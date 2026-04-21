<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { Trash2, Pencil, ChevronDown } from '@lucide/svelte';
	import {
		ChatSidebarConversationItem,
		ChatSidebarDocItem,
		DialogConfirmation
	} from '$lib/components/app';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import Label from '$lib/components/ui/label/label.svelte';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import Input from '$lib/components/ui/input/input.svelte';
	import {
		conversationsStore,
		conversations,
		buildConversationTree
	} from '$lib/stores/conversations.svelte';
	import { docsStore, docs } from '$lib/stores/docs.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { getPreviewText } from '$lib/utils';
	import ChatSidebarActions from './ChatSidebarActions.svelte';

	const sidebar = Sidebar.useSidebar();

	const COLLAPSE_KEY = 'ht-sidebar-collapsed';

	function readCollapsed(): Record<string, boolean> {
		if (typeof window === 'undefined') return {};
		try {
			const raw = window.localStorage.getItem(COLLAPSE_KEY);
			return raw ? (JSON.parse(raw) as Record<string, boolean>) : {};
		} catch {
			return {};
		}
	}

	function writeCollapsed(state: Record<string, boolean>) {
		if (typeof window === 'undefined') return;
		try {
			window.localStorage.setItem(COLLAPSE_KEY, JSON.stringify(state));
		} catch {
			/* ignore */
		}
	}

	let collapsed = $state<Record<string, boolean>>({});

	onMount(() => {
		if (!docsStore.isInitialized) {
			docsStore.initialize();
		}
		collapsed = readCollapsed();
	});

	function toggleGroup(key: 'conversations' | 'documents') {
		collapsed = { ...collapsed, [key]: !collapsed[key] };
		writeCollapsed(collapsed);
	}

	let currentRouteId = $derived(page.route.id);
	let currentChatId = $derived(currentRouteId === '/chat/[id]' ? page.params.id : undefined);
	let currentDocId = $derived(currentRouteId === '/doc/[id]' ? page.params.id : undefined);
	let isSearchModeActive = $state(false);
	let searchQuery = $state('');
	let showDeleteDialog = $state(false);
	let deleteWithForks = $state(false);
	let showEditDialog = $state(false);
	let selectedConversation = $state<DatabaseConversation | null>(null);
	let editedName = $state('');
	let selectedConversationNamePreview = $derived.by(() =>
		selectedConversation ? getPreviewText(selectedConversation.name) : ''
	);

	let showDocDeleteDialog = $state(false);
	let showDocEditDialog = $state(false);
	let selectedDoc = $state<DatabaseDoc | null>(null);
	let editedDocName = $state('');

	let filteredConversations = $derived.by(() => {
		if (searchQuery.trim().length > 0) {
			return conversations().filter((conversation: { name: string }) =>
				conversation.name.toLowerCase().includes(searchQuery.toLowerCase())
			);
		}

		return conversations();
	});

	let filteredDocs = $derived.by(() => {
		if (searchQuery.trim().length === 0) return docs();
		const q = searchQuery.toLowerCase();
		return docs().filter(
			(d) => d.name.toLowerCase().includes(q) || (d.content ?? '').toLowerCase().includes(q)
		);
	});

	let conversationTree = $derived(buildConversationTree(filteredConversations));

	let selectedConversationHasDescendants = $derived.by(() => {
		if (!selectedConversation) return false;

		const allConvs = conversations();
		const queue = [selectedConversation.id];

		while (queue.length > 0) {
			const parentId = queue.pop()!;

			for (const c of allConvs) {
				if (c.forkedFromConversationId === parentId) return true;
			}
		}

		return false;
	});

	async function handleDeleteConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
		if (conversation) {
			selectedConversation = conversation;
			deleteWithForks = false;
			showDeleteDialog = true;
		}
	}

	async function handleEditConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
		if (conversation) {
			selectedConversation = conversation;
			editedName = conversation.name;
			showEditDialog = true;
		}
	}

	function handleConfirmDelete() {
		if (selectedConversation) {
			const convId = selectedConversation.id;
			const withForks = deleteWithForks;
			showDeleteDialog = false;

			setTimeout(() => {
				conversationsStore.deleteConversation(convId, {
					deleteWithForks: withForks
				});
			}, 100); // Wait for animation to finish
		}
	}

	function handleConfirmEdit() {
		if (!editedName.trim() || !selectedConversation) return;

		showEditDialog = false;

		conversationsStore.updateConversationName(selectedConversation.id, editedName);
		selectedConversation = null;
	}

	async function handleEditDoc(id: string) {
		const doc = docs().find((d) => d.id === id);
		if (!doc) return;
		selectedDoc = doc;
		editedDocName = doc.name ?? '';
		showDocEditDialog = true;
	}

	async function handleDeleteDoc(id: string) {
		const doc = docs().find((d) => d.id === id);
		if (!doc) return;
		selectedDoc = doc;
		showDocDeleteDialog = true;
	}

	function handleConfirmDocEdit() {
		if (!selectedDoc || !editedDocName.trim()) return;
		const docId = selectedDoc.id;
		const name = editedDocName;
		showDocEditDialog = false;
		docsStore.renameDoc(docId, name);
		selectedDoc = null;
	}

	function handleConfirmDocDelete() {
		if (!selectedDoc) return;
		const docId = selectedDoc.id;
		showDocDeleteDialog = false;
		setTimeout(() => docsStore.deleteDoc(docId), 100);
		selectedDoc = null;
	}

	export function handleMobileSidebarItemClick() {
		if (sidebar.isMobile) {
			sidebar.toggle();
		}
	}

	export function activateSearchMode() {
		isSearchModeActive = true;
	}

	export function editActiveConversation() {
		if (currentChatId) {
			const activeConversation = filteredConversations.find((conv) => conv.id === currentChatId);

			if (activeConversation) {
				const event = new CustomEvent('edit-active-conversation', {
					detail: { conversationId: currentChatId }
				});
				document.dispatchEvent(event);
			}
		}
	}

	async function selectConversation(id: string) {
		if (isSearchModeActive) {
			isSearchModeActive = false;
			searchQuery = '';
		}

		await goto(`#/chat/${id}`);
	}

	async function selectDoc(id: string) {
		handleMobileSidebarItemClick();
		await goto(`#/doc/${id}`);
	}

	function handleStopGeneration(id: string) {
		chatStore.stopGenerationForChat(id);
	}
</script>

<ScrollArea class="h-[100vh]">
	<Sidebar.Header class=" top-0 z-10 gap-4 bg-sidebar/50 p-4 pb-2 backdrop-blur-lg md:sticky">
		<a href="#/" onclick={handleMobileSidebarItemClick}>
			<h1 class="inline-flex items-center gap-1 px-2 text-xl font-semibold">ht-llama.cpp</h1>
		</a>

		<ChatSidebarActions {handleMobileSidebarItemClick} bind:isSearchModeActive bind:searchQuery />
	</Sidebar.Header>

	<Sidebar.Group class="mt-2 space-y-2 p-0 px-4">
		{#if (filteredConversations.length > 0 && isSearchModeActive) || !isSearchModeActive}
			{#if isSearchModeActive}
				<Sidebar.GroupLabel>Search results</Sidebar.GroupLabel>
			{:else}
				<button
					type="button"
					class="group flex h-8 w-full cursor-pointer items-center justify-between rounded-md px-2 text-xs font-medium text-sidebar-foreground/70 transition-colors hover:text-sidebar-foreground focus-visible:ring-2 focus-visible:ring-sidebar-ring focus-visible:outline-hidden"
					onclick={() => toggleGroup('conversations')}
					aria-expanded={!collapsed.conversations}
				>
					<span>Conversations</span>
					<ChevronDown
						class="h-3.5 w-3.5 transition-transform {collapsed.conversations ? '-rotate-90' : ''}"
					/>
				</button>
			{/if}
		{/if}

		{#if isSearchModeActive || !collapsed.conversations}
			<Sidebar.GroupContent>
				<Sidebar.Menu>
					{#each conversationTree as { conversation, depth } (conversation.id)}
						<Sidebar.MenuItem class="mb-1 p-0">
							<ChatSidebarConversationItem
								conversation={{
									id: conversation.id,
									name: conversation.name,
									lastModified: conversation.lastModified,
									currNode: conversation.currNode,
									forkedFromConversationId: conversation.forkedFromConversationId
								}}
								{depth}
								{handleMobileSidebarItemClick}
								isActive={currentChatId === conversation.id}
								onSelect={selectConversation}
								onEdit={handleEditConversation}
								onDelete={handleDeleteConversation}
								onStop={handleStopGeneration}
							/>
						</Sidebar.MenuItem>
					{/each}

					{#if conversationTree.length === 0}
						<div class="px-2 py-4 text-center">
							<p class="mb-4 p-4 text-sm text-muted-foreground">
								{searchQuery.length > 0
									? 'No results found'
									: isSearchModeActive
										? 'Start typing to see results'
										: 'No conversations yet'}
							</p>
						</div>
					{/if}
				</Sidebar.Menu>
			</Sidebar.GroupContent>
		{/if}
	</Sidebar.Group>

	{#if isSearchModeActive ? filteredDocs.length > 0 : docs().length > 0}
		<Sidebar.Group class="mt-2 space-y-2 p-0 px-4">
			{#if isSearchModeActive}
				<Sidebar.GroupLabel>Documents</Sidebar.GroupLabel>
			{:else}
				<button
					type="button"
					class="group flex h-8 w-full cursor-pointer items-center justify-between rounded-md px-2 text-xs font-medium text-sidebar-foreground/70 transition-colors hover:text-sidebar-foreground focus-visible:ring-2 focus-visible:ring-sidebar-ring focus-visible:outline-hidden"
					onclick={() => toggleGroup('documents')}
					aria-expanded={!collapsed.documents}
				>
					<span>Documents</span>
					<ChevronDown
						class="h-3.5 w-3.5 transition-transform {collapsed.documents ? '-rotate-90' : ''}"
					/>
				</button>
			{/if}

			{#if isSearchModeActive || !collapsed.documents}
				<Sidebar.GroupContent>
					<Sidebar.Menu>
						{#each filteredDocs as doc (doc.id)}
							<Sidebar.MenuItem class="mb-1 p-0">
								<ChatSidebarDocItem
									{doc}
									isActive={currentDocId === doc.id}
									{handleMobileSidebarItemClick}
									onSelect={selectDoc}
									onEdit={handleEditDoc}
									onDelete={handleDeleteDoc}
								/>
							</Sidebar.MenuItem>
						{/each}
					</Sidebar.Menu>
				</Sidebar.GroupContent>
			{/if}
		</Sidebar.Group>
	{/if}
</ScrollArea>

<DialogConfirmation
	bind:open={showDeleteDialog}
	title="Delete Conversation"
	description={selectedConversation
		? `Are you sure you want to delete "${selectedConversationNamePreview}"? This action cannot be undone and will permanently remove all messages in this conversation.`
		: ''}
	confirmText="Delete"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmDelete}
	onCancel={() => {
		showDeleteDialog = false;
		selectedConversation = null;
	}}
>
	{#if selectedConversationHasDescendants}
		<div class="flex items-center gap-2 py-2">
			<Checkbox id="delete-with-forks" bind:checked={deleteWithForks} />

			<Label for="delete-with-forks" class="text-sm">Also delete all forked conversations</Label>
		</div>
	{/if}
</DialogConfirmation>

<DialogConfirmation
	bind:open={showEditDialog}
	title="Edit Conversation Name"
	description=""
	confirmText="Save"
	cancelText="Cancel"
	icon={Pencil}
	onConfirm={handleConfirmEdit}
	onCancel={() => {
		showEditDialog = false;
		selectedConversation = null;
	}}
	onKeydown={(e) => {
		if (e.key === 'Enter') {
			e.preventDefault();
			e.stopImmediatePropagation();
			handleConfirmEdit();
		}
	}}
>
	<Input
		class="text-foreground"
		placeholder="Enter a new name"
		type="text"
		bind:value={editedName}
	/>
</DialogConfirmation>

<DialogConfirmation
	bind:open={showDocDeleteDialog}
	title="Delete Document"
	description={selectedDoc
		? `Are you sure you want to delete "${selectedDoc.name || 'Untitled'}"? This cannot be undone.`
		: ''}
	confirmText="Delete"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmDocDelete}
	onCancel={() => {
		showDocDeleteDialog = false;
		selectedDoc = null;
	}}
/>

<DialogConfirmation
	bind:open={showDocEditDialog}
	title="Rename Document"
	description=""
	confirmText="Save"
	cancelText="Cancel"
	icon={Pencil}
	onConfirm={handleConfirmDocEdit}
	onCancel={() => {
		showDocEditDialog = false;
		selectedDoc = null;
	}}
	onKeydown={(e) => {
		if (e.key === 'Enter') {
			e.preventDefault();
			e.stopImmediatePropagation();
			handleConfirmDocEdit();
		}
	}}
>
	<Input
		class="text-foreground"
		placeholder="Enter a new name"
		type="text"
		bind:value={editedDocName}
	/>
</DialogConfirmation>
