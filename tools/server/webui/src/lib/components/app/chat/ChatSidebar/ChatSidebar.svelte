<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onMount, untrack } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { Trash2, Pencil, ChevronDown, X } from '@lucide/svelte';
	import {
		ChatSidebarConversationItem,
		ChatSidebarDocItem,
		DialogConfirmation
	} from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
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

	// Bulk selection / delete-all
	let selectionMode = $state(false);
	let selectedIds = new SvelteSet<string>();
	let anchorId = $state<string | undefined>(undefined);
	let showBulkDeleteDialog = $state(false);
	let showDeleteAllDialog = $state(false);

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

	// Flat id list in tree order — used as the domain for shift-range selection
	// and shift+arrow extensions in selection mode.
	let flatIds = $derived(conversationTree.map((n) => n.conversation.id));

	function enterSelectionMode(initialId?: string) {
		selectionMode = true;
		if (initialId) {
			selectedIds.add(initialId);
			anchorId = initialId;
		}
	}

	function exitSelectionMode() {
		selectionMode = false;
		selectedIds.clear();
		anchorId = undefined;
	}

	function toggleSelect(id: string, event: MouseEvent | KeyboardEvent) {
		if (event.shiftKey && anchorId) {
			const a = flatIds.indexOf(anchorId);
			const b = flatIds.indexOf(id);
			if (a !== -1 && b !== -1) {
				const [lo, hi] = a < b ? [a, b] : [b, a];
				for (let i = lo; i <= hi; i++) selectedIds.add(flatIds[i]);
				return;
			}
		}
		if (selectedIds.has(id)) {
			selectedIds.delete(id);
		} else {
			selectedIds.add(id);
		}
		anchorId = id;
	}

	function selectAllVisible() {
		for (const id of flatIds) selectedIds.add(id);
		anchorId = flatIds[flatIds.length - 1];
	}

	function extendSelectionByKey(direction: 1 | -1) {
		if (flatIds.length === 0) return;
		const current = anchorId ?? flatIds[0];
		const idx = flatIds.indexOf(current);
		if (idx === -1) return;
		const nextIdx = Math.max(0, Math.min(flatIds.length - 1, idx + direction));
		const nextId = flatIds[nextIdx];
		selectedIds.add(nextId);
		selectedIds.add(current);
		anchorId = nextId;
		// Scroll the now-focused row into view for visual feedback.
		const el = document.querySelector<HTMLElement>(`[data-conversation-id="${nextId}"]`);
		el?.scrollIntoView({ block: 'nearest' });
	}

	function handleGlobalKeydown(event: KeyboardEvent) {
		if (!selectionMode) return;
		// Don't hijack keys typed inside the edit/delete dialogs.
		if (showBulkDeleteDialog || showDeleteAllDialog || showEditDialog || showDocEditDialog) {
			return;
		}
		// Don't hijack keys while typing into form fields.
		const target = event.target as HTMLElement | null;
		if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;

		if (event.key === 'Escape') {
			event.preventDefault();
			exitSelectionMode();
			return;
		}
		if (event.key === 'ArrowDown' && event.shiftKey) {
			event.preventDefault();
			extendSelectionByKey(1);
			return;
		}
		if (event.key === 'ArrowUp' && event.shiftKey) {
			event.preventDefault();
			extendSelectionByKey(-1);
			return;
		}
		if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a') {
			event.preventDefault();
			selectAllVisible();
			return;
		}
		if ((event.key === 'Delete' || event.key === 'Backspace') && selectedIds.size > 0) {
			event.preventDefault();
			showBulkDeleteDialog = true;
		}
	}

	$effect(() => {
		// Drop stale ids if the visible list changes under us (e.g. new search query).
		const visible = new Set(flatIds);
		untrack(() => {
			for (const id of selectedIds) {
				if (!visible.has(id)) selectedIds.delete(id);
			}
			if (anchorId && !visible.has(anchorId)) anchorId = undefined;
		});
	});

	async function handleConfirmBulkDelete() {
		const ids = [...selectedIds];
		showBulkDeleteDialog = false;
		selectionMode = false;
		selectedIds.clear();
		anchorId = undefined;
		await conversationsStore.deleteMany(ids);
	}

	async function handleConfirmDeleteAll() {
		showDeleteAllDialog = false;
		exitSelectionMode();
		await conversationsStore.deleteAll();
	}

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

	async function handleDuplicateDoc(id: string) {
		try {
			await docsStore.duplicateDoc(id);
		} catch (err) {
			console.error('[docs] duplicate failed', err);
		}
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

	onMount(() => {
		window.addEventListener('keydown', handleGlobalKeydown);
		return () => window.removeEventListener('keydown', handleGlobalKeydown);
	});

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

		<ChatSidebarActions
			{handleMobileSidebarItemClick}
			bind:isSearchModeActive
			bind:searchQuery
			onEnterSelectionMode={() => enterSelectionMode()}
			onDeleteAll={() => (showDeleteAllDialog = true)}
		/>

		{#if selectionMode}
			<div
				class="flex items-center justify-between gap-2 rounded-md border border-sidebar-border bg-sidebar/60 px-2 py-1 text-xs"
			>
				<span class="truncate text-sidebar-foreground/80">
					{selectedIds.size} selected
				</span>
				<div class="flex items-center gap-1">
					<Button
						size="sm"
						variant="destructive"
						class="h-7 px-2"
						disabled={selectedIds.size === 0}
						onclick={() => (showBulkDeleteDialog = true)}
					>
						<Trash2 class="h-3.5 w-3.5" />
						Delete
					</Button>
					<Button
						size="sm"
						variant="ghost"
						class="h-7 px-2"
						onclick={exitSelectionMode}
						aria-label="Exit selection mode"
					>
						<X class="h-3.5 w-3.5" />
					</Button>
				</div>
			</div>
		{/if}
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
								{selectionMode}
								isSelected={selectedIds.has(conversation.id)}
								onToggleSelect={toggleSelect}
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
									onDuplicate={handleDuplicateDoc}
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

<DialogConfirmation
	bind:open={showBulkDeleteDialog}
	title="Delete selected conversations"
	description="Are you sure you want to delete {selectedIds.size} conversation{selectedIds.size === 1
		? ''
		: 's'}? This cannot be undone."
	confirmText="Delete"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmBulkDelete}
	onCancel={() => (showBulkDeleteDialog = false)}
/>

<DialogConfirmation
	bind:open={showDeleteAllDialog}
	title="Delete all conversations"
	description="Are you sure you want to delete every conversation? This cannot be undone."
	confirmText="Delete all"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmDeleteAll}
	onCancel={() => (showDeleteAllDialog = false)}
/>
