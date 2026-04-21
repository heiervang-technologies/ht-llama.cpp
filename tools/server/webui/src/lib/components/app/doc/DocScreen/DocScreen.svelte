<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { MarkdownContent } from '$lib/components/app';
	import { docsStore, activeDoc } from '$lib/stores/docs.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { IsMobile } from '$lib/hooks/is-mobile.svelte';
	import DocScreenHeader from './DocScreenHeader.svelte';
	import DocEditor from './DocEditor.svelte';

	interface Props {
		docId: string;
	}

	let { docId }: Props = $props();

	const VIEW_KEY = 'docViewMode';
	type ViewMode = 'edit' | 'preview' | 'split';
	let storedView: ViewMode = 'split';
	if (browser) {
		try {
			const raw = localStorage.getItem(VIEW_KEY);
			if (raw === 'edit' || raw === 'preview' || raw === 'split') storedView = raw;
		} catch {
			/* ignore */
		}
	}
	let viewMode = $state<ViewMode>(storedView);
	const isMobile = new IsMobile();
	let effectiveView = $derived<ViewMode>(
		isMobile.current ? (viewMode === 'split' ? 'edit' : viewMode) : viewMode
	);

	function setView(next: ViewMode) {
		viewMode = next;
		if (browser) {
			try {
				localStorage.setItem(VIEW_KEY, next);
			} catch {
				/* ignore */
			}
		}
	}

	let doc = $derived(activeDoc());
	let saving = $state(false);
	let pendingContent = $state<string | null>(null);
	let saveTimer: ReturnType<typeof setTimeout> | undefined;

	onMount(() => {
		if (!docsStore.isInitialized) {
			docsStore.initialize();
		}
	});

	$effect(() => {
		const id = docId;
		untrack(() => {
			docsStore.loadDoc(id).then((loaded) => {
				if (!loaded) {
					goto('#/');
				}
			});
		});
	});

	function scheduleSave(next: string) {
		pendingContent = next;
		saving = true;
		if (saveTimer) clearTimeout(saveTimer);
		saveTimer = setTimeout(async () => {
			if (!doc) return;
			const toSave = pendingContent;
			pendingContent = null;
			if (toSave === null) return;
			try {
				await docsStore.updateContent(doc.id, toSave);
			} finally {
				saving = false;
			}
		}, 500);
	}

	async function handleRename(name: string) {
		if (!doc) return;
		await docsStore.renameDoc(doc.id, name);
	}

	async function handleChatAbout() {
		if (!doc) return;
		const trimmedName = (doc.name || 'Untitled').trim();
		const trimmedContent = (pendingContent ?? doc.content).trim();
		// Flush any pending save first so sidebar shows the right version.
		if (pendingContent !== null) {
			await docsStore.updateContent(doc.id, pendingContent);
			pendingContent = null;
			saving = false;
		}
		const seed = trimmedContent
			? `I'd like to discuss this document titled "${trimmedName}":\n\n\`\`\`markdown\n${trimmedContent}\n\`\`\`\n\n`
			: `I'd like to start a chat about my document "${trimmedName}".\n\n`;
		try {
			globalThis.sessionStorage?.setItem('pendingDocSeed', seed);
		} catch {
			/* ignore storage errors */
		}
		await conversationsStore.createConversation(trimmedName || 'Untitled');
	}
</script>

<DocScreenHeader
	name={doc?.name ?? ''}
	view={effectiveView}
	{saving}
	onRename={handleRename}
	onViewChange={setView}
	onChatAbout={handleChatAbout}
/>

<main class="flex h-full w-full flex-col pt-14 md:pt-16">
	{#if !doc}
		<div class="flex flex-1 items-center justify-center text-sm text-muted-foreground">
			Loading document…
		</div>
	{:else}
		<div class="flex min-h-0 flex-1 overflow-hidden">
			{#if effectiveView !== 'preview'}
				<div
					class="flex min-h-0 flex-1 flex-col border-r border-border/60 {effectiveView === 'edit'
						? 'w-full'
						: 'w-1/2'}"
				>
					<DocEditor content={doc.content} onChange={scheduleSave} />
				</div>
			{/if}

			{#if effectiveView !== 'edit'}
				<div
					class="flex min-h-0 {effectiveView === 'preview'
						? 'w-full'
						: 'w-1/2'} flex-col overflow-auto bg-background/40"
				>
					<div class="prose prose-sm dark:prose-invert max-w-none px-6 py-6">
						{#if doc.content.trim().length === 0}
							<p class="text-muted-foreground">
								Start typing on the left — preview will render here.
							</p>
						{:else}
							<MarkdownContent content={doc.content} />
						{/if}
					</div>
				</div>
			{/if}
		</div>
	{/if}
</main>
