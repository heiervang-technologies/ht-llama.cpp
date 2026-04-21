<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { MarkdownContent } from '$lib/components/app';
	import { docsStore, activeDoc } from '$lib/stores/docs.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { aiCommandsStore } from '$lib/stores/ai-commands.svelte';
	import { IsMobile } from '$lib/hooks/is-mobile.svelte';
	import DocScreenHeader from './DocScreenHeader.svelte';
	import DocEditor, { type DocEditorApi } from './DocEditor.svelte';

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
	let editorApi: DocEditorApi | null = null;
	let commandsMenuOpen = $state(false);

	onMount(() => {
		if (!docsStore.isInitialized) {
			docsStore.initialize();
		}

		// Ctrl/Cmd+Shift+K — open the AI commands menu from anywhere in the doc
		// screen. Ctrl+K alone is reserved for the browser address bar; Ctrl+Shift
		// gives us a dedicated binding that doesn't collide with other editor
		// shortcuts or the browser's default.
		function onKeydown(e: KeyboardEvent) {
			const modifier = e.metaKey || e.ctrlKey;
			if (modifier && e.shiftKey && (e.key === 'k' || e.key === 'K')) {
				e.preventDefault();
				if (aiCommandsStore.runningId === null) {
					commandsMenuOpen = true;
				}
			}
		}
		window.addEventListener('keydown', onKeydown);
		return () => window.removeEventListener('keydown', onKeydown);
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

	async function handleRunAiCommand(commandId: string) {
		if (!doc) return;
		// Flush any pending save so the command runs against the saved state.
		if (pendingContent !== null) {
			await docsStore.updateContent(doc.id, pendingContent);
			pendingContent = null;
			saving = false;
		}

		const command = aiCommandsStore.list().find((c) => c.id === commandId);
		if (!command) return;

		const baseContent = doc.content;
		const sel = editorApi?.getSelection();
		const selectionText = sel?.text ?? '';
		const hasSelection = selectionText.length > 0;

		if (command.requiresSelection && !hasSelection) {
			toast.warning(`"${command.name}" requires a text selection in the editor.`);
			return;
		}

		if (command.mode === 'replace' && hasSelection && sel && editorApi) {
			// Stream the model output directly over the selected range.
			const from = sel.from;
			let currentEnd = sel.to;
			let outputSoFar = '';
			const api = editorApi;
			await aiCommandsStore.run(
				commandId,
				baseContent,
				(delta) => {
					outputSoFar += delta;
					api.replaceRange(from, currentEnd, outputSoFar, { stream: true });
					currentEnd = from + outputSoFar.length;
				},
				selectionText
			);
			// The editor's updateListener already streamed changes through onChange.
			// Persist the final doc state once streaming finishes.
			if (outputSoFar.length > 0) {
				const finalContent = baseContent.slice(0, from) + outputSoFar + baseContent.slice(sel.to);
				await docsStore.updateContent(doc.id, finalContent);
			}
			return;
		}

		// Append mode (default): add a separator and stream after the existing doc.
		const separator = baseContent.trimEnd().length === 0 ? '' : '\n\n---\n\n';
		let outputSoFar = '';
		const appendStart = baseContent.length + separator.length;
		let appendEnd = appendStart;
		// Seed the separator once up front so subsequent token writes just extend
		// the tail, and so the preview immediately shows the divider.
		if (editorApi) {
			editorApi.replaceRange(baseContent.length, baseContent.length, separator, { stream: true });
		}
		await aiCommandsStore.run(
			commandId,
			baseContent,
			(delta) => {
				outputSoFar += delta;
				if (editorApi) {
					editorApi.replaceRange(appendStart, appendEnd, outputSoFar, { stream: true });
					appendEnd = appendStart + outputSoFar.length;
				}
				// Keep the docs store mirror in sync so the preview pane and any
				// other reactive consumers re-render. The editor already has the
				// change applied, so DocEditor's $effect will no-op.
				docsStore.setContentLive(doc!.id, baseContent + separator + outputSoFar);
			},
			selectionText
		);
		if (outputSoFar.length > 0) {
			await docsStore.updateContent(doc.id, baseContent + separator + outputSoFar);
		}
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
	onRunAiCommand={handleRunAiCommand}
	bind:commandsMenuOpen
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
					{#key doc.id}
						<DocEditor
							content={doc.content}
							onChange={scheduleSave}
							onReady={(api) => (editorApi = api)}
						/>
					{/key}
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
