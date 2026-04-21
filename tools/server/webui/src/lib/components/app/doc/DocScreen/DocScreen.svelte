<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { Trash2 } from '@lucide/svelte';
	import { toast } from 'svelte-sonner';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { MarkdownContent, DialogConfirmation } from '$lib/components/app';
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
	let showDeleteDialog = $state(false);

	// A doc loaded with its default name and empty body, freshly created in
	// the last few seconds, is almost certainly a brand-new doc. Auto-focus
	// the title so the user can type a name without an extra click.
	let autofocusTitle = $derived(
		!!doc &&
			(doc.name ?? '') === 'Untitled' &&
			(doc.content ?? '').length === 0 &&
			Date.now() - doc.lastModified < 3000
	);

	onMount(() => {
		if (!docsStore.isInitialized) {
			docsStore.initialize();
		}

		// Ctrl/Cmd+Shift+K — open the AI commands menu from anywhere in the doc
		// screen. Ctrl+K alone is reserved for the browser address bar; Ctrl+Shift
		// gives us a dedicated binding that doesn't collide with other editor
		// shortcuts or the browser's default. Ctrl/Cmd+S flushes the debounced
		// save so the user can commit intentionally without waiting 500ms.
		function onKeydown(e: KeyboardEvent) {
			const modifier = e.metaKey || e.ctrlKey;
			if (modifier && e.shiftKey && (e.key === 'k' || e.key === 'K')) {
				e.preventDefault();
				if (aiCommandsStore.runningId === null) {
					commandsMenuOpen = true;
				}
			} else if (modifier && !e.shiftKey && !e.altKey && (e.key === 's' || e.key === 'S')) {
				// Browsers bind Ctrl+S to "Save Page As" by default — preventDefault
				// suppresses that dialog before flushing.
				e.preventDefault();
				flushSaveNow();
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

	// Strip leading #'s + whitespace from a markdown heading line. Returns
	// null if the line isn't a heading, is empty, or only contains #'s.
	function extractH1Title(content: string): string | null {
		const firstLine = content.split('\n', 1)[0] ?? '';
		const match = firstLine.match(/^\s*#{1,6}\s+(.+?)\s*#*\s*$/);
		if (!match) return null;
		const stripped = match[1].trim();
		if (!stripped) return null;
		// Cap overlong titles; the full heading is still in the doc body.
		return stripped.length > 80 ? stripped.slice(0, 80).trimEnd() + '…' : stripped;
	}

	async function persistContent(toSave: string) {
		if (!doc) return;
		try {
			await docsStore.updateContent(doc.id, toSave);
			// Auto-derive a title from the first markdown heading while the
			// doc is still Untitled. Stops after the first successful rename;
			// a user-picked title will never be overwritten.
			if (doc.name === 'Untitled') {
				const derived = extractH1Title(toSave);
				if (derived) {
					await docsStore.renameDoc(doc.id, derived);
				}
			}
		} finally {
			saving = false;
		}
	}

	function scheduleSave(next: string) {
		pendingContent = next;
		saving = true;
		if (saveTimer) clearTimeout(saveTimer);
		saveTimer = setTimeout(async () => {
			const toSave = pendingContent;
			pendingContent = null;
			if (toSave === null) return;
			await persistContent(toSave);
		}, 500);
	}

	async function flushSaveNow() {
		if (saveTimer) {
			clearTimeout(saveTimer);
			saveTimer = undefined;
		}
		const toSave = pendingContent;
		pendingContent = null;
		if (toSave === null) {
			// Nothing pending — likely already saved. Leave the UI alone.
			return;
		}
		await persistContent(toSave);
	}

	async function handleRename(name: string) {
		if (!doc) return;
		await docsStore.renameDoc(doc.id, name);
	}

	async function handleRunAiCommand(commandId: string) {
		if (!doc) return;
		// Flush any pending save so the command runs against the saved state.
		await flushSaveNow();

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

	function handleDeleteRequest() {
		if (!doc) return;
		showDeleteDialog = true;
	}

	async function handleDeleteConfirm() {
		showDeleteDialog = false;
		if (!doc) return;
		// Cancel any pending save so we don't write back a ghost after deletion.
		if (saveTimer) {
			clearTimeout(saveTimer);
			saveTimer = undefined;
		}
		pendingContent = null;
		saving = false;
		const name = doc.name?.trim() || 'Untitled';
		try {
			await docsStore.deleteDoc(doc.id);
			toast.success(`Deleted "${name}"`);
		} catch (err) {
			console.error('[docs] delete failed', err);
			toast.error(`Failed to delete "${name}"`);
		}
	}

	// Dictation drop point. Called by the header's mic button once STT returns.
	// Inserts the transcribed text at the current cursor (or replaces the
	// selection), then schedules a save the same way typing does.
	function handleDictation(text: string) {
		if (!text || !doc) return;
		const sel = editorApi?.getSelection();
		if (!editorApi || !sel) {
			// Editor API not ready: append to the doc, with a leading space so
			// dictation doesn't glue to the previous word.
			const current = doc.content ?? '';
			const needsSpace = current.length > 0 && !/\s$/.test(current);
			const next = current + (needsSpace ? ' ' : '') + text;
			docsStore.setContentLive(doc.id, next);
			scheduleSave(next);
			return;
		}
		// Caret case (no selection): prepend a space when the char before the
		// cursor is a non-whitespace word character, so dictated phrases don't
		// stick to the preceding token. When there's a selection, we're
		// replacing it outright, so no extra spacing is needed.
		let toInsert = text;
		if (sel.from === sel.to && sel.from > 0) {
			const prev = doc.content.charAt(sel.from - 1);
			if (prev && !/\s/.test(prev)) toInsert = ' ' + toInsert;
		}
		editorApi.replaceRange(sel.from, sel.to, toInsert);
		editorApi.focus();
	}

	async function handleChatAbout() {
		if (!doc) return;
		const trimmedName = (doc.name || 'Untitled').trim();
		const trimmedContent = (pendingContent ?? doc.content).trim();
		// Flush any pending save first so sidebar shows the right version.
		await flushSaveNow();
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
	content={doc?.content ?? ''}
	onRename={handleRename}
	onViewChange={setView}
	onChatAbout={handleChatAbout}
	onRunAiCommand={handleRunAiCommand}
	onDelete={handleDeleteRequest}
	onDictate={handleDictation}
	bind:commandsMenuOpen
	{autofocusTitle}
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

<DialogConfirmation
	bind:open={showDeleteDialog}
	title="Delete document"
	description={`Are you sure you want to delete "${(doc?.name || 'Untitled').trim()}"? This cannot be undone.`}
	confirmText="Delete"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleDeleteConfirm}
	onCancel={() => (showDeleteDialog = false)}
/>
