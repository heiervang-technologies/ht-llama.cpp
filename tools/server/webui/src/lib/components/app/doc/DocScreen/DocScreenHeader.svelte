<script lang="ts">
	import {
		Settings,
		MessageSquarePlus,
		Columns,
		FileText,
		Eye,
		Sparkles,
		Mic,
		Square,
		Loader2,
		NotebookPen
	} from '@lucide/svelte';
	import { toast } from 'svelte-sonner';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Popover from '$lib/components/ui/popover';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import { getChatSettingsDialogContext } from '$lib/contexts';
	import { BackendPill } from '$lib/components/app/navigation';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { SttService } from '$lib/services/stt.service';
	import {
		AudioRecorder,
		convertToWav,
		createAudioFile,
		isAudioRecordingSupported
	} from '$lib/utils/audio-recording';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { selectedModelId } from '$lib/stores/models.svelte';
	import ChainPicker from '$lib/components/app/chat/ChatForm/ChatFormActions/ChainPicker.svelte';
	import AiCommandsMenu from './AiCommandsMenu.svelte';
	import DocMoreActionsMenu from './DocMoreActionsMenu.svelte';

	interface Props {
		id: string;
		name: string;
		view: 'edit' | 'preview' | 'split';
		saving: boolean;
		content: string;
		onRename: (next: string) => void;
		onViewChange: (next: 'edit' | 'preview' | 'split') => void;
		onChatAbout: () => void;
		onRunAiCommand: (commandId: string) => void;
		onDelete: () => void;
		onDictate?: (text: string) => void;
		commandsMenuOpen?: boolean;
		/** If true, focus + select the title input on mount. Used for brand-new docs. */
		autofocusTitle?: boolean;
	}

	let {
		id,
		name,
		view,
		saving,
		content,
		onRename,
		onViewChange,
		onChatAbout,
		onRunAiCommand,
		onDelete,
		onDictate,
		commandsMenuOpen = $bindable(false),
		autofocusTitle = false
	}: Props = $props();

	// Lightweight word counter: split on whitespace runs and drop empties. Good
	// enough for prose; markdown syntax characters don't count as words. At ~50k
	// chars this is still sub-millisecond, so no need to debounce.
	let wordCount = $derived(
		content
			.trim()
			.split(/\s+/)
			.filter((s) => s.length > 0).length
	);

	let titleInputEl: HTMLInputElement | undefined = $state();

	$effect(() => {
		if (autofocusTitle && titleInputEl) {
			titleInputEl.focus();
			titleInputEl.select();
		}
	});

	const sidebar = useSidebar();
	const chatSettingsDialog = getChatSettingsDialogContext();

	let localName = $derived(name);
	let inlineOn = $derived(Boolean(config().inlineCompletionEnabled));

	// Mirror of the global system prompt while the popover is open. Flushed to
	// config only on Save so half-typed drafts don't leak into new chats.
	let systemPromptDraft = $state('');
	let systemPromptOpen = $state(false);
	$effect(() => {
		if (systemPromptOpen) systemPromptDraft = config().systemMessage ?? '';
	});
	function saveSystemPrompt() {
		settingsStore.updateConfig('systemMessage', systemPromptDraft);
		systemPromptOpen = false;
		toast.success('System prompt updated');
	}

	let activeModelId = $derived(selectedModelId());
	let isRouter = $derived(isRouterMode());

	// Dictation: identical pattern to ChatForm's mic button, but the transcribed
	// text is inserted at the editor cursor instead of into a textarea. Gated on
	// STT being configured + enabled; falls back to the record-only path if STT
	// isn't set up (user still gets to download the wav via the browser).
	let sttDictationReady = $derived(
		Boolean(config().sttEnabled) && SttService.isConfigured() && isAudioRecordingSupported()
	);
	let isRecording = $state(false);
	let isTranscribing = $state(false);
	let transcribeAbort: AbortController | null = null;
	const audioRecorder = new AudioRecorder();

	function commit() {
		if (localName !== name) onRename(localName);
	}

	function toggleInlineCompletion() {
		settingsStore.updateConfig('inlineCompletionEnabled', !config().inlineCompletionEnabled);
	}

	async function handleMicClick() {
		// Clicking while transcribing cancels the in-flight STT request, so a
		// slow/hung server doesn't strand the user in the spinner state.
		if (isTranscribing) {
			transcribeAbort?.abort();
			transcribeAbort = null;
			isTranscribing = false;
			return;
		}
		if (audioRecorder.isRecording()) {
			try {
				const blob = await audioRecorder.stopRecording();
				isRecording = false;
				const wav = await convertToWav(blob);
				const file = createAudioFile(wav, `dictation-${Date.now()}.wav`);
				isTranscribing = true;
				const controller = new AbortController();
				transcribeAbort = controller;
				try {
					const text = await SttService.transcribe(file, { signal: controller.signal });
					if (text) onDictate?.(text);
				} catch (err) {
					// Silent on user-initiated aborts — we already reset state above.
					if ((err as { name?: string })?.name === 'AbortError') return;
					console.error('[doc-dictate] transcribe failed', err);
					const msg = err instanceof Error ? err.message : String(err);
					toast.error(`Dictation failed: ${msg}`);
				} finally {
					if (transcribeAbort === controller) transcribeAbort = null;
					isTranscribing = false;
				}
			} catch (err) {
				console.error('[doc-dictate] stop failed', err);
				isRecording = false;
				isTranscribing = false;
			}
		} else {
			try {
				await audioRecorder.startRecording();
				isRecording = true;
			} catch (err) {
				console.error('[doc-dictate] start failed', err);
				const msg = err instanceof Error ? err.message : String(err);
				toast.error(`Could not start recording: ${msg}`);
			}
		}
	}
</script>

<header
	class="pointer-events-none fixed top-0 right-0 left-0 z-50 flex items-center gap-2 p-2 duration-200 ease-linear md:p-4 {sidebar.open
		? 'md:left-[var(--sidebar-width)]'
		: ''}"
>
	<div class="pointer-events-auto ml-12 flex min-w-0 flex-1 items-center gap-2 md:ml-12">
		<input
			bind:this={titleInputEl}
			type="text"
			class="min-w-0 flex-1 truncate rounded-md bg-transparent px-2 py-1 text-sm font-medium text-foreground outline-none focus:bg-muted/40"
			bind:value={localName}
			onblur={commit}
			onkeydown={(e) => {
				if (e.key === 'Enter') {
					e.preventDefault();
					(e.currentTarget as HTMLInputElement).blur();
				} else if (e.key === 'Escape') {
					// Revert any in-progress edit and blur. commit() no-ops when the
					// name hasn't changed, so resetting localName before blur is enough.
					e.preventDefault();
					localName = name;
					(e.currentTarget as HTMLInputElement).blur();
				}
			}}
			placeholder="Untitled"
		/>

		{#if saving}
			<span class="text-xs text-muted-foreground">Saving…</span>
		{:else if wordCount > 0}
			<span
				class="hidden text-xs text-muted-foreground md:inline"
				title="Word count (whitespace-separated)"
			>
				{wordCount.toLocaleString()}
				{wordCount === 1 ? 'word' : 'words'}
			</span>
		{/if}
	</div>

	<div class="pointer-events-auto flex items-center gap-1">
		<div class="hidden items-center rounded-md border bg-background/70 p-0.5 backdrop-blur md:flex">
			<Button
				variant={view === 'edit' ? 'secondary' : 'ghost'}
				size="sm"
				class="h-7 px-2"
				onclick={() => onViewChange('edit')}
				title="Editor only"
			>
				<FileText class="h-3.5 w-3.5" />
			</Button>
			<Button
				variant={view === 'split' ? 'secondary' : 'ghost'}
				size="sm"
				class="h-7 px-2"
				onclick={() => onViewChange('split')}
				title="Split"
			>
				<Columns class="h-3.5 w-3.5" />
			</Button>
			<Button
				variant={view === 'preview' ? 'secondary' : 'ghost'}
				size="sm"
				class="h-7 px-2"
				onclick={() => onViewChange('preview')}
				title="Preview only"
			>
				<Eye class="h-3.5 w-3.5" />
			</Button>
		</div>

		<Button
			variant={inlineOn ? 'secondary' : 'ghost'}
			size="sm"
			onclick={toggleInlineCompletion}
			class="gap-1.5 rounded-full backdrop-blur-lg"
			title={inlineOn
				? 'AI ghost-text completions are ON — Ctrl+Tab to force, Tab to accept, Esc to dismiss'
				: 'Enable AI ghost-text completions'}
		>
			<Sparkles class="h-4 w-4 {inlineOn ? 'text-primary' : ''}" />
			<span class="hidden md:inline">{inlineOn ? 'AI on' : 'AI off'}</span>
		</Button>

		{#if sttDictationReady}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<Button
						variant="ghost"
						size="sm"
						class="h-8 w-8 rounded-full p-0 backdrop-blur-lg {isTranscribing
							? 'bg-blue-500 text-white hover:bg-blue-600'
							: isRecording
								? 'animate-pulse bg-red-500 text-white hover:bg-red-600'
								: ''}"
						onclick={handleMicClick}
						aria-label={isTranscribing
							? 'Cancel transcription'
							: isRecording
								? 'Stop dictation'
								: 'Start dictation'}
					>
						{#if isTranscribing}
							<Loader2 class="h-4 w-4 animate-spin" />
						{:else if isRecording}
							<Square class="h-4 w-4 animate-pulse fill-white" />
						{:else}
							<Mic class="h-4 w-4" />
						{/if}
					</Button>
				</Tooltip.Trigger>
				<Tooltip.Content>
					<p>
						{isTranscribing
							? 'Click to cancel transcription'
							: isRecording
								? 'Click to stop dictation'
								: 'Dictate into the document at the cursor'}
					</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}

		<ChainPicker class="hidden md:flex" {activeModelId} {isRouter} />

		<Popover.Root bind:open={systemPromptOpen}>
			<Popover.Trigger>
				{#snippet child({ props })}
					<Button
						{...props}
						variant="ghost"
						size="sm"
						class="gap-1.5 rounded-full backdrop-blur-lg"
						title="Edit system prompt (global)"
					>
						<NotebookPen class="h-4 w-4" />
						<span class="hidden md:inline">System</span>
					</Button>
				{/snippet}
			</Popover.Trigger>
			<Popover.Content class="w-[min(28rem,90vw)]">
				<div class="flex flex-col gap-2">
					<p class="text-sm font-medium">System prompt</p>
					<p class="text-xs text-muted-foreground">
						Applies to all new chats and to the AI commands run from this editor.
					</p>
					<textarea
						bind:value={systemPromptDraft}
						rows="6"
						class="w-full rounded-md border bg-background p-2 text-sm outline-none focus:ring-2 focus:ring-ring"
						placeholder="You are a helpful assistant…"
					></textarea>
					<div class="flex justify-end gap-2">
						<Button variant="ghost" size="sm" onclick={() => (systemPromptOpen = false)}>
							Cancel
						</Button>
						<Button size="sm" onclick={saveSystemPrompt}>Save</Button>
					</div>
				</div>
			</Popover.Content>
		</Popover.Root>

		<AiCommandsMenu onRun={onRunAiCommand} bind:open={commandsMenuOpen} />

		<Button
			variant="ghost"
			size="sm"
			onclick={onChatAbout}
			class="gap-1.5 rounded-full backdrop-blur-lg"
			title="Start a chat about this doc"
		>
			<MessageSquarePlus class="h-4 w-4" />
			<span class="hidden md:inline">Chat about this</span>
		</Button>

		<DocMoreActionsMenu docId={id} docName={name} docContent={content} {onDelete} />

		<BackendPill />

		<Button
			variant="ghost"
			size="icon-lg"
			onclick={() => chatSettingsDialog.open()}
			class="rounded-full backdrop-blur-lg"
		>
			<Settings class="h-4 w-4" />
		</Button>
	</div>
</header>
