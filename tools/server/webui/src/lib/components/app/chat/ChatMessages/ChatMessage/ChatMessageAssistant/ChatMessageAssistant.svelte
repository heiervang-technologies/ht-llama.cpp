<script lang="ts">
	import {
		ChatMessageAgenticContent,
		ChatMessageActions,
		ChatMessageEditForm,
		ChatMessageStatistics,
		ModelBadge,
		ModelsSelectorDropdown
	} from '$lib/components/app';
	import { getMessageEditContext } from '$lib/contexts';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { isLoading, isChatStreaming } from '$lib/stores/chat.svelte';
	import { copyToClipboard, deriveAgenticSections } from '$lib/utils';
	import { AgenticSectionType } from '$lib/enums';
	import { REASONING_TAGS } from '$lib/constants/agentic';
	import { tick } from 'svelte';
	import { fade } from 'svelte/transition';
	import { MessageRole, ChatMessageStatsView } from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { ttsStore } from '$lib/stores/tts.svelte';
	import { artifactsStore } from '$lib/stores/artifacts.svelte';
	import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';
	import { extractGalleryArtifacts, hashString } from '$lib/utils/artifacts';
	import { SvelteMap } from 'svelte/reactivity';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { ServerModelStatus } from '$lib/enums';

	import { hasAgenticContent } from '$lib/utils';

	interface Props {
		class?: string;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		isLastAssistantMessage?: boolean;
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
		messageContent: string | undefined;
		onCopy: () => void;
		onConfirmDelete: () => void;
		onContinue?: () => void;
		onDelete: () => void;
		onEdit?: () => void;
		onForkConversation?: (options: { name: string; includeAttachments: boolean }) => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onRegenerate: (modelOverride?: string) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
		showDeleteDialog: boolean;
		siblingInfo?: ChatMessageSiblingInfo | null;
		textareaElement?: HTMLTextAreaElement;
	}

	let {
		class: className = '',
		deletionInfo,
		isLastAssistantMessage = false,
		message,
		toolMessages = [],
		messageContent,
		onConfirmDelete,
		onContinue,
		onCopy,
		onDelete,
		onEdit,
		onForkConversation,
		onNavigateToSibling,
		onRegenerate,
		onShowDeleteDialogChange,
		showDeleteDialog,
		siblingInfo = null,
		textareaElement = $bindable()
	}: Props = $props();

	// Get edit context
	const editCtx = getMessageEditContext();

	const isAgentic = $derived(hasAgenticContent(message, toolMessages));
	const hasReasoning = $derived(!!message.reasoningContent);
	const processingState = useProcessingState();

	let currentConfig = $derived(config());
	let isRouter = $derived(isRouterMode());
	let showRawOutput = $state(false);

	const ttsEnabled = $derived(Boolean(currentConfig.ttsEnabled) && ttsStore.isConfigured());
	const isSpeakingThis = $derived(ttsStore.speakingId === message.id);
	const isLoadingSpeechThis = $derived(ttsStore.loadingId === message.id);

	function stripForSpeech(text: string): string {
		return text
			.replace(/```[\s\S]*?```/g, ' ')
			.replace(/`([^`]+)`/g, '$1')
			.replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
			.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
			.replace(/[*_~#>]+/g, ' ')
			.replace(/\s+/g, ' ')
			.trim();
	}

	function handleSpeak() {
		const text = stripForSpeech(messageContent ?? '');
		if (!text) return;
		void ttsStore.toggle(message.id, text);
	}

	// Tracks which message id has already triggered autoplay so we never replay
	// after the user manually stops, after another message updates state, or
	// when ttsStore.speakingId transitions back to null.
	let autoplayedMessageId: string | null = null;

	// Only messages whose generation we witnessed in *this* session should
	// trigger autoplay — otherwise opening an existing chat would start
	// reading the last reply aloud unprompted.
	let hasSeenStreaming = $state(isChatStreaming() || isLoading());

	$effect(() => {
		if (isChatStreaming() || isLoading()) hasSeenStreaming = true;
	});

	$effect(() => {
		if (!ttsEnabled || !currentConfig.ttsAutoplay) return;
		if (!isLastAssistantMessage) return;
		if (isChatStreaming() || isLoading()) return;
		if (!hasSeenStreaming) return;
		if (autoplayedMessageId === message.id) return;
		const text = messageContent ?? '';
		if (!text.trim()) return;
		if (ttsStore.speakingId || ttsStore.loadingId) return;
		autoplayedMessageId = message.id;
		void ttsStore.speak(message.id, stripForSpeech(text));
	});

	// Register finished assistant content with the artifact store so HTML/SVG blocks
	// surface in the side drawer. Skips while streaming so we don't flicker on every
	// token — only the final content lands in the store.
	//
	// Component-local guard for the gallery capture path: $effect may re-run after
	// streaming ends (post-stream metadata updates, message object re-identity from
	// the store, remount during scroll virtualization). Without this guard, each
	// re-run fires another `captureFromChat` for the same slot, and the async
	// `findArtifactBySlot → createArtifact` pair inside the store races with itself,
	// producing duplicate artifacts ("same instance three times, multiplying each
	// turn"). We track the per-slot content hash we last submitted and short-circuit
	// on a match so the store only sees one call per unique content-per-slot.
	const submittedSlotHashes = new SvelteMap<string, string>();
	$effect(() => {
		if (isChatStreaming() || isLoading()) return;
		const text = messageContent ?? '';
		if (!text) return;
		artifactsStore.register(message.id, text);

		// Slash commands like `/image` already persist their artifact via
		// `saveManual` before writing the assistant message, and embed a
		// sentinel so we know the gallery entry exists. Auto-capture would
		// otherwise fire on the inline data URL and create a twin. Short-
		// circuit here — the sentinel is the contract between the slash
		// handler and this effect.
		if (text.includes('<!--ht-slash-artifact:')) return;

		// Also persist qualifying artifacts to the gallery. The slot id keys on
		// the message's parent, so regenerating the same turn (new assistant
		// sibling under the same user prompt) lands as a new revision of the
		// existing gallery entry rather than a twin artifact.
		const convId = message.convId;
		const slotParent = message.parent;
		if (!convId || !slotParent) return;
		const candidates = extractGalleryArtifacts(text);
		if (candidates.length === 0) return;
		for (const c of candidates) {
			const slot = `${slotParent}#${c.index}`;
			const contentKey = c.text != null ? hashString(c.text) : `blob:${c.blob?.size ?? 0}`;
			if (submittedSlotHashes.get(slot) === contentKey) continue;
			submittedSlotHashes.set(slot, contentKey);
			// Fire-and-forget; errors go to console rather than a toast so a
			// backgrounded capture can't hijack the chat UI.
			void artifactGalleryStore
				.captureFromChat(
					{
						conversationId: convId,
						slot,
						messageId: message.id,
						reason: 'initial'
					},
					{
						kind: c.kind,
						title: c.title,
						mimeType: c.mimeType,
						text: c.text,
						blob: c.blob,
						summary: c.summary
					}
				)
				.catch((err) => {
					// Undo the optimistic guard on failure so a retry with the same
					// content can actually retry.
					submittedSlotHashes.delete(slot);
					console.warn('[artifact-gallery] capture failed', err);
				});
		}
	});

	let rawOutputContent = $derived.by(() => {
		const sections = deriveAgenticSections(message, toolMessages, [], false);
		const parts: string[] = [];

		for (const section of sections) {
			switch (section.type) {
				case AgenticSectionType.REASONING:
				case AgenticSectionType.REASONING_PENDING:
					parts.push(`${REASONING_TAGS.START}\n${section.content}\n${REASONING_TAGS.END}`);
					break;

				case AgenticSectionType.TEXT:
					parts.push(section.content);
					break;

				case AgenticSectionType.TOOL_CALL:
				case AgenticSectionType.TOOL_CALL_PENDING:
				case AgenticSectionType.TOOL_CALL_STREAMING: {
					const callObj: Record<string, unknown> = { name: section.toolName };

					if (section.toolArgs) {
						try {
							callObj.arguments = JSON.parse(section.toolArgs);
						} catch {
							callObj.arguments = section.toolArgs;
						}
					}

					parts.push(JSON.stringify(callObj, null, 2));

					if (section.toolResult) {
						parts.push(`[Tool Result]\n${section.toolResult}`);
					}

					break;
				}
			}
		}

		return parts.join('\n\n\n');
	});

	let activeStatsView = $state<ChatMessageStatsView>(ChatMessageStatsView.GENERATION);
	let statsContainerEl: HTMLDivElement | undefined = $state();

	function getScrollParent(el: HTMLElement): HTMLElement | null {
		let parent = el.parentElement;
		while (parent) {
			const style = getComputedStyle(parent);
			if (/(auto|scroll)/.test(style.overflowY)) {
				return parent;
			}
			parent = parent.parentElement;
		}
		return null;
	}

	async function handleStatsViewChange(view: ChatMessageStatsView) {
		const el = statsContainerEl;
		if (!el) {
			activeStatsView = view;

			return;
		}

		const scrollParent = getScrollParent(el);
		if (!scrollParent) {
			activeStatsView = view;

			return;
		}

		const yBefore = el.getBoundingClientRect().top;

		activeStatsView = view;

		await tick();

		const delta = el.getBoundingClientRect().top - yBefore;
		if (delta !== 0) {
			scrollParent.scrollTop += delta;
		}

		// Correct any drift after browser paint
		requestAnimationFrame(() => {
			const drift = el.getBoundingClientRect().top - yBefore;

			if (Math.abs(drift) > 1) {
				scrollParent.scrollTop += drift;
			}
		});
	}

	let highlightAgenticTurns = $derived(
		isAgentic &&
			(currentConfig.alwaysShowAgenticTurns || activeStatsView === ChatMessageStatsView.SUMMARY)
	);

	let displayedModel = $derived(message.model ?? null);

	let isCurrentlyLoading = $derived(isLoading());
	let isStreaming = $derived(isChatStreaming());
	let hasNoContent = $derived(!message?.content?.trim());
	let isActivelyProcessing = $derived(isCurrentlyLoading || isStreaming);

	let showProcessingInfoTop = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			hasNoContent &&
			!isAgentic &&
			isLastAssistantMessage
	);

	let showProcessingInfoBottom = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			(!hasNoContent || isAgentic) &&
			isLastAssistantMessage
	);

	function handleCopyModel() {
		void copyToClipboard(displayedModel ?? '');
	}

	$effect(() => {
		if (showProcessingInfoTop || showProcessingInfoBottom) {
			processingState.startMonitoring();
		}
	});
</script>

<div
	class="text-md group w-full leading-7.5 {className}"
	role="group"
	aria-label="Assistant message with actions"
>
	{#if showProcessingInfoTop}
		<div class="mt-6 w-full max-w-[48rem]" in:fade>
			<div class="processing-container">
				<span class="processing-text">
					{processingState.getPromptProgressText() ??
						processingState.getProcessingMessage() ??
						'Processing...'}
				</span>
			</div>
		</div>
	{/if}

	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else if message.role === MessageRole.ASSISTANT}
		{#if showRawOutput}
			<pre class="raw-output">{rawOutputContent || ''}</pre>
		{:else}
			<ChatMessageAgenticContent
				{message}
				{toolMessages}
				isStreaming={isChatStreaming()}
				{isLastAssistantMessage}
				highlightTurns={highlightAgenticTurns}
			/>
		{/if}
	{:else}
		<div class="text-sm whitespace-pre-wrap">
			{messageContent}
		</div>
	{/if}

	{#if showProcessingInfoBottom}
		<div class="mt-4 w-full max-w-[48rem]" in:fade>
			<div class="processing-container">
				<span class="processing-text">
					{processingState.getPromptProgressText() ??
						processingState.getProcessingMessage() ??
						'Processing...'}
				</span>
			</div>
		</div>
	{/if}

	<div class="info my-6 grid gap-4 tabular-nums">
		{#if displayedModel}
			<div
				bind:this={statsContainerEl}
				class="inline-flex flex-wrap items-start gap-2 text-xs text-muted-foreground"
			>
				{#if isRouter}
					<ModelsSelectorDropdown
						currentModel={displayedModel}
						disabled={isLoading()}
						onModelChange={async (modelId: string, modelName: string) => {
							const status = modelsStore.getModelStatus(modelId);

							if (status !== ServerModelStatus.LOADED) {
								await modelsStore.loadModel(modelId);
							}

							onRegenerate(modelName);
							return true;
						}}
					/>
				{:else}
					<ModelBadge model={displayedModel || undefined} onclick={handleCopyModel} />
				{/if}

				{#if currentConfig.showMessageStats && message.timings && message.timings.predicted_n && message.timings.predicted_ms}
					{@const agentic = message.timings.agentic}
					<ChatMessageStatistics
						promptTokens={agentic ? agentic.llm.prompt_n : message.timings.prompt_n}
						promptMs={agentic ? agentic.llm.prompt_ms : message.timings.prompt_ms}
						predictedTokens={agentic ? agentic.llm.predicted_n : message.timings.predicted_n}
						predictedMs={agentic ? agentic.llm.predicted_ms : message.timings.predicted_ms}
						agenticTimings={agentic}
						onActiveViewChange={handleStatsViewChange}
					/>
				{:else if isLoading() && currentConfig.showMessageStats}
					{@const liveStats = processingState.getLiveProcessingStats()}
					{@const genStats = processingState.getLiveGenerationStats()}
					{@const promptProgress = processingState.processingState?.promptProgress}
					{@const isStillProcessingPrompt =
						promptProgress && promptProgress.processed < promptProgress.total}

					{#if liveStats || genStats}
						<ChatMessageStatistics
							isLive
							isProcessingPrompt={!!isStillProcessingPrompt}
							promptTokens={liveStats?.tokensProcessed}
							promptMs={liveStats?.timeMs}
							predictedTokens={genStats?.tokensGenerated}
							predictedMs={genStats?.timeMs}
						/>
					{/if}
				{/if}
			</div>
		{/if}
	</div>

	{#if message.timestamp && !editCtx.isEditing}
		<ChatMessageActions
			role={MessageRole.ASSISTANT}
			justify="start"
			actionsPosition="left"
			{siblingInfo}
			{showDeleteDialog}
			{deletionInfo}
			{onCopy}
			{onEdit}
			{onRegenerate}
			onContinue={currentConfig.enableContinueGeneration && !hasReasoning ? onContinue : undefined}
			{onForkConversation}
			{onDelete}
			{onConfirmDelete}
			{onNavigateToSibling}
			{onShowDeleteDialogChange}
			showRawOutputSwitch={currentConfig.showRawOutputSwitch}
			rawOutputEnabled={showRawOutput}
			onRawOutputToggle={(enabled: boolean) => (showRawOutput = enabled)}
			onSpeak={ttsEnabled ? handleSpeak : undefined}
			isSpeaking={isSpeakingThis}
			isLoadingSpeech={isLoadingSpeechThis}
		/>
	{/if}
</div>

<style>
	.processing-container {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.5rem;
	}

	.processing-text {
		background: linear-gradient(
			90deg,
			var(--muted-foreground),
			var(--foreground),
			var(--muted-foreground)
		);
		background-size: 200% 100%;
		background-clip: text;
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		animation: shine 1s linear infinite;
		font-weight: 500;
		font-size: 0.875rem;
	}

	@keyframes shine {
		to {
			background-position: -200% 0;
		}
	}

	.raw-output {
		width: 100%;
		max-width: 48rem;
		margin-top: 1.5rem;
		padding: 1rem 1.25rem;
		border-radius: 1rem;
		background: hsl(var(--muted) / 0.3);
		color: var(--foreground);
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas,
			'Liberation Mono', Menlo, monospace;
		font-size: 0.875rem;
		line-height: 1.6;
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>
