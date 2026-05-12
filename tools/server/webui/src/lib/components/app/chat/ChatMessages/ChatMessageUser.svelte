<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { ChatAttachmentsList, MarkdownContent } from '$lib/components/app';
	import { getMessageEditContext } from '$lib/contexts';
	import { config } from '$lib/stores/settings.svelte';
	import { ChatMessageActions } from '$lib/components/app';
	import ChatMessageEditForm from './ChatMessageEditForm.svelte';
	import { AttachmentType, MessageRole } from '$lib/enums';
	import { isInlineImageExtra } from '$lib/utils/extract-markdown-images';
	import { RefreshCw, ChevronDown, ChevronRight, CircleX } from '@lucide/svelte';
	import { isPatchRepairSource } from '$lib/editor/ai-patch/types';
	import { getPatchSession, stopPatchRepairLoop } from '$lib/editor/ai-patch/session-registry';

	interface Props {
		class?: string;
		message?: DatabaseMessage;
		siblingInfo?: ChatMessageSiblingInfo | null;
		deletionInfo?: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		showDeleteDialog?: boolean;
		onEdit?: () => void;
		onDelete?: () => void;
		onConfirmDelete?: () => void;
		onForkConversation?: (options: { name: string; includeAttachments: boolean }) => void;
		onShowDeleteDialogChange?: (show: boolean) => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onCopy?: () => void;
		content?: string;
		attachments?: DatabaseMessageExtra[];
		renderMarkdown?: boolean;
		textColorClass?: string;
		cardBgClass?: string;
		maxHeightStyle?: string;
	}

	let {
		class: className = '',
		message,
		siblingInfo = null,
		deletionInfo,
		showDeleteDialog,
		onEdit,
		onDelete,
		onConfirmDelete,
		onForkConversation,
		onShowDeleteDialogChange,
		onNavigateToSibling,
		onCopy
	}: Props = $props();

	// Get contexts
	const editCtx = getMessageEditContext();

	let isMultiline = $state(false);
	let messageElement: HTMLElement | undefined = $state();
	const currentConfig = config();

	// Patch-repair source — an auto-generated user turn produced by the
	// ai-patch repair loop. Rendered with muted styling and collapsed by
	// default so the retry prompt doesn't visually dominate a thread of
	// human turns. See `$lib/editor/ai-patch/repair-loop.ts` for the body
	// format and `$lib/editor/ai-patch/types.ts:MessageSource` for the
	// provenance shape.
	const repairSource = $derived.by(() => {
		const src = (message.metadata as { source?: unknown } | undefined)?.source;
		return isPatchRepairSource(src) ? src : null;
	});
	let repairExpanded = $state(false);
	// First non-empty line of the body, for the collapsed preview. Falls
	// back to the full content when the body has no newline — that's
	// unusual for a repair turn (the formatter always emits a header) but
	// defensive against empty/stubbed bodies.
	const repairPreviewLine = $derived.by(() => {
		if (!repairSource) return '';
		const content: string = message.content ?? '';
		const firstLine = content.split('\n').find((l: string) => l.trim().length > 0);
		return firstLine ?? '';
	});

	// Whether the parent patch-repair loop is still accepting retries. The
	// registry holds a handle while the loop is running; once it ends
	// (committed / exhausted / aborted) the handle is unregistered and the
	// × affordance disappears. The underlying store is a `SvelteMap`
	// (see `$lib/editor/ai-patch/session-registry.ts`), so `getPatchSession`
	// reads are tracked inside this `$derived` — we don't need to poll or
	// invalidate on a parent re-render.
	const repairLoopLive = $derived.by(() => {
		if (!repairSource) return false;
		const handle = getPatchSession(repairSource.parentSessionId);
		return handle !== null;
	});

	function discardRepairLoop() {
		if (!repairSource) return;
		stopPatchRepairLoop(repairSource.parentSessionId);
	}

	// Hide attachment chips for image extras that were lifted from inline
	// `![](data:image/...)` markdown — the same bytes already render in the
	// message body, so a chip would be a duplicate. Only dedup when markdown
	// rendering is on, otherwise the chip is the only place the user sees it.
	const visibleExtras = $derived.by(() => {
		if (!message.extra) return [];
		if (!currentConfig.renderUserContentAsMarkdown) return message.extra;
		const content = message.content ?? '';
		return message.extra.filter((e: DatabaseMessageExtra) => {
			if (e.type !== AttachmentType.IMAGE) return true;
			return !isInlineImageExtra(e, content);
		});
	});

	$effect(() => {
		if (!messageElement || !message.content.trim()) return;

		if (message.content.includes('\n')) {
			isMultiline = true;
			return;
		}

		const resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const element = entry.target as HTMLElement;
				const estimatedSingleLineHeight = 24; // Typical line height for text-md

				isMultiline = element.offsetHeight > estimatedSingleLineHeight * 1.5;
			}
		});

		resizeObserver.observe(messageElement);

		return () => {
			resizeObserver.disconnect();
		};
	});
</script>

<div
	aria-label="User message with actions"
	class="group flex flex-col items-end gap-3 md:gap-2 {className}"
	role="group"
>
	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else}
		{#if visibleExtras.length > 0}
			<div class="mb-2 max-w-[80%]">
				<ChatAttachmentsList attachments={visibleExtras} readonly imageHeight="h-80" />
			</div>
		{/if}

		{#if message.content.trim()}
			{#if repairSource}
				<!--
					Patch-repair synthetic turn. Muted styling, chip, italic
					attribution, collapsed by default. The body is always
					rendered via MarkdownContent because the repair formatter
					emits fenced-code snippets the model needs to see
					monospaced regardless of the user's renderUserContentAsMarkdown
					preference.
				-->
				<Card
					class="bg-base-200/40 text-base-content/70 max-w-[80%] overflow-y-auto rounded-[1.125rem] border border-muted-foreground/20 px-3.75 py-2 backdrop-blur-md"
					style="max-height: var(--max-message-height); overflow-wrap: anywhere; word-break: break-word;"
					aria-label="Auto-generated patch-feedback turn"
				>
					<div class="flex items-center gap-1">
						<button
							type="button"
							class="text-base-content/80 hover:text-base-content flex flex-1 items-center gap-1.5 text-left text-xs font-medium"
							onclick={() => (repairExpanded = !repairExpanded)}
							aria-expanded={repairExpanded}
						>
							{#if repairExpanded}
								<ChevronDown class="h-3.5 w-3.5" aria-hidden="true" />
							{:else}
								<ChevronRight class="h-3.5 w-3.5" aria-hidden="true" />
							{/if}
							<RefreshCw class="h-3.5 w-3.5" aria-hidden="true" />
							<span>Patch feedback</span>
						</button>
						{#if repairLoopLive}
							<!--
								Stop-retrying affordance. Present only while the parent
								repair loop is in flight; disappears once the loop
								terminates (committed / exhausted / aborted). The chip
								above stays — it's the message's identity and navigates
								conversation history even after the loop ends.
							-->
							<button
								type="button"
								aria-label="Stop retrying patch"
								title="Stop retrying"
								class="text-base-content/60 hover:text-base-content flex h-5 w-5 items-center justify-center rounded-sm hover:bg-accent"
								onclick={discardRepairLoop}
							>
								<CircleX class="h-3.5 w-3.5" aria-hidden="true" />
							</button>
						{/if}
					</div>
					<div class="text-base-content/60 mt-1 text-[0.7rem] italic">
						Auto-generated from parser errors — {repairSource.failureCode} (block {repairSource.blockIndex +
							1}, retry {repairSource.reflection})
					</div>
					{#if repairExpanded}
						<div class="mt-2" bind:this={messageElement}>
							<MarkdownContent class="markdown-user-content -my-4" content={message.content} />
						</div>
					{:else if repairPreviewLine}
						<div
							bind:this={messageElement}
							class="text-base-content/70 mt-2 truncate text-sm"
							title={repairPreviewLine}
						>
							{repairPreviewLine}
						</div>
					{/if}
				</Card>
			{:else}
				<Card
					class="max-w-[80%] overflow-y-auto rounded-[1.125rem] border-none bg-primary/5 px-3.75 py-1.5 text-foreground backdrop-blur-md data-[multiline]:py-2.5 dark:bg-primary/15"
					data-multiline={isMultiline ? '' : undefined}
					style="max-height: var(--max-message-height); overflow-wrap: anywhere; word-break: break-word;"
				>
					{#if currentConfig.renderUserContentAsMarkdown}
						<div bind:this={messageElement}>
							<MarkdownContent class="markdown-user-content -my-4" content={message.content} />
						</div>
					{:else}
						<span bind:this={messageElement} class="text-md whitespace-pre-wrap">
							{message.content}
						</span>
					{/if}
				</Card>
			{/if}
		{/if}

		{#if message.timestamp}
			<div class="max-w-[80%]">
				<ChatMessageActions
					actionsPosition="right"
					{deletionInfo}
					justify="end"
					{onConfirmDelete}
					{onCopy}
					{onDelete}
					{onEdit}
					{onForkConversation}
					{onNavigateToSibling}
					{onShowDeleteDialogChange}
					{siblingInfo}
					{showDeleteDialog}
					role={MessageRole.USER}
				/>
			</div>
		{/if}
	{/if}
</div>
