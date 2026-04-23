<script lang="ts">
	import { Wrench, ChevronRight } from '@lucide/svelte';
	import { Card } from '$lib/components/ui/card';

	interface Props {
		class?: string;
		message: DatabaseMessage;
	}

	let { class: className = '', message }: Props = $props();

	// The tool result is the visible content; the `toolCallId` (set on
	// role=tool messages) lets us thread the display back to the calling
	// assistant turn if we ever want cross-links. For now we just show the
	// id as a dim monospace chip so users can correlate visually.
	let toolName = $derived((message.metadata?.toolName as string | undefined) ?? 'tool');
	let toolCallId = $derived(message.toolCallId ?? '');
	let content = $derived(message.content ?? '');

	let expanded = $state(false);
	// Pretty-print JSON payloads where possible; fall back to the raw
	// string. Tool results are frequently JSON from the agentic loop but
	// can be plain text for legacy MCP servers.
	let prettyContent = $derived.by(() => {
		if (!content.trim()) return '';
		try {
			const parsed = JSON.parse(content);
			return JSON.stringify(parsed, null, 2);
		} catch {
			return content;
		}
	});

	let preview = $derived(content.length > 160 ? `${content.slice(0, 160)}…` : content);
</script>

<div aria-label="Tool message" class="group flex w-full justify-start {className}" role="group">
	<Card
		class="w-full max-w-[85%] overflow-hidden rounded-[1.125rem] !border !border-dashed !border-border/60 bg-muted/40 px-3 py-2"
	>
		<button
			type="button"
			class="flex w-full items-start gap-2 text-left"
			onclick={() => (expanded = !expanded)}
			aria-expanded={expanded}
		>
			<Wrench class="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-muted-foreground" aria-hidden="true" />
			<div class="flex min-w-0 flex-1 flex-col gap-0.5">
				<div class="flex items-center gap-2 text-xs text-muted-foreground">
					<span class="font-medium tracking-wide uppercase">tool</span>
					<span class="font-mono">{toolName}</span>
					{#if toolCallId}
						<span class="truncate font-mono text-[10px] opacity-60">{toolCallId}</span>
					{/if}
				</div>
				{#if !expanded}
					<span class="truncate text-xs text-foreground/80">{preview || '(empty)'}</span>
				{/if}
			</div>
			<ChevronRight
				class="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-muted-foreground transition-transform {expanded
					? 'rotate-90'
					: ''}"
				aria-hidden="true"
			/>
		</button>

		{#if expanded}
			<pre
				class="mt-2 max-h-96 overflow-auto rounded-md bg-muted/60 p-2 text-[11px] leading-snug break-words whitespace-pre-wrap">{prettyContent ||
					'(empty)'}</pre>
		{/if}
	</Card>
</div>
