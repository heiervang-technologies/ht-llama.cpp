<script lang="ts">
	import { Wrench, ChevronRight, Settings2, Eye } from '@lucide/svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { Card } from '$lib/components/ui/card';
	import { config } from '$lib/stores/settings.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import type { OpenAIToolDefinition } from '$lib/types/mcp';

	interface Props {
		class?: string;
	}

	let { class: className = '' }: Props = $props();

	// Reactive view of what would be sent to the model on the very next turn:
	// the system prompt (as configured) + the full tool surface (built-ins +
	// any connected MCP servers). This is "phantom" — it renders above the
	// first real message so the user can eyeball prompt context at a glance,
	// but it is not itself a turn in history.
	let systemMessage = $derived((config().systemMessage ?? '').trim());

	let tools = $derived<OpenAIToolDefinition[]>(mcpStore.getToolDefinitionsForLLM());

	let hasContent = $derived(systemMessage.length > 0 || tools.length > 0);

	let systemExpanded = $state(false);
	let expandedTools = new SvelteSet<string>();

	function toggleTool(name: string) {
		if (expandedTools.has(name)) expandedTools.delete(name);
		else expandedTools.add(name);
	}

	function describeSchema(def: OpenAIToolDefinition): string {
		const params = def.function.parameters as Record<string, unknown> | undefined;
		return JSON.stringify(params ?? {}, null, 2);
	}
</script>

{#if hasContent}
	<section
		aria-label="Phantom context — visible to you only"
		class="mx-auto flex w-full max-w-[48rem] flex-col gap-2 {className}"
	>
		<header class="flex items-center gap-2 px-1 text-xs text-muted-foreground">
			<Eye class="h-3.5 w-3.5" aria-hidden="true" />
			<span class="tracking-wide uppercase">Phantom context</span>
			<span class="opacity-60">— what the model sees before your next turn</span>
		</header>

		{#if systemMessage}
			<Card
				class="w-full overflow-hidden rounded-[1.125rem] !border !border-dashed !border-border/60 bg-muted/30 px-3 py-2"
			>
				<button
					type="button"
					class="flex w-full items-start gap-2 text-left"
					onclick={() => (systemExpanded = !systemExpanded)}
					aria-expanded={systemExpanded}
				>
					<Settings2
						class="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-muted-foreground"
						aria-hidden="true"
					/>
					<div class="flex min-w-0 flex-1 flex-col gap-0.5">
						<div class="flex items-center gap-2 text-xs text-muted-foreground">
							<span class="font-medium tracking-wide uppercase">system</span>
							<span class="opacity-60">prompt</span>
						</div>
						{#if !systemExpanded}
							<span class="truncate text-xs text-foreground/80">{systemMessage}</span>
						{/if}
					</div>
					<ChevronRight
						class="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-muted-foreground transition-transform {systemExpanded
							? 'rotate-90'
							: ''}"
						aria-hidden="true"
					/>
				</button>
				{#if systemExpanded}
					<pre
						class="mt-2 max-h-96 overflow-auto rounded-md bg-muted/60 p-2 text-[11px] leading-snug break-words whitespace-pre-wrap">{systemMessage}</pre>
				{/if}
			</Card>
		{/if}

		{#if tools.length > 0}
			<Card
				class="w-full overflow-hidden rounded-[1.125rem] !border !border-dashed !border-border/60 bg-muted/30 px-3 py-2"
			>
				<div class="flex items-center gap-2 text-xs text-muted-foreground">
					<Wrench class="h-3.5 w-3.5" aria-hidden="true" />
					<span class="font-medium tracking-wide uppercase">tools</span>
					<span class="opacity-60">{tools.length} available</span>
				</div>
				<ul class="mt-1 flex flex-col divide-y divide-border/40">
					{#each tools as tool (tool.function.name)}
						{@const expanded = expandedTools.has(tool.function.name)}
						<li class="py-1.5">
							<button
								type="button"
								class="flex w-full items-start gap-2 text-left"
								onclick={() => toggleTool(tool.function.name)}
								aria-expanded={expanded}
							>
								<div class="flex min-w-0 flex-1 flex-col gap-0.5">
									<span class="font-mono text-xs text-foreground/90">{tool.function.name}</span>
									{#if !expanded && tool.function.description}
										<span class="truncate text-[11px] text-muted-foreground">
											{tool.function.description}
										</span>
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
								{#if tool.function.description}
									<p class="mt-1 text-[11px] leading-snug text-muted-foreground">
										{tool.function.description}
									</p>
								{/if}
								<pre
									class="mt-1 max-h-64 overflow-auto rounded-md bg-muted/60 p-2 text-[11px] leading-snug break-words whitespace-pre-wrap">{describeSchema(
										tool
									)}</pre>
							{/if}
						</li>
					{/each}
				</ul>
			</Card>
		{/if}
	</section>
{/if}
