<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { toolTimings } from '$lib/stores/tool-timings.svelte';

	// Inline progress indicator for a tool call that is currently
	// pending. Reads the rolling median for this tool from
	// `toolTimings.median(name)` and fills the bar based on elapsed
	// vs that estimate. Past the estimate the bar pegs at 100% and
	// switches to an amber pulse so the user sees the call is now
	// taking longer than usual rather than being stuck — the
	// agentic loop will record the actual duration when the call
	// finally completes, which drags the estimate right back to
	// truth over the next few calls.
	//
	// 100 ms ticker is plenty for a smooth fill — coarser would
	// look choppy on sub-second calls, finer is wasted work since
	// human-perception threshold is ~50ms.

	interface Props {
		toolName: string | undefined;
	}

	let { toolName }: Props = $props();

	const startedAt = performance.now();
	let now = $state(performance.now());
	let timer: ReturnType<typeof setInterval> | undefined;

	onMount(() => {
		timer = setInterval(() => {
			now = performance.now();
		}, 100);
	});
	onDestroy(() => {
		if (timer) clearInterval(timer);
	});

	let elapsedMs = $derived(now - startedAt);
	let expectedMs = $derived(toolTimings.median(toolName) ?? 0);
	let progress = $derived(expectedMs > 0 ? Math.min(1, elapsedMs / expectedMs) : 0);
	let overdue = $derived(expectedMs > 0 && elapsedMs > expectedMs);
	let sampleCount = $derived(toolTimings.sampleCount(toolName));

	function fmt(ms: number): string {
		if (ms < 1000) return `${Math.round(ms)}ms`;
		if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
		const m = Math.floor(ms / 60_000);
		const s = Math.round((ms % 60_000) / 1000);
		return `${m}m ${s}s`;
	}
</script>

<div class="my-1 flex items-center gap-2 px-1 text-[10px] text-muted-foreground">
	<div class="relative h-1 flex-1 overflow-hidden rounded-full bg-muted">
		{#if expectedMs > 0}
			<div
				class="h-full rounded-full transition-[width] duration-150 ease-linear {overdue
					? 'animate-pulse bg-amber-500'
					: 'bg-primary'}"
				style="width: {progress * 100}%"
			></div>
		{:else}
			<!-- No estimate yet (first call of an untracked tool) —
			     indeterminate shimmer so the user sees activity. -->
			<div class="absolute inset-0 animate-pulse bg-primary/40"></div>
		{/if}
	</div>
	<span
		class="whitespace-nowrap tabular-nums {overdue ? 'text-amber-600 dark:text-amber-400' : ''}"
		title={sampleCount > 0
			? `median over ${sampleCount} past call${sampleCount === 1 ? '' : 's'}`
			: 'first call — using a seeded default estimate'}
	>
		{fmt(elapsedMs)}{#if expectedMs > 0}<span class="opacity-60"> / ~{fmt(expectedMs)}</span>{/if}
	</span>
</div>
