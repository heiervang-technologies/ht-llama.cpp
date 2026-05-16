<script lang="ts">
	/**
	 * Persistent floating panel showing in-flight image / video gen jobs.
	 *
	 * Bottom-right corner; collapses to a small chip with the running
	 * count when idle. Only renders when at least one job is being
	 * tracked, so it's invisible if you never touch the image surface.
	 */

	import { imageJobs, type ImageJob } from '$lib/stores/image-jobs.svelte';
	import {
		ImageIcon,
		Video,
		X,
		Loader2,
		CheckCircle2,
		XCircle,
		ChevronUp,
		ChevronDown
	} from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	let collapsed = $state(true);
	let jobs = $derived(imageJobs.jobs);
	let runningCount = $derived(imageJobs.runningCount);

	// Tick once per second so elapsed-time labels update.
	let now = $state(Date.now());
	$effect(() => {
		const t = setInterval(() => (now = Date.now()), 1000);
		return () => clearInterval(t);
	});

	function elapsedLabel(j: ImageJob): string {
		const end = j.endedAt ?? now;
		const ms = Math.max(0, end - j.startedAt);
		const s = Math.floor(ms / 1000);
		if (s < 60) return `${s}s`;
		const m = Math.floor(s / 60);
		const rs = s % 60;
		return `${m}m ${rs}s`;
	}

	function statusIcon(j: ImageJob) {
		if (j.status === 'running') return Loader2;
		if (j.status === 'completed') return CheckCircle2;
		return XCircle;
	}

	function kindIcon(j: ImageJob) {
		if (j.kind === 'i2v' || j.kind === 's2v' || j.kind === 'flf') return Video;
		return ImageIcon;
	}

	function truncatePrompt(p: string, n = 60): string {
		const flat = p.replace(/\s+/g, ' ').trim();
		return flat.length > n ? flat.slice(0, n - 1) + '…' : flat;
	}
</script>

{#if jobs.length > 0}
	<div
		class="fixed right-4 bottom-4 z-[1000] flex max-w-[26rem] min-w-[18rem] flex-col gap-2 rounded-lg border border-border bg-card/95 p-3 shadow-lg backdrop-blur-md"
		role="region"
		aria-label="Image job queue"
	>
		<button
			type="button"
			class="flex items-center justify-between gap-2 text-left text-sm font-medium"
			onclick={() => (collapsed = !collapsed)}
			aria-expanded={!collapsed}
		>
			<span class="inline-flex items-center gap-2">
				{#if runningCount > 0}
					<Loader2 class="h-4 w-4 animate-spin text-primary" />
					<span>
						{runningCount} job{runningCount === 1 ? '' : 's'} running
					</span>
				{:else}
					<CheckCircle2 class="h-4 w-4 text-muted-foreground" />
					<span class="text-muted-foreground">
						{jobs.length} recent job{jobs.length === 1 ? '' : 's'}
					</span>
				{/if}
			</span>
			{#if collapsed}
				<ChevronUp class="h-4 w-4 text-muted-foreground" />
			{:else}
				<ChevronDown class="h-4 w-4 text-muted-foreground" />
			{/if}
		</button>

		{#if !collapsed}
			<ul class="flex max-h-[50vh] flex-col gap-2 overflow-y-auto">
				{#each jobs as job (job.id)}
					{@const StatusIcon = statusIcon(job)}
					{@const KindIcon = kindIcon(job)}
					<li
						class="flex items-start gap-2 rounded-md border border-border/50 bg-background/40 p-2 text-xs"
						class:opacity-60={job.status === 'completed' || job.status === 'cancelled'}
					>
						<KindIcon class="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
						<div class="min-w-0 flex-1">
							<div class="flex items-center justify-between gap-2">
								<span class="truncate font-medium" title={job.model}>{job.model}</span>
								<span class="shrink-0 font-mono text-[10px] text-muted-foreground">
									{elapsedLabel(job)}
								</span>
							</div>
							<div class="truncate text-muted-foreground" title={job.prompt}>
								{truncatePrompt(job.prompt)}
							</div>
							{#if job.status === 'failed' && job.error}
								<div class="mt-1 text-destructive">{job.error}</div>
							{/if}
						</div>
						<div class="flex shrink-0 items-center gap-1">
							<StatusIcon
								class={`h-3.5 w-3.5 ${job.status === 'running' ? 'animate-spin text-primary' : job.status === 'completed' ? 'text-emerald-500' : job.status === 'failed' ? 'text-destructive' : 'text-muted-foreground'}`}
							/>
							{#if job.status === 'running' && job.abort}
								<Button
									variant="ghost"
									size="icon-sm"
									aria-label="Cancel job"
									onclick={() => imageJobs.cancel(job.id)}
								>
									<X class="h-3 w-3" />
								</Button>
							{:else if job.status !== 'running'}
								<Button
									variant="ghost"
									size="icon-sm"
									aria-label="Dismiss"
									onclick={() => imageJobs.dismiss(job.id)}
								>
									<X class="h-3 w-3" />
								</Button>
							{/if}
						</div>
					</li>
				{/each}
			</ul>
		{/if}
	</div>
{/if}
