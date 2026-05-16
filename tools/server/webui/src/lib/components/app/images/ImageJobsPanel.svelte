<script lang="ts">
	/**
	 * Persistent floating panel showing in-flight image / video gen jobs.
	 *
	 * Two tabs:
	 *   - Mine   — outbound HTTP requests this webview submitted (imageJobs).
	 *   - Server — comfy-openai proxy's /v1/images/queue view (the whole
	 *              cluster, not just this client). Polled when expanded.
	 *
	 * The "Server" tab is auto-hidden when the proxy doesn't ship the
	 * queue endpoint (older builds → 404 → soft state). Collapses to a
	 * small chip when idle. Renders only when at least one entry exists
	 * in either source.
	 */

	import { imageJobs, type ImageJob } from '$lib/stores/image-jobs.svelte';
	import { serverImageQueue } from '$lib/stores/server-image-queue.svelte';
	import {
		ImageIcon,
		Video,
		X,
		Loader2,
		CheckCircle2,
		XCircle,
		ChevronUp,
		ChevronDown,
		RefreshCw
	} from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	let collapsed = $state(true);
	let activeTab = $state<'mine' | 'server'>('mine');

	let jobs = $derived(imageJobs.jobs);
	let runningCount = $derived(imageJobs.runningCount);
	let serverSnapshot = $derived(serverImageQueue.snapshot);
	let serverUnavailable = $derived(serverImageQueue.lastError === 'unavailable');
	let serverHasAny = $derived(
		serverSnapshot.running.length +
			serverSnapshot.pending.length +
			serverSnapshot.recently_done.length >
			0
	);
	let serverRunningCount = $derived(serverSnapshot.running.length + serverSnapshot.pending.length);

	// Show the panel as long as there's something to look at. The server
	// snapshot can carry signal even when the webview hasn't submitted
	// anything (e.g. another client is generating).
	let panelVisible = $derived(jobs.length > 0 || serverHasAny);

	// Tick once per second so elapsed-time labels update.
	let now = $state(Date.now());
	$effect(() => {
		const t = setInterval(() => (now = Date.now()), 1000);
		return () => clearInterval(t);
	});

	// Subscribe to server queue polling whenever the panel is expanded.
	// When collapsed we still want a slow trickle so the badge counts
	// stay current — keep the subscription for the whole lifetime of
	// the panel mount, the store handles fast/slow cadence internally.
	$effect(() => {
		serverImageQueue.subscribe();
		return () => serverImageQueue.unsubscribe();
	});

	function elapsedLabel(j: ImageJob): string {
		const end = j.endedAt ?? now;
		const ms = Math.max(0, end - j.startedAt);
		return msLabel(ms);
	}

	function msLabel(ms: number | null): string {
		if (ms == null) return '—';
		const s = Math.floor(Math.max(0, ms) / 1000);
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

	function kindIcon(kind: string | null | undefined) {
		if (kind === 'video' || kind === 'i2v' || kind === 's2v' || kind === 'flf') return Video;
		return ImageIcon;
	}

	function statusColor(status: string): string {
		if (status === 'completed') return 'text-emerald-500';
		if (status === 'failed' || status === 'timeout') return 'text-destructive';
		if (status === 'cancelled') return 'text-muted-foreground';
		return 'text-muted-foreground';
	}

	function statusIconForServer(status: string) {
		if (status === 'completed') return CheckCircle2;
		if (status === 'failed' || status === 'timeout' || status === 'cancelled') return XCircle;
		return CheckCircle2;
	}

	function truncatePrompt(p: string, n = 60): string {
		const flat = p.replace(/\s+/g, ' ').trim();
		return flat.length > n ? flat.slice(0, n - 1) + '…' : flat;
	}

	function shortPromptId(id: string): string {
		return id.length > 10 ? id.slice(0, 8) + '…' : id;
	}
</script>

{#if panelVisible}
	<div
		class="fixed right-4 bottom-4 z-[1000] flex max-w-[28rem] min-w-[20rem] flex-col gap-2 rounded-lg border border-border bg-card/95 p-3 shadow-lg backdrop-blur-md"
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
				{#if runningCount > 0 || serverRunningCount > 0}
					<Loader2 class="h-4 w-4 animate-spin text-primary" />
					<span>
						{#if runningCount > 0}
							{runningCount} mine
						{/if}
						{#if runningCount > 0 && serverRunningCount > 0}
							·
						{/if}
						{#if serverRunningCount > 0}
							{serverRunningCount} on server
						{/if}
					</span>
				{:else}
					<CheckCircle2 class="h-4 w-4 text-muted-foreground" />
					<span class="text-muted-foreground">Image queue idle</span>
				{/if}
			</span>
			{#if collapsed}
				<ChevronUp class="h-4 w-4 text-muted-foreground" />
			{:else}
				<ChevronDown class="h-4 w-4 text-muted-foreground" />
			{/if}
		</button>

		{#if !collapsed}
			{#if !serverUnavailable}
				<div class="flex items-center gap-1 border-b border-border/40 pb-1 text-xs">
					<button
						type="button"
						class="rounded px-2 py-1 {activeTab === 'mine'
							? 'bg-muted text-foreground'
							: 'text-muted-foreground hover:text-foreground'}"
						onclick={() => (activeTab = 'mine')}
					>
						Mine ({jobs.length})
					</button>
					<button
						type="button"
						class="rounded px-2 py-1 {activeTab === 'server'
							? 'bg-muted text-foreground'
							: 'text-muted-foreground hover:text-foreground'}"
						onclick={() => (activeTab = 'server')}
					>
						Server ({serverSnapshot.running.length +
							serverSnapshot.pending.length +
							serverSnapshot.recently_done.length})
					</button>
					<div class="ml-auto">
						<Button
							variant="ghost"
							size="icon-sm"
							aria-label="Refresh"
							onclick={() => serverImageQueue.refresh()}
						>
							<RefreshCw class="h-3 w-3" />
						</Button>
					</div>
				</div>
			{/if}

			{#if activeTab === 'mine' || serverUnavailable}
				<ul class="flex max-h-[50vh] flex-col gap-2 overflow-y-auto">
					{#each jobs as job (job.id)}
						{@const StatusIcon = statusIcon(job)}
						{@const KindIcon = kindIcon(job.kind)}
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
			{:else}
				<ul class="flex max-h-[50vh] flex-col gap-2 overflow-y-auto">
					{#each serverSnapshot.running as item (item.prompt_id)}
						{@const KindIcon = kindIcon(item.kind)}
						<li class="flex items-start gap-2 rounded-md border border-primary/40 bg-primary/5 p-2 text-xs">
							<KindIcon class="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
							<div class="min-w-0 flex-1">
								<div class="flex items-center justify-between gap-2">
									<span class="truncate font-medium" title={item.model ?? ''}>
										{item.model ?? '—'}
									</span>
									<span class="shrink-0 font-mono text-[10px] text-muted-foreground">
										{msLabel(item.elapsed_ms)}
									</span>
								</div>
								<div class="truncate text-[10px] text-muted-foreground" title={item.prompt_id}>
									running · {shortPromptId(item.prompt_id)}
								</div>
							</div>
							<Loader2 class="h-3.5 w-3.5 shrink-0 animate-spin text-primary" />
						</li>
					{/each}
					{#each serverSnapshot.pending as item (item.prompt_id)}
						{@const KindIcon = kindIcon(item.kind)}
						<li class="flex items-start gap-2 rounded-md border border-border/50 bg-background/40 p-2 text-xs">
							<KindIcon class="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
							<div class="min-w-0 flex-1">
								<div class="flex items-center justify-between gap-2">
									<span class="truncate font-medium" title={item.model ?? ''}>
										{item.model ?? '—'}
									</span>
									<span class="shrink-0 font-mono text-[10px] text-muted-foreground">
										queued
									</span>
								</div>
								<div class="truncate text-[10px] text-muted-foreground" title={item.prompt_id}>
									pending · {shortPromptId(item.prompt_id)}
								</div>
							</div>
						</li>
					{/each}
					{#each serverSnapshot.recently_done as item (item.prompt_id)}
						{@const KindIcon = kindIcon(item.kind)}
						{@const StatusIcon = statusIconForServer(item.status)}
						<li
							class="flex items-start gap-2 rounded-md border border-border/50 bg-background/40 p-2 text-xs opacity-60"
						>
							<KindIcon class="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
							<div class="min-w-0 flex-1">
								<div class="flex items-center justify-between gap-2">
									<span class="truncate font-medium" title={item.model ?? ''}>
										{item.model ?? '—'}
									</span>
									<span class="shrink-0 font-mono text-[10px] text-muted-foreground">
										{msLabel(item.duration_ms)}
									</span>
								</div>
								<div class="truncate text-[10px] text-muted-foreground" title={item.prompt_id}>
									{item.status} · {shortPromptId(item.prompt_id)}
								</div>
							</div>
							<StatusIcon class={`h-3.5 w-3.5 shrink-0 ${statusColor(item.status)}`} />
						</li>
					{/each}
					{#if serverSnapshot.running.length + serverSnapshot.pending.length + serverSnapshot.recently_done.length === 0}
						<li class="rounded-md border border-dashed border-border/40 p-3 text-center text-xs text-muted-foreground">
							Server queue is empty.
						</li>
					{/if}
				</ul>
			{/if}
		{/if}
	</div>
{/if}
