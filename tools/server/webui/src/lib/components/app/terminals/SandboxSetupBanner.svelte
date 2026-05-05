<script lang="ts">
	import { AlertTriangle, ExternalLink } from '@lucide/svelte';
	import type { SandboxStatus } from '$lib/services/termd.service';

	interface Props {
		status: SandboxStatus | null;
	}

	let { status }: Props = $props();

	// Each row = one invariant the sidecar enforces in
	// `assert_sandbox_ready`. Showing them individually helps the user
	// understand *why* creation is refused, instead of a single opaque
	// "not ready" pill.
	const ROWS = [
		{ key: 'docker_ok', label: 'Docker daemon reachable' },
		{ key: 'runsc_ok', label: 'gVisor runtime (runsc) registered' },
		{ key: 'network_ok', label: 'unleash-sandbox network with icc=off' },
		{ key: 'iptables_ok', label: 'iptables LAN-drop rules' },
		{ key: 'image_ok', label: 'Container image unleash:latest present' }
	] as const;

	function flag(key: (typeof ROWS)[number]['key']): 'ok' | 'bad' | 'unknown' {
		if (!status) return 'unknown';
		const v = status[key];
		if (typeof v === 'boolean') return v ? 'ok' : 'bad';
		if (v === 'ok') return 'ok';
		if (v === 'unknown') return 'unknown';
		return 'bad';
	}
</script>

<div class="flex flex-col gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-4">
	<div class="flex items-center gap-2">
		<AlertTriangle class="h-4 w-4 text-amber-500" aria-hidden="true" />
		<h3 class="text-sm font-semibold">Sandbox setup incomplete</h3>
	</div>
	<p class="text-xs text-muted-foreground">
		The sidecar refuses to spawn containers until all four security invariants hold. Run the
		one-time setup on the host and refresh:
	</p>

	<pre
		class="overflow-x-auto rounded-md bg-black/40 p-2 text-xs leading-snug text-foreground/90">sudo unleash sandbox setup</pre>

	<ul class="flex flex-col gap-1 text-xs">
		{#each ROWS as row (row.key)}
			{@const state = flag(row.key)}
			<li class="flex items-center gap-2">
				<span
					class="inline-flex h-3.5 w-10 flex-shrink-0 items-center justify-center rounded-full text-[10px] font-medium uppercase
					{state === 'ok'
						? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
						: state === 'bad'
							? 'bg-destructive/15 text-destructive'
							: 'bg-muted text-muted-foreground'}"
				>
					{state === 'ok' ? 'ok' : state === 'bad' ? 'fix' : '?'}
				</span>
				<span>{row.label}</span>
			</li>
		{/each}
	</ul>

	<a
		href="https://github.com/heiervang-technologies/agent-tools"
		target="_blank"
		rel="noopener"
		class="inline-flex items-center gap-1 self-start text-xs underline"
	>
		unleash sandbox docs <ExternalLink class="h-3 w-3" />
	</a>
</div>
