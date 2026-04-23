<script lang="ts">
	import { onMount, onDestroy as onTileDestroy } from 'svelte';
	import { goto } from '$app/navigation';
	import { Button } from '$lib/components/ui/button';
	import { Card } from '$lib/components/ui/card';
	import { TerminalSquare, Trash2, Clock } from '@lucide/svelte';
	import type { TerminalHandle } from '$lib/services/termd.service';

	interface Props {
		terminal: TerminalHandle;
		onDestroy: (id: string) => void;
	}

	let { terminal, onDestroy }: Props = $props();

	// Reactive clock: bumps once a minute so "just now" → "1m ago" → "2m ago"
	// updates without needing a route change. Kept coarse (60s) since the
	// label is coarse; avoids a sub-second repaint storm on the gallery.
	let now = $state(Date.now());
	let timer: ReturnType<typeof setInterval> | undefined;
	onMount(() => {
		timer = setInterval(() => (now = Date.now()), 60_000);
	});
	onTileDestroy(() => {
		if (timer) clearInterval(timer);
	});

	// The container_id we get from Docker is 64 hex chars — way too
	// long for a tile. xterm users care more about "when did I open
	// this", so we show an age label instead.
	let age = $derived.by(() => {
		const createdAt = terminal.created_at;
		if (!createdAt) return 'just now';
		// Docker's `created` from the list endpoint is seconds, not ms.
		const ts = createdAt > 1e12 ? createdAt : createdAt * 1000;
		const delta = now - ts;
		if (delta < 60_000) return 'just now';
		if (delta < 3_600_000) return `${Math.round(delta / 60_000)}m ago`;
		if (delta < 86_400_000) return `${Math.round(delta / 3_600_000)}h ago`;
		return `${Math.round(delta / 86_400_000)}d ago`;
	});

	function open() {
		goto(`#/terminals/${terminal.id}`);
	}

	function handleKeydown(e: KeyboardEvent) {
		// Enter / Space on a role="button" Card should navigate, matching
		// what a native <button> would do. Anything else passes through.
		if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			open();
		}
	}
</script>

<Card
	class="group flex cursor-pointer flex-col gap-2 overflow-hidden p-4 transition hover:border-primary/60 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
	role="button"
	tabindex={0}
	aria-label={`Open terminal ${terminal.name}`}
	onclick={open}
	onkeydown={handleKeydown}
>
	<header class="flex items-center justify-between gap-2">
		<div class="flex min-w-0 items-center gap-2">
			<TerminalSquare class="h-4 w-4 flex-shrink-0 text-primary" aria-hidden="true" />
			<span class="truncate text-sm font-medium">{terminal.name}</span>
		</div>
		<span
			class="flex-shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium uppercase
				{terminal.status === 'running'
				? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
				: 'bg-muted text-muted-foreground'}"
		>
			{terminal.status || 'unknown'}
		</span>
	</header>
	<div class="flex items-center gap-2 text-xs text-muted-foreground">
		<Clock class="h-3 w-3" aria-hidden="true" />
		<span>{age}</span>
	</div>
	<div class="truncate font-mono text-[11px] text-muted-foreground/70">
		{terminal.image}
	</div>
	<div class="flex items-center justify-end gap-1 opacity-0 transition group-hover:opacity-100">
		<Button
			variant="ghost"
			size="sm"
			class="h-7 px-2 text-destructive hover:text-destructive"
			onclick={(e) => {
				e.stopPropagation();
				if (confirm(`Destroy terminal "${terminal.name}" and wipe its workspace?`)) {
					onDestroy(terminal.id);
				}
			}}
			title="Destroy + wipe workspace"
		>
			<Trash2 class="h-3.5 w-3.5" />
		</Button>
	</div>
</Card>
