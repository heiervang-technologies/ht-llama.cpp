<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { Button } from '$lib/components/ui/button';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import { ArrowLeft, Trash2 } from '@lucide/svelte';
	import { terminalsStore } from '$lib/stores/terminals.svelte';
	import { TerminalView } from '$lib/components/app/terminals';

	const sidebar = useSidebar();

	let id = $derived(page.params.id);
	let terminal = $derived(terminalsStore.terminals.find((t) => t.id === id) ?? null);

	onMount(async () => {
		if (terminalsStore.terminals.length === 0) {
			await terminalsStore.refresh();
		}
	});

	async function handleDestroy() {
		if (!terminal) return;
		if (!confirm(`Destroy terminal "${terminal.name}" and wipe its workspace?`)) return;
		const ok = await terminalsStore.destroy(terminal.id);
		if (ok) {
			toast.success('Terminal destroyed');
			goto('#/terminals');
		}
	}

	function handleDisconnect(clean: boolean) {
		if (clean) toast.message('Terminal disconnected');
		else toast.error('Terminal connection lost');
	}
</script>

<svelte:head>
	<title>{terminal?.name ?? 'Terminal'} · ht-llama.cpp</title>
</svelte:head>

<div
	class="flex h-full flex-col duration-200 ease-linear {sidebar.open
		? 'md:ml-[var(--sidebar-width)]'
		: ''}"
>
	<header class="flex items-center gap-2 border-b p-3 md:p-4">
		<Button variant="ghost" size="sm" onclick={() => goto('#/terminals')}>
			<ArrowLeft class="h-4 w-4" />
			<span class="hidden md:inline">All terminals</span>
		</Button>
		<div class="min-w-0 flex-1">
			<h1 class="truncate text-sm font-medium">{terminal?.name ?? id}</h1>
			{#if terminal}
				<p class="font-mono text-[10px] text-muted-foreground">
					{terminal.container_id.slice(0, 12)} · {terminal.image}
				</p>
			{/if}
		</div>
		<Button
			variant="ghost"
			size="sm"
			class="text-destructive hover:text-destructive"
			onclick={handleDestroy}
			disabled={!terminal}
		>
			<Trash2 class="h-4 w-4" />
		</Button>
	</header>

	<main class="min-h-0 flex-1">
		{#if terminal}
			{#key terminal.id}
				<TerminalView terminalId={terminal.id} onDisconnect={handleDisconnect} />
			{/key}
		{:else}
			<p class="p-6 text-sm text-muted-foreground">Loading terminal…</p>
		{/if}
	</main>
</div>
