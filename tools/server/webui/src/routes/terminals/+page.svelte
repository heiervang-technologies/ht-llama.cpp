<script lang="ts">
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { Button } from '$lib/components/ui/button';
	import { TerminalSquare, Plus, RefreshCw } from '@lucide/svelte';
	import { terminalsStore } from '$lib/stores/terminals.svelte';
	import {
		SandboxSetupBanner,
		TerminalTile,
		CreateTerminalDialog
	} from '$lib/components/app/terminals';
	import { resolveTermdUrl, type CreateTerminalBody } from '$lib/services/termd.service';

	let showCreateDialog = $state(false);

	let creating = $derived(terminalsStore.creating);
	let terminals = $derived(terminalsStore.terminals);
	let sandbox = $derived(terminalsStore.sandbox);
	let needsSetup = $derived(terminalsStore.needsSetup);
	let error = $derived(terminalsStore.error);
	let available = $derived(terminalsStore.available);

	onMount(() => {
		terminalsStore.refresh();
	});

	async function handleCreate(body: CreateTerminalBody) {
		showCreateDialog = false;
		const t = await terminalsStore.create(body);
		if (t) toast.success(`Spawned sandbox "${t.name}"`);
	}

	async function handleDestroy(id: string) {
		const ok = await terminalsStore.destroy(id);
		if (ok) toast.success('Terminal destroyed');
	}

	$effect(() => {
		if (error) toast.error(error);
	});
</script>

<svelte:head>
	<title>Sandbox terminals · heierchat</title>
</svelte:head>

<div class="flex h-full flex-col gap-4 p-4 md:p-6">
	<header class="flex items-center gap-3 border-b pb-3">
		<TerminalSquare class="h-5 w-5 text-primary" />
		<div class="flex-1">
			<h1 class="text-lg font-semibold">Sandbox terminals</h1>
			<p class="text-xs text-muted-foreground">
				gVisor-hardened containers on the unleash-sandbox network. Internet yes, LAN no.
			</p>
		</div>
		<Button variant="outline" size="sm" onclick={() => terminalsStore.refresh()}>
			<RefreshCw class="h-3.5 w-3.5" />
			<span class="hidden sm:inline">Refresh</span>
		</Button>
		<Button
			size="sm"
			onclick={() => (showCreateDialog = true)}
			disabled={creating || needsSetup || !available}
		>
			<Plus class="h-3.5 w-3.5" />
			{creating ? 'Spawning…' : 'New terminal'}
		</Button>
	</header>

	{#if !available}
		<div class="rounded-lg border border-dashed p-6 text-sm text-muted-foreground">
			<p class="mb-2 font-medium">ht-termd not configured.</p>
			<p class="text-xs">
				The Tauri app auto-spawns the sidecar on launch. For the web UI, start <code>ht-termd</code>
				on the host and either pass
				<code>--termd-url http://127.0.0.1:43127</code>
				to <code>llama-server</code>, or paste the URL into Settings → Terminals.
			</p>
		</div>
	{:else if needsSetup}
		<SandboxSetupBanner status={sandbox} />
	{/if}

	{#if available}
		{#if terminals.length === 0 && !needsSetup}
			<div
				class="flex flex-1 flex-col items-center justify-center gap-3 rounded-lg border border-dashed p-10 text-center text-sm text-muted-foreground"
			>
				<TerminalSquare class="h-8 w-8 opacity-40" />
				<p>No sandboxes yet.</p>
				<Button size="sm" onclick={() => (showCreateDialog = true)} disabled={creating}>
					<Plus class="h-3.5 w-3.5" />
					Spawn your first terminal
				</Button>
			</div>
		{:else if terminals.length > 0}
			<div class="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
				{#each terminals as terminal (terminal.id)}
					<TerminalTile {terminal} onDestroy={handleDestroy} />
				{/each}
			</div>
		{/if}
	{/if}

	<footer class="pt-4 text-[11px] text-muted-foreground/70">
		{#if available}
			<span>Connected to <code>{resolveTermdUrl()}</code></span>
		{/if}
	</footer>
</div>

<CreateTerminalDialog bind:open={showCreateDialog} onSubmit={handleCreate} />
