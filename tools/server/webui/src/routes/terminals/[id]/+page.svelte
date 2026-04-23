<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { ArrowLeft, Trash2, Palette } from '@lucide/svelte';
	import { terminalsStore } from '$lib/stores/terminals.svelte';
	import { TerminalView } from '$lib/components/app/terminals';
	import {
		TERMINAL_THEMES,
		DEFAULT_TERMINAL_THEME_ID,
		resolveTheme
	} from '$lib/components/app/terminals/terminal-themes';

	let id = $derived(page.params.id);
	let terminal = $derived(terminalsStore.terminals.find((t) => t.id === id) ?? null);

	// Theme selection persists in localStorage per-device so a user's
	// preferred CRT skin survives a reload without adding another
	// settings field.
	const THEME_KEY = 'ht-llama.terminalThemeId';
	let themeId = $state<string>(DEFAULT_TERMINAL_THEME_ID);
	if (typeof localStorage !== 'undefined') {
		themeId = localStorage.getItem(THEME_KEY) ?? DEFAULT_TERMINAL_THEME_ID;
	}
	let activeTheme = $derived(resolveTheme(themeId));
	function pickTheme(next: string) {
		themeId = next;
		try {
			localStorage.setItem(THEME_KEY, next);
		} catch {
			/* private mode — ignore */
		}
	}

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

<div class="flex h-full flex-col">
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
		<DropdownMenu.Root>
			<DropdownMenu.Trigger>
				<Button variant="ghost" size="sm" title="Terminal theme">
					<Palette class="h-4 w-4" />
					<span class="hidden md:inline">{activeTheme.label}</span>
				</Button>
			</DropdownMenu.Trigger>
			<DropdownMenu.Content align="end" class="w-64">
				{#each TERMINAL_THEMES as opt (opt.id)}
					<DropdownMenu.Item
						onclick={() => pickTheme(opt.id)}
						class={opt.id === themeId ? 'bg-accent' : ''}
					>
						<div class="flex flex-col">
							<span class="font-medium">{opt.label}</span>
							<span class="text-[10px] text-muted-foreground">{opt.description}</span>
						</div>
					</DropdownMenu.Item>
				{/each}
			</DropdownMenu.Content>
		</DropdownMenu.Root>

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

	<main class="min-h-0 flex-1 p-3 md:p-4">
		{#if terminal}
			{#key terminal.id}
				<div
					class="relative h-full w-full overflow-hidden rounded-xl border border-border/60 p-2 shadow-sm"
					style:background={activeTheme.palette.background}
				>
					<TerminalView terminalId={terminal.id} {themeId} onDisconnect={handleDisconnect} />
				</div>
			{/key}
		{:else}
			<p class="p-6 text-sm text-muted-foreground">Loading terminal…</p>
		{/if}
	</main>
</div>
