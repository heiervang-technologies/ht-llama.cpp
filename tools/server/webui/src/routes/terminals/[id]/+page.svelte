<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onMount, onDestroy } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import {
		ArrowLeft,
		Trash2,
		Palette,
		Users,
		Lock,
		ClipboardCheck,
		ScrollText
	} from '@lucide/svelte';
	import { terminalsStore } from '$lib/stores/terminals.svelte';
	import {
		terminalModes,
		TERMINAL_MODES,
		type TerminalMode
	} from '$lib/stores/terminal-modes.svelte';
	import { TerminalView } from '$lib/components/app/terminals';
	import {
		TERMINAL_THEMES,
		DEFAULT_TERMINAL_THEME_ID,
		resolveTheme
	} from '$lib/components/app/terminals/terminal-themes';
	import { TermdService } from '$lib/services/termd.service';

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

	// Interaction mode (per-terminal). Reactive: switching updates
	// both the picker and the small badge on the terminal header.
	let mode = $state<TerminalMode>('solo');
	$effect(() => {
		if (!id) return;
		mode = terminalModes.get(id);
	});
	function pickMode(next: TerminalMode) {
		if (!id) return;
		mode = next;
		terminalModes.set(id, next);
		toast.success(`Terminal mode → ${TERMINAL_MODES.find((m) => m.id === next)?.label}`);
	}

	const MODE_ICON: Record<TerminalMode, typeof Lock> = {
		solo: Lock,
		shared: Users,
		review: ClipboardCheck
	};

	// Bootstrap log panel. Polls every 2s while open so long-running
	// setup scripts (apt install, git clone) stay live-tailed. Closed
	// by default to keep the terminal the primary view.
	let logOpen = $state(false);
	let logText = $state('');
	let logTimer: ReturnType<typeof setInterval> | undefined;
	async function refreshLog() {
		if (!id) return;
		try {
			logText = await TermdService.bootstrapLog(id);
		} catch (err) {
			console.warn('[terminal] bootstrap log fetch failed', err);
		}
	}
	$effect(() => {
		if (logOpen) {
			refreshLog();
			logTimer = setInterval(refreshLog, 2000);
		} else if (logTimer) {
			clearInterval(logTimer);
			logTimer = undefined;
		}
	});

	onMount(async () => {
		if (terminalsStore.terminals.length === 0) {
			await terminalsStore.refresh();
		}
	});

	onDestroy(() => {
		if (logTimer) clearInterval(logTimer);
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

		<!-- Mode picker — pill variant so the current mode reads at a
			   glance; dropdown opens full descriptions. -->
		<DropdownMenu.Root>
			<DropdownMenu.Trigger>
				{#snippet child({ props })}
					<Button {...props} variant="ghost" size="sm" title="Who can type into this terminal">
						{@const Icon = MODE_ICON[mode]}
						<Icon class="h-4 w-4" />
						<span class="hidden capitalize md:inline">{mode}</span>
					</Button>
				{/snippet}
			</DropdownMenu.Trigger>
			<DropdownMenu.Content align="end" class="w-72">
				<DropdownMenu.Label class="text-xs text-muted-foreground">
					Interaction mode
				</DropdownMenu.Label>
				{#each TERMINAL_MODES as opt (opt.id)}
					{@const Icon = MODE_ICON[opt.id]}
					<DropdownMenu.Item
						onclick={() => pickMode(opt.id)}
						class={opt.id === mode ? 'bg-accent' : ''}
					>
						<Icon class="mr-2 h-3.5 w-3.5" />
						<div class="flex min-w-0 flex-col">
							<span class="font-medium">{opt.label}</span>
							<span class="text-[10px] whitespace-normal text-muted-foreground">
								{opt.description}
							</span>
						</div>
					</DropdownMenu.Item>
				{/each}
			</DropdownMenu.Content>
		</DropdownMenu.Root>

		<Button
			variant={logOpen ? 'default' : 'ghost'}
			size="sm"
			onclick={() => (logOpen = !logOpen)}
			title="Bootstrap log"
		>
			<ScrollText class="h-4 w-4" />
			<span class="hidden md:inline">Log</span>
		</Button>

		<DropdownMenu.Root>
			<DropdownMenu.Trigger>
				{#snippet child({ props })}
					<Button {...props} variant="ghost" size="sm" title="Terminal theme">
						<Palette class="h-4 w-4" />
						<span class="hidden md:inline">{activeTheme.label}</span>
					</Button>
				{/snippet}
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

	<main class="flex min-h-0 flex-1 gap-3 p-3 md:p-4">
		{#if terminal}
			{#key terminal.id}
				<div
					class="relative flex-1 overflow-hidden rounded-xl border border-border/60 p-2 shadow-sm"
					style:background={activeTheme.palette.background}
				>
					<TerminalView terminalId={terminal.id} {themeId} onDisconnect={handleDisconnect} />
				</div>
			{/key}
		{:else}
			<p class="p-6 text-sm text-muted-foreground">Loading terminal…</p>
		{/if}

		{#if logOpen}
			<aside
				class="flex w-80 flex-shrink-0 flex-col gap-2 overflow-hidden rounded-xl border bg-card p-3 shadow-sm"
			>
				<header class="flex items-center justify-between">
					<h2 class="flex items-center gap-1.5 text-xs font-semibold">
						<ScrollText class="h-3.5 w-3.5 text-primary" />
						Bootstrap log
					</h2>
					<Button variant="ghost" size="sm" class="h-6 px-2" onclick={refreshLog}>Refresh</Button>
				</header>
				<p class="text-[10px] text-muted-foreground">
					Output of the bootstrap script (if any) that ran once when this container started. Polls
					every 2s while this panel is open.
				</p>
				<pre
					class="min-h-0 flex-1 overflow-auto rounded-md bg-muted/40 p-2 font-mono text-[11px] leading-snug break-words whitespace-pre-wrap">{logText ||
						'(no bootstrap log yet)'}</pre>
			</aside>
		{/if}
	</main>
</div>
