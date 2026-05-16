<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onMount, onDestroy } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import {
		ArrowLeft,
		Trash2,
		Palette,
		Users,
		Lock,
		ClipboardCheck,
		ScrollText,
		Check,
		X
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
	import { terminalProposals, type TerminalProposal } from '$lib/stores/terminal-proposals.svelte';

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

	// Review-mode proposal queue. Each entry is a keystroke the
	// model asked to send; the user approves or rejects. Approve
	// dispatches through `TermdService.sendInput` — the same
	// path `send_keys` would take in Shared mode.
	let pendingProposals = $derived<TerminalProposal[]>(id ? terminalProposals.pending(id) : []);

	async function approveProposal(p: TerminalProposal) {
		try {
			await TermdService.sendInput(p.terminalId, {
				text: p.text,
				auto_enter: p.autoEnter
			});
			terminalProposals.remove(p.id);
			toast.success('Keystrokes approved');
		} catch (err) {
			toast.error(`Approve failed: ${err instanceof Error ? err.message : String(err)}`);
		}
	}

	function rejectProposal(p: TerminalProposal) {
		terminalProposals.remove(p.id);
		toast.message('Proposal rejected');
	}

	function preview(text: string): string {
		// Escape visible control chars so \r, Ctrl+C, ESC etc. don't
		// vanish into the monospace block. Keeps the preview truthful
		// about what the model actually wants to send. We avoid a regex
		// literal with any raw 0x1b / 0x00-0x1f bytes — ESLint's
		// `no-control-regex` rule forbids them — so the sweep runs as
		// a char-by-char loop.
		const esc = String.fromCharCode(0x1b);
		let out = text
			.split(esc)
			.join('\\e')
			.replace(/\r/g, '\\r')
			.replace(/\n/g, '\\n')
			.replace(/\t/g, '\\t');
		let result = '';
		for (const c of out) {
			const code = c.charCodeAt(0);
			if ((code >= 0x00 && code <= 0x1f) || code === 0x7f) {
				result += `\\x${code.toString(16).padStart(2, '0')}`;
			} else {
				result += c;
			}
		}
		return result;
	}

	// In-app confirmation rather than `window.confirm`. Native modals
	// in webkit2gtk steal focus from the terminal pty and look out of
	// place against the themed UI.
	let showDestroyConfirm = $state(false);

	async function handleDestroyConfirmed() {
		if (!terminal) return;
		showDestroyConfirm = false;
		const ok = await terminalsStore.destroy(terminal.id);
		if (ok) {
			toast.success('Terminal destroyed');
			goto('#/terminals');
		}
	}

	function handleDestroy() {
		if (!terminal) return;
		showDestroyConfirm = true;
	}

	function handleDisconnect(clean: boolean) {
		if (clean) toast.message('Terminal disconnected');
		else toast.error('Terminal connection lost');
	}
</script>

<svelte:head>
	<title>{terminal?.name ?? 'Terminal'} · heierchat</title>
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

		{#if mode === 'review'}
			<aside
				class="flex w-80 flex-shrink-0 flex-col gap-2 overflow-hidden rounded-xl border bg-card p-3 shadow-sm"
			>
				<header class="flex items-center justify-between">
					<h2 class="flex items-center gap-1.5 text-xs font-semibold">
						<ClipboardCheck class="h-3.5 w-3.5 text-primary" />
						Review queue
						<span class="ml-1 rounded-full bg-primary/15 px-1.5 py-0.5 text-[10px] text-primary">
							{pendingProposals.length}
						</span>
					</h2>
				</header>
				<p class="text-[10px] text-muted-foreground">
					Keystrokes the model has proposed. Approve lands them in the PTY; reject drops them.
				</p>
				{#if pendingProposals.length === 0}
					<div class="flex flex-1 items-center justify-center text-[11px] text-muted-foreground/70">
						No pending proposals.
					</div>
				{:else}
					<ul class="flex min-h-0 flex-1 flex-col gap-2 overflow-auto">
						{#each pendingProposals as p (p.id)}
							<li class="flex flex-col gap-1.5 rounded-md border border-border/60 bg-muted/20 p-2">
								<div class="flex items-start justify-between gap-2">
									<span class="font-mono text-[10px] text-muted-foreground">
										{new Date(p.createdAt).toLocaleTimeString()}
										{#if p.autoEnter}
											<span class="ml-1 rounded bg-muted px-1 text-[9px]">+ enter</span>
										{/if}
									</span>
									<div class="flex gap-1">
										<Button
											variant="ghost"
											size="sm"
											class="h-6 w-6 p-0 text-emerald-600 hover:text-emerald-600 dark:text-emerald-400"
											title="Approve"
											onclick={() => approveProposal(p)}
										>
											<Check class="h-3.5 w-3.5" />
										</Button>
										<Button
											variant="ghost"
											size="sm"
											class="h-6 w-6 p-0 text-destructive hover:text-destructive"
											title="Reject"
											onclick={() => rejectProposal(p)}
										>
											<X class="h-3.5 w-3.5" />
										</Button>
									</div>
								</div>
								<pre
									class="max-h-32 overflow-auto rounded bg-muted/50 p-1.5 font-mono text-[11px] leading-snug break-all whitespace-pre-wrap">{preview(
										p.text
									)}</pre>
							</li>
						{/each}
					</ul>
				{/if}
			</aside>
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

<AlertDialog.Root bind:open={showDestroyConfirm}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>Destroy terminal?</AlertDialog.Title>
			<AlertDialog.Description>
				This will stop and remove
				<span class="font-mono">"{terminal?.name ?? ''}"</span> and wipe its workspace. Anything you
				didn't push or copy out will be gone.
			</AlertDialog.Description>
		</AlertDialog.Header>
		<AlertDialog.Footer>
			<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>
			<AlertDialog.Action
				class="text-destructive-foreground bg-destructive hover:bg-destructive/90"
				onclick={handleDestroyConfirmed}
			>
				Destroy
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
