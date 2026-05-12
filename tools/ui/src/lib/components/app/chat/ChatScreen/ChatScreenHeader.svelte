<script lang="ts">
	import { Settings, PanelRight, Wrench, Eye, EyeOff, TerminalSquare } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import { getChatSettingsDialogContext } from '$lib/contexts';
	import { BackendPill } from '$lib/components/app/navigation';
	import { artifactsStore } from '$lib/stores/artifacts.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { DialogAvailableTools } from '$lib/components/app/dialogs';
	import { chatTerminalAttachment } from '$lib/stores/chat-terminal-attachment.svelte';
	import { terminalsStore } from '$lib/stores/terminals.svelte';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	const sidebar = useSidebar();
	const chatSettingsDialog = getChatSettingsDialogContext();

	let showToolsDialog = $state(false);
	// Badge count: number of MCP tools currently serialised into the
	// request. Zero is still worth showing so users see "no tools" is real,
	// not a UI bug.
	let toolCount = $derived(mcpStore.getToolDefinitionsForLLM().length);

	// Prior-turn transparency: when on, system messages + tool
	// calls/results render as their own cards in the log. Flips three
	// flags in unison so the user gets a single affordance instead of
	// three scattered toggles.
	let transparent = $derived(
		Boolean(config().showSystemMessage) &&
			Boolean(config().alwaysShowAgenticTurns) &&
			Boolean(config().showToolMessagesAsStandalone)
	);

	function toggleTransparency() {
		const next = !transparent;
		settingsStore.updateConfig('showSystemMessage', next);
		settingsStore.updateConfig('alwaysShowAgenticTurns', next);
		settingsStore.updateConfig('showToolMessagesAsStandalone', next);
	}

	let artifactCount = $derived(artifactsStore.entries.length);
	let hasArtifacts = $derived(artifactCount > 0);
	let artifactTooltip = $derived(
		artifactCount === 1
			? '1 artifact · click to toggle drawer'
			: `${artifactCount} artifacts · click to toggle drawer`
	);

	// Terminal button — shown only when ht-termd is reachable. Refresh
	// the live list once on mount so we know whether to spawn or attach
	// when the button is pressed; subsequent attaches use the cached
	// list to avoid a network round-trip per click.
	let termdAvailable = $derived(terminalsStore.available);
	let liveTerminals = $derived(terminalsStore.terminals);
	let attachedId = $derived(chatTerminalAttachment.terminalId);
	let drawerVisible = $derived(chatTerminalAttachment.visible);
	let needsSetup = $derived(terminalsStore.needsSetup);
	let creatingTerminal = $derived(terminalsStore.creating);
	let terminalCount = $derived(liveTerminals.length);

	let terminalTooltip = $derived(
		!termdAvailable
			? 'Terminal sandbox not configured — set Settings → Terminals → Base URL'
			: needsSetup
				? 'Sandbox prerequisites missing — run `unleash sandbox setup`'
				: drawerVisible
					? 'Hide terminal drawer'
					: terminalCount === 0
						? 'Spawn a sandbox terminal'
						: attachedId
							? 'Show terminal drawer'
							: `Attach most recent terminal (${terminalCount} available)`
	);

	onMount(() => {
		// Best-effort discovery; failures are surfaced via the badge
		// state, not a toast. Refresh is cheap enough to run on every
		// chat-screen mount so a sandbox spawned in another tab shows
		// up the moment the user comes back.
		if (terminalsStore.available) terminalsStore.refresh();
	});

	async function handleTerminalClick() {
		if (!termdAvailable) {
			toast.error('Terminal sandbox not configured.');
			chatSettingsDialog.open();
			return;
		}
		if (drawerVisible) {
			chatTerminalAttachment.dismiss();
			return;
		}
		if (attachedId) {
			// Drawer was dismissed but we still know which terminal —
			// re-attach the same id rather than picking a "fresh" one.
			chatTerminalAttachment.attach(attachedId);
			return;
		}
		// No attached terminal. Prefer reusing an existing sandbox to
		// keep the user's container count manageable; only spawn fresh
		// when the list is empty.
		if (liveTerminals.length > 0) {
			chatTerminalAttachment.attach(liveTerminals[0].id);
			return;
		}
		if (needsSetup) {
			toast.error('Sandbox prerequisites missing — run `unleash sandbox setup` on the host first.');
			return;
		}
		const t = await terminalsStore.create({});
		if (t) {
			chatTerminalAttachment.attach(t.id);
			toast.success(`Spawned sandbox "${t.name}"`);
		}
	}
</script>

<header
	class="pointer-events-none fixed top-0 right-0 left-0 z-50 flex items-center justify-end p-2 duration-200 ease-linear md:p-4 {sidebar.open
		? 'md:left-[var(--sidebar-width)]'
		: ''}"
>
	<div class="pointer-events-auto flex items-center space-x-2">
		<BackendPill />

		{#if hasArtifacts}
			<Button
				variant="ghost"
				size="icon-lg"
				onclick={() => artifactsStore.toggle()}
				class="relative rounded-full backdrop-blur-lg"
				title={artifactTooltip}
				aria-label={artifactTooltip}
			>
				<PanelRight class="h-4 w-4" />
				{#if artifactCount > 1}
					<span
						class="absolute -top-1 -right-1 flex h-4 min-w-[1rem] items-center justify-center rounded-full bg-primary px-1 text-[10px] font-semibold text-primary-foreground"
						aria-hidden="true"
					>
						{artifactCount > 9 ? '9+' : artifactCount}
					</span>
				{/if}
			</Button>
		{/if}

		<Button
			variant="ghost"
			size="icon-lg"
			onclick={toggleTransparency}
			class="rounded-full backdrop-blur-lg {transparent ? 'text-primary' : ''}"
			title={transparent
				? 'Hide system + tool messages (currently visible in the log)'
				: 'Show system + tool messages inline in the chat log'}
			aria-label="Toggle prior-turn transparency"
			aria-pressed={transparent}
		>
			{#if transparent}
				<Eye class="h-4 w-4" />
			{:else}
				<EyeOff class="h-4 w-4" />
			{/if}
		</Button>

		<Button
			variant="ghost"
			size="icon-lg"
			onclick={() => (showToolsDialog = true)}
			class="relative rounded-full backdrop-blur-lg"
			title="Available tools — what the model sees on the next turn"
			aria-label="Inspect available tools"
		>
			<Wrench class="h-4 w-4" />
			{#if toolCount > 0}
				<span
					class="absolute -top-1 -right-1 flex h-4 min-w-[1rem] items-center justify-center rounded-full bg-primary px-1 text-[10px] font-semibold text-primary-foreground"
					aria-hidden="true"
				>
					{toolCount > 9 ? '9+' : toolCount}
				</span>
			{/if}
		</Button>

		{#if termdAvailable}
			<Button
				variant="ghost"
				size="icon-lg"
				onclick={handleTerminalClick}
				disabled={creatingTerminal}
				class="relative rounded-full backdrop-blur-lg {drawerVisible ? 'text-primary' : ''}"
				title={terminalTooltip}
				aria-label={terminalTooltip}
				aria-pressed={drawerVisible}
			>
				<TerminalSquare class="h-4 w-4" />
				{#if terminalCount > 0}
					<span
						class="absolute -top-1 -right-1 flex h-4 min-w-[1rem] items-center justify-center rounded-full bg-primary px-1 text-[10px] font-semibold text-primary-foreground"
						aria-hidden="true"
					>
						{terminalCount > 9 ? '9+' : terminalCount}
					</span>
				{/if}
			</Button>
		{/if}

		<Button
			variant="ghost"
			size="icon-lg"
			onclick={() => chatSettingsDialog.open()}
			class="rounded-full backdrop-blur-lg"
		>
			<Settings class="h-4 w-4" />
		</Button>
	</div>
</header>

<DialogAvailableTools bind:open={showToolsDialog} />
