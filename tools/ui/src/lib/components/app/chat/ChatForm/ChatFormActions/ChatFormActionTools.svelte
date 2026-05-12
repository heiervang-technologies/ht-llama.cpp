<script lang="ts">
	import {
		Wrench,
		Image as ImageIcon,
		Video,
		Search,
		TerminalSquare,
		Telescope,
		Settings,
		Check
	} from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { Switch } from '$lib/components/ui/switch';
	import { TOOLTIP_DELAY_DURATION, SETTINGS_KEYS, SETTINGS_SECTION_TITLES } from '$lib/constants';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { getChatSettingsDialogContext } from '$lib/contexts';
	import type { Component } from 'svelte';

	// Composer "Tools" menu — replaces the standalone image-gen pill with
	// a single dropdown that surfaces every model-callable capability the
	// user can opt into for this turn:
	//   • Image generation         (imageGenEnabled)
	//   • Video generation         (videoGenEnabled)
	//   • Web search               (webSearchEnabled)
	//   • Terminal                 (terminalToolsEnabled)
	//   • Deep research            (not implemented — disabled)
	//
	// By default the toggles are mutually exclusive (radio-style): turning
	// one on turns the others off. The "Allow multiple at once" advanced
	// switch at the bottom lifts that to checkbox semantics so the model
	// can chain capabilities (web_search → terminal → generate_image).
	//
	// "Advanced…" opens the Settings dialog at MCP, where the user can
	// configure per-tool details (URLs, API keys, max results) and any
	// extra MCP-provided tools they've wired in.

	interface Props {
		disabled?: boolean;
		class?: string;
	}

	let { disabled = false, class: className = '' }: Props = $props();

	let dropdownOpen = $state(false);
	const chatSettingsDialog = getChatSettingsDialogContext();

	type ToolKey = 'image' | 'video' | 'search' | 'terminal' | 'deep-research';

	interface ToolEntry {
		key: ToolKey;
		label: string;
		settingsKey: keyof typeof SETTINGS_KEYS | null; // null → not implemented
		icon: Component;
		hint: string;
		comingSoon?: boolean;
	}

	const tools: ToolEntry[] = [
		{
			key: 'image',
			label: 'Image generation',
			settingsKey: 'IMAGE_GEN_ENABLED',
			icon: ImageIcon,
			hint: 'generate_image / edit_image — model can produce / restyle images via the configured proxy.'
		},
		{
			key: 'video',
			label: 'Video generation',
			settingsKey: 'VIDEO_GEN_ENABLED',
			icon: Video,
			hint: 'generate_video — text / image / sound-driven clips. Async (~minutes per clip).'
		},
		{
			key: 'search',
			label: 'Web search',
			settingsKey: 'WEB_SEARCH_ENABLED',
			icon: Search,
			hint: 'web_search — SearXNG metasearch (DDG, Brave, Wikipedia, GitHub, arXiv, …) at the configured base URL.'
		},
		{
			key: 'terminal',
			label: 'Terminal',
			settingsKey: 'TERMINAL_TOOLS_ENABLED',
			icon: TerminalSquare,
			hint: 'list_terminals / send_keys — model can drive a sandboxed shell. Self-disables when no terminal is open.'
		},
		{
			key: 'deep-research',
			label: 'Deep research',
			settingsKey: null,
			icon: Telescope,
			hint: 'Long-running multi-source research loop. Not implemented yet.',
			comingSoon: true
		}
	];

	type ConfigKey =
		| 'imageGenEnabled'
		| 'videoGenEnabled'
		| 'webSearchEnabled'
		| 'terminalToolsEnabled';

	function configKeyFor(t: ToolEntry): ConfigKey | null {
		if (!t.settingsKey) return null;
		return SETTINGS_KEYS[t.settingsKey] as ConfigKey;
	}

	let cfg = $derived(config());
	let allowMultiple = $derived(Boolean(cfg.allowMultipleTools));

	function isOn(t: ToolEntry): boolean {
		const k = configKeyFor(t);
		if (!k) return false;
		return Boolean(cfg[k]);
	}

	let activeTools = $derived(tools.filter((t) => isOn(t)));
	let activeCount = $derived(activeTools.length);

	function setTool(t: ToolEntry, on: boolean) {
		const k = configKeyFor(t);
		if (!k) return;
		settingsStore.updateConfig(k, on);
	}

	function clickTool(t: ToolEntry) {
		if (t.comingSoon) return;
		const turningOn = !isOn(t);

		if (turningOn && !allowMultiple) {
			// Radio mode — turn off every other implemented tool first so
			// only the just-clicked one ends up active.
			for (const other of tools) {
				if (other.key !== t.key) setTool(other, false);
			}
		}
		setTool(t, turningOn);
	}

	function toggleAllowMultiple() {
		settingsStore.updateConfig('allowMultipleTools', !allowMultiple);
		// Switching FROM multi → radio with several tools active would
		// leave a violated invariant; collapse to whichever was first
		// in the registered list so the dropdown's radio indicator is
		// honest.
		if (allowMultiple) {
			// allowMultiple was true (about to become false) — only
			// runs the next tick after the update commits, so check
			// active count against the snapshot here.
			const active = tools.filter((t) => isOn(t));
			if (active.length > 1) {
				const keep = active[0];
				for (const t of active) {
					if (t.key !== keep.key) setTool(t, false);
				}
			}
		}
	}

	function openAdvanced() {
		dropdownOpen = false;
		chatSettingsDialog.open(SETTINGS_SECTION_TITLES.MCP);
	}

	let triggerLabel = $derived.by(() => {
		if (activeCount === 0) return 'Tools';
		if (activeCount === 1) return activeTools[0].label;
		return `Tools · ${activeCount}`;
	});

	let triggerTooltip = $derived.by(() => {
		if (activeCount === 0) return 'Tools — pick capabilities the model can use this turn.';
		const names = activeTools.map((t) => t.label).join(', ');
		return `Active: ${names}${allowMultiple ? '' : ' (radio — pick one)'}`;
	});
</script>

<DropdownMenu.Root bind:open={dropdownOpen}>
	<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
		<Tooltip.Trigger>
			{#snippet child({ props: tipProps })}
				<DropdownMenu.Trigger {disabled}>
					{#snippet child({ props: ddProps })}
						<Button
							{...tipProps}
							{...ddProps}
							type="button"
							variant={activeCount > 0 ? 'default' : 'ghost'}
							size="sm"
							{disabled}
							class="h-8 gap-1.5 rounded-full px-3 text-xs font-medium {className}"
						>
							<Wrench class="h-3.5 w-3.5" />
							<span>{triggerLabel}</span>
						</Button>
					{/snippet}
				</DropdownMenu.Trigger>
			{/snippet}
		</Tooltip.Trigger>
		<Tooltip.Content side="top">
			<p class="max-w-xs text-xs">{triggerTooltip}</p>
		</Tooltip.Content>
	</Tooltip.Root>

	<DropdownMenu.Content align="start" class="w-64">
		<DropdownMenu.Label class="text-xs tracking-wide text-muted-foreground uppercase">
			Capabilities
		</DropdownMenu.Label>

		{#each tools as t (t.key)}
			{@const on = isOn(t)}
			{@const Icon = t.icon}
			<DropdownMenu.Item
				disabled={t.comingSoon}
				onSelect={(e) => {
					// Don't close the menu — let the user toggle multiple
					// in a row when allowMultiple is on, and the click
					// itself is the toggle action.
					e.preventDefault();
					clickTool(t);
				}}
				class="flex items-center justify-between gap-2"
			>
				<div class="flex min-w-0 items-center gap-2">
					<Icon class="h-4 w-4 flex-shrink-0 {on ? 'text-primary' : 'opacity-70'}" />
					<div class="flex min-w-0 flex-col">
						<span class="truncate text-sm {t.comingSoon ? 'opacity-50' : ''}">{t.label}</span>
						{#if t.comingSoon}
							<span class="text-[10px] tracking-wide text-muted-foreground uppercase">
								Coming soon
							</span>
						{/if}
					</div>
				</div>

				{#if t.comingSoon}
					<span class="text-[10px] text-muted-foreground">—</span>
				{:else if allowMultiple}
					<!-- Checkbox-style indicator -->
					<span
						class="flex h-4 w-4 flex-shrink-0 items-center justify-center rounded border {on
							? 'border-primary bg-primary text-primary-foreground'
							: 'border-muted-foreground/40'}"
					>
						{#if on}
							<Check class="h-3 w-3" />
						{/if}
					</span>
				{:else}
					<!-- Radio-style indicator -->
					<span
						class="flex h-4 w-4 flex-shrink-0 items-center justify-center rounded-full border-2 {on
							? 'border-primary'
							: 'border-muted-foreground/40'}"
					>
						{#if on}
							<span class="h-2 w-2 rounded-full bg-primary"></span>
						{/if}
					</span>
				{/if}
			</DropdownMenu.Item>
		{/each}

		<DropdownMenu.Separator />

		<DropdownMenu.Item
			class="flex items-center justify-between gap-2"
			onSelect={(e) => {
				e.preventDefault();
				toggleAllowMultiple();
			}}
		>
			<div class="flex min-w-0 flex-col">
				<span class="text-sm">Allow multiple at once</span>
				<span class="text-[10px] text-muted-foreground"> Chain tools in one turn (advanced) </span>
			</div>
			<Switch checked={allowMultiple} class="pointer-events-none" />
		</DropdownMenu.Item>

		<DropdownMenu.Separator />

		<DropdownMenu.Item onSelect={openAdvanced} class="flex items-center gap-2">
			<Settings class="h-4 w-4 opacity-70" />
			<span class="text-sm">Advanced…</span>
		</DropdownMenu.Item>
	</DropdownMenu.Content>
</DropdownMenu.Root>
