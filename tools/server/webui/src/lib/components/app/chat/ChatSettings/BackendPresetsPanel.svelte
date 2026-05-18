<script lang="ts">
	/**
	 * Editor for `config.backendPresets` — a JSON-stringified array of
	 * `{name, url}` entries the tray's "Switch backend" submenu reads.
	 *
	 * Round-trip: parses the JSON string from localConfig, renders one
	 * row per preset with name + url inputs, calls onConfigChange with
	 * the re-serialised JSON on any edit / add / remove.
	 *
	 * Malformed JSON (rare — only happens if the user hand-edited the
	 * value to invalid syntax) is surfaced as a small warning and we
	 * start from an empty list rather than block the panel.
	 */

	import { Plus, Trash2, AlertTriangle, Server } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { SETTINGS_KEYS } from '$lib/constants';
	import type { SettingsConfigType } from '$lib/types';

	interface Props {
		localConfig: SettingsConfigType;
		onConfigChange: (key: string, value: string | boolean) => void;
	}

	let { localConfig, onConfigChange }: Props = $props();

	interface PresetRow {
		name: string;
		url: string;
	}

	let parseError = $state<string | null>(null);

	let presets = $state<PresetRow[]>(parseInitial());

	function parseInitial(): PresetRow[] {
		const raw = String(localConfig[SETTINGS_KEYS.BACKEND_PRESETS] ?? '').trim();
		if (!raw) return [];
		try {
			const parsed = JSON.parse(raw);
			if (!Array.isArray(parsed)) {
				parseError = 'Stored value is not an array — starting fresh.';
				return [];
			}
			return parsed
				.filter((p) => p && typeof p.url === 'string')
				.map((p) => ({
					name: typeof p.name === 'string' ? p.name : '',
					url: typeof p.url === 'string' ? p.url : ''
				}));
		} catch (err) {
			parseError = `Couldn't parse stored JSON (${err instanceof Error ? err.message : String(err)}). Starting fresh.`;
			return [];
		}
	}

	function persist() {
		// Drop rows where both fields are empty so trailing blank rows
		// don't pollute the tray submenu, but keep partial rows so the
		// user can type incrementally without rows disappearing.
		const filtered = presets.filter((p) => p.name.trim() || p.url.trim());
		const json = filtered.length === 0 ? '' : JSON.stringify(filtered);
		onConfigChange(SETTINGS_KEYS.BACKEND_PRESETS, json);
	}

	function updateRow(index: number, key: 'name' | 'url', value: string) {
		presets[index] = { ...presets[index], [key]: value };
		persist();
	}

	function addRow() {
		presets.push({ name: '', url: '' });
		// Don't persist the empty row — `persist` filters it out, which
		// means saving and re-opening the panel would drop the new
		// blank row mid-edit. The next keystroke will persist.
	}

	function removeRow(index: number) {
		presets.splice(index, 1);
		persist();
	}
</script>

<div class="space-y-3 rounded-md border border-border bg-muted/30 p-3">
	<div class="flex items-center gap-2">
		<Server class="h-4 w-4 text-muted-foreground" />
		<Label class="font-medium">Backend presets</Label>
	</div>
	<p class="text-xs text-muted-foreground">
		Named backends for the tray's "Switch backend" submenu. The currently active
		<code>Backend Base URL</code> is marked with a check. Leave empty to fall back
		to a single auto-derived "Default" entry.
	</p>

	{#if parseError}
		<div class="flex items-start gap-2 rounded border border-amber-500/50 bg-amber-500/10 p-2 text-xs">
			<AlertTriangle class="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
			<span>{parseError}</span>
		</div>
	{/if}

	{#if presets.length === 0}
		<p class="rounded border border-dashed border-border/60 p-3 text-center text-xs text-muted-foreground">
			No presets yet. Add one to make the tray switcher useful.
		</p>
	{:else}
		<ul class="space-y-2">
			{#each presets as preset, idx (idx)}
				<li class="flex items-start gap-2">
					<div class="flex flex-1 flex-col gap-1">
						<Input
							class="text-sm"
							placeholder="Name (e.g. Titan)"
							value={preset.name}
							oninput={(e) => updateRow(idx, 'name', e.currentTarget.value)}
						/>
						<Input
							class="font-mono text-xs"
							placeholder="https://host:port"
							value={preset.url}
							oninput={(e) => updateRow(idx, 'url', e.currentTarget.value)}
						/>
					</div>
					<Button
						variant="ghost"
						size="icon-sm"
						aria-label="Remove preset"
						onclick={() => removeRow(idx)}
					>
						<Trash2 class="h-3.5 w-3.5" />
					</Button>
				</li>
			{/each}
		</ul>
	{/if}

	<Button variant="outline" size="sm" onclick={addRow} class="w-full">
		<Plus class="mr-1 h-3.5 w-3.5" />
		Add preset
	</Button>
</div>
