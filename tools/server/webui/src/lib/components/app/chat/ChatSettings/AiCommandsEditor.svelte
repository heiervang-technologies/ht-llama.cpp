<script lang="ts">
	import { Plus, Pencil, Copy, Trash2, RotateCcw, Save, X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import {
		DEFAULT_AI_COMMANDS,
		parseAiCommands,
		type AiCommand,
		type AiCommandMode
	} from '$lib/constants/ai-commands';

	interface Props {
		value: string;
		onChange: (next: string) => void;
	}

	let { value, onChange }: Props = $props();

	type Draft = {
		id: string;
		name: string;
		template: string;
		mode: AiCommandMode;
		requiresSelection: boolean;
		isNew: boolean;
	};

	function readCommands(raw: string): AiCommand[] {
		if (!raw || raw.trim().length === 0) return [...DEFAULT_AI_COMMANDS];
		return parseAiCommands(raw);
	}

	// svelte-ignore state_referenced_locally
	let commands = $state<AiCommand[]>(readCommands(value));
	let editing = $state<Draft | null>(null);
	// svelte-ignore state_referenced_locally
	let lastSyncedValue = value;

	// If the underlying config value is reset externally (e.g. Reset button in settings footer),
	// re-read the list so the UI reflects it.
	$effect(() => {
		if (value !== lastSyncedValue) {
			lastSyncedValue = value;
			commands = readCommands(value);
			editing = null;
		}
	});

	function emit(next: AiCommand[]) {
		commands = next;
		const serialized = JSON.stringify(next);
		lastSyncedValue = serialized;
		onChange(serialized);
	}

	function startNew() {
		editing = {
			id: `custom-${Date.now()}`,
			name: '',
			template: '',
			mode: 'append',
			requiresSelection: false,
			isNew: true
		};
	}

	function startEdit(cmd: AiCommand) {
		editing = {
			id: cmd.id,
			name: cmd.name,
			template: cmd.template,
			mode: cmd.mode,
			requiresSelection: cmd.requiresSelection ?? false,
			isNew: false
		};
	}

	function startDuplicate(cmd: AiCommand) {
		editing = {
			id: `custom-${Date.now()}`,
			name: `${cmd.name} (copy)`,
			template: cmd.template,
			mode: cmd.mode,
			requiresSelection: cmd.requiresSelection ?? false,
			isNew: true
		};
	}

	function cancelEdit() {
		editing = null;
	}

	function saveEdit() {
		if (!editing) return;
		const name = editing.name.trim();
		const template = editing.template.trim();
		if (!name || !template) return;
		const next: AiCommand = {
			id: editing.id,
			name,
			template,
			mode: editing.mode,
			...(editing.mode === 'replace' && editing.requiresSelection
				? { requiresSelection: true }
				: {})
		};
		const existingIdx = commands.findIndex((c) => c.id === next.id);
		const updated = [...commands];
		if (existingIdx >= 0) updated[existingIdx] = next;
		else updated.push(next);
		emit(updated);
		editing = null;
	}

	function remove(id: string) {
		if (editing?.id === id) editing = null;
		emit(commands.filter((c) => c.id !== id));
	}

	function restoreDefaults() {
		editing = null;
		emit([...DEFAULT_AI_COMMANDS]);
	}

	function move(id: string, dir: -1 | 1) {
		const idx = commands.findIndex((c) => c.id === id);
		if (idx < 0) return;
		const targetIdx = idx + dir;
		if (targetIdx < 0 || targetIdx >= commands.length) return;
		const updated = [...commands];
		[updated[idx], updated[targetIdx]] = [updated[targetIdx], updated[idx]];
		emit(updated);
	}
</script>

<div class="space-y-4">
	<div>
		<h3 class="text-base font-semibold">AI Commands</h3>
		<p class="mt-1 text-sm text-muted-foreground">
			Commands appear in the doc editor's Commands menu. Use <code
				class="rounded bg-muted px-1 py-0.5 text-xs">{'{{document}}'}</code
			>
			and
			<code class="rounded bg-muted px-1 py-0.5 text-xs">{'{{selection}}'}</code> placeholders in the
			template.
		</p>
	</div>

	<div class="space-y-2">
		{#each commands as cmd, i (cmd.id)}
			<div
				class="flex items-center gap-2 rounded-md border border-border/50 bg-background/40 px-3 py-2"
			>
				<div class="min-w-0 flex-1">
					<div class="flex items-center gap-2">
						<span class="truncate text-sm font-medium">{cmd.name}</span>
						<span
							class="rounded-full border border-border/50 px-1.5 py-0.5 text-[10px] text-muted-foreground"
						>
							{cmd.mode}
						</span>
						{#if cmd.requiresSelection}
							<span
								class="rounded-full border border-border/50 px-1.5 py-0.5 text-[10px] text-muted-foreground"
							>
								needs selection
							</span>
						{/if}
					</div>
					<div class="truncate text-xs text-muted-foreground">
						{cmd.template.replace(/\s+/g, ' ').slice(0, 120)}
					</div>
				</div>

				<div class="flex shrink-0 items-center gap-1">
					<Button
						variant="ghost"
						size="sm"
						class="h-7 px-2"
						title="Move up"
						disabled={i === 0}
						onclick={() => move(cmd.id, -1)}
					>
						↑
					</Button>
					<Button
						variant="ghost"
						size="sm"
						class="h-7 px-2"
						title="Move down"
						disabled={i === commands.length - 1}
						onclick={() => move(cmd.id, 1)}
					>
						↓
					</Button>
					<Button
						variant="ghost"
						size="sm"
						class="h-7 px-2"
						title="Edit"
						onclick={() => startEdit(cmd)}
					>
						<Pencil class="h-3.5 w-3.5" />
					</Button>
					<Button
						variant="ghost"
						size="sm"
						class="h-7 px-2"
						title="Duplicate"
						disabled={editing !== null}
						onclick={() => startDuplicate(cmd)}
					>
						<Copy class="h-3.5 w-3.5" />
					</Button>
					<Button
						variant="ghost"
						size="sm"
						class="h-7 px-2 text-destructive hover:text-destructive"
						title="Delete"
						onclick={() => remove(cmd.id)}
					>
						<Trash2 class="h-3.5 w-3.5" />
					</Button>
				</div>
			</div>
		{/each}

		{#if commands.length === 0}
			<div class="rounded-md border border-dashed border-border/50 p-4 text-center">
				<p class="text-sm text-muted-foreground">No commands. Add one or restore the defaults.</p>
			</div>
		{/if}
	</div>

	<div class="flex items-center gap-2">
		<Button variant="outline" size="sm" onclick={startNew} disabled={editing !== null}>
			<Plus class="mr-1 h-3.5 w-3.5" />
			New command
		</Button>
		<Button variant="ghost" size="sm" onclick={restoreDefaults} title="Restore default commands">
			<RotateCcw class="mr-1 h-3.5 w-3.5" />
			Restore defaults
		</Button>
	</div>

	{#if editing}
		<div class="space-y-3 rounded-md border border-border/60 bg-muted/30 p-4">
			<div class="flex items-center justify-between">
				<h4 class="text-sm font-semibold">
					{editing.isNew ? 'New command' : 'Edit command'}
				</h4>
				<Button variant="ghost" size="sm" class="h-7 px-2" onclick={cancelEdit} title="Cancel">
					<X class="h-3.5 w-3.5" />
				</Button>
			</div>

			<div class="space-y-1">
				<label for="ai-cmd-name" class="text-xs font-medium text-muted-foreground">Name</label>
				<input
					id="ai-cmd-name"
					type="text"
					class="w-full rounded-md border border-border/60 bg-background px-2 py-1.5 text-sm outline-none focus:border-primary"
					placeholder="e.g. Rewrite more formal"
					bind:value={editing.name}
				/>
			</div>

			<div class="space-y-1">
				<label for="ai-cmd-template" class="text-xs font-medium text-muted-foreground">
					Prompt template
				</label>
				<textarea
					id="ai-cmd-template"
					rows="6"
					class="w-full rounded-md border border-border/60 bg-background px-2 py-1.5 font-mono text-xs outline-none focus:border-primary"
					placeholder={'Rewrite the following passage…\n\n{{selection}}'}
					bind:value={editing.template}
				></textarea>
				<p class="text-[11px] text-muted-foreground">
					{'Use {{document}} for the full document and {{selection}} for the highlighted text.'}
				</p>
			</div>

			<div class="flex flex-wrap items-center gap-3">
				<div class="space-y-1">
					<label for="ai-cmd-mode" class="text-xs font-medium text-muted-foreground">Mode</label>
					<select
						id="ai-cmd-mode"
						class="rounded-md border border-border/60 bg-background px-2 py-1 text-sm outline-none focus:border-primary"
						bind:value={editing.mode}
					>
						<option value="append">Append to document</option>
						<option value="replace">Replace selection</option>
					</select>
				</div>

				{#if editing.mode === 'replace'}
					<label class="mt-4 flex cursor-pointer items-center gap-2 text-sm">
						<input type="checkbox" bind:checked={editing.requiresSelection} />
						<span>Require active selection</span>
					</label>
				{/if}
			</div>

			<div class="flex items-center gap-2 pt-1">
				<Button
					size="sm"
					onclick={saveEdit}
					disabled={!editing.name.trim() || !editing.template.trim()}
				>
					<Save class="mr-1 h-3.5 w-3.5" />
					{editing.isNew ? 'Add command' : 'Save changes'}
				</Button>
				<Button variant="ghost" size="sm" onclick={cancelEdit}>Cancel</Button>
			</div>
		</div>
	{/if}
</div>
