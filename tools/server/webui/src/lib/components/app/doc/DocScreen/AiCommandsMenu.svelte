<script lang="ts">
	import { Loader2, Square, Wand2 } from '@lucide/svelte';
	import { browser } from '$app/environment';
	import { Button } from '$lib/components/ui/button';
	import {
		DropdownMenu,
		DropdownMenuContent,
		DropdownMenuItem,
		DropdownMenuLabel,
		DropdownMenuSeparator,
		DropdownMenuTrigger
	} from '$lib/components/ui/dropdown-menu';
	import { aiCommandsStore } from '$lib/stores/ai-commands.svelte';

	interface Props {
		onRun: (commandId: string) => void;
		open?: boolean;
	}

	let { onRun, open = $bindable(false) }: Props = $props();

	let commands = $derived(aiCommandsStore.list());
	let runningId = $derived(aiCommandsStore.runningId);
	let runningCommand = $derived(runningId ? commands.find((c) => c.id === runningId) : undefined);

	// The DocScreen shortcut accepts both Ctrl and Cmd; show the conventional
	// label for the user's platform instead of always saying "Ctrl".
	const isMac = browser && /Mac|iPhone|iPad|iPod/.test(navigator.platform);
	const shortcutLabel = isMac ? '⌘⇧K' : 'Ctrl+Shift+K';
</script>

{#if runningId}
	<Button
		variant="ghost"
		size="sm"
		class="gap-1.5 rounded-full backdrop-blur-lg"
		title={runningCommand
			? `Stop "${runningCommand.name}" (Esc)`
			: 'Stop the running AI command (Esc)'}
		onclick={() => aiCommandsStore.stop()}
	>
		<Loader2 class="h-4 w-4 animate-spin" />
		<Square class="h-3 w-3" />
		<span class="hidden max-w-[10rem] truncate md:inline">
			{runningCommand ? runningCommand.name : 'Stop'}
		</span>
	</Button>
{:else}
	<DropdownMenu bind:open>
		<DropdownMenuTrigger>
			{#snippet child({ props })}
				<Button
					{...props}
					variant="ghost"
					size="sm"
					class="gap-1.5 rounded-full backdrop-blur-lg"
					title="Run an AI command on this document ({shortcutLabel})"
				>
					<Wand2 class="h-4 w-4" />
					<span class="hidden md:inline">Commands</span>
				</Button>
			{/snippet}
		</DropdownMenuTrigger>

		<DropdownMenuContent align="end" class="w-56">
			<DropdownMenuLabel>AI commands</DropdownMenuLabel>
			<DropdownMenuSeparator />
			{#if commands.length === 0}
				<div class="px-2 py-1.5 text-xs text-muted-foreground">
					No commands. Edit <code>aiCommands</code> in settings (JSON).
				</div>
			{:else}
				{#each commands as command (command.id)}
					<DropdownMenuItem
						onclick={() => {
							open = false;
							onRun(command.id);
						}}
					>
						<span class="flex-1">{command.name}</span>
						{#if command.mode === 'replace'}
							<span class="text-[10px] text-muted-foreground" title="Acts on current selection">
								sel
							</span>
						{/if}
					</DropdownMenuItem>
				{/each}
			{/if}
		</DropdownMenuContent>
	</DropdownMenu>
{/if}
