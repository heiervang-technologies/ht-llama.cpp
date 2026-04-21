<script lang="ts">
	import { Loader2, Wand2 } from '@lucide/svelte';
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
	}

	let { onRun }: Props = $props();

	let open = $state(false);
	let commands = $derived(aiCommandsStore.list());
	let running = $derived(aiCommandsStore.runningId !== null);
</script>

<DropdownMenu bind:open>
	<DropdownMenuTrigger>
		{#snippet child({ props })}
			<Button
				{...props}
				variant="ghost"
				size="sm"
				class="gap-1.5 rounded-full backdrop-blur-lg"
				title="Run an AI command on this document"
				disabled={running}
			>
				{#if running}
					<Loader2 class="h-4 w-4 animate-spin" />
				{:else}
					<Wand2 class="h-4 w-4" />
				{/if}
				<span class="hidden md:inline">{running ? 'Running…' : 'Commands'}</span>
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
					{command.name}
				</DropdownMenuItem>
			{/each}
		{/if}
		{#if running}
			<DropdownMenuSeparator />
			<DropdownMenuItem
				onclick={() => {
					open = false;
					aiCommandsStore.stop();
				}}
			>
				Stop current command
			</DropdownMenuItem>
		{/if}
	</DropdownMenuContent>
</DropdownMenu>
