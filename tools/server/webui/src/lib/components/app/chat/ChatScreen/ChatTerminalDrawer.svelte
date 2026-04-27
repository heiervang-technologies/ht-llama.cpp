<script lang="ts">
	import { TerminalSquare, X, ExternalLink, ChevronDown, ChevronUp } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { TerminalView } from '$lib/components/app/terminals';
	import { chatTerminalAttachment } from '$lib/stores/chat-terminal-attachment.svelte';
	import { terminalsStore } from '$lib/stores/terminals.svelte';
	import { goto } from '$app/navigation';

	// Drawer is dismissable AND collapsible. Dismiss = "the model
	// shouldn't pop this back up automatically" until the model acts
	// on a different terminal. Collapse = "I just want it out of the
	// way for a sec but keep it attached".
	let collapsed = $state(false);

	let id = $derived(chatTerminalAttachment.terminalId);
	let visible = $derived(chatTerminalAttachment.visible);
	let terminal = $derived(id ? terminalsStore.terminals.find((t) => t.id === id) : null);
	let displayName = $derived(terminal?.name ?? id?.slice(0, 8) ?? 'terminal');

	function close() {
		chatTerminalAttachment.dismiss();
		collapsed = false;
	}

	function openFull() {
		if (!id) return;
		goto(`#/terminals/${id}`);
	}
</script>

{#if visible && id}
	<div
		class="mx-auto mt-2 w-full max-w-[48rem] overflow-hidden rounded-lg border border-border/40 bg-muted/30"
	>
		<header class="flex items-center gap-2 border-b border-border/30 bg-muted/40 px-3 py-1.5">
			<TerminalSquare class="h-3.5 w-3.5 flex-shrink-0 text-primary" aria-hidden="true" />
			<div class="min-w-0 flex-1 truncate text-xs">
				<span class="font-medium">{displayName}</span>
				{#if terminal?.image}
					<span class="text-muted-foreground"> · {terminal.image}</span>
				{/if}
			</div>
			<Button
				variant="ghost"
				size="sm"
				class="h-6 w-6 p-0"
				onclick={() => (collapsed = !collapsed)}
				title={collapsed ? 'Expand' : 'Collapse'}
			>
				{#if collapsed}
					<ChevronUp class="h-3.5 w-3.5" />
				{:else}
					<ChevronDown class="h-3.5 w-3.5" />
				{/if}
			</Button>
			<Button
				variant="ghost"
				size="sm"
				class="h-6 w-6 p-0"
				onclick={openFull}
				title="Open full terminal view"
			>
				<ExternalLink class="h-3.5 w-3.5" />
			</Button>
			<Button variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={close} title="Detach drawer">
				<X class="h-3.5 w-3.5" />
			</Button>
		</header>

		{#if !collapsed}
			<!-- Fixed-height pane so the chat composer stays anchored.
			     Tall enough for ~16 lines + a couple of chrome rows; if
			     the user wants more they can hit "Open full". -->
			{#key id}
				<div class="h-64 w-full">
					<TerminalView terminalId={id} />
				</div>
			{/key}
		{/if}
	</div>
{/if}
