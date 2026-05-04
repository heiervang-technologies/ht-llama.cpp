<script lang="ts">
	import {
		TerminalSquare,
		X,
		ExternalLink,
		ChevronDown,
		ChevronUp,
		PictureInPicture2
	} from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { TerminalView } from '$lib/components/app/terminals';
	import { chatTerminalAttachment } from '$lib/stores/chat-terminal-attachment.svelte';
	import { terminalsStore } from '$lib/stores/terminals.svelte';
	import { openInNewWindow } from '$lib/utils/tauri-window';
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

	async function popOut() {
		if (!id) return;
		// Stable label keyed on terminal id so a second pop-out for
		// the same terminal focuses the existing window instead of
		// spawning a duplicate.
		await openInNewWindow(`#/terminals/${id}`, {
			title: `${displayName} · ht-llama.cpp`,
			label: `term-${id.replace(/[^a-zA-Z0-9_-]/g, '_')}`,
			width: 960,
			height: 720
		});
	}
</script>

<!-- Once an `id` is set we keep the drawer markup mounted across dismiss / re-attach
     cycles, hiding it via `display: none` instead of unmounting. The TerminalView
     therefore retains its WebSocket and xterm state — closing and re-opening the
     drawer (or hammering the header terminal button) no longer triggers a fresh
     handshake + backlog repaint every time. We still remount on `id` change via
     `{#key id}` because that's a different sandbox and the WS has to point
     somewhere else. -->
{#if id}
	<div
		class="mx-auto mt-2 w-full max-w-[48rem] overflow-hidden rounded-lg border border-border/40 bg-muted/30"
		class:hidden={!visible}
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
				onclick={popOut}
				title="Pop out into its own window"
			>
				<PictureInPicture2 class="h-3.5 w-3.5" />
			</Button>
			<Button
				variant="ghost"
				size="sm"
				class="h-6 w-6 p-0"
				onclick={openFull}
				title="Open full terminal view in this window"
			>
				<ExternalLink class="h-3.5 w-3.5" />
			</Button>
			<Button variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={close} title="Detach drawer">
				<X class="h-3.5 w-3.5" />
			</Button>
		</header>

		<!-- Collapse hides the pane visually but keeps the WS alive,
		     same rationale as the dismiss case above. Fixed-height
		     pane so the chat composer stays anchored. -->
		{#key id}
			<div class="h-64 w-full" class:hidden={collapsed}>
				<TerminalView terminalId={id} />
			</div>
		{/key}
	</div>
{/if}
