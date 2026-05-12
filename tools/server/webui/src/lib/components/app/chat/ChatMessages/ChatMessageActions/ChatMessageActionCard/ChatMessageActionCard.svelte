<script lang="ts">
	import type { Snippet, Component } from 'svelte';

	interface Props {
		icon?: Component<{ class?: string }>;
		message?: Snippet;
		actions?: Snippet;
		children?: Snippet;
		actionsPosition?: 'left' | 'right' | 'top' | 'bottom';
		justify?: 'start' | 'end' | 'center' | 'between';
		role?: string;
		isLastAssistantMessage?: boolean;
		content?: string;
		uploadedFile?: unknown;
		uploadedFiles?: unknown;
		item?: unknown;
		isLoading?: boolean;
		showAddButton?: boolean;
		disabled?: boolean;
		deletionInfo?: unknown;
		siblingInfo?: unknown;
		onConfirmDelete?: (...args: any[]) => any;
		onCopy?: (...args: any[]) => any;
		onDelete?: (...args: any[]) => any;
		onEdit?: (...args: any[]) => any;
		onForkConversation?: (...args: any[]) => any;
		onNavigateToSibling?: (...args: any[]) => any;
		onShowDeleteDialogChange?: (...args: any[]) => any;
		showDeleteDialog?: boolean;
		onRegenerate?: (...args: any[]) => any;
		onContinue?: (...args: any[]) => any;
		showRawOutputSwitch?: boolean;
		rawOutputEnabled?: boolean;
		onRawOutputToggle?: (...args: any[]) => any;
		onSpeak?: (...args: any[]) => any;
		isSpeaking?: boolean;
		isLoadingSpeech?: boolean;
		highlightTurns?: boolean;
	}

	let { icon: IconComponent, message, actions, children }: Props = $props();
</script>

<div class="my-2 rounded-lg border border-border bg-card p-3">
	<div class="mb-3 flex items-center gap-2 text-sm">
		{#if IconComponent}<IconComponent class="h-4 w-4 shrink-0 text-muted-foreground" />{/if}
		<span>
			{#if message}{@render message()}{/if}
		</span>
	</div>
	<div class="flex flex-wrap items-center gap-2">
		{#if actions}{@render actions()}{/if}
	</div>
</div>
