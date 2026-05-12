<script lang="ts">
	import { Trash2, Pencil, Copy, MoreHorizontal, FileText } from '@lucide/svelte';
	import { DropdownMenuActions } from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { Checkbox } from '$lib/components/ui/checkbox';

	interface Props {
		doc: DatabaseDoc;
		isActive?: boolean;
		handleMobileSidebarItemClick?: () => void;
		onSelect?: (id: string) => void;
		onEdit?: (id: string) => void;
		onDuplicate?: (id: string) => void;
		onDelete?: (id: string) => void;
		selectionMode?: boolean;
		isSelected?: boolean;
		onToggleSelect?: (id: string, event: MouseEvent | KeyboardEvent) => void;
	}

	let {
		doc,
		isActive = false,
		handleMobileSidebarItemClick,
		onSelect,
		onEdit,
		onDuplicate,
		onDelete,
		onToggleSelect,
		selectionMode = false,
		isSelected = false
	}: Props = $props();

	let renderActionsDropdown = $state(false);
	let dropdownOpen = $state(false);

	function handleSelect(event: MouseEvent) {
		if (selectionMode) {
			onToggleSelect?.(doc.id, event);
			return;
		}
		onSelect?.(doc.id);
	}

	function handleEdit(event: Event) {
		event.stopPropagation();
		onEdit?.(doc.id);
	}

	function handleDuplicate(event: Event) {
		event.stopPropagation();
		onDuplicate?.(doc.id);
	}

	function handleDelete(event: Event) {
		event.stopPropagation();
		onDelete?.(doc.id);
	}

	function handleMouseLeave() {
		if (!dropdownOpen) renderActionsDropdown = false;
	}

	function handleMouseOver() {
		renderActionsDropdown = true;
	}

	$effect(() => {
		if (!dropdownOpen) renderActionsDropdown = false;
	});

	let displayName = $derived(doc.name?.trim() || 'Untitled');
</script>

<!-- svelte-ignore a11y_mouse_events_have_key_events -->
<button
	class="group flex min-h-9 w-full cursor-pointer items-center justify-between space-x-3 rounded-lg px-3 py-1.5 text-left transition-colors hover:bg-foreground/10 {isActive
		? 'bg-foreground/5 text-accent-foreground'
		: ''} {selectionMode && isSelected ? 'bg-primary/15 hover:bg-primary/20' : ''}"
	data-doc-id={doc.id}
	onclick={handleSelect}
	onmouseover={handleMouseOver}
	onmouseleave={handleMouseLeave}
	title={displayName}
>
	<div class="flex min-w-0 flex-1 items-center gap-2">
		{#if selectionMode}
			<Checkbox
				checked={isSelected}
				aria-label={isSelected ? 'Deselect document' : 'Select document'}
				class="pointer-events-none shrink-0"
			/>
		{/if}

		<Tooltip.Root>
			<Tooltip.Trigger>
				<FileText class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
			</Tooltip.Trigger>
			<Tooltip.Content>
				<p>Document</p>
			</Tooltip.Content>
		</Tooltip.Root>

		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<span class="truncate text-sm font-medium" onclick={handleMobileSidebarItemClick}>
			{displayName}
		</span>
	</div>

	{#if renderActionsDropdown && !selectionMode}
		<div class="actions flex items-center">
			<DropdownMenuActions
				triggerIcon={MoreHorizontal}
				triggerTooltip="More actions"
				bind:open={dropdownOpen}
				actions={[
					{
						icon: Pencil,
						label: 'Rename',
						onclick: handleEdit
					},
					{
						icon: Copy,
						label: 'Duplicate',
						onclick: handleDuplicate
					},
					{
						icon: Trash2,
						label: 'Delete',
						onclick: handleDelete,
						variant: 'destructive',
						separator: true
					}
				]}
			/>
		</div>
	{/if}
</button>

<style>
	button {
		:global([data-slot='dropdown-menu-trigger']:not([data-state='open'])) {
			opacity: 0;
		}
		&:is(:hover) :global([data-slot='dropdown-menu-trigger']) {
			opacity: 1;
		}
		@media (max-width: 768px) {
			:global([data-slot='dropdown-menu-trigger']) {
				opacity: 1 !important;
			}
		}
	}
</style>
