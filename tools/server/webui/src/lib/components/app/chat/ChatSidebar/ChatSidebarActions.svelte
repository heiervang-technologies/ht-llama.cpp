<script lang="ts">
	import { FileText, CheckSquare, Search, SquarePen, Trash2, X } from '@lucide/svelte';
	import { KeyboardShortcutInfo } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { McpLogo } from '$lib/components/app';
	import { SETTINGS_SECTION_TITLES } from '$lib/constants';
	import { getChatSettingsDialogContext } from '$lib/contexts';
	import { conversations } from '$lib/stores/conversations.svelte';
	import { docsStore } from '$lib/stores/docs.svelte';

	interface Props {
		handleMobileSidebarItemClick: () => void;
		isSearchModeActive: boolean;
		searchQuery: string;
		onEnterSelectionMode?: () => void;
		onDeleteAll?: () => void;
	}

	let {
		handleMobileSidebarItemClick,
		isSearchModeActive = $bindable(),
		searchQuery = $bindable(),
		onEnterSelectionMode,
		onDeleteAll
	}: Props = $props();

	let conversationCount = $derived(conversations().length);

	let searchInput: HTMLInputElement | null = $state(null);

	const chatSettingsDialog = getChatSettingsDialogContext();

	function handleSearchModeDeactivate() {
		isSearchModeActive = false;
		searchQuery = '';
	}

	$effect(() => {
		if (isSearchModeActive) {
			searchInput?.focus();
		}
	});
</script>

<div class="my-1 space-y-1">
	{#if isSearchModeActive}
		<div class="relative">
			<Search class="absolute top-2.5 left-2 h-4 w-4 text-muted-foreground" />

			<Input
				bind:ref={searchInput}
				bind:value={searchQuery}
				onkeydown={(e) => e.key === 'Escape' && handleSearchModeDeactivate()}
				placeholder="Search conversations..."
				class="pl-8"
			/>

			<X
				class="cursor-pointertext-muted-foreground absolute top-2.5 right-2 h-4 w-4"
				onclick={handleSearchModeDeactivate}
			/>
		</div>
	{:else}
		<Button
			class="w-full justify-between backdrop-blur-none! hover:[&>kbd]:opacity-100"
			href="?new_chat=true#/"
			onclick={handleMobileSidebarItemClick}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<SquarePen class="h-4 w-4" />

				New chat
			</div>

			<KeyboardShortcutInfo keys={['shift', 'cmd', 'o']} />
		</Button>

		<Button
			class="w-full justify-start backdrop-blur-none!"
			onclick={async () => {
				handleMobileSidebarItemClick();
				await docsStore.createDoc();
			}}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<FileText class="h-4 w-4" />

				New doc
			</div>
		</Button>

		<Button
			class="w-full justify-between backdrop-blur-none! hover:[&>kbd]:opacity-100"
			onclick={() => {
				isSearchModeActive = true;
			}}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<Search class="h-4 w-4" />

				Search
			</div>

			<KeyboardShortcutInfo keys={['cmd', 'k']} />
		</Button>

		<Button
			class="w-full justify-between backdrop-blur-none! hover:[&>kbd]:opacity-100"
			onclick={() => {
				chatSettingsDialog.open(SETTINGS_SECTION_TITLES.MCP);
			}}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<McpLogo class="h-4 w-4" />

				MCP Servers
			</div>
		</Button>

		{#if conversationCount > 0 && (onEnterSelectionMode || onDeleteAll)}
			<div class="flex gap-1 pt-1">
				{#if onEnterSelectionMode}
					<Button
						class="flex-1 justify-start backdrop-blur-none!"
						onclick={() => onEnterSelectionMode?.()}
						variant="ghost"
						size="sm"
						title="Select multiple conversations to delete"
					>
						<CheckSquare class="h-4 w-4" />
						Select
					</Button>
				{/if}

				{#if onDeleteAll}
					<Button
						class="flex-1 justify-start text-destructive backdrop-blur-none! hover:text-destructive"
						onclick={() => onDeleteAll?.()}
						variant="ghost"
						size="sm"
						title="Delete every conversation"
					>
						<Trash2 class="h-4 w-4" />
						Delete all
					</Button>
				{/if}
			</div>
		{/if}
	{/if}
</div>
