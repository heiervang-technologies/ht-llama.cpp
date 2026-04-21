<script lang="ts">
	import { Settings, MessageSquarePlus, Columns, FileText, Eye, Sparkles } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import { getChatSettingsDialogContext } from '$lib/contexts';
	import { BackendPill } from '$lib/components/app/navigation';
	import { config, settingsStore } from '$lib/stores/settings.svelte';

	interface Props {
		name: string;
		view: 'edit' | 'preview' | 'split';
		saving: boolean;
		onRename: (next: string) => void;
		onViewChange: (next: 'edit' | 'preview' | 'split') => void;
		onChatAbout: () => void;
	}

	let { name, view, saving, onRename, onViewChange, onChatAbout }: Props = $props();

	const sidebar = useSidebar();
	const chatSettingsDialog = getChatSettingsDialogContext();

	let localName = $derived(name);
	let inlineOn = $derived(Boolean(config().inlineCompletionEnabled));

	function commit() {
		if (localName !== name) onRename(localName);
	}

	function toggleInlineCompletion() {
		settingsStore.updateConfig('inlineCompletionEnabled', !config().inlineCompletionEnabled);
	}
</script>

<header
	class="pointer-events-none fixed top-0 right-0 left-0 z-50 flex items-center gap-2 p-2 duration-200 ease-linear md:p-4 {sidebar.open
		? 'md:left-[var(--sidebar-width)]'
		: ''}"
>
	<div class="pointer-events-auto ml-12 flex min-w-0 flex-1 items-center gap-2 md:ml-12">
		<input
			type="text"
			class="min-w-0 flex-1 truncate rounded-md bg-transparent px-2 py-1 text-sm font-medium text-foreground outline-none focus:bg-muted/40"
			bind:value={localName}
			onblur={commit}
			onkeydown={(e) => {
				if (e.key === 'Enter') {
					e.preventDefault();
					(e.currentTarget as HTMLInputElement).blur();
				}
			}}
			placeholder="Untitled"
		/>

		{#if saving}
			<span class="text-xs text-muted-foreground">Saving…</span>
		{/if}
	</div>

	<div class="pointer-events-auto flex items-center gap-1">
		<div class="hidden items-center rounded-md border bg-background/70 p-0.5 backdrop-blur md:flex">
			<Button
				variant={view === 'edit' ? 'secondary' : 'ghost'}
				size="sm"
				class="h-7 px-2"
				onclick={() => onViewChange('edit')}
				title="Editor only"
			>
				<FileText class="h-3.5 w-3.5" />
			</Button>
			<Button
				variant={view === 'split' ? 'secondary' : 'ghost'}
				size="sm"
				class="h-7 px-2"
				onclick={() => onViewChange('split')}
				title="Split"
			>
				<Columns class="h-3.5 w-3.5" />
			</Button>
			<Button
				variant={view === 'preview' ? 'secondary' : 'ghost'}
				size="sm"
				class="h-7 px-2"
				onclick={() => onViewChange('preview')}
				title="Preview only"
			>
				<Eye class="h-3.5 w-3.5" />
			</Button>
		</div>

		<Button
			variant={inlineOn ? 'secondary' : 'ghost'}
			size="sm"
			onclick={toggleInlineCompletion}
			class="gap-1.5 rounded-full backdrop-blur-lg"
			title={inlineOn
				? 'AI ghost-text completions are ON (Tab to accept, Esc to dismiss)'
				: 'Enable AI ghost-text completions'}
		>
			<Sparkles class="h-4 w-4 {inlineOn ? 'text-primary' : ''}" />
			<span class="hidden md:inline">{inlineOn ? 'AI on' : 'AI off'}</span>
		</Button>

		<Button
			variant="ghost"
			size="sm"
			onclick={onChatAbout}
			class="gap-1.5 rounded-full backdrop-blur-lg"
			title="Start a chat about this doc"
		>
			<MessageSquarePlus class="h-4 w-4" />
			<span class="hidden md:inline">Chat about this</span>
		</Button>

		<BackendPill />

		<Button
			variant="ghost"
			size="icon-lg"
			onclick={() => chatSettingsDialog.open()}
			class="rounded-full backdrop-blur-lg"
		>
			<Settings class="h-4 w-4" />
		</Button>
	</div>
</header>
