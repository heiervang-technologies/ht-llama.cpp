<script lang="ts">
	import { Settings, PanelRight } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import { getChatSettingsDialogContext } from '$lib/contexts';
	import { BackendPill } from '$lib/components/app/navigation';
	import { artifactsStore } from '$lib/stores/artifacts.svelte';

	const sidebar = useSidebar();
	const chatSettingsDialog = getChatSettingsDialogContext();

	let hasArtifacts = $derived(artifactsStore.entries.length > 0);
</script>

<header
	class="pointer-events-none fixed top-0 right-0 left-0 z-50 flex items-center justify-end p-2 duration-200 ease-linear md:p-4 {sidebar.open
		? 'md:left-[var(--sidebar-width)]'
		: ''}"
>
	<div class="pointer-events-auto flex items-center space-x-2">
		<BackendPill />

		{#if hasArtifacts}
			<Button
				variant="ghost"
				size="icon-lg"
				onclick={() => artifactsStore.toggle()}
				class="rounded-full backdrop-blur-lg"
				title="Toggle artifact drawer"
				aria-label="Toggle artifact drawer"
			>
				<PanelRight class="h-4 w-4" />
			</Button>
		{/if}

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
