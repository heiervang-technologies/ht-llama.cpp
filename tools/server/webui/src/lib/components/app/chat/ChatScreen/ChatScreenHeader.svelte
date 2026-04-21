<script lang="ts">
	import { Settings, PanelRight } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import { getChatSettingsDialogContext } from '$lib/contexts';
	import { BackendPill } from '$lib/components/app/navigation';
	import { artifactsStore } from '$lib/stores/artifacts.svelte';

	const sidebar = useSidebar();
	const chatSettingsDialog = getChatSettingsDialogContext();

	let artifactCount = $derived(artifactsStore.entries.length);
	let hasArtifacts = $derived(artifactCount > 0);
	let artifactTooltip = $derived(
		artifactCount === 1
			? '1 artifact · click to toggle drawer'
			: `${artifactCount} artifacts · click to toggle drawer`
	);
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
				class="relative rounded-full backdrop-blur-lg"
				title={artifactTooltip}
				aria-label={artifactTooltip}
			>
				<PanelRight class="h-4 w-4" />
				{#if artifactCount > 1}
					<span
						class="absolute -top-1 -right-1 flex h-4 min-w-[1rem] items-center justify-center rounded-full bg-primary px-1 text-[10px] font-semibold text-primary-foreground"
						aria-hidden="true"
					>
						{artifactCount > 9 ? '9+' : artifactCount}
					</span>
				{/if}
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
