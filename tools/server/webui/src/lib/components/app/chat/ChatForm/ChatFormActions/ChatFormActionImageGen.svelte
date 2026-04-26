<script lang="ts">
	import { Image, ImageOff } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { config, settingsStore } from '$lib/stores/settings.svelte';

	interface Props {
		disabled?: boolean;
		class?: string;
	}

	let { disabled = false, class: className = '' }: Props = $props();

	let enabled = $derived(Boolean(config().imageGenEnabled));

	function toggle() {
		settingsStore.updateConfig('imageGenEnabled', !enabled);
	}
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		{#snippet child({ props })}
			<Button
				{...props}
				type="button"
				variant={enabled ? 'default' : 'ghost'}
				size="sm"
				{disabled}
				onclick={toggle}
				aria-pressed={enabled}
				class="h-8 gap-1.5 rounded-full px-3 text-xs font-medium transition-colors {className}"
			>
				{#if enabled}
					<Image class="h-3.5 w-3.5" />
					<span>Image</span>
				{:else}
					<ImageOff class="h-3.5 w-3.5 opacity-70" />
					<span class="opacity-70">Image</span>
				{/if}
			</Button>
		{/snippet}
	</Tooltip.Trigger>

	<Tooltip.Content side="top">
		{#if enabled}
			<p>
				Image generation enabled — model can call generate_image / edit_image, and /image works in
				the composer.
			</p>
		{:else}
			<p>
				Image generation disabled — click to enable. Same flag as Settings → Images → Enable image
				generation.
			</p>
		{/if}
	</Tooltip.Content>
</Tooltip.Root>
