<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import { config } from '$lib/stores/settings.svelte';
	import { IsMobile } from '$lib/hooks/is-mobile.svelte';
	import { TtsService } from '$lib/services/tts.service';
	import LoraAdapters from '$lib/components/app/lora/LoraAdapters.svelte';
	import { ModelsSelector, ModelsSelectorSheet } from '$lib/components/app';
	import VoicePicker from './VoicePicker.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		isOffline?: boolean;
		activeModelId?: string | null;
		conversationModel?: string | null;
		isRouter?: boolean;
	}

	let {
		class: className = '',
		disabled = false,
		isOffline = false,
		activeModelId = null,
		conversationModel = null,
		isRouter = false
	}: Props = $props();

	let currentConfig = $derived(config());
	let showVoice = $derived(currentConfig.showVoicePicker !== false && TtsService.isConfigured());

	let isMobile = new IsMobile();

	let selectorDesktop: ModelsSelector | undefined = $state(undefined);
	let selectorMobile: ModelsSelectorSheet | undefined = $state(undefined);

	export function openModelSelector() {
		if (isMobile.current) {
			selectorMobile?.open();
		} else {
			selectorDesktop?.open();
		}
	}
</script>

<!--
  ChainPicker wraps voice + LoRA + model selectors into a single rounded strip.
  Each child keeps its own text/icons; only its pill background and rounded
  corners are neutralised so everything slots into one continuous rectangle.

  Between segments we draw a stepped, asymmetric seam: an SVG path that
  crosses the junction with a tongue on the bottom half and a notch on the
  top half, so the seam looks like two puzzle pieces locked together rather
  than a symmetrical divider.
-->
<div class={cn('chain-picker', className)} data-slot="chain-picker">
	{#if showVoice}
		<div class="chain-seg chain-seg-voice">
			<VoicePicker {disabled} />
		</div>

		<span class="chain-notch" aria-hidden="true">
			<!-- Asymmetric puzzle seam: notch on top, tab on bottom. -->
			<svg viewBox="0 0 10 24" preserveAspectRatio="none">
				<path
					d="M 5 0
					   L 5 7
					   C 1 7, 1 11, 5 11
					   L 5 13
					   C 9 13, 9 17, 5 17
					   L 5 24"
				/>
			</svg>
		</span>
	{/if}

	<div class="chain-seg chain-seg-lora">
		<LoraAdapters
			disabled={disabled || isOffline}
			modelId={isRouter ? (activeModelId ?? undefined) : undefined}
		/>
	</div>

	<span class="chain-notch" aria-hidden="true">
		<svg viewBox="0 0 10 24" preserveAspectRatio="none">
			<path
				d="M 5 0
				   L 5 7
				   C 1 7, 1 11, 5 11
				   L 5 13
				   C 9 13, 9 17, 5 17
				   L 5 24"
			/>
		</svg>
	</span>

	<div class="chain-seg chain-seg-model">
		{#if isMobile.current}
			<ModelsSelectorSheet
				disabled={disabled || isOffline}
				bind:this={selectorMobile}
				currentModel={conversationModel}
				forceForegroundText
				useGlobalSelection
			/>
		{:else}
			<ModelsSelector
				disabled={disabled || isOffline}
				bind:this={selectorDesktop}
				currentModel={conversationModel}
				forceForegroundText
				useGlobalSelection
			/>
		{/if}
	</div>
</div>

<style>
	.chain-picker {
		position: relative;
		display: inline-flex;
		align-items: stretch;
		gap: 0;
		background-color: color-mix(in srgb, var(--muted-foreground) 10%, transparent);
		border-radius: 8px;
		padding: 2px;
	}

	.chain-seg {
		position: relative;
		display: inline-flex;
		align-items: center;
		padding: 0 4px;
		z-index: 1;
	}

	.chain-notch {
		position: relative;
		display: inline-flex;
		align-items: stretch;
		width: 10px;
		flex-shrink: 0;
		z-index: 2;
	}

	.chain-notch svg {
		width: 100%;
		height: 100%;
		fill: none;
		stroke: color-mix(in srgb, var(--muted-foreground) 35%, transparent);
		stroke-width: 1.25;
	}

	/*
	 * Children render their own <Popover.Trigger> / <button> with
	 * `rounded-sm bg-muted-foreground/10`. Neutralise both so the unified
	 * strip reads as one rectangle. Keep padding / icons / hover intact.
	 */
	:global(.chain-picker .chain-seg button),
	:global(.chain-picker .chain-seg span[aria-disabled]) {
		background-color: transparent !important;
		border-radius: 4px !important;
	}

	/*
	 * When a segment's trigger is hovered or open, give it a subtle
	 * highlight so the user sees it as an interactive region inside the
	 * merged strip. Tighter than the pill bg since the strip already has
	 * a muted background.
	 */
	:global(.chain-picker .chain-seg button:hover) {
		background-color: color-mix(in srgb, var(--muted-foreground) 12%, transparent) !important;
	}
</style>
