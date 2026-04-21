<script lang="ts">
	import { Shuffle, RotateCcw } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import Label from '$lib/components/ui/label/label.svelte';

	interface Props {
		primary: number;
		secondary: number;
		onChange: (key: 'themePrimaryHue' | 'themeSecondaryHue', value: number) => void;
	}

	let { primary, secondary, onChange }: Props = $props();

	const DEFAULT_PRIMARY = 295;
	const DEFAULT_SECONDARY = 190;

	function clampHue(n: number): number {
		if (!Number.isFinite(n)) return 0;
		const mod = ((n % 360) + 360) % 360;
		return Math.round(mod);
	}

	function randomize() {
		const p = Math.floor(Math.random() * 360);
		// Keep secondary visually distinct from primary (at least 60° apart).
		let s = Math.floor(Math.random() * 360);
		if (Math.abs(((s - p + 540) % 360) - 180) > 120) {
			s = (p + 180 + Math.floor(Math.random() * 60) - 30 + 360) % 360;
		}
		onChange('themePrimaryHue', p);
		onChange('themeSecondaryHue', s);
	}

	function reset() {
		onChange('themePrimaryHue', DEFAULT_PRIMARY);
		onChange('themeSecondaryHue', DEFAULT_SECONDARY);
	}

	function swatch(hue: number, lightness = 0.55): string {
		return `oklch(${lightness} 0.18 ${hue})`;
	}
</script>

<div class="space-y-3 rounded-md border border-border/60 bg-muted/30 p-4">
	<div class="flex items-center justify-between gap-2">
		<Label class="text-sm font-medium">Theme hues</Label>
		<div class="flex items-center gap-1">
			<Button
				type="button"
				variant="ghost"
				size="sm"
				class="gap-1.5"
				onclick={randomize}
				title="Randomize primary + secondary hue"
			>
				<Shuffle class="h-3.5 w-3.5" />
				Randomize
			</Button>
			<Button
				type="button"
				variant="ghost"
				size="sm"
				class="gap-1.5"
				onclick={reset}
				title="Reset to HT turquoise/purple"
			>
				<RotateCcw class="h-3.5 w-3.5" />
				Reset
			</Button>
		</div>
	</div>

	<div class="space-y-2">
		<div class="flex items-center gap-3">
			<span
				class="h-5 w-5 shrink-0 rounded-full border border-border/60"
				style:background-color={swatch(primary, 0.5)}
				aria-hidden="true"
			></span>
			<Label for="theme-primary-hue" class="min-w-24 text-xs text-muted-foreground">
				Primary ({primary}°)
			</Label>
			<input
				id="theme-primary-hue"
				type="range"
				min="0"
				max="359"
				step="1"
				value={primary}
				oninput={(e) =>
					onChange('themePrimaryHue', clampHue(Number((e.target as HTMLInputElement).value)))}
				class="hue-slider flex-1"
				aria-label="Primary hue"
			/>
		</div>
		<div class="flex items-center gap-3">
			<span
				class="h-5 w-5 shrink-0 rounded-full border border-border/60"
				style:background-color={swatch(secondary, 0.75)}
				aria-hidden="true"
			></span>
			<Label for="theme-secondary-hue" class="min-w-24 text-xs text-muted-foreground">
				Secondary ({secondary}°)
			</Label>
			<input
				id="theme-secondary-hue"
				type="range"
				min="0"
				max="359"
				step="1"
				value={secondary}
				oninput={(e) =>
					onChange(
						'themeSecondaryHue',
						clampHue(Number((e.target as HTMLInputElement).value))
					)}
				class="hue-slider flex-1"
				aria-label="Secondary hue"
			/>
		</div>
	</div>

	<p class="text-xs text-muted-foreground">
		Primary drives text and accents; secondary drives neutral surfaces. Changes apply live.
	</p>
</div>

<style>
	.hue-slider {
		appearance: none;
		height: 0.5rem;
		border-radius: 9999px;
		background: linear-gradient(
			to right,
			oklch(0.7 0.2 0),
			oklch(0.7 0.2 60),
			oklch(0.7 0.2 120),
			oklch(0.7 0.2 180),
			oklch(0.7 0.2 240),
			oklch(0.7 0.2 300),
			oklch(0.7 0.2 360)
		);
		cursor: pointer;
	}

	.hue-slider::-webkit-slider-thumb {
		appearance: none;
		height: 1rem;
		width: 1rem;
		border-radius: 9999px;
		background: var(--background);
		border: 2px solid var(--foreground);
		cursor: pointer;
	}

	.hue-slider::-moz-range-thumb {
		height: 1rem;
		width: 1rem;
		border-radius: 9999px;
		background: var(--background);
		border: 2px solid var(--foreground);
		cursor: pointer;
	}
</style>
