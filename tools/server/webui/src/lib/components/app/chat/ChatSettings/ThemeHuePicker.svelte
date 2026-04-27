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
		// The expression computes circular distance in [0, 180]; below 60 means
		// the two hues read as almost the same colour, so push secondary to the
		// opposite side of the wheel with some jitter.
		let s = Math.floor(Math.random() * 360);
		if (Math.abs(((s - p + 540) % 360) - 180) < 60) {
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

	/**
	 * Convert an `#rrggbb` hex string to its HSL hue in degrees [0, 360).
	 * The theme system only stores the hue channel — saturation and
	 * lightness are derived per-context via OKLCH ramps in app.css — so
	 * we deliberately discard everything except H. Picking a colour with
	 * S=0 (pure grey) is degenerate (hue is undefined); fall back to
	 * keeping the current value in that case so the slider doesn't jump.
	 */
	function hexToHue(hex: string, fallback: number): number {
		const m = /^#([0-9a-f]{6})$/i.exec(hex.trim());
		if (!m) return fallback;
		const n = parseInt(m[1], 16);
		const r = ((n >> 16) & 0xff) / 255;
		const g = ((n >> 8) & 0xff) / 255;
		const b = (n & 0xff) / 255;
		const max = Math.max(r, g, b);
		const min = Math.min(r, g, b);
		const d = max - min;
		if (d === 0) return fallback; // achromatic — keep current hue
		let h: number;
		if (max === r) h = ((g - b) / d) % 6;
		else if (max === g) h = (b - r) / d + 2;
		else h = (r - g) / d + 4;
		return clampHue(h * 60);
	}

	/**
	 * Hue → `#rrggbb` round-trip so the colour input opens with the
	 * current pick selected. Uses HSL with S=70%, L=55% — visually
	 * matches the swatch chips for a consistent reference colour.
	 */
	function hueToHex(hue: number): string {
		const s = 0.7;
		const l = 0.55;
		const c = (1 - Math.abs(2 * l - 1)) * s;
		const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
		const m = l - c / 2;
		let r = 0,
			g = 0,
			b = 0;
		if (hue < 60) {
			r = c;
			g = x;
		} else if (hue < 120) {
			r = x;
			g = c;
		} else if (hue < 180) {
			g = c;
			b = x;
		} else if (hue < 240) {
			g = x;
			b = c;
		} else if (hue < 300) {
			r = x;
			b = c;
		} else {
			r = c;
			b = x;
		}
		const toHex = (v: number) =>
			Math.round((v + m) * 255)
				.toString(16)
				.padStart(2, '0');
		return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
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
			<!-- Native color input. Click the swatch and the OS opens its
			     full HSL/HSV picker — no extra dependency, accessible by
			     default. We only keep the hue channel since the theme
			     system derives S/L per-context via OKLCH ramps. -->
			<label
				class="relative h-5 w-5 shrink-0 cursor-pointer overflow-hidden rounded-full border border-border/60"
				style:background-color={swatch(primary, 0.5)}
				title="Click for full color picker"
			>
				<input
					type="color"
					value={hueToHex(primary)}
					oninput={(e) =>
						onChange('themePrimaryHue', hexToHue((e.target as HTMLInputElement).value, primary))}
					class="absolute inset-0 h-full w-full cursor-pointer opacity-0"
					aria-label="Primary color picker"
				/>
			</label>
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
			<label
				class="relative h-5 w-5 shrink-0 cursor-pointer overflow-hidden rounded-full border border-border/60"
				style:background-color={swatch(secondary, 0.75)}
				title="Click for full color picker"
			>
				<input
					type="color"
					value={hueToHex(secondary)}
					oninput={(e) =>
						onChange(
							'themeSecondaryHue',
							hexToHue((e.target as HTMLInputElement).value, secondary)
						)}
					class="absolute inset-0 h-full w-full cursor-pointer opacity-0"
					aria-label="Secondary color picker"
				/>
			</label>
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
					onChange('themeSecondaryHue', clampHue(Number((e.target as HTMLInputElement).value)))}
				class="hue-slider flex-1"
				aria-label="Secondary hue"
			/>
		</div>
	</div>

	<p class="text-xs text-muted-foreground">
		Click a swatch for a full color picker, or drag the slider for hue only. Saturation and
		lightness are derived per-context — only the hue channel is stored.
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
