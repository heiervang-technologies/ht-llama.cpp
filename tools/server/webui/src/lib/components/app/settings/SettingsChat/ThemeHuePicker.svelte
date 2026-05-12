<script lang="ts">
	import { Shuffle, RotateCcw, Circle, CircleDashed } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import Label from '$lib/components/ui/label/label.svelte';

	export type ThemeMode = 'colorful' | 'pure-black' | 'pure-white';
	type HueKey = 'themePrimaryHue' | 'themeSecondaryHue';

	interface Props {
		primary: number;
		secondary: number;
		chroma: number;
		mode: ThemeMode;
		onHueChange: (key: HueKey, value: number) => void;
		onChromaChange: (value: number) => void;
		onModeChange: (mode: ThemeMode) => void;
	}

	let { primary, secondary, chroma, mode, onHueChange, onChromaChange, onModeChange }: Props =
		$props();

	const DEFAULT_PRIMARY = 295;
	const DEFAULT_SECONDARY = 190;
	const DEFAULT_CHROMA = 1;

	function clampHue(n: number): number {
		if (!Number.isFinite(n)) return 0;
		const mod = ((n % 360) + 360) % 360;
		return Math.round(mod);
	}

	function clamp01(n: number): number {
		if (!Number.isFinite(n)) return 1;
		return Math.max(0, Math.min(1, n));
	}

	function randomize() {
		const p = Math.floor(Math.random() * 360);
		// Keep secondary visually distinct from primary (at least 60° apart).
		let s = Math.floor(Math.random() * 360);
		if (Math.abs(((s - p + 540) % 360) - 180) < 60) {
			s = (p + 180 + Math.floor(Math.random() * 60) - 30 + 360) % 360;
		}
		onHueChange('themePrimaryHue', p);
		onHueChange('themeSecondaryHue', s);
		onChromaChange(DEFAULT_CHROMA);
		onModeChange('colorful');
	}

	function reset() {
		onHueChange('themePrimaryHue', DEFAULT_PRIMARY);
		onHueChange('themeSecondaryHue', DEFAULT_SECONDARY);
		onChromaChange(DEFAULT_CHROMA);
		onModeChange('colorful');
	}

	// Swatch chip — when chroma is 0 or mode is mono, show the actual
	// neutral tone the rest of the UI will use, so the chip never lies
	// about what the user has selected.
	function swatch(hue: number, lightness = 0.55): string {
		if (mode === 'pure-black') return 'oklch(0.06 0 0)';
		if (mode === 'pure-white') return 'oklch(0.95 0 0)';
		return `oklch(${lightness} ${0.18 * chroma} ${hue})`;
	}

	/**
	 * Decompose an `#rrggbb` hex into (hue, saturation-ish, lightness).
	 * `chromaEstimate` is `max - min` in [0, 1] — a quick proxy for OKLCH
	 * chroma. Used to round-trip the OS picker's *colourfulness*, not
	 * just its hue. Pure black / white have chromaEstimate = 0; we route
	 * those into the dedicated theme modes so the user gets exactly
	 * what they picked instead of the hue snapping the surface back to
	 * a tinted palette.
	 */
	function hexToTriplet(
		hex: string
	): { hue: number; chromaEstimate: number; lightness: number } | null {
		const m = /^#([0-9a-f]{6})$/i.exec(hex.trim());
		if (!m) return null;
		const n = parseInt(m[1], 16);
		const r = ((n >> 16) & 0xff) / 255;
		const g = ((n >> 8) & 0xff) / 255;
		const b = (n & 0xff) / 255;
		const max = Math.max(r, g, b);
		const min = Math.min(r, g, b);
		const d = max - min;
		const lightness = (max + min) / 2;
		let h = 0;
		if (d !== 0) {
			if (max === r) h = ((g - b) / d) % 6;
			else if (max === g) h = (b - r) / d + 2;
			else h = (r - g) / d + 4;
			h = clampHue(h * 60);
		}
		return { hue: h, chromaEstimate: d, lightness };
	}

	function applyPickedColor(key: HueKey, hex: string, currentHue: number) {
		const t = hexToTriplet(hex);
		if (!t) return;

		// Near-black or near-white achromatic pick → snap into the
		// matching dedicated theme mode. The hue value still gets stored
		// in case the user toggles back to colourful later.
		if (t.chromaEstimate < 0.04) {
			if (t.lightness < 0.08) {
				onModeChange('pure-black');
				return;
			}
			if (t.lightness > 0.92) {
				onModeChange('pure-white');
				return;
			}
			// Mid-tone grey → keep current hue, just zero the chroma.
			onChromaChange(0);
			onModeChange('colorful');
			return;
		}

		// Coloured pick → store hue + chroma + lightness.
		onHueChange(key, t.hue || currentHue);
		// Chroma estimate runs 0..1 already; give the slider a slight
		// bias so an "ordinary" colour reads as roughly full chroma
		// rather than half — most UI colours pick land in [0.4, 0.7].
		onChromaChange(clamp01(Math.min(1, t.chromaEstimate * 1.4)));
		if (mode !== 'colorful') onModeChange('colorful');
	}

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

	<!-- Mode presets — pure black / pure white short-circuit the
	     OKLCH ramps in app.css for users who want a strictly greyscale
	     UI. Tapping one of these swaps the theme; tapping Colourful
	     returns to hue + chroma controls. -->
	<div class="flex flex-wrap items-center gap-2">
		<Label class="text-xs text-muted-foreground">Mode</Label>
		<div class="flex flex-1 gap-1 rounded-md border border-border/60 bg-background/60 p-1">
			<Button
				type="button"
				variant={mode === 'colorful' ? 'default' : 'ghost'}
				size="sm"
				class="flex-1 gap-1.5"
				onclick={() => onModeChange('colorful')}
				title="Use the hue-driven palette"
			>
				<CircleDashed class="h-3.5 w-3.5" />
				Colourful
			</Button>
			<Button
				type="button"
				variant={mode === 'pure-black' ? 'default' : 'ghost'}
				size="sm"
				class="flex-1 gap-1.5"
				onclick={() => onModeChange('pure-black')}
				title="True greyscale on a near-black background"
			>
				<Circle class="h-3.5 w-3.5 fill-foreground" />
				Pure black
			</Button>
			<Button
				type="button"
				variant={mode === 'pure-white' ? 'default' : 'ghost'}
				size="sm"
				class="flex-1 gap-1.5"
				onclick={() => onModeChange('pure-white')}
				title="True greyscale on a near-white background"
			>
				<Circle class="h-3.5 w-3.5" />
				Pure white
			</Button>
		</div>
	</div>

	<!-- Hue + chroma controls only meaningful when the colourful mode is
	     active; in pure-black / pure-white they're informational, so
	     visibly dim them but keep them around — flipping back to
	     Colourful brings the previous hues straight back. -->
	<div class="space-y-2" class:opacity-50={mode !== 'colorful'}>
		<div class="flex items-center gap-3">
			<label
				class="relative h-5 w-5 shrink-0 cursor-pointer overflow-hidden rounded-full border border-border/60"
				style:background-color={swatch(primary, 0.5)}
				title="Click for full color picker — black or white auto-switches mode"
			>
				<input
					type="color"
					value={hueToHex(primary)}
					oninput={(e) =>
						applyPickedColor('themePrimaryHue', (e.target as HTMLInputElement).value, primary)}
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
					onHueChange('themePrimaryHue', clampHue(Number((e.target as HTMLInputElement).value)))}
				class="hue-slider flex-1"
				aria-label="Primary hue"
			/>
		</div>
		<div class="flex items-center gap-3">
			<label
				class="relative h-5 w-5 shrink-0 cursor-pointer overflow-hidden rounded-full border border-border/60"
				style:background-color={swatch(secondary, 0.75)}
				title="Click for full color picker — black or white auto-switches mode"
			>
				<input
					type="color"
					value={hueToHex(secondary)}
					oninput={(e) =>
						applyPickedColor('themeSecondaryHue', (e.target as HTMLInputElement).value, secondary)}
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
					onHueChange('themeSecondaryHue', clampHue(Number((e.target as HTMLInputElement).value)))}
				class="hue-slider flex-1"
				aria-label="Secondary hue"
			/>
		</div>
		<div class="flex items-center gap-3">
			<div
				class="relative h-5 w-5 shrink-0 overflow-hidden rounded-full border border-border/60"
				style:background={`linear-gradient(135deg, oklch(0.55 0 ${primary}), oklch(0.55 ${0.2 * chroma} ${primary}))`}
				aria-hidden="true"
			></div>
			<Label for="theme-chroma" class="min-w-24 text-xs text-muted-foreground">
				Saturation ({Math.round(chroma * 100)}%)
			</Label>
			<input
				id="theme-chroma"
				type="range"
				min="0"
				max="100"
				step="1"
				value={Math.round(chroma * 100)}
				oninput={(e) => onChromaChange(clamp01(Number((e.target as HTMLInputElement).value) / 100))}
				class="chroma-slider flex-1"
				aria-label="Saturation"
			/>
		</div>
	</div>

	<p class="text-xs text-muted-foreground">
		Click a swatch for an OS colour picker — black or white will switch the theme to a true
		greyscale mode. The saturation slider goes from full HT chroma to neutral grey.
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

	.chroma-slider {
		appearance: none;
		height: 0.5rem;
		border-radius: 9999px;
		/* Greyscale at the left → most-saturated mid hue at the right. */
		background: linear-gradient(to right, oklch(0.6 0 0), oklch(0.6 0.25 295));
		cursor: pointer;
	}

	.chroma-slider::-webkit-slider-thumb {
		appearance: none;
		height: 1rem;
		width: 1rem;
		border-radius: 9999px;
		background: var(--background);
		border: 2px solid var(--foreground);
		cursor: pointer;
	}

	.chroma-slider::-moz-range-thumb {
		height: 1rem;
		width: 1rem;
		border-radius: 9999px;
		background: var(--background);
		border: 2px solid var(--foreground);
		cursor: pointer;
	}
</style>
