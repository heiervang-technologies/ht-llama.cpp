<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { Terminal } from '@xterm/xterm';
	import { FitAddon } from '@xterm/addon-fit';
	import { WebLinksAddon } from '@xterm/addon-web-links';
	import { WebglAddon } from '@xterm/addon-webgl';
	import { TermdService } from '$lib/services/termd.service';
	import { DEFAULT_TERMINAL_THEME_ID, MONO_FONT_STACK, resolveTheme } from './terminal-themes';
	import '@xterm/xterm/css/xterm.css';

	interface Props {
		terminalId: string;
		/** Theme id from `TERMINAL_THEMES`. Falls back to default. */
		themeId?: string | null;
		onDisconnect?: (clean: boolean) => void;
	}

	let { terminalId, themeId = DEFAULT_TERMINAL_THEME_ID, onDisconnect }: Props = $props();

	let hostEl: HTMLDivElement | undefined = $state();
	let term: Terminal | undefined;
	let fit: FitAddon | undefined;
	let ws: WebSocket | undefined;
	let ro: ResizeObserver | undefined;
	let webglAddon: WebglAddon | undefined;

	let connected = $state(false);
	let everOpened = $state(false);

	let theme = $derived(resolveTheme(themeId));

	function sendResize() {
		if (!term || !fit || !ws || ws.readyState !== WebSocket.OPEN) return;
		try {
			fit.fit();
			const { cols, rows } = term;
			ws.send(JSON.stringify({ t: 'resize', cols, rows }));
		} catch (err) {
			console.warn('[terminal] resize failed', err);
		}
	}

	// Live palette swap on theme change. `options.theme` alone updates
	// the canvas renderer but the WebGL renderer caches its glyph
	// atlas — the old colours stay on-screen until something triggers
	// a full redraw. We reset both explicitly so switching themes
	// doesn't require re-entering the route.
	$effect(() => {
		if (!term) return;
		term.options.theme = theme.palette;
		term.options.fontFamily = theme.fontFamily ?? MONO_FONT_STACK;
		// `clearTextureAtlas` lives on the Terminal when the WebGL
		// addon is active; bail if we're on canvas-only.
		type WithAtlasClear = Terminal & { clearTextureAtlas?: () => void };
		(term as WithAtlasClear).clearTextureAtlas?.();
		term.refresh(0, term.rows - 1);
	});

	onMount(() => {
		if (!hostEl) return;
		term = new Terminal({
			// Literal stack so the canvas renderer can measure glyphs.
			fontFamily: theme.fontFamily ?? MONO_FONT_STACK,
			fontSize: 13,
			fontWeight: 400,
			fontWeightBold: 600,
			letterSpacing: 0,
			// Tighter vertical rhythm makes block characters (█ ▄ ▀ ▌ ▐)
			// abut without the ghost-grid gap the canvas renderer
			// otherwise draws. Pure text still reads fine at 1.05.
			lineHeight: 1.05,
			cursorBlink: true,
			cursorStyle: 'block',
			scrollback: 5000,
			allowProposedApi: true,
			theme: theme.palette
		});
		fit = new FitAddon();
		term.loadAddon(fit);
		term.loadAddon(new WebLinksAddon());
		term.open(hostEl);

		// Try WebGL — it renders block glyphs edge-to-edge without
		// the inter-cell seams the canvas renderer leaves, which is
		// what makes ANSI art look grid-ish. Fall back silently if
		// the context can't be created (older webviews, forced-
		// software GL, etc.).
		try {
			webglAddon = new WebglAddon();
			webglAddon.onContextLoss(() => {
				webglAddon?.dispose();
				webglAddon = undefined;
			});
			term.loadAddon(webglAddon);
		} catch (err) {
			console.warn('[terminal] WebGL renderer unavailable, falling back to canvas', err);
		}

		fit.fit();

		const url = TermdService.wsUrl(terminalId);
		ws = new WebSocket(url);
		ws.binaryType = 'arraybuffer';
		ws.onopen = () => {
			connected = true;
			everOpened = true;
			sendResize();
			term?.focus();
		};
		ws.onmessage = (ev) => {
			if (typeof ev.data === 'string') {
				term?.write(ev.data);
			} else if (ev.data instanceof ArrayBuffer) {
				term?.write(new Uint8Array(ev.data));
			}
		};
		ws.onclose = (ev) => {
			connected = false;
			onDisconnect?.(ev.wasClean);
		};
		ws.onerror = () => {
			connected = false;
		};

		term.onData((data) => {
			if (ws && ws.readyState === WebSocket.OPEN) {
				ws.send(new TextEncoder().encode(data));
			}
		});

		ro = new ResizeObserver(() => sendResize());
		ro.observe(hostEl);
	});

	onDestroy(() => {
		ro?.disconnect();
		ws?.close();
		webglAddon?.dispose();
		term?.dispose();
	});
</script>

<div class="term-shell term-shell--{theme.overlay ?? 'flat'} relative h-full w-full">
	<div bind:this={hostEl} class="term-host absolute inset-0"></div>

	<!-- Scanline + vignette overlays. Pure CSS — the real terminal
		   lives underneath. `pointer-events: none` so they don't eat
		   mouse / keyboard. -->
	{#if theme.overlay === 'crt' || theme.overlay === 'amber-crt'}
		<div class="term-overlay term-overlay--scanlines"></div>
		<div class="term-overlay term-overlay--vignette"></div>
		<div class="term-overlay term-overlay--glow term-overlay--glow-{theme.overlay}"></div>
	{:else if theme.overlay === 'matrix'}
		<div class="term-overlay term-overlay--scanlines term-overlay--scanlines-soft"></div>
	{:else if theme.overlay === 'blueprint'}
		<div class="term-overlay term-overlay--grid"></div>
	{/if}

	{#if !connected}
		<div
			class="pointer-events-none absolute inset-x-0 top-0 flex justify-center p-2 text-xs text-muted-foreground"
		>
			{everOpened ? 'Disconnected.' : 'Connecting…'}
		</div>
	{/if}
</div>

<style>
	.term-shell {
		isolation: isolate;
	}

	.term-overlay {
		position: absolute;
		inset: 0;
		pointer-events: none;
		z-index: 2;
	}

	/* Classic CRT scanlines: 2-pixel rows of translucent black every
	   3 CSS px. Offsetting slightly with a keyframe gives the
	   rolling-sync shimmer without being distracting. */
	.term-overlay--scanlines {
		background-image: linear-gradient(
			to bottom,
			transparent 0,
			transparent 1px,
			rgba(0, 0, 0, 0.22) 1px,
			rgba(0, 0, 0, 0.22) 2px
		);
		background-size: 100% 3px;
		mix-blend-mode: multiply;
		animation: term-scanroll 8s linear infinite;
	}
	.term-overlay--scanlines-soft {
		background-image: linear-gradient(
			to bottom,
			transparent 0,
			transparent 1px,
			rgba(0, 0, 0, 0.1) 1px,
			rgba(0, 0, 0, 0.1) 2px
		);
		background-size: 100% 4px;
		animation: none;
	}

	@keyframes term-scanroll {
		from {
			background-position: 0 0;
		}
		to {
			background-position: 0 3px;
		}
	}

	/* Barrel-vignette — a soft radial fade that darkens the corners
	   and mimics CRT glass bulge. Subtler than real curvature; if
	   we ever want actual warp we can swap this for an SVG filter. */
	.term-overlay--vignette {
		background: radial-gradient(ellipse at center, transparent 55%, rgba(0, 0, 0, 0.55) 100%);
	}

	/* Phosphor bloom — coloured inner glow that tints the whole
	   frame toward the cursor colour. One class per overlay so the
	   tint matches the theme. */
	.term-overlay--glow {
		box-shadow: inset 0 0 80px 0 rgba(0, 0, 0, 0);
	}
	.term-overlay--glow-crt {
		box-shadow:
			inset 0 0 120px rgba(51, 255, 102, 0.18),
			inset 0 0 40px rgba(51, 255, 102, 0.25);
	}
	.term-overlay--glow-amber-crt {
		box-shadow:
			inset 0 0 120px rgba(255, 187, 51, 0.18),
			inset 0 0 40px rgba(255, 187, 51, 0.25);
	}

	/* Blueprint grid paper. Stays static — no animation. */
	.term-overlay--grid {
		background-image:
			linear-gradient(to right, rgba(154, 230, 255, 0.08) 1px, transparent 1px),
			linear-gradient(to bottom, rgba(154, 230, 255, 0.08) 1px, transparent 1px);
		background-size: 24px 24px;
	}

	/* Barrel-curve the xterm canvas itself for CRT themes. The
	   canvas is inside .term-host; perspective + slight
	   translateZ(0) enforces compositing so the filter stays
	   GPU-accelerated. */
	.term-shell--crt .term-host,
	.term-shell--amber-crt .term-host {
		transform: translateZ(0);
		filter: contrast(1.06) saturate(1.1);
	}

	/* Paper theme: tiny warm vignette so full-screen whites don't
	   blow out the rest of the app. */
	.term-shell--paper {
		background: #f6f4ee;
	}
</style>
