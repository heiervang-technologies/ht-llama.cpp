<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { Terminal } from '@xterm/xterm';
	import { FitAddon } from '@xterm/addon-fit';
	import { WebLinksAddon } from '@xterm/addon-web-links';
	import { TermdService } from '$lib/services/termd.service';
	import '@xterm/xterm/css/xterm.css';

	interface Props {
		terminalId: string;
		/** Fired when the WS closes; lets the caller show a "disconnected"
		 *  banner without coupling this component to a store. */
		onDisconnect?: (clean: boolean) => void;
	}

	let { terminalId, onDisconnect }: Props = $props();

	let hostEl: HTMLDivElement | undefined = $state();
	let term: Terminal | undefined;
	let fit: FitAddon | undefined;
	let ws: WebSocket | undefined;
	let ro: ResizeObserver | undefined;

	// Live status. We keep reactive state out of $state() where it
	// isn't rendered reactively — the view only flips on disconnect.
	let connected = $state(false);
	let everOpened = $state(false);

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

	onMount(() => {
		if (!hostEl) return;
		term = new Terminal({
			fontFamily: 'var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace)',
			fontSize: 13,
			cursorBlink: true,
			scrollback: 5000,
			allowProposedApi: true,
			// Palette follows the app theme loosely. A proper theme
			// switch hook can come later; for now these colours track
			// CSS custom props that already exist.
			theme: {
				background: '#0b0b10',
				foreground: '#e6e6e6',
				cursor: '#e6e6e6',
				cursorAccent: '#0b0b10',
				selectionBackground: '#394267'
			}
		});
		fit = new FitAddon();
		term.loadAddon(fit);
		term.loadAddon(new WebLinksAddon());
		term.open(hostEl);
		fit.fit();

		// Open WS + wire I/O.
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
		term?.dispose();
	});
</script>

<div class="relative h-full w-full">
	<div bind:this={hostEl} class="h-full w-full bg-black"></div>
	{#if !connected}
		<div
			class="pointer-events-none absolute inset-x-0 top-0 flex justify-center p-2 text-xs text-muted-foreground"
		>
			{everOpened ? 'Disconnected.' : 'Connecting…'}
		</div>
	{/if}
</div>
