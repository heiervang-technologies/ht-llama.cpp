import { toast } from 'svelte-sonner';
import { TtsService } from '$lib/services/tts.service';
import { settingsStore } from '$lib/stores/settings.svelte';

const MAX_CONSECUTIVE_FAILURES = 2;

class TtsStore {
	#speakingId = $state<string | null>(null);
	#loadingId = $state<string | null>(null);
	#current: HTMLAudioElement | null = null;
	#currentUrl: string | null = null;
	#controller: AbortController | null = null;
	#consecutiveFailures = 0;

	get speakingId(): string | null {
		return this.#speakingId;
	}

	get loadingId(): string | null {
		return this.#loadingId;
	}

	isConfigured(): boolean {
		return TtsService.isConfigured();
	}

	stop(): void {
		if (this.#controller) {
			this.#controller.abort();
			this.#controller = null;
		}
		if (this.#current) {
			this.#current.pause();
			this.#current.src = '';
			this.#current = null;
		}
		if (this.#currentUrl) {
			URL.revokeObjectURL(this.#currentUrl);
			this.#currentUrl = null;
		}
		this.#speakingId = null;
		this.#loadingId = null;
	}

	async speak(id: string, text: string): Promise<void> {
		this.stop();

		const controller = new AbortController();
		this.#controller = controller;
		this.#loadingId = id;

		let blob: Blob;
		try {
			blob = await TtsService.synthesize(text, { signal: controller.signal });
		} catch (err) {
			if (this.#loadingId === id) this.#loadingId = null;
			if (err instanceof DOMException && err.name === 'AbortError') return;
			const msg = err instanceof Error ? err.message : String(err);
			console.error('[tts]', err);
			this.#consecutiveFailures += 1;
			toast.error(`TTS failed: ${msg}`);
			if (
				this.#consecutiveFailures >= MAX_CONSECUTIVE_FAILURES &&
				settingsStore.config.ttsAutoplay
			) {
				settingsStore.updateConfig('ttsAutoplay', false);
				toast.warning(
					'TTS autoplay disabled after repeated failures. Re-enable in Settings once TTS is working.'
				);
			}
			return;
		}

		this.#consecutiveFailures = 0;
		if (controller.signal.aborted) return;

		const url = URL.createObjectURL(blob);
		this.#currentUrl = url;

		const audio = new Audio(url);
		this.#current = audio;
		this.#loadingId = null;
		this.#speakingId = id;

		const cleanup = () => {
			if (this.#speakingId === id) this.#speakingId = null;
			if (this.#currentUrl === url) {
				URL.revokeObjectURL(url);
				this.#currentUrl = null;
			}
			if (this.#current === audio) this.#current = null;
		};

		audio.addEventListener('ended', cleanup, { once: true });
		audio.addEventListener('error', cleanup, { once: true });
		controller.signal.addEventListener('abort', () => {
			audio.pause();
			cleanup();
		});

		try {
			await audio.play();
		} catch (err) {
			cleanup();
			const msg = err instanceof Error ? err.message : String(err);
			console.error('[tts] audio.play failed', err);
			toast.error(`TTS playback failed: ${msg}`);
		}
	}

	async toggle(id: string, text: string): Promise<void> {
		if (this.#speakingId === id || this.#loadingId === id) {
			this.stop();
			return;
		}
		await this.speak(id, text);
	}
}

export const ttsStore = new TtsStore();
