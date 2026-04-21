import { TtsService } from '$lib/services/tts.service';

class TtsStore {
	#speakingId = $state<string | null>(null);
	#loadingId = $state<string | null>(null);
	#current: HTMLAudioElement | null = null;
	#currentUrl: string | null = null;
	#controller: AbortController | null = null;

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
			if (!(err instanceof DOMException && err.name === 'AbortError')) {
				throw err;
			}
			return;
		}

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
			throw err;
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
