import { TtsService, type TtsVoice } from '$lib/services/tts.service';

/**
 * voicesStore - reactive list of TTS voices fetched from the configured server.
 *
 * `/v1/audio/voices` is an OpenAI-style extension; not every TTS backend
 * implements it. The store degrades to an empty list on 404 / unreachable, so
 * callers should fall back to a free-text voice input rather than treating the
 * empty list as a hard error.
 */
class VoicesStore {
	voices = $state<TtsVoice[]>([]);
	loading = $state(false);
	/** Human-readable error from the last fetch, or null if the last fetch succeeded (or returned the documented empty-on-404 case). */
	lastError = $state<string | null>(null);
	/** Timestamp of last successful fetch; 0 = never fetched. */
	lastFetchedAt = $state(0);

	private inflight: AbortController | null = null;

	async fetch(): Promise<void> {
		this.inflight?.abort();
		this.inflight = new AbortController();
		this.loading = true;
		this.lastError = null;
		try {
			const list = await TtsService.fetchVoices(this.inflight.signal);
			this.voices = list;
			this.lastFetchedAt = Date.now();
		} catch (err) {
			// Aborts happen whenever a newer fetch supersedes this one — not a
			// real error, don't surface it.
			if ((err as { name?: string })?.name === 'AbortError') return;
			// Don't clobber the previously-fetched list — keep showing whatever
			// worked last, and surface the error alongside it.
			this.lastError = err instanceof Error ? err.message : String(err);
		} finally {
			this.loading = false;
			this.inflight = null;
		}
	}

	clear(): void {
		this.voices = [];
		this.lastFetchedAt = 0;
		this.lastError = null;
	}
}

export const voicesStore = new VoicesStore();
