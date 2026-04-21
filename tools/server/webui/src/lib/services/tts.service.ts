import { config } from '$lib/stores/settings.svelte';

export interface TtsSynthesizeOptions {
	signal?: AbortSignal;
}

/**
 * Stateless client for OpenAI-compatible TTS servers (incl. Qwen3-TTS).
 * Posts to `<baseUrl>/v1/audio/speech` and returns the audio Blob.
 *
 * Settings consumed:
 * - ttsBaseUrl   (required; empty = disabled)
 * - ttsApiKey    (optional bearer token)
 * - ttsModel     (required)
 * - ttsVoice     (optional — ignored by Qwen3 when ref_audio is set)
 * - ttsFormat    (default 'wav')
 * - ttsRefAudio  (optional data: URI — enables Qwen3-TTS voice cloning.
 *                 When present, request includes ref_audio and
 *                 x_vector_only_mode=true)
 */
export class TtsService {
	static isConfigured(): boolean {
		const c = config();
		return Boolean(c.ttsBaseUrl?.toString().trim() && c.ttsModel?.toString().trim());
	}

	static async synthesize(text: string, opts: TtsSynthesizeOptions = {}): Promise<Blob> {
		const c = config();
		const baseUrl = c.ttsBaseUrl?.toString().trim().replace(/\/+$/, '') ?? '';
		if (!baseUrl || !c.ttsModel?.toString().trim()) {
			throw new Error('TTS is not configured. Set a Base URL and Model in settings.');
		}

		const format = (c.ttsFormat?.toString().trim() || 'wav').toLowerCase();

		const headers: Record<string, string> = {
			'Content-Type': 'application/json'
		};
		const apiKey = c.ttsApiKey?.toString().trim();
		if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

		const body: Record<string, unknown> = {
			model: c.ttsModel.toString(),
			input: text,
			response_format: format
		};
		const voice = c.ttsVoice?.toString().trim();
		if (voice) body.voice = voice;

		const refAudio = c.ttsRefAudio?.toString().trim();
		if (refAudio && refAudio.startsWith('data:')) {
			body.ref_audio = refAudio;
			body.x_vector_only_mode = true;
		}

		// Hard-cap the request so a hung preflight (no CORS on the TTS server) or a
		// dropped connection doesn't leave the speaker stuck in the loading state.
		const timeout = new AbortController();
		const timer = setTimeout(() => timeout.abort(), 30_000);
		const signal = opts.signal ? anySignal([opts.signal, timeout.signal]) : timeout.signal;

		let response: Response;
		try {
			response = await fetch(`${baseUrl}/v1/audio/speech`, {
				method: 'POST',
				headers,
				body: JSON.stringify(body),
				signal
			});
		} catch (err) {
			if (timeout.signal.aborted) {
				throw new Error(
					`TTS request timed out after 30s. Check the base URL (${baseUrl}) and that the server responds to CORS preflight.`
				);
			}
			throw err;
		} finally {
			clearTimeout(timer);
		}

		if (!response.ok) {
			const msg = await response.text().catch(() => '');
			throw new Error(`TTS request failed (${response.status}): ${msg || response.statusText}`);
		}

		return await response.blob();
	}
}

function anySignal(signals: AbortSignal[]): AbortSignal {
	if (
		typeof (AbortSignal as unknown as { any?: (s: AbortSignal[]) => AbortSignal }).any ===
		'function'
	) {
		return (AbortSignal as unknown as { any: (s: AbortSignal[]) => AbortSignal }).any(signals);
	}
	const controller = new AbortController();
	for (const s of signals) {
		if (s.aborted) {
			controller.abort((s as AbortSignal & { reason?: unknown }).reason);
			break;
		}
		s.addEventListener(
			'abort',
			() => controller.abort((s as AbortSignal & { reason?: unknown }).reason),
			{ once: true }
		);
	}
	return controller.signal;
}
