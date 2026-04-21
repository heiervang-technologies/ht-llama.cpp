import { config } from '$lib/stores/settings.svelte';

export interface TtsSynthesizeOptions {
	signal?: AbortSignal;
}

export interface TtsSynthesizeResult {
	blob: Blob;
	/** True when the request had to fall back to the bundled default reference clip. */
	usedDefaultRef: boolean;
}

const DEFAULT_REF_URL = '/tts-default-ref.mp3';
let cachedDefaultRef: string | null = null;

async function loadDefaultRefAudio(): Promise<string> {
	if (cachedDefaultRef) return cachedDefaultRef;
	const resp = await fetch(DEFAULT_REF_URL);
	if (!resp.ok) throw new Error(`Failed to load default ref audio (${resp.status})`);
	const blob = await resp.blob();
	cachedDefaultRef = await new Promise<string>((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(String(reader.result));
		reader.onerror = () => reject(reader.error ?? new Error('FileReader failed'));
		reader.readAsDataURL(blob);
	});
	return cachedDefaultRef;
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
 *                 When absent, falls back to the bundled default ref clip
 *                 so Base-task servers don't 400 out-of-the-box.)
 */
export class TtsService {
	static isConfigured(): boolean {
		const c = config();
		return Boolean(c.ttsBaseUrl?.toString().trim() && c.ttsModel?.toString().trim());
	}

	static async synthesize(
		text: string,
		opts: TtsSynthesizeOptions = {}
	): Promise<TtsSynthesizeResult> {
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
			response_format: format,
			// Qwen3-TTS Base task is the only one that works without a server-side
			// speaker registry (i.e. without the multiplexer). Default to it so the
			// client works against a raw vllm deploy.
			task_type: 'Base'
		};
		const voice = c.ttsVoice?.toString().trim();
		if (voice) body.voice = voice;

		const userRefAudio = c.ttsRefAudio?.toString().trim();
		let usedDefaultRef = false;
		if (userRefAudio && userRefAudio.startsWith('data:')) {
			body.ref_audio = userRefAudio;
		} else {
			body.ref_audio = await loadDefaultRefAudio();
			usedDefaultRef = true;
		}
		// Skip text conditioning on the reference clip: we rarely know the exact
		// transcript (especially for the bundled default) and the speaker embedding
		// alone is enough for Qwen3-TTS voice cloning.
		body.x_vector_only_mode = true;

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

		return { blob: await response.blob(), usedDefaultRef };
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
