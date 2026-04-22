import { config } from '$lib/stores/settings.svelte';

export interface TtsSynthesizeOptions {
	signal?: AbortSignal;
}

export interface TtsSynthesizeResult {
	blob: Blob;
	/** True when the request had to fall back to the bundled default reference clip. */
	usedDefaultRef: boolean;
}

/** Minimal shape of a voice returned by an OpenAI-compatible `/v1/audio/voices` endpoint. */
export interface TtsVoice {
	id: string;
	name?: string;
	language?: string;
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
		const userRefAudio = c.ttsRefAudio?.toString().trim();
		let usedDefaultRef = false;

		if (voice) {
			// User picked a named voice (e.g. from the multiplexer's voice
			// registry). Send only the name — adding ref_audio here makes the
			// server fall back to cloning and the voice name is ignored.
			body.voice = voice;
		} else if (userRefAudio && userRefAudio.startsWith('data:')) {
			// No named voice, but user uploaded a ref clip → voice cloning.
			body.ref_audio = userRefAudio;
			// Speaker-embedding-only mode: we don't know the reference transcript,
			// so text conditioning would corrupt the clone.
			body.x_vector_only_mode = true;
		} else {
			// Nothing configured — fall back to the bundled default ref clip so
			// Base-task servers don't 400 out-of-the-box.
			body.ref_audio = await loadDefaultRefAudio();
			body.x_vector_only_mode = true;
			usedDefaultRef = true;
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
			// fetch throws a TypeError on connection failures (refused, DNS, CORS
			// reject). WebKit surfaces this as "Load failed", Chromium as "Failed
			// to fetch" — neither names the host. Rewrite so the toast is useful.
			if (err instanceof TypeError) {
				throw new Error(`Could not reach TTS server at ${baseUrl}. Is it running and reachable?`);
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

	/**
	 * Fetch available voices from the configured TTS server.
	 *
	 * Hits `GET <baseUrl>/v1/audio/voices` (an OpenAI-style extension shared by
	 * vLLM's Qwen3-TTS multiplexer and a handful of other TTS servers). Parses
	 * either `{ data: [...] }` or a bare array.
	 *
	 * - Returns `[]` only on 404 (endpoint genuinely not implemented).
	 * - Throws with a useful message on network / CORS / DNS failure and on
	 *   any other non-ok status (401, 500, etc). Silently degrading to an
	 *   empty list would hide a real misconfiguration (wrong base URL, CORS
	 *   preflight rejected, etc).
	 */
	static async fetchVoices(signal?: AbortSignal): Promise<TtsVoice[]> {
		const c = config();
		const baseUrl = c.ttsBaseUrl?.toString().trim().replace(/\/+$/, '') ?? '';
		if (!baseUrl) return [];

		const headers: Record<string, string> = { Accept: 'application/json' };
		const apiKey = c.ttsApiKey?.toString().trim();
		if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

		let response: Response;
		try {
			response = await fetch(`${baseUrl}/v1/audio/voices`, { headers, signal });
		} catch (err) {
			// Aborts (e.g. because a newer fetch superseded this one) are not
			// errors — bubble as-is so the caller can distinguish.
			if ((err as { name?: string })?.name === 'AbortError') throw err;
			// fetch throws a TypeError for network / CORS / DNS failures. The
			// browser error message ("Load failed" on WebKit, "Failed to fetch"
			// on Chromium) doesn't name the host — rewrite to something useful.
			const base = `Could not reach TTS server at ${baseUrl}/v1/audio/voices`;
			const hint =
				err instanceof TypeError
					? ' (network unreachable, DNS failure, or CORS preflight rejected). Confirm the server is running and that its CORS config allows this app.'
					: '';
			throw new Error(base + hint);
		}

		if (response.status === 404) return [];
		if (!response.ok) {
			const body = await response.text().catch(() => '');
			const snippet = body.trim().slice(0, 400);
			throw new Error(
				`Voices request failed (${response.status} ${response.statusText})` +
					(snippet ? `: ${snippet}` : '')
			);
		}

		let payload: unknown;
		try {
			payload = await response.json();
		} catch {
			return [];
		}

		// Accept several response shapes. Observed in the wild:
		// - Bare array:         `[...]`
		// - OpenAI-style list:  `{ data: [...] }`
		// - Qwen3-TTS mux:      `{ voices: [...], uploaded_voices: [...] }`
		// - Voice name list:    `{ voices: ["alice", ...] }`
		const p = payload as {
			data?: unknown[];
			voices?: unknown[];
			uploaded_voices?: unknown[];
		} | null;

		const raw: unknown[] = Array.isArray(payload)
			? payload
			: Array.isArray(p?.data)
				? p.data
				: Array.isArray(p?.voices) || Array.isArray(p?.uploaded_voices)
					? [...(p?.voices ?? []), ...(p?.uploaded_voices ?? [])]
					: [];

		const seen = new Set<string>();
		return raw
			.map((entry): TtsVoice | null => {
				if (typeof entry === 'string') return { id: entry, name: entry };
				if (entry && typeof entry === 'object') {
					const e = entry as Record<string, unknown>;
					const id = (e.id ?? e.voice ?? e.name) as string | undefined;
					if (!id || typeof id !== 'string') return null;
					return {
						id,
						name: typeof e.name === 'string' ? e.name : id,
						language: typeof e.language === 'string' ? e.language : undefined
					};
				}
				return null;
			})
			.filter((v): v is TtsVoice => {
				if (!v) return false;
				if (seen.has(v.id)) return false;
				seen.add(v.id);
				return true;
			});
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
