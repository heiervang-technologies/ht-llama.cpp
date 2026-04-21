import { config } from '$lib/stores/settings.svelte';

export interface SttTranscribeOptions {
	signal?: AbortSignal;
	language?: string;
}

/**
 * Stateless client for OpenAI-compatible Speech-to-Text servers (Qwen3-ASR, Whisper).
 * Posts multipart form-data to `<baseUrl>/v1/audio/transcriptions` and returns
 * the transcribed text.
 *
 * Settings consumed:
 * - sttBaseUrl  (required; empty = disabled)
 * - sttApiKey   (optional bearer token)
 * - sttModel    (required)
 * - sttLanguage (optional ISO 639-1 code; omitted when blank)
 */
export class SttService {
	static isConfigured(): boolean {
		const c = config();
		return Boolean(c.sttBaseUrl?.toString().trim() && c.sttModel?.toString().trim());
	}

	static async transcribe(audio: Blob | File, opts: SttTranscribeOptions = {}): Promise<string> {
		const c = config();
		const baseUrl = c.sttBaseUrl?.toString().trim().replace(/\/+$/, '') ?? '';
		if (!baseUrl || !c.sttModel?.toString().trim()) {
			throw new Error('STT is not configured. Set a Base URL and Model in settings.');
		}

		const file =
			audio instanceof File
				? audio
				: new File([audio], 'recording.wav', { type: audio.type || 'audio/wav' });

		const form = new FormData();
		form.set('model', c.sttModel.toString());
		form.set('file', file, file.name);
		form.set('response_format', 'json');
		const language = opts.language ?? c.sttLanguage?.toString().trim();
		if (language) form.set('language', language);

		const headers: Record<string, string> = {};
		const apiKey = c.sttApiKey?.toString().trim();
		if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

		// Hard-cap the request: a hung STT server would otherwise leave the mic
		// button pinned in the "transcribing" spinner state indefinitely.
		const timeout = new AbortController();
		const timer = setTimeout(() => timeout.abort(), 60_000);
		const signal = opts.signal ? anySignal([opts.signal, timeout.signal]) : timeout.signal;

		let response: Response;
		try {
			response = await fetch(`${baseUrl}/v1/audio/transcriptions`, {
				method: 'POST',
				headers,
				body: form,
				signal
			});
		} catch (err) {
			if (timeout.signal.aborted) {
				throw new Error(
					`STT request timed out after 60s. Check the base URL (${baseUrl}) and that the server is responding.`
				);
			}
			// Same treatment as TtsService: fetch throws TypeError on connection
			// refused / DNS / CORS reject, and the browser-native message ("Load
			// failed" on WebKit, "Failed to fetch" on Chromium) doesn't name the
			// host. Surface a message the user can act on.
			if (err instanceof TypeError) {
				throw new Error(`Could not reach STT server at ${baseUrl}. Is it running and reachable?`);
			}
			throw err;
		} finally {
			clearTimeout(timer);
		}

		if (!response.ok) {
			const msg = await response.text().catch(() => '');
			throw new Error(`STT request failed (${response.status}): ${msg || response.statusText}`);
		}

		// OpenAI-compatible servers return {text: string} for response_format=json.
		// Some return plain text if the header/format differs — fall back gracefully.
		const contentType = response.headers.get('content-type') ?? '';
		if (contentType.includes('application/json')) {
			const data = (await response.json()) as { text?: string };
			return (data?.text ?? '').toString().trim();
		}
		return (await response.text()).trim();
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
