import { config } from '$lib/stores/settings.svelte';

export interface TtsSynthesizeOptions {
	signal?: AbortSignal;
}

/**
 * Stateless client for OpenAI-compatible TTS servers.
 * Posts {model, voice, input, response_format} to `<baseUrl>/v1/audio/speech`
 * and returns the response as a Blob.
 *
 * Settings consumed:
 * - ttsBaseUrl  (required; empty = disabled)
 * - ttsApiKey   (optional bearer token)
 * - ttsModel    (required)
 * - ttsVoice    (optional)
 * - ttsFormat   (default 'wav')
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

		const response = await fetch(`${baseUrl}/v1/audio/speech`, {
			method: 'POST',
			headers,
			body: JSON.stringify(body),
			signal: opts.signal
		});

		if (!response.ok) {
			const msg = await response.text().catch(() => '');
			throw new Error(`TTS request failed (${response.status}): ${msg || response.statusText}`);
		}

		return await response.blob();
	}
}
