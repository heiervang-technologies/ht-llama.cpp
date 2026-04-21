import { config } from '$lib/stores/settings.svelte';
import { resolveApiUrl } from '$lib/utils/backend-url';

/**
 * Minimal client for llama-server's native `/completion` endpoint.
 * Used by the inline ghost-text completions in the doc editor.
 * Separate from chat.service so it stays small and zero-state.
 *
 * Settings consumed:
 * - apiKey                    (optional, forwarded as Bearer)
 * - inlineCompletionMaxTokens (default 48)
 */

export interface InlineCompletionOptions {
	prompt: string;
	maxTokens?: number;
	stop?: string[];
	temperature?: number;
	signal?: AbortSignal;
}

export interface InlineCompletionResult {
	content: string;
}

export class CompletionService {
	static async complete(opts: InlineCompletionOptions): Promise<InlineCompletionResult> {
		const c = config();
		const apiKey = c.apiKey?.toString().trim();

		const headers: Record<string, string> = {
			'Content-Type': 'application/json'
		};
		if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

		const body = {
			prompt: opts.prompt,
			n_predict: opts.maxTokens ?? Number(c.inlineCompletionMaxTokens ?? 48),
			stream: false,
			cache_prompt: true,
			stop: opts.stop ?? ['\n\n', '\n#', '\n- ', '\n* '],
			temperature: opts.temperature ?? 0.2
		};

		const response = await fetch(resolveApiUrl('/completion'), {
			method: 'POST',
			headers,
			body: JSON.stringify(body),
			signal: opts.signal
		});

		if (!response.ok) {
			const msg = await response.text().catch(() => '');
			throw new Error(`Completion failed (${response.status}): ${msg || response.statusText}`);
		}

		const data = (await response.json()) as { content?: string };
		return { content: (data.content ?? '').trimEnd() };
	}

	/**
	 * Streaming variant used by AI commands. Calls the OpenAI-compatible
	 * `/v1/chat/completions` endpoint and yields content deltas.
	 */
	static async *chatStream(
		messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
		opts: { signal?: AbortSignal; maxTokens?: number; temperature?: number } = {}
	): AsyncGenerator<string> {
		const c = config();
		const apiKey = c.apiKey?.toString().trim();

		const headers: Record<string, string> = {
			'Content-Type': 'application/json'
		};
		if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

		const body = {
			messages,
			stream: true,
			temperature: opts.temperature ?? 0.4,
			max_tokens: opts.maxTokens ?? 1024
		};

		const response = await fetch(resolveApiUrl('/v1/chat/completions'), {
			method: 'POST',
			headers,
			body: JSON.stringify(body),
			signal: opts.signal
		});

		if (!response.ok || !response.body) {
			const msg = await response.text().catch(() => '');
			throw new Error(`Chat stream failed (${response.status}): ${msg || response.statusText}`);
		}

		const reader = response.body.getReader();
		const decoder = new TextDecoder();
		let buffer = '';

		try {
			while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n');
				buffer = lines.pop() ?? '';
				for (const line of lines) {
					const trimmed = line.trim();
					if (!trimmed.startsWith('data:')) continue;
					const payload = trimmed.slice(5).trim();
					if (!payload || payload === '[DONE]') continue;
					try {
						const json = JSON.parse(payload) as {
							choices?: Array<{ delta?: { content?: string } }>;
						};
						const delta = json.choices?.[0]?.delta?.content;
						if (delta) yield delta;
					} catch {
						/* ignore partial JSON */
					}
				}
			}
		} finally {
			reader.releaseLock();
		}
	}
}
