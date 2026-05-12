import { config } from '$lib/stores/settings.svelte';
import { selectedModelName } from '$lib/stores/models.svelte';
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
	/** Text before the cursor. Required. */
	prefix: string;
	/** Text after the cursor. Optional but strongly recommended — without it
	 *  FIM degrades to one-sided prediction. */
	suffix?: string;
	maxTokens?: number;
	stop?: string[];
	temperature?: number;
	signal?: AbortSignal;
}

export interface InlineCompletionResult {
	content: string;
}

export class CompletionService {
	/**
	 * Inline ghost-text completion via llama-server's `/infill` endpoint.
	 *
	 * Why /infill and not /completion: instruct models fed raw prefix text
	 * try to "respond" instead of continue, producing repetition,
	 * preambles, or apologetic disclaimers. `/infill` injects the model's
	 * own FIM (fill-in-middle) tokens — `<|fim_prefix|>` /
	 * `<|fim_suffix|>` / `<|fim_middle|>` for Qwen-Coder / DeepSeek-Coder,
	 * `[PREFIX]` / `[SUFFIX]` / `[MIDDLE]` for Codestral, etc. — which
	 * keeps the model on-distribution. llama.cpp auto-detects the right
	 * tokens from GGUF metadata.
	 *
	 * Sampling defaults follow the llama.vscode reference client:
	 *   top_k 40, top_p 0.99, samplers: top_k → top_p → infill,
	 *   temperature 0.2, repeat_penalty to dampen the loops the user was
	 *   seeing on instruct-only models.
	 */
	static async complete(opts: InlineCompletionOptions): Promise<InlineCompletionResult> {
		const c = config();
		const apiKey = c.apiKey?.toString().trim();

		const headers: Record<string, string> = {
			'Content-Type': 'application/json'
		};
		if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

		const body: Record<string, unknown> = {
			input_prefix: opts.prefix,
			input_suffix: opts.suffix ?? '',
			n_predict: opts.maxTokens ?? Number(c.inlineCompletionMaxTokens ?? 48),
			stream: false,
			cache_prompt: true,
			samplers: ['top_k', 'top_p', 'infill'],
			top_k: 40,
			top_p: 0.99,
			temperature: opts.temperature ?? 0.2,
			// Repetition guard. FIM-trained models rarely loop, but the
			// fallback path through instruct models (when the GGUF lacks
			// FIM tokens and llama-server stitches a regular prompt) often
			// does — repeat_penalty 1.1 is the inflection point that kills
			// loops without flattening valid repeats like list bullets.
			repeat_penalty: 1.1,
			repeat_last_n: 64,
			// Stop on a paragraph break to keep ghost text inline-shaped.
			// `\n\n` covers prose, `\n#` / `\n- ` / `\n* ` cover markdown
			// structural breaks the user is unlikely to want auto-filled.
			stop: opts.stop ?? ['\n\n', '\n#', '\n- ', '\n* '],
			// Bound generation latency so a slow model can't stall the
			// editor. Triggers only after the first newline so short
			// completions still come through.
			t_max_predict_ms: 2000
		};
		// Router mode requires a model name on every request.
		const model = selectedModelName();
		if (model) body.model = model;

		const response = await fetch(resolveApiUrl('/infill'), {
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

		const body: Record<string, unknown> = {
			messages,
			stream: true,
			temperature: opts.temperature ?? 0.4,
			max_tokens: opts.maxTokens ?? 1024
		};
		// llama-server router mode rejects the request with 400 "model name is
		// missing" when no model is named. Use the currently-selected chat model
		// so AI commands hit the same backend the user is chatting with.
		const model = selectedModelName();
		if (model) body.model = model;

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
