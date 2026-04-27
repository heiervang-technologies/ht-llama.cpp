/**
 * Built-in function-calling tools exposed to the model, alongside any
 * configured MCP servers.
 *
 * Each tool has a stable definition (OpenAI-style) and an executor that
 * the agentic loop can dispatch to. Tools are merged into the request's
 * `tools[]` array by the MCP store's `getToolDefinitionsForLLM` and
 * dispatched from its `executeTool` via the `dispatchBuiltin` bridge
 * below — so the call-site story stays single-channel.
 *
 * Current surface:
 *   - list_artifacts — browse the gallery
 *   - get_artifact   — read a specific revision
 *   - fork_artifact  — create an independent copy as a new gallery entry
 *   - send_keys      — type into a sandbox terminal (mode-gated)
 *   - generate_image — produce images via an OpenAI-compatible proxy
 *                      (e.g. ComfyUI at images.ht.local); result lands
 *                      in the gallery as an `image` artifact
 *   - edit_image     — restyle / inpaint an existing image; creates a
 *                      new artifact, never overwrites the source
 *   - generate_video — text/image/sound-driven video clips via the
 *                      same proxy; async with per-model poll budget
 *
 * All tools read and write through the same DatabaseService / gallery
 * store the UI uses, so anything the model does shows up in the
 * gallery immediately (no separate sync path).
 */

import { DatabaseService } from './database.service';
import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';
import { config } from '$lib/stores/settings.svelte';
import type { DatabaseArtifactKind } from '$lib/types/database';
import type { MCPToolCall, OpenAIToolDefinition, ToolExecutionResult } from '$lib/types/mcp';

/**
 * A built-in tool is defined in the OpenAI function-calling shape the
 * chat pipeline already understands, paired with an executor that
 * receives pre-parsed arguments. Errors thrown here are caught by the
 * agentic loop and surfaced to the model as the tool result — do NOT
 * try to render UI toasts from here; that would blur the boundary
 * between "tool errored" (model sees it) and "UI reacted" (user sees
 * it).
 */
export interface BuiltinTool {
	definition: OpenAIToolDefinition;
	execute(args: Record<string, unknown>, signal?: AbortSignal): Promise<ToolExecutionResult>;
	/**
	 * Optional gate — when present and returns false the tool is
	 * hidden from `getBuiltinToolDefinitions()` (so the model never
	 * sees it in its tool list) AND its executor rejects if somehow
	 * invoked (stale tool_call in conversation history after the
	 * user toggled it off, replay of an old request, etc.). Gates
	 * are evaluated fresh on every call so config changes take
	 * effect immediately.
	 */
	gate?: () => boolean;
	/** Human-readable name used when reporting a gated refusal. */
	gateLabel?: string;
}

const registry = new Map<string, BuiltinTool>();

function register(tool: BuiltinTool): void {
	registry.set(tool.definition.function.name, tool);
}

function isEnabled(tool: BuiltinTool): boolean {
	return tool.gate ? tool.gate() : true;
}

export function getBuiltinToolDefinitions(): OpenAIToolDefinition[] {
	return [...registry.values()].filter(isEnabled).map((t) => t.definition);
}

export function hasBuiltinTool(name: string): boolean {
	const t = registry.get(name);
	return Boolean(t && isEnabled(t));
}

/**
 * Bridge from the MCP store's `executeTool` path. Matches the return
 * shape so callers don't need a separate code path — a built-in tool is
 * "just another tool" as far as the agentic loop is concerned.
 */
export async function dispatchBuiltin(
	call: MCPToolCall,
	signal?: AbortSignal
): Promise<ToolExecutionResult> {
	const tool = registry.get(call.function.name);
	if (!tool) {
		return { content: `Error: unknown built-in tool ${call.function.name}`, isError: true };
	}
	if (!isEnabled(tool)) {
		const label = tool.gateLabel ?? call.function.name;
		return {
			content: `Error: ${label} is currently disabled in Settings. Ask the user to enable it before calling again.`,
			isError: true
		};
	}
	const args = parseArgs(call.function.arguments);
	try {
		return await tool.execute(args, signal);
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		return { content: `Error: ${message}`, isError: true };
	}
}

function parseArgs(raw: MCPToolCall['function']['arguments']): Record<string, unknown> {
	if (typeof raw === 'string') {
		if (!raw.trim()) return {};
		try {
			const parsed = JSON.parse(raw);
			return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : {};
		} catch {
			return {};
		}
	}
	return raw ?? {};
}

function ok(payload: unknown): ToolExecutionResult {
	return { content: JSON.stringify(payload), isError: false };
}

function err(message: string): ToolExecutionResult {
	return { content: JSON.stringify({ error: message }), isError: true };
}

const ARTIFACT_KINDS: DatabaseArtifactKind[] = [
	'html',
	'svg',
	'image',
	'code',
	'audio',
	'video',
	'pdf',
	'markdown'
];

register({
	definition: {
		type: 'function',
		function: {
			name: 'list_artifacts',
			description:
				'Browse artifacts saved in the gallery. Returns id, title, kind, and last-updated timestamp for each. Use this to discover an artifact id before calling get_artifact, fork_artifact, or emitting a SEARCH/REPLACE patch against one.',
			parameters: {
				type: 'object',
				properties: {
					kind: {
						type: 'string',
						enum: [...ARTIFACT_KINDS, 'all'],
						description:
							'Filter by modality (html, svg, image, code, audio, video, pdf, markdown) or "all" for everything. Defaults to "all".'
					},
					query: {
						type: 'string',
						description:
							'Case-insensitive substring match against title, summary, and tags. Optional.'
					},
					limit: {
						type: 'integer',
						minimum: 1,
						maximum: 100,
						description: 'Max artifacts returned (default 50).'
					}
				},
				required: []
			}
		}
	},
	async execute(args) {
		const kind = typeof args.kind === 'string' ? args.kind : 'all';
		const limit = Math.min(100, Math.max(1, Number(args.limit) || 50));
		const query = typeof args.query === 'string' ? args.query.trim().toLowerCase() : '';

		const all = await DatabaseService.listArtifacts();
		let filtered = kind === 'all' ? all : all.filter((a) => a.kind === kind);
		if (query) {
			filtered = filtered.filter(
				(a) =>
					a.title.toLowerCase().includes(query) ||
					a.summary?.toLowerCase().includes(query) ||
					a.tags.some((t) => t.toLowerCase().includes(query))
			);
		}

		const items = filtered.slice(0, limit).map((a) => ({
			id: a.id,
			title: a.title,
			kind: a.kind,
			tags: a.tags,
			summary: a.summary ?? null,
			updatedAt: new Date(a.updatedAt).toISOString(),
			sourceConversationId: a.sourceConversationId ?? null
		}));

		return ok({
			total: filtered.length,
			returned: items.length,
			items
		});
	}
});

register({
	definition: {
		type: 'function',
		function: {
			name: 'get_artifact',
			description:
				"Read an artifact's current revision (or a specific revision) so you can reason about its content or produce an exact SEARCH/REPLACE anchor. For binary kinds (image/audio/video/pdf) only metadata is returned; use the UI to view the payload.",
			parameters: {
				type: 'object',
				properties: {
					artifactId: {
						type: 'string',
						description: 'Id of the artifact. Use list_artifacts to discover it.'
					},
					revisionId: {
						type: 'string',
						description: 'Optional specific revision to read. Omit for the current pinned revision.'
					},
					maxChars: {
						type: 'integer',
						minimum: 512,
						maximum: 200000,
						description: 'Truncate returned content to this many characters (default 16000).'
					}
				},
				required: ['artifactId']
			}
		}
	},
	async execute(args) {
		const artifactId = String(args.artifactId ?? '');
		if (!artifactId) return err('artifactId is required');
		const maxChars = Math.min(200000, Math.max(512, Number(args.maxChars) || 16000));
		const revisionId =
			typeof args.revisionId === 'string' && args.revisionId ? args.revisionId : undefined;

		const artifact = await DatabaseService.getArtifact(artifactId);
		if (!artifact) return err(`artifact ${artifactId} not found`);
		const revs = await DatabaseService.listArtifactRevisions(artifactId);
		const rev = revisionId
			? revs.find((r) => r.id === revisionId)
			: (revs.find((r) => r.id === artifact.currentRevisionId) ?? revs.at(-1));
		if (!rev) return err('no revision available');

		const raw = rev.text;
		const isBinary = rev.blob && !raw;
		const content = raw ? raw.slice(0, maxChars) : null;
		const truncated = raw ? raw.length > maxChars : false;

		return ok({
			id: artifact.id,
			title: artifact.title,
			kind: artifact.kind,
			revisionId: rev.id,
			revisionNumber: rev.revisionNumber,
			totalRevisions: revs.length,
			currentRevisionId: artifact.currentRevisionId,
			mimeType: rev.mimeType,
			reason: rev.reason,
			updatedAt: new Date(artifact.updatedAt).toISOString(),
			content,
			truncated,
			fullCharCount: raw ? raw.length : null,
			binaryByteSize: isBinary ? (rev.blob?.size ?? null) : null,
			note: isBinary
				? 'Binary payload; only metadata returned. Use the gallery UI to view.'
				: truncated
					? `Content truncated to ${maxChars} chars; ${raw!.length - maxChars} more remain. Request a larger maxChars to see the tail.`
					: null
		});
	}
});

register({
	definition: {
		type: 'function',
		function: {
			name: 'send_keys',
			description:
				"Type into a sandbox terminal's shared PTY. The user sees everything you type appear in their xterm in real time; use this for pair-debugging, running commands on the user's behalf, or reacting to program output. The terminal must already exist and have been opened at least once (so a bash session is live). Ends with a newline automatically if `auto_enter` is true.",
			parameters: {
				type: 'object',
				properties: {
					terminalId: {
						type: 'string',
						description: 'The id of the terminal (from list_terminals or the URL the user shared).'
					},
					text: {
						type: 'string',
						description:
							'Exact characters to type. Include control chars verbatim — e.g. "\\u0003" for Ctrl+C. Mutually exclusive with `base64`.'
					},
					auto_enter: {
						type: 'boolean',
						description:
							'If true, append a newline to `text` so the shell executes it as a single command. Default false — use false when injecting partial input or control sequences.'
					}
				},
				required: ['terminalId']
			}
		}
	},
	async execute(args) {
		const terminalId = String(args.terminalId ?? '');
		if (!terminalId) return err('terminalId is required');
		const text = typeof args.text === 'string' ? args.text : '';
		if (!text) return err('text is required (non-empty)');
		const autoEnter = Boolean(args.auto_enter);

		// Per-terminal mode gate. Default is `solo` so a freshly
		// created terminal is opaque to the model until the user
		// explicitly flips it to `shared` or `review`. Imported
		// lazily to avoid pulling Svelte reactivity into the
		// builtin-tools module graph.
		const { terminalModes } = await import('$lib/stores/terminal-modes.svelte');
		const mode = terminalModes.snapshot(terminalId);
		if (mode === 'solo') {
			return err(
				`Terminal ${terminalId} is in "solo" mode; send_keys is blocked. Ask the user to switch it to "shared" (type live) or "review" (user approves each keystroke) mode first.`
			);
		}
		if (mode === 'review') {
			// Park the proposal for user approval. Return with a
			// structured "queued" response the model can reason
			// about without blocking — approval happens out-of-band.
			const { terminalProposals } = await import('$lib/stores/terminal-proposals.svelte');
			const proposal = terminalProposals.propose(terminalId, { text, autoEnter });
			return ok({
				queued: true,
				proposalId: proposal.id,
				mode,
				note: 'User must approve this keystroke via the Review panel before it lands in the PTY. Proceed with other reasoning; a later tool call or follow-up can check status.'
			});
		}

		// Reuse the termd service's URL resolution so `send_keys`
		// works identically from the Tauri sidecar or a manually-
		// configured endpoint. Imported lazily to avoid a circular
		// dependency with `services/termd.service.ts` at module load.
		const { resolveTermdUrl } = await import('./termd.service');
		const base = resolveTermdUrl();
		if (!base) return err('terminals are not configured (no ht-termd URL)');

		const res = await fetch(
			`${base.replace(/\/+$/, '')}/v1/terminals/${encodeURIComponent(terminalId)}/input`,
			{
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ text, auto_enter: autoEnter })
			}
		);
		if (!res.ok) {
			let detail = '';
			try {
				detail = ((await res.json()) as { error?: string }).error ?? '';
			} catch {
				/* ignore non-JSON error body */
			}
			return err(detail || `HTTP ${res.status}`);
		}
		return ok({ sent: text.length, mode });
	}
});

register({
	definition: {
		type: 'function',
		function: {
			name: 'fork_artifact',
			description:
				'Create an independent copy of an artifact (current revision by default) as a new gallery entry. The fork starts its own revision chain; use this when you want to iterate in a new direction without touching the original. The original is unchanged; the new artifact gets a "forked" tag and metadata pointers back to the source.',
			parameters: {
				type: 'object',
				properties: {
					artifactId: {
						type: 'string',
						description: 'Id of the source artifact.'
					},
					revisionId: {
						type: 'string',
						description:
							'Optional specific revision to fork from. Omit to fork the current pinned revision.'
					},
					newTitle: {
						type: 'string',
						description: 'Optional title for the fork. Defaults to "<original title> (fork)".'
					}
				},
				required: ['artifactId']
			}
		}
	},
	async execute(args) {
		const artifactId = String(args.artifactId ?? '');
		if (!artifactId) return err('artifactId is required');
		const revisionId =
			typeof args.revisionId === 'string' && args.revisionId ? args.revisionId : undefined;
		const newTitle =
			typeof args.newTitle === 'string' && args.newTitle.trim() ? args.newTitle.trim() : undefined;

		const source = await DatabaseService.getArtifact(artifactId);
		if (!source) return err(`artifact ${artifactId} not found`);
		const revs = await DatabaseService.listArtifactRevisions(artifactId);
		const rev = revisionId
			? revs.find((r) => r.id === revisionId)
			: (revs.find((r) => r.id === source.currentRevisionId) ?? revs.at(-1));
		if (!rev) return err('source revision unavailable');

		const tagSet = new Set([...source.tags, 'forked']);
		const { artifact: forked, revision: forkedRev } = await DatabaseService.createArtifact(
			{
				title: newTitle ?? `${source.title} (fork)`,
				kind: source.kind,
				tags: [...tagSet],
				summary: source.summary
			},
			{
				reason: 'fork',
				contentHash: rev.contentHash,
				mimeType: rev.mimeType,
				text: rev.text,
				blob: rev.blob,
				metadata: {
					...(rev.metadata ?? {}),
					forkedFromArtifactId: source.id,
					forkedFromRevisionId: rev.id
				}
			}
		);
		// Reload the gallery so the fork is visible without a page refresh.
		await artifactGalleryStore.load();

		return ok({
			id: forked.id,
			title: forked.title,
			kind: forked.kind,
			revisionId: forkedRev.id,
			revisionNumber: forkedRev.revisionNumber,
			sourceArtifactId: source.id,
			sourceRevisionId: rev.id
		});
	}
});

// ----- generate_image --------------------------------------------------
//
// The images proxy is an OpenAI-compat `/v1/images/generations` endpoint
// backed by ComfyUI. We always request `b64_json` so we don't depend on
// the client being able to fetch a ComfyUI /view URL (the proxy host
// might be LAN-only). Each returned image gets persisted as an
// `image` artifact so it's visible in the gallery and the chat render
// pipeline without extra plumbing — satisfies the "every image the
// model sees must be visible in the UI" contract.

/**
 * Resolve the images proxy base URL with the same fallback semantics as
 * the rest of the service layer: explicit config wins, then a Tauri
 * bundle-time default (currently not wired — reserved for future APK
 * builds that want to preconfigure the cluster proxy).
 */
function resolveImagesBaseUrl(): string {
	const cfg = String(config().imagesBaseUrl ?? '').trim();
	if (cfg) return cfg.replace(/\/+$/, '');
	if (typeof window !== 'undefined') {
		const fallback = (window as unknown as { __HT_DEFAULT_IMAGES_URL__?: string })
			.__HT_DEFAULT_IMAGES_URL__;
		if (typeof fallback === 'string' && fallback.trim()) {
			return fallback.trim().replace(/\/+$/, '');
		}
	}
	return '';
}

function resolveImagesApiKey(): string {
	const cfg = String(config().imagesApiKey ?? '').trim();
	if (cfg) return cfg;
	if (typeof window !== 'undefined') {
		const fallback = (window as unknown as { __HT_DEFAULT_IMAGES_KEY__?: string })
			.__HT_DEFAULT_IMAGES_KEY__;
		if (typeof fallback === 'string' && fallback.trim()) return fallback.trim();
	}
	return '';
}

/**
 * Strip a `data:image/...;base64,` prefix if present so the proxy
 * receives only the raw base64 payload. Returning unchanged input is
 * fine when the caller already passed clean base64.
 */
function stripDataUrlPrefix(input: string): string {
	const m = /^data:[a-z0-9.+/-]+;base64,(.+)$/i.exec(input);
	return m ? m[1] : input;
}

/**
 * Decode a base64 string into a Blob with the given MIME type. Splits
 * into 1 MB chunks so very large payloads don't blow the JS engine's
 * single-allocation limit — ComfyUI can return multi-megabyte PNGs.
 */
function base64ToBlob(base64: string, mimeType: string): Blob {
	const byteChars = atob(base64);
	const chunkSize = 1 << 20;
	const byteArrays: Uint8Array[] = [];
	for (let offset = 0; offset < byteChars.length; offset += chunkSize) {
		const slice = byteChars.slice(offset, offset + chunkSize);
		const bytes = new Uint8Array(slice.length);
		for (let i = 0; i < slice.length; i++) bytes[i] = slice.charCodeAt(i);
		byteArrays.push(bytes);
	}
	return new Blob(byteArrays, { type: mimeType });
}

/**
 * Core image-generation call, shared by the `generate_image` built-in
 * tool (LLM-driven) and the `/image` slash command (user-driven). Each
 * caller passes its own `source` tag so the resulting artifact records
 * *who* triggered it, and the gallery can split LLM-generated from
 * direct-invoke output downstream.
 *
 * Keeping this one function is the point — both paths POST the same
 * body to the same endpoint, persist into the same gallery, and need
 * the same failure modes. Divergence here would bite us the first time
 * we change the proxy contract.
 */
export interface RunImageGenerationOptions {
	source: 'generate_image' | 'direct' | 'playground';
	prompt: string;
	model?: string;
	size?: string;
	n?: number;
	/** Negative prompt — proxies that don't support it ignore the key. */
	negativePrompt?: string;
	/** Seed for reproducible runs. Omit (or -1) for random. */
	seed?: number;
	signal?: AbortSignal;
}

export interface RunImageGenerationResult {
	model: string;
	size: string | null;
	prompt: string;
	images: Array<{
		artifactId: string;
		revisionId: string;
		title: string;
		mimeType: string;
	}>;
}

export async function runImageGeneration(
	opts: RunImageGenerationOptions
): Promise<RunImageGenerationResult> {
	const prompt = opts.prompt.trim();
	if (!prompt) throw new Error('prompt is required');
	if (!config().imageGenEnabled) {
		throw new Error('Image generation is currently disabled in Settings → Images.');
	}

	const model = opts.model?.trim() || 'z-image-turbo';
	const size = opts.size?.trim() || undefined;
	const n = Math.min(4, Math.max(1, opts.n ?? 1));

	const base = resolveImagesBaseUrl();
	if (!base) {
		throw new Error(
			'Image generation is not configured. Set Settings → Images → Base URL (e.g. http://images.ht.local).'
		);
	}

	const apiKey = resolveImagesApiKey();
	const headers: Record<string, string> = {
		'Content-Type': 'application/json',
		Authorization: `Bearer ${apiKey || 'no-auth'}`
	};

	const body: Record<string, unknown> = {
		prompt,
		model,
		n,
		response_format: 'b64_json'
	};
	if (size) body.size = size;
	const trimmedNeg = opts.negativePrompt?.trim();
	if (trimmedNeg) {
		// Same alias hedge as the prompt — the proxy adapter may key on
		// any of these names depending on the workflow.
		body.negative_prompt = trimmedNeg;
		body.negativePrompt = trimmedNeg;
	}
	if (typeof opts.seed === 'number' && opts.seed >= 0) {
		body.seed = opts.seed;
	}

	let res: Response;
	try {
		res = await fetch(`${base}/v1/images/generations`, {
			method: 'POST',
			headers,
			body: JSON.stringify(body),
			signal: opts.signal
		});
	} catch (fetchErr) {
		const message = fetchErr instanceof Error ? fetchErr.message : String(fetchErr);
		throw new Error(`Network error reaching ${base}: ${message}`);
	}

	if (!res.ok) {
		let detail = '';
		try {
			detail = JSON.stringify(await res.json());
		} catch {
			try {
				detail = await res.text();
			} catch {
				/* ignore */
			}
		}
		throw new Error(`Images proxy HTTP ${res.status}: ${detail || 'no body'}`);
	}

	type GenResp = {
		created?: number;
		data?: Array<{ b64_json?: string; url?: string; revised_prompt?: string }>;
	};
	const payload = (await res.json()) as GenResp;
	const items = payload.data ?? [];
	if (items.length === 0) {
		throw new Error('Images proxy returned no images.');
	}

	const saved: RunImageGenerationResult['images'] = [];
	for (let i = 0; i < items.length; i++) {
		const item = items[i];
		let blob: Blob | undefined;
		const mimeType = 'image/png';

		if (typeof item.b64_json === 'string' && item.b64_json.length > 0) {
			blob = base64ToBlob(item.b64_json, mimeType);
		} else if (typeof item.url === 'string' && item.url.length > 0) {
			try {
				const imgRes = await fetch(item.url, { signal: opts.signal });
				if (imgRes.ok) blob = await imgRes.blob();
			} catch {
				/* ignore */
			}
		}
		if (!blob) continue;

		const title = `Generated · ${prompt.slice(0, 80)}${prompt.length > 80 ? '…' : ''}${
			items.length > 1 ? ` (${i + 1}/${items.length})` : ''
		}`;
		const artifact = await artifactGalleryStore.saveManual({
			kind: 'image' as DatabaseArtifactKind,
			title,
			mimeType,
			blob,
			tags: [
				'generated',
				model,
				...(opts.source === 'direct'
					? ['direct']
					: opts.source === 'playground'
						? ['playground']
						: [])
			],
			metadata: {
				source: opts.source,
				model,
				prompt,
				size: size ?? null,
				revisedPrompt: item.revised_prompt ?? null,
				generatedAt: new Date().toISOString()
			}
		});
		saved.push({
			artifactId: artifact.id,
			revisionId: artifact.currentRevisionId,
			title: artifact.title,
			mimeType
		});
	}

	if (saved.length === 0) {
		throw new Error('Images proxy returned rows but none had usable b64_json or a fetchable url.');
	}

	return { model, size: size ?? null, prompt, images: saved };
}

register({
	// Gated behind a user toggle so the model only sees / invokes
	// generate_image when the user has explicitly opted in. Same
	// mental model as ChatGPT's "Create image" button: media
	// generation is a capability the user turns on, not a default.
	gate: () => Boolean(config().imageGenEnabled),
	gateLabel: 'Image generation',
	definition: {
		type: 'function',
		function: {
			name: 'generate_image',
			description:
				'Generate images from a text prompt via the OpenAI-compatible images proxy (ComfyUI-backed). Each returned image is saved as an `image` artifact in the gallery automatically, so the user sees it inline. Use this when the user explicitly asks for an image, a mockup, a diagram mock, or a visual reference. Two reliable models: `z-image-turbo` (general-purpose, p50 ~52s, p95 ~68s) is the default; `newbie-image` (anime / manga style, ~22s — the fastest option) is the right pick when the user asks for anime. Two experimental models: `qwen-image` works but takes ~10 minutes per image (VRAM-paging on a 24GB GPU) — only use when the user explicitly asks, and warn them about the wait; `flux2-klein` is currently broken (text encoder needs >24GB VRAM, OOMs during load). Before any call, warn the user about the expected wait: ~60s for z-image-turbo, ~25s for newbie-image, ~10 minutes for qwen-image.',
			parameters: {
				type: 'object',
				properties: {
					prompt: {
						type: 'string',
						description:
							'The image description. Be specific — composition, subject, style, colours. The proxy passes this straight to the ComfyUI workflow.'
					},
					model: {
						type: 'string',
						enum: ['z-image-turbo', 'newbie-image', 'qwen-image', 'flux2-klein'],
						description:
							'Model id on the proxy. Defaults to `z-image-turbo` (general-purpose, ~60s). `newbie-image` (~22s) is reliable for anime / manga style. `qwen-image` works but ~10 minutes per image. `flux2-klein` is currently broken (OOMs loading a 33GB text encoder on a 24GB GPU). Only override the default if the user asked for a specific model by name.'
					},
					size: {
						type: 'string',
						description:
							'OpenAI-style size string, e.g. `1024x1024`, `1024x1536`, `1536x1024`. Passed through; unsupported sizes may be coerced by the workflow.'
					},
					n: {
						type: 'integer',
						minimum: 1,
						maximum: 4,
						description: 'Number of images to generate (default 1, max 4).'
					}
				},
				required: ['prompt']
			}
		}
	},
	async execute(args, signal) {
		try {
			const result = await runImageGeneration({
				source: 'generate_image',
				prompt: String(args.prompt ?? ''),
				model: typeof args.model === 'string' ? args.model : undefined,
				size: typeof args.size === 'string' ? args.size : undefined,
				n: Number(args.n) || undefined,
				signal
			});
			return ok({
				...result,
				note: "Each image is in the user's gallery; reference by artifactId via get_artifact in follow-ups."
			});
		} catch (e) {
			return err(e instanceof Error ? e.message : String(e));
		}
	}
});

// ----- edit_image ------------------------------------------------------
//
// POST /v1/images/edits against the same OpenAI-compatible proxy.
// Single reliable model on the backend right now (`qwen-image-edit`,
// ~2.5 min at 1024x1024, qwen 20GB weights). Result lands in the
// gallery as a new `image` artifact — we never overwrite the source;
// an "edit" is a new asset with `sourceArtifactId` metadata so forks
// are reproducible. Gated on the SAME `imageGenEnabled` toggle as
// generate_image, since from the user's perspective both are
// "produce an image via the cluster GPU" and a single switch is
// easier to reason about than two.

register({
	gate: () => Boolean(config().imageGenEnabled),
	gateLabel: 'Image generation',
	definition: {
		type: 'function',
		function: {
			name: 'edit_image',
			description:
				"Edit an existing image according to a natural-language prompt (inpaint, restyle, change subject, etc.) via the OpenAI-compatible images proxy. The reference image is sent as base64 or a data URL; the returned image is saved as a NEW `image` artifact in the gallery — the source is never overwritten. One reliable model today: `qwen-image-edit` (~2.5 min at 1024x1024). Warn the user about the ~2-3 minute wait before calling. To edit an existing gallery artifact, call `get_artifact` first to fetch its data URL, then pass it here. The revised artifact's metadata includes `sourceArtifactId` so follow-up forks stay traceable.",
			parameters: {
				type: 'object',
				properties: {
					prompt: {
						type: 'string',
						description:
							'The edit instruction. Be concrete — what to change, keep, or add. The proxy passes this verbatim to the ComfyUI workflow.'
					},
					image: {
						type: 'string',
						description:
							'The source image. Either a raw base64 string OR a `data:image/...;base64,...` data URL. Required. Use `get_artifact` to fetch an existing gallery artifact as a data URL.'
					},
					model: {
						type: 'string',
						enum: ['qwen-image-edit'],
						description:
							'Edit model id on the proxy. Currently only `qwen-image-edit` is wired; the enum is kept for future expansion.'
					},
					size: {
						type: 'string',
						description:
							'Output size, `WIDTHxHEIGHT` (e.g. `1024x1024`). Default `1024x1024`. Sent through to the workflow; unsupported sizes may be coerced.'
					},
					n: {
						type: 'integer',
						minimum: 1,
						maximum: 4,
						description:
							'Number of edited variants (default 1, max 4). Each variant becomes its own artifact.'
					},
					sourceArtifactId: {
						type: 'string',
						description:
							'Optional: the gallery artifact id the source image came from. When supplied, the new artifact metadata records it so the edit chain stays traceable; purely informational.'
					}
				},
				required: ['prompt', 'image']
			}
		}
	},
	async execute(args, signal) {
		try {
			const result = await runImageEdit({
				source: 'edit_image',
				prompt: String(args.prompt ?? ''),
				image: String(args.image ?? ''),
				model: typeof args.model === 'string' ? args.model : undefined,
				size: typeof args.size === 'string' ? args.size : undefined,
				n: Number(args.n) || undefined,
				sourceArtifactId:
					typeof args.sourceArtifactId === 'string' ? args.sourceArtifactId : undefined,
				signal
			});
			return ok({
				...result,
				note: 'Each edited image is a NEW artifact in the gallery; the original is untouched. Reference edits by artifactId in follow-ups.'
			});
		} catch (e) {
			return err(e instanceof Error ? e.message : String(e));
		}
	}
});

/**
 * Symmetric to runImageGeneration, for the /v1/images/edits proxy.
 * Both the `edit_image` tool and the playground page call this.
 */
export interface RunImageEditOptions {
	source: 'edit_image' | 'direct' | 'playground';
	prompt: string;
	/** base64 string OR `data:image/...;base64,...` data URL */
	image: string;
	model?: string;
	size?: string;
	n?: number;
	negativePrompt?: string;
	seed?: number;
	sourceArtifactId?: string | null;
	signal?: AbortSignal;
}

export interface RunImageEditResult {
	model: string;
	size: string;
	prompt: string;
	sourceArtifactId: string | null;
	images: Array<{
		artifactId: string;
		revisionId: string;
		title: string;
		mimeType: string;
	}>;
}

export async function runImageEdit(opts: RunImageEditOptions): Promise<RunImageEditResult> {
	const prompt = opts.prompt.trim();
	if (!prompt) throw new Error('prompt is required');
	const rawImage = opts.image.trim();
	if (!rawImage) throw new Error('image is required (base64 string or data URL)');
	if (!config().imageGenEnabled) {
		throw new Error('Image generation is currently disabled in Settings → Images.');
	}

	const model = opts.model?.trim() || 'qwen-image-edit';
	const size = opts.size?.trim() || '1024x1024';
	const n = Math.min(4, Math.max(1, opts.n ?? 1));
	const sourceArtifactId = opts.sourceArtifactId?.trim() || null;

	const base = resolveImagesBaseUrl();
	if (!base) {
		throw new Error(
			'Image editing is not configured. Set Settings → Images → Base URL (e.g. http://images.ht.local).'
		);
	}

	const apiKey = resolveImagesApiKey();
	// The proxy on this fork accepts JSON with `image` as a base64
	// string — multipart/form-data is the OpenAI canonical shape but
	// snoop's images-proxy doesn't speak it (request dies as a generic
	// WebKit "Load failed"). The previous JSON path was almost right;
	// the bug was that we sent the full `data:image/...;base64,...`
	// data URL when the user picked a source from gallery / file
	// upload. The proxy then either preserved the `data:` prefix in
	// the file it wrote, or naively base64-decoded the whole string,
	// either way producing bytes PIL couldn't parse. Strip the prefix
	// here so the proxy always receives pure base64.
	const cleanImage = stripDataUrlPrefix(rawImage);
	const headers: Record<string, string> = {
		'Content-Type': 'application/json',
		Authorization: `Bearer ${apiKey || 'no-auth'}`
	};
	// Hedge: include the prompt under a few common aliases. Saw a case
	// where the source image arrived intact but the proxy ignored
	// `prompt` and qwen-image-edit returned the source unchanged. The
	// canonical OpenAI key is `prompt`; the others cost a few bytes
	// and let the proxy pick whichever its workflow adapter expects.
	const body: Record<string, unknown> = {
		prompt,
		instruction: prompt,
		text: prompt,
		model,
		n,
		size,
		image: cleanImage,
		response_format: 'b64_json'
	};
	const trimmedNeg = opts.negativePrompt?.trim();
	if (trimmedNeg) {
		body.negative_prompt = trimmedNeg;
		body.negativePrompt = trimmedNeg;
	}
	if (typeof opts.seed === 'number' && opts.seed >= 0) {
		body.seed = opts.seed;
	}

	let res: Response;
	try {
		res = await fetch(`${base}/v1/images/edits`, {
			method: 'POST',
			headers,
			body: JSON.stringify(body),
			signal: opts.signal
		});
	} catch (fetchErr) {
		const message = fetchErr instanceof Error ? fetchErr.message : String(fetchErr);
		throw new Error(`Network error reaching ${base}: ${message}`);
	}

	if (!res.ok) {
		let detail = '';
		try {
			detail = JSON.stringify(await res.json());
		} catch {
			try {
				detail = await res.text();
			} catch {
				/* ignore */
			}
		}
		throw new Error(`Images-edit proxy HTTP ${res.status}: ${detail || 'no body'}`);
	}

	type EditResp = {
		created?: number;
		data?: Array<{ b64_json?: string; url?: string; revised_prompt?: string }>;
	};
	const payload = (await res.json()) as EditResp;
	const items = payload.data ?? [];
	if (items.length === 0) throw new Error('Images-edit proxy returned no images.');

	const saved: RunImageEditResult['images'] = [];
	for (let i = 0; i < items.length; i++) {
		const item = items[i];
		let blob: Blob | undefined;
		const mimeType = 'image/png';

		if (typeof item.b64_json === 'string' && item.b64_json.length > 0) {
			blob = base64ToBlob(item.b64_json, mimeType);
		} else if (typeof item.url === 'string' && item.url.length > 0) {
			try {
				const imgRes = await fetch(item.url, { signal: opts.signal });
				if (imgRes.ok) blob = await imgRes.blob();
			} catch {
				/* ignore */
			}
		}
		if (!blob) continue;

		const title = `Edited · ${prompt.slice(0, 80)}${prompt.length > 80 ? '…' : ''}${
			items.length > 1 ? ` (${i + 1}/${items.length})` : ''
		}`;
		const artifact = await artifactGalleryStore.saveManual({
			kind: 'image' as DatabaseArtifactKind,
			title,
			mimeType,
			blob,
			tags: [
				'generated',
				'edited',
				model,
				...(opts.source === 'playground'
					? ['playground']
					: opts.source === 'direct'
						? ['direct']
						: [])
			],
			metadata: {
				source: opts.source,
				model,
				prompt,
				size,
				sourceArtifactId,
				revisedPrompt: item.revised_prompt ?? null,
				generatedAt: new Date().toISOString()
			}
		});
		saved.push({
			artifactId: artifact.id,
			revisionId: artifact.currentRevisionId,
			title: artifact.title,
			mimeType
		});
	}

	if (saved.length === 0) {
		throw new Error(
			'Images-edit proxy returned rows but none had usable b64_json or a fetchable url.'
		);
	}

	return { model, size, prompt, sourceArtifactId, images: saved };
}

// ----- generate_video --------------------------------------------------
//
// Async: POST /v1/videos returns 202 + {id, status:"queued"}; we poll
// GET /v1/videos/{id} until status == "completed" (or "failed"), then
// GET /v1/videos/{id}/content for the mp4 bytes. The typical wait is
// ~60 s for a 17-frame short and ~3 min for 81 frames at 832x480, so
// the agentic loop has to tolerate a multi-minute tool call. We cap
// the poll loop at POLL_BUDGET_MS below — if the job isn't done by
// then we return a partial result with the job id so the model can
// tell the user where to check. Video lands in the gallery as a
// `video` artifact the same way images do.

const VIDEO_POLL_INTERVAL_MS = 2000;

/**
 * Poll budgets are per-model since runtimes vary widely. Upper-bound
 * with headroom so a user at the long end of the distribution still
 * completes instead of the tool timing out prematurely.
 *
 *   wan22-i2v:       ~60 s (17f) → ~3 min (81f)  → cap 6 min
 *   wan22-i2v-hq:    ~5× slower than wan22-i2v    → cap 15 min
 *   ltx-2.3:         ~4 min (49f 960x544)         → cap 10 min
 *   wan22-s2v:       ~3.5 min (49f 512x288)       → cap 8 min
 */
const VIDEO_POLL_BUDGET_MS: Record<string, number> = {
	'wan22-i2v': 6 * 60 * 1000,
	'wan22-i2v-hq': 15 * 60 * 1000,
	'ltx-2.3': 10 * 60 * 1000,
	'wan22-s2v': 8 * 60 * 1000
};
const DEFAULT_VIDEO_POLL_BUDGET_MS = 10 * 60 * 1000;

type VideoJobStatus = {
	id: string;
	model?: string;
	status: 'queued' | 'in_progress' | 'completed' | 'failed' | string;
	error?: string;
	content_url?: string;
	created?: number;
};

async function pollVideoJob(
	base: string,
	id: string,
	headers: Record<string, string>,
	budgetMs: number,
	signal?: AbortSignal
): Promise<VideoJobStatus> {
	const deadline = Date.now() + budgetMs;
	while (Date.now() < deadline) {
		if (signal?.aborted) throw new Error('aborted');
		const res = await fetch(`${base}/v1/videos/${encodeURIComponent(id)}`, {
			method: 'GET',
			headers,
			signal
		});
		if (!res.ok) {
			let detail = '';
			try {
				detail = await res.text();
			} catch {
				/* ignore */
			}
			throw new Error(`poll HTTP ${res.status}: ${detail || 'no body'}`);
		}
		const status = (await res.json()) as VideoJobStatus;
		if (status.status === 'completed' || status.status === 'failed') return status;
		await new Promise((r) => setTimeout(r, VIDEO_POLL_INTERVAL_MS));
	}
	throw new Error(
		`video job ${id} still in progress after ${budgetMs / 1000}s — check the gallery later`
	);
}

/**
 * Shared video-generation path. The `generate_video` tool and the
 * /images playground "Video" mode call this — divergence here would
 * mean two ways to talk to the same proxy, with predictable drift.
 *
 * Async contract:
 *   POST {base}/v1/videos             → 200/202 + { id, status }
 *   GET  {base}/v1/videos/{id}        (poll until completed | failed)
 *   GET  {base}/v1/videos/{id}/content → raw mp4 bytes
 *
 * On success the mp4 is saved as a `video` artifact in the gallery
 * (so the gallery has a single ingest path for media output) and a
 * compact reference is returned. The full Blob is NOT included in
 * the result so LLM tool callers don't shovel megabytes back into
 * context.
 */
export interface RunVideoGenerationOptions {
	source: 'generate_video' | 'playground';
	prompt: string;
	model?: string;
	/** data:image/...;base64,... — REQUIRED for every current model. */
	image: string;
	/** data:audio/...;base64,... — required for wan22-s2v, ignored otherwise. */
	audio?: string;
	size?: string;
	frames?: number;
	signal?: AbortSignal;
}

export interface RunVideoGenerationResult {
	model: string;
	size: string;
	frames: number;
	prompt: string;
	jobId: string;
	video: {
		artifactId: string;
		revisionId: string;
		title: string;
		mimeType: string;
		bytes: number;
	};
}

export async function runVideoGeneration(
	opts: RunVideoGenerationOptions
): Promise<RunVideoGenerationResult> {
	const prompt = opts.prompt.trim();
	if (!prompt) throw new Error('prompt is required');
	if (!config().videoGenEnabled) {
		throw new Error('Video generation is currently disabled in Settings → Images.');
	}

	const model = opts.model?.trim() || 'wan22-i2v';
	const defaultSize =
		model === 'ltx-2.3' ? '960x544' : model === 'wan22-s2v' ? '512x288' : '832x480';
	const size = opts.size?.trim() || defaultSize;
	const frames = Math.min(121, Math.max(1, opts.frames ?? 17));
	const image = opts.image.trim();
	if (!image) {
		throw new Error(
			`${model} requires a reference image data URL (all current video models are i2v / s2v).`
		);
	}
	const audio = opts.audio?.trim() || undefined;
	if (model === 'wan22-s2v' && !audio) {
		throw new Error('wan22-s2v is sound-to-video; pass an audio data URL (wav/mp3/ogg/flac).');
	}

	const base = resolveImagesBaseUrl();
	if (!base) {
		throw new Error(
			'Video generation is not configured. Set Settings → Images → Base URL (shared with image generation).'
		);
	}

	const apiKey = resolveImagesApiKey();
	const headers: Record<string, string> = {
		'Content-Type': 'application/json',
		Authorization: `Bearer ${apiKey || 'no-auth'}`
	};

	const body: Record<string, unknown> = { prompt, model, size, frames, image };
	if (audio) body.audio = audio;

	let submitRes: Response;
	try {
		submitRes = await fetch(`${base}/v1/videos`, {
			method: 'POST',
			headers,
			body: JSON.stringify(body),
			signal: opts.signal
		});
	} catch (fetchErr) {
		const message = fetchErr instanceof Error ? fetchErr.message : String(fetchErr);
		throw new Error(`Network error reaching ${base}: ${message}`);
	}
	if (!submitRes.ok) {
		let detail = '';
		try {
			detail = JSON.stringify(await submitRes.json());
		} catch {
			try {
				detail = await submitRes.text();
			} catch {
				/* ignore */
			}
		}
		throw new Error(`Videos proxy HTTP ${submitRes.status}: ${detail || 'no body'}`);
	}

	const job = (await submitRes.json()) as VideoJobStatus;
	if (!job.id) throw new Error('Videos proxy returned no job id.');

	const budget = VIDEO_POLL_BUDGET_MS[model] ?? DEFAULT_VIDEO_POLL_BUDGET_MS;
	const final = await pollVideoJob(base, job.id, headers, budget, opts.signal);
	if (final.status === 'failed') {
		throw new Error(`Video job ${job.id} failed: ${final.error ?? 'no error detail provided'}`);
	}

	const contentRes = await fetch(`${base}/v1/videos/${encodeURIComponent(job.id)}/content`, {
		method: 'GET',
		headers: { Authorization: headers['Authorization'] },
		signal: opts.signal
	});
	if (!contentRes.ok) {
		throw new Error(
			`Content fetch HTTP ${contentRes.status} for job ${job.id} (completed but no bytes).`
		);
	}
	const blob = await contentRes.blob();
	const mimeType = blob.type || 'video/mp4';
	const title = `Generated video · ${prompt.slice(0, 80)}${prompt.length > 80 ? '…' : ''}`;

	const artifact = await artifactGalleryStore.saveManual({
		kind: 'video' as DatabaseArtifactKind,
		title,
		mimeType,
		blob,
		tags: ['generated', model, ...(opts.source === 'playground' ? ['playground'] : [])],
		metadata: {
			source: opts.source,
			model,
			prompt,
			size,
			frames,
			audioDriven: Boolean(audio),
			jobId: job.id,
			generatedAt: new Date().toISOString()
		}
	});

	return {
		model,
		size,
		frames,
		prompt,
		jobId: job.id,
		video: {
			artifactId: artifact.id,
			revisionId: artifact.currentRevisionId,
			title: artifact.title,
			mimeType,
			bytes: blob.size
		}
	};
}

register({
	// Same toggle pattern as generate_image. Video is slower (~60 s
	// minimum, ~3 min for longer clips) so the user really should opt
	// in explicitly.
	gate: () => Boolean(config().videoGenEnabled),
	gateLabel: 'Video generation',
	definition: {
		type: 'function',
		function: {
			name: 'generate_video',
			description:
				'Generate a short video clip via the OpenAI-compatible videos proxy. Async: the tool submits a job, polls until completion, then saves the mp4 as a `video` artifact in the gallery. Every generation ties up the chat turn for minutes — warn the user about wait time before calling.\n\nModel matrix:\n  • `wan22-i2v` — image-to-video with 4-step lightning LoRAs. Fast (~60s for a 17-frame short, ~3min for 81 frames). Default.\n  • `wan22-i2v-hq` — same i2v pipeline without LoRAs, 20 steps. ~5× slower than wan22-i2v but noticeably sharper. Use when the user asks for quality over speed.\n  • `ltx-2.3` — LTX 2.3 distilled (i2v). ~4 min for 49 frames at 960x544. Good for slightly longer cinematic clips.\n  • `wan22-s2v` — sound-driven i2v (lip-sync / motion from audio). Needs BOTH `image` and `audio`. ~3.5 min for a 49-frame 512x288 clip.\n\nAll four models are image-to-video and require `image`. Call `get_artifact` first if the user wants to animate an existing gallery artifact.',
			parameters: {
				type: 'object',
				properties: {
					prompt: {
						type: 'string',
						description:
							'Motion / scene description. For i2v and s2v models the prompt tells the workflow how the still should animate.'
					},
					model: {
						type: 'string',
						enum: ['wan22-i2v', 'wan22-i2v-hq', 'ltx-2.3', 'wan22-s2v'],
						description:
							'Model id on the proxy. See the main description for the per-model speed/quality trade. Default `wan22-i2v`.'
					},
					image: {
						type: 'string',
						description:
							'Reference image as a `data:image/...;base64,...` URL. Required for every model (all current models are image-to-video or sound-to-video). Use `get_artifact` to fetch an existing artifact.'
					},
					audio: {
						type: 'string',
						description:
							'Reference audio as a `data:audio/...;base64,...` URL (wav / mp3 / ogg / flac). Required for `wan22-s2v`, ignored by the others. The model uses this to drive lip-sync / motion.'
					},
					size: {
						type: 'string',
						description:
							'Frame dimensions in `WIDTHxHEIGHT` form. Default `832x480` for wan22-i2v*, `960x544` for ltx-2.3, `512x288` for wan22-s2v. Higher resolutions roughly cube the runtime.'
					},
					frames: {
						type: 'integer',
						minimum: 1,
						maximum: 121,
						description:
							'Number of output frames. 17 ≈ 1 s at 16 fps. 49 is a good balance. 81 is long-form. Larger values risk hitting the per-model poll budget (see the tool description for expected runtimes).'
					}
				},
				required: ['prompt']
			}
		}
	},
	async execute(args, signal) {
		try {
			const result = await runVideoGeneration({
				source: 'generate_video',
				prompt: String(args.prompt ?? ''),
				model: typeof args.model === 'string' ? args.model : undefined,
				image: String(args.image ?? ''),
				audio: typeof args.audio === 'string' ? args.audio : undefined,
				size: typeof args.size === 'string' ? args.size : undefined,
				frames: Number(args.frames) || undefined,
				signal
			});
			return ok({
				...result,
				note: "Video is in the user's gallery; reference by artifactId in follow-ups."
			});
		} catch (e) {
			return err(e instanceof Error ? e.message : String(e));
		}
	}
});
