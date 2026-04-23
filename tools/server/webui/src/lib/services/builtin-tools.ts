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
 *
 * All tools read and write through the same DatabaseService / gallery
 * store the UI uses, so anything the model does shows up in the
 * gallery immediately (no separate sync path).
 */

import { DatabaseService } from './database.service';
import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';
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
}

const registry = new Map<string, BuiltinTool>();

function register(tool: BuiltinTool): void {
	registry.set(tool.definition.function.name, tool);
}

export function getBuiltinToolDefinitions(): OpenAIToolDefinition[] {
	return [...registry.values()].map((t) => t.definition);
}

export function hasBuiltinTool(name: string): boolean {
	return registry.has(name);
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
