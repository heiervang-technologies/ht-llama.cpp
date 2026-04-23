/**
 * Live-chat bootstrap (commit 5) — lazily opens a `PatchSession` when the
 * assistant stream produces a `<<<<<<< SEARCH` marker, and flushes its
 * `CommitResult` into the chat-integration registry on stream end so
 * `chatStore.maybeRunPatchRepairLoop` can consume it.
 *
 * Contract:
 *
 *   - `feed(chunk)` is called for every text delta from the ChatService
 *     SSE stream. Until a sniffer parser recognises the first SEARCH
 *     marker, the chunks are buffered but NO session is opened, NO DB
 *     writes occur, and NO registry entries are created.
 *   - On first `block-open` the bootstrap resolves a target
 *     asynchronously (filename line → doc, otherwise current artifact,
 *     otherwise autocapture slot), opens a `PatchSession` against it,
 *     and replays the buffered bytes into the session. Subsequent
 *     chunks go straight through.
 *   - On `end()` the session (if any) is finalised and its `CommitResult`
 *     is recorded via `recordCompletedPatchSession`. When NO session was
 *     opened, `end()` is a true no-op — the chat turn completes with
 *     zero ai-patch state transitions.
 *   - `abort()` cancels any open session without committing.
 *
 * The raw assistant text continues to render in the chat message
 * unchanged — this bootstrap taps the stream, it does not replace it.
 * The message record still shows SEARCH/REPLACE verbatim in scrollback;
 * the bootstrap produces the *effect* of the patch on the target buffer.
 *
 * Scope: one target per session. A turn whose first block resolves to
 * doc `foo.md` pins the session to that doc; subsequent blocks in the
 * same turn that target a different buffer are recorded as
 * `E_NO_TARGET`. Multi-target turns are a commit 6 concern.
 */

import { PatchSession, resolveTarget } from './dispatcher';
import { StreamingPatchParser } from './parser';
import { recordCompletedPatchSession } from './chat-integration';
import {
	resolveTargetFromAssistantContext,
	type InlineSeed,
	type TargetResolutionContext
} from './target-resolution';
import {
	PatchFailureCode,
	type CommitResult,
	type ParserEvent,
	type PatchErrorRecord
} from './types';

export interface ChatPatchBootstrapOptions {
	/** Assistant message id — becomes the parentSessionId for any repair loop. */
	messageId: string;
	/** Model id for the ai-patch source attribution. */
	modelId: string;
	/** Conversation id — threaded into inline targets. */
	conversationId: string;
	/** User message the assistant turn is a child of — keys the autocapture slot. */
	parentMessageId: string;
	/** Current open artifact, if any (route param / gallery selection). */
	getCurrentArtifactId?: () => string | null | undefined;
	/** Builder for the inline seed — invoked only on the autocapture fallback. */
	getInlineSeed?: () => InlineSeed | null | undefined;
	/** Per-block SEARCH+REPLACE byte budget override. Defaults to 16 KB. */
	byteBudget?: number;
	/**
	 * Toast emitter for `E_NO_TARGET` (which repair cannot fix). Defaults
	 * to `console.warn`. Supplied by the chat store so the sonner toast
	 * uses the same surface as the rest of the app.
	 */
	onNoTarget?: (reason: string) => void;
}

export interface ChatPatchBootstrap {
	/** Forward one SSE text delta. Safe to call after `end()` (no-op). */
	feed(chunk: string): void;
	/**
	 * Flush the parser, commit any open session, and record the result
	 * for the repair loop to pick up. Idempotent.
	 */
	end(): Promise<void>;
	/** Cancel any in-flight session without committing. Idempotent. */
	abort(): void;
	/** Introspection for tests: true once the parser saw its first block-open. */
	hasOpenedSession(): boolean;
}

type Phase =
	/** No SEARCH marker seen yet; buffering raw chunks. */
	| { kind: 'IDLE' }
	/** block-open seen; async target resolution in flight. */
	| { kind: 'RESOLVING'; filename: string | undefined }
	/** Target resolved + session open; feed passes straight through. */
	| { kind: 'ACTIVE'; session: PatchSession }
	/** Target resolution failed; drop chunks until end() (still recorded). */
	| { kind: 'NO_TARGET' }
	/** abort()/end() called. */
	| { kind: 'CLOSED' };

/**
 * Build a bootstrap handle for one assistant turn. Call `feed(chunk)`
 * inside the stream callbacks and `end()` from the stream-complete hook.
 */
export function createChatPatchBootstrap(opts: ChatPatchBootstrapOptions): ChatPatchBootstrap {
	const preSessionErrors: PatchErrorRecord[] = [];
	const onNoTarget = opts.onNoTarget ?? defaultNoTargetToast;

	let phase: Phase = { kind: 'IDLE' };
	// Raw stream bytes buffered until the session is open. Once we enter
	// ACTIVE these are feed()-ed into the session in order and the buffer
	// is discarded.
	let buffer = '';

	// Sniffer parser — purely detects `block-open` so we can trigger
	// target resolution. Events are informational; the real parsing
	// happens inside `PatchSession`.
	const sniffer = new StreamingPatchParser({
		onEvent(ev: ParserEvent) {
			if (ev.type === 'block-open' && phase.kind === 'IDLE') {
				phase = { kind: 'RESOLVING', filename: ev.filename };
				// Kick off async resolution; don't await.
				void beginResolution(ev.filename);
			}
		}
	});

	// Cheap read helper that launders `phase.kind` through `string` so TS
	// doesn't incorrectly narrow mutable module state across `await`
	// expressions. The real `phase` union is still enforced at every write
	// site; this accessor only loosens the *read* type.
	const currentPhase = (): string => phase.kind;

	async function beginResolution(filename: string | undefined): Promise<void> {
		try {
			const ctx: TargetResolutionContext = {
				conversationId: opts.conversationId,
				parentMessageId: opts.parentMessageId,
				currentArtifactId: opts.getCurrentArtifactId?.() ?? null,
				inlineSeed: opts.getInlineSeed?.() ?? null
			};
			const target = await resolveTargetFromAssistantContext(
				{ search: '', replace: '', filename },
				ctx
			);
			if (currentPhase() === 'CLOSED') return;
			if (!target) {
				const reason = filename
					? `no doc matches filename "${filename}"`
					: 'no target buffer resolvable for naked SEARCH/REPLACE fence';
				preSessionErrors.push({
					code: PatchFailureCode.E_NO_TARGET,
					reason,
					blockIndex: 0
				});
				onNoTarget(reason);
				phase = { kind: 'NO_TARGET' };
				buffer = '';
				return;
			}
			const resolved = await resolveTarget(target);
			if (currentPhase() === 'CLOSED') return;
			const session = new PatchSession(resolved, {
				messageId: opts.messageId,
				modelId: opts.modelId,
				byteBudget: opts.byteBudget
			});
			// Replay the buffered bytes. The session's own parser drives
			// its state machine off the raw bytes; we must feed from the
			// stream start so the `<<<<<<< SEARCH` line lands in the
			// session's buffer too.
			if (buffer.length > 0) {
				session.feed(buffer);
				buffer = '';
			}
			phase = { kind: 'ACTIVE', session };
		} catch (err) {
			console.warn('[ai-patch] target resolution threw', err);
			if (currentPhase() === 'CLOSED') return;
			preSessionErrors.push({
				code: PatchFailureCode.E_NO_TARGET,
				reason: `target resolution threw: ${(err as Error).message}`,
				blockIndex: 0
			});
			phase = { kind: 'NO_TARGET' };
			buffer = '';
		}
	}

	return {
		feed(chunk: string) {
			if (!chunk) return;
			switch (phase.kind) {
				case 'IDLE':
					buffer += chunk;
					sniffer.feed(chunk);
					return;
				case 'RESOLVING':
					// Still waiting on async resolve; keep buffering.
					buffer += chunk;
					return;
				case 'ACTIVE':
					phase.session.feed(chunk);
					return;
				case 'NO_TARGET':
				case 'CLOSED':
					return;
			}
		},

		async end() {
			if (phase.kind === 'CLOSED') return;
			sniffer.end();
			// If resolution is still pending, wait for it. We give it a
			// microtask loop rather than a real timeout — Dexie calls
			// resolve fast in the hot path and the tests mock them to
			// be synchronous.
			while (phase.kind === 'RESOLVING') {
				await Promise.resolve();
			}
			if (phase.kind === 'ACTIVE') {
				try {
					const session = phase.session;
					phase = { kind: 'CLOSED' };
					const result = await session.end();
					recordCompletedPatchSession(opts.messageId, mergeErrors(result, preSessionErrors));
				} catch (err) {
					console.warn('[ai-patch] session.end threw', err);
					recordCompletedPatchSession(opts.messageId, {
						committed: false,
						reason: 'aborted',
						errors: [
							...preSessionErrors,
							{
								code: PatchFailureCode.E_NO_TARGET,
								reason: `session.end threw: ${(err as Error).message}`
							}
						],
						repairable: false
					});
				}
				return;
			}
			if (phase.kind === 'NO_TARGET') {
				phase = { kind: 'CLOSED' };
				recordCompletedPatchSession(opts.messageId, {
					committed: false,
					reason: 'no-blocks',
					errors: preSessionErrors,
					repairable: false
				});
				return;
			}
			// IDLE — parser never saw a block. Zero state transitions.
			phase = { kind: 'CLOSED' };
		},

		abort() {
			if (phase.kind === 'CLOSED') return;
			if (phase.kind === 'ACTIVE') phase.session.abort();
			phase = { kind: 'CLOSED' };
		},

		hasOpenedSession() {
			return phase.kind === 'ACTIVE';
		}
	};
}

function mergeErrors(result: CommitResult, extras: PatchErrorRecord[]): CommitResult {
	if (extras.length === 0) return result;
	return {
		...result,
		errors: [...extras, ...(result.errors ?? [])]
	};
}

function defaultNoTargetToast(reason: string): void {
	console.warn('[ai-patch] E_NO_TARGET', reason);
}
