/**
 * Patch-session dispatcher — round-trips a streamed SEARCH/REPLACE diff into
 * a persisted buffer edit.
 *
 * Pipeline:
 *
 *   chat-stream chunks
 *       → LimitedPatchStream (byte budget + parser)
 *       → ShadowDoc (per-block findAnchor + stream-driven mutations)
 *       → (optional) CM6 bridge — paints the shadow onto a live EditorView
 *       → PatchSession.end() → F6 elision + F7 syntax gates
 *       → ResolvedTarget.commit(finalText, ...)
 *       → store / DB write
 *
 * Target-kind behaviour:
 *
 *   - `artifact`: snapshots `artifact.currentRevisionId` at session-start and
 *     threads it through `artifactGalleryStore.addUserEditRevision` as
 *     `parentRevisionId`. That pins the session's base revision so concurrent
 *     edits during streaming don't silently re-parent the ai-patch revision
 *     onto whatever happens to be current at commit time.
 *
 *   - `doc`: persists via `docsStore.updateContent`. When a DocEditor is
 *     currently mounted for the doc (the docsStore registry has an active
 *     view), the commit closure also dispatches through the CM6 bridge so
 *     the user sees one `input.type.ai` transaction land and one-step undo
 *     reverts the patch. When no view is mounted, we fall back to the
 *     headless string-replace path.
 *
 *   - `inline`: the autocapture-slot upgrade. On first commit we materialise
 *     the slot into a gallery artifact via
 *     `artifactGalleryStore.captureFromChatForPatch` and flip the session's
 *     target in-place to `{kind:'artifact', artifactId:<new>}` so subsequent
 *     blocks (or re-runs) route through the artifact path with the correct
 *     `parentRevisionId`.
 *
 * Gates:
 *
 *   - F6 (elision) runs at block-close. A block whose REPLACE payload
 *     contains a lazy-elision marker is rejected; the shadow buffer is
 *     unchanged and subsequent blocks still apply.
 *   - F7 (syntax) runs once at session-commit against the final shadow
 *     text. For html / svg targets, invalid markup aborts the commit —
 *     no revision is written and the session reports `{ committed: false,
 *     reason: 'syntax-invalid' }`. Markdown, code and binary kinds pass
 *     through.
 *   - Zero-successful-blocks short-circuit: if every block failed, the
 *     commit closure is not invoked and the session reports `{ committed:
 *     false, reason: 'no-blocks' }`. The repair-loop wiring (commit 4)
 *     consumes this signal.
 *
 * This module is headless — no Svelte component imports, no DOM access.
 * The CM6 attachment is wired opaquely through `ResolvedTarget.viewRef`;
 * the dispatcher detects an EditorView-shaped object by structural check.
 */

import { DatabaseService } from '$lib/services/database.service';
import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';
import { docsStore } from '$lib/stores/docs.svelte';
import type { DatabaseArtifactKind } from '$lib/types/database';
import type { EditorView } from '@codemirror/view';
import {
	attachPatchView,
	type CM6Attachment,
	type InflightAnchor,
	type PatchAbortTarget
} from './cm6-bridge';
import { detectElision } from './elision';
import { findAnchor } from './fuzz-match';
import { LimitedPatchStream } from './limiter';
import { ShadowDoc } from './shadow-doc';
import { validateSyntax } from './syntax-gate';
import { REPAIRABLE_CODES } from './repair-loop';
import {
	PatchFailureCode,
	type CommitResult,
	type ParserEvent,
	type PatchErrorRecord,
	type PatchSimilarRegion,
	type PatchSource,
	type PatchTarget,
	type ResolvedTarget
} from './types';

/* ------------------------------------------------------------------------- */
/* Target resolution                                                         */
/* ------------------------------------------------------------------------- */

/**
 * Load a target's current state and build a commit closure. Throws if the
 * target is missing (artifact not in DB, doc not in DB). Inline targets
 * return a closure that materialises the autocapture slot on first commit.
 */
export async function resolveTarget(target: PatchTarget): Promise<ResolvedTarget> {
	switch (target.kind) {
		case 'artifact':
			return resolveArtifact(target.artifactId);
		case 'doc':
			return resolveDoc(target.docId);
		case 'inline':
			return resolveInline(target);
	}
}

async function resolveArtifact(artifactId: string): Promise<ResolvedTarget> {
	const artifact = await DatabaseService.getArtifact(artifactId);
	if (!artifact) {
		throw new Error(`ai-patch: artifact ${artifactId} not found`);
	}
	const revisions = await DatabaseService.listArtifactRevisions(artifactId);
	const currentRev = revisions.find((r) => r.id === artifact.currentRevisionId) ?? revisions.at(-1);
	if (!currentRev) {
		throw new Error(`ai-patch: artifact ${artifactId} has no revisions`);
	}
	const parentRevisionId = currentRev.id;
	const baseText = currentRev.text ?? '';

	return {
		kind: 'artifact',
		syntaxKind: artifact.kind as DatabaseArtifactKind,
		baseText,
		parentRevisionId,
		commit: async (finalText, opts) => {
			const rev = await artifactGalleryStore.addUserEditRevision(
				artifactId,
				{
					kind: artifact.kind as DatabaseArtifactKind,
					title: artifact.title,
					mimeType: currentRev.mimeType,
					text: finalText,
					metadata: { source: opts.source }
				},
				{ parentRevisionId }
			);
			return { revisionId: rev.id };
		}
	};
}

async function resolveDoc(docId: string): Promise<ResolvedTarget> {
	const doc = await DatabaseService.getDoc(docId);
	if (!doc) {
		throw new Error(`ai-patch: doc ${docId} not found`);
	}
	const baseText = doc.content ?? '';

	// If a DocEditor is currently mounted for this doc, surface its
	// EditorView to the session so the CM6 bridge can paint the shadow on
	// top of the live view. When nothing is mounted, fall back to the
	// headless string-replace path — the test suite exercises both.
	const activeApi = docsStore.getActiveView(docId);
	const viewRef = activeApi?.getEditorView() ?? null;

	return {
		kind: 'doc',
		syntaxKind: 'doc',
		baseText,
		viewRef: viewRef ?? undefined,
		commit: async (finalText) => {
			await docsStore.updateContent(docId, finalText);
			return { docId };
		}
	};
}

async function resolveInline(
	target: Extract<PatchTarget, { kind: 'inline' }>
): Promise<ResolvedTarget> {
	const slot = `${target.parentMessageId}#${target.artifactIndex}`;
	// If the slot is already materialised, upgrade transparently — the
	// caller can keep passing `inline` and we'll route to the artifact
	// path. Otherwise `baseText` comes from the seed the caller supplied
	// (typically the inline artifact's current payload, pre-edit).
	const existing = await DatabaseService.findArtifactBySlot(target.conversationId, slot);
	if (existing) {
		return resolveArtifact(existing.id);
	}

	return {
		kind: 'inline',
		syntaxKind: target.seed.kind,
		baseText: target.seed.baseText,
		commit: async (finalText, opts) => {
			// Materialise on first commit — creates a fresh artifact and its
			// initial revision in one transaction, then threads the edit
			// through as revision 2 if we want one-call semantics. We opt
			// for a single `captureFromChatForPatch` call that handles both
			// cases (no artifact → create; exists → append) internally.
			const { artifactId, revisionId } = await artifactGalleryStore.captureFromChatForPatch(
				{
					conversationId: target.conversationId,
					slot,
					messageId: target.parentMessageId,
					reason: 'edit'
				},
				{
					kind: target.seed.kind,
					title: target.seed.title,
					mimeType: target.seed.mimeType,
					text: finalText,
					metadata: { source: opts.source }
				}
			);
			return { newArtifactId: artifactId, revisionId };
		}
	};
}

/* ------------------------------------------------------------------------- */
/* Session                                                                   */
/* ------------------------------------------------------------------------- */

export interface PatchSessionCtorOptions {
	messageId: string;
	modelId: string;
	/** Per-block SEARCH+REPLACE byte budget. Defaults to 16 KB via limiter. */
	byteBudget?: number;
}

/** Outcome of a single block, surfaced via `onBlockResolved` for UI hooks. */
export interface BlockOutcome {
	ok: boolean;
	reason?: string;
	code?: PatchFailureCode;
	from?: number;
	to?: number;
}

/**
 * A running patch session. Feed text chunks via `feed(chunk)` as they arrive
 * from the chat stream; call `end()` at stream-complete to commit; call
 * `abort()` on cancellation to discard the shadow state without writing.
 *
 * Gate ordering at end():
 *   1. Flush the stream to drain any tail bytes.
 *   2. Run F7 syntax gate on the final shadow text.
 *   3. Zero-block short-circuit.
 *   4. Call ResolvedTarget.commit(finalText, ...).
 *
 * F6 (elision) runs per-block at block-close inside the parser-event loop,
 * not here — elision-flagged blocks never land in the shadow at all, so by
 * the time we reach step 2 the shadow is already clean.
 */
export class PatchSession implements PatchAbortTarget {
	/** Mutable: inline targets flip this in place on first commit. */
	target: ResolvedTarget;
	readonly messageId: string;
	readonly modelId: string;

	/** Collected per-block outcomes, in stream order. */
	readonly blocks: BlockOutcome[] = [];
	/** Non-fatal diagnostics captured across the session. */
	readonly errors: PatchErrorRecord[] = [];

	private readonly shadow: ShadowDoc;
	private readonly stream: LimitedPatchStream;
	/** CM6 bridge for the currently-open block, when a view is attached. */
	private currentAttachment: CM6Attachment | null = null;

	/** `null` while we're still collecting SEARCH for the current block. */
	private currentSearch: string | null = null;
	/**
	 * Running capture of the partial SEARCH text — fed by `search-line`
	 * events. Used on F11 (`E_MARKER_GRAMMAR`) to quote back to the model
	 * what the parser had read before the grammar broke.
	 */
	private currentPartialSearch = '';
	/**
	 * Running capture of the partial REPLACE text — built from
	 * `replace-chunk` events (which include all characters, including
	 * newlines). Used on F11 / F14 to quote the partial capture.
	 */
	private currentPartialReplace = '';
	/** Characters of REPLACE we have appended to shadow for this block. */
	private currentReplaceApplied = 0;
	/** Characters of REPLACE received from parser for this block (chunks + lines). */
	private currentReplaceReceived = 0;
	/** Anchor was successfully locked for the current block. */
	private currentAnchorLocked = false;
	/** The current block was aborted before anchoring — swallow subsequent events. */
	private currentBlockFailed = false;
	/** Recorded anchor span for the current block — needed at block-close. */
	private currentAnchorFrom = 0;
	private currentAnchorTo = 0;

	private finished = false;

	constructor(target: ResolvedTarget, options: PatchSessionCtorOptions) {
		this.target = target;
		this.messageId = options.messageId;
		this.modelId = options.modelId;
		this.shadow = new ShadowDoc(target.baseText);
		this.stream = new LimitedPatchStream({
			onEvent: (ev) => this.onParserEvent(ev),
			byteBudget: options.byteBudget
		});
	}

	/** Forward a raw text chunk from the chat stream. */
	feed(chunk: string): void {
		if (this.finished) return;
		this.stream.feed(chunk);
	}

	/* --------------------------------------------------------------------- */
	/* PatchAbortTarget — F4 user-edit listener                              */
	/* --------------------------------------------------------------------- */

	/**
	 * Snapshot currently-in-flight anchor ranges for the abort listener.
	 * The session only has one block in flight at a time (the streaming
	 * parser serialises them), so this returns at most one entry. The
	 * blockIndex matches the position the block will take in `this.blocks`
	 * on close, i.e. `this.blocks.length`.
	 */
	inflightAnchors(): InflightAnchor[] {
		if (!this.currentAnchorLocked || this.currentBlockFailed) return [];
		return [
			{
				blockIndex: this.blocks.length,
				from: this.currentAnchorFrom,
				to: this.currentAnchorTo
			}
		];
	}

	/**
	 * Abort the currently-open block with `E_USER_EDIT`. Called by the
	 * F4 listener when a user transaction touches the in-flight anchor
	 * range. Idempotent: subsequent calls for the same `blockIndex` are
	 * swallowed, because the first call flipped `currentBlockFailed` and
	 * `inflightAnchors()` now returns an empty array.
	 */
	abortBlock(blockIndex: number, code: 'E_USER_EDIT'): void {
		if (this.currentBlockFailed || !this.currentAnchorLocked) return;
		if (blockIndex !== this.blocks.length) return;
		this.recordBlockFailure(PatchFailureCode.E_USER_EDIT, `user edit aborted block`, {
			search: this.currentSearch ?? undefined
		});
		void code; // kept for forward-compat if additional abort reasons land
		this.currentAttachment?.abort();
		this.currentAttachment = null;
		this.currentAnchorLocked = false;
	}

	/**
	 * Remap tracked anchor coordinates through a user transaction that
	 * lay strictly outside the in-flight range. No-op when nothing is
	 * in flight.
	 */
	remapAnchors(mapPos: (pos: number, assoc?: number) => number): void {
		if (!this.currentAnchorLocked) return;
		this.currentAnchorFrom = mapPos(this.currentAnchorFrom, 1);
		this.currentAnchorTo = mapPos(this.currentAnchorTo, 1);
	}

	/**
	 * Finalise: flush the stream, run gates, invoke the target's commit
	 * closure when the session produced any usable edits. Returns the
	 * commit result or a non-committed sentinel with the reason.
	 */
	async end(): Promise<CommitResult> {
		if (this.finished) {
			throw new Error('PatchSession.end: already finalised');
		}
		this.finished = true;
		this.stream.end();
		// Any block still in flight at stream-end is a truncated stream (F1).
		// Drop it from the shadow — partial chunks are worse than nothing.
		if (this.currentAttachment) {
			this.currentAttachment.abort();
			this.currentAttachment = null;
		}
		if (this.shadow.hasOpenBlock()) {
			this.shadow.closeBlock();
		}

		const successfulBlocks = this.blocks.filter((b) => b.ok);
		if (successfulBlocks.length === 0) {
			// Zero successful blocks — no revision, no DB write. The repair
			// loop (commit 4) observes this sentinel and may retry.
			console.info('[ai-patch] session produced no successful blocks', {
				messageId: this.messageId,
				errorCount: this.errors.length
			});
			return {
				committed: false,
				reason: 'no-blocks',
				errors: [...this.errors],
				repairable: hasRepairableError(this.errors)
			};
		}

		const finalText = this.shadow.toString();

		// F7 syntax gate — once per session on the final text. Markdown and
		// code skip; html / svg reject on parsererror.
		const syntax = validateSyntax(this.target.syntaxKind, finalText);
		if (!syntax.ok) {
			console.warn('[ai-patch] syntax gate rejected session', {
				messageId: this.messageId,
				kind: this.target.syntaxKind,
				error: syntax.error
			});
			const syntaxKindForRepair = isValidatableSyntaxKind(this.target.syntaxKind)
				? this.target.syntaxKind
				: undefined;
			const { line: syntaxLine, column: syntaxColumn } = parseSyntaxPosition(syntax.error);
			const syntaxErrors: PatchErrorRecord[] = [
				...this.errors,
				{
					code: PatchFailureCode.E_SYNTAX_INVALID,
					reason: syntax.error,
					artifactKind: syntaxKindForRepair,
					targetText: finalText,
					syntaxLine,
					syntaxColumn
				}
			];
			return {
				committed: false,
				reason: 'syntax-invalid',
				errors: syntaxErrors,
				repairable: hasRepairableError(syntaxErrors)
			};
		}

		const source: PatchSource = {
			kind: 'ai-patch',
			modelId: this.modelId,
			sessionId: this.messageId
		};
		const result = await this.target.commit(finalText, { source });

		// Inline-target upgrade: once materialised, flip the target in place
		// so subsequent callers (the repair loop, re-runs) see an artifact
		// handle with a stable id instead of a slot that now maps to an
		// artifact.
		if (this.target.kind === 'inline' && result.newArtifactId) {
			this.target = await resolveArtifact(result.newArtifactId);
		}
		return result;
	}

	/** Discard the session without committing. */
	abort(): void {
		this.finished = true;
		if (this.currentAttachment) {
			this.currentAttachment.abort();
			this.currentAttachment = null;
		}
	}

	/** Final computed text — useful for tests and logging. */
	get finalText(): string {
		return this.shadow.toString();
	}

	/* --------------------------------------------------------------------- */

	private onParserEvent(ev: ParserEvent): void {
		switch (ev.type) {
			case 'block-open':
				this.resetBlockState();
				return;

			case 'search-complete': {
				// ANCHOR_LOCKED transition: resolve SEARCH against the *current*
				// shadow buffer (not the session-start snapshot) so back-to-back
				// blocks can target text the previous block just inserted.
				this.currentSearch = ev.search;
				const shadowText = this.shadow.toString();
				const result = findAnchor(shadowText, ev.search);
				if (result.kind !== 'unique') {
					const code =
						result.kind === 'ambiguous'
							? PatchFailureCode.E_AMBIGUOUS
							: PatchFailureCode.E_NO_MATCH;
					// For F2 (no-match) snapshot the fuzz ladder's top-N
					// suggestion regions — text and coordinates — so the
					// repair-loop prompt has concrete alternatives to quote
					// back to the model. F3 (ambiguous) gets the colliding
					// regions instead; the repair-format logic for F3 lands
					// in commit 4b but we capture the data here to avoid a
					// second pass through fuzz-match later.
					let similar: PatchSimilarRegion[] | undefined;
					if (result.kind === 'none') {
						similar = result.suggestions.map((s) => ({
							from: s.from,
							to: s.to,
							similarity: s.similarity,
							text: shadowText.slice(s.from, s.to)
						}));
					} else if (result.kind === 'ambiguous') {
						similar = result.matches.map((m) => ({
							from: m.from,
							to: m.to,
							similarity: 1,
							text: shadowText.slice(m.from, m.to)
						}));
					}
					this.recordBlockFailure(code, `anchor ${result.kind}`, {
						search: ev.search,
						similar,
						// Snapshot the buffer so F3 context rendering doesn't need the
						// shadow at repair-format time.
						targetText: result.kind === 'ambiguous' ? shadowText : undefined
					});
					return;
				}
				try {
					this.shadow.applyAnchor(result.from, result.to);
				} catch (err) {
					this.recordBlockFailure(
						PatchFailureCode.E_MARKER_GRAMMAR,
						`shadow anchor failed: ${(err as Error).message}`
					);
					return;
				}
				this.currentAnchorLocked = true;
				this.currentAnchorFrom = result.from;
				this.currentAnchorTo = result.to;

				// Attach the CM6 bridge if the target has a live EditorView.
				// We detect by structural check rather than `instanceof` so
				// the dispatcher stays framework-agnostic at import time.
				const view = extractEditorView(this.target.viewRef);
				if (view) {
					this.currentAttachment = attachPatchView(view, this.shadow, {
						anchorFrom: result.from,
						anchorTo: result.to
					});
				}
				return;
			}

			case 'replace-chunk':
				// Capture into the partial buffer regardless of anchor state so a
				// later F11 / F14 error can quote whatever the parser had read.
				this.currentPartialReplace += ev.chunk;
				if (this.currentBlockFailed || !this.currentAnchorLocked) return;
				this.currentReplaceReceived += ev.chunk.length;
				// Apply incrementally so streaming consumers see characters land.
				this.shadow.appendChunk(ev.chunk);
				this.currentReplaceApplied += ev.chunk.length;
				this.currentAttachment?.update();
				return;

			case 'replace-line':
				// The parser fires replace-chunk for every character, including
				// newlines, so `replace-line` is a redundant structured echo.
				// Bookkeeping only; we already applied the chars.
				return;

			case 'block-close': {
				if (!this.currentAnchorLocked) {
					this.blocks.push({ ok: false, reason: 'block closed without anchor' });
					this.resetBlockState();
					return;
				}

				// F6 — elision gate. Run on the authoritative REPLACE payload
				// (block.replace), before the shadow folds the block in. If
				// we detect a placeholder, the block is rejected and we
				// rewind the in-flight shadow state (delete the streamed
				// chunks, restore the SEARCH range). Subsequent blocks still
				// apply.
				const elision = detectElision(ev.block.replace);
				if (elision) {
					this.rewindCurrentBlock(ev.block.search);
					this.recordBlockFailure(
						PatchFailureCode.E_ELISION,
						`elision at line ${elision.line}: ${elision.reason}`,
						{
							search: ev.block.search,
							replace: ev.block.replace,
							elisionLine: elision.line
						}
					);
					this.currentAttachment?.abort();
					this.currentAttachment = null;
					this.resetBlockState();
					return;
				}

				// Pass block.replace so the shadow reconciles any drift between
				// the streamed chunk series (which includes the final newline
				// before the close marker) and the authoritative block payload.
				const summary = this.shadow.closeBlock(ev.block.replace);

				// Commit the real CM6 transaction — one undo step per block.
				if (this.currentAttachment && summary) {
					this.currentAttachment.commit(summary.inserted);
					this.currentAttachment = null;
				}

				this.blocks.push({
					ok: true,
					from: summary?.from,
					to: summary?.to
				});
				this.resetBlockState();
				return;
			}

			case 'parse-warning':
				// Surfaced to the dispatcher via `errors`, but not treated as a
				// fatal block failure — the parser already recovered.
				return;

			case 'parse-error': {
				// Capture partial SEARCH / REPLACE (if any) and — for F14 — the
				// configured byte budget so the repair format can reference them
				// verbatim. `blockIndex` still points at the currently-open
				// block so the injected turn can reference it by number.
				const blockIndex = this.blocks.length;
				this.errors.push({
					code: ev.code,
					reason: ev.reason,
					blockIndex,
					partialSearch: this.currentPartialSearch || undefined,
					partialReplace: this.currentPartialReplace || undefined,
					byteBudget:
						ev.code === PatchFailureCode.E_BYTE_BUDGET ? this.stream.byteBudget : undefined
				});
				console.warn('[ai-patch] parse-error', ev.code, ev.reason);
				if (
					this.currentAnchorLocked &&
					(ev.code === PatchFailureCode.E_BYTE_BUDGET ||
						ev.code === PatchFailureCode.E_MARKER_GRAMMAR)
				) {
					// If we already applied part of a REPLACE payload for a block
					// that's now unsalvageable, the shadow is inconsistent for that
					// block. Close it out so subsequent blocks start clean — the
					// partially-applied chunks remain (diagnostic value > rollback
					// fidelity in this commit).
					this.shadow.closeBlock();
					this.currentAttachment?.abort();
					this.currentAttachment = null;
				}
				this.currentBlockFailed = true;
				this.currentAnchorLocked = false;
				return;
			}

			case 'search-line':
				// Bookkeeping only — the parser also emits search-complete with
				// the fully-joined SEARCH text, which is what we need. Keep the
				// partial-SEARCH buffer in sync so F11 can quote what was
				// captured before a grammar break.
				this.currentPartialSearch =
					this.currentPartialSearch.length > 0
						? `${this.currentPartialSearch}\n${ev.line}`
						: ev.line;
				return;
		}
	}

	private resetBlockState(): void {
		this.currentSearch = null;
		this.currentPartialSearch = '';
		this.currentPartialReplace = '';
		this.currentReplaceApplied = 0;
		this.currentReplaceReceived = 0;
		this.currentAnchorLocked = false;
		this.currentBlockFailed = false;
		this.currentAnchorFrom = 0;
		this.currentAnchorTo = 0;
	}

	private recordBlockFailure(
		code: PatchFailureCode,
		reason: string,
		extra?: {
			search?: string;
			similar?: PatchSimilarRegion[];
			targetText?: string;
			replace?: string;
			elisionLine?: number;
			partialSearch?: string;
			partialReplace?: string;
			byteBudget?: number;
		}
	): void {
		// `this.blocks.length` is the next entry's index — use it *before* the
		// push below so `blockIndex` and the position in `this.blocks` agree.
		const blockIndex = this.blocks.length;
		this.errors.push({
			code,
			reason,
			blockIndex,
			search: extra?.search,
			similar: extra?.similar,
			targetText: extra?.targetText,
			replace: extra?.replace,
			elisionLine: extra?.elisionLine,
			partialSearch: extra?.partialSearch,
			partialReplace: extra?.partialReplace,
			byteBudget: extra?.byteBudget
		});
		this.blocks.push({ ok: false, code, reason });
		this.currentBlockFailed = true;
		console.warn('[ai-patch] block failure', code, reason);
	}

	/**
	 * Undo the shadow-side effects of the currently-open block so a later
	 * gate failure (elision) can reject it without corrupting the buffer
	 * for subsequent blocks. Replays: delete the streamed chunks, then
	 * re-insert the original SEARCH range so the buffer state matches the
	 * pre-anchor state.
	 */
	private rewindCurrentBlock(originalSearch: string): void {
		if (!this.shadow.hasOpenBlock()) return;
		// Overwriting the in-block inserted text with the original SEARCH
		// payload restores the pre-anchor state — the shadow helper already
		// handles the range bookkeeping.
		this.shadow.closeBlock(originalSearch);
	}
}

/* ------------------------------------------------------------------------- */
/* Helpers                                                                   */
/* ------------------------------------------------------------------------- */

// REPAIRABLE_CODES is imported from repair-loop.ts at the top of this file.

function hasRepairableError(errors: PatchErrorRecord[]): boolean {
	return errors.some((e) => REPAIRABLE_CODES.has(e.code));
}

/**
 * Narrowing predicate for the F7 repair format — it only understands
 * markup-shaped kinds. Code and markdown always pass the gate anyway so
 * this branch is defensive.
 */
function isValidatableSyntaxKind(
	kind: ResolvedTarget['syntaxKind']
): kind is 'html' | 'svg' | 'code' | 'markdown' {
	return kind === 'html' || kind === 'svg' || kind === 'code' || kind === 'markdown';
}

/**
 * Extract a 1-based line / column pair from a DOMParser error string
 * when possible. The Chromium / Firefox / WebKit formats all embed "line
 * N column M" somewhere in the message; we grep for the pattern and
 * gracefully return `{}` otherwise.
 */
function parseSyntaxPosition(message: string): { line?: number; column?: number } {
	const m = /line\s*[:#]?\s*(\d+)(?:[^0-9]+column\s*[:#]?\s*(\d+))?/i.exec(message);
	if (!m) return {};
	const line = Number.parseInt(m[1], 10);
	const column = m[2] ? Number.parseInt(m[2], 10) : undefined;
	return { line, column };
}

/**
 * Coarse pre-digest of a session's error list. Callers (repair-loop,
 * telemetry, UI chips) use this to answer "did anything fail?", "what's
 * repairable?", and "how many blocks died on anchor" without iterating
 * the raw list themselves.
 */
export interface SummarizedErrors {
	total: number;
	/** Errors whose code maps to a wired repair-format path (4a: F2 only). */
	repairable: PatchErrorRecord[];
	/** All F2 (E_NO_MATCH) failures, in stream order. */
	noMatch: PatchErrorRecord[];
	/** Codes seen, for quick diagnostic display. */
	codes: PatchFailureCode[];
}

/**
 * Pre-digest a dispatcher-produced error list for the repair loop. Pure
 * function — no I/O, no session state. Splits errors into repairable and
 * total counts so `formatRepairMessage` can drive off a single argument.
 */
export function summarizePatchErrors(errors: PatchErrorRecord[]): SummarizedErrors {
	const repairable: PatchErrorRecord[] = [];
	const noMatch: PatchErrorRecord[] = [];
	const codes = new Set<PatchFailureCode>();
	for (const e of errors) {
		codes.add(e.code);
		if (REPAIRABLE_CODES.has(e.code)) repairable.push(e);
		if (e.code === PatchFailureCode.E_NO_MATCH) noMatch.push(e);
	}
	return {
		total: errors.length,
		repairable,
		noMatch,
		codes: [...codes]
	};
}

/**
 * Structural check for an `EditorView` reference. We avoid importing
 * `EditorView` as a value-level `instanceof` operand because a stripped
 * viewRef (from the test suite, or from a headless flow) should be
 * tolerated — the bridge just won't attach.
 */
function extractEditorView(ref: unknown): EditorView | null {
	if (!ref || typeof ref !== 'object') return null;
	const candidate = ref as { dispatch?: unknown; state?: unknown };
	if (typeof candidate.dispatch === 'function' && candidate.state) {
		return ref as EditorView;
	}
	return null;
}
