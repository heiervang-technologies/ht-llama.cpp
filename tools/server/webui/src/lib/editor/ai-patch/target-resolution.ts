/**
 * Live-chat target resolution heuristic (commit 5).
 *
 * Given a `SearchReplaceBlock` extracted from an assistant stream plus a
 * small context object describing the turn, decide which buffer the block
 * should edit. Three paths, in priority order:
 *
 *   1. **Filename line above the fence.** The parser already captured a
 *      `pendingFilename` when the `<<<<<<< SEARCH` line was preceded by
 *      something shaped like a path. We re-apply a stricter regex here
 *      (extension-aware) so inline code references like `MyClass` don't
 *      resolve to a stray doc. If the regex hits and Dexie finds a doc by
 *      that name (case-insensitive), we return a `doc` target. If the
 *      regex hits but Dexie returns nothing, we do NOT fall through — it's
 *      an unambiguous `E_NO_TARGET` (the model declared intent; we won't
 *      silently retarget onto an unrelated autocapture slot).
 *
 *   2. **Naked fence, explicit current-artifact.** If the caller already
 *      knows which artifact the turn is editing (they opened
 *      `/artifacts/<id>` in the background, or a previous block this turn
 *      materialised one), resolve to that artifact.
 *
 *   3. **Naked fence, autocapture slot fallback.** The assistant hasn't
 *      finished its turn yet, but it may have opened a capture slot via a
 *      previous block in the same turn. We target artifact-index 0 under
 *      the assistant's parent (user) message id — matching the slot key
 *      shape `${slotParent}#${index}` used by `ChatMessageAssistant`. The
 *      `inline` target carries a seed so the dispatcher can materialise
 *      the slot on first commit (see `captureFromChatForPatch`).
 *
 *   4. **None of the above** → return `null`. The caller emits
 *      `E_NO_TARGET` and skips the block.
 *
 * This module is headless: no CM6, no Svelte, no DOM. Only Dexie (via
 * `DatabaseService`) and the `PatchTarget` contract.
 */

import { DatabaseService } from '$lib/services/database.service';
import type { PatchTarget, SearchReplaceBlock } from './types';

/**
 * Supported filename extensions for the "filename-line-above-fence"
 * heuristic. Kept in sync with the parser's looser `looksLikeFilename`
 * guard but stricter: we require a recognised extension here so
 * free-floating identifiers don't accidentally resolve.
 */
const FILENAME_PATTERN =
	/^([A-Za-z0-9_.\-/]+\.(?:html|svg|md|markdown|txt|json|ts|tsx|js|jsx|css|py|sh|yml|yaml))\s*$/;

/**
 * Seed payload used when falling through to the autocapture-slot path.
 * The dispatcher needs a `kind`, `title`, `mimeType`, and `baseText` to
 * materialise a fresh artifact on first commit. The caller supplies the
 * `baseText` (the artifact's current payload pre-edit); the other fields
 * default to the "code" kind in v1 — we don't yet sniff html/svg from the
 * REPLACE payload at target-resolution time.
 */
export interface InlineSeed {
	kind: 'html' | 'svg' | 'code' | 'markdown';
	title: string;
	mimeType: string;
	baseText: string;
}

/**
 * Context threaded from the chat-stream bootstrap into the target
 * resolver. Everything is optional because a cold start (no artifact
 * open, no autocapture yet) still needs to run the filename-above-fence
 * path.
 */
export interface TargetResolutionContext {
	/** Conversation the assistant stream belongs to. Required for inline. */
	conversationId?: string;
	/**
	 * The user message id whose child is the streaming assistant message.
	 * This keys the autocapture slot: `${parentMessageId}#0`.
	 */
	parentMessageId?: string;
	/**
	 * Id of the artifact the turn is explicitly targeting. Populated when
	 * the user opened `/artifacts/<id>` before sending, or when a previous
	 * block in the same turn flipped an inline target into an artifact
	 * handle. Wins over the autocapture fallback.
	 */
	currentArtifactId?: string | null;
	/**
	 * Seed for the inline fallback — the artifact's base text and kind.
	 * Optional; when absent we can't materialise an autocapture slot and
	 * the naked-fence branch returns `null`.
	 */
	inlineSeed?: InlineSeed | null;
}

/**
 * True when `candidate` looks like a filename the resolver would accept.
 * Exported so the parser's pendingFilename capture (which is looser) can
 * cross-validate without re-importing the regex.
 */
export function isResolvableFilename(candidate: string): boolean {
	return FILENAME_PATTERN.test(candidate.trim());
}

/**
 * Resolve a streamed SEARCH/REPLACE block to a concrete `PatchTarget`.
 * Returns `null` when no target can be determined — the caller should
 * record an `E_NO_TARGET` error and skip the block.
 */
export async function resolveTargetFromAssistantContext(
	block: SearchReplaceBlock,
	ctx: TargetResolutionContext
): Promise<PatchTarget | null> {
	// Path 1 — filename above fence.
	if (block.filename && isResolvableFilename(block.filename)) {
		const doc = await DatabaseService.findDocByName(block.filename.trim());
		if (doc) return { kind: 'doc', docId: doc.id };
		// Filename declared but not found → unambiguous miss. We do NOT
		// fall through to the autocapture slot; that would silently
		// retarget the edit onto an unrelated buffer.
		return null;
	}

	// Path 2 — explicit current artifact wins over autocapture.
	if (ctx.currentArtifactId) {
		return { kind: 'artifact', artifactId: ctx.currentArtifactId };
	}

	// Path 3 — autocapture slot fallback. Requires conversation + parent +
	// a seed to materialise the slot on first commit.
	if (ctx.conversationId && ctx.parentMessageId && ctx.inlineSeed) {
		return {
			kind: 'inline',
			conversationId: ctx.conversationId,
			parentMessageId: ctx.parentMessageId,
			artifactIndex: 0,
			seed: ctx.inlineSeed
		};
	}

	// Path 4 — give up.
	return null;
}
