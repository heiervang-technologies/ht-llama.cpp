/**
 * Chat-store ↔ ai-patch glue.
 *
 * The repair loop is headless (see `stream-hook.ts` — `runPatchRepairLoop`
 * drives its own stream factory) and its unit tests mock the chat service.
 * The live chat flow, however, is driven by `chatStore.sendMessage` which
 * owns the abort controller + streaming state machine. We can't hand that
 * machine to the orchestrator, so the wiring is one-directional: this
 * module is imported by `chatStore`, not the other way round, and
 * provides two pieces:
 *
 *   1. A "completed patch session" registry — anywhere that opens a
 *      `PatchSession` for a chat turn (future commit 5) calls
 *      `recordCompletedPatchSession(messageId, result)` at `session.end()`
 *      time. The chat store picks the result up synchronously at the end
 *      of `streamChatCompletion`.
 *
 *   2. `handleCompletedPatchSession(ctx)` — the "one step of the loop"
 *      helper. Checks the result against the reflection budget, injects
 *      a synthetic `patch-repair` user turn if it should retry, then
 *      invokes the caller-supplied `sendMessage('')` to recurse through
 *      the real chat pipeline. The counter lives in this module keyed by
 *      `parentSessionId` so it survives the recursion.
 *
 * No imports from `$lib/stores/chat.svelte.ts` here — the dep goes one
 * way only, so chatStore can import this module at top-level without
 * circular-import pain.
 */

import { formatRepairMessage, MAX_REFLECTIONS, REPAIRABLE_CODES } from './repair-loop';
import { injectRepairTurn } from './repair-injector';
import {
	registerPatchSession,
	unregisterPatchSession,
	type SessionHandle
} from './session-registry';
import type { CommitResult } from './types';

/**
 * Callback shape for the retriggering a new assistant stream off the
 * just-injected patch-repair user turn. The chat store implements this by
 * creating a fresh assistant message parented to the conversation's new
 * leaf and running `streamChatCompletion` — it can't re-enter
 * `sendMessage` because that guard-returns on empty content. Declared
 * structurally so this module has no dep on the chat store (cycle
 * prevention).
 */
export type RunAssistantTurn = () => Promise<void>;

/** Context passed into `handleCompletedPatchSession`. */
export interface HandleCompletedPatchSessionContext {
	conversationId: string;
	/**
	 * Stable id used to key the reflection counter. Matches the assistant
	 * message id that produced the original failing patch — the repair
	 * loop re-uses it across retries so the × affordance cancels the
	 * whole loop, not just the current retry.
	 */
	parentSessionId: string;
	result: CommitResult;
	/** Retrigger the stream after the synthetic turn has been injected. */
	runAssistantTurn: RunAssistantTurn;
	/** Optional override for the reflection budget (tests). */
	maxReflections?: number;
	/** Optional toast emitter; defaults to `console.warn`. */
	onExhausted?: (message: string) => void;
}

/* ------------------------------------------------------------------------- */
/* Completed-session registry                                                */
/* ------------------------------------------------------------------------- */

const completed = new Map<string, CommitResult>();

/**
 * Record the `CommitResult` for a patch session that just ended. Typically
 * called from the `session.end()` path in whatever bootstraps the
 * PatchSession for a chat turn. The chat store calls
 * `consumeCompletedPatchSession(messageId)` at stream-finish to pick it up.
 */
export function recordCompletedPatchSession(messageId: string, result: CommitResult): void {
	completed.set(messageId, result);
}

/**
 * Pop the pending `CommitResult` for `messageId`, if any. Returns `null`
 * when no session was recorded for the message — which is the common case
 * (most assistant turns are not patch sessions).
 */
export function consumeCompletedPatchSession(messageId: string): CommitResult | null {
	const result = completed.get(messageId);
	if (!result) return null;
	completed.delete(messageId);
	return result;
}

/** Test helper — drop both the completed-results and counter state. */
export function __resetChatIntegrationForTest(): void {
	completed.clear();
	reflections.clear();
}

/* ------------------------------------------------------------------------- */
/* Reflection accounting                                                     */
/* ------------------------------------------------------------------------- */

const reflections = new Map<string, number>();

/** Current reflection count for a given parent session. 0 when unset. */
export function getReflectionCount(parentSessionId: string): number {
	return reflections.get(parentSessionId) ?? 0;
}

/* ------------------------------------------------------------------------- */
/* Orchestrator — one step of the loop                                       */
/* ------------------------------------------------------------------------- */

/**
 * Given a just-finished patch session's `CommitResult`, decide whether to
 * inject a synthetic user turn + recurse through `sendMessage`, or end
 * the loop (clean commit, non-repairable failure, or budget exhausted).
 *
 * Idempotent per call: the reflection counter is incremented before the
 * recursive `sendMessage` runs, so a second stream-complete for the same
 * `parentSessionId` reads the bumped value and stops when the budget is
 * out.
 *
 * Registers a `SessionHandle` in the session registry for the duration of
 * the injected turn so the UI × affordance can stop the loop
 * mid-recursion. The handle is unregistered once the recursive
 * `sendMessage` call returns (success or throw) so the × disappears the
 * moment the loop has no more outstanding work.
 */
export async function handleCompletedPatchSession(
	ctx: HandleCompletedPatchSessionContext
): Promise<void> {
	const budget = ctx.maxReflections ?? MAX_REFLECTIONS;
	const result = ctx.result;

	// Clean commit → clear any lingering counter and bail.
	if (result.committed !== false) {
		reflections.delete(ctx.parentSessionId);
		return;
	}

	// Non-repairable failure → clear counter, bail. `repairable` may be
	// undefined on older sentinels; treat as non-repairable to be safe.
	if (result.repairable !== true) {
		reflections.delete(ctx.parentSessionId);
		return;
	}

	const repairable = (result.errors ?? []).find((e) => REPAIRABLE_CODES.has(e.code));
	if (!repairable) {
		reflections.delete(ctx.parentSessionId);
		return;
	}

	const current = reflections.get(ctx.parentSessionId) ?? 0;
	if (current >= budget) {
		const msg = `Patch retries exhausted after ${budget} attempts. Committed blocks kept.`;
		(ctx.onExhausted ?? defaultExhaustedToast)(msg);
		reflections.delete(ctx.parentSessionId);
		return;
	}

	const nextReflection = current + 1;
	reflections.set(ctx.parentSessionId, nextReflection);

	// Short-lived SessionHandle so the discard × affordance can cancel the
	// injected turn's stream. `abort()` is best-effort: we don't own the
	// chat-store's AbortController from here, but marking exhausted is
	// enough to keep the loop from firing again on the next result.
	let cancelled = false;
	const handle: SessionHandle = {
		abort: () => {
			cancelled = true;
		},
		markExhausted: () => {
			reflections.set(ctx.parentSessionId, budget);
			cancelled = true;
		},
		reflectionCount: () => nextReflection
	};
	registerPatchSession(ctx.parentSessionId, handle);

	try {
		const body = formatRepairMessage(result.errors ?? [], []);
		await injectRepairTurn(ctx.conversationId, body, {
			parentSessionId: ctx.parentSessionId,
			failureCode: repairable.code,
			blockIndex: repairable.blockIndex ?? 0,
			reflection: nextReflection
		});

		if (cancelled) return;

		// Retrigger the stream via the caller-supplied callback. The
		// chatStore implementation creates a fresh assistant message
		// parented to the just-injected repair turn and runs
		// `streamChatCompletion`. We can't re-enter `sendMessage` here —
		// it guard-returns on empty content — so the callback owns the
		// retrigger path.
		await ctx.runAssistantTurn();
	} finally {
		unregisterPatchSession(ctx.parentSessionId);
	}
}

function defaultExhaustedToast(message: string): void {
	console.warn('[ai-patch]', message);
}
