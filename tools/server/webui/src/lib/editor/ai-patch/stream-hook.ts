/**
 * Chat-stream adapter + loop orchestrator.
 *
 * Two layers live here:
 *
 * 1. `createPatchStreamHook(session)` — the original low-level adapter
 *    that turns a `PatchSession` into the three callbacks a ChatService
 *    stream caller already consumes. Useful for tests and for callers
 *    that want to drive a single session without the retry loop.
 *
 * 2. `runPatchRepairLoop({...})` — the high-level orchestrator that
 *    closes the loop:
 *
 *       - opens a fresh `PatchSession` per attempt,
 *       - hands the hook to the caller's `runStream` so it can plumb into
 *         whatever sendMessage wiring the host uses,
 *       - after each stream ends, inspects the `CommitResult`,
 *       - on `{repairable:true, reflectionCount<MAX_REFLECTIONS}`:
 *         formats + injects a synthetic `patch-repair` user turn, then
 *         re-invokes `runStream` with an incremented reflection counter,
 *       - on exhaustion: fires a toast and stops (committed blocks from
 *         earlier attempts stay on disk),
 *       - on committed success or non-repairable failure: returns the
 *         final result.
 *
 *    The orchestrator NEVER patches `chat.service.ts` — it only calls the
 *    caller-supplied `runStream`, so the chat service stays the single
 *    owner of the AbortController / sendMessage contract. Callers pass a
 *    factory for signals so we can plumb a cancel path through the
 *    session-registry (the discard × affordance calls `handle.abort()`).
 */

import type { CommitResult, PatchFailureCode } from './types';
import type { PatchSession } from './dispatcher';
import { formatRepairMessage, MAX_REFLECTIONS, REPAIRABLE_CODES } from './repair-loop';
import { injectRepairTurn } from './repair-injector';
import {
	registerPatchSession,
	unregisterPatchSession,
	type SessionHandle
} from './session-registry';

export interface PatchStreamHook {
	/** Forward one text delta from the chat stream. */
	onChunk: (text: string) => void;
	/** Flush + commit. Resolves with the commit result. */
	onFinish: () => Promise<CommitResult>;
	/** Tear down the session without committing. */
	onAbort: () => void;
}

/**
 * Create the set of callbacks a chat-stream caller can feed straight into
 * ChatService.sendMessage (or any equivalent observer-shaped API).
 */
export function createPatchStreamHook(session: PatchSession): PatchStreamHook {
	return {
		onChunk: (text: string) => {
			if (!text) return;
			session.feed(text);
		},
		onFinish: async () => {
			return await session.end();
		},
		onAbort: () => {
			session.abort();
		}
	};
}

/* ------------------------------------------------------------------------- */
/* Orchestrator                                                              */
/* ------------------------------------------------------------------------- */

export interface RunPatchRepairLoopOptions {
	/**
	 * Conversation hosting the originating assistant message. Used as the
	 * target for injected `patch-repair` user turns.
	 */
	conversationId: string;
	/**
	 * Id of the assistant message whose stream produced the original
	 * (failing) patch. All injected repair turns carry this as
	 * `metadata.source.parentSessionId` so the UI can group retries.
	 */
	parentSessionId: string;
	/**
	 * Factory — called once per attempt — that returns a freshly-opened
	 * `PatchSession` bound to the same target. Necessary because shadow
	 * state is single-use; we cannot reuse a session that has already
	 * been `end()`-ed.
	 */
	sessionFactory: (attempt: PatchSessionAttempt) => PatchSession | Promise<PatchSession>;
	/**
	 * Caller-owned bridge into the underlying streaming API. The
	 * orchestrator invokes this per attempt with a ready-to-use stream
	 * hook; the caller is responsible for plumbing the hook's three
	 * callbacks into `ChatService.sendMessage` (and providing the correct
	 * `messages` array for each retry — typically the active-messages list
	 * re-read from the store after the repair turn was injected).
	 *
	 * The orchestrator then calls `hook.onFinish()` itself; the caller
	 * MUST NOT invoke it. The caller SHOULD propagate any abort from the
	 * returned signal into its sendMessage call.
	 */
	runStream: (ctx: RunStreamContext) => Promise<void>;
	/** Optional override for the reflection budget. Defaults to MAX_REFLECTIONS. */
	maxReflections?: number;
	/**
	 * Optional toast emitter. When the loop exhausts its budget, the
	 * orchestrator calls this with a short message. Decoupled from
	 * `svelte-sonner` so tests can intercept without DOM plumbing.
	 */
	onExhausted?: (message: string) => void;
}

export interface PatchSessionAttempt {
	/** 0 on the initial attempt, 1 on the first retry, etc. */
	reflectionCount: number;
	/** Reason the previous attempt failed (absent on attempt 0). */
	previousFailureCode?: PatchFailureCode;
}

export interface RunStreamContext {
	hook: PatchStreamHook;
	/** Abort signal the caller MUST forward into ChatService.sendMessage. */
	signal: AbortSignal;
	reflectionCount: number;
}

/**
 * Drive a patch session through up to `maxReflections` retries. Returns
 * the final `CommitResult` — either the one from a successful commit, or
 * the last non-repairable / exhausted failure sentinel.
 *
 * See module header for the full contract.
 */
export async function runPatchRepairLoop(
	options: RunPatchRepairLoopOptions
): Promise<CommitResult> {
	const maxReflections = options.maxReflections ?? MAX_REFLECTIONS;

	let reflectionCount = 0;
	let exhaustedByDiscard = false;
	let controller: AbortController = new AbortController();
	let lastFailureCode: PatchFailureCode | undefined;

	const handle: SessionHandle = {
		abort: () => {
			controller.abort();
		},
		markExhausted: () => {
			exhaustedByDiscard = true;
		},
		reflectionCount: () => reflectionCount
	};
	registerPatchSession(options.parentSessionId, handle);

	try {
		for (;;) {
			const session = await options.sessionFactory({
				reflectionCount,
				previousFailureCode: lastFailureCode
			});
			const hook = createPatchStreamHook(session);
			controller = new AbortController();

			try {
				await options.runStream({
					hook,
					signal: controller.signal,
					reflectionCount
				});
			} catch (err) {
				// A stream-level abort (user cancel) ends the loop — we drop
				// the session and treat it as aborted. Genuine errors bubble
				// up; unwinding the registry happens in `finally`.
				if ((err as { name?: string })?.name === 'AbortError' || controller.signal.aborted) {
					hook.onAbort();
					return { committed: false, reason: 'aborted' };
				}
				throw err;
			}

			const result = await hook.onFinish();

			// Committed or non-repairable → we're done.
			if (!isFailedCommit(result)) return result;
			if (exhaustedByDiscard) return result;
			if (!result.repairable) return result;

			const repairable = (result.errors ?? []).find((e) => REPAIRABLE_CODES.has(e.code));
			if (!repairable) return result;

			if (reflectionCount >= maxReflections) {
				const msg = `Patch retries exhausted after ${maxReflections} attempts. Committed blocks kept.`;
				options.onExhausted?.(msg);
				return result;
			}

			// Inject the synthetic user turn and loop.
			const body = formatRepairMessage(result.errors ?? [], []);
			reflectionCount += 1;
			lastFailureCode = repairable.code;
			await injectRepairTurn(options.conversationId, body, {
				parentSessionId: options.parentSessionId,
				failureCode: repairable.code,
				blockIndex: repairable.blockIndex ?? 0,
				reflection: reflectionCount
			});
		}
	} finally {
		unregisterPatchSession(options.parentSessionId);
	}
}

function isFailedCommit(result: CommitResult): boolean {
	return result.committed === false;
}
