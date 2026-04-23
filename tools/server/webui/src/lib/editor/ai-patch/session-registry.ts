/**
 * Session registry — a live directory of in-flight patch sessions so the
 * discard-affordance on a repair card can find the controller to abort.
 *
 * Keyed by `parentSessionId` (the id of the assistant message whose stream
 * produced the original failing patch). The repair loop re-uses that id
 * across retries, so the × button on any repair card for the same parent
 * cancels the entire loop, not just the current retry.
 *
 * Reactivity
 * ----------
 * The backing store is a `SvelteMap` so callers reading via `.get(id)`
 * inside a `$derived` (or `$effect`) get tracked dependencies for free —
 * when the loop (un)registers a handle, any UI surface that asked "is
 * there still a live loop for this parent?" re-derives on the next tick
 * without a parent re-render. In particular `ChatMessageUser.svelte`
 * leans on this to hide the × discard affordance the moment a session
 * transitions from `in-flight` to `committed` / `exhausted` / `aborted`.
 *
 * Public API is unchanged from the pre-SvelteMap version; the reactivity
 * is a property of the read path, not a new method.
 */

import { SvelteMap } from 'svelte/reactivity';
import { MAX_REFLECTIONS } from './repair-loop';

export interface SessionHandle {
	/**
	 * Aborts the currently-running ChatService stream for this session.
	 * No-op if the stream is not running at call time.
	 */
	abort(): void;
	/**
	 * Saturate the reflection counter so the loop-self-closing check bails
	 * out on the next `repairable` outcome. Called from the × handler to
	 * prevent another retry from firing after the in-flight stream
	 * resolves.
	 */
	markExhausted(): void;
	/**
	 * Current reflection count — 0 on the initial session, 1 on the first
	 * retry, etc. Used by the UI to decide whether to render the discard
	 * affordance.
	 */
	reflectionCount(): number;
}

/**
 * SvelteMap of live session handles. Exported for callers that want to
 * iterate (telemetry, debug panels) — direct mutation is not supported;
 * use the `register*` / `unregister*` helpers. Reads on `.get(id)` are
 * reactive and `$derived`-friendly.
 */
const sessions = new SvelteMap<string, SessionHandle>();

/** Register (or replace) a handle for `parentSessionId`. */
export function registerPatchSession(parentSessionId: string, handle: SessionHandle): void {
	sessions.set(parentSessionId, handle);
}

/** Remove the handle — called on loop termination (committed / exhausted / aborted). */
export function unregisterPatchSession(parentSessionId: string): void {
	sessions.delete(parentSessionId);
}

/**
 * Return the handle for `parentSessionId`, or `null` if the loop has ended.
 *
 * Reactive: reads via `SvelteMap.get` are tracked, so calling this inside
 * `$derived` / `$effect` re-runs when the handle is (un)registered.
 */
export function getPatchSession(parentSessionId: string): SessionHandle | null {
	return sessions.get(parentSessionId) ?? null;
}

/**
 * Discard affordance entry point. Aborts the in-flight stream and marks
 * the handle as exhausted so no further retry can fire. Idempotent.
 */
export function stopPatchRepairLoop(parentSessionId: string): void {
	const handle = sessions.get(parentSessionId);
	if (!handle) return;
	handle.markExhausted();
	handle.abort();
}

/** Test helper — never call from app code. */
export function __resetSessionRegistryForTest(): void {
	sessions.clear();
}

/** Mirror of the loop's cap so UI code doesn't have to thread it. */
export { MAX_REFLECTIONS };
