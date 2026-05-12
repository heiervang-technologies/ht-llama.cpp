/**
 * session-registry — SvelteMap-backed reactivity tests.
 *
 * Commit 4d swaps the plain `Map<string, SessionHandle>` for a
 * `SvelteMap` so callers that read via `.get(id)` inside a `$derived` or
 * `$effect` pick up register/unregister transitions without a parent
 * re-render. The × discard affordance on a `patch-repair` card uses this
 * to disappear the moment the loop commits / exhausts / aborts.
 *
 * Running a full `$effect` harness from a node-env unit test is fragile
 * — the svelte compiler may not process a plain `.ts` file the same way
 * it does a `.svelte` module. We therefore test the behaviour via the
 * public API (Map semantics still hold) and verify structurally that the
 * backing store is a `SvelteMap` by spying on its prototype. If a
 * refactor reverts to a plain `Map`, the spy assertions trip. End-to-end
 * reactivity is exercised by the Storybook / browser-env UI tests.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { SvelteMap } from 'svelte/reactivity';
import {
	__resetSessionRegistryForTest,
	getPatchSession,
	registerPatchSession,
	unregisterPatchSession,
	type SessionHandle
} from '$lib/editor/ai-patch/session-registry';

function makeHandle(status: 'in-flight' | 'exhausted'): SessionHandle {
	let exhausted = status === 'exhausted';
	return {
		abort: () => {},
		markExhausted: () => {
			exhausted = true;
		},
		reflectionCount: () => (exhausted ? 3 : 1)
	};
}

beforeEach(() => {
	__resetSessionRegistryForTest();
});

describe('session-registry — plain Map semantics still hold', () => {
	it('register / get / unregister round-trip', () => {
		registerPatchSession('a', makeHandle('in-flight'));
		registerPatchSession('b', makeHandle('in-flight'));
		expect(getPatchSession('a')).not.toBeNull();
		expect(getPatchSession('b')).not.toBeNull();
		expect(getPatchSession('missing')).toBeNull();

		unregisterPatchSession('a');
		expect(getPatchSession('a')).toBeNull();
		expect(getPatchSession('b')).not.toBeNull();
	});

	it('register upserts — second call replaces the first handle', () => {
		const first = makeHandle('in-flight');
		const second = makeHandle('exhausted');
		registerPatchSession('c', first);
		registerPatchSession('c', second);
		expect(getPatchSession('c')?.reflectionCount()).toBe(3);
	});

	it('__resetSessionRegistryForTest clears the entire map', () => {
		registerPatchSession('x', makeHandle('in-flight'));
		registerPatchSession('y', makeHandle('in-flight'));
		__resetSessionRegistryForTest();
		expect(getPatchSession('x')).toBeNull();
		expect(getPatchSession('y')).toBeNull();
	});
});

describe('session-registry — SvelteMap is the backing store', () => {
	it('getPatchSession reads through SvelteMap.prototype.get', () => {
		const spy = vi.spyOn(SvelteMap.prototype, 'get');
		try {
			registerPatchSession('k', makeHandle('in-flight'));
			getPatchSession('k');
			// Filter for the call that actually queried key 'k' — other
			// SvelteMap instances elsewhere in the process may share the
			// prototype.
			expect(spy.mock.calls.some((args) => args[0] === 'k')).toBe(true);
		} finally {
			spy.mockRestore();
		}
	});

	it('registerPatchSession writes through SvelteMap.prototype.set', () => {
		const spy = vi.spyOn(SvelteMap.prototype, 'set');
		try {
			registerPatchSession('kk', makeHandle('in-flight'));
			expect(spy.mock.calls.some((args) => args[0] === 'kk')).toBe(true);
		} finally {
			spy.mockRestore();
		}
	});

	it('unregisterPatchSession writes through SvelteMap.prototype.delete', () => {
		registerPatchSession('kkk', makeHandle('in-flight'));
		const spy = vi.spyOn(SvelteMap.prototype, 'delete');
		try {
			unregisterPatchSession('kkk');
			expect(spy.mock.calls.some((args) => args[0] === 'kkk')).toBe(true);
		} finally {
			spy.mockRestore();
		}
	});
});
