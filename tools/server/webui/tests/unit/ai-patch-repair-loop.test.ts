/**
 * RepairLoop state-machine tests.
 *
 * The loop is pure logic — no I/O, no Svelte — so we drive it with
 * hand-rolled failure records. Commit 4a wires only F2 (E_NO_MATCH);
 * these tests lock in that scope.
 */

import { describe, expect, it } from 'vitest';
import {
	RepairLoop,
	PatchFailureCode,
	type DispatcherFailure,
	type PatchErrorRecord
} from '$lib/editor/ai-patch';

function noMatchError(overrides: Partial<PatchErrorRecord> = {}): PatchErrorRecord {
	return {
		code: PatchFailureCode.E_NO_MATCH,
		reason: 'anchor none',
		blockIndex: 0,
		search: 'const target = 1;',
		similar: [{ from: 0, to: 17, similarity: 0.85, text: 'const target = 2;' }],
		...overrides
	};
}

function ambiguousError(): PatchErrorRecord {
	return {
		code: PatchFailureCode.E_AMBIGUOUS,
		reason: 'anchor ambiguous',
		blockIndex: 0
	};
}

function failure(errors: PatchErrorRecord[]): DispatcherFailure {
	return { errors, repairable: errors.some((e) => e.code === PatchFailureCode.E_NO_MATCH) };
}

describe('RepairLoop', () => {
	it('emits repair-pending on the first F2 failure and increments the counter', () => {
		const loop = new RepairLoop();
		const event = loop.consume(failure([noMatchError()]));
		expect(event.kind).toBe('repair-pending');
		if (event.kind !== 'repair-pending') throw new Error();
		expect(event.reflection).toBe(1);
		expect(event.failureCode).toBe(PatchFailureCode.E_NO_MATCH);
		expect(event.blockIndex).toBe(0);
		expect(loop.reflectionCount).toBe(1);
		expect(loop.remaining).toBe(2);
	});

	it('exhausts after 3 failed reflections and then marks itself done', () => {
		const loop = new RepairLoop();
		expect(loop.consume(failure([noMatchError()])).kind).toBe('repair-pending');
		expect(loop.consume(failure([noMatchError()])).kind).toBe('repair-pending');
		expect(loop.consume(failure([noMatchError()])).kind).toBe('repair-pending');
		const fourth = loop.consume(failure([noMatchError()]));
		expect(fourth.kind).toBe('repair-exhausted');
		if (fourth.kind !== 'repair-exhausted') throw new Error();
		expect(fourth.reflection).toBe(3);
		expect(fourth.toast).toContain('3 attempts');
		expect(loop.isDone).toBe(true);
		expect(() => loop.consume(failure([noMatchError()]))).toThrow(/already ended/);
	});

	it('honours an explicit repairable:false even with F2 errors present', () => {
		const loop = new RepairLoop();
		const event = loop.consume({
			errors: [noMatchError()],
			repairable: false
		});
		expect(event.kind).toBe('no-repair');
		expect(loop.isDone).toBe(true);
	});

	it('treats ambiguous (F3) as repairable now that 4b wires F3/F6/F7/F11/F14', () => {
		const loop = new RepairLoop();
		const event = loop.consume({
			errors: [
				{
					code: PatchFailureCode.E_AMBIGUOUS,
					reason: 'anchor ambiguous',
					blockIndex: 0,
					search: 'duplicate',
					similar: [
						{ from: 0, to: 9, similarity: 1, text: 'duplicate' },
						{ from: 50, to: 59, similarity: 1, text: 'duplicate' }
					],
					targetText: 'duplicate line 1\nother\nduplicate line 2'
				}
			]
		});
		expect(event.kind).toBe('repair-pending');
		expect(loop.reflectionCount).toBe(1);
	});

	it('ignores E_USER_EDIT — user takeover is not an auto-retry trigger', () => {
		const loop = new RepairLoop();
		const event = loop.consume({
			errors: [
				{
					code: PatchFailureCode.E_USER_EDIT,
					reason: 'user edit aborted block',
					blockIndex: 0
				}
			]
		});
		expect(event.kind).toBe('no-repair');
		expect(loop.reflectionCount).toBe(0);
	});

	it('picks the first repairable error when a session has mixed failure codes', () => {
		const loop = new RepairLoop();
		// With commit 4b, F3 is itself repairable — so when F3 comes first
		// the loop drives off it (block 0). Under 4a, F3 was ignored and
		// F2 at block 1 was picked; the test now reflects the broader set.
		const event = loop.consume(
			failure([ambiguousError(), noMatchError({ blockIndex: 1, search: 'second failing search' })])
		);
		expect(event.kind).toBe('repair-pending');
		if (event.kind !== 'repair-pending') throw new Error();
		expect(event.blockIndex).toBe(0);
		expect(event.failureCode).toBe(PatchFailureCode.E_AMBIGUOUS);
	});

	it('supports configurable reflection budget (smoke test for 4b)', () => {
		const loop = new RepairLoop(1);
		expect(loop.consume(failure([noMatchError()])).kind).toBe('repair-pending');
		const second = loop.consume(failure([noMatchError()]));
		expect(second.kind).toBe('repair-exhausted');
	});

	it('reset() restores the pre-consume state', () => {
		const loop = new RepairLoop();
		loop.consume(failure([noMatchError()]));
		loop.consume(failure([noMatchError()]));
		expect(loop.reflectionCount).toBe(2);
		loop.reset();
		expect(loop.reflectionCount).toBe(0);
		expect(loop.isDone).toBe(false);
		expect(loop.consume(failure([noMatchError()])).kind).toBe('repair-pending');
	});
});
