/**
 * F4 — user-edit abort listener.
 *
 * The listener watches CM6 transactions for non-`input.type.ai` changes
 * and either (a) aborts an in-flight block whose anchor range the edit
 * touched, or (b) remaps the anchors of still-active blocks so subsequent
 * streamed chunks land at the right offset.
 *
 * These tests drive `handleUserTransaction` directly against synthetic
 * `PatchAbortTarget` fakes — no live `EditorView`, no DOM — plus a
 * second group that exercises `PatchSession.abortBlock` end-to-end
 * against the dispatcher's own abort-target implementation.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ChangeSet } from '@codemirror/state';

/* ------------------------------------------------------------------------- */
/* Dispatcher fixtures (needed for PatchSession integration tests)           */
/* ------------------------------------------------------------------------- */

interface ArtifactFixture {
	id: string;
	currentRevisionId: string;
	kind?: string;
	title?: string;
}
interface RevisionFixture {
	id: string;
	artifactId: string;
	text: string;
	mimeType: string;
	reason: string;
}
const fixtures = {
	artifacts: new Map<string, ArtifactFixture>(),
	revisions: new Map<string, RevisionFixture[]>()
};
function resetFixtures() {
	fixtures.artifacts.clear();
	fixtures.revisions.clear();
}

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		getArtifact: async (id: string) => fixtures.artifacts.get(id),
		listArtifactRevisions: async (id: string) => fixtures.revisions.get(id) ?? [],
		getArtifactRevision: async () => undefined,
		getDoc: async () => undefined,
		findArtifactBySlot: async () => undefined,
		appendArtifactRevision: async () => ({})
	}
}));

vi.mock('$lib/stores/artifact-gallery.svelte', () => ({
	artifactGalleryStore: {
		addUserEditRevision: async () => ({ id: 'unused' }),
		captureFromChatForPatch: async () => ({ artifactId: 'unused', revisionId: 'unused' })
	}
}));

vi.mock('$lib/stores/docs.svelte', () => ({
	docsStore: {
		updateContent: async () => {},
		getActiveView: () => undefined,
		registerActiveView: () => {},
		unregisterActiveView: () => {}
	}
}));

import {
	PatchFailureCode,
	PatchSession,
	handleUserTransaction,
	resolveTarget,
	type InflightAnchor,
	type PatchAbortTarget
} from '$lib/editor/ai-patch';

beforeEach(() => resetFixtures());

/* ------------------------------------------------------------------------- */
/* Helpers                                                                   */
/* ------------------------------------------------------------------------- */

/**
 * Hand-rolled stand-in for `PatchSession`'s abort-target surface. Captures
 * every abort / remap call so the test can assert on them, and supports
 * point-mutating the tracked anchor array to model a session whose
 * coordinates shift under the listener's feet.
 */
interface TargetHarness extends PatchAbortTarget {
	readonly aborted: number[];
	readonly remapCalls: { n: number };
	readonly anchors: InflightAnchor[];
}

function makeTarget(initial: InflightAnchor[]): TargetHarness {
	const state = {
		anchors: [...initial],
		aborted: [] as number[],
		remapCalls: { n: 0 }
	};
	const harness: TargetHarness = {
		inflightAnchors() {
			return state.anchors.filter((a) => !state.aborted.includes(a.blockIndex));
		},
		abortBlock(blockIndex) {
			state.aborted.push(blockIndex);
		},
		remapAnchors(mapPos) {
			state.remapCalls.n += 1;
			for (const a of state.anchors) {
				if (state.aborted.includes(a.blockIndex)) continue;
				a.from = mapPos(a.from, 1);
				a.to = mapPos(a.to, 1);
			}
		},
		aborted: state.aborted,
		remapCalls: state.remapCalls,
		anchors: state.anchors
	};
	return harness;
}

function seedArtifact(id: string, text: string): void {
	fixtures.artifacts.set(id, { id, currentRevisionId: 'r', kind: 'code', title: 'X' });
	fixtures.revisions.set(id, [
		{ id: 'r', artifactId: id, text, mimeType: 'text/plain', reason: 'initial' }
	]);
}

/* ------------------------------------------------------------------------- */
/* handleUserTransaction (pure)                                              */
/* ------------------------------------------------------------------------- */

describe('handleUserTransaction — abort / remap decision', () => {
	it('(a) aborts only the block whose range overlaps the edit', () => {
		const target = makeTarget([
			{ blockIndex: 0, from: 10, to: 20 },
			{ blockIndex: 1, from: 30, to: 40 },
			{ blockIndex: 2, from: 50, to: 60 }
		]);
		// Edit inside block 1 (at [30..40)): replace two chars with one.
		const changes = ChangeSet.of({ from: 33, to: 35, insert: 'X' }, 100);
		handleUserTransaction(changes, target);
		expect(target.aborted).toEqual([1]);
		// Block 0 untouched (edit is after it); block 2 remapped back by 1
		// because the edit deleted 2 chars and inserted 1.
		const block0 = target.anchors.find((a) => a.blockIndex === 0)!;
		const block2 = target.anchors.find((a) => a.blockIndex === 2)!;
		expect(block0).toMatchObject({ from: 10, to: 20 });
		expect(block2).toMatchObject({ from: 49, to: 59 });
		expect(target.remapCalls.n).toBe(1);
	});

	it('(b) remaps anchors when the edit lies strictly before them', () => {
		const target = makeTarget([{ blockIndex: 0, from: 50, to: 60 }]);
		const changes = ChangeSet.of({ from: 20, to: 20, insert: 'hello' }, 100);
		handleUserTransaction(changes, target);
		expect(target.aborted).toEqual([]);
		expect(target.anchors[0]).toMatchObject({ blockIndex: 0, from: 55, to: 65 });
	});

	it('(c) a cross-boundary delete straddling the anchor aborts the block', () => {
		const target = makeTarget([
			{ blockIndex: 0, from: 10, to: 20 },
			{ blockIndex: 1, from: 40, to: 60 }
		]);
		// Delete 35..45 — starts before block 1's range, ends inside it.
		const changes = ChangeSet.of({ from: 35, to: 45, insert: '' }, 100);
		handleUserTransaction(changes, target);
		expect(target.aborted).toEqual([1]);
		// Block 0 is before the edit: unchanged.
		const block0 = target.anchors.find((a) => a.blockIndex === 0)!;
		expect(block0).toMatchObject({ from: 10, to: 20 });
	});

	it('inclusive boundary: an edit at the anchor edge counts as overlap', () => {
		const target = makeTarget([{ blockIndex: 0, from: 10, to: 20 }]);
		const changes = ChangeSet.of({ from: 20, to: 22, insert: 'X' }, 100);
		handleUserTransaction(changes, target);
		expect(target.aborted).toEqual([0]);
	});

	it('no-op when there are no in-flight anchors', () => {
		const target = makeTarget([]);
		const changes = ChangeSet.of({ from: 0, to: 0, insert: 'anything' }, 100);
		handleUserTransaction(changes, target);
		expect(target.aborted).toEqual([]);
		expect(target.remapCalls.n).toBe(0);
	});
});

/* ------------------------------------------------------------------------- */
/* PatchSession — integration                                                */
/* ------------------------------------------------------------------------- */

describe('PatchSession — F4 E_USER_EDIT integration', () => {
	it('records E_USER_EDIT on abortBlock and swallows subsequent REPLACE chunks', async () => {
		seedArtifact('art-1', 'alpha beta gamma');
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-1' });
		const session = new PatchSession(target, { messageId: 'm', modelId: 'm' });

		// Feed the parser up through the separator — anchor locks on beta.
		session.feed('<<<<<<< SEARCH\nbeta\n=======\n');
		const anchors = session.inflightAnchors();
		expect(anchors).toHaveLength(1);
		expect(anchors[0].blockIndex).toBe(0);

		// Simulate the F4 listener invoking abortBlock.
		session.abortBlock(0, 'E_USER_EDIT');
		expect(session.inflightAnchors()).toHaveLength(0);

		// REPLACE + close arrive but are swallowed: the block is marked failed.
		session.feed('BETA\n>>>>>>> REPLACE\n');
		const result = await session.end();

		const userEdit = session.errors.find((e) => e.code === PatchFailureCode.E_USER_EDIT);
		expect(userEdit).toBeDefined();
		expect(userEdit!.blockIndex).toBe(0);
		// Zero successful blocks → non-commit sentinel, but NOT repairable:
		// E_USER_EDIT is deliberately not in the repairable set.
		expect(result.committed).toBe(false);
		expect(result.repairable).toBe(false);
	});

	it('remapAnchors shifts tracked anchor coords by the provided mapPos', async () => {
		seedArtifact('art-2', 'abcdefghij');
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-2' });
		const session = new PatchSession(target, { messageId: 'm', modelId: 'm' });
		session.feed('<<<<<<< SEARCH\ncd\n=======\n');
		const before = session.inflightAnchors()[0];
		session.remapAnchors((pos) => pos + 7);
		const after = session.inflightAnchors()[0];
		expect(after.from).toBe(before.from + 7);
		expect(after.to).toBe(before.to + 7);
	});
});
