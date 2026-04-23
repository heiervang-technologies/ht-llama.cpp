/**
 * End-to-end driver: a streaming patch session feeding a fake chat stream
 * into a fake CM6 `EditorView`, asserting the visible-effect contract
 * from the design brief §4.3.
 *
 * We cannot instantiate a real `EditorView` in the unit-test environment
 * (node, no DOM, no jsdom per the "no new deps" rule), so we stand up a
 * minimal `FakeEditorView` that forwards `dispatch` into `EditorState`
 * updates. The bridge only relies on `.state`, `.dispatch`, and the CM6
 * state layer — all of which work headlessly against a real
 * `EditorState` from `@codemirror/state`.
 *
 * Contract under test:
 *
 *   - During streaming: the `patchStateField` carries a non-empty
 *     DecorationSet (the in-flight widget), and the real document's
 *     string is unchanged.
 *   - At block close: exactly ONE "real" change transaction lands with
 *     `userEvent === 'input.type.ai'`; the doc matches the expected
 *     post-patch content.
 *   - Undo: one step of history reverts the entire patch, not per-token.
 *
 * Because we run headlessly, we assert on `EditorState` contents (doc
 * length, slice, decoration count) rather than on rendered DOM. The
 * browser-side visual confirmation is out of scope for unit tests.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { EditorState, Transaction, type Extension, type TransactionSpec } from '@codemirror/state';
import { history, undo } from '@codemirror/commands';

/* ------------------------------------------------------------------------- */
/* Fixtures — the dispatcher speaks to these via module mocks                */
/* ------------------------------------------------------------------------- */

interface Doc {
	id: string;
	content: string;
}
const fixtures = {
	docs: new Map<string, Doc>(),
	updateContentCalls: [] as Array<{ id: string; content: string }>,
	activeView: null as null | { getEditorView(): unknown }
};
function reset() {
	fixtures.docs.clear();
	fixtures.updateContentCalls = [];
	fixtures.activeView = null;
}

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		getArtifact: async () => undefined,
		listArtifactRevisions: async () => [],
		getDoc: async (id: string) => fixtures.docs.get(id),
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
		updateContent: async (id: string, content: string) => {
			fixtures.updateContentCalls.push({ id, content });
			const doc = fixtures.docs.get(id);
			if (doc) doc.content = content;
		},
		getActiveView: (id: string) => (fixtures.docs.has(id) ? fixtures.activeView : undefined),
		registerActiveView: () => {},
		unregisterActiveView: () => {}
	}
}));

import { PatchSession, patchStateField, resolveTarget } from '$lib/editor/ai-patch';

beforeEach(reset);

/* ------------------------------------------------------------------------- */
/* FakeEditorView — headless CM6 state driver                                */
/* ------------------------------------------------------------------------- */

/**
 * Minimal stand-in for `EditorView`. We only implement the two things the
 * bridge calls into: `.state` (current EditorState) and `.dispatch(spec)`.
 * The real `EditorView` composes spec → transaction → apply — we mirror
 * that via `EditorState.update`, and keep a flat `transactionLog` for
 * assertions.
 */
class FakeEditorView {
	state: EditorState;
	readonly transactions: Transaction[] = [];

	constructor(doc: string, extensions: Extension[] = []) {
		this.state = EditorState.create({
			doc,
			extensions: [history(), patchStateField, ...extensions]
		});
	}

	dispatch(...specs: TransactionSpec[]) {
		const tr = this.state.update(...specs);
		this.transactions.push(tr);
		this.state = tr.state;
	}
}

/* ------------------------------------------------------------------------- */

function block(search: string, replace: string): string {
	return `<<<<<<< SEARCH\n${search}\n=======\n${replace}\n>>>>>>> REPLACE\n`;
}

function decorationCount(state: EditorState): number {
	const decorations = state.field(patchStateField, false);
	if (!decorations) return 0;
	let n = 0;
	decorations.between(0, state.doc.length, () => {
		n += 1;
	});
	return n;
}

describe('end-to-end: streaming patch into a headless FakeEditorView', () => {
	it('streams through the widget layer, lands one real transaction, and is reversible in one undo step', async () => {
		const initial = '# Hello\n\nWorld body.\n';
		fixtures.docs.set('doc-1', { id: 'doc-1', content: initial });

		const fakeView = new FakeEditorView(initial);
		fixtures.activeView = {
			getEditorView: () => fakeView
		};

		const target = await resolveTarget({ kind: 'doc', docId: 'doc-1' });
		// Sanity — the dispatcher threaded the live view through the
		// registry into the resolved target.
		expect(target.viewRef).toBe(fakeView);

		const session = new PatchSession(target, { messageId: 'msg-1', modelId: 'm' });
		const expected = initial.replace('World body.', 'Earth body!');

		// Feed the block in two halves so the mid-stream assertion below
		// is meaningful — the shadow sees chunks before the close marker.
		const patch = block('World body.', 'Earth body!');
		const mid = Math.floor(patch.length * 0.6);
		session.feed(patch.slice(0, mid));

		// Mid-stream contract: the real doc is unchanged, and the
		// decoration layer is non-empty (widget is rendering shadow text).
		expect(fakeView.state.doc.toString()).toBe(initial);
		expect(decorationCount(fakeView.state)).toBeGreaterThan(0);

		// Finish feeding and close the session.
		session.feed(patch.slice(mid));
		const result = await session.end();

		// Post-close: real doc matches expected, decoration layer is empty.
		expect(fakeView.state.doc.toString()).toBe(expected);
		expect(decorationCount(fakeView.state)).toBe(0);

		// The commit transaction lives on the transaction log. Identify it
		// by the userEvent — only the bridge's real `commit()` sets
		// `input.type.ai`. Widget set/clear effects have no userEvent.
		const inputTypeAi = fakeView.transactions.filter((tr) => tr.isUserEvent('input.type.ai'));
		expect(inputTypeAi).toHaveLength(1);
		// And that transaction carries the net change (anchor range → final text).
		expect(inputTypeAi[0].changes.empty).toBe(false);

		// The dispatcher's commit closure also persists via docsStore — the
		// CM6 transaction handles the visible edit, `updateContent` keeps
		// IndexedDB in sync.
		expect(fixtures.updateContentCalls).toEqual([{ id: 'doc-1', content: expected }]);
		expect(result.docId).toBe('doc-1');

		// One-step undo reverts the whole patch, not per-token. The bridge
		// emits the commit as a single `input.type.ai` transaction with
		// default `addToHistory`; history merges only same-userEvent
		// adjacent transactions, but we only push one so this is trivial.
		const undoTransaction = undo({
			state: fakeView.state,
			dispatch: (tr: Transaction) => {
				fakeView.transactions.push(tr);
				fakeView.state = tr.state;
			}
		});
		expect(undoTransaction).toBe(true);
		expect(fakeView.state.doc.toString()).toBe(initial);
	});

	it('falls back to headless updateContent when no DocEditor is mounted for the doc', async () => {
		fixtures.docs.set('doc-2', { id: 'doc-2', content: 'one two three' });
		// No `activeView` registered — dispatcher takes the fallback path.
		fixtures.activeView = null;

		const target = await resolveTarget({ kind: 'doc', docId: 'doc-2' });
		expect(target.viewRef).toBeUndefined();

		const session = new PatchSession(target, { messageId: 'msg-2', modelId: 'm' });
		session.feed(block('two', 'TWO'));
		const result = await session.end();

		expect(result.docId).toBe('doc-2');
		expect(fixtures.updateContentCalls).toEqual([{ id: 'doc-2', content: 'one TWO three' }]);
	});
});
