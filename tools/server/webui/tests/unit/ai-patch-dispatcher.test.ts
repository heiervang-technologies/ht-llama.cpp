/**
 * Dispatcher round-trip tests.
 *
 * We don't pull in fake-indexeddb here — this project runs unit tests under
 * node, and Dexie + Svelte runes in the artifact store don't play nicely
 * in that environment. Instead we mock the two modules the dispatcher
 * talks to (`DatabaseService` and the two stores) so we can assert
 * precisely on how the dispatcher threads `parentRevisionId` and
 * metadata through the existing store API.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

/* ------------------------------------------------------------------------- */
/* Fixtures                                                                  */
/* ------------------------------------------------------------------------- */

interface Artifact {
	id: string;
	title: string;
	kind: string;
	currentRevisionId: string;
}

interface Revision {
	id: string;
	artifactId: string;
	text: string;
	mimeType: string;
	parentRevisionId?: string;
	reason: string;
	metadata?: unknown;
}

interface Doc {
	id: string;
	content: string;
}

/** In-memory store shared between the mocks below. */
const fixtures = {
	artifacts: new Map<string, Artifact>(),
	revisions: new Map<string, Revision[]>(),
	docs: new Map<string, Doc>(),
	/** Spy target: every addUserEditRevision call is recorded here. */
	editCalls: [] as Array<{
		artifactId: string;
		payload: Record<string, unknown>;
		opts?: { parentRevisionId?: string };
	}>,
	updateContentCalls: [] as Array<{ id: string; content: string }>,
	loadCalls: 0
};

function resetFixtures() {
	fixtures.artifacts.clear();
	fixtures.revisions.clear();
	fixtures.docs.clear();
	fixtures.editCalls = [];
	fixtures.updateContentCalls = [];
	fixtures.loadCalls = 0;
}

/* ------------------------------------------------------------------------- */
/* Mocks (must be declared before the module under test is imported)         */
/* ------------------------------------------------------------------------- */

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		getArtifact: async (id: string) => fixtures.artifacts.get(id),
		listArtifactRevisions: async (id: string) => fixtures.revisions.get(id) ?? [],
		getDoc: async (id: string) => fixtures.docs.get(id),
		findArtifactBySlot: async () => undefined,
		appendArtifactRevision: async (
			artifactId: string,
			rev: Omit<Revision, 'id' | 'artifactId'>
		) => {
			const id = `rev-${Math.random().toString(36).slice(2, 8)}`;
			const full: Revision = { ...rev, id, artifactId };
			const list = fixtures.revisions.get(artifactId) ?? [];
			list.push(full);
			fixtures.revisions.set(artifactId, list);
			const art = fixtures.artifacts.get(artifactId);
			if (art) art.currentRevisionId = id;
			return full;
		}
	}
}));

vi.mock('$lib/stores/artifact-gallery.svelte', () => ({
	artifactGalleryStore: {
		addUserEditRevision: async (
			artifactId: string,
			payload: Record<string, unknown>,
			opts?: { parentRevisionId?: string }
		) => {
			fixtures.editCalls.push({ artifactId, payload, opts });
			// Thread through to the mocked DatabaseService so the fixtures
			// reflect the append — mirrors the real store's behaviour.
			const { DatabaseService } = await import('$lib/services/database.service');
			let parentRevisionId = opts?.parentRevisionId;
			if (!parentRevisionId) {
				const revs = await DatabaseService.listArtifactRevisions(artifactId);
				parentRevisionId = revs.at(-1)?.id;
			}
			const rev = await DatabaseService.appendArtifactRevision(artifactId, {
				reason: 'edit',
				parentRevisionId,
				contentHash: 'mock-hash',
				mimeType: payload.mimeType as string,
				text: payload.text as string,
				metadata: payload.metadata as Record<string, unknown> | undefined
			} as any);
			fixtures.loadCalls += 1;
			return rev;
		}
	}
}));

vi.mock('$lib/stores/docs.svelte', () => ({
	docsStore: {
		updateContent: async (id: string, content: string) => {
			fixtures.updateContentCalls.push({ id, content });
			const doc = fixtures.docs.get(id);
			if (doc) doc.content = content;
		},
		// No live DocEditor in the unit environment — the dispatcher's
		// commit closure falls through to the headless updateContent path.
		getActiveView: () => undefined,
		registerActiveView: () => {},
		unregisterActiveView: () => {}
	}
}));

/* ------------------------------------------------------------------------- */
/* Test subject                                                              */
/* ------------------------------------------------------------------------- */

import {
	PatchSession,
	PatchFailureCode,
	createPatchStreamHook,
	resolveTarget
} from '$lib/editor/ai-patch';

beforeEach(() => {
	resetFixtures();
});

/* ------------------------------------------------------------------------- */
/* Helpers                                                                   */
/* ------------------------------------------------------------------------- */

function seedArtifact(opts: { id: string; text: string; kind?: string; title?: string }): {
	revisionId: string;
} {
	const revisionId = 'rev-initial';
	fixtures.artifacts.set(opts.id, {
		id: opts.id,
		title: opts.title ?? 'Untitled',
		kind: opts.kind ?? 'code',
		currentRevisionId: revisionId
	});
	fixtures.revisions.set(opts.id, [
		{
			id: revisionId,
			artifactId: opts.id,
			text: opts.text,
			mimeType: 'text/plain',
			reason: 'initial'
		}
	]);
	return { revisionId };
}

function buildBlock(search: string, replace: string): string {
	return ['<<<<<<< SEARCH', search, '=======', replace, '>>>>>>> REPLACE', ''].join('\n');
}

/* ------------------------------------------------------------------------- */
/* Tests                                                                     */
/* ------------------------------------------------------------------------- */

describe('resolveTarget — artifact', () => {
	it('snapshots the current revision as parentRevisionId and exposes baseText', async () => {
		const { revisionId } = seedArtifact({ id: 'art-1', text: 'hello world' });
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-1' });
		expect(target.kind).toBe('artifact');
		expect(target.baseText).toBe('hello world');
		expect(target.parentRevisionId).toBe(revisionId);
	});

	it('throws when the artifact does not exist', async () => {
		await expect(resolveTarget({ kind: 'artifact', artifactId: 'missing' })).rejects.toThrow(
			/not found/
		);
	});
});

describe('resolveTarget — doc', () => {
	it('returns the current content as baseText', async () => {
		fixtures.docs.set('doc-1', { id: 'doc-1', content: '# hi' });
		const target = await resolveTarget({ kind: 'doc', docId: 'doc-1' });
		expect(target.kind).toBe('doc');
		expect(target.baseText).toBe('# hi');
	});

	it('commit writes via docsStore.updateContent', async () => {
		fixtures.docs.set('doc-1', { id: 'doc-1', content: 'old' });
		const target = await resolveTarget({ kind: 'doc', docId: 'doc-1' });
		await target.commit('new content', {
			source: { kind: 'ai-patch', modelId: 'm', sessionId: 's' }
		});
		expect(fixtures.updateContentCalls).toEqual([{ id: 'doc-1', content: 'new content' }]);
	});
});

describe('resolveTarget — inline', () => {
	it('returns a resolved target whose commit materialises the slot (covered in inline-upgrade.test.ts)', async () => {
		const target = await resolveTarget({
			kind: 'inline',
			conversationId: 'conv-1',
			parentMessageId: 'msg-1',
			artifactIndex: 0,
			seed: {
				kind: 'code',
				title: 'Untitled',
				mimeType: 'text/plain',
				baseText: 'seed body'
			}
		});
		expect(target.kind).toBe('inline');
		expect(target.baseText).toBe('seed body');
	});
});

describe('PatchSession — end-to-end artifact round-trip', () => {
	it('parses → anchors → applies → commits a new revision with correct parent and metadata', async () => {
		const initial = 'const greeting = "hello";\nconsole.log(greeting);\n';
		const { revisionId: parentRevId } = seedArtifact({ id: 'art-1', text: initial });

		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-1' });
		const session = new PatchSession(target, {
			messageId: 'msg-42',
			modelId: 'test-model'
		});

		// Simulate the stream arriving in three chunks to exercise the streaming
		// parser path, not a single-shot feed.
		const block = buildBlock('"hello"', '"howdy"');
		const mid = Math.floor(block.length / 2);
		session.feed(block.slice(0, mid));
		session.feed(block.slice(mid, mid + 5));
		session.feed(block.slice(mid + 5));

		const result = await session.end();

		// Exactly one block, applied successfully.
		expect(session.blocks).toHaveLength(1);
		expect(session.blocks[0].ok).toBe(true);
		expect(session.errors).toEqual([]);

		// Shadow buffer reflects the edit.
		const expected = initial.replace('"hello"', '"howdy"');
		expect(session.finalText).toBe(expected);

		// The store received one call with the session-start parent pinned and
		// ai-patch source metadata threaded through.
		expect(fixtures.editCalls).toHaveLength(1);
		const call = fixtures.editCalls[0];
		expect(call.artifactId).toBe('art-1');
		expect(call.opts?.parentRevisionId).toBe(parentRevId);
		expect(call.payload.text).toBe(expected);
		expect(call.payload.mimeType).toBe('text/plain');
		expect(call.payload.metadata).toEqual({
			source: { kind: 'ai-patch', modelId: 'test-model', sessionId: 'msg-42' }
		});

		// A new revision landed and the returned commit result points at it.
		const revs = fixtures.revisions.get('art-1')!;
		expect(revs).toHaveLength(2);
		expect(revs[1].reason).toBe('edit');
		expect(revs[1].parentRevisionId).toBe(parentRevId);
		expect(result.revisionId).toBe(revs[1].id);
	});

	it('applies multiple blocks in order against the evolving buffer', async () => {
		const initial = 'alpha\nbeta\ngamma\n';
		seedArtifact({ id: 'art-m', text: initial });
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-m' });
		const session = new PatchSession(target, { messageId: 'msg-m', modelId: 'm' });

		const blockA = buildBlock('alpha', 'ALPHA');
		const blockB = buildBlock('gamma', 'GAMMA');
		session.feed(blockA + blockB);
		await session.end();

		expect(session.finalText).toBe('ALPHA\nbeta\nGAMMA\n');
		expect(session.blocks.every((b) => b.ok)).toBe(true);
	});

	it('records an E_NO_MATCH error when SEARCH does not anchor and short-circuits the commit', async () => {
		const initial = 'whatever body';
		seedArtifact({ id: 'art-x', text: initial });
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-x' });
		const session = new PatchSession(target, { messageId: 'msg-x', modelId: 'm' });

		const block = buildBlock('this text is nowhere in the buffer at all', 'replacement');
		session.feed(block);
		const result = await session.end();

		expect(session.errors.some((e) => e.code === PatchFailureCode.E_NO_MATCH)).toBe(true);
		// Failed block means no shadow mutation, so final text == initial.
		expect(session.finalText).toBe(initial);
		// Zero-successful-blocks short-circuit: no commit closure runs, no
		// revision lands, and the result carries the sentinel reason. The
		// repair loop (commit 4) consumes this signal.
		expect(fixtures.editCalls).toHaveLength(0);
		expect(result.committed).toBe(false);
		expect(result.reason).toBe('no-blocks');
	});

	it('logs a parse-error event on malformed marker grammar without throwing', async () => {
		const initial = 'some content';
		seedArtifact({ id: 'art-e', text: initial });
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-e' });
		const session = new PatchSession(target, { messageId: 'msg-e', modelId: 'm' });

		// Emit a bare separator in IDLE state — the parser flags this as F11.
		session.feed('=======\nuh oh\n');
		await session.end();

		expect(session.errors.some((e) => e.code === PatchFailureCode.E_MARKER_GRAMMAR)).toBe(true);
	});
});

describe('createPatchStreamHook — callback adapter', () => {
	it('onChunk drives the session; onFinish resolves with commit result', async () => {
		const initial = 'old value';
		seedArtifact({ id: 'art-h', text: initial });
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-h' });
		const session = new PatchSession(target, { messageId: 'msg-h', modelId: 'm' });
		const hook = createPatchStreamHook(session);

		const block = buildBlock('old', 'NEW');
		hook.onChunk(block);
		const result = await hook.onFinish();

		expect(session.finalText).toBe('NEW value');
		expect(result.revisionId).toBeDefined();
	});

	it('onAbort discards the session without committing', async () => {
		const initial = 'old value';
		seedArtifact({ id: 'art-a', text: initial });
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-a' });
		const session = new PatchSession(target, { messageId: 'msg-a', modelId: 'm' });
		const hook = createPatchStreamHook(session);

		hook.onChunk(buildBlock('old', 'NEW'));
		hook.onAbort();
		// After abort, end() should no-op / not produce another commit.
		expect(fixtures.editCalls).toHaveLength(0);
		// Subsequent feed is ignored (finished).
		hook.onChunk('anything');
		expect(fixtures.editCalls).toHaveLength(0);
	});
});
