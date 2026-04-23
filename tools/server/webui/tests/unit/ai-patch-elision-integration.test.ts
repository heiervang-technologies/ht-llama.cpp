/**
 * Elision integration — F6 wired into the dispatcher event loop.
 *
 * The unit test for the detector itself (`ai-patch-elision.test.ts`)
 * exercises the regex ladder. This file checks that the dispatcher
 * *actually rejects* a block whose REPLACE payload contains a lazy
 * elision marker, and that subsequent clean blocks still apply against
 * the un-corrupted shadow buffer.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

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

const fixtures = {
	artifacts: new Map<string, Artifact>(),
	revisions: new Map<string, Revision[]>(),
	editCalls: [] as Array<{
		artifactId: string;
		payload: Record<string, unknown>;
		opts?: { parentRevisionId?: string };
	}>
};

function resetFixtures() {
	fixtures.artifacts.clear();
	fixtures.revisions.clear();
	fixtures.editCalls = [];
}

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		getArtifact: async (id: string) => fixtures.artifacts.get(id),
		listArtifactRevisions: async (id: string) => fixtures.revisions.get(id) ?? [],
		getDoc: async () => undefined,
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
			return { id: 'rev-new', artifactId } as unknown as Revision;
		}
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

import { PatchFailureCode, PatchSession, resolveTarget } from '$lib/editor/ai-patch';

beforeEach(() => resetFixtures());

function seed(id: string, text: string) {
	fixtures.artifacts.set(id, {
		id,
		title: 't',
		kind: 'code',
		currentRevisionId: 'rev-0'
	});
	fixtures.revisions.set(id, [
		{ id: 'rev-0', artifactId: id, text, mimeType: 'text/plain', reason: 'initial' }
	]);
}

function block(search: string, replace: string): string {
	return `<<<<<<< SEARCH\n${search}\n=======\n${replace}\n>>>>>>> REPLACE\n`;
}

describe('dispatcher — F6 elision at block close', () => {
	it('rejects a block whose REPLACE contains a bare ellipsis line, leaves the buffer untouched for that block', async () => {
		const initial = 'alpha\nbeta\ngamma\n';
		seed('art-1', initial);
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-1' });
		const session = new PatchSession(target, { messageId: 'msg-1', modelId: 'm' });

		session.feed(block('alpha\nbeta\ngamma', 'ALPHA\n...\nGAMMA'));
		const result = await session.end();

		expect(session.errors.some((e) => e.code === PatchFailureCode.E_ELISION)).toBe(true);
		// No successful blocks → short-circuit → no commit.
		expect(result.committed).toBe(false);
		expect(result.reason).toBe('no-blocks');
		expect(fixtures.editCalls).toHaveLength(0);
		// The shadow buffer is rewound to the pre-anchor state so subsequent
		// blocks (if any) have a clean baseline.
		expect(session.finalText).toBe(initial);
	});

	it('flags a "// ... rest unchanged" placeholder comment', async () => {
		const initial = 'one\ntwo\nthree\n';
		seed('art-2', initial);
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-2' });
		const session = new PatchSession(target, { messageId: 'msg-2', modelId: 'm' });

		session.feed(block('one\ntwo\nthree', 'ONE\n// ... rest unchanged\nTHREE'));
		await session.end();

		expect(session.errors.some((e) => e.code === PatchFailureCode.E_ELISION)).toBe(true);
	});

	it('subsequent clean blocks still apply after an elision rejection', async () => {
		const initial = 'one\ntwo\nthree\n';
		seed('art-3', initial);
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-3' });
		const session = new PatchSession(target, { messageId: 'msg-3', modelId: 'm' });

		// Block A: elision — rejected.
		// Block B: clean — applies.
		const stream = block('one', 'ONE\n// ... existing code') + block('three', 'THREE');
		session.feed(stream);
		const result = await session.end();

		// One rejected, one successful → commit runs with the partial result.
		expect(session.errors.some((e) => e.code === PatchFailureCode.E_ELISION)).toBe(true);
		expect(session.blocks.filter((b) => b.ok)).toHaveLength(1);
		expect(result.committed).not.toBe(false);
		expect(session.finalText).toBe('one\ntwo\nTHREE\n');
	});

	it('a non-elision REPLACE body containing a legitimate ellipsis in prose is not flagged', async () => {
		const initial = 'hello world';
		seed('art-4', initial);
		const target = await resolveTarget({ kind: 'artifact', artifactId: 'art-4' });
		const session = new PatchSession(target, { messageId: 'msg-4', modelId: 'm' });

		// Ellipsis is inline inside actual prose, not its own line or comment.
		session.feed(block('hello world', 'say hello... and goodbye.'));
		await session.end();

		expect(session.errors.some((e) => e.code === PatchFailureCode.E_ELISION)).toBe(false);
		expect(session.finalText).toBe('say hello... and goodbye.');
	});
});
