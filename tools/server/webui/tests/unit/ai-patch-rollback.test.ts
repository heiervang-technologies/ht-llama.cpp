/**
 * artifactGalleryStore.rollbackToRevision — creates a new `reason:
 * 'rollback'` revision that duplicates a prior revision's payload, points
 * `currentRevisionId` at it, and threads provenance via
 * `metadata.rolledBackFrom` / `rolledBackTo`.
 *
 * We mock `DatabaseService` only — the store calls into it and reads the
 * result back through `listArtifacts()`; every write ends up reflected
 * in the fake db.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

interface FakeArtifact {
	id: string;
	currentRevisionId: string;
	title: string;
	kind: 'code';
	tags: string[];
	createdAt: number;
	updatedAt: number;
}
interface FakeRevision {
	id: string;
	artifactId: string;
	revisionNumber: number;
	createdAt: number;
	reason: string;
	parentRevisionId?: string;
	contentHash: string;
	mimeType: string;
	text?: string;
	metadata?: Record<string, unknown>;
}

const fixtures = {
	artifacts: new Map<string, FakeArtifact>(),
	revisions: new Map<string, FakeRevision[]>(),
	idCounter: 0
};

function resetFixtures() {
	fixtures.artifacts.clear();
	fixtures.revisions.clear();
	fixtures.idCounter = 0;
}

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		listArtifacts: async () => [...fixtures.artifacts.values()],
		getArtifact: async (id: string) => fixtures.artifacts.get(id),
		listArtifactRevisions: async (id: string) =>
			(fixtures.revisions.get(id) ?? [])
				.slice()
				.sort((a, b) => a.revisionNumber - b.revisionNumber),
		getArtifactRevision: async (revId: string) => {
			for (const list of fixtures.revisions.values()) {
				const hit = list.find((r) => r.id === revId);
				if (hit) return hit;
			}
			return undefined;
		},
		appendArtifactRevision: async (
			artifactId: string,
			rev: Omit<FakeRevision, 'id' | 'artifactId' | 'revisionNumber' | 'createdAt'>
		) => {
			const art = fixtures.artifacts.get(artifactId);
			if (!art) throw new Error('not found');
			const list = fixtures.revisions.get(artifactId) ?? [];
			const last = list[list.length - 1];
			const stored: FakeRevision = {
				...rev,
				id: `rev-${++fixtures.idCounter}`,
				artifactId,
				revisionNumber: (last?.revisionNumber ?? 0) + 1,
				createdAt: Date.now()
			};
			list.push(stored);
			fixtures.revisions.set(artifactId, list);
			art.currentRevisionId = stored.id;
			art.updatedAt = Date.now();
			return stored;
		},
		updateArtifact: async () => {}
	}
}));

import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';

beforeEach(() => resetFixtures());

function seed(artifactId: string, revisions: Array<{ text: string; reason: string }>): string[] {
	fixtures.artifacts.set(artifactId, {
		id: artifactId,
		currentRevisionId: '',
		title: 'X',
		kind: 'code',
		tags: [],
		createdAt: 0,
		updatedAt: 0
	});
	const ids: string[] = [];
	for (let i = 0; i < revisions.length; i++) {
		const id = `rev-${artifactId}-${i}`;
		ids.push(id);
		const list = fixtures.revisions.get(artifactId) ?? [];
		list.push({
			id,
			artifactId,
			revisionNumber: i + 1,
			createdAt: i,
			reason: revisions[i].reason,
			contentHash: `h-${revisions[i].text}`,
			mimeType: 'text/plain',
			text: revisions[i].text
		});
		fixtures.revisions.set(artifactId, list);
	}
	fixtures.artifacts.get(artifactId)!.currentRevisionId = ids[ids.length - 1];
	return ids;
}

describe('artifactGalleryStore.rollbackToRevision', () => {
	it('appends a rollback revision with the target payload and rolledBack metadata', async () => {
		const [rev1, rev2, rev3] = seed('art-1', [
			{ text: 'first', reason: 'initial' },
			{ text: 'second', reason: 'edit' },
			{ text: 'third', reason: 'edit' }
		]);
		// Current tip is rev3. Roll back to rev1.
		const newRev = await artifactGalleryStore.rollbackToRevision('art-1', rev1);
		expect(newRev.reason).toBe('rollback');
		expect(newRev.text).toBe('first');
		expect(newRev.contentHash).toBe('h-first');
		expect(newRev.metadata).toMatchObject({
			rolledBackFrom: rev3,
			rolledBackTo: rev1
		});
		// Current tip advances to the new rollback revision.
		const art = fixtures.artifacts.get('art-1')!;
		expect(art.currentRevisionId).toBe(newRev.id);
		// parentRevisionId points at the prior tip (rev3), not the target.
		expect(newRev.parentRevisionId).toBe(rev3);
		void rev2; // silence unused-var warning
	});

	it('dedup — rolling back to current is a no-op that returns the existing tip', async () => {
		const [rev1, rev2] = seed('art-2', [
			{ text: 'alpha', reason: 'initial' },
			{ text: 'beta', reason: 'edit' }
		]);
		// Current tip is rev2. Roll back to rev2 → no new revision written.
		const returned = await artifactGalleryStore.rollbackToRevision('art-2', rev2);
		const list = fixtures.revisions.get('art-2')!;
		expect(list).toHaveLength(2);
		expect(returned.id).toBe(rev2);
		// Current pointer unchanged.
		expect(fixtures.artifacts.get('art-2')!.currentRevisionId).toBe(rev2);
		void rev1;
	});

	it('dedup — same content as tip under a different id returns the tip revision', async () => {
		// Revision 3 has the same contentHash as revision 1 (same text). A
		// rollback to rev1 would be byte-identical to the current tip — the
		// store short-circuits to avoid flooding the timeline.
		const [rev1] = seed('art-3', [
			{ text: 'alpha', reason: 'initial' },
			{ text: 'beta', reason: 'edit' },
			{ text: 'alpha', reason: 'edit' } // identical text → identical hash
		]);
		const list = fixtures.revisions.get('art-3')!;
		expect(list).toHaveLength(3);
		const currentTipId = fixtures.artifacts.get('art-3')!.currentRevisionId;
		const returned = await artifactGalleryStore.rollbackToRevision('art-3', rev1);
		// No new revision appended.
		expect(fixtures.revisions.get('art-3')!).toHaveLength(3);
		// Returned is the existing tip, not rev1.
		expect(returned.id).toBe(currentTipId);
	});

	it('throws when the artifact does not exist', async () => {
		await expect(artifactGalleryStore.rollbackToRevision('ghost', 'any-rev')).rejects.toThrow(
			/not found/
		);
	});

	it('throws when the target revision belongs to a different artifact', async () => {
		seed('art-a', [{ text: 'a', reason: 'initial' }]);
		seed('art-b', [{ text: 'b', reason: 'initial' }]);
		// rev of art-a doesn't belong to art-b.
		const revA = fixtures.revisions.get('art-a')![0].id;
		await expect(artifactGalleryStore.rollbackToRevision('art-b', revA)).rejects.toThrow(
			/not found/
		);
	});
});
