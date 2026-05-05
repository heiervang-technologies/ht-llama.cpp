/**
 * artifactGalleryStore.addUserEditRevision dedup behaviour.
 *
 * The override path (`opts.parentRevisionId`) was added in commit 2 to let
 * the ai-patch dispatcher pin the session-start revision as the parent so
 * concurrent edits during streaming don't silently re-parent the new
 * revision. In commit 3 we close the dedup hole that override path left
 * open:
 *
 *   - Tip-match + parent-omitted  → return existing (pre-existing behaviour)
 *   - Tip-match + parent=tip      → return existing (the new override path)
 *   - Tip-match + parent=OLDER    → append (divergent branch that happens
 *                                   to match the tip text)
 *   - Content-differs             → append (always)
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

interface Revision {
	id: string;
	artifactId: string;
	revisionNumber: number;
	contentHash: string;
	parentRevisionId?: string;
	text?: string;
	mimeType: string;
	reason: string;
}
interface Artifact {
	id: string;
	currentRevisionId: string;
}

const fixtures = {
	artifacts: new Map<string, Artifact>(),
	revisions: new Map<string, Revision[]>(),
	appendCalls: 0
};
function reset() {
	fixtures.artifacts.clear();
	fixtures.revisions.clear();
	fixtures.appendCalls = 0;
}

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		listArtifacts: async () => Array.from(fixtures.artifacts.values()),
		listArtifactRevisions: async (id: string) => fixtures.revisions.get(id) ?? [],
		appendArtifactRevision: async (
			artifactId: string,
			rev: Omit<Revision, 'id' | 'artifactId' | 'revisionNumber'>
		) => {
			fixtures.appendCalls += 1;
			const list = fixtures.revisions.get(artifactId) ?? [];
			const id = `rev-${list.length + 1}`;
			const full: Revision = {
				...rev,
				id,
				artifactId,
				revisionNumber: list.length + 1
			};
			list.push(full);
			fixtures.revisions.set(artifactId, list);
			const art = fixtures.artifacts.get(artifactId);
			if (art) art.currentRevisionId = id;
			return full;
		},
		updateArtifact: async () => {}
	}
}));

/* hashString: content-addressable dedup — same text → same hash. */
vi.mock('$lib/utils/artifacts', () => ({
	hashString: (s: string) => `h:${s}`
}));

import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';

beforeEach(reset);

function seed(id: string, text: string, hash = `h:${text}`) {
	fixtures.artifacts.set(id, { id, currentRevisionId: 'rev-1' });
	fixtures.revisions.set(id, [
		{
			id: 'rev-1',
			artifactId: id,
			revisionNumber: 1,
			contentHash: hash,
			text,
			mimeType: 'text/plain',
			reason: 'initial'
		}
	]);
}

describe('addUserEditRevision — dedup matrix', () => {
	it('no-override + tip-match → returns existing, no append', async () => {
		seed('art-1', 'hello');
		const rev = await artifactGalleryStore.addUserEditRevision('art-1', {
			kind: 'code',
			title: 't',
			mimeType: 'text/plain',
			text: 'hello'
		});
		expect(fixtures.appendCalls).toBe(0);
		expect(rev.id).toBe('rev-1');
	});

	it('override-pinned-to-tip + tip-match → returns existing, no append (NEW in commit 3)', async () => {
		seed('art-2', 'hello');
		const rev = await artifactGalleryStore.addUserEditRevision(
			'art-2',
			{ kind: 'code', title: 't', mimeType: 'text/plain', text: 'hello' },
			{ parentRevisionId: 'rev-1' }
		);
		expect(fixtures.appendCalls).toBe(0);
		expect(rev.id).toBe('rev-1');
	});

	it('override-pinned-to-tip + content-differs → appends normally', async () => {
		seed('art-3', 'hello');
		const rev = await artifactGalleryStore.addUserEditRevision(
			'art-3',
			{ kind: 'code', title: 't', mimeType: 'text/plain', text: 'hello, world' },
			{ parentRevisionId: 'rev-1' }
		);
		expect(fixtures.appendCalls).toBe(1);
		expect(rev.revisionNumber).toBe(2);
		expect(rev.parentRevisionId).toBe('rev-1');
	});

	it('override-pinned-to-OLDER + tip-match → still appends (divergent branch that happens to match the tip)', async () => {
		// Seed two revisions so we have an "older" one to pin to.
		seed('art-4', 'hello');
		fixtures.revisions.get('art-4')!.push({
			id: 'rev-2',
			artifactId: 'art-4',
			revisionNumber: 2,
			contentHash: 'h:hello-v2',
			text: 'hello-v2',
			mimeType: 'text/plain',
			reason: 'edit'
		});
		fixtures.artifacts.get('art-4')!.currentRevisionId = 'rev-2';

		// Now the caller pins to rev-1 (the OLDER one) but submits content
		// whose hash collides with the current tip's... wait, we need to
		// pin to rev-1 (hash h:hello) while submitting new content that
		// matches rev-2 (h:hello-v2). That's the "divergent branch matches
		// tip" case. Append should still fire.
		const rev = await artifactGalleryStore.addUserEditRevision(
			'art-4',
			{ kind: 'code', title: 't', mimeType: 'text/plain', text: 'hello-v2' },
			{ parentRevisionId: 'rev-1' }
		);
		expect(fixtures.appendCalls).toBe(1);
		expect(rev.parentRevisionId).toBe('rev-1');
	});
});
