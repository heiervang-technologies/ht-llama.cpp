/**
 * Inline-target → artifact-handle upgrade (design brief §7.1).
 *
 * First commit on an `inline` target materialises the autocapture slot
 * into a real gallery artifact via
 * `artifactGalleryStore.captureFromChatForPatch`. The dispatcher then
 * flips the session's `target` in place to the artifact handle so a
 * follow-up run (the repair loop, a re-stream) sees a stable id.
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
}

const fixtures = {
	artifacts: new Map<string, Artifact>(),
	revisions: new Map<string, Revision[]>(),
	captureCalls: [] as Array<{
		conversationId: string;
		slot: string;
		payload: Record<string, unknown>;
	}>
};
function reset() {
	fixtures.artifacts.clear();
	fixtures.revisions.clear();
	fixtures.captureCalls = [];
}

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		getArtifact: async (id: string) => fixtures.artifacts.get(id),
		listArtifactRevisions: async (id: string) => fixtures.revisions.get(id) ?? [],
		getDoc: async () => undefined,
		findArtifactBySlot: async (conversationId: string, slot: string) => {
			for (const art of fixtures.artifacts.values()) {
				// The mock threads source via `captureFromChatForPatch` below.
				if (
					(art as unknown as Record<string, string>).sourceConversationId === conversationId &&
					(art as unknown as Record<string, string>).sourceMessageSlot === slot
				) {
					return art;
				}
			}
			return undefined;
		}
	}
}));

vi.mock('$lib/stores/artifact-gallery.svelte', () => ({
	artifactGalleryStore: {
		captureFromChatForPatch: async (
			source: { conversationId: string; slot: string; messageId?: string },
			payload: Record<string, unknown>
		) => {
			fixtures.captureCalls.push({
				conversationId: source.conversationId,
				slot: source.slot,
				payload
			});
			const artifactId = 'art-new-1';
			const revisionId = 'rev-new-1';
			const art: Artifact = {
				id: artifactId,
				title: payload.title as string,
				kind: payload.kind as string,
				currentRevisionId: revisionId
			};
			// Stash the slot coordinates on the artifact so findArtifactBySlot
			// can find it on the subsequent resolveArtifact call the
			// dispatcher makes during upgrade.
			(art as unknown as Record<string, string>).sourceConversationId = source.conversationId;
			(art as unknown as Record<string, string>).sourceMessageSlot = source.slot;
			fixtures.artifacts.set(artifactId, art);
			fixtures.revisions.set(artifactId, [
				{
					id: revisionId,
					artifactId,
					text: payload.text as string,
					mimeType: payload.mimeType as string,
					reason: 'initial'
				}
			]);
			return { artifactId, revisionId };
		},
		// Unused in this test path, kept to satisfy the dispatcher import.
		addUserEditRevision: async () => ({ id: 'unused' })
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

import { PatchSession, resolveTarget } from '$lib/editor/ai-patch';

beforeEach(reset);

function block(search: string, replace: string): string {
	return `<<<<<<< SEARCH\n${search}\n=======\n${replace}\n>>>>>>> REPLACE\n`;
}

describe('inline → artifact upgrade', () => {
	it('first commit materialises the slot via captureFromChatForPatch and flips the session target', async () => {
		const target = await resolveTarget({
			kind: 'inline',
			conversationId: 'conv-1',
			parentMessageId: 'msg-42',
			artifactIndex: 0,
			seed: {
				kind: 'code',
				title: 'Example',
				mimeType: 'text/plain',
				baseText: 'alpha\nbeta\ngamma'
			}
		});
		expect(target.kind).toBe('inline');
		expect(target.baseText).toBe('alpha\nbeta\ngamma');

		const session = new PatchSession(target, {
			messageId: 'msg-42',
			modelId: 'test-model'
		});
		session.feed(block('alpha', 'ALPHA'));
		const result = await session.end();

		// captureFromChatForPatch fired once with the expected slot.
		expect(fixtures.captureCalls).toHaveLength(1);
		expect(fixtures.captureCalls[0]).toMatchObject({
			conversationId: 'conv-1',
			slot: 'msg-42#0'
		});
		expect(fixtures.captureCalls[0].payload.text).toBe('ALPHA\nbeta\ngamma');

		// Commit result carries the freshly-materialised ids.
		expect(result.newArtifactId).toBe('art-new-1');
		expect(result.revisionId).toBe('rev-new-1');

		// The session's target is now an artifact handle, not inline — a
		// follow-up run (repair loop, re-stream) will route through the
		// stable artifact path.
		expect(session.target.kind).toBe('artifact');
		expect(session.target.parentRevisionId).toBe('rev-new-1');
	});

	it('resolveTarget short-circuits to the artifact path when the slot is already materialised', async () => {
		// Seed a pre-existing artifact for the slot.
		fixtures.artifacts.set('art-existing', {
			id: 'art-existing',
			title: 'pre',
			kind: 'code',
			currentRevisionId: 'rev-existing'
		} as Artifact);
		(
			fixtures.artifacts.get('art-existing') as unknown as Record<string, string>
		).sourceConversationId = 'conv-2';
		(
			fixtures.artifacts.get('art-existing') as unknown as Record<string, string>
		).sourceMessageSlot = 'msg-7#0';
		fixtures.revisions.set('art-existing', [
			{
				id: 'rev-existing',
				artifactId: 'art-existing',
				text: 'prior body',
				mimeType: 'text/plain',
				reason: 'initial'
			}
		]);

		const target = await resolveTarget({
			kind: 'inline',
			conversationId: 'conv-2',
			parentMessageId: 'msg-7',
			artifactIndex: 0,
			seed: {
				kind: 'code',
				title: 'ignored',
				mimeType: 'text/plain',
				baseText: 'ignored'
			}
		});
		// Already-materialised slot resolves as the artifact, baseText and
		// parent pinned from the persisted revision.
		expect(target.kind).toBe('artifact');
		expect(target.baseText).toBe('prior body');
		expect(target.parentRevisionId).toBe('rev-existing');
	});
});
