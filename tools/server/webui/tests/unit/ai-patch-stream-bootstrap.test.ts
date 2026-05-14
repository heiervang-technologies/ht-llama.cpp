/**
 * Commit 5 — live-chat stream bootstrap end-to-end.
 *
 * Drives `createChatPatchBootstrap` with a fake-streamed assistant turn
 * containing a single SEARCH/REPLACE block against a known artifact, and
 * asserts the full chain:
 *
 *   stream chunks
 *       → sniffer parser detects block-open
 *       → async target resolution (currentArtifactId)
 *       → PatchSession opens + replays buffered chunks
 *       → session.end() commits
 *       → recordCompletedPatchSession stores the result under the message id
 *
 * The test does NOT exercise the repair-loop retrigger — that lives in
 * `ai-patch-loop-close.test.ts` and `ai-patch-chat-integration.test.ts`.
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
	reason: string;
	parentRevisionId?: string;
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

function reset() {
	fixtures.artifacts.clear();
	fixtures.revisions.clear();
	fixtures.editCalls = [];
}

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		findDocByName: async () => undefined,
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
			const { DatabaseService } = await import('$lib/services/database.service');
			let parentRevisionId = opts?.parentRevisionId;
			if (!parentRevisionId) {
				const revs = await DatabaseService.listArtifactRevisions(artifactId);
				parentRevisionId = revs.at(-1)?.id;
			}
			return await DatabaseService.appendArtifactRevision(artifactId, {
				reason: 'edit',
				parentRevisionId,
				contentHash: 'mock-hash',
				mimeType: payload.mimeType as string,
				text: payload.text as string,
				metadata: payload.metadata as Record<string, unknown> | undefined
			} as any);
		},
		captureFromChatForPatch: async () => ({
			artifactId: 'art-new',
			revisionId: 'rev-new'
		})
	}
}));

vi.mock('$lib/stores/docs.svelte', () => ({
	docsStore: {
		getActiveView: () => undefined,
		updateContent: async () => {}
	}
}));

import { createChatPatchBootstrap } from '$lib/editor/ai-patch/chat-bootstrap';
import {
	__resetChatIntegrationForTest,
	consumeCompletedPatchSession
} from '$lib/editor/ai-patch/chat-integration';
import { __resetSessionRegistryForTest } from '$lib/editor/ai-patch/session-registry';

function seedArtifact(opts: { id: string; text: string; kind?: string }): { revisionId: string } {
	const revisionId = 'rev-initial';
	fixtures.artifacts.set(opts.id, {
		id: opts.id,
		title: 'Untitled',
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

beforeEach(() => {
	reset();
	__resetChatIntegrationForTest();
	__resetSessionRegistryForTest();
});

describe('createChatPatchBootstrap — one SEARCH/REPLACE block against a known artifact', () => {
	it('opens a session lazily, applies the block, commits a revision, and records the result', async () => {
		seedArtifact({ id: 'art-1', text: 'hello world\n' });

		const bootstrap = createChatPatchBootstrap({
			messageId: 'asst-1',
			modelId: 'test-model',
			conversationId: 'conv-1',
			parentMessageId: 'user-1',
			getCurrentArtifactId: () => 'art-1'
		});

		const stream = [
			"Here's the edit you asked for:\n\n",
			'<<<<<<< SEARCH\n',
			'hello world\n',
			'=======\n',
			'hello there\n',
			'>>>>>>> REPLACE\n'
		];
		for (const c of stream) bootstrap.feed(c);
		await bootstrap.end();

		// Exactly one revision appended to the artifact.
		expect(fixtures.editCalls).toHaveLength(1);
		const call = fixtures.editCalls[0];
		expect(call.artifactId).toBe('art-1');
		expect(call.payload.text).toBe('hello there\n');
		// The parent-revision id was snapshotted at session-start.
		expect(call.opts?.parentRevisionId).toBe('rev-initial');

		// The CommitResult is stashed for the chat store to consume.
		const recorded = consumeCompletedPatchSession('asst-1');
		expect(recorded).not.toBeNull();
		expect(recorded?.committed).not.toBe(false);
		expect(recorded?.revisionId).toBeDefined();
	});

	it('records a repairable F2 (no-match) failure when SEARCH does not anchor', async () => {
		seedArtifact({ id: 'art-1', text: 'completely different text\n' });

		const bootstrap = createChatPatchBootstrap({
			messageId: 'asst-2',
			modelId: 'test-model',
			conversationId: 'conv-1',
			parentMessageId: 'user-1',
			getCurrentArtifactId: () => 'art-1'
		});

		const stream = [
			'<<<<<<< SEARCH\n',
			'this will not match anything\n',
			'=======\n',
			'replacement\n',
			'>>>>>>> REPLACE\n'
		];
		for (const c of stream) bootstrap.feed(c);
		await bootstrap.end();

		// No commit — session rejected the block.
		expect(fixtures.editCalls).toHaveLength(0);
		const recorded = consumeCompletedPatchSession('asst-2');
		expect(recorded?.committed).toBe(false);
		expect(recorded?.repairable).toBe(true);
		expect(recorded?.errors?.[0].code).toBe('E_NO_MATCH');
	});

	it('drops the block with E_NO_TARGET when a filename above the fence matches no doc', async () => {
		// No seeded artifact / doc at all.
		const toasts: string[] = [];
		const bootstrap = createChatPatchBootstrap({
			messageId: 'asst-3',
			modelId: 'test-model',
			conversationId: 'conv-1',
			parentMessageId: 'user-1',
			onNoTarget: (reason) => toasts.push(reason)
		});

		const stream = [
			'notes.md\n',
			'<<<<<<< SEARCH\n',
			'anything\n',
			'=======\n',
			'anything else\n',
			'>>>>>>> REPLACE\n'
		];
		for (const c of stream) bootstrap.feed(c);
		await bootstrap.end();

		expect(fixtures.editCalls).toHaveLength(0);
		expect(toasts).toHaveLength(1);
		expect(toasts[0]).toMatch(/notes\.md/);
		const recorded = consumeCompletedPatchSession('asst-3');
		expect(recorded?.committed).toBe(false);
		expect(recorded?.errors?.some((e) => e.code === 'E_NO_TARGET')).toBe(true);
		// E_NO_TARGET is not in REPAIRABLE_CODES — a second model pass
		// cannot invent a target. The failure must not be repairable.
		expect(recorded?.repairable).toBe(false);
	});
});
