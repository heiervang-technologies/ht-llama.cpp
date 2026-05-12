/**
 * Commit 5 — inert-stream contract.
 *
 * An assistant turn that produces no `<<<<<<< SEARCH` marker must leave
 * the ai-patch subsystem completely untouched:
 *
 *   - no `PatchSession` is created,
 *   - no DB writes,
 *   - no entries in the session registry,
 *   - no completed-session result recorded for the message,
 *   - no toasts emitted.
 *
 * We drive the bootstrap with a fake-streamed "plain assistant reply"
 * and assert each of those invariants.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

const fixtures = {
	findDocCalls: [] as string[],
	getArtifactCalls: [] as string[],
	captureCalls: [] as unknown[],
	toastCalls: [] as unknown[]
};

function reset() {
	fixtures.findDocCalls = [];
	fixtures.getArtifactCalls = [];
	fixtures.captureCalls = [];
	fixtures.toastCalls = [];
}

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		findDocByName: async (name: string) => {
			fixtures.findDocCalls.push(name);
			return undefined;
		},
		getArtifact: async (id: string) => {
			fixtures.getArtifactCalls.push(id);
			return undefined;
		},
		listArtifactRevisions: async () => [],
		getDoc: async () => undefined,
		findArtifactBySlot: async () => undefined
	}
}));

vi.mock('$lib/stores/artifact-gallery.svelte', () => ({
	artifactGalleryStore: {
		captureFromChatForPatch: async (source: unknown, payload: unknown) => {
			fixtures.captureCalls.push({ source, payload });
			return { artifactId: 'art-never-1', revisionId: 'rev-never-1' };
		},
		addUserEditRevision: async () => {
			throw new Error('should never be called for an inert stream');
		}
	}
}));

vi.mock('$lib/stores/docs.svelte', () => ({
	docsStore: {
		getActiveView: () => undefined,
		updateContent: async () => {
			throw new Error('should never be called for an inert stream');
		}
	}
}));

import { createChatPatchBootstrap } from '$lib/editor/ai-patch/chat-bootstrap';
import {
	__resetChatIntegrationForTest,
	consumeCompletedPatchSession
} from '$lib/editor/ai-patch/chat-integration';
import {
	__resetSessionRegistryForTest,
	getPatchSession
} from '$lib/editor/ai-patch/session-registry';

beforeEach(() => {
	reset();
	__resetChatIntegrationForTest();
	__resetSessionRegistryForTest();
});

describe('bootstrap — stream without SEARCH/REPLACE markers', () => {
	it('performs zero ai-patch state transitions for a plain assistant turn', async () => {
		const bootstrap = createChatPatchBootstrap({
			messageId: 'asst-1',
			modelId: 'test-model',
			conversationId: 'conv-1',
			parentMessageId: 'user-1',
			onNoTarget: (reason) => fixtures.toastCalls.push(reason)
		});

		// Simulate a multi-chunk plain-text reply — prose, punctuation,
		// even an isolated `<` and `>` that MUST NOT trip the parser
		// into thinking a fence is starting.
		const chunks = [
			"Hello, I'm an assistant.\n",
			'Here is a <tag> that should not matter.\n',
			'And some prose spanning several tokens ',
			'with no special markers anywhere.\n'
		];
		for (const c of chunks) bootstrap.feed(c);
		await bootstrap.end();

		// 1. No session open at end.
		expect(bootstrap.hasOpenedSession()).toBe(false);
		// 2. No DB probes — resolution was never invoked.
		expect(fixtures.findDocCalls).toEqual([]);
		expect(fixtures.getArtifactCalls).toEqual([]);
		expect(fixtures.captureCalls).toEqual([]);
		// 3. No registry entry for this message.
		expect(getPatchSession('asst-1')).toBeNull();
		// 4. No CommitResult stashed for the chat-store consumer.
		expect(consumeCompletedPatchSession('asst-1')).toBeNull();
		// 5. No toasts.
		expect(fixtures.toastCalls).toEqual([]);
	});

	it('is still inert when feed() is followed by abort() (user stopped generation early)', async () => {
		const bootstrap = createChatPatchBootstrap({
			messageId: 'asst-2',
			modelId: 'm',
			conversationId: 'conv-1',
			parentMessageId: 'user-1'
		});
		bootstrap.feed('partial reply...');
		bootstrap.abort();
		await bootstrap.end();
		expect(consumeCompletedPatchSession('asst-2')).toBeNull();
	});
});
