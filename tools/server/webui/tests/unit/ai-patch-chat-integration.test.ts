/**
 * handleCompletedPatchSession — chat-store ↔ ai-patch glue tests.
 *
 * The glue module is the live-chat counterpart to the headless
 * `runPatchRepairLoop` in `stream-hook.ts`. It's driven by the chat store
 * at `streamChatCompletion` completion time: given a `CommitResult` for
 * the assistant turn's PatchSession, it either injects a synthetic
 * `patch-repair` user turn + recurses through the caller's
 * `runAssistantTurn` callback, or — on budget exhaustion — emits a toast
 * and stops.
 *
 * We mock the database + conversations store the same way
 * `ai-patch-loop-close.test.ts` does, then drive the helper directly with
 * a scripted `runAssistantTurn` that simulates the chat store's
 * retrigger path.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

interface StoredMessage {
	id: string;
	convId: string;
	role: string;
	content: string;
	type: string;
	timestamp: number;
	toolCalls?: string;
	children: string[];
	parent: string | null;
	metadata?: Record<string, unknown>;
}

const fixtures = {
	createCalls: [] as Array<{ message: Omit<StoredMessage, 'id'>; parentId: string | null }>,
	activeConversation: null as { id: string } | null,
	activeMessages: [] as StoredMessage[],
	conversationMessages: new Map<string, StoredMessage[]>(),
	idCounter: 0,
	exhaustedToasts: [] as string[]
};

function reset() {
	fixtures.createCalls = [];
	fixtures.activeConversation = null;
	fixtures.activeMessages = [];
	fixtures.conversationMessages.clear();
	fixtures.idCounter = 0;
	fixtures.exhaustedToasts = [];
}

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		createMessageBranch: async (message: Omit<StoredMessage, 'id'>, parentId: string | null) => {
			fixtures.createCalls.push({ message, parentId });
			const stored: StoredMessage = { ...message, id: `msg-${++fixtures.idCounter}` };
			const list = fixtures.conversationMessages.get(message.convId) ?? [];
			list.push(stored);
			fixtures.conversationMessages.set(message.convId, list);
			return stored;
		},
		createRootMessage: async (convId: string) => `root-${convId}`
	}
}));

vi.mock('$lib/stores/conversations.svelte', () => ({
	conversationsStore: {
		get activeConversation() {
			return fixtures.activeConversation;
		},
		get activeMessages() {
			return fixtures.activeMessages;
		},
		addMessageToActive: (m: StoredMessage) => {
			fixtures.activeMessages.push(m);
		},
		updateCurrentNode: async () => {},
		updateConversationTimestamp: () => {},
		getConversationMessages: async (convId: string) =>
			fixtures.conversationMessages.get(convId) ?? []
	}
}));

import {
	__resetChatIntegrationForTest,
	consumeCompletedPatchSession,
	getReflectionCount,
	handleCompletedPatchSession,
	recordCompletedPatchSession
} from '$lib/editor/ai-patch/chat-integration';
import {
	getPatchSession,
	__resetSessionRegistryForTest
} from '$lib/editor/ai-patch/session-registry';
import { PatchFailureCode, type CommitResult } from '$lib/editor/ai-patch/types';

beforeEach(() => {
	reset();
	__resetChatIntegrationForTest();
	__resetSessionRegistryForTest();
	fixtures.activeConversation = { id: 'conv-1' };
	fixtures.conversationMessages.set('conv-1', [
		{
			id: 'msg-parent',
			convId: 'conv-1',
			role: 'assistant',
			content: '(patch attempt)',
			type: 'text',
			timestamp: 1,
			children: [],
			parent: null
		}
	]);
	fixtures.activeMessages = fixtures.conversationMessages.get('conv-1')!.slice();
});

describe('handleCompletedPatchSession', () => {
	it('injects a repair turn and retriggers the stream for a repairable F2 failure', async () => {
		const f2Failure: CommitResult = {
			committed: false,
			reason: 'no-blocks',
			repairable: true,
			errors: [
				{
					code: PatchFailureCode.E_NO_MATCH,
					reason: 'no match',
					blockIndex: 0,
					search: 'failed-search'
				}
			]
		};
		const retriggerCalls: number[] = [];

		await handleCompletedPatchSession({
			conversationId: 'conv-1',
			parentSessionId: 'msg-parent',
			result: f2Failure,
			runAssistantTurn: async () => {
				retriggerCalls.push(getReflectionCount('msg-parent'));
				// Simulate the retriggered stream producing a clean commit.
				// The chat store would call back into
				// `maybeRunPatchRepairLoop` with the new assistant message
				// id — no recorded result means no-op, which is what we
				// want here.
			}
		});

		// One repair turn persisted.
		expect(fixtures.createCalls).toHaveLength(1);
		expect(fixtures.createCalls[0].message.role).toBe('user');
		expect(
			(fixtures.createCalls[0].message.metadata as { source: { kind: string } }).source.kind
		).toBe('patch-repair');
		// The retrigger callback fired exactly once, with the bumped counter
		// already visible.
		expect(retriggerCalls).toEqual([1]);
		// Handle was unregistered after the retrigger returned.
		expect(getPatchSession('msg-parent')).toBeNull();
	});

	it('stops after MAX_REFLECTIONS failed attempts and emits a toast', async () => {
		const failing: CommitResult = {
			committed: false,
			reason: 'no-blocks',
			repairable: true,
			errors: [
				{
					code: PatchFailureCode.E_NO_MATCH,
					reason: 'no match',
					blockIndex: 0,
					search: 'oops'
				}
			]
		};
		const toasts: string[] = [];
		const maxReflections = 3;

		// Simulate maxReflections attempts in a row by re-entering the
		// helper from inside the retrigger callback — this mirrors what
		// chatStore.maybeRunPatchRepairLoop does on a recursive failure.
		const runRetry = async () => {
			await handleCompletedPatchSession({
				conversationId: 'conv-1',
				parentSessionId: 'msg-parent',
				result: failing,
				maxReflections,
				onExhausted: (m) => toasts.push(m),
				runAssistantTurn: runRetry
			});
		};
		await runRetry();

		// maxReflections=3 → 3 injected turns then exhaustion.
		expect(fixtures.createCalls).toHaveLength(maxReflections);
		expect(toasts).toHaveLength(1);
		expect(toasts[0]).toMatch(/exhausted after 3 attempts/);
		expect(getReflectionCount('msg-parent')).toBe(0); // cleared on exhaustion
	});

	it('does not retry E_USER_EDIT — user overrides are not bugs', async () => {
		let retriggered = false;
		await handleCompletedPatchSession({
			conversationId: 'conv-1',
			parentSessionId: 'msg-parent',
			result: {
				committed: false,
				reason: 'no-blocks',
				repairable: false,
				errors: [{ code: PatchFailureCode.E_USER_EDIT, reason: 'user edit', blockIndex: 0 }]
			},
			runAssistantTurn: async () => {
				retriggered = true;
			}
		});
		expect(retriggered).toBe(false);
		expect(fixtures.createCalls).toHaveLength(0);
	});

	it('clears state on successful commit', async () => {
		// Bump the counter first so we can assert it's cleared.
		await handleCompletedPatchSession({
			conversationId: 'conv-1',
			parentSessionId: 'msg-parent',
			result: {
				committed: false,
				reason: 'no-blocks',
				repairable: true,
				errors: [
					{
						code: PatchFailureCode.E_NO_MATCH,
						reason: 'no match',
						blockIndex: 0,
						search: 's'
					}
				]
			},
			runAssistantTurn: async () => {}
		});
		expect(getReflectionCount('msg-parent')).toBe(1);

		await handleCompletedPatchSession({
			conversationId: 'conv-1',
			parentSessionId: 'msg-parent',
			result: { revisionId: 'rev-final' },
			runAssistantTurn: async () => {
				throw new Error('should not be called for a committed result');
			}
		});
		expect(getReflectionCount('msg-parent')).toBe(0);
	});
});

describe('completed-session registry', () => {
	it('round-trips a CommitResult keyed by message id', () => {
		const result: CommitResult = { revisionId: 'rev-1' };
		recordCompletedPatchSession('msg-a', result);
		expect(consumeCompletedPatchSession('msg-a')).toBe(result);
		// Consume is one-shot.
		expect(consumeCompletedPatchSession('msg-a')).toBeNull();
	});

	it('returns null for unknown ids', () => {
		expect(consumeCompletedPatchSession('nope')).toBeNull();
	});
});
