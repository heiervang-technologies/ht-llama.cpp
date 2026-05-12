/**
 * runPatchRepairLoop — loop-self-closing orchestrator tests.
 *
 * The orchestrator opens a fresh `PatchSession` per attempt, drives the
 * caller-supplied `runStream` with the matching hook, and re-injects a
 * synthetic user turn whenever the previous attempt emitted a repairable
 * failure. These tests fake the ChatService plumbing end of the contract
 * (by scripting each session's `.end()` return value) and assert on the
 * resulting injected messages + final `CommitResult`.
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
	PatchFailureCode,
	runPatchRepairLoop,
	stopPatchRepairLoop,
	type RunStreamContext
} from '$lib/editor/ai-patch';
import { __resetSessionRegistryForTest } from '$lib/editor/ai-patch/session-registry';
import type { PatchSession } from '$lib/editor/ai-patch/dispatcher';
import type { CommitResult } from '$lib/editor/ai-patch/types';

/**
 * Build a throwaway object that mimics the narrow surface the
 * orchestrator calls into on a real `PatchSession`. Cast to
 * `PatchSession` at the boundary — the type is intentionally wide but
 * the orchestrator only touches these three methods.
 */
function fakeSession(scripted: () => CommitResult): PatchSession {
	let ended = false;
	const stub = {
		feed: () => {},
		end: async () => {
			if (ended) throw new Error('already ended');
			ended = true;
			return scripted();
		},
		abort: () => {}
	};
	return stub as unknown as PatchSession;
}

beforeEach(() => {
	reset();
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

describe('runPatchRepairLoop', () => {
	it('injects a synthetic user turn when the first attempt has repairable errors, then commits', async () => {
		const outcomes: CommitResult[] = [
			{
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
			},
			{ revisionId: 'rev-final' }
		];
		const attempts: number[] = [];
		const result = await runPatchRepairLoop({
			conversationId: 'conv-1',
			parentSessionId: 'msg-parent',
			sessionFactory: (att) => {
				attempts.push(att.reflectionCount);
				return fakeSession(() => outcomes.shift()!);
			},
			runStream: async () => {
				// Real callers would feed chunks through the hook here; we
				// no-op because the scripted `.end()` is driven by the
				// outcome queue.
			},
			onExhausted: (msg) => fixtures.exhaustedToasts.push(msg)
		});
		expect(result.revisionId).toBe('rev-final');
		// Two attempts: initial + one retry.
		expect(attempts).toEqual([0, 1]);
		// One repair message persisted.
		expect(fixtures.createCalls).toHaveLength(1);
		expect(fixtures.createCalls[0].message.role).toBe('user');
		expect(
			(fixtures.createCalls[0].message.metadata as { source: { kind: string } }).source.kind
		).toBe('patch-repair');
		expect(fixtures.exhaustedToasts).toHaveLength(0);
	});

	it('surfaces a toast and stops after maxReflections failed retries', async () => {
		// maxReflections=2 → initial attempt + 2 retries = 3 total attempts,
		// then the loop emits the exhausted toast and returns the last
		// failure sentinel.
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
		const attempts: number[] = [];
		const result = await runPatchRepairLoop({
			conversationId: 'conv-1',
			parentSessionId: 'msg-parent',
			maxReflections: 2,
			sessionFactory: (att) => {
				attempts.push(att.reflectionCount);
				return fakeSession(() => failing);
			},
			runStream: async () => {},
			onExhausted: (msg) => fixtures.exhaustedToasts.push(msg)
		});
		expect(attempts).toEqual([0, 1, 2]);
		expect(result.committed).toBe(false);
		expect(fixtures.exhaustedToasts).toHaveLength(1);
		expect(fixtures.exhaustedToasts[0]).toMatch(/exhausted after 2 attempts/);
		// Two repair turns were injected (one after each of the first two
		// failed attempts).
		expect(fixtures.createCalls).toHaveLength(2);
	});

	it('does not retry E_USER_EDIT failures — they are user overrides, not bugs', async () => {
		const result = await runPatchRepairLoop({
			conversationId: 'conv-1',
			parentSessionId: 'msg-parent',
			sessionFactory: () =>
				fakeSession(() => ({
					committed: false,
					reason: 'no-blocks',
					// The dispatcher sets repairable=false when only E_USER_EDIT
					// errors are present (it's not in REPAIRABLE_CODES).
					repairable: false,
					errors: [{ code: PatchFailureCode.E_USER_EDIT, reason: 'user edit', blockIndex: 0 }]
				})),
			runStream: async () => {},
			onExhausted: (msg) => fixtures.exhaustedToasts.push(msg)
		});
		expect(result.committed).toBe(false);
		expect(fixtures.createCalls).toHaveLength(0);
		expect(fixtures.exhaustedToasts).toHaveLength(0);
	});

	it('stopPatchRepairLoop aborts the in-flight stream and prevents further retries', async () => {
		const failing: CommitResult = {
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
		};
		const result = await runPatchRepairLoop({
			conversationId: 'conv-1',
			parentSessionId: 'msg-parent',
			sessionFactory: () => fakeSession(() => failing),
			runStream: async (ctx: RunStreamContext) => {
				// User clicks the × during the first attempt.
				if (ctx.reflectionCount === 0) {
					stopPatchRepairLoop('msg-parent');
					const err = new Error('aborted');
					err.name = 'AbortError';
					throw err;
				}
			}
		});
		expect(result.committed).toBe(false);
		expect(result.reason).toBe('aborted');
		expect(fixtures.createCalls).toHaveLength(0);
	});
});
