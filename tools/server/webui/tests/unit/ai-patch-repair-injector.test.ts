/**
 * injectRepairTurn — thin glue between the repair loop and the chat
 * persistence layer. We mock `DatabaseService` and `conversationsStore`
 * so we can assert on exactly what the injector writes.
 *
 * The renderer contract is: `metadata.source.kind === 'patch-repair'`
 * with the full `MessageSource` shape. If that contract drifts the
 * ChatMessageUser.svelte branch needs to drift with it; this test is a
 * tripwire for that.
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
	updateCurrentNodeCalls: [] as string[],
	activeMessagesAppended: [] as StoredMessage[],
	timestampBumped: 0,
	activeConversation: null as { id: string } | null,
	activeMessages: [] as StoredMessage[],
	conversationMessages: new Map<string, StoredMessage[]>(),
	idCounter: 0
};

function resetFixtures() {
	fixtures.createCalls = [];
	fixtures.updateCurrentNodeCalls = [];
	fixtures.activeMessagesAppended = [];
	fixtures.timestampBumped = 0;
	fixtures.activeConversation = null;
	fixtures.activeMessages = [];
	fixtures.conversationMessages.clear();
	fixtures.idCounter = 0;
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
		createRootMessage: async (convId: string) => {
			const id = `root-${convId}`;
			const root: StoredMessage = {
				id,
				convId,
				role: 'system',
				content: '',
				type: 'root',
				timestamp: 0,
				children: [],
				parent: null
			};
			const list = fixtures.conversationMessages.get(convId) ?? [];
			list.push(root);
			fixtures.conversationMessages.set(convId, list);
			return id;
		}
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
			fixtures.activeMessagesAppended.push(m);
		},
		updateCurrentNode: async (id: string) => {
			fixtures.updateCurrentNodeCalls.push(id);
		},
		updateConversationTimestamp: () => {
			fixtures.timestampBumped++;
		},
		getConversationMessages: async (convId: string) => {
			return fixtures.conversationMessages.get(convId) ?? [];
		}
	}
}));

import { injectRepairTurn, PatchFailureCode } from '$lib/editor/ai-patch';

beforeEach(() => resetFixtures());

describe('injectRepairTurn', () => {
	it('persists a user-role message with the patch-repair source metadata', async () => {
		fixtures.activeConversation = { id: 'conv-1' };
		fixtures.activeMessages = [
			{
				id: 'msg-parent',
				convId: 'conv-1',
				role: 'assistant',
				content: 'search/replace attempt',
				type: 'text',
				timestamp: 1,
				children: [],
				parent: 'root-conv-1'
			}
		];

		const result = await injectRepairTurn('conv-1', 'body goes here', {
			parentSessionId: 'msg-parent',
			failureCode: PatchFailureCode.E_NO_MATCH,
			blockIndex: 1,
			reflection: 1
		});

		expect(fixtures.createCalls).toHaveLength(1);
		const call = fixtures.createCalls[0];
		expect(call.parentId).toBe('msg-parent');
		expect(call.message.role).toBe('user');
		expect(call.message.content).toBe('body goes here');
		expect(call.message.convId).toBe('conv-1');

		const source = (call.message.metadata as { source: Record<string, unknown> } | undefined)
			?.source;
		expect(source).toEqual({
			kind: 'patch-repair',
			parentSessionId: 'msg-parent',
			failureCode: PatchFailureCode.E_NO_MATCH,
			blockIndex: 1,
			reflection: 1
		});

		expect(result.id).toBe('msg-1');
		expect(fixtures.activeMessagesAppended).toHaveLength(1);
		expect(fixtures.updateCurrentNodeCalls).toEqual(['msg-1']);
		expect(fixtures.timestampBumped).toBe(1);
	});

	it('skips the active-messages mirror when targeting a non-active conversation', async () => {
		fixtures.activeConversation = { id: 'conv-A' };
		fixtures.activeMessages = [];
		fixtures.conversationMessages.set('conv-B', [
			{
				id: 'leaf-B',
				convId: 'conv-B',
				role: 'assistant',
				content: 'prior',
				type: 'text',
				timestamp: 0,
				children: [],
				parent: null
			}
		]);

		await injectRepairTurn('conv-B', 'body', {
			parentSessionId: 'leaf-B',
			failureCode: PatchFailureCode.E_NO_MATCH,
			blockIndex: 0,
			reflection: 1
		});

		expect(fixtures.createCalls).toHaveLength(1);
		expect(fixtures.createCalls[0].parentId).toBe('leaf-B');
		expect(fixtures.activeMessagesAppended).toHaveLength(0);
		expect(fixtures.updateCurrentNodeCalls).toHaveLength(0);
	});

	it('creates a root message when the conversation is empty', async () => {
		fixtures.activeConversation = null;
		fixtures.conversationMessages.set('conv-empty', []);

		await injectRepairTurn('conv-empty', 'body', {
			parentSessionId: 'unknown',
			failureCode: PatchFailureCode.E_NO_MATCH,
			blockIndex: 0,
			reflection: 1
		});

		expect(fixtures.createCalls).toHaveLength(1);
		expect(fixtures.createCalls[0].parentId).toBe('root-conv-empty');
	});
});
