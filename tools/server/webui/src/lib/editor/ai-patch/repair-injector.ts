/**
 * Repair-turn injector — thin glue between the pure `RepairLoop` and the
 * chat persistence layer.
 *
 * The repair loop itself produces a formatted body string; the injector
 * stamps that into a `DatabaseMessage` with role `user`, attaches the
 * `patch-repair` source tag, persists it via `DatabaseService`, and
 * surfaces it through the active-messages store so the UI can render it
 * immediately.
 *
 * We deliberately do **not** call `ChatService.sendMessage` here — the
 * existing chat flow will pick up the injected user turn on its next
 * trigger (either the session's own re-drive, or an explicit user
 * retry). Wiring that trigger into the stream callbacks is commit 4b's
 * problem.
 */

import { DatabaseService } from '$lib/services/database.service';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types';
import type { MessageSource, PatchFailureCode } from './types';

/**
 * Metadata threaded onto the injected user turn. The renderer switches on
 * `metadata.source.kind === 'patch-repair'` to pick up the muted styling.
 */
export interface RepairInjectionMeta {
	/** The assistant-message id that produced the failed patch session. */
	parentSessionId: string;
	/** The first repair-format-supported error code. */
	failureCode: PatchFailureCode;
	/** Stream-order block index the retry is aimed at. */
	blockIndex: number;
	/** 1-based reflection counter. */
	reflection: number;
}

/**
 * Append a patch-repair user turn to the given conversation. Returns the
 * persisted `DatabaseMessage` so the caller can assert on / forward its
 * id. The new message is parented to the conversation's current leaf,
 * mirroring how a human-typed turn lands.
 *
 * @param conversationId - Target conversation.
 * @param body - Pre-formatted message body from `formatRepairMessage`.
 * @param meta  - Provenance for the renderer and telemetry.
 */
export async function injectRepairTurn(
	conversationId: string,
	body: string,
	meta: RepairInjectionMeta
): Promise<DatabaseMessage> {
	const parentId = await resolveParentId(conversationId);

	const source: MessageSource = {
		kind: 'patch-repair',
		parentSessionId: meta.parentSessionId,
		failureCode: meta.failureCode,
		blockIndex: meta.blockIndex,
		reflection: meta.reflection
	};

	const message = await DatabaseService.createMessageBranch(
		{
			convId: conversationId,
			role: MessageRole.USER,
			content: body,
			type: MessageType.TEXT,
			timestamp: Date.now(),
			toolCalls: '',
			children: [],
			// The dispatcher's source metadata is stored alongside this one
			// under `metadata.ai-patch.*` by commit 4b — for now the repair
			// source is the only member.
			metadata: { source }
		},
		parentId
	);

	// Only mirror into the active-messages array if this injection targets
	// the currently-open conversation. Running a repair loop against an
	// inactive conversation (e.g. from a background retry) is legal: the
	// DB write happens, and the next time the user opens that conversation
	// the message will load through the regular path.
	const active = conversationsStore.activeConversation;
	if (active?.id === conversationId) {
		conversationsStore.addMessageToActive(message);
		await conversationsStore.updateCurrentNode(message.id);
		conversationsStore.updateConversationTimestamp();
	}

	return message;
}

/* ------------------------------------------------------------------------- */
/* Internal                                                                  */
/* ------------------------------------------------------------------------- */

/**
 * Find the parent for the injected turn. Prefers the active leaf (matches
 * the normal `chatStore.addMessage` path) and falls back to the root
 * message lookup when the conversation isn't active in memory.
 */
async function resolveParentId(conversationId: string): Promise<string | null> {
	const active = conversationsStore.activeConversation;
	if (active?.id === conversationId) {
		const leaf = conversationsStore.activeMessages.at(-1);
		if (leaf) return leaf.id;
	}

	const all = await conversationsStore.getConversationMessages(conversationId);
	const leaf = all.at(-1);
	if (leaf) return leaf.id;

	const root = all.find((m) => m.parent === null && m.type === 'root');
	if (root) return root.id;

	// Last-ditch: create a root if the conversation is empty. Mirrors the
	// behaviour of chatStore.addMessage so an injected turn into an
	// otherwise-empty conversation doesn't drop on the floor.
	return await DatabaseService.createRootMessage(conversationId);
}
