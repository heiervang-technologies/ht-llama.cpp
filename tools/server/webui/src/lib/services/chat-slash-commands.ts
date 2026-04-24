/**
 * Composer slash commands.
 *
 * The chat composer calls `tryHandleSlashCommand` on every submit. If
 * the message starts with a recognised slash, this module drives the
 * whole turn itself (adding user + assistant messages, dispatching
 * out-of-band, swapping the placeholder for the result) and returns
 * `true` so the chat store's normal send path is skipped. This keeps
 * direct-invoke media generation out of the LLM turn entirely — the
 * model never sees the `/image cats on the roof` text, and the proxy
 * call runs in parallel with whatever else the user is doing.
 *
 * Grammar today:
 *
 *   /image <prompt>    → POST /v1/images/generations (metadata.source='direct')
 *   /edit  <prompt>    → reserved, announces "coming soon"
 *   /video <prompt>    → reserved, announces "coming soon"
 *
 * `/edit` and `/video` are intentional stubs: mentioning the grammar
 * in-product teaches the user it exists, and when we wire the full
 * path the toast goes away with zero new muscle memory required.
 *
 * All assistant messages produced here include `ARTIFACT_SENTINEL` so
 * the auto-capture hook in `ChatMessageAssistant.svelte` doesn't
 * duplicate the artifact we already persisted via `runImageGeneration`.
 */

import { toast } from 'svelte-sonner';

import { chatStore } from '$lib/stores/chat.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { DatabaseService } from './database.service';
import { runImageGeneration } from './builtin-tools';
import { config } from '$lib/stores/settings.svelte';
import { MessageRole, MessageType } from '$lib/enums/chat';

/**
 * Sentinel markup embedded in assistant messages produced by
 * slash commands. `ChatMessageAssistant.svelte` reads this to skip
 * gallery auto-capture — the artifact already went in via saveManual.
 * Format: `<!--ht-slash-artifact:${id},${id}-->`.
 */
export const ARTIFACT_SENTINEL_PREFIX = '<!--ht-slash-artifact:';
export const ARTIFACT_SENTINEL_SUFFIX = '-->';

const IMAGE_SLASH = /^\/image(?:\s+([\s\S]+))?$/i;
const EDIT_SLASH = /^\/edit(?:\s+([\s\S]+))?$/i;
const VIDEO_SLASH = /^\/video(?:\s+([\s\S]+))?$/i;

export async function tryHandleSlashCommand(rawMessage: string): Promise<boolean> {
	const trimmed = rawMessage.trim();
	if (!trimmed.startsWith('/')) return false;

	const imageMatch = IMAGE_SLASH.exec(trimmed);
	if (imageMatch) {
		await handleImageSlash(trimmed, (imageMatch[1] ?? '').trim());
		return true;
	}
	if (EDIT_SLASH.test(trimmed)) {
		toast.info(
			'/edit is coming soon. For now, ask the model to use the `edit_image` tool with a source image.'
		);
		return true;
	}
	if (VIDEO_SLASH.test(trimmed)) {
		toast.info(
			'/video is coming soon. For now, ask the model to use the `generate_video` tool with a prompt.'
		);
		return true;
	}

	return false;
}

async function handleImageSlash(rawMessage: string, prompt: string): Promise<void> {
	if (!conversationsStore.activeConversation) {
		await conversationsStore.createConversation();
	}

	if (!prompt) {
		toast.info('Usage: /image <prompt>');
		return;
	}
	if (!config().imageGenEnabled) {
		toast.error('Image generation is disabled. Enable it in Settings → Tools.');
		return;
	}

	// Echo the raw `/image cats…` message into the transcript so the
	// thread reads coherently. This also anchors a slot (messageId +
	// ordinal) that the placeholder can attach to.
	const userMsg = await chatStore.addMessage(MessageRole.USER, rawMessage, MessageType.TEXT);

	const placeholderText = `_🎨 Generating image…_\n\n> ${prompt}`;
	const assistantMsg = await chatStore.addMessage(
		MessageRole.ASSISTANT,
		placeholderText,
		MessageType.TEXT,
		userMsg.id
	);

	try {
		const result = await runImageGeneration({ source: 'direct', prompt });
		const parts: string[] = [];
		for (const img of result.images) {
			const revision = await DatabaseService.getArtifactRevision(img.revisionId);
			if (!revision?.blob) continue;
			const dataUrl = await blobToDataUrl(revision.blob);
			const alt = `Generated · ${prompt.slice(0, 80)}`;
			parts.push(`![${escapeMd(alt)}](${dataUrl})`);
		}
		if (parts.length === 0) {
			await finaliseMessage(
				assistantMsg.id,
				`❌ Image generation completed but no inline preview could be loaded. Open the gallery to view the result.`
			);
			return;
		}

		const idsMarker =
			ARTIFACT_SENTINEL_PREFIX +
			result.images.map((i) => i.artifactId).join(',') +
			ARTIFACT_SENTINEL_SUFFIX;
		const finalContent =
			`${idsMarker}\n\n**${escapeMd(prompt)}** · ${result.model}\n\n${parts.join('\n\n')}`.trim();
		await finaliseMessage(assistantMsg.id, finalContent);
	} catch (e) {
		const message = e instanceof Error ? e.message : String(e);
		await finaliseMessage(assistantMsg.id, `❌ Image generation failed: ${message}`);
		toast.error(`Image generation failed: ${message}`);
	}
}

async function finaliseMessage(messageId: string, content: string): Promise<void> {
	await DatabaseService.updateMessage(messageId, { content });
	const idx = conversationsStore.findMessageIndex(messageId);
	if (idx >= 0) {
		conversationsStore.updateMessageAtIndex(idx, { content });
	}
}

function blobToDataUrl(blob: Blob): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onloadend = () => {
			if (typeof reader.result === 'string') resolve(reader.result);
			else reject(new Error('FileReader did not return a string'));
		};
		reader.onerror = () => reject(reader.error ?? new Error('FileReader failed'));
		reader.readAsDataURL(blob);
	});
}

function escapeMd(text: string): string {
	return text.replace(/[[\]()]/g, (m) => `\\${m}`);
}
