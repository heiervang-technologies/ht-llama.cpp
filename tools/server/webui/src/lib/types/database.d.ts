import type { ChatMessageTimings, ChatRole, ChatMessageType } from '$lib/types/chat';
import { AttachmentType } from '$lib/enums';

export interface McpServerOverride {
	serverId: string;
	enabled: boolean;
}

export interface DatabaseConversation {
	currNode: string | null;
	id: string;
	lastModified: number;
	name: string;
	mcpServerOverrides?: McpServerOverride[];
	forkedFromConversationId?: string;
}

export interface DatabaseMessageExtraAudioFile {
	type: AttachmentType.AUDIO;
	name: string;
	base64Data: string;
	mimeType: string;
}

export interface DatabaseMessageExtraImageFile {
	type: AttachmentType.IMAGE;
	name: string;
	base64Url: string;
}

export interface DatabaseMessageExtraVideoFile {
	type: AttachmentType.VIDEO;
	name: string;
	/** `data:video/<format>;base64,…` so the chat history can render a
	 *  native <video controls> element directly from the stored extra. */
	base64Url: string;
	mimeType: string;
	/** Pre-rendered poster frame (first/middle frame) to show in attachment
	 *  chips without having to remount the video element. */
	posterDataUrl?: string;
	durationSec?: number;
	widthPx?: number;
	heightPx?: number;
	/** Precomputed frame sequence for models without native video input.
	 *  Each entry is a data URL (JPEG). Populated at attachment time so
	 *  we don't re-decode the video on every send. */
	fallbackFrames?: string[];
	/** Base64-encoded WAV audio extracted from the video track. Same
	 *  rationale — cached at attachment time. */
	fallbackAudioBase64?: string;
}

/**
 * Legacy format from old webui - pasted content was stored as "context" type
 * @deprecated Use DatabaseMessageExtraTextFile instead
 */
export interface DatabaseMessageExtraLegacyContext {
	type: AttachmentType.LEGACY_CONTEXT;
	name: string;
	content: string;
}

export interface DatabaseMessageExtraPdfFile {
	type: AttachmentType.PDF;
	base64Data: string;
	name: string;
	content: string;
	images?: string[];
	processedAsImages: boolean;
}

export interface DatabaseMessageExtraTextFile {
	type: AttachmentType.TEXT;
	name: string;
	content: string;
}

export interface DatabaseMessageExtraMcpPrompt {
	type: AttachmentType.MCP_PROMPT;
	name: string;
	serverName: string;
	promptName: string;
	content: string;
	arguments?: Record<string, string>;
}

export interface DatabaseMessageExtraMcpResource {
	type: AttachmentType.MCP_RESOURCE;
	name: string;
	uri: string;
	serverName: string;
	content: string;
	mimeType?: string;
}

export type DatabaseMessageExtra =
	| DatabaseMessageExtraImageFile
	| DatabaseMessageExtraTextFile
	| DatabaseMessageExtraAudioFile
	| DatabaseMessageExtraVideoFile
	| DatabaseMessageExtraPdfFile
	| DatabaseMessageExtraMcpPrompt
	| DatabaseMessageExtraMcpResource
	| DatabaseMessageExtraLegacyContext;

export interface DatabaseMessage {
	id: string;
	convId: string;
	type: ChatMessageType;
	timestamp: number;
	role: ChatRole;
	content: string;
	parent: string | null;
	/**
	 * @deprecated - left for backward compatibility
	 */
	thinking?: string;
	/** Reasoning content produced by the model (separate from visible content) */
	reasoningContent?: string;
	/** Serialized JSON array of tool calls made by assistant messages */
	toolCalls?: string;
	/** Tool call ID for tool result messages (role: 'tool') */
	toolCallId?: string;
	children: string[];
	extra?: DatabaseMessageExtra[];
	timings?: ChatMessageTimings;
	model?: string;
}

export type ExportedConversation = {
	conv: DatabaseConversation;
	messages: DatabaseMessage[];
};

export type ExportedConversations = ExportedConversation | ExportedConversation[];

export interface DatabaseDoc {
	id: string;
	name: string;
	content: string;
	createdAt: number;
	lastModified: number;
}

export type DatabaseArtifactKind =
	| 'html'
	| 'svg'
	| 'image'
	| 'code'
	| 'audio'
	| 'video'
	| 'pdf'
	| 'markdown';

export interface DatabaseArtifact {
	id: string;
	title: string;
	kind: DatabaseArtifactKind;
	currentRevisionId: string;
	tags: string[];
	createdAt: number;
	updatedAt: number;
	sourceConversationId?: string;
	sourceMessageSlot?: string;
	summary?: string;
}

export interface DatabaseArtifactRevision {
	id: string;
	artifactId: string;
	revisionNumber: number;
	createdAt: number;
	reason: 'initial' | 'regenerate' | 'edit' | 'fork';
	parentRevisionId?: string;
	contentHash: string;
	mimeType: string;
	/** Present for text-like kinds (html/svg/code/markdown). */
	text?: string;
	/** Present for binary kinds (image/audio/video/pdf) — stored as a Blob so
	 *  IndexedDB avoids the base64 inflation you'd get from a string field. */
	blob?: Blob;
	sourceMessageId?: string;
	metadata?: Record<string, unknown>;
}
