import { AttachmentType, FileTypeCategory, SpecialFileType } from '$lib/enums';
import { getFileTypeCategory, getFileTypeCategoryByExtension, isImageFile } from '$lib/utils';
import type {
	AttachmentDisplayItemsOptions,
	ChatAttachmentDisplayItem,
	ChatUploadedFile,
	DatabaseMessageExtra
} from '$lib/types';

/** True when a display item represents an MCP prompt invocation. */
export function isMcpPrompt(item: ChatAttachmentDisplayItem): boolean {
	return Boolean(item.isMcpPrompt);
}

/** True when a display item represents an MCP resource binding. */
export function isMcpResource(item: ChatAttachmentDisplayItem): boolean {
	return Boolean(item.isMcpResource);
}

/**
 * Check if an uploaded file is an MCP prompt
 */
function isMcpPromptUpload(file: ChatUploadedFile): boolean {
	return file.type === SpecialFileType.MCP_PROMPT && !!file.mcpPrompt;
}

/**
 * Check if an attachment is an MCP prompt
 */
function isMcpPromptAttachment(attachment: DatabaseMessageExtra): boolean {
	return attachment.type === AttachmentType.MCP_PROMPT;
}

/**
 * Check if an attachment is an MCP resource
 */
function isMcpResourceAttachment(attachment: DatabaseMessageExtra): boolean {
	return attachment.type === AttachmentType.MCP_RESOURCE;
}

/**
 * Gets the file type category from an uploaded file, checking both MIME type and extension
 */
function getUploadedFileCategory(file: ChatUploadedFile): FileTypeCategory | null {
	const categoryByMime = getFileTypeCategory(file.type);

	if (categoryByMime) {
		return categoryByMime;
	}

	return getFileTypeCategoryByExtension(file.name);
}

/**
 * Creates a unified list of display items from uploaded files and stored attachments.
 * Items are returned in reverse order (newest first).
 */
export function getAttachmentDisplayItems(
	options: AttachmentDisplayItemsOptions
): ChatAttachmentDisplayItem[] {
	const { uploadedFiles = [], attachments = [] } = options;
	const items: ChatAttachmentDisplayItem[] = [];

	// Add uploaded files (ChatForm)
	for (const file of uploadedFiles) {
		const uploadedCategory = getUploadedFileCategory(file);
		items.push({
			id: file.id,
			name: file.name,
			size: file.size,
			preview: file.preview,
			isImage: uploadedCategory === FileTypeCategory.IMAGE,
			isVideo: uploadedCategory === FileTypeCategory.VIDEO,
			isMcpPrompt: isMcpPromptUpload(file),
			isLoading: file.isLoading,
			loadError: file.loadError,
			uploadedFile: file,
			textContent: file.textContent
		});
	}

	// Add stored attachments (ChatMessage)
	for (const [index, attachment] of attachments.entries()) {
		const isImage = isImageFile(attachment);
		const isVideo = attachment.type === AttachmentType.VIDEO;
		const isMcpPrompt = isMcpPromptAttachment(attachment);
		const isMcpResource = isMcpResourceAttachment(attachment);

		let preview: string | undefined;
		if (isImage && 'base64Url' in attachment) {
			preview = attachment.base64Url;
		} else if (isVideo && 'base64Url' in attachment) {
			// Prefer the full video data URL so the attachment preview can
			// play inline; the poster is only used when we want a still
			// thumbnail (attachment list in history).
			preview = attachment.base64Url;
		}

		items.push({
			id: `attachment-${index}`,
			name: attachment.name,
			preview,
			isImage,
			isVideo,
			isMcpPrompt,
			isMcpResource,
			attachment,
			attachmentIndex: index,
			textContent: 'content' in attachment ? attachment.content : undefined
		});
	}

	return items.reverse();
}
