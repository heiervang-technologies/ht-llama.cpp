import { AttachmentType } from '$lib/enums/attachment';
import type { DatabaseMessageExtraImageFile } from '$lib/types/database';

// ![alt](data:image/<mime>;base64,<payload>) — tolerates an optional "title" after URL.
// Kept deliberately strict: only data URLs are extracted client-side because
// http(s) URLs would require a fetch that usually fails CORS for arbitrary
// hosts. Those are left in the text (rendering only) and can be attached
// manually via the paperclip to reach the vision encoder.
const MARKDOWN_DATA_IMAGE_REGEX =
	/!\[([^\]]*)\]\((data:image\/[a-zA-Z0-9+.-]+;base64,[A-Za-z0-9+/=]+)(?:\s+"[^"]*")?\)/g;

/**
 * Extracts inline `![alt](data:image/...)` markdown images from a user
 * message and returns them as attachment extras suitable for feeding to the
 * model's vision encoder. Keeps the original text untouched so the image
 * still renders inline in the message bubble (duplicated chips are filtered
 * at the render layer via `isInlineImageExtra`).
 */
export function extractMarkdownDataImageAttachments(text: string): DatabaseMessageExtraImageFile[] {
	const extras: DatabaseMessageExtraImageFile[] = [];
	const seen = new Set<string>();

	for (const match of text.matchAll(MARKDOWN_DATA_IMAGE_REGEX)) {
		const alt = match[1] ?? '';
		const dataUrl = match[2] ?? '';
		if (!dataUrl || seen.has(dataUrl)) continue;
		seen.add(dataUrl);

		const mime = dataUrl.match(/^data:(image\/[^;]+)/)?.[1] ?? 'image/png';
		const ext = (mime.split('/')[1] ?? 'png').replace('+xml', '');
		const baseName = alt.trim() || 'pasted-image';
		const name = baseName.endsWith(`.${ext}`) ? baseName : `${baseName}.${ext}`;

		extras.push({
			type: AttachmentType.IMAGE,
			name,
			base64Url: dataUrl
		});
	}

	return extras;
}

/**
 * True when an image extra's base64 URL is already embedded in the message
 * content — i.e. it was lifted from an inline markdown image and will render
 * via the markdown pipeline. The attachment chip row should skip these to
 * avoid a duplicate view of the same image.
 */
export function isInlineImageExtra(extra: DatabaseMessageExtraImageFile, content: string): boolean {
	if (!extra.base64Url || !content) return false;
	return content.includes(extra.base64Url);
}
