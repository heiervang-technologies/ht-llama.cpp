import { describe, expect, it } from 'vitest';
import {
	extractMarkdownDataImageAttachments,
	isInlineImageExtra
} from '$lib/utils/extract-markdown-images';
import { AttachmentType } from '$lib/enums/attachment';

const PNG_PIXEL =
	'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2z9EAAAAAASUVORK5CYII=';
const JPEG_PIXEL =
	'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ==';

describe('extractMarkdownDataImageAttachments', () => {
	it('returns no extras when the text has no markdown images', () => {
		expect(extractMarkdownDataImageAttachments('hello world')).toEqual([]);
	});

	it('ignores http and relative image URLs — only data: URLs are lifted', () => {
		const text = `![cat](https://example.com/cat.png) and ![local](/img.png)`;
		expect(extractMarkdownDataImageAttachments(text)).toEqual([]);
	});

	it('extracts a single data: image and uses the alt text for the name', () => {
		const text = `![my-pic](${PNG_PIXEL})`;
		const extras = extractMarkdownDataImageAttachments(text);
		expect(extras).toHaveLength(1);
		expect(extras[0]).toMatchObject({
			type: AttachmentType.IMAGE,
			name: 'my-pic.png',
			base64Url: PNG_PIXEL
		});
	});

	it('falls back to a default name when alt text is empty', () => {
		const text = `![](${JPEG_PIXEL})`;
		const extras = extractMarkdownDataImageAttachments(text);
		expect(extras).toHaveLength(1);
		expect(extras[0].name).toBe('pasted-image.jpeg');
	});

	it('extracts multiple distinct data images', () => {
		const text = `first ![a](${PNG_PIXEL}) middle ![b](${JPEG_PIXEL}) last`;
		const extras = extractMarkdownDataImageAttachments(text);
		expect(extras).toHaveLength(2);
		expect(extras.map((e) => e.base64Url)).toEqual([PNG_PIXEL, JPEG_PIXEL]);
	});

	it('deduplicates the same data URL used twice', () => {
		const text = `![a](${PNG_PIXEL}) again ![b](${PNG_PIXEL})`;
		const extras = extractMarkdownDataImageAttachments(text);
		expect(extras).toHaveLength(1);
	});

	it('tolerates a markdown title after the URL', () => {
		const text = `![a](${PNG_PIXEL} "caption")`;
		const extras = extractMarkdownDataImageAttachments(text);
		expect(extras).toHaveLength(1);
		expect(extras[0].base64Url).toBe(PNG_PIXEL);
	});

	it('does not append the extension twice when alt already contains it', () => {
		const text = `![photo.png](${PNG_PIXEL})`;
		const extras = extractMarkdownDataImageAttachments(text);
		expect(extras[0].name).toBe('photo.png');
	});
});

describe('isInlineImageExtra', () => {
	const extra = {
		type: AttachmentType.IMAGE,
		name: 'pasted-image.png',
		base64Url: PNG_PIXEL
	} as const;

	it('returns true when the content embeds the same base64 URL', () => {
		expect(isInlineImageExtra(extra, `look at this ![](${PNG_PIXEL})`)).toBe(true);
	});

	it('returns false when the content is unrelated text', () => {
		expect(isInlineImageExtra(extra, 'just some words')).toBe(false);
	});

	it('returns false for an empty base64 URL', () => {
		expect(isInlineImageExtra({ ...extra, base64Url: '' }, PNG_PIXEL)).toBe(false);
	});
});
