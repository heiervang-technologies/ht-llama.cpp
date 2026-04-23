/**
 * Extracts renderable artifacts (currently HTML/SVG) from assistant message content.
 *
 * Returns an ordered list, one entry per fenced code block of the supported languages.
 * The extractor is deliberately conservative: it only picks up blocks that look like
 * real standalone content (length > 60 chars, contains angle brackets) so short inline
 * snippets don't steal the drawer.
 */

export type ArtifactKind = 'html' | 'svg';

export interface ExtractedArtifact {
	index: number;
	kind: ArtifactKind;
	language: string;
	content: string;
	title: string;
}

const FENCE_RE = /```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g;

function classify(lang: string, content: string): ArtifactKind | null {
	const l = lang.toLowerCase().trim();
	if (l === 'html' || l === 'htm') return 'html';
	if (l === 'svg') return 'svg';
	// Detect bare HTML/SVG blocks that were fenced without a language tag.
	if (!l) {
		const trimmed = content.trimStart();
		if (/^<!doctype html/i.test(trimmed)) return 'html';
		if (/^<svg[\s>]/i.test(trimmed)) return 'svg';
		if (/^<html[\s>]/i.test(trimmed)) return 'html';
	}
	return null;
}

function deriveTitle(content: string, kind: ArtifactKind, fallbackIndex: number): string {
	if (kind === 'html') {
		const match = content.match(/<title>([^<]+)<\/title>/i);
		if (match) return match[1].trim();
		const h1 = content.match(/<h1[^>]*>([^<]+)<\/h1>/i);
		if (h1) return h1[1].trim();
	}
	return kind === 'html'
		? `HTML artifact ${fallbackIndex + 1}`
		: `SVG artifact ${fallbackIndex + 1}`;
}

export function extractArtifacts(content: string): ExtractedArtifact[] {
	if (!content || !content.includes('```')) return [];
	const out: ExtractedArtifact[] = [];
	let match: RegExpExecArray | null;
	FENCE_RE.lastIndex = 0;
	while ((match = FENCE_RE.exec(content)) !== null) {
		const language = match[1] ?? '';
		const body = match[2] ?? '';
		const kind = classify(language, body);
		if (!kind) continue;
		const trimmed = body.trim();
		if (trimmed.length < 60) continue;
		if (!/[<>]/.test(trimmed)) continue;
		out.push({
			index: out.length,
			kind,
			language: language || kind,
			content: body,
			title: deriveTitle(body, kind, out.length)
		});
	}
	return out;
}

/** Cheap hash used to dedupe reruns during streaming without keeping full text. */
export function hashString(text: string): string {
	let h = 0;
	for (let i = 0; i < text.length; i++) {
		h = (h * 31 + text.charCodeAt(i)) | 0;
	}
	return h.toString(36);
}

export type GalleryKind = 'html' | 'svg' | 'image' | 'code' | 'markdown';

export interface GalleryArtifactCandidate {
	index: number;
	kind: GalleryKind;
	title: string;
	mimeType: string;
	text?: string;
	blob?: Blob;
	summary?: string;
	language?: string;
}

// Hybrid capture thresholds — anything smaller stays in the ephemeral drawer.
const MIN_CODE_LINES = 30;
const HTML_MIN_CHARS = 2000;
const SVG_MIN_CHARS = 1000;

const FULL_HTML_RE = /(<!doctype html[\s>]|<html[\s>])/i;
const SVG_ROOT_RE = /^<svg[\s>]/i;
const IMG_DATA_URL_RE = /!\[[^\]]*]\((data:image\/[a-z0-9+.-]+;base64,[^)]+)\)/gi;

function looksLikeHtml(lang: string, body: string): boolean {
	const l = lang.toLowerCase();
	if (l === 'html' || l === 'htm') return true;
	if (!l && FULL_HTML_RE.test(body)) return true;
	return false;
}

function looksLikeSvg(lang: string, body: string): boolean {
	const l = lang.toLowerCase();
	if (l === 'svg') return true;
	if (!l && SVG_ROOT_RE.test(body.trimStart())) return true;
	return false;
}

function dataUrlToBlob(dataUrl: string): { blob: Blob; mime: string } | null {
	const match = /^data:([^;,]+);base64,(.*)$/.exec(dataUrl);
	if (!match) return null;
	const [, mime, b64] = match;
	try {
		const bin = atob(b64);
		const buf = new Uint8Array(bin.length);
		for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
		return { blob: new Blob([buf], { type: mime }), mime };
	} catch {
		return null;
	}
}

/**
 * Extracts gallery-worthy artifacts (above the hybrid thresholds) from a
 * finished assistant message. The `index` is a deterministic ordinal so the
 * caller can build a slot id that stays stable across regenerations of the
 * same turn — letting the gallery append a revision instead of forking a
 * new artifact.
 */
export function extractGalleryArtifacts(content: string): GalleryArtifactCandidate[] {
	if (!content) return [];
	const out: GalleryArtifactCandidate[] = [];
	let ordinal = 0;

	if (content.includes('```')) {
		FENCE_RE.lastIndex = 0;
		let match: RegExpExecArray | null;
		while ((match = FENCE_RE.exec(content)) !== null) {
			const language = (match[1] ?? '').trim();
			const body = match[2] ?? '';
			const trimmed = body.trim();
			if (!trimmed) continue;

			if (looksLikeHtml(language, trimmed)) {
				if (FULL_HTML_RE.test(trimmed) || trimmed.length >= HTML_MIN_CHARS) {
					out.push({
						index: ordinal++,
						kind: 'html',
						title: deriveTitle(body, 'html', out.length),
						mimeType: 'text/html',
						text: body,
						language: language || 'html'
					});
				}
				continue;
			}

			if (looksLikeSvg(language, trimmed)) {
				if (trimmed.length >= SVG_MIN_CHARS || SVG_ROOT_RE.test(trimmed)) {
					out.push({
						index: ordinal++,
						kind: 'svg',
						title: deriveTitle(body, 'svg', out.length),
						mimeType: 'image/svg+xml',
						text: body,
						language: language || 'svg'
					});
				}
				continue;
			}

			const lines = body.split('\n').length;
			if (lines >= MIN_CODE_LINES) {
				const firstLine = body.split('\n').find((l) => l.trim().length > 0) ?? '';
				const title = language ? `${language} (${lines} lines)` : `code (${lines} lines)`;
				out.push({
					index: ordinal++,
					kind: 'code',
					title,
					mimeType: language ? `text/x-${language}` : 'text/plain',
					text: body,
					summary: firstLine.slice(0, 120),
					language: language || undefined
				});
			}
		}
	}

	IMG_DATA_URL_RE.lastIndex = 0;
	let imgMatch: RegExpExecArray | null;
	while ((imgMatch = IMG_DATA_URL_RE.exec(content)) !== null) {
		const dataUrl = imgMatch[1];
		const converted = dataUrlToBlob(dataUrl);
		if (!converted) continue;
		out.push({
			index: ordinal++,
			kind: 'image',
			title: `Generated image ${out.filter((a) => a.kind === 'image').length + 1}`,
			mimeType: converted.mime,
			blob: converted.blob
		});
	}

	return out;
}
