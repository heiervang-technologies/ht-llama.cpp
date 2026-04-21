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
