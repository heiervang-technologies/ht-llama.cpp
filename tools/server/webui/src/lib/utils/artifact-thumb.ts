/**
 * Picks a lightweight preview representation for a gallery card.
 *
 * - images return an object URL for the stored Blob (caller revokes it).
 * - audio/video return the Blob URL too (cards render <audio>/<video> poster).
 * - html/svg/markdown return a sanitized inline snippet (no scripts run —
 *   the real renderer lives in the detail route).
 * - pdf/code fall back to a monospace excerpt.
 */

import type { DatabaseArtifactKind, DatabaseArtifactRevision } from '$lib/types/database';

export type ArtifactThumb =
	| { kind: 'image' | 'audio' | 'video' | 'pdf'; url: string }
	| { kind: 'svg'; markup: string }
	| { kind: 'code' | 'html' | 'markdown'; excerpt: string };

const EXCERPT_LINES = 12;
const EXCERPT_CHARS = 480;

export function buildThumb(
	kind: DatabaseArtifactKind,
	revision: DatabaseArtifactRevision | undefined
): ArtifactThumb | null {
	if (!revision) return null;

	if (kind === 'image' || kind === 'audio' || kind === 'video' || kind === 'pdf') {
		if (!revision.blob) return null;
		return { kind, url: URL.createObjectURL(revision.blob) };
	}

	const text = revision.text ?? '';
	if (!text) return null;

	if (kind === 'svg') {
		// Strip <script> defensively; the full sandbox is applied in the
		// detail/iframe renderer, but we'd rather not even parse script
		// content for a thumbnail.
		const markup = text.replace(/<script[\s\S]*?<\/script>/gi, '');
		return { kind: 'svg', markup };
	}

	const excerpt = text
		.split('\n')
		.slice(0, EXCERPT_LINES)
		.join('\n')
		.slice(0, EXCERPT_CHARS);
	return { kind: kind === 'html' ? 'html' : kind === 'markdown' ? 'markdown' : 'code', excerpt };
}

export function revokeThumb(thumb: ArtifactThumb | null): void {
	if (thumb && 'url' in thumb) URL.revokeObjectURL(thumb.url);
}
