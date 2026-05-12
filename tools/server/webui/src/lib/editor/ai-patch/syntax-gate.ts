/**
 * Syntax gate — failure bucket F7 from the design brief §6.
 *
 * Runs once per session against the final shadow text, against the target
 * artifact/doc kind. When the post-apply text is syntactically invalid for
 * its kind, we reject the whole session rather than persist a broken
 * document. SWE-agent's +3pt improvement on syntax-gated commits
 * ([arXiv:2405.15793](https://arxiv.org/abs/2405.15793)) is the precedent.
 *
 * Kind-by-kind policy:
 *
 *   - `markdown`: **always** `{ ok: true }`. Markdown's grammar is permissive
 *     enough that parsing noise isn't a useful signal; the design brief is
 *     explicit that markdown warns but never blocks (§6, F7 row).
 *   - `html` / `svg`: `DOMParser` with the appropriate mime type. If the
 *     parsed document contains a `<parsererror>` element, the text is
 *     rejected with the extracted message.
 *   - `code`: skipped in v1 — a per-language parser stack (tree-sitter or
 *     similar) is a separate effort. We prefer false negatives over false
 *     positives here; a repair loop on top of a string-match parser would
 *     be worse than no check at all.
 *   - `image` / `audio` / `video` / `pdf`: not patchable in v1. The
 *     dispatcher already refuses these upstream; if one reaches us we
 *     pass-through without rejecting.
 *
 * SSR / node safety: DOMParser is browser-only. On the server we return
 * `{ ok: true, skipped: true }`. Unit tests that need the real check path
 * install a `globalThis.DOMParser` shim per-suite — see the syntax-gate
 * test file for the pattern.
 */

import type { DatabaseArtifactKind } from '$lib/types/database';

export type SyntaxResult =
	| { ok: true; skipped?: boolean; warning?: string }
	| { ok: false; error: string };

/**
 * Validate `text` against the grammar implied by `kind`.
 *
 * This is intentionally a pure function — it performs no I/O and no store
 * writes. The dispatcher calls it once at session commit; failures surface
 * as `{ ok: false }` and abort the commit. See the module header for the
 * per-kind policy.
 */
export function validateSyntax(kind: DatabaseArtifactKind | 'doc', text: string): SyntaxResult {
	// Markdown and docs always pass. A heading-heavy or broken table is still
	// a valid markdown document — rejecting it on a parser heuristic would be
	// more annoying than helpful.
	if (kind === 'markdown' || kind === 'doc') {
		return { ok: true };
	}

	if (kind === 'html') {
		return validateViaDOMParser(text, 'text/html');
	}

	if (kind === 'svg') {
		return validateViaDOMParser(text, 'image/svg+xml');
	}

	// Code / binary kinds — not gated in v1.
	return { ok: true, skipped: true };
}

function validateViaDOMParser(text: string, mime: 'text/html' | 'image/svg+xml'): SyntaxResult {
	// SSR / node / any environment without a DOM — skip the check. The check
	// will run in the browser (where the user is editing), which is where
	// invalid markup actually matters.
	if (typeof DOMParser === 'undefined') {
		return { ok: true, skipped: true };
	}

	let doc: Document;
	try {
		doc = new DOMParser().parseFromString(text, mime);
	} catch (err) {
		return { ok: false, error: `DOMParser threw: ${(err as Error).message}` };
	}

	// `text/html` never throws and always returns a document — errors appear
	// as <parsererror> children. `image/svg+xml` can return a parsererror
	// document for malformed XML; same detection.
	const parserError = doc.querySelector('parsererror');
	if (parserError) {
		const message = parserError.textContent?.trim() ?? 'parser error';
		// Keep the message reasonably short; repair prompts aren't helped by
		// multi-line DOMParser prose.
		const trimmed = message.length > 240 ? `${message.slice(0, 240)}…` : message;
		return { ok: false, error: trimmed };
	}

	return { ok: true };
}
