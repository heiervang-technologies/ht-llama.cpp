/**
 * Lazy-elision detector — failure bucket F6 from the design brief §6.
 *
 * LLMs occasionally emit placeholder "unchanged" markers inside REPLACE
 * payloads — `// ... rest unchanged`, `# existing code`, bare `...` lines,
 * etc. Silently accepting these destroys the target buffer (see
 * https://github.com/google-gemini/gemini-cli/issues/4836). We reject the
 * block at commit time when any of these patterns show up.
 *
 * Detection is deliberately conservative: a literal `...` that is obviously
 * part of real content (e.g. inside a string, a template literal, prose,
 * JSX text) must not trip us. We only flag when the marker is *plainly
 * placeholder text*: bare on its line, or the sole non-whitespace body of a
 * comment line.
 */

export interface ElisionHit {
	/** Line number within the REPLACE payload, zero-indexed. */
	line: number;
	/** The raw line text that tripped the detector. */
	text: string;
	/** Human-readable reason, for the repair-loop prompt. */
	reason: string;
}

/** Bare `...` (possibly with leading/trailing whitespace) on its own line. */
const BARE_ELLIPSIS_RE = /^\s*\.{3,}\s*$/;

/**
 * Comment-style placeholder. We match common single-line comment openers
 * (`//`, `#`, `--`), block comment pairs (`/* ... *\/`), and HTML comments
 * (`<!-- ... -->`). The body must contain an ellipsis **and** a placeholder
 * keyword ("unchanged", "existing", "rest", "previous", "same", "omitted",
 * "truncated", "snip") — or be a bare ellipsis.
 */
const PLACEHOLDER_WORDS_RE =
	/\b(unchanged|existing|rest|previous|same|omitted|truncated|snip|as\s+before)\b/i;

const ELLIPSIS_RE = /\.{3,}|…/;

/** Extract the comment body from a line, if the line is a pure comment. */
function commentBody(line: string): string | null {
	const trimmed = line.trim();
	// Line comments.
	const line1 = /^\/\/\s?(.*)$/.exec(trimmed);
	if (line1) return line1[1];
	const line2 = /^#\s?(.*)$/.exec(trimmed);
	if (line2) return line2[1];
	const line3 = /^--\s?(.*)$/.exec(trimmed);
	if (line3) return line3[1];
	// Block comments — whole line is /* ... */
	const block = /^\/\*\s?(.*?)\s?\*\/\s*$/.exec(trimmed);
	if (block) return block[1];
	// HTML comments — whole line is <!-- ... -->
	const html = /^<!--\s?(.*?)\s?-->\s*$/.exec(trimmed);
	if (html) return html[1];
	return null;
}

/** True when the comment body itself looks like a placeholder. */
function bodyLooksPlaceholder(body: string): boolean {
	// Bare "..." inside a comment: "// ...".
	if (/^\.{3,}$|^…$/.test(body.trim())) return true;
	// An ellipsis paired with a placeholder keyword — "... rest unchanged",
	// "existing code ...", "... as before".
	if (ELLIPSIS_RE.test(body) && PLACEHOLDER_WORDS_RE.test(body)) return true;
	// Keyword-only (no ellipsis) — e.g. "// existing code", "# rest unchanged".
	// Only when the body is *just* the placeholder phrase, nothing else —
	// otherwise we'd flag real comments like "# the rest of the logic follows".
	// We require the body to be short (<= 40 chars) and match a tight pattern.
	if (body.length <= 40) {
		if (/^existing\s+code\b\s*\.?$/i.test(body.trim())) return true;
		if (/^rest\s+unchanged\b\s*\.?$/i.test(body.trim())) return true;
		if (/^unchanged\b\s*\.?$/i.test(body.trim())) return true;
		if (/^previous\s+code\b\s*\.?$/i.test(body.trim())) return true;
		if (/^code\s+unchanged\b\s*\.?$/i.test(body.trim())) return true;
	}
	return false;
}

/**
 * Scan a REPLACE payload for lazy-elision markers.
 * Returns the first hit, or `null` if the payload is clean.
 *
 * We short-circuit at the first hit because one match is enough to fail
 * commit (F6); the repair prompt only needs one concrete example.
 */
export function detectElision(replaceText: string): ElisionHit | null {
	if (!replaceText) return null;
	const lines = replaceText.split('\n');
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		// Bare ellipsis — "  ...  ".
		if (BARE_ELLIPSIS_RE.test(line)) {
			return { line: i, text: line, reason: 'bare ellipsis line' };
		}
		const body = commentBody(line);
		if (body !== null && bodyLooksPlaceholder(body)) {
			return { line: i, text: line, reason: 'placeholder comment' };
		}
	}
	return null;
}
