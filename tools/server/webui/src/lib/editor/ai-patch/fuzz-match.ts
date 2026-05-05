/**
 * Anchor finder — the COLLECTING_SEARCH → ANCHOR_LOCKED transition from the
 * design brief §3.2.
 *
 * Implements Aider's `replace_most_similar_chunk` ladder:
 *   1. Exact match.
 *   2. Whitespace-stripped match (ignore intra-line indentation drift).
 *   3. Leading-blank-line tolerant.
 *   4. `...` elision expansion (SEARCH contains `...`; we stitch around it).
 *   5. Sequence-similarity ≥ 0.8 (LCS ratio — our own, no external library).
 *
 * Ambiguity guard (design §3.2 step 3): enforce strictly when the SEARCH
 * target is short. If the non-whitespace length is < 10 chars, *any* >1
 * match is ambiguous; otherwise only >1 exact matches are ambiguous (later
 * ladder rungs that each produce a single best match are allowed through).
 *
 * Returns offsets into `buffer` — character indices, not byte indices — so
 * the dispatcher can use them directly with CM6 `ChangeSet` ranges later.
 */

export type AnchorVia = 'exact' | 'ws' | 'leading-blank' | 'elision' | 'similarity';

export type AnchorResult =
	| {
			kind: 'unique';
			from: number;
			to: number;
			via: AnchorVia;
	  }
	| {
			kind: 'none';
			/** Top-3 similar regions, for the F2 repair prompt. */
			suggestions: Array<{ from: number; to: number; similarity: number }>;
	  }
	| {
			kind: 'ambiguous';
			matches: Array<{ from: number; to: number }>;
	  };

/* ------------------------------------------------------------------------- */

function nonWsLength(s: string): number {
	let n = 0;
	for (let i = 0; i < s.length; i++) {
		const c = s.charCodeAt(i);
		if (c !== 32 && c !== 9 && c !== 10 && c !== 13) n++;
	}
	return n;
}

function countMatches(haystack: string, needle: string): number {
	if (!needle) return 0;
	let count = 0;
	let i = 0;
	while (true) {
		const at = haystack.indexOf(needle, i);
		if (at < 0) break;
		count++;
		i = at + needle.length;
	}
	return count;
}

function findAllIndexes(haystack: string, needle: string): number[] {
	const out: number[] = [];
	if (!needle) return out;
	let i = 0;
	while (true) {
		const at = haystack.indexOf(needle, i);
		if (at < 0) break;
		out.push(at);
		i = at + needle.length;
	}
	return out;
}

/** Drop leading entirely-blank lines. */
function trimLeadingBlank(text: string): string {
	return text.replace(/^(?:[ \t]*\n)+/, '');
}

/* ------------------------------------------------------------------------- */
/* Rung 1: exact                                                             */
/* ------------------------------------------------------------------------- */

function rungExact(buffer: string, search: string): AnchorResult | null {
	const idxs = findAllIndexes(buffer, search);
	if (idxs.length === 0) return null;
	const short = nonWsLength(search) < 10;
	if (idxs.length > 1) {
		if (short) {
			return {
				kind: 'ambiguous',
				matches: idxs.map((i) => ({ from: i, to: i + search.length }))
			};
		}
		// ≥ 10 non-ws chars and still >1 match → ambiguous per design.
		return {
			kind: 'ambiguous',
			matches: idxs.map((i) => ({ from: i, to: i + search.length }))
		};
	}
	return {
		kind: 'unique',
		from: idxs[0],
		to: idxs[0] + search.length,
		via: 'exact'
	};
}

/* ------------------------------------------------------------------------- */
/* Rung 2: whitespace-stripped                                               */
/* ------------------------------------------------------------------------- */

/**
 * Line-oriented whitespace-tolerant match. Project both buffer and search
 * into per-line indent-stripped form, scan line-by-line, and report the
 * original-buffer character range of any matching window.
 */
function rungWhitespace(buffer: string, search: string): AnchorResult | null {
	const bufLines = buffer.split('\n');
	const searchLines = search.split('\n');
	if (searchLines.length === 0) return null;

	const searchStripped = searchLines.map((l) => l.replace(/^[ \t]+/, ''));
	const bufStripped = bufLines.map((l) => l.replace(/^[ \t]+/, ''));

	// Short-circuit if stripping changed nothing on either side — then the
	// exact rung already handled it.
	if (searchStripped.join('\n') === search && bufStripped.join('\n') === buffer) {
		return null;
	}

	// Precompute line start offsets in the original buffer.
	const lineStarts = new Int32Array(bufLines.length + 1);
	{
		let acc = 0;
		for (let i = 0; i < bufLines.length; i++) {
			lineStarts[i] = acc;
			acc += bufLines[i].length + 1; // +1 for '\n'
		}
		lineStarts[bufLines.length] = acc;
	}

	const matches: Array<{ from: number; to: number }> = [];
	const window = searchStripped.length;
	const maxStart = bufLines.length - window;
	for (let s = 0; s <= maxStart; s++) {
		let ok = true;
		for (let k = 0; k < window; k++) {
			if (bufStripped[s + k] !== searchStripped[k]) {
				ok = false;
				break;
			}
		}
		if (!ok) continue;
		const from = lineStarts[s];
		// end of last matched line — the last line has no trailing '\n' in the
		// window, so `to` is lineStarts[s+window] - 1 when another line follows.
		let to: number;
		if (s + window < bufLines.length) {
			to = lineStarts[s + window] - 1; // drop trailing '\n'
		} else {
			to = lineStarts[s + window];
		}
		matches.push({ from, to });
	}

	if (matches.length === 0) return null;
	if (matches.length > 1) return { kind: 'ambiguous', matches };
	return { kind: 'unique', from: matches[0].from, to: matches[0].to, via: 'ws' };
}

/* ------------------------------------------------------------------------- */
/* Rung 3: leading-blank-line tolerant                                       */
/* ------------------------------------------------------------------------- */

function rungLeadingBlank(buffer: string, search: string): AnchorResult | null {
	const trimmed = trimLeadingBlank(search);
	if (trimmed === search || trimmed.length === 0) return null;
	const idxs = findAllIndexes(buffer, trimmed);
	if (idxs.length === 0) return null;
	if (idxs.length > 1) {
		return {
			kind: 'ambiguous',
			matches: idxs.map((i) => ({ from: i, to: i + trimmed.length }))
		};
	}
	return {
		kind: 'unique',
		from: idxs[0],
		to: idxs[0] + trimmed.length,
		via: 'leading-blank'
	};
}

/* ------------------------------------------------------------------------- */
/* Rung 4: `...` elision expansion                                           */
/* ------------------------------------------------------------------------- */

/**
 * If SEARCH contains a bare `...` line, treat it as "anything here". We
 * anchor on head and tail segments and claim the span between the
 * earliest head match and a tail match that follows it.
 *
 * This rung is distinct from the F6 REPLACE-elision detector: here we let
 * the *model* elide context in SEARCH on purpose, matching Aider behaviour.
 */
function rungElision(buffer: string, search: string): AnchorResult | null {
	const lines = search.split('\n');
	let gapIdx = -1;
	for (let i = 0; i < lines.length; i++) {
		if (/^\s*\.{3,}\s*$/.test(lines[i])) {
			gapIdx = i;
			break;
		}
	}
	if (gapIdx < 0) return null;
	const head = lines.slice(0, gapIdx).join('\n');
	const tail = lines.slice(gapIdx + 1).join('\n');
	if (!head || !tail) return null;
	const headAt = buffer.indexOf(head);
	if (headAt < 0) return null;
	if (buffer.indexOf(head, headAt + head.length) >= 0) {
		return null; // ambiguous head; be conservative and fall through
	}
	const tailAt = buffer.indexOf(tail, headAt + head.length);
	if (tailAt < 0) return null;
	if (buffer.indexOf(tail, tailAt + tail.length) >= 0) {
		return null;
	}
	return {
		kind: 'unique',
		from: headAt,
		to: tailAt + tail.length,
		via: 'elision'
	};
}

/* ------------------------------------------------------------------------- */
/* Rung 5: similarity (LCS ratio ≥ 0.8)                                      */
/* ------------------------------------------------------------------------- */

/**
 * Longest-common-subsequence length between two strings, computed in O(n·m)
 * time and O(min(n,m)) space. Plenty fast for the block sizes we care about
 * (a few hundred chars × a few thousand chars of buffer window).
 */
function lcsLength(a: string, b: string): number {
	if (a.length === 0 || b.length === 0) return 0;
	// Make `b` the shorter one for memory.
	if (b.length > a.length) {
		const t = a;
		a = b;
		b = t;
	}
	const m = b.length;
	const prev = new Int32Array(m + 1);
	const cur = new Int32Array(m + 1);
	for (let i = 1; i <= a.length; i++) {
		const ca = a.charCodeAt(i - 1);
		for (let j = 1; j <= m; j++) {
			if (ca === b.charCodeAt(j - 1)) {
				cur[j] = prev[j - 1] + 1;
			} else {
				const u = prev[j];
				const l = cur[j - 1];
				cur[j] = u > l ? u : l;
			}
		}
		prev.set(cur);
		cur.fill(0);
	}
	return prev[m];
}

/** 2·LCS / (|a|+|b|) — matches difflib's SequenceMatcher.ratio() closely. */
function similarity(a: string, b: string): number {
	if (!a && !b) return 1;
	if (!a || !b) return 0;
	const l = lcsLength(a, b);
	return (2 * l) / (a.length + b.length);
}

/**
 * Slide a window the size of `search` across `buffer` at line boundaries
 * and score similarity. Returns the top-N by score.
 */
function similarityScan(
	buffer: string,
	search: string,
	threshold: number,
	topN: number
): Array<{ from: number; to: number; similarity: number }> {
	const searchLines = search.split('\n');
	const windowLines = searchLines.length;
	if (windowLines === 0) return [];
	const bufLines = buffer.split('\n');
	if (bufLines.length < windowLines) return [];

	// Precompute line start offsets so we can turn (startLine, endLine) into
	// absolute character ranges without re-scanning.
	const lineStarts = new Int32Array(bufLines.length + 1);
	{
		let acc = 0;
		for (let i = 0; i < bufLines.length; i++) {
			lineStarts[i] = acc;
			acc += bufLines[i].length + 1; // +1 for the \n we split on
		}
		lineStarts[bufLines.length] = acc;
	}

	const hits: Array<{ from: number; to: number; similarity: number }> = [];
	const maxStart = bufLines.length - windowLines;
	for (let start = 0; start <= maxStart; start++) {
		const window = bufLines.slice(start, start + windowLines).join('\n');
		const sim = similarity(window, search);
		if (sim >= threshold) {
			const from = lineStarts[start];
			const to = from + window.length;
			hits.push({ from, to, similarity: sim });
		}
	}
	hits.sort((a, b) => b.similarity - a.similarity);
	return hits.slice(0, topN);
}

function rungSimilarity(buffer: string, search: string): AnchorResult | null {
	const hits = similarityScan(buffer, search, 0.8, 5);
	if (hits.length === 0) {
		// Below threshold — surface the best-3 as *suggestions*, not matches,
		// so the 'none' branch in the caller can use them for the F2 prompt.
		return null;
	}
	// De-dupe near-duplicates: overlapping hits with the same score are often
	// the same region windowed at adjacent lines.
	const dedup: typeof hits = [];
	for (const h of hits) {
		const overlap = dedup.find((d) => d.from < h.to && h.from < d.to);
		if (!overlap) dedup.push(h);
	}
	if (dedup.length > 1) {
		return {
			kind: 'ambiguous',
			matches: dedup.map(({ from, to }) => ({ from, to }))
		};
	}
	const only = dedup[0];
	return { kind: 'unique', from: only.from, to: only.to, via: 'similarity' };
}

/* ------------------------------------------------------------------------- */
/* Entry point                                                               */
/* ------------------------------------------------------------------------- */

/**
 * Run the fuzz ladder against `buffer` for SEARCH text `search`.
 *
 * Ambiguity guard: per design §3.2 step 3, if the SEARCH target has < 10
 * non-whitespace chars, *any* multiple match is ambiguous. The exact-match
 * rung applies this directly; later rungs only produce a single best match
 * per scan, but when they tie (e.g. similarity rung returns two disjoint
 * top regions) we still flag ambiguous.
 */
export function findAnchor(buffer: string, search: string): AnchorResult {
	if (!search) {
		return { kind: 'none', suggestions: [] };
	}

	// Short-circuit: single multiple-match on the exact rung when the target
	// is too short to be unambiguous even in principle.
	if (nonWsLength(search) < 10 && countMatches(buffer, search) > 1) {
		const idxs = findAllIndexes(buffer, search);
		return {
			kind: 'ambiguous',
			matches: idxs.map((i) => ({ from: i, to: i + search.length }))
		};
	}

	const rungs: Array<(b: string, s: string) => AnchorResult | null> = [
		rungExact,
		rungWhitespace,
		rungLeadingBlank,
		rungElision,
		rungSimilarity
	];
	for (const rung of rungs) {
		const r = rung(buffer, search);
		if (!r) continue;
		if (r.kind === 'unique' || r.kind === 'ambiguous') return r;
	}
	// All rungs missed — return the best similarity suggestions (< 0.8) so
	// the repair prompt can show the user what the model *almost* matched.
	const suggestions = similarityScan(buffer, search, 0.0, 3);
	return { kind: 'none', suggestions };
}
