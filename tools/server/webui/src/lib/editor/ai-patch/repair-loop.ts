/**
 * Repair-loop orchestrator — pure logic layer for the reflection loop.
 *
 * Design brief §6.1 — "repair loop": when the dispatcher reports a failed
 * session with at least one repairable error, we inject a synthetic user
 * turn into the conversation that quotes the failed SEARCH block (or
 * equivalent context for other failure codes) and asks the model to try
 * again. `runPatchRepairLoop` (in `stream-hook.ts`) calls into
 * `formatRepairMessage` to produce that turn and into `injectRepairTurn`
 * to persist it.
 *
 * This module is intentionally I/O-free:
 *
 *   - No chat service calls. The caller owns the injection.
 *   - No Dexie, no DOM, no Svelte runes.
 *   - No max-reflection persistence. The orchestrator tracks the counter
 *     per user-initiated patch.
 *
 * Commit 4b — formats cover F2 (E_NO_MATCH), F3 (E_AMBIGUOUS), F6
 * (E_ELISION), F7 (E_SYNTAX_INVALID), F11 (E_MARKER_GRAMMAR), and F14
 * (E_BYTE_BUDGET). F4 (E_USER_EDIT) is intentionally excluded from the
 * repairable set — user-initiated edits over an in-flight block are a
 * manual override, not an auto-retry trigger.
 */

import type { SummarizedErrors } from './dispatcher';
import {
	PatchFailureCode,
	type PatchErrorRecord,
	type PatchSimilarRegion,
	type SearchReplaceBlock
} from './types';

/* ------------------------------------------------------------------------- */
/* Constants                                                                 */
/* ------------------------------------------------------------------------- */

/**
 * Max retry turns per user-initiated patch. Hardcoded for commit 4a; 4b
 * will lift this onto the settings store so the user can tune it.
 */
export const MAX_REFLECTIONS = 3;

/**
 * Similar-region quota embedded in a single repair prompt. The fuzz ladder
 * already truncates to 3 candidates at the `none` branch, so this is a
 * ceiling rather than a filter — we just never render more than this.
 */
export const MAX_SUGGESTIONS = 3;

/* ------------------------------------------------------------------------- */
/* Public types                                                              */
/* ------------------------------------------------------------------------- */

/**
 * Event surfaced by `RepairLoop.consume`. The caller decides what to do
 * with each: `repair-pending` → inject a new user turn; `repair-exhausted`
 * → surface a toast and stop; `no-repair` → the session succeeded (or
 * failed for a non-repairable reason) and the loop is done.
 */
export type RepairEvent =
	| {
			kind: 'repair-pending';
			/** Formatted user-turn body, ready to pass to the chat store. */
			body: string;
			/** 1-based reflection counter — mirrored on the injected message. */
			reflection: number;
			/** The first error we're asking the model to fix on this retry. */
			failureCode: PatchFailureCode;
			/** Stream-order block index the retry is aimed at. */
			blockIndex: number;
	  }
	| {
			kind: 'repair-exhausted';
			/** Reflection count at the moment of exhaustion. */
			reflection: number;
			/** Copy the caller can surface as-is in a toast. */
			toast: string;
	  }
	| { kind: 'no-repair' };

/**
 * Argument to `RepairLoop.consume`. Matches the fields the dispatcher's
 * `CommitResult` surfaces — we duplicate the shape here rather than
 * importing it so the loop can be driven from non-dispatcher callers too
 * (e.g. parser-level failures in commit 4b).
 */
export interface DispatcherFailure {
	errors: PatchErrorRecord[];
	/**
	 * The SEARCH/REPLACE blocks the parser extracted for this session, in
	 * stream order. Used by `formatRepairMessage` to quote the original
	 * SEARCH text when the block failed before anchoring (when
	 * `PatchErrorRecord.search` is unset).
	 */
	blocks?: SearchReplaceBlock[];
	/** Dispatcher-reported repairable flag. If explicitly `false`, we emit
	 *  `no-repair` without consulting the errors. */
	repairable?: boolean;
}

/* ------------------------------------------------------------------------- */
/* Formatting                                                                */
/* ------------------------------------------------------------------------- */

/**
 * Build the body of the synthetic user turn that asks the model to fix a
 * failed patch. Pure: string in, string out.
 *
 * Covers F2/F3/F6/F7/F11/F14. F4 (`E_USER_EDIT`) is intentionally NOT
 * handled here — when the user typed over an in-flight block we *do not*
 * auto-retry; the user has already taken over. The loop filters
 * `E_USER_EDIT` out before calling this function.
 */
export function formatRepairMessage(
	errors: PatchErrorRecord[],
	originalBlocks: SearchReplaceBlock[] = []
): string {
	if (errors.length === 0) {
		throw new Error('formatRepairMessage: no errors to repair');
	}

	// Drive the message off the *first* repairable error in stream order.
	// Fixing the earliest failure usually unblocks the rest on the retry.
	const primary = errors.find((e) => REPAIRABLE_CODES.has(e.code)) ?? errors[0];

	switch (primary.code) {
		case PatchFailureCode.E_NO_MATCH:
			return formatNoMatch(primary, originalBlocks);
		case PatchFailureCode.E_AMBIGUOUS:
			return formatAmbiguous(primary, originalBlocks);
		case PatchFailureCode.E_ELISION:
			return formatElision(primary, originalBlocks);
		case PatchFailureCode.E_SYNTAX_INVALID:
			return formatSyntaxInvalid(primary);
		case PatchFailureCode.E_MARKER_GRAMMAR:
			return formatMarkerGrammar(primary);
		case PatchFailureCode.E_BYTE_BUDGET:
			return formatByteBudget(primary);
		case PatchFailureCode.E_USER_EDIT:
			// Defensive: should have been filtered out upstream. The loop's
			// `repairableErrors` helper excludes this code, so hitting this
			// branch is a programmer error in the caller.
			throw new Error(
				'formatRepairMessage: E_USER_EDIT is not an auto-retryable failure — ' +
					'the user took over the block manually.'
			);
		case PatchFailureCode.E_NO_TARGET:
			// Commit 5 bootstrap code — never in REPAIRABLE_CODES because
			// a repeat model pass cannot invent a target buffer. Reaching
			// this branch would mean a caller tried to repair-format an
			// unrepairable failure.
			throw new Error(
				'formatRepairMessage: E_NO_TARGET is not an auto-retryable failure — ' +
					'no target buffer could be resolved for the SEARCH/REPLACE block.'
			);
	}
}

/* ------------------------------------------------------------------------- */
/* Per-code formatters                                                       */
/* ------------------------------------------------------------------------- */

function formatNoMatch(error: PatchErrorRecord, originalBlocks: SearchReplaceBlock[]): string {
	const blockIndex = error.blockIndex ?? 0;
	const search =
		error.search ??
		originalBlocks[blockIndex]?.search ??
		'(SEARCH text unavailable — the stream terminated before the separator)';

	const suggestions = (error.similar ?? []).slice(0, MAX_SUGGESTIONS);

	const header = [
		'The previous patch attempt failed: the SEARCH block below did not',
		'match any region of the target buffer. Please retry with a SEARCH',
		'that is character-identical to text that actually exists in the buffer.'
	].join(' ');

	const parts: string[] = [];
	parts.push('## Patch feedback');
	parts.push('');
	parts.push(header);
	parts.push('');
	parts.push(`Block ${blockIndex + 1} — failed SEARCH:`);
	parts.push(fence(search));

	if (suggestions.length > 0) {
		parts.push('');
		parts.push('Closest regions in the buffer (for reference — pick one or quote the');
		parts.push('buffer exactly):');
		for (let i = 0; i < suggestions.length; i++) {
			const s = suggestions[i];
			const pct = Math.round(s.similarity * 100);
			parts.push('');
			parts.push(`Candidate ${i + 1} (similarity ${pct}%):`);
			parts.push(fence(s.text));
		}
	} else {
		parts.push('');
		parts.push('No close candidates were found. The buffer may have diverged from');
		parts.push('what you expected — re-read the current content before retrying.');
	}

	parts.push('');
	parts.push(
		'Emit a new SEARCH/REPLACE block for this change using text that exists verbatim in the target.'
	);
	return parts.join('\n');
}

/**
 * F3 — SEARCH matched multiple regions. Quote the SEARCH, then for each
 * match show 3 lines of context above + the match (marked) + 3 lines
 * below. Cap at 4 matches; for 5+ append a "(N more...)" note.
 */
export const F3_MAX_MATCHES = 4;
export const F3_CONTEXT_LINES = 3;

function formatAmbiguous(error: PatchErrorRecord, originalBlocks: SearchReplaceBlock[]): string {
	const blockIndex = error.blockIndex ?? 0;
	const search = error.search ?? originalBlocks[blockIndex]?.search ?? '(SEARCH text unavailable)';

	const matches = (error.similar ?? []).slice();
	// Sort by start offset so the "Match 1 / Match 2" enumeration reads
	// top-to-bottom in buffer order. Callers already produce this order for
	// F3, but we sort defensively.
	matches.sort((a, b) => a.from - b.from);

	const shownMatches = matches.slice(0, F3_MAX_MATCHES);
	const hiddenCount = matches.length - shownMatches.length;

	const parts: string[] = [];
	parts.push('## Patch feedback');
	parts.push('');
	parts.push(
		'The previous patch attempt failed: the SEARCH block below matched multiple ' +
			'regions of the target buffer. SEARCH must anchor to exactly one location — ' +
			'please re-emit the block with more surrounding context so only one match survives.'
	);
	parts.push('');
	parts.push(`Block ${blockIndex + 1} — ambiguous SEARCH:`);
	parts.push(fence(search));
	parts.push('');
	parts.push(
		`Found ${matches.length} matching region${matches.length === 1 ? '' : 's'} in the buffer:`
	);

	const targetText = error.targetText ?? '';
	for (let i = 0; i < shownMatches.length; i++) {
		const m = shownMatches[i];
		const line = lineFromOffset(targetText, m.from);
		parts.push('');
		parts.push(`Match ${i + 1} — around line ${line}:`);
		parts.push(fence(buildMatchContext(targetText, m.from, m.to, F3_CONTEXT_LINES)));
	}

	if (hiddenCount > 0) {
		parts.push('');
		parts.push(`(${hiddenCount} more match${hiddenCount === 1 ? '' : 'es'} not shown)`);
	}

	parts.push('');
	parts.push(
		'Please re-emit the block with more surrounding context so exactly one match survives.'
	);
	return parts.join('\n');
}

function formatElision(error: PatchErrorRecord, originalBlocks: SearchReplaceBlock[]): string {
	const blockIndex = error.blockIndex ?? 0;
	const replace =
		error.replace ?? originalBlocks[blockIndex]?.replace ?? '(REPLACE text unavailable)';
	const elisionLine = error.elisionLine ?? 0;

	const parts: string[] = [];
	parts.push('## Patch feedback');
	parts.push('');
	parts.push(
		'The previous patch attempt failed: the REPLACE block below contains a ' +
			'lazy-elision placeholder (e.g. `...`, "existing code", "unchanged"). ' +
			'This tool cannot preserve elided content — please re-emit the block ' +
			'with the real content in place of the placeholder.'
	);
	parts.push('');
	parts.push(`Block ${blockIndex + 1} — REPLACE with placeholder marked:`);
	parts.push(fence(markElisionLine(replace, elisionLine)));
	parts.push('');
	parts.push(
		'Please re-emit the block with the real content in place of the placeholder — ' +
			'no `...`, no "existing code", no "unchanged" stubs.'
	);
	return parts.join('\n');
}

function formatSyntaxInvalid(error: PatchErrorRecord): string {
	const kind = error.artifactKind ?? 'html';
	const fenceLang = kind === 'svg' ? 'xml' : kind;
	const text = error.targetText ?? '';
	const line = error.syntaxLine;
	const column = error.syntaxColumn;

	const parts: string[] = [];
	parts.push('## Patch feedback');
	parts.push('');
	parts.push(
		'The patch committed, but the resulting document failed the syntax gate: ' +
			`the ${kind.toUpperCase()} parser reported the following error.`
	);
	parts.push('');
	parts.push('Parser message:');
	parts.push(fence(error.reason));

	if (line !== undefined && text) {
		const snippet = buildSyntaxContext(text, line, column ?? 1, 2);
		parts.push('');
		parts.push(`Near line ${line}${column !== undefined ? `, column ${column}` : ''}:`);
		parts.push(fenceLabel(fenceLang, snippet));
	} else if (text) {
		// No position available — quote the whole text if short, else tail it.
		const quote = text.length > 4000 ? text.slice(-4000) : text;
		parts.push('');
		parts.push('Committed document:');
		parts.push(fenceLabel(fenceLang, quote));
	}

	parts.push('');
	parts.push('Please fix the syntax at the marked position and re-emit the block.');
	return parts.join('\n');
}

function formatMarkerGrammar(error: PatchErrorRecord): string {
	const partialSearch = error.partialSearch ?? '';
	const partialReplace = error.partialReplace ?? '';

	const parts: string[] = [];
	parts.push('## Patch feedback');
	parts.push('');
	parts.push(
		'The previous patch attempt failed: your SEARCH/REPLACE markers were ' +
			'malformed and the streaming parser could not recover. The partial ' +
			'capture (what the parser managed to read before the break) is ' +
			'reproduced below for reference.'
	);
	parts.push('');
	parts.push(`Parser error: ${error.reason}`);

	if (partialSearch) {
		parts.push('');
		parts.push('Partial SEARCH (before the break):');
		parts.push(fence(partialSearch));
	}
	if (partialReplace) {
		parts.push('');
		parts.push('Partial REPLACE (before the break):');
		parts.push(fence(partialReplace));
	}
	if (!partialSearch && !partialReplace) {
		parts.push('');
		parts.push('The parser captured no content before the grammar break.');
	}

	parts.push('');
	parts.push(
		'Please re-emit the block using exactly `<<<<<<< SEARCH`, `=======`, and ' +
			'`>>>>>>> REPLACE` — 7 characters on each marker, no extras, no trailing ' +
			'text on the marker lines.'
	);
	return parts.join('\n');
}

function formatByteBudget(error: PatchErrorRecord): string {
	const budget = error.byteBudget;
	const parts: string[] = [];
	parts.push('## Patch feedback');
	parts.push('');
	parts.push(
		'The previous patch attempt failed: one of the SEARCH/REPLACE blocks exceeded ' +
			(budget !== undefined ? `the ${formatByteSize(budget)} byte budget.` : 'the byte budget.')
	);
	parts.push('');
	parts.push(
		'Please split this into smaller blocks — typically one function or section per ' +
			'SEARCH/REPLACE. Large blocks are brittle (more chance of whitespace drift) ' +
			'and harder for the tool to apply reliably.'
	);
	return parts.join('\n');
}

/* ------------------------------------------------------------------------- */
/* Fence helpers                                                             */
/* ------------------------------------------------------------------------- */

/**
 * Wrap a block of text in a fenced code block. We use a long-enough fence
 * that an arbitrary payload can't close it early; 4 backticks are enough
 * for anything that doesn't itself contain 4-backtick fences (which would
 * be pathological for SEARCH content). Wrapping suggestion text in a fence
 * also guarantees the renderer won't try to interpret embedded markdown.
 */
function fence(text: string): string {
	return '````\n' + text + '\n````';
}

function fenceLabel(lang: string, text: string): string {
	return '````' + lang + '\n' + text + '\n````';
}

/* ------------------------------------------------------------------------- */
/* Context helpers (F3 / F7)                                                 */
/* ------------------------------------------------------------------------- */

/**
 * Build a fenced-context snippet around a match region: N lines of
 * context above, the matching line(s) marked with `>>> `, N lines below.
 * `from` / `to` are UTF-16 offsets into `text`. The snippet respects file
 * edges — it never pads past start or end.
 */
function buildMatchContext(text: string, from: number, to: number, contextLines: number): string {
	if (!text) return '(buffer text unavailable)';
	const lines = text.split('\n');
	const startLine = lineFromOffset(text, from) - 1; // 0-based
	const endLine = lineFromOffset(text, Math.max(to - 1, from)) - 1;
	const ctxStart = Math.max(0, startLine - contextLines);
	const ctxEnd = Math.min(lines.length - 1, endLine + contextLines);
	const out: string[] = [];
	for (let i = ctxStart; i <= ctxEnd; i++) {
		const marker = i >= startLine && i <= endLine ? '>>> ' : '    ';
		out.push(marker + lines[i]);
	}
	return out.join('\n');
}

/**
 * 1-based line number at a given offset. Uses newline count up to `offset`.
 */
function lineFromOffset(text: string, offset: number): number {
	if (!text) return 1;
	let n = 1;
	const end = Math.min(offset, text.length);
	for (let i = 0; i < end; i++) {
		if (text.charCodeAt(i) === 10) n++;
	}
	return n;
}

/**
 * Mark the elision line inside a REPLACE payload with visible bookends so
 * the model can see exactly which line we think is a placeholder. `line`
 * is 1-based (from `ElisionHit.line`); defensive against out-of-range.
 */
function markElisionLine(text: string, line: number): string {
	if (line <= 0) return text;
	const lines = text.split('\n');
	const idx = Math.min(line - 1, lines.length - 1);
	if (idx < 0) return text;
	lines[idx] = '>>> ' + lines[idx] + ' <<<  (placeholder)';
	return lines.join('\n');
}

/**
 * Build a fenced syntax-error context: `context` lines around `line`,
 * plus a caret line under the `column` indicating the position. The
 * column caret is best-effort — some browsers return meaningless column
 * numbers, so we clamp to the line length.
 */
function buildSyntaxContext(text: string, line: number, column: number, context: number): string {
	const lines = text.split('\n');
	const idx = Math.min(Math.max(1, line), lines.length) - 1;
	const ctxStart = Math.max(0, idx - context);
	const ctxEnd = Math.min(lines.length - 1, idx + context);
	const out: string[] = [];
	for (let i = ctxStart; i <= ctxEnd; i++) {
		out.push(lines[i]);
		if (i === idx) {
			const caretColumn = Math.min(Math.max(1, column), (lines[i] ?? '').length + 1);
			out.push(' '.repeat(Math.max(0, caretColumn - 1)) + '^ here');
		}
	}
	return out.join('\n');
}

function formatByteSize(bytes: number): string {
	if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
	return `${bytes} B`;
}

/* ------------------------------------------------------------------------- */
/* State machine                                                             */
/* ------------------------------------------------------------------------- */

/**
 * Per-session retry budget + event emitter. Callers construct one when
 * they start an ai-patch session; each call to `consume` turns a
 * dispatcher outcome into a `RepairEvent` the caller can act on.
 *
 * Explicitly *not* a Svelte rune — callers that want reactivity wrap this
 * in their own state; the loop itself is plain JS.
 */
export class RepairLoop {
	readonly maxReflections: number;
	/** 0 until the first repair-pending; incremented before each emit. */
	private reflection = 0;
	/** Flipped true once we emit `repair-exhausted` or decide no-repair. */
	private done = false;

	constructor(maxReflections: number = MAX_REFLECTIONS) {
		if (maxReflections < 1) {
			throw new Error('RepairLoop: maxReflections must be >= 1');
		}
		this.maxReflections = maxReflections;
	}

	/** Current reflection count. 0 before any repair. */
	get reflectionCount(): number {
		return this.reflection;
	}

	/** Remaining reflections in the budget. */
	get remaining(): number {
		return Math.max(0, this.maxReflections - this.reflection);
	}

	/** True once the loop has stopped accepting further consume() calls. */
	get isDone(): boolean {
		return this.done;
	}

	/**
	 * Feed a dispatcher failure (or summarized equivalent) to the loop.
	 * Returns the event the caller should act on. Side-effect-free beyond
	 * updating internal counters.
	 *
	 * Contract:
	 *
	 *   - If the caller-supplied `repairable` flag is explicitly `false`,
	 *     we emit `no-repair` and mark done.
	 *   - If no errors are repairable by commit-4a's format, we emit
	 *     `no-repair` and mark done.
	 *   - Otherwise, if the budget is exhausted, we emit
	 *     `repair-exhausted` and mark done.
	 *   - Otherwise, we increment the reflection counter and emit
	 *     `repair-pending` with the formatted body.
	 */
	consume(failure: DispatcherFailure, summary?: SummarizedErrors | null): RepairEvent {
		if (this.done) {
			throw new Error('RepairLoop.consume: loop already ended');
		}

		// Short-circuit on explicit opt-out. The dispatcher already knows
		// whether any error is repairable; prefer its verdict over
		// re-scanning the list here.
		if (failure.repairable === false) {
			this.done = true;
			return { kind: 'no-repair' };
		}

		const repairable = repairableErrors(failure.errors, summary);
		if (repairable.length === 0) {
			this.done = true;
			return { kind: 'no-repair' };
		}

		if (this.reflection >= this.maxReflections) {
			this.done = true;
			return {
				kind: 'repair-exhausted',
				reflection: this.reflection,
				toast:
					`Patch repair gave up after ${this.maxReflections} attempts. Committed ` +
					`blocks (if any) stand; review the conversation and retry manually.`
			};
		}

		this.reflection += 1;
		const primary = repairable[0];
		const body = formatRepairMessage(failure.errors, failure.blocks ?? []);

		return {
			kind: 'repair-pending',
			body,
			reflection: this.reflection,
			failureCode: primary.code,
			blockIndex: primary.blockIndex ?? 0
		};
	}

	/**
	 * Reset the loop — used by test fixtures and (in 4b) the user-edit
	 * listener that treats a manual buffer mutation as fresh consent.
	 */
	reset(): void {
		this.reflection = 0;
		this.done = false;
	}
}

/* ------------------------------------------------------------------------- */
/* Helpers                                                                   */
/* ------------------------------------------------------------------------- */

/**
 * Codes the repair loop knows how to auto-retry. `E_USER_EDIT` is *not*
 * in this set: when the user manually edits inside an in-flight block we
 * treat it as their override, not as a failure to apologise for.
 *
 * Kept in sync with the dispatcher's equivalent set; both are the single
 * source of truth for "what counts as a retryable failure".
 */
export const REPAIRABLE_CODES: ReadonlySet<PatchFailureCode> = new Set([
	PatchFailureCode.E_NO_MATCH,
	PatchFailureCode.E_AMBIGUOUS,
	PatchFailureCode.E_ELISION,
	PatchFailureCode.E_SYNTAX_INVALID,
	PatchFailureCode.E_MARKER_GRAMMAR,
	PatchFailureCode.E_BYTE_BUDGET
]);

/**
 * Walk the error list and surface only those codes the repair format
 * knows how to handle. Commit 4b: F2 / F3 / F6 / F7 / F11 / F14.
 */
function repairableErrors(
	errors: PatchErrorRecord[],
	summary?: SummarizedErrors | null
): PatchErrorRecord[] {
	if (summary) return summary.repairable;
	return errors.filter((e) => REPAIRABLE_CODES.has(e.code));
}

// Re-export for convenience — the suggestion shape is repair-loop-adjacent
// and callers don't need a second import path to get at it.
export type { PatchSimilarRegion };
