/**
 * Streaming SEARCH/REPLACE parser — state machine from design brief §3.
 *
 * Scope: this is a pure lexer/recognizer. It turns a byte stream into
 * structured events. The dispatcher (a later commit) is responsible for
 * anchoring SEARCH text into the target buffer, opening CM6 transactions,
 * committing revisions, etc.
 *
 * States:
 *   IDLE                → awaiting opening `<<<<<<< SEARCH` (optional filename line above).
 *   COLLECTING_SEARCH   → accumulating SEARCH lines.
 *   AWAIT_SEPARATOR     → (implicit) SEARCH complete, about to enter REPLACE.
 *   COLLECTING_REPLACE  → accumulating REPLACE lines + streaming per-char chunks.
 *   BLOCK_DONE          → closed, back to IDLE for the next block.
 *
 * Line-buffered: we accumulate raw bytes until `\n`, then feed the completed
 * line to the state machine. Within COLLECTING_REPLACE we additionally
 * stream per-character `replace-chunk` events so the UI renders characters
 * as they arrive; we hold back chars that *might* be the start of the
 * closing `>>>>>>> REPLACE` marker, then commit or revoke them once the
 * line resolves.
 *
 * Event contract:
 *   - `replace-chunk` is fired once per character, in order, for every char
 *     of REPLACE content (including the `\n` at line ends). A consumer that
 *     concatenates every `chunk` field reconstructs the full REPLACE text
 *     exactly. The closing marker's characters are NOT emitted as chunks.
 *   - `replace-line` is fired once per complete REPLACE line (without the
 *     trailing `\n`). Use this for structured consumption; `replace-chunk`
 *     is for live UI.
 *
 * Malformed-marker recovery (design §8.1 fallback contract): tolerate minor
 * drift — extra `<` / `>`, wrong case, trailing whitespace. Emit
 * `parse-warning` when we recover; fall through to `parse-error` (F11) only
 * when the line is not plausibly a marker at all given the current state.
 */

import { PatchFailureCode, type ParserEvent, type SearchReplaceBlock } from './types';

type State = 'IDLE' | 'COLLECTING_SEARCH' | 'COLLECTING_REPLACE';

export interface StreamingPatchParserOptions {
	onEvent: (ev: ParserEvent) => void;
}

/* ------------------------------------------------------------------------- */
/* Marker recognition                                                        */
/* ------------------------------------------------------------------------- */

const STRICT_SEARCH = /^<{7}\s+SEARCH\s*$/;
const STRICT_SEPARATOR = /^={7}\s*$/;
const STRICT_REPLACE = /^>{7}\s+REPLACE\s*$/;

/** Relaxed: 5–10 angle brackets, optional space, SEARCH/REPLACE case-insensitive. */
const FUZZY_SEARCH = /^<{5,10}\s*SEARCH\b[ \t]*$/i;
const FUZZY_SEPARATOR = /^={5,10}\s*$/;
const FUZZY_REPLACE = /^>{5,10}\s*REPLACE\b[ \t]*$/i;

type MarkerKind = 'search-open' | 'separator' | 'replace-close';

interface MarkerMatch {
	kind: MarkerKind;
	strict: boolean;
	line: string;
}

function recogniseMarker(line: string): MarkerMatch | null {
	if (STRICT_SEARCH.test(line)) return { kind: 'search-open', strict: true, line };
	if (STRICT_SEPARATOR.test(line)) return { kind: 'separator', strict: true, line };
	if (STRICT_REPLACE.test(line)) return { kind: 'replace-close', strict: true, line };
	if (FUZZY_SEARCH.test(line)) return { kind: 'search-open', strict: false, line };
	if (FUZZY_SEPARATOR.test(line)) return { kind: 'separator', strict: false, line };
	if (FUZZY_REPLACE.test(line)) return { kind: 'replace-close', strict: false, line };
	return null;
}

/**
 * True when `tail` (a partial line in the stream buffer) could still resolve
 * into a closing `>>>>>>> REPLACE` marker once more bytes arrive. Used to
 * hold back per-char streaming of REPLACE content so we don't flash a `>`
 * that will immediately be revoked.
 */
function isPossibleCloseMarker(tail: string): boolean {
	if (!tail) return false;
	// Allow up to 10 angle brackets then any prefix of " REPLACE".
	return /^>{1,10}(?:[ \t]*(?:R(?:E(?:P(?:L(?:A(?:C(?:E)?)?)?)?)?)?)?[ \t]*)?$/i.test(tail);
}

/**
 * Conservative "looks like a filename" check for the optional line above
 * the SEARCH fence (design §2.3). No spaces (after trim), no weird chars,
 * at least one `.` or `/`.
 */
function looksLikeFilename(line: string): boolean {
	const t = line.trim();
	if (!t || t.length > 200) return false;
	if (/\s/.test(t)) return false;
	return /[./]/.test(t) && /^[\w./\\:-]+$/.test(t);
}

/* ------------------------------------------------------------------------- */
/* Parser                                                                    */
/* ------------------------------------------------------------------------- */

export class StreamingPatchParser {
	private readonly onEvent: (ev: ParserEvent) => void;
	private state: State = 'IDLE';
	/** Bytes received since the last `\n`. */
	private buf = '';
	/** How many characters of the current partial REPLACE line we've already streamed as chunks. */
	private streamedInTail = 0;
	/** Completed SEARCH lines. */
	private searchLines: string[] = [];
	/** Completed REPLACE lines. */
	private replaceLines: string[] = [];
	/** Pending filename candidate from IDLE state. */
	private pendingFilename: string | null = null;
	/** Filename for the currently-open block, if any. */
	private currentFilename: string | undefined;
	private finished = false;

	constructor(opts: StreamingPatchParserOptions) {
		this.onEvent = opts.onEvent;
	}

	feed(chunk: string): void {
		if (this.finished) return;
		if (!chunk) return;
		this.buf += chunk;

		while (true) {
			const nl = this.buf.indexOf('\n');
			if (nl < 0) break;
			const line = this.buf.slice(0, nl);
			this.buf = this.buf.slice(nl + 1);
			// If we were in COLLECTING_REPLACE and had pre-streamed part of
			// this tail, flush the remainder as chunks *before* emitting the
			// structured line — unless the completed line turns out to be the
			// close marker, in which case we must revoke. We can't literally
			// un-send chunks; the guard `isPossibleCloseMarker` ensures we
			// never emitted chunks for a line that is now revealed to be a
			// close marker.
			if (this.state === 'COLLECTING_REPLACE') {
				const marker = recogniseMarker(line);
				if (!marker || marker.kind !== 'replace-close') {
					// Safe to flush remaining tail chars as chunks + newline.
					if (line.length > this.streamedInTail) {
						this.onEvent({
							type: 'replace-chunk',
							chunk: line.slice(this.streamedInTail)
						});
					}
					this.onEvent({ type: 'replace-chunk', chunk: '\n' });
				}
			}
			this.streamedInTail = 0;
			this.processLine(line);
			if (this.finished) return;
		}

		// Partial tail: if in REPLACE and the tail is *not* a plausible close
		// marker prefix, stream the newly-arrived characters.
		if (this.state === 'COLLECTING_REPLACE' && this.buf.length > this.streamedInTail) {
			if (!isPossibleCloseMarker(this.buf)) {
				const delta = this.buf.slice(this.streamedInTail);
				this.onEvent({ type: 'replace-chunk', chunk: delta });
				this.streamedInTail = this.buf.length;
			}
		}
	}

	/** End-of-stream marker. Any buffered partial line is flushed if it's safe. */
	end(): void {
		if (this.finished) return;
		if (this.buf.length > 0) {
			// Only process a trailing line-without-newline in IDLE / SEARCH;
			// in REPLACE, an unterminated line signals F1 (stream-ended
			// mid-block) and the dispatcher handles that — we do nothing.
			if (this.state !== 'COLLECTING_REPLACE') {
				this.processLine(this.buf);
			}
			this.buf = '';
		}
		this.finished = true;
	}

	/* --------------------------------------------------------------------- */

	private processLine(line: string): void {
		const marker = recogniseMarker(line);

		switch (this.state) {
			case 'IDLE': {
				if (marker?.kind === 'search-open') {
					if (!marker.strict) {
						this.onEvent({
							type: 'parse-warning',
							reason: 'relaxed SEARCH marker',
							line: marker.line
						});
					}
					this.currentFilename = this.pendingFilename ?? undefined;
					this.pendingFilename = null;
					this.searchLines = [];
					this.replaceLines = [];
					this.state = 'COLLECTING_SEARCH';
					this.onEvent({ type: 'block-open', filename: this.currentFilename });
					return;
				}
				if (marker?.kind === 'separator' || marker?.kind === 'replace-close') {
					this.onEvent({
						type: 'parse-error',
						code: PatchFailureCode.E_MARKER_GRAMMAR,
						reason: `unexpected marker "${marker.kind}" in IDLE state`
					});
					return;
				}
				if (looksLikeFilename(line)) {
					this.pendingFilename = line.trim();
				} else if (line.trim().length > 0) {
					this.pendingFilename = null;
				}
				return;
			}

			case 'COLLECTING_SEARCH': {
				if (marker?.kind === 'separator') {
					if (!marker.strict) {
						this.onEvent({
							type: 'parse-warning',
							reason: 'relaxed separator',
							line: marker.line
						});
					}
					const search = this.searchLines.join('\n');
					this.onEvent({ type: 'search-complete', search });
					this.state = 'COLLECTING_REPLACE';
					return;
				}
				if (marker?.kind === 'replace-close') {
					this.onEvent({
						type: 'parse-error',
						code: PatchFailureCode.E_MARKER_GRAMMAR,
						reason: 'REPLACE close encountered before separator'
					});
					this.resetToIdle();
					return;
				}
				if (marker?.kind === 'search-open') {
					this.onEvent({
						type: 'parse-error',
						code: PatchFailureCode.E_MARKER_GRAMMAR,
						reason: 'nested SEARCH open inside SEARCH'
					});
					this.resetToIdle();
					return;
				}
				this.searchLines.push(line);
				this.onEvent({ type: 'search-line', line });
				return;
			}

			case 'COLLECTING_REPLACE': {
				if (marker?.kind === 'replace-close') {
					if (!marker.strict) {
						this.onEvent({
							type: 'parse-warning',
							reason: 'relaxed REPLACE close',
							line: marker.line
						});
					}
					const block: SearchReplaceBlock = {
						search: this.searchLines.join('\n'),
						replace: this.replaceLines.join('\n'),
						filename: this.currentFilename
					};
					this.onEvent({ type: 'block-close', block });
					this.resetToIdle();
					return;
				}
				if (marker?.kind === 'separator') {
					this.onEvent({
						type: 'parse-error',
						code: PatchFailureCode.E_MARKER_GRAMMAR,
						reason: 'duplicate separator inside REPLACE'
					});
					this.resetToIdle();
					return;
				}
				if (marker?.kind === 'search-open') {
					this.onEvent({
						type: 'parse-error',
						code: PatchFailureCode.E_MARKER_GRAMMAR,
						reason: 'SEARCH open inside REPLACE body'
					});
					this.resetToIdle();
					return;
				}
				this.replaceLines.push(line);
				this.onEvent({ type: 'replace-line', line });
				return;
			}
		}
	}

	private resetToIdle(): void {
		this.state = 'IDLE';
		this.searchLines = [];
		this.replaceLines = [];
		this.currentFilename = undefined;
		this.pendingFilename = null;
		this.streamedInTail = 0;
	}
}
