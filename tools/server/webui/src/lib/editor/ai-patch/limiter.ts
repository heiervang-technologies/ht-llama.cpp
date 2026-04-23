/**
 * Byte-budget limiter — preprocessing wrapper around StreamingPatchParser.
 *
 * Protects downstream consumers (shadow-doc, UI, persistence) from a
 * malfunctioning assistant that emits an unbounded SEARCH or REPLACE payload.
 * We track cumulative SEARCH+REPLACE characters per *open block*: once the
 * budget is exceeded, we emit a synthetic `parse-error` (F14 /
 * `E_BYTE_BUDGET`) and drop further bytes until the block closes. Counter
 * resets on every `block-open`.
 *
 * Scope: this is a soft guard on stream volume, not a security boundary. The
 * hard guard is at the HTTP layer. 16 KB is the default because real-world
 * artifact-scale edits (a single HTML/SVG/code block) sit well under that,
 * while still leaving headroom for a stray long-line regeneration. Callers
 * operating on larger buffers pass `{ byteBudget }`.
 *
 * Non-goals:
 *   - Counting bytes of anything outside SEARCH/REPLACE content. Filename
 *     lines, markers, warnings, etc. are not charged.
 *   - Token-accurate accounting. UTF-16 code unit count is close enough and
 *     avoids pulling in a TextEncoder per chunk.
 */

import { StreamingPatchParser } from './parser';
import { PatchFailureCode, type ParserEvent } from './types';

export const DEFAULT_BYTE_BUDGET = 16 * 1024;

export interface LimitedPatchStreamOptions {
	onEvent: (ev: ParserEvent) => void;
	/** Max combined SEARCH+REPLACE chars per block. Defaults to 16 KB. */
	byteBudget?: number;
}

/**
 * Streaming wrapper: feed `.feed(chunk)` / `.end()` exactly as you would the
 * raw parser. Observer signature matches `StreamingPatchParser` — same events
 * flow through, plus a potential extra `parse-error` with code
 * `E_BYTE_BUDGET` when the budget is tripped.
 */
export class LimitedPatchStream {
	private readonly parser: StreamingPatchParser;
	private readonly onEvent: (ev: ParserEvent) => void;
	private readonly budget: number;
	/** Characters charged against the current block's budget. */
	private charged = 0;
	/** Once tripped, we ignore further SEARCH/REPLACE content until block-close. */
	private tripped = false;

	constructor(opts: LimitedPatchStreamOptions) {
		this.onEvent = opts.onEvent;
		this.budget = opts.byteBudget ?? DEFAULT_BYTE_BUDGET;
		this.parser = new StreamingPatchParser({
			onEvent: (ev) => this.handleEvent(ev)
		});
	}

	feed(chunk: string): void {
		this.parser.feed(chunk);
	}

	end(): void {
		this.parser.end();
	}

	/**
	 * Configured byte budget (in UTF-16 code units) for a single block.
	 * Exposed so downstream error records can quote the limit verbatim in
	 * the repair-format prompt (F14). Read-only; the budget is fixed at
	 * construction time.
	 */
	get byteBudget(): number {
		return this.budget;
	}

	/* --------------------------------------------------------------------- */

	private handleEvent(ev: ParserEvent): void {
		switch (ev.type) {
			case 'block-open':
				this.charged = 0;
				this.tripped = false;
				this.onEvent(ev);
				return;

			case 'search-line':
				// +1 for the implicit newline that separates SEARCH lines.
				this.charge(ev.line.length + 1);
				if (this.tripped) return;
				this.onEvent(ev);
				return;

			case 'search-complete':
				// No additional charge here — the content was already counted
				// line-by-line. Forward verbatim unless tripped.
				if (this.tripped) return;
				this.onEvent(ev);
				return;

			case 'replace-line':
				this.charge(ev.line.length + 1);
				if (this.tripped) return;
				this.onEvent(ev);
				return;

			case 'replace-chunk':
				this.charge(ev.chunk.length);
				if (this.tripped) return;
				this.onEvent(ev);
				return;

			case 'block-close':
				// Forward close regardless so downstream state machines can
				// reset even on a tripped block, then reset our counters.
				if (!this.tripped) this.onEvent(ev);
				this.charged = 0;
				this.tripped = false;
				return;

			case 'parse-warning':
			case 'parse-error':
				this.onEvent(ev);
				return;
		}
	}

	private charge(n: number): void {
		if (this.tripped) return;
		this.charged += n;
		if (this.charged > this.budget) {
			this.tripped = true;
			this.onEvent({
				type: 'parse-error',
				code: PatchFailureCode.E_BYTE_BUDGET,
				reason: `SEARCH+REPLACE exceeded byte budget (${this.charged} > ${this.budget})`
			});
		}
	}
}
