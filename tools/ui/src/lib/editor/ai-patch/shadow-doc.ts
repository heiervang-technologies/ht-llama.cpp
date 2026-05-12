/**
 * ShadowDoc — per-session buffer primitive that stages SEARCH/REPLACE edits
 * as they stream in.
 *
 * The parser produces events; findAnchor turns SEARCH text into a
 * (from, to) range in the base text; this primitive then:
 *
 *   1. applyAnchor(from, to)   — deletes the SEARCH range, remembers the
 *                                insertion offset for subsequent chunks.
 *   2. appendChunk(text)       — appends streaming REPLACE bytes at the
 *                                current insertion point.
 *   3. closeBlock()            — finalises the block, shifting the insertion
 *                                cursor so follow-on blocks anchor against
 *                                an up-to-date buffer.
 *
 * We wrap `@codemirror/state`'s `Text` because the CM6 mount in commit 3 will
 * consume the emitted `ChangeSet`s directly — no translation layer needed.
 * For this commit the consumer just calls `toString()` at session end.
 *
 * Invariants (enforced by assertions in the test suite, upheld here):
 *   - Between `applyAnchor` and `closeBlock`, `appendChunk` may be called
 *     any number of times; concatenating the chunks yields exactly the
 *     text the shadow buffer now holds at that insertion range.
 *   - `applyAnchor` on the same block is idempotent per-block *only* to the
 *     extent that the parser never fires it twice; we throw if misused.
 *   - `closeBlock` resets the in-block state but not the underlying buffer
 *     — multi-block patches compose left-to-right across the evolving text.
 */

import { ChangeSet, Text } from '@codemirror/state';

export interface ShadowBlockSummary {
	from: number;
	/** Where the replacement text ends after the block closes. */
	to: number;
	/** Concatenated chunks that were appended during this block. */
	inserted: string;
}

export class ShadowDoc {
	private text: Text;
	/** Raw ChangeSets emitted across the lifetime of the session — the CM6 mount
	 *  in commit 3 consumes these directly. Each entry's pre-image length
	 *  matches the buffer length *at the time it was created*, so consumers
	 *  apply them in order or compose them themselves. */
	private readonly changes: ChangeSet[] = [];
	/** Insertion offset inside the *current* buffer (post-anchor deletion). */
	private insertAt: number | null = null;
	/** Characters appended to the current block. */
	private inBlockLen = 0;
	private inBlockStart = 0;
	private inBlockChunks: string[] = [];

	constructor(initial: string) {
		this.text = Text.of(initial.split('\n'));
	}

	/** Current shadow buffer contents. */
	toString(): string {
		return this.text.toString();
	}

	/** Total length of the current buffer (chars). */
	get length(): number {
		return this.text.length;
	}

	/**
	 * Ordered list of raw ChangeSets produced during the session. Each is
	 * valid against the buffer length *immediately before* it was applied —
	 * exactly what CM6 wants when replaying a sequence of transactions.
	 */
	getChanges(): readonly ChangeSet[] {
		return this.changes;
	}

	/**
	 * Concatenated REPLACE text currently staged in the open block, or the
	 * empty string if no block is in flight. The CM6 bridge reads this on
	 * every chunk to update its in-flight widget without touching private
	 * state.
	 */
	currentBlockText(): string {
		return this.inBlockChunks.join('');
	}

	/** Zero-based offset at which the currently-open block's insertion begins,
	 *  or `null` if no block is in flight. */
	currentBlockStart(): number | null {
		return this.insertAt === null ? null : this.inBlockStart;
	}

	/**
	 * Compute the net single `ChangeSet` equivalent of the whole session so
	 * far, against the supplied `baseDoc` (which must be the session-start
	 * baseline — i.e. what the shadow was initialised with).
	 *
	 * The CM6 bridge doesn't use this — it dispatches a block-scoped
	 * ChangeSet at `commit()` time. The helper exists for headless
	 * consumers (virtualised artifacts, driver tests) that want to replay
	 * "what the bridge would have done" as one transaction.
	 */
	getChangeSet(baseDoc: Text): ChangeSet {
		return ChangeSet.of({ from: 0, to: baseDoc.length, insert: this.toString() }, baseDoc.length);
	}

	/**
	 * Called when the dispatcher has anchored the current block's SEARCH text
	 * into the buffer. Deletes the SEARCH range and parks the insertion
	 * cursor at `from` ready for streaming chunks.
	 */
	applyAnchor(from: number, to: number): void {
		if (this.insertAt !== null) {
			throw new Error('ShadowDoc.applyAnchor: anchor already active — close the block first');
		}
		if (from < 0 || to < from || to > this.text.length) {
			throw new Error(
				`ShadowDoc.applyAnchor: invalid range ${from}..${to} for buffer of length ${this.text.length}`
			);
		}
		if (to > from) {
			const change = ChangeSet.of({ from, to }, this.text.length);
			this.changes.push(change);
			this.text = change.apply(this.text);
		}
		this.insertAt = from;
		this.inBlockStart = from;
		this.inBlockLen = 0;
		this.inBlockChunks = [];
	}

	/**
	 * Append a streaming REPLACE chunk at the current insertion offset. Safe
	 * to call for single characters or larger runs — the parser may fire at
	 * either granularity.
	 */
	appendChunk(chunk: string): void {
		if (this.insertAt === null) {
			throw new Error('ShadowDoc.appendChunk: no active anchor — call applyAnchor first');
		}
		if (!chunk) return;
		const at = this.insertAt + this.inBlockLen;
		const change = ChangeSet.of({ from: at, to: at, insert: chunk }, this.text.length);
		this.changes.push(change);
		this.text = change.apply(this.text);
		this.inBlockLen += chunk.length;
		this.inBlockChunks.push(chunk);
	}

	/**
	 * Finalise the current block. Returns a summary describing the span we
	 * occupied so callers (future CM6 decorations) can reason about block
	 * boundaries. No-op if no anchor was applied — some blocks fail anchoring
	 * and are discarded by the dispatcher before reaching us.
	 *
	 * `finalReplace`, when provided, is the authoritative REPLACE payload as
	 * the parser observed it at block-close. The streaming chunk path picks
	 * up a trailing `\n` (the one that terminates the final REPLACE line
	 * before the close marker) that is NOT part of `SearchReplaceBlock.replace`;
	 * we reconcile by rewriting the block's inserted range to match
	 * `finalReplace` exactly. No-op when the streamed chunks already agree.
	 */
	closeBlock(finalReplace?: string): ShadowBlockSummary | null {
		if (this.insertAt === null) return null;
		let inserted = this.inBlockChunks.join('');
		if (finalReplace !== undefined && inserted !== finalReplace) {
			const start = this.inBlockStart;
			const end = start + this.inBlockLen;
			const change = ChangeSet.of({ from: start, to: end, insert: finalReplace }, this.text.length);
			this.changes.push(change);
			this.text = change.apply(this.text);
			this.inBlockLen = finalReplace.length;
			this.inBlockChunks = [finalReplace];
			inserted = finalReplace;
		}
		const summary: ShadowBlockSummary = {
			from: this.inBlockStart,
			to: this.inBlockStart + this.inBlockLen,
			inserted
		};
		this.insertAt = null;
		this.inBlockLen = 0;
		this.inBlockChunks = [];
		return summary;
	}

	/** True when a block is currently open (between applyAnchor and closeBlock). */
	hasOpenBlock(): boolean {
		return this.insertAt !== null;
	}
}
