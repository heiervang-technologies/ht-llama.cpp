import { describe, expect, it } from 'vitest';
import {
	PatchFailureCode,
	StreamingPatchParser,
	type ParserEvent,
	type SearchReplaceBlock
} from '$lib/editor/ai-patch';

function collect(input: string, chunkSizes?: number[]): ParserEvent[] {
	const events: ParserEvent[] = [];
	const parser = new StreamingPatchParser({ onEvent: (ev) => events.push(ev) });
	if (!chunkSizes) {
		parser.feed(input);
	} else {
		let off = 0;
		for (const size of chunkSizes) {
			parser.feed(input.slice(off, off + size));
			off += size;
		}
		if (off < input.length) parser.feed(input.slice(off));
	}
	parser.end();
	return events;
}

const SIMPLE_BLOCK = [
	'<<<<<<< SEARCH',
	'old line one',
	'old line two',
	'=======',
	'new line one',
	'new line two',
	'>>>>>>> REPLACE',
	''
].join('\n');

describe('StreamingPatchParser — full-input happy path', () => {
	it('emits block-open, search lines, search-complete, replace lines, block-close', () => {
		const events = collect(SIMPLE_BLOCK);
		const types = events.map((e) => e.type);
		expect(types).toContain('block-open');
		expect(types).toContain('search-complete');
		expect(types).toContain('block-close');
		const close = events.find((e) => e.type === 'block-close');
		expect(close).toBeDefined();
		if (close && close.type === 'block-close') {
			const block: SearchReplaceBlock = close.block;
			expect(block.search).toBe('old line one\nold line two');
			expect(block.replace).toBe('new line one\nnew line two');
			expect(block.filename).toBeUndefined();
		}
	});

	it('records a filename from the line directly above the SEARCH marker', () => {
		const input = `src/foo.ts\n${SIMPLE_BLOCK}`;
		const events = collect(input);
		const open = events.find((e) => e.type === 'block-open');
		expect(open).toBeDefined();
		if (open && open.type === 'block-open') {
			expect(open.filename).toBe('src/foo.ts');
		}
	});

	it('drops a filename candidate if prose follows before the fence', () => {
		const input = `src/foo.ts\nSome prose about the change.\n${SIMPLE_BLOCK}`;
		const events = collect(input);
		const open = events.find((e) => e.type === 'block-open');
		if (open && open.type === 'block-open') {
			expect(open.filename).toBeUndefined();
		}
	});
});

describe('StreamingPatchParser — streaming boundaries', () => {
	it('produces the same block regardless of chunk boundaries (byte-by-byte)', () => {
		const chunkSizes = Array(SIMPLE_BLOCK.length).fill(1);
		const events = collect(SIMPLE_BLOCK, chunkSizes);
		const close = events.find((e) => e.type === 'block-close');
		expect(close).toBeDefined();
		if (close && close.type === 'block-close') {
			expect(close.block.search).toBe('old line one\nold line two');
			expect(close.block.replace).toBe('new line one\nnew line two');
		}
	});

	it('streams REPLACE chars via replace-chunk events', () => {
		const events = collect(SIMPLE_BLOCK);
		const chunks = events.filter((e) => e.type === 'replace-chunk');
		expect(chunks.length).toBeGreaterThan(0);
		const reconstructed = chunks.map((e) => (e.type === 'replace-chunk' ? e.chunk : '')).join('');
		expect(reconstructed).toBe('new line one\nnew line two\n');
	});

	it('does not leak the close-marker text into replace-chunk stream', () => {
		const events = collect(SIMPLE_BLOCK);
		const chunks = events
			.filter((e) => e.type === 'replace-chunk')
			.map((e) => (e.type === 'replace-chunk' ? e.chunk : ''))
			.join('');
		expect(chunks.includes('>>>>>>> REPLACE')).toBe(false);
		expect(chunks.includes('>>>>>>>')).toBe(false);
	});

	it('produces identical chunks regardless of chunk boundaries', () => {
		const wholeEvents = collect(SIMPLE_BLOCK);
		const wholeChunks = wholeEvents
			.filter((e) => e.type === 'replace-chunk')
			.map((e) => (e.type === 'replace-chunk' ? e.chunk : ''))
			.join('');

		const singleByte = collect(SIMPLE_BLOCK, Array(SIMPLE_BLOCK.length).fill(1));
		const sbChunks = singleByte
			.filter((e) => e.type === 'replace-chunk')
			.map((e) => (e.type === 'replace-chunk' ? e.chunk : ''))
			.join('');

		expect(sbChunks).toBe(wholeChunks);
	});
});

describe('StreamingPatchParser — multiple blocks', () => {
	it('emits two block-close events for two sequential blocks', () => {
		const input = SIMPLE_BLOCK + '\n' + SIMPLE_BLOCK;
		const events = collect(input);
		const closes = events.filter((e) => e.type === 'block-close');
		expect(closes.length).toBe(2);
	});
});

describe('StreamingPatchParser — malformed-marker recovery', () => {
	it('accepts `<<<<<<<<SEARCH` (extra bracket, no space) with a warning', () => {
		const input = ['<<<<<<<<SEARCH', 'old', '=======', 'new', '>>>>>>> REPLACE', ''].join('\n');
		const events = collect(input);
		const warn = events.find((e) => e.type === 'parse-warning');
		expect(warn).toBeDefined();
		const close = events.find((e) => e.type === 'block-close');
		expect(close).toBeDefined();
	});

	it('accepts lowercase `search` / `replace` with a warning', () => {
		const input = ['<<<<<<< search', 'old', '=======', 'new', '>>>>>>> replace', ''].join('\n');
		const events = collect(input);
		const warns = events.filter((e) => e.type === 'parse-warning');
		expect(warns.length).toBeGreaterThanOrEqual(2);
		const close = events.find((e) => e.type === 'block-close');
		expect(close).toBeDefined();
	});

	it('accepts separator of length 6 or 8', () => {
		const input = ['<<<<<<< SEARCH', 'old', '========', 'new', '>>>>>>> REPLACE', ''].join('\n');
		const events = collect(input);
		const close = events.find((e) => e.type === 'block-close');
		expect(close).toBeDefined();
	});
});

describe('StreamingPatchParser — contradictory / malformed blocks', () => {
	it('emits parse-error when `>>>>>>> REPLACE` appears before `=======`', () => {
		const input = ['<<<<<<< SEARCH', 'old', '>>>>>>> REPLACE', ''].join('\n');
		const events = collect(input);
		const err = events.find((e) => e.type === 'parse-error');
		expect(err).toBeDefined();
		if (err && err.type === 'parse-error') {
			expect(err.code).toBe(PatchFailureCode.E_MARKER_GRAMMAR);
		}
	});

	it('emits parse-error when a nested SEARCH open appears inside SEARCH', () => {
		const input = [
			'<<<<<<< SEARCH',
			'old',
			'<<<<<<< SEARCH',
			'other',
			'=======',
			'new',
			'>>>>>>> REPLACE',
			''
		].join('\n');
		const events = collect(input);
		const err = events.find((e) => e.type === 'parse-error');
		expect(err).toBeDefined();
	});

	it('emits parse-error for a duplicate separator inside REPLACE', () => {
		const input = [
			'<<<<<<< SEARCH',
			'old',
			'=======',
			'new',
			'=======',
			'more',
			'>>>>>>> REPLACE',
			''
		].join('\n');
		const events = collect(input);
		const err = events.find((e) => e.type === 'parse-error');
		expect(err).toBeDefined();
	});

	it('emits parse-error for an orphan `=======` in IDLE', () => {
		const input = ['some prose', '=======', 'more prose'].join('\n');
		const events = collect(input);
		const err = events.find((e) => e.type === 'parse-error');
		expect(err).toBeDefined();
		if (err && err.type === 'parse-error') {
			expect(err.code).toBe(PatchFailureCode.E_MARKER_GRAMMAR);
		}
	});

	it('does not emit a block-close when the stream ends mid-REPLACE (F1)', () => {
		const partial = ['<<<<<<< SEARCH', 'old', '=======', 'new line one', 'new line two'].join('\n');
		// No trailing newline, no close marker — EOS mid-block.
		const events = collect(partial);
		const close = events.find((e) => e.type === 'block-close');
		expect(close).toBeUndefined();
	});
});

describe('StreamingPatchParser — resilience to random chunk sizes', () => {
	function seededRandom(seed: number): () => number {
		let s = seed >>> 0;
		return () => {
			s = (s * 1664525 + 1013904223) >>> 0;
			return s / 0x100000000;
		};
	}

	it('produces the same block-close payload across 10 random chunkings', () => {
		const input = SIMPLE_BLOCK + '\n' + SIMPLE_BLOCK;
		const expected = collect(input).find((e) => e.type === 'block-close');
		expect(expected).toBeDefined();

		for (let seed = 1; seed <= 10; seed++) {
			const rnd = seededRandom(seed);
			const sizes: number[] = [];
			let remaining = input.length;
			while (remaining > 0) {
				const s = 1 + Math.floor(rnd() * 7);
				sizes.push(Math.min(s, remaining));
				remaining -= s;
			}
			const evs = collect(input, sizes);
			const closes = evs.filter((e) => e.type === 'block-close');
			expect(closes.length).toBe(2);
			for (const close of closes) {
				if (close.type === 'block-close') {
					expect(close.block.search).toBe('old line one\nold line two');
					expect(close.block.replace).toBe('new line one\nnew line two');
				}
			}
		}
	});
});
