import { describe, expect, it } from 'vitest';
import {
	DEFAULT_BYTE_BUDGET,
	LimitedPatchStream,
	PatchFailureCode,
	type ParserEvent
} from '$lib/editor/ai-patch';

function run(input: string, byteBudget?: number): ParserEvent[] {
	const events: ParserEvent[] = [];
	const stream = new LimitedPatchStream({ onEvent: (ev) => events.push(ev), byteBudget });
	stream.feed(input);
	stream.end();
	return events;
}

const SMALL_BLOCK = [
	'<<<<<<< SEARCH',
	'old one',
	'old two',
	'=======',
	'new one',
	'new two',
	'>>>>>>> REPLACE',
	''
].join('\n');

describe('LimitedPatchStream — under budget passes through', () => {
	it('forwards every structural event when payload fits the budget', () => {
		const events = run(SMALL_BLOCK);
		const types = events.map((e) => e.type);
		expect(types).toContain('block-open');
		expect(types).toContain('search-complete');
		expect(types).toContain('block-close');
		expect(types).not.toContain('parse-error');
	});

	it('respects a custom byteBudget that still fits', () => {
		const events = run(SMALL_BLOCK, 4096);
		expect(events.some((e) => e.type === 'block-close')).toBe(true);
		expect(events.some((e) => e.type === 'parse-error')).toBe(false);
	});

	it('has a 16 KB default budget', () => {
		expect(DEFAULT_BYTE_BUDGET).toBe(16 * 1024);
	});
});

describe('LimitedPatchStream — budget enforcement', () => {
	it('emits E_BYTE_BUDGET when combined SEARCH+REPLACE exceeds the budget', () => {
		// Budget of 50 chars — a 200-char REPLACE line will blow past it.
		const longLine = 'x'.repeat(200);
		const block = ['<<<<<<< SEARCH', 'old one', '=======', longLine, '>>>>>>> REPLACE', ''].join(
			'\n'
		);
		const events = run(block, 50);
		const err = events.find(
			(e) => e.type === 'parse-error' && e.code === PatchFailureCode.E_BYTE_BUDGET
		);
		expect(err).toBeDefined();
	});

	it('charges SEARCH content, not just REPLACE', () => {
		const bigSearch = 'q'.repeat(500);
		const block = ['<<<<<<< SEARCH', bigSearch, '=======', 'tiny', '>>>>>>> REPLACE', ''].join(
			'\n'
		);
		const events = run(block, 100);
		const err = events.find(
			(e) => e.type === 'parse-error' && e.code === PatchFailureCode.E_BYTE_BUDGET
		);
		expect(err).toBeDefined();
	});

	it('does not charge filename lines or markers against the budget', () => {
		// Filename + markers alone (~45 chars). With budget 100 and a 40-char
		// combined SEARCH+REPLACE payload it must pass.
		const block = [
			'some-file.ts',
			'<<<<<<< SEARCH',
			'abcdefghij',
			'=======',
			'klmnopqrst',
			'>>>>>>> REPLACE',
			''
		].join('\n');
		const events = run(block, 100);
		expect(events.some((e) => e.type === 'parse-error')).toBe(false);
		expect(events.some((e) => e.type === 'block-close')).toBe(true);
	});
});

describe('LimitedPatchStream — multi-block counter reset', () => {
	it('resets the charged counter between blocks', () => {
		// Two blocks, each ~60 chars of content, budget 80. Without a reset
		// the second block would trip. With the reset, both should succeed.
		const block = [
			'<<<<<<< SEARCH',
			'a'.repeat(20),
			'=======',
			'b'.repeat(20),
			'>>>>>>> REPLACE',
			''
		].join('\n');
		const twoBlocks = block + block;
		const events = run(twoBlocks, 80);
		const closes = events.filter((e) => e.type === 'block-close');
		expect(closes.length).toBe(2);
		expect(events.some((e) => e.type === 'parse-error')).toBe(false);
	});

	it('trips the second block when only the second exceeds the budget', () => {
		const smallBlock = ['<<<<<<< SEARCH', 'small', '=======', 'tiny', '>>>>>>> REPLACE', ''].join(
			'\n'
		);
		const fatBlock = [
			'<<<<<<< SEARCH',
			'whatever',
			'=======',
			'z'.repeat(500),
			'>>>>>>> REPLACE',
			''
		].join('\n');
		const events = run(smallBlock + fatBlock, 100);
		const errs = events.filter(
			(e) => e.type === 'parse-error' && e.code === PatchFailureCode.E_BYTE_BUDGET
		);
		expect(errs.length).toBe(1);
		// First block should have closed cleanly before the second tripped.
		const firstClose = events.findIndex((e) => e.type === 'block-close');
		const firstErr = events.findIndex(
			(e) => e.type === 'parse-error' && e.code === PatchFailureCode.E_BYTE_BUDGET
		);
		expect(firstClose).toBeLessThan(firstErr);
	});
});
