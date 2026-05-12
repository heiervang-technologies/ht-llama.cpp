/**
 * formatRepairMessage — body-formatting tests for the synthetic user turn
 * the repair loop injects on F2 failures.
 *
 * We assert against the contract the model sees: the failed SEARCH text
 * appears verbatim, the fuzz-ladder suggestions are quoted, and the
 * framing makes the "patch feedback" nature obvious to a human reader
 * scanning the conversation.
 */

import { describe, expect, it } from 'vitest';
import {
	formatRepairMessage,
	PatchFailureCode,
	type PatchErrorRecord,
	type SearchReplaceBlock
} from '$lib/editor/ai-patch';

describe('formatRepairMessage (F2)', () => {
	const FAILED_SEARCH = ["const greeting = 'hello world';", 'console.log(greeting);'].join('\n');

	const SIMILAR_1 = ["const greeting = 'hello, world!';", 'console.log(greeting);'].join('\n');

	const SIMILAR_2 = ["const farewell = 'goodbye';", 'console.log(farewell);'].join('\n');

	function noMatch(): PatchErrorRecord {
		return {
			code: PatchFailureCode.E_NO_MATCH,
			reason: 'anchor none',
			blockIndex: 2,
			search: FAILED_SEARCH,
			similar: [
				{ from: 0, to: SIMILAR_1.length, similarity: 0.92, text: SIMILAR_1 },
				{ from: 100, to: 100 + SIMILAR_2.length, similarity: 0.41, text: SIMILAR_2 }
			]
		};
	}

	it('quotes the failed SEARCH verbatim inside a fenced block', () => {
		const body = formatRepairMessage([noMatch()]);
		expect(body).toContain(FAILED_SEARCH);
		// Must be wrapped in a fence — check both directions so a partial
		// match doesn't slip through.
		const idx = body.indexOf(FAILED_SEARCH);
		expect(body.slice(0, idx)).toMatch(/````/);
		expect(body.slice(idx + FAILED_SEARCH.length)).toMatch(/````/);
	});

	it('lists the top-N similar regions with percentage similarity', () => {
		const body = formatRepairMessage([noMatch()]);
		expect(body).toContain(SIMILAR_1);
		expect(body).toContain(SIMILAR_2);
		expect(body).toContain('Candidate 1');
		expect(body).toContain('Candidate 2');
		expect(body).toMatch(/92%/);
		expect(body).toMatch(/41%/);
	});

	it('references the failing block by 1-based index', () => {
		const body = formatRepairMessage([noMatch()]);
		// blockIndex=2 → "Block 3"
		expect(body).toContain('Block 3');
	});

	it('labels the turn as patch feedback so the human reader can skim past it', () => {
		const body = formatRepairMessage([noMatch()]);
		expect(body).toMatch(/Patch feedback/i);
		expect(body.toLowerCase()).toContain('search');
	});

	it('falls back to the original block text when the error has no search field', () => {
		const err: PatchErrorRecord = {
			code: PatchFailureCode.E_NO_MATCH,
			reason: 'anchor none',
			blockIndex: 0
		};
		const blocks: SearchReplaceBlock[] = [
			{ search: 'fallback-search-text', replace: 'irrelevant' }
		];
		const body = formatRepairMessage([err], blocks);
		expect(body).toContain('fallback-search-text');
	});

	it('surfaces a "no candidates" note when the fuzz ladder returned nothing', () => {
		const err: PatchErrorRecord = {
			code: PatchFailureCode.E_NO_MATCH,
			reason: 'anchor none',
			blockIndex: 0,
			search: 'totally absent text',
			similar: []
		};
		const body = formatRepairMessage([err]);
		expect(body).toMatch(/No close candidates/i);
	});

	it('refuses E_USER_EDIT — manual user edits are not auto-retried', () => {
		const userEdit: PatchErrorRecord = {
			code: PatchFailureCode.E_USER_EDIT,
			reason: 'user edit aborted block',
			blockIndex: 0
		};
		expect(() => formatRepairMessage([userEdit])).toThrow(/not an auto-retryable/);
	});

	it('throws when called with an empty error list', () => {
		expect(() => formatRepairMessage([])).toThrow(/no errors/);
	});
});

describe('formatRepairMessage (F3 — ambiguous)', () => {
	const TARGET = [
		'function hello() {',
		'  return "world";',
		'}',
		'',
		'function hello() {',
		'  return "galaxy";',
		'}',
		'',
		'// end'
	].join('\n');

	it('quotes the SEARCH and enumerates each match with context', () => {
		const search = 'function hello() {';
		const firstFrom = TARGET.indexOf(search);
		const secondFrom = TARGET.indexOf(search, firstFrom + 1);
		const body = formatRepairMessage([
			{
				code: PatchFailureCode.E_AMBIGUOUS,
				reason: 'anchor ambiguous',
				blockIndex: 0,
				search,
				similar: [
					{ from: firstFrom, to: firstFrom + search.length, similarity: 1, text: search },
					{ from: secondFrom, to: secondFrom + search.length, similarity: 1, text: search }
				],
				targetText: TARGET
			}
		]);
		expect(body).toContain('Match 1');
		expect(body).toContain('Match 2');
		expect(body).toMatch(/Found 2 matching/);
		// Matching line must be marked with the `>>>` prefix.
		expect(body).toMatch(/>>> function hello\(\)/);
		// Context line above is not marked.
		expect(body).not.toMatch(/>>> \/\/ end/);
	});

	it('caps at 4 matches and appends an elision note for the rest', () => {
		// Construct 5 hits against a synthetic target. We don't care that
		// the target text is repetitive for this assertion — only the
		// "(1 more…)" footer.
		const hits = Array.from({ length: 5 }, (_, i) => ({
			from: i * 10,
			to: i * 10 + 5,
			similarity: 1,
			text: 'alpha'
		}));
		const body = formatRepairMessage([
			{
				code: PatchFailureCode.E_AMBIGUOUS,
				reason: 'anchor ambiguous',
				blockIndex: 0,
				search: 'alpha',
				similar: hits,
				targetText: 'alpha     alpha     alpha     alpha     alpha'
			}
		]);
		expect(body).toContain('Found 5 matching');
		expect(body).toContain('Match 4');
		expect(body).not.toContain('Match 5');
		expect(body).toMatch(/\(1 more match[^)]*not shown\)/);
	});

	it('does not pad context past the start of the buffer (single-line SEARCH at file edge)', () => {
		const tgt = 'alpha\nbeta';
		const body = formatRepairMessage([
			{
				code: PatchFailureCode.E_AMBIGUOUS,
				reason: 'anchor ambiguous',
				blockIndex: 0,
				search: 'alpha',
				similar: [{ from: 0, to: 5, similarity: 1, text: 'alpha' }],
				targetText: tgt
			}
		]);
		// With a single-line match at offset 0 and 3 lines of context, we
		// must not emit leading blank/padded lines (the buffer only has two
		// lines). The match must appear marked on the first context line.
		expect(body).toMatch(/>>> alpha/);
		expect(body).toMatch(/ {4}beta/);
	});
});

describe('formatRepairMessage (F6 — elision)', () => {
	it('marks the placeholder line and asks for the real content', () => {
		const replace = [
			'function build() {',
			'  // ... existing code ...',
			'  return value;',
			'}'
		].join('\n');
		const body = formatRepairMessage([
			{
				code: PatchFailureCode.E_ELISION,
				reason: 'elision at line 2',
				blockIndex: 0,
				replace,
				elisionLine: 2
			}
		]);
		expect(body).toContain('lazy-elision');
		expect(body).toMatch(/>>> .*existing code.*<<< {2}\(placeholder\)/);
		expect(body).toContain('return value;'); // other lines survive untouched
	});
});

describe('formatRepairMessage (F7 — syntax invalid)', () => {
	it('quotes the parser message and context with a caret at the column', () => {
		const text = ['<!doctype html>', '<html>', '<body>', '  <p>oops', '</body>', '</html>'].join(
			'\n'
		);
		const body = formatRepairMessage([
			{
				code: PatchFailureCode.E_SYNTAX_INVALID,
				reason: 'error on line 4, column 7: malformed start tag',
				targetText: text,
				artifactKind: 'html',
				syntaxLine: 4,
				syntaxColumn: 7
			}
		]);
		expect(body).toMatch(/Near line 4, column 7/);
		expect(body).toMatch(/malformed start tag/);
		// Caret alignment — column 7 → 6 leading spaces then "^ here"
		expect(body).toMatch(/ {6}\^ here/);
		// Fenced with html label
		expect(body).toContain('````html');
	});

	it('falls back to quoting the whole text when no position is available', () => {
		const text = '<svg><g><rect/></g>'; // missing </svg>
		const body = formatRepairMessage([
			{
				code: PatchFailureCode.E_SYNTAX_INVALID,
				reason: 'premature end of data',
				targetText: text,
				artifactKind: 'svg'
			}
		]);
		expect(body).toContain(text);
		expect(body).toContain('````xml');
	});
});

describe('formatRepairMessage (F11 — marker grammar)', () => {
	it('quotes partial SEARCH / REPLACE and reminds the model of the marker grammar', () => {
		const body = formatRepairMessage([
			{
				code: PatchFailureCode.E_MARKER_GRAMMAR,
				reason: 'duplicate separator inside REPLACE',
				blockIndex: 0,
				partialSearch: 'const x = 1;',
				partialReplace: 'const x = 2;'
			}
		]);
		expect(body).toContain('const x = 1;');
		expect(body).toContain('const x = 2;');
		expect(body).toMatch(/<<<<<<< SEARCH/);
		expect(body).toMatch(/>>>>>>> REPLACE/);
	});

	it('handles the zero-capture case gracefully', () => {
		const body = formatRepairMessage([
			{
				code: PatchFailureCode.E_MARKER_GRAMMAR,
				reason: 'unexpected separator',
				blockIndex: 0
			}
		]);
		expect(body).toContain('captured no content');
	});
});

describe('formatRepairMessage (F14 — byte budget)', () => {
	it('reports the budget in KB and asks for smaller blocks', () => {
		const body = formatRepairMessage([
			{
				code: PatchFailureCode.E_BYTE_BUDGET,
				reason: 'block exceeded 16384 B budget',
				blockIndex: 0,
				byteBudget: 16 * 1024
			}
		]);
		expect(body).toMatch(/16 KB/);
		expect(body).toMatch(/split this into smaller blocks/);
	});
});
