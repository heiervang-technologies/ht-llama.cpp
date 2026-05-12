import { describe, expect, it } from 'vitest';
import { findAnchor } from '$lib/editor/ai-patch/fuzz-match';

describe('findAnchor — exact rung', () => {
	it('returns unique:exact for an unambiguous exact match', () => {
		const buf = 'alpha\nbeta\ngamma delta epsilon\n';
		const search = 'gamma delta epsilon';
		const r = findAnchor(buf, search);
		expect(r.kind).toBe('unique');
		if (r.kind === 'unique') {
			expect(r.via).toBe('exact');
			expect(buf.slice(r.from, r.to)).toBe(search);
		}
	});

	it('returns ambiguous for >1 exact match of a long search', () => {
		const buf = ['function foo() { return x; }', 'function foo() { return x; }'].join('\n');
		const search = 'function foo() { return x; }';
		const r = findAnchor(buf, search);
		expect(r.kind).toBe('ambiguous');
		if (r.kind === 'ambiguous') {
			expect(r.matches.length).toBe(2);
		}
	});

	it('returns ambiguous for any >1 match of a short (<10 non-ws) search', () => {
		const buf = 'ab ab ab';
		const r = findAnchor(buf, 'ab');
		expect(r.kind).toBe('ambiguous');
		if (r.kind === 'ambiguous') {
			expect(r.matches.length).toBe(3);
		}
	});
});

describe('findAnchor — whitespace rung', () => {
	it('matches when SEARCH has less indent than buffer', () => {
		const buf = '    const x = 1;\n    const y = 2;\n';
		const search = 'const x = 1;\nconst y = 2;';
		const r = findAnchor(buf, search);
		expect(r.kind).toBe('unique');
		if (r.kind === 'unique') {
			expect(['ws', 'exact']).toContain(r.via);
			expect(buf.slice(r.from, r.to)).toContain('const x = 1;');
		}
	});
});

describe('findAnchor — leading-blank rung', () => {
	it('tolerates leading blank lines in SEARCH', () => {
		const buf = 'apple\nbanana\ncherry\n';
		const search = '\n\nbanana\ncherry';
		const r = findAnchor(buf, search);
		expect(r.kind).toBe('unique');
		if (r.kind === 'unique') {
			expect(['leading-blank', 'exact', 'ws']).toContain(r.via);
			expect(buf.slice(r.from, r.to).includes('banana\ncherry')).toBe(true);
		}
	});
});

describe('findAnchor — elision rung', () => {
	it('expands `...` between unique head and tail', () => {
		const buf = ['function header() {', '  a();', '  b();', '  c();', '  d();', '}'].join('\n');
		const search = ['function header() {', '...', '}'].join('\n');
		const r = findAnchor(buf, search);
		expect(r.kind).toBe('unique');
		if (r.kind === 'unique') {
			expect(r.via).toBe('elision');
			expect(buf.slice(r.from, r.to).startsWith('function header() {')).toBe(true);
			expect(buf.slice(r.from, r.to).endsWith('}')).toBe(true);
		}
	});

	it('does not fire elision rung when head is ambiguous', () => {
		const buf = [
			'function foo() {',
			'  return 1;',
			'}',
			'function foo() {',
			'  return 2;',
			'}'
		].join('\n');
		const search = ['function foo() {', '...', '}'].join('\n');
		const r = findAnchor(buf, search);
		// Elision rung should bail out; similarity will find best matches.
		// Either 'ambiguous' from similarity or 'unique' from another rung is acceptable,
		// but the 'via' must not be 'elision'.
		if (r.kind === 'unique') {
			expect(r.via).not.toBe('elision');
		}
	});
});

describe('findAnchor — similarity rung', () => {
	it('finds a near-match at ≥ 0.8 similarity', () => {
		const buf = ['function compute(x) {', '  const y = x + 1;', '  return y * 2;', '}'].join('\n');
		// Same shape, minor typo in variable name
		const search = ['function compute(x) {', '  const z = x + 1;', '  return y * 2;', '}'].join(
			'\n'
		);
		const r = findAnchor(buf, search);
		expect(r.kind).toBe('unique');
		if (r.kind === 'unique') {
			expect(r.via).toBe('similarity');
		}
	});

	it('returns none with suggestions when similarity is too low', () => {
		const buf = 'completely unrelated content that has nothing in common';
		const search = 'function parseIntoAST(tokens) {\n  return ast;\n}';
		const r = findAnchor(buf, search);
		expect(r.kind).toBe('none');
		if (r.kind === 'none') {
			expect(Array.isArray(r.suggestions)).toBe(true);
		}
	});
});

describe('findAnchor — edge cases', () => {
	it('returns none for empty search', () => {
		const r = findAnchor('anything', '');
		expect(r.kind).toBe('none');
	});

	it('returns unique for a long multi-line exact match', () => {
		const buf = ['pre', '<<<MARKER>>>', 'line1', 'line2', 'line3', '<<<END>>>', 'post'].join('\n');
		const search = 'line1\nline2\nline3';
		const r = findAnchor(buf, search);
		expect(r.kind).toBe('unique');
		if (r.kind === 'unique') {
			expect(r.via).toBe('exact');
		}
	});
});
