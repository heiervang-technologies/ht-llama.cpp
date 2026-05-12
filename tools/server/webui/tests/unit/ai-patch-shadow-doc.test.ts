import { describe, expect, it } from 'vitest';
import { ShadowDoc } from '$lib/editor/ai-patch';

describe('ShadowDoc — basic anchor + chunk + close', () => {
	it('replaces the anchored range with the concatenation of chunks', () => {
		const initial = 'alpha beta gamma';
		const shadow = new ShadowDoc(initial);
		// Replace "beta" with "delta epsilon".
		const from = initial.indexOf('beta');
		const to = from + 'beta'.length;
		shadow.applyAnchor(from, to);
		shadow.appendChunk('delta');
		shadow.appendChunk(' ');
		shadow.appendChunk('epsilon');
		const summary = shadow.closeBlock();
		expect(shadow.toString()).toBe('alpha delta epsilon gamma');
		expect(summary?.inserted).toBe('delta epsilon');
		expect(summary?.from).toBe(from);
		expect(summary?.to).toBe(from + 'delta epsilon'.length);
	});

	it('handles pure-insertion anchors (from == to)', () => {
		const shadow = new ShadowDoc('one\ntwo\n');
		shadow.applyAnchor(4, 4); // between "one\n" and "two"
		shadow.appendChunk('MIDDLE\n');
		shadow.closeBlock();
		expect(shadow.toString()).toBe('one\nMIDDLE\ntwo\n');
	});

	it('handles pure-deletion anchors (empty chunks)', () => {
		const shadow = new ShadowDoc('keep\nDELETE ME\nkeep2\n');
		const from = 'keep\n'.length;
		const to = from + 'DELETE ME\n'.length;
		shadow.applyAnchor(from, to);
		shadow.closeBlock();
		expect(shadow.toString()).toBe('keep\nkeep2\n');
	});
});

describe('ShadowDoc — multi-block mutation chain', () => {
	it('composes sequential blocks against the evolving buffer', () => {
		const shadow = new ShadowDoc('one\ntwo\nthree\n');
		// Block A: replace "one" → "ONE"
		shadow.applyAnchor(0, 3);
		shadow.appendChunk('ONE');
		shadow.closeBlock();
		expect(shadow.toString()).toBe('ONE\ntwo\nthree\n');
		// Block B: replace "three" → "THREE" — offsets must be computed against
		// the *current* buffer after block A.
		const text = shadow.toString();
		const from = text.indexOf('three');
		const to = from + 'three'.length;
		shadow.applyAnchor(from, to);
		shadow.appendChunk('THREE');
		shadow.closeBlock();
		expect(shadow.toString()).toBe('ONE\ntwo\nTHREE\n');
	});

	it('allows blocks that target newly-inserted text from a prior block', () => {
		const shadow = new ShadowDoc('<body></body>');
		// Block A: insert a div into the empty body.
		shadow.applyAnchor('<body>'.length, '<body>'.length);
		shadow.appendChunk('<div>hi</div>');
		shadow.closeBlock();
		expect(shadow.toString()).toBe('<body><div>hi</div></body>');
		// Block B: replace the just-inserted text.
		const text = shadow.toString();
		const from = text.indexOf('hi');
		const to = from + 'hi'.length;
		shadow.applyAnchor(from, to);
		shadow.appendChunk('hello');
		shadow.closeBlock();
		expect(shadow.toString()).toBe('<body><div>hello</div></body>');
	});
});

describe('ShadowDoc — invariants and error cases', () => {
	it('N chunks concatenated == shadow-state-delta for the block', () => {
		const shadow = new ShadowDoc('prefix DELTA suffix');
		const from = 'prefix '.length;
		const to = from + 'DELTA'.length;
		const before = shadow.toString();
		shadow.applyAnchor(from, to);
		const pieces = ['hel', 'lo ', 'world'];
		for (const p of pieces) shadow.appendChunk(p);
		const summary = shadow.closeBlock();
		const after = shadow.toString();
		// Classical diff-application invariant: before[0..from] + concat(chunks) +
		// before[to..] == after.
		expect(before.slice(0, from) + pieces.join('') + before.slice(to)).toBe(after);
		expect(summary?.inserted).toBe(pieces.join(''));
	});

	it('throws when applyAnchor is called while a block is already open', () => {
		const shadow = new ShadowDoc('abcdef');
		shadow.applyAnchor(0, 1);
		expect(() => shadow.applyAnchor(2, 3)).toThrow(/already active/);
	});

	it('throws when appendChunk is called with no active anchor', () => {
		const shadow = new ShadowDoc('abcdef');
		expect(() => shadow.appendChunk('x')).toThrow(/no active anchor/);
	});

	it('rejects invalid anchor ranges', () => {
		const shadow = new ShadowDoc('abcdef');
		expect(() => shadow.applyAnchor(-1, 1)).toThrow();
		expect(() => shadow.applyAnchor(5, 2)).toThrow();
		expect(() => shadow.applyAnchor(0, 999)).toThrow();
	});

	it('closeBlock with no open block returns null (no-op)', () => {
		const shadow = new ShadowDoc('abc');
		expect(shadow.closeBlock()).toBeNull();
		expect(shadow.toString()).toBe('abc');
	});

	it('tracks hasOpenBlock correctly', () => {
		const shadow = new ShadowDoc('abcdef');
		expect(shadow.hasOpenBlock()).toBe(false);
		shadow.applyAnchor(0, 3);
		expect(shadow.hasOpenBlock()).toBe(true);
		shadow.closeBlock();
		expect(shadow.hasOpenBlock()).toBe(false);
	});
});
