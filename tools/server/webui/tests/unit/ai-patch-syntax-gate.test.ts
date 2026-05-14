/**
 * Syntax gate — F7 detection behaviour across the kinds we care about in
 * v1. The runtime target is the browser; unit tests run in node without
 * DOMParser, so we install a minimal shim for the suite that exercises
 * the DOM-parsing branch. The SSR-skip branch is tested with the shim
 * absent.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { validateSyntax } from '$lib/editor/ai-patch/syntax-gate';

/* ------------------------------------------------------------------------- */
/* Minimal DOMParser shim — just enough to exercise the parsererror branch   */
/* ------------------------------------------------------------------------- */

/**
 * We hand-roll a single-file DOMParser fake rather than pulling in jsdom /
 * happy-dom (the branch's "no new dependencies" rule). The shim only needs
 * to return an object with a `.querySelector('parsererror')` that either
 * resolves to an element-ish value or null — which is the exact shape the
 * syntax gate inspects.
 *
 * Detection heuristics:
 *   - html: unbalanced `<tag>` without closing, or a literal `<parsererror>`
 *   - svg: not wrapped in `<svg>`, or missing closing svg
 *
 * This is intentionally crude — real validation runs in the browser. The
 * goal here is to assert the gate's plumbing (forward the error message,
 * respect the kind, etc.) not to test DOMParser itself.
 */
class FakeParserError {
	constructor(public textContent: string) {}
}

class FakeDocument {
	constructor(private readonly error: string | null) {}
	querySelector(sel: string): FakeParserError | null {
		if (sel === 'parsererror' && this.error) return new FakeParserError(this.error);
		return null;
	}
}

class FakeDOMParser {
	parseFromString(text: string, mime: string): FakeDocument {
		if (mime === 'text/html') {
			// Trip on `<unclosed` (no matching `>`) — our test inputs use this
			// as a sentinel for "definitely invalid".
			if (text.includes('<<<BAD_HTML>>>')) return new FakeDocument('html parse failure');
			return new FakeDocument(null);
		}
		if (mime === 'image/svg+xml') {
			if (!text.trim().startsWith('<svg')) return new FakeDocument('not an svg document');
			if (text.includes('<<<BAD_SVG>>>')) return new FakeDocument('svg parse failure');
			// Detect obvious malformed-XML sentinel.
			if (text.includes('<unclosed')) return new FakeDocument('unclosed tag');
			return new FakeDocument(null);
		}
		return new FakeDocument(null);
	}
}

/* ------------------------------------------------------------------------- */

describe('validateSyntax — SSR-safe fallbacks', () => {
	it('markdown and doc always pass, regardless of DOMParser availability', () => {
		expect(validateSyntax('markdown', 'definitely not html')).toEqual({ ok: true });
		expect(validateSyntax('doc', '# a heading\n\n- bullet')).toEqual({ ok: true });
	});

	it('code / image / audio / video / pdf pass through as skipped', () => {
		for (const kind of ['code', 'image', 'audio', 'video', 'pdf'] as const) {
			const res = validateSyntax(kind, 'anything');
			expect(res.ok).toBe(true);
		}
	});

	it('html / svg skip gracefully when DOMParser is not defined (node / SSR)', () => {
		// Sanity check the test environment — node has no DOMParser unless
		// we've installed the shim. Guard so the matrix is explicit.
		expect(typeof DOMParser).toBe('undefined');
		const html = validateSyntax('html', '<<<BAD_HTML>>>');
		expect(html.ok).toBe(true);
		expect(html).toMatchObject({ skipped: true });
		const svg = validateSyntax('svg', '<<<BAD_SVG>>>');
		expect(svg.ok).toBe(true);
		expect(svg).toMatchObject({ skipped: true });
	});
});

describe('validateSyntax — with a DOMParser shim installed', () => {
	beforeAll(() => {
		(globalThis as any).DOMParser = FakeDOMParser;
	});
	afterAll(() => {
		delete (globalThis as any).DOMParser;
	});

	it('valid html passes', () => {
		const res = validateSyntax('html', '<!doctype html><html><body>ok</body></html>');
		expect(res.ok).toBe(true);
	});

	it('invalid html is rejected with the parsererror message', () => {
		const res = validateSyntax('html', '<<<BAD_HTML>>>');
		expect(res.ok).toBe(false);
		if (!res.ok) expect(res.error).toContain('html parse failure');
	});

	it('valid svg passes', () => {
		const res = validateSyntax(
			'svg',
			'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'
		);
		expect(res.ok).toBe(true);
	});

	it('invalid svg (unclosed tag) is rejected', () => {
		const res = validateSyntax('svg', '<svg><unclosed');
		expect(res.ok).toBe(false);
		if (!res.ok) expect(res.error).toContain('unclosed');
	});

	it('markdown still always passes even with DOMParser available', () => {
		expect(validateSyntax('markdown', '# broken? maybe, but we do not reject')).toEqual({
			ok: true
		});
	});

	it('truncates very long parser errors to keep repair prompts bounded', () => {
		const originalParser = (globalThis as any).DOMParser;

		(globalThis as any).DOMParser = class {
			parseFromString() {
				return {
					querySelector(sel: string) {
						return sel === 'parsererror' ? { textContent: 'x'.repeat(1000) } : null;
					}
				};
			}
		};
		try {
			const res = validateSyntax('html', 'any');
			expect(res.ok).toBe(false);
			if (!res.ok) {
				expect(res.error.length).toBeLessThanOrEqual(241);
				expect(res.error.endsWith('…')).toBe(true);
			}
		} finally {
			(globalThis as any).DOMParser = originalParser;
		}
	});
});
