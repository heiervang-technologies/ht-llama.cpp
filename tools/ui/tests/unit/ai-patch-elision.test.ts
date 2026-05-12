import { describe, expect, it } from 'vitest';
import { detectElision } from '$lib/editor/ai-patch/elision';

describe('detectElision', () => {
	it('returns null for empty input', () => {
		expect(detectElision('')).toBeNull();
	});

	it('returns null for plain code without placeholders', () => {
		const code = ['function greet(name) {', '  return `Hello, ${name}!`;', '}'].join('\n');
		expect(detectElision(code)).toBeNull();
	});

	describe('bare ellipsis', () => {
		it('flags a `...` line alone', () => {
			const hit = detectElision(['line1', '...', 'line3'].join('\n'));
			expect(hit).not.toBeNull();
			expect(hit!.line).toBe(1);
			expect(hit!.reason).toBe('bare ellipsis line');
		});

		it('flags a `...` line with leading whitespace', () => {
			const hit = detectElision(['a', '    ...', 'b'].join('\n'));
			expect(hit).not.toBeNull();
			expect(hit!.line).toBe(1);
		});

		it('flags more than three dots too', () => {
			const hit = detectElision('.....');
			expect(hit).not.toBeNull();
		});
	});

	describe('placeholder comments', () => {
		it('flags `// ... rest unchanged`', () => {
			const hit = detectElision(['function foo() {', '  // ... rest unchanged', '}'].join('\n'));
			expect(hit).not.toBeNull();
			expect(hit!.line).toBe(1);
			expect(hit!.reason).toBe('placeholder comment');
		});

		it('flags `# ... existing code`', () => {
			const hit = detectElision(['def foo():', '    # ... existing code', '    pass'].join('\n'));
			expect(hit).not.toBeNull();
		});

		it('flags `// existing code` with no ellipsis', () => {
			const hit = detectElision(['function foo() {', '  // existing code', '}'].join('\n'));
			expect(hit).not.toBeNull();
		});

		it('flags `# unchanged`', () => {
			const hit = detectElision(['a = 1', '# unchanged', 'b = 2'].join('\n'));
			expect(hit).not.toBeNull();
		});

		it('flags `/* ... */` block comment with placeholder word', () => {
			const hit = detectElision(['a;', '/* ... rest unchanged */', 'b;'].join('\n'));
			expect(hit).not.toBeNull();
		});

		it('flags `<!-- ... existing -->` HTML comment', () => {
			const hit = detectElision(['<div>', '<!-- ... existing content -->', '</div>'].join('\n'));
			expect(hit).not.toBeNull();
		});

		it('flags `-- ... unchanged` SQL-style comment', () => {
			const hit = detectElision(['SELECT 1;', '-- ... unchanged', 'SELECT 2;'].join('\n'));
			expect(hit).not.toBeNull();
		});

		it('flags unicode single-char ellipsis', () => {
			const hit = detectElision(['a', '// … rest unchanged', 'b'].join('\n'));
			expect(hit).not.toBeNull();
		});
	});

	describe('negative cases', () => {
		it('does not flag prose with `...` inside a string literal', () => {
			const code = 'const msg = "Loading...";';
			expect(detectElision(code)).toBeNull();
		});

		it('does not flag a comment that mentions `existing` in normal prose', () => {
			const code = [
				'function foo() {',
				'  // Parse the existing configuration file before applying defaults.',
				'  return 1;',
				'}'
			].join('\n');
			expect(detectElision(code)).toBeNull();
		});

		it('does not flag a real ellipsis inside English prose', () => {
			const code = 'const msg = "Thinking...";';
			expect(detectElision(code)).toBeNull();
		});

		it('does not flag a real comment like "// the rest of the logic follows"', () => {
			const code = [
				'function foo() {',
				'  // the rest of the logic follows a different pattern',
				'  return 1;',
				'}'
			].join('\n');
			expect(detectElision(code)).toBeNull();
		});

		it('does not flag dots inside a normal expression line', () => {
			const code = 'const arr = [...other, 1, 2];';
			expect(detectElision(code)).toBeNull();
		});

		it('does not flag a comment that contains a URL with three dots', () => {
			const code = '// see https://example.com/docs/api for details';
			expect(detectElision(code)).toBeNull();
		});
	});

	it('returns the first hit when multiple placeholders exist', () => {
		const txt = ['a', '// ... rest unchanged', 'b', '...', 'c'].join('\n');
		const hit = detectElision(txt);
		expect(hit).not.toBeNull();
		expect(hit!.line).toBe(1);
	});
});
