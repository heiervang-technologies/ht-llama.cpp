/**
 * In-memory proposal queue used by Review-mode sandbox terminals.
 *
 * The store is tiny but its contract matters: `send_keys` in Review
 * mode calls `propose()` and returns without blocking; the user later
 * calls `approve()` / `reject()` via the UI which in turn calls
 * `remove()`. Regressions here would either drop proposals silently
 * or leak them across terminals — neither is visible without
 * exercising the store directly.
 */

import { beforeEach, describe, expect, it } from 'vitest';
import {
	terminalProposals,
	type TerminalProposal
} from '../../src/lib/stores/terminal-proposals.svelte';

// The module exports a singleton; reset between tests by flushing
// every terminal we've touched. Using `clearTerminal` (the public
// API) instead of reaching into internals so the reset path stays
// exercised by the test suite too.
function reset(ids: string[]) {
	for (const id of ids) terminalProposals.clearTerminal(id);
}

describe('terminalProposals store', () => {
	beforeEach(() => reset(['t-a', 't-b', 't-c']));

	it('starts empty for any terminal id', () => {
		expect(terminalProposals.pending('t-a')).toEqual([]);
		expect(terminalProposals.totalPending()).toBe(0);
	});

	it('propose() returns a fresh proposal and makes it visible in pending()', () => {
		const p = terminalProposals.propose('t-a', { text: 'ls\n', autoEnter: false });
		expect(p.id).toMatch(/^prop-/);
		expect(p.terminalId).toBe('t-a');
		expect(p.text).toBe('ls\n');
		expect(p.autoEnter).toBe(false);
		expect(p.source).toBe('model');
		expect(p.createdAt).toBeLessThanOrEqual(Date.now());

		const pending = terminalProposals.pending('t-a');
		expect(pending).toHaveLength(1);
		expect(pending[0]).toEqual(p);
	});

	it('keeps proposals isolated per terminal id', () => {
		terminalProposals.propose('t-a', { text: 'pwd', autoEnter: true });
		terminalProposals.propose('t-b', { text: 'whoami', autoEnter: true });

		expect(terminalProposals.pending('t-a')).toHaveLength(1);
		expect(terminalProposals.pending('t-b')).toHaveLength(1);
		expect(terminalProposals.pending('t-c')).toEqual([]);
		expect(terminalProposals.totalPending()).toBe(2);
	});

	it('preserves insertion order within a terminal', () => {
		const first = terminalProposals.propose('t-a', { text: '1', autoEnter: false });
		const second = terminalProposals.propose('t-a', { text: '2', autoEnter: false });
		const third = terminalProposals.propose('t-a', { text: '3', autoEnter: false });

		expect(terminalProposals.pending('t-a').map((p) => p.id)).toEqual([
			first.id,
			second.id,
			third.id
		]);
	});

	it('assigns unique ids across many rapid proposals', () => {
		const ids = new Set<string>();
		for (let i = 0; i < 200; i++) {
			const p = terminalProposals.propose('t-a', { text: String(i), autoEnter: false });
			ids.add(p.id);
		}
		expect(ids.size).toBe(200);
	});

	it('remove() drops the matching proposal and returns it', () => {
		const p = terminalProposals.propose('t-a', { text: 'echo hi', autoEnter: true });
		terminalProposals.propose('t-a', { text: 'echo bye', autoEnter: true });

		const removed = terminalProposals.remove(p.id);
		expect(removed?.id).toBe(p.id);
		expect(terminalProposals.pending('t-a')).toHaveLength(1);
		expect(terminalProposals.pending('t-a')[0].text).toBe('echo bye');
	});

	it('remove() with an unknown id is a no-op and returns null', () => {
		terminalProposals.propose('t-a', { text: 'x', autoEnter: false });
		const removed = terminalProposals.remove('prop-does-not-exist');
		expect(removed).toBeNull();
		expect(terminalProposals.pending('t-a')).toHaveLength(1);
	});

	it('clearTerminal() wipes only that terminal', () => {
		terminalProposals.propose('t-a', { text: '1', autoEnter: false });
		terminalProposals.propose('t-a', { text: '2', autoEnter: false });
		terminalProposals.propose('t-b', { text: 'untouched', autoEnter: true });

		terminalProposals.clearTerminal('t-a');
		expect(terminalProposals.pending('t-a')).toEqual([]);
		expect(terminalProposals.pending('t-b')).toHaveLength(1);
	});

	it('totalPending() aggregates across terminals', () => {
		terminalProposals.propose('t-a', { text: 'a1', autoEnter: false });
		terminalProposals.propose('t-a', { text: 'a2', autoEnter: false });
		terminalProposals.propose('t-b', { text: 'b1', autoEnter: false });
		expect(terminalProposals.totalPending()).toBe(3);

		terminalProposals.clearTerminal('t-a');
		expect(terminalProposals.totalPending()).toBe(1);
	});

	it('preserves payload fidelity: auto_enter flag, control chars', () => {
		const ctrlC = '\x03';
		const p = terminalProposals.propose('t-a', {
			text: `top${ctrlC}`,
			autoEnter: false
		});
		const pending = terminalProposals.pending('t-a') as TerminalProposal[];
		expect(pending[0].text).toBe(`top${ctrlC}`);
		expect(p.autoEnter).toBe(false);
	});
});
