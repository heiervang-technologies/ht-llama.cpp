/**
 * Queue of pending `send_keys` proposals per terminal. Used by the
 * Review interaction mode: instead of typing the model's keystrokes
 * straight into the PTY (which Shared mode does), we park them here
 * and let the user approve or reject each one from the terminal
 * view's side panel.
 *
 * The store is in-memory only — proposals are ephemeral. If the user
 * reloads, the queue clears; the model's agentic loop will observe
 * that its earlier "queued" response never resulted in visible output
 * and can recover. Persisting these to localStorage would create its
 * own class of stale-queue bugs and wasn't worth the UX of "oh you
 * had a pending approval from 3 days ago".
 */

import { SvelteMap } from 'svelte/reactivity';

export interface TerminalProposal {
	id: string;
	terminalId: string;
	text: string;
	autoEnter: boolean;
	createdAt: number;
	/** Origin hint so the UI can show who asked. Currently always
	 *  `'model'`; left open for future MCP / teammate sources. */
	source: 'model';
}

class TerminalProposalsStore {
	// Map<terminalId, proposals[]>. A SvelteMap so lookups stay
	// reactive — the panel re-renders when we push/remove.
	private byTerminal = new SvelteMap<string, TerminalProposal[]>();

	pending(terminalId: string): TerminalProposal[] {
		return this.byTerminal.get(terminalId) ?? [];
	}

	totalPending(): number {
		let n = 0;
		for (const list of this.byTerminal.values()) n += list.length;
		return n;
	}

	propose(terminalId: string, payload: { text: string; autoEnter: boolean }): TerminalProposal {
		const id = `prop-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
		const proposal: TerminalProposal = {
			id,
			terminalId,
			text: payload.text,
			autoEnter: payload.autoEnter,
			createdAt: Date.now(),
			source: 'model'
		};
		const list = [...(this.byTerminal.get(terminalId) ?? []), proposal];
		this.byTerminal.set(terminalId, list);
		return proposal;
	}

	remove(proposalId: string): TerminalProposal | null {
		for (const [tid, list] of this.byTerminal.entries()) {
			const idx = list.findIndex((p) => p.id === proposalId);
			if (idx === -1) continue;
			const [removed] = list.splice(idx, 1);
			// Trigger a map-level notification by re-setting the value —
			// splice mutates in place which the SvelteMap doesn't
			// observe by itself.
			this.byTerminal.set(tid, [...list]);
			return removed;
		}
		return null;
	}

	clearTerminal(terminalId: string): void {
		this.byTerminal.delete(terminalId);
	}
}

export const terminalProposals = new TerminalProposalsStore();
