/**
 * Per-terminal interaction mode. Controls who can put bytes into
 * the shared PTY:
 *
 *   - `solo`    — only the human; the `send_keys` built-in tool is
 *                 refused with a clear "blocked by user" message
 *                 the model can reason about.
 *   - `shared`  — anyone can type. Model + user land in the same
 *                 bash, keystrokes interleave live.
 *   - `review`  — model's input queues for explicit human approval
 *                 before hitting the PTY (wired in a follow-up;
 *                 today this behaves like `solo`).
 *
 * Mode lives in localStorage keyed by terminal id so it survives
 * reloads without adding a server round-trip. Default = solo, so a
 * freshly-spawned terminal is model-opaque until the user opts in.
 */

import { SvelteMap } from 'svelte/reactivity';

export type TerminalMode = 'solo' | 'shared' | 'review';

export const TERMINAL_MODES: {
	id: TerminalMode;
	label: string;
	description: string;
}[] = [
	{
		id: 'solo',
		label: 'Solo',
		description:
			'Only you can type. The model sees `send_keys` blocked and has to ask you to run commands.'
	},
	{
		id: 'shared',
		label: 'Shared',
		description: 'Model and user type into the same PTY. Chaos-grade pair programming.'
	},
	{
		id: 'review',
		label: 'Review',
		description:
			'Model proposes keystrokes; you approve each one before it lands. (Today: same as Solo.)'
	}
];

export const DEFAULT_TERMINAL_MODE: TerminalMode = 'solo';

const STORAGE_KEY = 'ht-llama.terminalModes';

class TerminalModesStore {
	private modes = new SvelteMap<string, TerminalMode>();
	private loaded = false;

	private load(): void {
		if (this.loaded || typeof localStorage === 'undefined') {
			this.loaded = true;
			return;
		}
		try {
			const raw = localStorage.getItem(STORAGE_KEY);
			if (raw) {
				const parsed = JSON.parse(raw) as Record<string, string>;
				for (const [id, mode] of Object.entries(parsed)) {
					if (mode === 'solo' || mode === 'shared' || mode === 'review') {
						this.modes.set(id, mode);
					}
				}
			}
		} catch {
			/* localStorage corrupt — ignore, start fresh */
		}
		this.loaded = true;
	}

	private persist(): void {
		if (typeof localStorage === 'undefined') return;
		try {
			localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(this.modes)));
		} catch {
			/* private mode / quota — ignore */
		}
	}

	get(id: string): TerminalMode {
		this.load();
		return this.modes.get(id) ?? DEFAULT_TERMINAL_MODE;
	}

	set(id: string, mode: TerminalMode): void {
		this.load();
		this.modes.set(id, mode);
		this.persist();
	}

	/** Peek without triggering a load — used from non-Svelte code
	 *  paths (the built-in `send_keys` tool) where we don't want to
	 *  pay the lazy-init on every call. */
	snapshot(id: string): TerminalMode {
		this.load();
		return this.modes.get(id) ?? DEFAULT_TERMINAL_MODE;
	}
}

export const terminalModes = new TerminalModesStore();
