/**
 * Module-level state for the /images playground so a generation in
 * flight survives the user navigating away and back. Component-local
 * `$state` would be discarded on unmount, leaving the user with no
 * indication a fetch is still pending — and no way to see the result
 * if it lands while they're on /chat.
 *
 * Held intentionally small: at most one active run at a time (the
 * playground UI only lets you submit one), plus the most recent
 * finished run so the canvas can repopulate on re-entry.
 *
 * Note: the AbortController is here too — calling `cancel()` aborts
 * the fetch even if the originating component has unmounted. That's
 * the whole point.
 */

import type { RunImageEditResult, RunImageGenerationResult } from '$lib/services/builtin-tools';

type Mode = 'generate' | 'edit';

export interface PlaygroundActiveRun {
	mode: Mode;
	prompt: string;
	model: string;
	startedAt: number;
	abort: () => void;
}

export interface PlaygroundFinishedRun {
	mode: Mode;
	result: RunImageGenerationResult | RunImageEditResult;
	dataUrls: string[];
	finishedAt: number;
}

class ImagePlaygroundStore {
	active = $state<PlaygroundActiveRun | null>(null);
	lastFinished = $state<PlaygroundFinishedRun | null>(null);

	beginRun(run: PlaygroundActiveRun) {
		this.active = run;
	}

	finishRun(finished: PlaygroundFinishedRun) {
		this.active = null;
		this.lastFinished = finished;
	}

	failRun() {
		this.active = null;
	}

	cancel() {
		this.active?.abort();
	}
}

export const imagePlaygroundStore = new ImagePlaygroundStore();
