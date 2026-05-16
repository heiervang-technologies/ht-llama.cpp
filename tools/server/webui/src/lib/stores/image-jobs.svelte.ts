/**
 * Tracker for in-flight image / video generation jobs the webui has
 * submitted to the OpenAI-compat images proxy.
 *
 * The proxy itself does not (yet) expose a queue endpoint, so the
 * webui only ever sees its own outbound jobs. That's still useful:
 * the user gets immediate visibility into "what am I waiting on,
 * how long has it been, is something stuck" without having to keep
 * the ImagesScreen open.
 *
 * Job lifecycle:
 *   submit(...)  →  status 'running', startedAt = Date.now()
 *   complete(id) →  status 'completed', endedAt = Date.now()
 *   fail(id, err)→  status 'failed', endedAt = Date.now(), error preserved
 *   cancel(id)   →  status 'cancelled', endedAt = Date.now()
 *
 * Completed jobs are retained briefly (RECENT_RETENTION_MS) so the
 * panel can show "just finished" entries with a fade-out, then dropped.
 * Failed jobs stick around longer so the user has time to inspect the
 * error.
 */

import { SvelteMap } from 'svelte/reactivity';

export type ImageJobKind = 't2i' | 'i2i' | 'i2v' | 's2v' | 'flf' | 'edit';
export type ImageJobStatus = 'running' | 'completed' | 'failed' | 'cancelled';

export interface ImageJob {
	id: string;
	kind: ImageJobKind;
	model: string;
	prompt: string;
	source: 'playground' | 'chat-tool' | 'slash-command' | 'edit';
	startedAt: number;
	endedAt?: number;
	status: ImageJobStatus;
	error?: string;
	/** Optional abort handle so the panel can offer a cancel button. */
	abort?: () => void;
}

const RECENT_RETENTION_MS = 30_000;
const FAILED_RETENTION_MS = 5 * 60_000;

class ImageJobsStore {
	private _jobs = new SvelteMap<string, ImageJob>();
	private _nextId = 1;

	/** Reactive view of all currently tracked jobs, newest first. */
	get jobs(): ImageJob[] {
		return Array.from(this._jobs.values()).sort((a, b) => b.startedAt - a.startedAt);
	}

	/** Only the running ones — what the panel's badge counts. */
	get runningCount(): number {
		let n = 0;
		for (const j of this._jobs.values()) if (j.status === 'running') n++;
		return n;
	}

	submit(opts: {
		kind: ImageJobKind;
		model: string;
		prompt: string;
		source: ImageJob['source'];
		abort?: () => void;
	}): string {
		const id = `imgjob-${this._nextId++}-${Date.now().toString(36)}`;
		this._jobs.set(id, {
			id,
			kind: opts.kind,
			model: opts.model,
			prompt: opts.prompt,
			source: opts.source,
			startedAt: Date.now(),
			status: 'running',
			abort: opts.abort
		});
		return id;
	}

	complete(id: string): void {
		const j = this._jobs.get(id);
		if (!j) return;
		j.status = 'completed';
		j.endedAt = Date.now();
		j.abort = undefined;
		this._scheduleRemoval(id, RECENT_RETENTION_MS);
	}

	fail(id: string, error: string): void {
		const j = this._jobs.get(id);
		if (!j) return;
		j.status = 'failed';
		j.endedAt = Date.now();
		j.error = error;
		j.abort = undefined;
		this._scheduleRemoval(id, FAILED_RETENTION_MS);
	}

	cancel(id: string): void {
		const j = this._jobs.get(id);
		if (!j) return;
		try {
			j.abort?.();
		} catch (_) {
			// abort() can throw on already-aborted controllers — ignore.
		}
		j.status = 'cancelled';
		j.endedAt = Date.now();
		j.abort = undefined;
		this._scheduleRemoval(id, RECENT_RETENTION_MS);
	}

	/** Drop a finished job immediately (used by the panel's dismiss button). */
	dismiss(id: string): void {
		const j = this._jobs.get(id);
		if (!j) return;
		if (j.status === 'running') return; // never dismiss a running job
		this._jobs.delete(id);
	}

	private _scheduleRemoval(id: string, after: number): void {
		setTimeout(() => {
			const j = this._jobs.get(id);
			if (j && j.status !== 'running') {
				this._jobs.delete(id);
			}
		}, after);
	}
}

export const imageJobs = new ImageJobsStore();
