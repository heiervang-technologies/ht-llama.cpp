/**
 * Polls the comfy-openai proxy's `/v1/images/queue` endpoint and exposes
 * the server-wide queue state alongside the webui's in-flight `imageJobs`.
 *
 * Spec landed in cloud commit (TBD) — comfy-openai:queue-v1. Response:
 *   {
 *     running:        [{ prompt_id, model, kind, submitted_at, elapsed_ms }],
 *     pending:        [{ prompt_id, model, kind, queued_at }],
 *     recently_done:  [{ prompt_id, model, kind, status, ended_at, duration_ms }]
 *   }
 *
 * `model` / `kind` / timing fields may be null for prompts submitted
 * before the proxy pod last restarted (PROMPT_TRACKER is in-memory).
 * Render those as "—" rather than dropping the row.
 *
 * Polling cadence: only ticks while at least one consumer is mounted
 * (the panel calls `subscribe()` on mount, `unsubscribe()` on unmount).
 * Interval is 1s when any local job is running OR the last response had
 * non-empty running/pending, otherwise 5s. Stops entirely when nobody
 * is subscribed.
 */

import { config } from '$lib/stores/settings.svelte';
import { imageJobs } from '$lib/stores/image-jobs.svelte';

export type ServerQueueKind = 'image' | 'video' | 'edit' | 'unknown' | string;
export type ServerQueueStatus = 'completed' | 'failed' | 'cancelled' | 'timeout' | 'unknown' | string;

export interface ServerQueueRunning {
	prompt_id: string;
	model: string | null;
	kind: ServerQueueKind | null;
	submitted_at: number | null;
	elapsed_ms: number | null;
}

export interface ServerQueuePending {
	prompt_id: string;
	model: string | null;
	kind: ServerQueueKind | null;
	queued_at: number | null;
}

export interface ServerQueueRecent {
	prompt_id: string;
	model: string | null;
	kind: ServerQueueKind | null;
	status: ServerQueueStatus;
	ended_at: number | null;
	duration_ms: number | null;
}

export interface ServerQueueSnapshot {
	running: ServerQueueRunning[];
	pending: ServerQueuePending[];
	recently_done: ServerQueueRecent[];
}

const EMPTY_SNAPSHOT: ServerQueueSnapshot = {
	running: [],
	pending: [],
	recently_done: []
};

const FAST_POLL_MS = 1000;
const SLOW_POLL_MS = 5000;

class ServerImageQueueStore {
	snapshot = $state<ServerQueueSnapshot>(EMPTY_SNAPSHOT);
	lastError = $state<string | null>(null);
	lastFetchedAt = $state<number | null>(null);

	private subscribers = 0;
	private timer: ReturnType<typeof setTimeout> | null = null;
	private inFlight = false;

	subscribe(): void {
		this.subscribers++;
		if (this.subscribers === 1) {
			void this.pollOnce();
			this.scheduleNext();
		}
	}

	unsubscribe(): void {
		this.subscribers = Math.max(0, this.subscribers - 1);
		if (this.subscribers === 0 && this.timer != null) {
			clearTimeout(this.timer);
			this.timer = null;
		}
	}

	/** Manual refresh — used by the panel's "refresh" button. */
	refresh(): void {
		void this.pollOnce();
	}

	private resolveBase(): string | null {
		const base = String(config().imagesBaseUrl ?? '').trim();
		if (!base) return null;
		return base.replace(/\/+$/, '');
	}

	private resolveHeaders(): Record<string, string> {
		const key = String(config().imagesApiKey ?? '').trim();
		const headers: Record<string, string> = { Accept: 'application/json' };
		if (key) headers.Authorization = `Bearer ${key}`;
		return headers;
	}

	private async pollOnce(): Promise<void> {
		if (this.inFlight) return;
		const base = this.resolveBase();
		if (!base) {
			this.snapshot = EMPTY_SNAPSHOT;
			this.lastError = null;
			return;
		}
		this.inFlight = true;
		try {
			const res = await fetch(`${base}/v1/images/queue`, {
				method: 'GET',
				headers: this.resolveHeaders()
			});
			if (!res.ok) {
				// 404 is the expected response from proxies that haven't shipped
				// the queue endpoint yet. Surface it as a soft state, not an
				// error — the panel just shows "Server view unavailable".
				this.lastError =
					res.status === 404 ? 'unavailable' : `HTTP ${res.status}`;
				this.snapshot = EMPTY_SNAPSHOT;
				return;
			}
			const payload = (await res.json()) as Partial<ServerQueueSnapshot>;
			this.snapshot = {
				running: Array.isArray(payload.running) ? payload.running : [],
				pending: Array.isArray(payload.pending) ? payload.pending : [],
				recently_done: Array.isArray(payload.recently_done) ? payload.recently_done : []
			};
			this.lastError = null;
			this.lastFetchedAt = Date.now();
		} catch (err) {
			this.lastError = err instanceof Error ? err.message : String(err);
		} finally {
			this.inFlight = false;
		}
	}

	private scheduleNext(): void {
		if (this.subscribers === 0) return;
		const fast =
			imageJobs.runningCount > 0 ||
			this.snapshot.running.length > 0 ||
			this.snapshot.pending.length > 0;
		const delay = fast ? FAST_POLL_MS : SLOW_POLL_MS;
		this.timer = setTimeout(async () => {
			await this.pollOnce();
			this.scheduleNext();
		}, delay);
	}
}

export const serverImageQueue = new ServerImageQueueStore();
