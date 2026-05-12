/**
 * Rolling median tool-call duration tracker.
 *
 * Every completed built-in / MCP tool call records its duration here
 * (from the agentic loop). The UI's pending-call progress bar reads
 * back the median to size its "expected duration" estimate. Median
 * (not mean) so a single outlier — proxy stall, cold cache — doesn't
 * skew the bar's pace; the next handful of normal calls drag the
 * estimate right back to truth.
 *
 * Last 20 samples per tool, persisted to localStorage so the estimate
 * survives reloads. Default seed values cover the slow built-ins so
 * the user doesn't stare at a 0% bar on the first invocation.
 */

const STORAGE_KEY = 'ht-llama:tool-timings';
const MAX_SAMPLES = 20;

// First-call hints. Tuned to the actual workflow latencies on this
// fork's cluster setup (z-image-turbo / wan22-i2v / etc). After the
// first sample lands these get superseded by the rolling median.
const DEFAULTS_MS: Record<string, number> = {
	generate_image: 55_000,
	edit_image: 150_000,
	generate_video: 180_000,
	web_search: 1_500,
	search_images: 1_800,
	search_news: 1_500,
	fetch_url: 4_000,
	fetch_image: 2_500,
	compose_collage: 1_000,
	send_keys: 250,
	list_terminals: 200,
	list_artifacts: 150,
	get_artifact: 150,
	fork_artifact: 250
};

function loadInitial(): Record<string, number[]> {
	if (typeof localStorage === 'undefined') return {};
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (!raw) return {};
		const parsed = JSON.parse(raw);
		if (parsed && typeof parsed === 'object') {
			const out: Record<string, number[]> = {};
			for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
				if (Array.isArray(v)) {
					const nums = v.filter((x): x is number => typeof x === 'number' && Number.isFinite(x));
					if (nums.length) out[k] = nums.slice(-MAX_SAMPLES);
				}
			}
			return out;
		}
	} catch {
		/* corrupt blob; start fresh */
	}
	return {};
}

class ToolTimingsStore {
	private samples = $state<Record<string, number[]>>(loadInitial());

	/** Record a successful tool-call duration. Non-finite / negative values ignored. */
	record(toolName: string | undefined, ms: number, success: boolean = true): void {
		if (!toolName || !success) return;
		if (!Number.isFinite(ms) || ms < 0) return;
		const arr = this.samples[toolName] ? [...this.samples[toolName]] : [];
		arr.push(ms);
		while (arr.length > MAX_SAMPLES) arr.shift();
		this.samples[toolName] = arr;
		this.persist();
	}

	/**
	 * Expected duration for a tool in ms, or null if we have no data
	 * and no default. The result is reactive — when a new sample lands
	 * via `record`, components reading via `median()` re-derive.
	 */
	median(toolName: string | undefined): number | null {
		if (!toolName) return null;
		const arr = this.samples[toolName];
		if (!arr || arr.length === 0) {
			return DEFAULTS_MS[toolName] ?? null;
		}
		const sorted = [...arr].sort((a, b) => a - b);
		const mid = Math.floor(sorted.length / 2);
		// Even-count median = average of the two middle values.
		return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
	}

	/** Sample count for a tool. */
	sampleCount(toolName: string | undefined): number {
		if (!toolName) return 0;
		return this.samples[toolName]?.length ?? 0;
	}

	private persist(): void {
		if (typeof localStorage === 'undefined') return;
		try {
			localStorage.setItem(STORAGE_KEY, JSON.stringify(this.samples));
		} catch {
			/* quota — drop silently rather than throwing into the agentic
			   loop; the bar just won't survive reloads. */
		}
	}
}

export const toolTimings = new ToolTimingsStore();
