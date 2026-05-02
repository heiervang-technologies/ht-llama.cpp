/**
 * Typed client for the `ht-termd` sidecar — the Rust HTTP+WS daemon
 * that owns Docker + PTY plumbing for sandbox terminals.
 *
 * The webui itself never constructs IPs or container ids; callers
 * discover the termd base URL via `resolveTermdUrl()`, then use
 * these thin wrappers to list / create / delete terminals and open
 * a WebSocket to attach an xterm.js pane to the container shell.
 */

import { config } from '$lib/stores/settings.svelte';

/** Raw sandbox readiness flags — mirrors the `SandboxStatus` struct
 *  in `tools/termd/src/sandbox_guard.rs`. */
export interface SandboxStatus {
	docker_ok: boolean;
	runsc_ok: boolean;
	network_ok: boolean;
	iptables_ok: 'ok' | 'missing' | 'unknown';
	image_ok: boolean;
}

export interface TerminalHandle {
	id: string;
	name: string;
	container_id: string;
	image: string;
	status: string;
	created_at: number;
}

export interface CreateTerminalBody {
	name?: string;
	/** Shell snippet that runs once, as root, inside the container
	 *  after files have been written. Output captured in the
	 *  per-terminal bootstrap log accessible via
	 *  `fetchBootstrapLog(id)`. */
	bootstrap?: string;
	/** Extra env vars for every docker-exec invocation. */
	env?: Record<string, string>;
	/** Files to drop before the bootstrap runs. Text passes through
	 *  verbatim; prefix `base64:` for binary. */
	files?: Array<{ path: string; content: string; mode?: number }>;
}

export class TermdUnavailable extends Error {
	constructor() {
		super('ht-termd not configured — set a terminals URL in Settings.');
	}
}

/**
 * URL resolution precedence (highest wins):
 *
 *   1. `window.__HT_TERMD_URL__` — injected by the Tauri shell when
 *      it auto-spawns the sidecar. Always wins in desktop mode.
 *   2. `config().terminalsBaseUrl` — optional user setting. Lets web
 *      deployments point at a remote termd.
 *   3. A future `/props.terminals.url` from llama-server (wired in a
 *      later commit).
 *   4. Undefined — the feature is simply off; UI hides the entry.
 */
export function resolveTermdUrl(): string | undefined {
	if (typeof window !== 'undefined') {
		const injected = (window as unknown as { __HT_TERMD_URL__?: string }).__HT_TERMD_URL__;
		if (typeof injected === 'string' && injected.trim()) return injected.trim();
	}
	const cfg = (config().terminalsBaseUrl as string | undefined)?.trim();
	if (cfg) return cfg;
	// Bundle-time default injected by the Tauri shell (from
	// `HT_DEFAULT_TERMINALS_URL` at build). Empty on desktop, set to
	// the tailnet-reachable termd on the Android APK.
	if (typeof window !== 'undefined') {
		const fallback = (window as unknown as { __HT_DEFAULT_TERMINALS_URL__?: string })
			.__HT_DEFAULT_TERMINALS_URL__;
		if (typeof fallback === 'string' && fallback.trim()) return fallback.trim();
	}
	return undefined;
}

/**
 * Bearer token to present to the termd daemon, if any. Precedence
 * mirrors `resolveTermdUrl`: Tauri injection > user config >
 * bundle-time default. Empty string means "no auth" — the daemon
 * will accept us as long as it was started without `--token`.
 */
export function resolveTermdToken(): string {
	if (typeof window !== 'undefined') {
		const injected = (window as unknown as { __HT_TERMD_TOKEN__?: string }).__HT_TERMD_TOKEN__;
		if (typeof injected === 'string' && injected.trim()) return injected.trim();
	}
	const cfg = (config().terminalsToken as string | undefined)?.trim();
	if (cfg) return cfg;
	if (typeof window !== 'undefined') {
		const fallback = (window as unknown as { __HT_DEFAULT_TERMINALS_TOKEN__?: string })
			.__HT_DEFAULT_TERMINALS_TOKEN__;
		if (typeof fallback === 'string' && fallback.trim()) return fallback.trim();
	}
	return '';
}

function baseOrThrow(): string {
	const base = resolveTermdUrl();
	if (!base) throw new TermdUnavailable();
	return base.replace(/\/+$/, '');
}

/** Per-request timeout. Termd serves loopback in single-digit ms; if the
 *  webview pool stalls, we'd rather surface a clean error than hang the
 *  agentic tool loop forever. The Tauri-plugin-http path is unaffected
 *  by webview throttling but still benefits from a deadline. */
const REQUEST_TIMEOUT_MS = 5000;

/** Lazy import of the Tauri-plugin-http `fetch`. Falls back to
 *  `window.fetch` in the browser / dev mode. We use the plugin in the
 *  desktop shell to side-step WebKit2GTK's resource-loader stall when
 *  the Tauri window sits on a non-active Hyprland workspace — reqwest
 *  on the Rust runtime is not subject to page throttling, so HTTP
 *  requests proceed even when JS is suspended-ish. */
async function getFetch(): Promise<typeof fetch> {
	if (typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window) {
		try {
			const mod = await import('@tauri-apps/plugin-http');
			return mod.fetch as unknown as typeof fetch;
		} catch {
			/* plugin missing (dev w/o tauri context) — fall through */
		}
	}
	return fetch;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const token = resolveTermdToken();
	const headers = new Headers(init?.headers ?? {});
	if (token && !headers.has('Authorization')) {
		headers.set('Authorization', `Bearer ${token}`);
	}
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
	const doFetch = await getFetch();
	let res: Response;
	try {
		res = await doFetch(`${baseOrThrow()}${path}`, {
			...init,
			headers,
			signal: init?.signal ?? controller.signal
		});
	} catch (err) {
		if ((err as { name?: string })?.name === 'AbortError') {
			throw new Error(`ht-termd ${path} timed out after ${REQUEST_TIMEOUT_MS}ms`);
		}
		throw err;
	} finally {
		clearTimeout(timer);
	}
	if (!res.ok) {
		let detail = '';
		try {
			detail = (await res.json()).error ?? '';
		} catch {
			/* non-JSON error bodies are possible on plain 5xx */
		}
		throw new Error(detail || `ht-termd ${path} → HTTP ${res.status}`);
	}
	if (res.status === 204) return undefined as unknown as T;
	return (await res.json()) as T;
}

export const TermdService = {
	async health(): Promise<{ status: string; sandbox: SandboxStatus }> {
		return request('/health');
	},

	async status(): Promise<SandboxStatus> {
		return request('/v1/sandbox/status');
	},

	async list(): Promise<TerminalHandle[]> {
		const r = await request<{ terminals: TerminalHandle[] }>('/v1/terminals');
		return r.terminals;
	},

	async create(body: CreateTerminalBody = {}): Promise<TerminalHandle> {
		return request('/v1/terminals', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
	},

	async destroy(id: string): Promise<void> {
		await request(`/v1/terminals/${encodeURIComponent(id)}`, { method: 'DELETE' });
	},

	/**
	 * Send keys into an existing terminal's shared PTY. Matches the
	 * server-side `/input` shape exactly; used both by the Review-mode
	 * "approve" button and by the model via the `send_keys` built-in
	 * tool.
	 */
	async sendInput(
		id: string,
		body: { text?: string; base64?: string; auto_enter?: boolean }
	): Promise<void> {
		await request(`/v1/terminals/${encodeURIComponent(id)}/input`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
	},

	/** Reads the bootstrap log (stdout + stderr of the setup script). */
	async bootstrapLog(id: string): Promise<string> {
		const r = await request<{ log: string }>(
			`/v1/terminals/${encodeURIComponent(id)}/bootstrap-log`
		);
		return r.log;
	},

	/**
	 * WebSocket URL for attaching an xterm.js pane. Caller is
	 * responsible for opening the WS and wiring it to the terminal.
	 */
	wsUrl(id: string): string {
		const base = baseOrThrow();
		const proto = base.startsWith('https://') ? 'wss' : 'ws';
		const hostPath = base.replace(/^https?:\/\//, '');
		// Browsers won't let us set `Authorization` on `new WebSocket()`
		// so auth for the WS handshake goes via query string. The server
		// accepts `?token=…` on the upgrade path only; everything else
		// still requires the bearer header.
		const token = resolveTermdToken();
		const query = token ? `?token=${encodeURIComponent(token)}` : '';
		return `${proto}://${hostPath}/v1/terminals/${encodeURIComponent(id)}/ws${query}`;
	}
};
