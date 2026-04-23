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
	return undefined;
}

function baseOrThrow(): string {
	const base = resolveTermdUrl();
	if (!base) throw new TermdUnavailable();
	return base.replace(/\/+$/, '');
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(`${baseOrThrow()}${path}`, init);
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
	 * WebSocket URL for attaching an xterm.js pane. Caller is
	 * responsible for opening the WS and wiring it to the terminal.
	 */
	wsUrl(id: string): string {
		const base = baseOrThrow();
		const proto = base.startsWith('https://') ? 'wss' : 'ws';
		const hostPath = base.replace(/^https?:\/\//, '');
		return `${proto}://${hostPath}/v1/terminals/${encodeURIComponent(id)}/ws`;
	}
};
