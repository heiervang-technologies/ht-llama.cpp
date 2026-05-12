/**
 * Terminals store — reactive view of live sandbox terminals managed
 * by the `ht-termd` sidecar. Thin wrapper around `TermdService`; the
 * UI binds directly to the `$state` arrays here and calls the
 * methods to mutate them.
 */

import {
	TermdService,
	TermdUnavailable,
	resolveTermdUrl,
	type SandboxStatus,
	type TerminalHandle
} from '$lib/services/termd.service';

class TerminalsStore {
	terminals = $state<TerminalHandle[]>([]);
	sandbox = $state<SandboxStatus | null>(null);
	loading = $state(false);
	creating = $state(false);
	/** Last error from a user-triggered action — UI surfaces it as a
	 *  toast or an inline banner. Cleared on success. */
	error = $state<string | null>(null);
	/** True when the sidecar URL is known (Tauri injection, llama-server
	 *  props, or user setting). When false, the UI hides / explains. */
	available = $derived(Boolean(resolveTermdUrl()));

	async refresh(): Promise<void> {
		if (!this.available) return;
		this.loading = true;
		try {
			// Intentionally parallel — the two endpoints are independent.
			const [list, status] = await Promise.all([
				TermdService.list(),
				TermdService.status().catch(() => null)
			]);
			this.terminals = list;
			this.sandbox = status;
			this.error = null;
		} catch (err) {
			this.error = err instanceof Error ? err.message : String(err);
		} finally {
			this.loading = false;
		}
	}

	async create(
		body: Parameters<typeof TermdService.create>[0] = {}
	): Promise<TerminalHandle | null> {
		if (!this.available) {
			this.error = new TermdUnavailable().message;
			return null;
		}
		this.creating = true;
		try {
			const t = await TermdService.create(body);
			this.terminals = [t, ...this.terminals.filter((x) => x.id !== t.id)];
			this.error = null;
			return t;
		} catch (err) {
			this.error = err instanceof Error ? err.message : String(err);
			return null;
		} finally {
			this.creating = false;
		}
	}

	async destroy(id: string): Promise<boolean> {
		try {
			await TermdService.destroy(id);
			this.terminals = this.terminals.filter((t) => t.id !== id);
			return true;
		} catch (err) {
			this.error = err instanceof Error ? err.message : String(err);
			return false;
		}
	}

	clearError(): void {
		this.error = null;
	}

	/**
	 * True when `assert_sandbox_ready` in the sidecar would refuse to
	 * spawn — UI uses this to show a "run `unleash sandbox setup`"
	 * banner instead of the create button.
	 */
	get needsSetup(): boolean {
		const s = this.sandbox;
		if (!s) return true;
		return !(s.docker_ok && s.runsc_ok && s.network_ok && s.image_ok && s.iptables_ok === 'ok');
	}
}

export const terminalsStore = new TerminalsStore();
