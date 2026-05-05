/**
 * Tauri-aware fetch shim — returns `@tauri-apps/plugin-http`'s fetch
 * when running inside the Tauri shell, otherwise the native one.
 *
 * Why: webkit2gtk's resource loader stalls when the host workspace
 * is backgrounded on Hyprland, and CORS preflights silently fail
 * for cross-origin image / media URLs. Routing through reqwest on
 * the Rust runtime sidesteps both — page throttling can't reach the
 * Tauri runtime, and reqwest doesn't enforce CORS.
 *
 * Already used by `termd.service`; this module exists so other
 * call sites (image / video input fetching, future external probes)
 * can share the same plumbing without duplicating the dynamic-import
 * dance.
 */

import { isTauri } from './tauri-window';

let cached: typeof fetch | undefined;

export async function getFetch(): Promise<typeof fetch> {
	if (cached) return cached;
	if (isTauri()) {
		try {
			const mod = await import('@tauri-apps/plugin-http');
			cached = mod.fetch as unknown as typeof fetch;
			return cached;
		} catch {
			/* plugin missing (rare — webview context but no plugin
			   wired up; happens during `cargo tauri dev` if the JS
			   runs before the plugin handshake completes). Fall
			   through to the native fetch. */
		}
	}
	cached = fetch;
	return cached;
}
