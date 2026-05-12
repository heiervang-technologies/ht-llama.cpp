import { base } from '$app/paths';
import { config } from '$lib/stores/settings.svelte';
import { UrlProtocol } from '$lib/enums';

/**
 * Returns the configured backend base URL with no trailing slash,
 * or an empty string when same-origin should be used.
 */
export function getBackendBaseUrl(): string {
	const raw = config().backendBaseUrl?.toString().trim() ?? '';
	if (raw) return raw.replace(/\/+$/, '');
	// Bundle-time default injected by the Tauri shell via
	// `HT_DEFAULT_BACKEND_URL`. Lets the Android APK ship with the
	// user's tailnet llama.cpp endpoint preconfigured so a fresh
	// install works out of the box without opening Settings first.
	if (typeof window !== 'undefined') {
		const fallback = (window as unknown as { __HT_DEFAULT_BACKEND_URL__?: string })
			.__HT_DEFAULT_BACKEND_URL__;
		if (typeof fallback === 'string' && fallback.trim()) {
			return fallback.trim().replace(/\/+$/, '');
		}
	}
	return '';
}

/**
 * Returns the hostname (or host:port for non-default ports) of the configured
 * backend, or an empty string when no backend is configured.
 * Safe to call in browser and SSR contexts.
 */
export function getBackendHostLabel(): string {
	const base = getBackendBaseUrl();
	if (!base) return '';
	try {
		const url = new URL(base);
		const defaultPort =
			(url.protocol === 'http:' && (url.port === '' || url.port === '80')) ||
			(url.protocol === 'https:' && (url.port === '' || url.port === '443'));
		return defaultPort ? url.hostname : `${url.hostname}:${url.port}`;
	} catch {
		return base.replace(/^https?:\/\//, '');
	}
}

/**
 * Resolve an API path to an absolute or SvelteKit-prefixed URL.
 * - Absolute URLs (http/https) are returned as-is.
 * - When a backend base URL is configured, it is prepended (path normalized to start with `/`).
 * - Otherwise the SvelteKit `base` path is used for same-origin requests. Relative
 *   prefixes (`./`) are preserved so the existing behavior is unchanged.
 */
export function resolveApiUrl(path: string): string {
	if (path.startsWith(UrlProtocol.HTTP) || path.startsWith(UrlProtocol.HTTPS)) {
		return path;
	}
	const backend = getBackendBaseUrl();
	if (backend) {
		const stripped = path.replace(/^\.\//, '/');
		const normalized = stripped.startsWith('/') ? stripped : `/${stripped}`;
		return `${backend}${normalized}`;
	}
	if (path.startsWith('./')) {
		return path;
	}
	return `${base}${path}`;
}
