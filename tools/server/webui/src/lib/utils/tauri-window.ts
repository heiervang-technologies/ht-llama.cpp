/**
 * Lightweight feature-detect + open-in-new-window helpers for the
 * webui. The same component renders inside the Tauri shell and on the
 * web (llm.ht.local), so anywhere we want a "pop out" surface we
 * branch:
 *   - Tauri  → spawn a native WebviewWindow at the same hash route
 *              (`#/foo`), so navigation, theming, and the WS to
 *              ht-termd all work the same as the in-app drawer.
 *   - web    → `window.open(...)` so plain browser users still get
 *              a second-window experience without any Tauri dep.
 */

export function isTauri(): boolean {
	if (typeof window === 'undefined') return false;
	const w = window as unknown as { __TAURI_INTERNALS__?: unknown; __TAURI__?: unknown };
	return Boolean(w.__TAURI_INTERNALS__ ?? w.__TAURI__);
}

/**
 * Persisted zoom factor for the Tauri webview. Browsers handle their
 * own zoom (Ctrl+= / Ctrl+- maps to the chrome zoom UI), but
 * webkit2gtk doesn't bind those keys by default — so the desktop
 * shell needs to forward them to `setZoom()` on the active webview.
 */
const ZOOM_STORAGE_KEY = 'ht-llama:webview-zoom';
const ZOOM_MIN = 0.5;
const ZOOM_MAX = 3;
const ZOOM_DEFAULT = 1;
const ZOOM_STEP = 0.1;

function clampZoom(z: number): number {
	if (!Number.isFinite(z)) return ZOOM_DEFAULT;
	return Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z));
}

export function getStoredZoom(): number {
	if (typeof localStorage === 'undefined') return ZOOM_DEFAULT;
	const raw = localStorage.getItem(ZOOM_STORAGE_KEY);
	if (!raw) return ZOOM_DEFAULT;
	const n = Number(raw);
	return clampZoom(n);
}

async function applyZoom(level: number): Promise<void> {
	const clamped = clampZoom(level);
	if (typeof localStorage !== 'undefined') {
		localStorage.setItem(ZOOM_STORAGE_KEY, String(clamped));
	}
	if (isTauri()) {
		try {
			const { getCurrentWebview } = await import('@tauri-apps/api/webview');
			await getCurrentWebview().setZoom(clamped);
		} catch (err) {
			console.warn('[tauri] setZoom failed', err);
		}
	} else {
		// Browser path — best-effort. Some Chromium builds expose
		// `document.body.style.zoom`; everywhere else this is a no-op
		// and the browser's own Ctrl+= handler runs first anyway.
		if (typeof document !== 'undefined') {
			const body = document.body as HTMLBodyElement & { style: CSSStyleDeclaration };
			body.style.zoom = String(clamped);
		}
	}
}

export async function zoomIn(): Promise<void> {
	await applyZoom(getStoredZoom() + ZOOM_STEP);
}

export async function zoomOut(): Promise<void> {
	await applyZoom(getStoredZoom() - ZOOM_STEP);
}

export async function zoomReset(): Promise<void> {
	await applyZoom(ZOOM_DEFAULT);
}

/**
 * Reapply the persisted zoom on app boot so a reload doesn't snap
 * back to 1.0×. Safe to call before mount; bails on SSR.
 */
export async function restoreZoom(): Promise<void> {
	if (typeof window === 'undefined') return;
	await applyZoom(getStoredZoom());
}

/**
 * Open `route` (e.g. `#/terminals/<id>`) in a separate Tauri window if
 * available, otherwise a new browser tab. The Tauri label needs to be
 * unique per spawn — the optional `label` arg lets the caller pin it
 * to a stable id (so re-popping the same terminal focuses the existing
 * window instead of opening a duplicate).
 *
 * Title is best-effort: we set it on creation; OS chrome may further
 * decorate it. Width/height are sensible defaults; the user is free to
 * resize after.
 */
export async function openInNewWindow(
	route: string,
	opts: { title?: string; label?: string; width?: number; height?: number } = {}
): Promise<void> {
	const { title = 'heierchat', label, width = 960, height = 720 } = opts;
	const url = window.location.pathname + route;

	if (isTauri()) {
		try {
			const { WebviewWindow } = await import('@tauri-apps/api/webviewWindow');
			const finalLabel = label ?? `term-${Math.random().toString(36).slice(2, 10)}`;
			// Reuse path: if a window with this label already exists,
			// focus it instead of spawning a duplicate.
			const existing = await WebviewWindow.getByLabel(finalLabel);
			if (existing) {
				await existing.setFocus();
				return;
			}
			const w = new WebviewWindow(finalLabel, {
				url,
				title,
				width,
				height,
				resizable: true,
				focus: true
			});
			// Surface creation errors (rare — usually a missing capability
			// permission) instead of swallowing.
			await new Promise<void>((resolve, reject) => {
				w.once('tauri://created', () => resolve());
				w.once('tauri://error', (e) => reject(new Error(JSON.stringify(e.payload))));
			});
		} catch (err) {
			console.warn('[tauri] pop-out failed, falling back to window.open', err);
			window.open(url, '_blank', 'noopener');
		}
	} else {
		window.open(url, '_blank', `noopener,width=${width},height=${height}`);
	}
}
