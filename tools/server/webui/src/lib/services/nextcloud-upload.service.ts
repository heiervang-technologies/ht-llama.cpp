/**
 * Nextcloud auto-upload pipeline.
 *
 * One artifact ←→ one remote file. The path follows the convention the
 * cloud-ops handoff specified, derived once at upload time and stored
 * on the artifact so subsequent re-uploads land at the same place
 * (idempotent overwrite via WebDAV PUT):
 *
 *   /<remoteRoot>/<YYYY-MM-DD>/<sessionId>/<artifactId>-<slug>.<ext>
 *
 * `<remoteRoot>` is what the user typed in Settings → Connections →
 * Nextcloud (default `/AI/`). The WebDAV client takes care of
 * prepending it to the user-facing path, so the producer side only
 * deals in everything-after-the-root.
 *
 *   <YYYY-MM-DD>  date partition so the folder doesn't grow unbounded
 *   <sessionId>   `metadata.source === 'chat'` → conversationId
 *                 `metadata.source === 'upload'` / direct → 'manual'
 *                 `metadata.source === 'generate_image' / 'edit_image' /
 *                                       'generate_video' / 'playground'`
 *                                                       → 'generated'
 *                 fallback                              → 'misc'
 *   <slug>        artifact.title sanitized to ASCII safe chars
 *   <ext>         derived from the revision's mimeType
 *
 * A `nextcloudSync` block is stamped into `artifact.metadata` on
 * success / failure so the gallery card can render a sync badge and
 * the mirror-deletes path can find the remote-side file when the
 * user later removes the artifact.
 *
 * In-flight `syncing` state lives in a Svelte store so the badge can
 * show a spinner without re-querying IndexedDB. Reload while a
 * sync is in flight = treat as not-synced (`undefined`); the next
 * mutation will re-fire.
 */

import { SvelteMap } from 'svelte/reactivity';
import { DatabaseService } from './database.service';
import { WebDavClient, WebDavError, WebDavNetworkError } from './webdav.service';
import { config } from '$lib/stores/settings.svelte';
import { SETTINGS_KEYS } from '$lib/constants';
import type { DatabaseArtifact } from '$lib/types/database';

const PASSWORD_KEY = 'nextcloud-app-password';

export type NextcloudSyncStatus = 'synced' | 'failed';

export interface NextcloudSyncState {
	status: NextcloudSyncStatus;
	/** Friendly path under `remoteRoot`, e.g. `/2026-04-27/manual/<id>-foo.png`. */
	remotePath: string;
	/** DAV href as the server reported it (URL-decoded). */
	remoteHref?: string;
	/** Absolute URL — handy for the gallery's "Open in Nextcloud" action. */
	remoteUrl?: string;
	/** Last server-reported ETag; populated when the response exposed it. */
	etag?: string;
	lastSyncedAt: number;
	/** Only when status === 'failed'. */
	error?: string;
}

/** In-flight "uploading" state per artifactId — for spinner UI. */
class NextcloudSyncRuntimeStore {
	private inFlight = new SvelteMap<string, true>();

	isUploading(artifactId: string): boolean {
		return this.inFlight.has(artifactId);
	}

	mark(artifactId: string): void {
		this.inFlight.set(artifactId, true);
	}

	clear(artifactId: string): void {
		this.inFlight.delete(artifactId);
	}
}

export const nextcloudSyncRuntime = new NextcloudSyncRuntimeStore();

/** Read the `nextcloudSync` block off an artifact, if any. */
export function getNextcloudSync(artifact: DatabaseArtifact): NextcloudSyncState | undefined {
	const raw = artifact.metadata?.nextcloudSync;
	if (!raw || typeof raw !== 'object') return undefined;
	const r = raw as Record<string, unknown>;
	if (typeof r.status !== 'string' || typeof r.remotePath !== 'string') return undefined;
	if (r.status !== 'synced' && r.status !== 'failed') return undefined;
	return {
		status: r.status,
		remotePath: r.remotePath,
		remoteHref: typeof r.remoteHref === 'string' ? r.remoteHref : undefined,
		remoteUrl: typeof r.remoteUrl === 'string' ? r.remoteUrl : undefined,
		etag: typeof r.etag === 'string' ? r.etag : undefined,
		lastSyncedAt: typeof r.lastSyncedAt === 'number' ? r.lastSyncedAt : Date.now(),
		error: typeof r.error === 'string' ? r.error : undefined
	};
}

interface ConnectionConfig {
	url: string;
	username: string;
	remoteRoot: string;
	autoUpload: boolean;
	mirrorDeletes: boolean;
}

export function readConnectionConfig(): ConnectionConfig {
	const c = config();
	return {
		url: String(c[SETTINGS_KEYS.NEXTCLOUD_URL] ?? '').trim(),
		username: String(c[SETTINGS_KEYS.NEXTCLOUD_USERNAME] ?? '').trim(),
		remoteRoot: String(c[SETTINGS_KEYS.NEXTCLOUD_REMOTE_ROOT] ?? '/AI/').trim(),
		autoUpload: Boolean(c[SETTINGS_KEYS.NEXTCLOUD_AUTO_UPLOAD]),
		mirrorDeletes: Boolean(c[SETTINGS_KEYS.NEXTCLOUD_MIRROR_DELETES])
	};
}

export function isNextcloudConfigured(): boolean {
	const cfg = readConnectionConfig();
	return Boolean(cfg.url && cfg.username);
}

async function buildClient(): Promise<WebDavClient | null> {
	const cfg = readConnectionConfig();
	if (!cfg.url || !cfg.username) return null;
	const password = await DatabaseService.getSecret(PASSWORD_KEY);
	if (!password) return null;
	return new WebDavClient({
		baseUrl: cfg.url,
		username: cfg.username,
		password,
		remoteRoot: cfg.remoteRoot || '/'
	});
}

/**
 * Compute the post-root path for a given artifact + revision. Stable:
 * the same artifact + revision pair always yields the same path so
 * re-upload overwrites cleanly.
 */
export function buildArtifactPath(
	artifact: DatabaseArtifact,
	mimeType: string | undefined
): string {
	const date = formatDate(artifact.createdAt ?? Date.now());
	const session = sessionFor(artifact);
	const slug = slugify(artifact.title || artifact.id);
	const ext = extFor(artifact, mimeType);
	const filename = `${artifact.id}-${slug}${ext ? `.${ext}` : ''}`;
	return `/${date}/${session}/${filename}`;
}

function formatDate(ts: number): string {
	const d = new Date(ts);
	const yyyy = d.getFullYear();
	const mm = String(d.getMonth() + 1).padStart(2, '0');
	const dd = String(d.getDate()).padStart(2, '0');
	return `${yyyy}-${mm}-${dd}`;
}

function sessionFor(artifact: DatabaseArtifact): string {
	if (artifact.sourceConversationId) return artifact.sourceConversationId;
	const source = (artifact.metadata as Record<string, unknown> | undefined)?.source;
	if (typeof source === 'string') {
		if (source === 'upload' || source === 'direct' || source === 'manual') return 'manual';
		if (source === 'playground') return 'generated';
		if (source === 'generate_image' || source === 'edit_image' || source === 'generate_video') {
			return 'generated';
		}
	}
	return 'misc';
}

function slugify(s: string): string {
	const base = s
		.normalize('NFKD')
		.replace(/[̀-ͯ]/g, '')
		.replace(/[^a-zA-Z0-9._-]+/g, '-')
		.replace(/^-+|-+$/g, '')
		.slice(0, 60);
	return base || 'artifact';
}

function extFor(artifact: DatabaseArtifact, mimeType: string | undefined): string {
	const fromTitle = artifact.title.match(/\.([a-zA-Z0-9]{1,8})$/)?.[1]?.toLowerCase();
	if (fromTitle) return fromTitle;
	const m = (mimeType ?? '').toLowerCase();
	if (m === 'image/png') return 'png';
	if (m === 'image/jpeg') return 'jpg';
	if (m === 'image/webp') return 'webp';
	if (m === 'image/gif') return 'gif';
	if (m === 'image/svg+xml') return 'svg';
	if (m === 'audio/mpeg') return 'mp3';
	if (m === 'audio/wav' || m === 'audio/x-wav') return 'wav';
	if (m === 'audio/ogg') return 'ogg';
	if (m === 'video/mp4') return 'mp4';
	if (m === 'video/webm') return 'webm';
	if (m === 'application/pdf') return 'pdf';
	if (m === 'text/html') return 'html';
	if (m === 'text/markdown' || m === 'text/x-markdown') return 'md';
	if (m === 'application/json') return 'json';
	if (m.startsWith('text/')) return 'txt';
	switch (artifact.kind) {
		case 'html':
			return 'html';
		case 'svg':
			return 'svg';
		case 'markdown':
			return 'md';
		case 'pdf':
			return 'pdf';
		default:
			return '';
	}
}

/**
 * Upload an artifact's current revision to Nextcloud. Resolves with
 * the resulting sync state regardless of success — the failure case
 * lands on `artifact.metadata.nextcloudSync.status === 'failed'` so
 * the gallery card can show a retry chip.
 *
 * No-ops (returns `undefined`) when:
 *   - The connection is not configured
 *   - The artifact came from Nextcloud (`metadata.source === 'nextcloud'`)
 *   - The artifact has no fetchable bytes
 */
export async function uploadArtifact(
	artifact: DatabaseArtifact
): Promise<NextcloudSyncState | undefined> {
	if (!isNextcloudConfigured()) return undefined;
	const source = (artifact.metadata as Record<string, unknown> | undefined)?.source;
	if (source === 'nextcloud') return undefined;

	const revision = await DatabaseService.getArtifactRevision(artifact.currentRevisionId);
	if (!revision) return undefined;

	const body = revision.blob ?? revision.text;
	if (!body) return undefined;

	nextcloudSyncRuntime.mark(artifact.id);
	const cfg = readConnectionConfig();
	const remotePath = buildArtifactPath(artifact, revision.mimeType);
	const folder = remotePath.slice(0, remotePath.lastIndexOf('/'));

	let result: NextcloudSyncState;
	try {
		const client = await buildClient();
		if (!client) {
			result = {
				status: 'failed',
				remotePath,
				lastSyncedAt: Date.now(),
				error: 'No app password saved. Open Settings → Connections → Nextcloud.'
			};
		} else {
			await client.ensureFolder(folder);
			const { etag } = await client.put(remotePath, body, { contentType: revision.mimeType });
			const remoteHref = `/remote.php/dav/files/${encodeURIComponent(cfg.username)}${encodePath(
				normaliseRoot(cfg.remoteRoot) + remotePath
			)}`;
			const remoteUrl = `${cfg.url.replace(/\/+$/, '')}${remoteHref}`;
			result = {
				status: 'synced',
				remotePath,
				remoteHref,
				remoteUrl,
				etag: etag ?? undefined,
				lastSyncedAt: Date.now()
			};
		}
	} catch (err) {
		result = {
			status: 'failed',
			remotePath,
			lastSyncedAt: Date.now(),
			error: translateError(err)
		};
	} finally {
		nextcloudSyncRuntime.clear(artifact.id);
	}

	const nextMetadata = { ...(artifact.metadata ?? {}), nextcloudSync: result };
	await DatabaseService.updateArtifact(artifact.id, { metadata: nextMetadata });
	return result;
}

/**
 * Fired by the gallery store after a new artifact lands. Background
 * (no await on the caller side) so the user-visible operation stays
 * snappy. Honours the `nextcloudAutoUpload` toggle.
 */
export function maybeAutoUpload(artifact: DatabaseArtifact): void {
	const cfg = readConnectionConfig();
	if (!cfg.url || !cfg.username || !cfg.autoUpload) return;
	const source = (artifact.metadata as Record<string, unknown> | undefined)?.source;
	if (source === 'nextcloud') return;
	void uploadArtifact(artifact);
}

/**
 * Mirror-delete: when the user removes an artifact locally and has
 * `mirrorDeletes` on, also DELETE the remote file. Best-effort —
 * the local removal already happened by the time we run, so a
 * remote failure is logged but doesn't propagate. WebDAV DELETE
 * treats 404 as success so a never-uploaded artifact is a no-op.
 */
export async function maybeMirrorDelete(artifact: DatabaseArtifact): Promise<void> {
	const cfg = readConnectionConfig();
	if (!cfg.mirrorDeletes) return;
	const sync = getNextcloudSync(artifact);
	if (!sync || !sync.remotePath) return;
	const client = await buildClient();
	if (!client) return;
	try {
		await client.delete(sync.remotePath);
	} catch (err) {
		console.warn('[nextcloud] mirror-delete failed for', artifact.id, err);
	}
}

function translateError(err: unknown): string {
	if (err instanceof WebDavError) {
		if (err.status === 401) return 'Auth rejected (app password may have been revoked).';
		if (err.status === 403) return 'Forbidden — app password lacks write access.';
		if (err.status === 404) return 'Folder not found on the server.';
		if (err.status === 0) return 'Bad response (URL might not be a Nextcloud / WebDAV endpoint).';
		return `HTTP ${err.status} ${err.statusText}`;
	}
	if (err instanceof WebDavNetworkError) {
		return 'Network error reaching the server (offline / CORS / DNS).';
	}
	return err instanceof Error ? err.message : String(err);
}

function normaliseRoot(input: string): string {
	const trimmed = input.trim();
	if (!trimmed || trimmed === '/') return '';
	const leading = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
	return leading.replace(/\/+$/, '');
}

function encodePath(p: string): string {
	return encodeURI(p).replace(/\/{2,}/g, '/');
}
