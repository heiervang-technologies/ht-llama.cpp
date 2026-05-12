/**
 * Tiny WebDAV client. Targeted specifically at Nextcloud's `/remote.php/dav/files/<user>/`
 * tree but the protocol bits are vanilla — drop-in for any server that speaks WebDAV
 * with HTTP basic auth.
 *
 * Why not the `webdav` npm package: ~50 KB minified, opinionated about Node
 * `Buffer`, ships its own XML parser. We need a handful of verbs (PROPFIND,
 * PUT, GET, DELETE, MKCOL) and the browser already has DOMParser. Cheaper to
 * write directly.
 *
 * CORS: the cluster's nginx ingress echoes Access-Control-* for an allow-list
 * of origins (llm.ht.local, localhost:5173, localhost:1420, tauri://localhost).
 * Pure-web mode and Tauri both work; if a future origin needs adding it has
 * to land on the cluster side, not here.
 *
 * Path resolution: callers pass user-friendly paths ("/AI", "/AI/2026-04-27").
 * The client stitches them onto `/remote.php/dav/files/<user>/` so users
 * never have to learn the DAV layout. The optional `remoteRoot` is prepended
 * before the friendly path — that's the "Remote folder root" setting.
 */

export interface WebDavConfig {
	/** Base server URL, e.g. `http://nextcloud.ht.local`. No trailing slash required. */
	baseUrl: string;
	username: string;
	/** Nextcloud app password (NOT the admin password). */
	password: string;
	/** User-facing root prefix appended after `/remote.php/dav/files/<user>/`.
	 *  e.g. `/AI/`. Leading and trailing slashes are normalised. */
	remoteRoot?: string;
}

/** Result of a PROPFIND on a single resource. */
export interface WebDavResource {
	/** The DAV `<href>`, exactly as the server returned it. URL-decoded. */
	href: string;
	/** Display name (basename). */
	name: string;
	/** True if the resource is a collection (folder). */
	isCollection: boolean;
	contentLength?: number;
	contentType?: string;
	lastModified?: string;
	etag?: string;
}

export class WebDavError extends Error {
	constructor(
		public readonly status: number,
		public readonly statusText: string,
		public readonly body: string,
		message?: string
	) {
		super(message ?? `WebDAV ${status} ${statusText}`);
		this.name = 'WebDavError';
	}
}

/** Thrown when the user is offline or the server is unreachable. */
export class WebDavNetworkError extends Error {
	constructor(public readonly cause: unknown) {
		super('Network error reaching WebDAV server');
		this.name = 'WebDavNetworkError';
	}
}

export class WebDavClient {
	private readonly baseUrl: string;
	private readonly userPath: string;
	private readonly remoteRoot: string;
	private readonly authHeader: string;

	constructor(cfg: WebDavConfig) {
		this.baseUrl = cfg.baseUrl.replace(/\/+$/, '');
		this.userPath = `/remote.php/dav/files/${encodeURIComponent(cfg.username)}`;
		this.remoteRoot = normaliseFolder(cfg.remoteRoot ?? '');
		// btoa works fine for ASCII basic auth; passwords containing
		// non-Latin1 are rare for app passwords (Nextcloud generates
		// ASCII). If someone hits this, encode the credential into
		// UTF-8 bytes manually first.
		this.authHeader = `Basic ${btoa(`${cfg.username}:${cfg.password}`)}`;
	}

	/** Resolve a user-facing path ("/2026-04-27/foo.png") to a full URL. */
	private url(friendlyPath: string): string {
		const normalised = friendlyPath.startsWith('/') ? friendlyPath : `/${friendlyPath}`;
		// encodeURI keeps `/` separators intact; encodeURIComponent
		// would mangle them. Filenames with `#` or `?` are still safe
		// because encodeURI escapes those.
		const encoded = encodeURI(`${this.userPath}${this.remoteRoot}${normalised}`).replace(
			/\/{2,}/g,
			'/'
		);
		return `${this.baseUrl}${encoded}`;
	}

	private async request(
		method: string,
		path: string,
		init: { headers?: Record<string, string>; body?: BodyInit | null } = {}
	): Promise<Response> {
		const headers: Record<string, string> = {
			Authorization: this.authHeader,
			...(init.headers ?? {})
		};
		try {
			return await fetch(this.url(path), {
				method,
				headers,
				body: init.body ?? null,
				// Nextcloud doesn't need cookies; sending credentials
				// would force the credentialed-CORS path which the
				// cluster ingress only echoes for explicit origins.
				// Basic auth via the header is enough.
				credentials: 'omit',
				mode: 'cors'
			});
		} catch (cause) {
			throw new WebDavNetworkError(cause);
		}
	}

	private async assertOk(
		method: string,
		path: string,
		res: Response,
		allow: number[]
	): Promise<void> {
		if (allow.includes(res.status)) return;
		const body = await res.text().catch(() => '');
		throw new WebDavError(res.status, res.statusText, body, `${method} ${path} -> ${res.status}`);
	}

	/**
	 * PROPFIND — list a resource (or its children when `depth=1`).
	 * Returns the parsed `<response>` entries. The first entry is the
	 * resource itself; subsequent entries (when depth=1) are its
	 * children.
	 */
	async propfind(path: string, depth: 0 | 1 = 1): Promise<WebDavResource[]> {
		const body = `<?xml version="1.0" encoding="utf-8"?>
<d:propfind xmlns:d="DAV:">
  <d:prop>
    <d:displayname/>
    <d:resourcetype/>
    <d:getcontentlength/>
    <d:getcontenttype/>
    <d:getlastmodified/>
    <d:getetag/>
  </d:prop>
</d:propfind>`;
		const res = await this.request('PROPFIND', path, {
			headers: {
				Depth: String(depth),
				'Content-Type': 'application/xml; charset=utf-8'
			},
			body
		});
		await this.assertOk('PROPFIND', path, res, [207]);
		const xml = await res.text();
		return parsePropfind(xml);
	}

	/** GET a file's bytes. */
	async get(path: string): Promise<Blob> {
		const res = await this.request('GET', path);
		await this.assertOk('GET', path, res, [200]);
		return res.blob();
	}

	/** PUT bytes at `path`, creating or overwriting. Server-side validation
	 *  decides whether to allow overwrite — pass `If-None-Match: *` via
	 *  `extraHeaders` to refuse overwrite. Returns the server-assigned
	 *  ETag if the response exposed one (Nextcloud's nginx ingress
	 *  whitelists ETag in Access-Control-Expose-Headers). */
	async put(
		path: string,
		body: Blob | ArrayBuffer | Uint8Array | string,
		opts: { contentType?: string; extraHeaders?: Record<string, string> } = {}
	): Promise<{ etag: string | null }> {
		const headers: Record<string, string> = {
			'Content-Type': opts.contentType ?? 'application/octet-stream',
			...(opts.extraHeaders ?? {})
		};
		const res = await this.request('PUT', path, { headers, body: body as BodyInit });
		// 201 Created (new), 204 No Content (overwrite). Some servers
		// also return 200 on overwrite — accept.
		await this.assertOk('PUT', path, res, [200, 201, 204]);
		return { etag: res.headers.get('ETag') };
	}

	/** DELETE a file or collection. 404 is treated as success — caller's
	 *  intent ("ensure gone") is satisfied. */
	async delete(path: string): Promise<void> {
		const res = await this.request('DELETE', path);
		if (res.status === 404) return;
		await this.assertOk('DELETE', path, res, [200, 204]);
	}

	/** MKCOL — create a collection. 405 means the collection already
	 *  exists; treat as success per cloud-ops guidance (idempotent). */
	async mkcol(path: string): Promise<void> {
		const res = await this.request('MKCOL', path);
		await this.assertOk('MKCOL', path, res, [201, 405]);
	}

	/**
	 * Idempotent recursive folder create. Walks the path one segment at
	 * a time so we don't fail when an intermediate directory already
	 * exists. WebDAV has no "MKCOL recursive" verb so this is the
	 * accepted pattern.
	 */
	async ensureFolder(path: string): Promise<void> {
		const parts = path.split('/').filter(Boolean);
		let acc = '';
		for (const seg of parts) {
			acc += `/${seg}`;
			await this.mkcol(acc);
		}
	}
}

/** Normalise to leading slash + trailing slash. Empty string in -> `/` out. */
function normaliseFolder(input: string): string {
	const trimmed = input.trim();
	if (!trimmed || trimmed === '/') return '/';
	const leading = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
	return leading.endsWith('/') ? leading : `${leading}/`;
}

function parsePropfind(xml: string): WebDavResource[] {
	// PROPFIND returns multistatus XML. The DOMParser path is fine for
	// the size we expect (folder listings on Nextcloud cap out around
	// a few MB even for big directories). Streaming would be overkill.
	const doc = new DOMParser().parseFromString(xml, 'application/xml');
	if (doc.querySelector('parsererror')) {
		throw new WebDavError(0, 'parse', xml, 'malformed multistatus XML');
	}
	const responses = Array.from(doc.getElementsByTagNameNS('DAV:', 'response'));
	return responses.map((r) => {
		const hrefRaw = r.getElementsByTagNameNS('DAV:', 'href')[0]?.textContent ?? '';
		const href = safeDecode(hrefRaw.trim());
		const propstat = r.getElementsByTagNameNS('DAV:', 'propstat')[0];
		const prop = propstat?.getElementsByTagNameNS('DAV:', 'prop')[0];
		const isCollection = !!prop
			?.getElementsByTagNameNS('DAV:', 'resourcetype')[0]
			?.getElementsByTagNameNS('DAV:', 'collection').length;
		const displayname =
			prop?.getElementsByTagNameNS('DAV:', 'displayname')[0]?.textContent?.trim() || '';
		const name = displayname || basename(href);
		const lengthStr =
			prop?.getElementsByTagNameNS('DAV:', 'getcontentlength')[0]?.textContent?.trim() ?? '';
		const contentLength = lengthStr ? Number(lengthStr) : undefined;
		const contentType =
			prop?.getElementsByTagNameNS('DAV:', 'getcontenttype')[0]?.textContent?.trim() || undefined;
		const lastModified =
			prop?.getElementsByTagNameNS('DAV:', 'getlastmodified')[0]?.textContent?.trim() || undefined;
		const etag =
			prop?.getElementsByTagNameNS('DAV:', 'getetag')[0]?.textContent?.trim() || undefined;
		return { href, name, isCollection, contentLength, contentType, lastModified, etag };
	});
}

function basename(path: string): string {
	const trimmed = path.replace(/\/+$/, '');
	const slash = trimmed.lastIndexOf('/');
	return slash >= 0 ? trimmed.slice(slash + 1) : trimmed;
}

function safeDecode(s: string): string {
	try {
		return decodeURIComponent(s);
	} catch {
		return s;
	}
}
