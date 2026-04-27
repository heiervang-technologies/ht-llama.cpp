<script lang="ts">
	import * as Sheet from '$lib/components/ui/sheet';
	import { Button } from '$lib/components/ui/button';
	import {
		Cloud,
		ChevronRight,
		Folder,
		FileText,
		Image as ImageIcon,
		Music,
		Video,
		FileCode,
		File as FileIcon,
		Home,
		RefreshCw,
		Loader2,
		ExternalLink,
		AlertTriangle
	} from '@lucide/svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { DatabaseService } from '$lib/services/database.service';
	import {
		WebDavClient,
		WebDavError,
		WebDavNetworkError,
		type WebDavResource
	} from '$lib/services/webdav.service';
	interface Props {
		open: boolean;
		onOpenChange: (next: boolean) => void;
	}

	let { open, onOpenChange }: Props = $props();

	const PASSWORD_KEY = 'nextcloud-app-password';

	// Path is *relative to* the user's configured `nextcloudRemoteRoot`.
	// "" or "/" both mean "the root the user opted into in Settings".
	// We store the segment list rather than a string to keep nav math
	// trivial — push to descend, pop to climb.
	let segments = $state<string[]>([]);
	let entries = $state<WebDavResource[]>([]);
	let isLoading = $state(false);
	let error = $state<{
		kind: 'config' | 'auth' | 'net' | 'http' | 'parse';
		message: string;
	} | null>(null);
	let selected = $state<WebDavResource | null>(null);

	// Avoid building a fresh client per render by memoising the active
	// config snapshot. The store is reactive, so anytime URL / username
	// / root changes, $derived rebuilds.
	let cfgSnapshot = $derived({
		url: String(config().nextcloudUrl ?? '').trim(),
		username: String(config().nextcloudUsername ?? '').trim(),
		remoteRoot: String(config().nextcloudRemoteRoot ?? '/AI/').trim()
	});

	let currentPath = $derived(segments.length === 0 ? '' : `/${segments.join('/')}`);
	let breadcrumbs = $derived(
		segments.map((seg, i) => ({ label: seg, depth: i + 1 }) as { label: string; depth: number })
	);

	$effect(() => {
		if (open) {
			void loadCurrent();
		}
	});

	async function loadCurrent(): Promise<void> {
		if (!cfgSnapshot.url || !cfgSnapshot.username) {
			error = {
				kind: 'config',
				message:
					'Nextcloud is not configured yet. Open Settings → Connections → Nextcloud and run Test connection first.'
			};
			entries = [];
			return;
		}
		const password = await DatabaseService.getSecret(PASSWORD_KEY);
		if (!password) {
			error = {
				kind: 'auth',
				message: 'No app password saved. Open Settings → Connections → Nextcloud and re-enter it.'
			};
			entries = [];
			return;
		}
		isLoading = true;
		error = null;
		selected = null;
		try {
			const client = new WebDavClient({
				baseUrl: cfgSnapshot.url,
				username: cfgSnapshot.username,
				password,
				remoteRoot: cfgSnapshot.remoteRoot || '/'
			});
			// Depth: 1 — the cloud-ops handoff explicitly recommended this
			// over `infinity` to keep the response cheap. Drilldown reloads
			// per click.
			const result = await client.propfind(currentPath, 1);
			if (result.length === 0) {
				entries = [];
				return;
			}
			// First entry is the resource itself; rest are children.
			entries = result.slice(1).sort(sortEntries);
		} catch (err) {
			error = translateError(err);
			entries = [];
		} finally {
			isLoading = false;
		}
	}

	function sortEntries(a: WebDavResource, b: WebDavResource): number {
		// Folders first, then files; both alphabetical.
		if (a.isCollection !== b.isCollection) return a.isCollection ? -1 : 1;
		return a.name.localeCompare(b.name, undefined, { numeric: true });
	}

	function translateError(err: unknown): {
		kind: 'auth' | 'net' | 'http' | 'parse';
		message: string;
	} {
		if (err instanceof WebDavError) {
			if (err.status === 401) {
				return {
					kind: 'auth',
					message:
						'Authentication rejected. The app password may have been revoked — re-enter it in Settings.'
				};
			}
			if (err.status === 403) {
				return {
					kind: 'http',
					message: 'Access denied. The app password lacks permission to read this folder.'
				};
			}
			if (err.status === 404) {
				return {
					kind: 'http',
					message: `Folder not found at /${segments.join('/')}.`
				};
			}
			if (err.status === 0) {
				return {
					kind: 'parse',
					message:
						'Could not parse the server response — the configured URL may not be a Nextcloud / WebDAV endpoint.'
				};
			}
			return { kind: 'http', message: `Server returned ${err.status} ${err.statusText}.` };
		}
		if (err instanceof WebDavNetworkError) {
			return {
				kind: 'net',
				message:
					'Network error reaching the server. The Nextcloud instance may be offline or blocked by CORS.'
			};
		}
		return {
			kind: 'http',
			message: err instanceof Error ? err.message : String(err)
		};
	}

	function descend(folder: WebDavResource): void {
		segments = [...segments, folder.name];
	}

	function jumpTo(depth: number): void {
		// depth 0 = root; otherwise truncate.
		segments = segments.slice(0, depth);
	}

	function refresh(): void {
		void loadCurrent();
	}

	function iconFor(entry: WebDavResource): typeof FileIcon {
		if (entry.isCollection) return Folder;
		const t = (entry.contentType ?? '').toLowerCase();
		const n = entry.name.toLowerCase();
		if (t.startsWith('image/') || /\.(png|jpe?g|gif|webp|svg|bmp|tiff?|avif)$/.test(n))
			return ImageIcon;
		if (t.startsWith('audio/') || /\.(mp3|wav|flac|ogg|opus|m4a)$/.test(n)) return Music;
		if (t.startsWith('video/') || /\.(mp4|mkv|webm|mov|avi)$/.test(n)) return Video;
		if (
			/\.(js|ts|tsx|jsx|py|rs|go|java|c|cc|cpp|h|hpp|rb|sh|json|yaml|yml|toml|xml|html?|css|scss|svelte|vue|md|markdown)$/.test(
				n
			)
		)
			return FileCode;
		if (t.startsWith('text/') || /\.(txt|log)$/.test(n)) return FileText;
		return FileIcon;
	}

	function fmtSize(bytes: number | undefined): string {
		if (typeof bytes !== 'number' || !Number.isFinite(bytes)) return '';
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
		if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
		return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
	}

	function fmtDate(s: string | undefined): string {
		if (!s) return '';
		const d = new Date(s);
		return Number.isFinite(d.getTime()) ? d.toLocaleString() : s;
	}

	function remoteUrl(entry: WebDavResource | null | undefined): string {
		if (!entry || !cfgSnapshot.url) return '';
		// `entry.href` is the server-relative DAV path. Combine with
		// the base URL for an "Open in Nextcloud" link. Nextcloud's
		// web UI uses a different URL shape (/apps/files/?dir=...) so
		// this opens the raw file rather than the file manager view —
		// good enough for the v1 drawer.
		return `${cfgSnapshot.url.replace(/\/+$/, '')}${entry.href}`;
	}

	function close(): void {
		onOpenChange(false);
	}
</script>

<Sheet.Root {open} {onOpenChange}>
	<Sheet.Content side="right" class="flex w-full flex-col gap-0 p-0 sm:max-w-2xl md:max-w-3xl">
		<Sheet.Header class="gap-1 border-b border-border/40 px-4 py-3">
			<div class="flex items-center gap-2">
				<Cloud class="h-4 w-4 text-primary" aria-hidden="true" />
				<Sheet.Title class="text-base font-medium">Nextcloud</Sheet.Title>
			</div>
			<Sheet.Description class="text-xs text-muted-foreground">
				{#if cfgSnapshot.url}
					Browsing <code class="rounded bg-muted px-1 py-0.5">{cfgSnapshot.remoteRoot || '/'}</code>
					on
					{cfgSnapshot.url.replace(/^https?:\/\//, '')}
				{:else}
					Configure the Nextcloud connection in Settings to browse remote files.
				{/if}
			</Sheet.Description>
		</Sheet.Header>

		<!-- Breadcrumbs + actions -->
		<div class="flex items-center gap-1 border-b border-border/40 px-3 py-2 text-xs">
			<button
				type="button"
				class="inline-flex items-center gap-1 rounded px-2 py-1 hover:bg-accent/40"
				title="Go to root"
				onclick={() => jumpTo(0)}
			>
				<Home class="h-3.5 w-3.5" />
				{cfgSnapshot.remoteRoot || '/'}
			</button>
			{#each breadcrumbs as crumb (crumb.depth)}
				<ChevronRight class="h-3 w-3 text-muted-foreground" aria-hidden="true" />
				<button
					type="button"
					class="rounded px-2 py-1 hover:bg-accent/40"
					onclick={() => jumpTo(crumb.depth)}
				>
					{crumb.label}
				</button>
			{/each}
			<div class="ml-auto flex items-center gap-1">
				<Button variant="ghost" size="sm" onclick={refresh} disabled={isLoading} title="Refresh">
					{#if isLoading}
						<Loader2 class="h-3.5 w-3.5 animate-spin" />
					{:else}
						<RefreshCw class="h-3.5 w-3.5" />
					{/if}
				</Button>
			</div>
		</div>

		<!-- Listing -->
		<div class="min-h-0 flex-1 overflow-auto">
			{#if error}
				<div
					class="m-4 flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs"
				>
					<AlertTriangle class="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-500" aria-hidden="true" />
					<div>{error.message}</div>
				</div>
			{:else if isLoading && entries.length === 0}
				<div class="flex h-full items-center justify-center text-xs text-muted-foreground">
					<Loader2 class="mr-2 h-3.5 w-3.5 animate-spin" />
					Loading…
				</div>
			{:else if entries.length === 0}
				<div class="flex h-full items-center justify-center text-xs text-muted-foreground">
					Empty folder.
				</div>
			{:else}
				<ul class="divide-y divide-border/40">
					{#each entries as entry (entry.href)}
						{@const Icon = iconFor(entry)}
						<li>
							<button
								type="button"
								class="flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm hover:bg-accent/40
									{selected?.href === entry.href ? 'bg-accent/40' : ''}"
								onclick={() => {
									if (entry.isCollection) {
										descend(entry);
									} else {
										selected = selected?.href === entry.href ? null : entry;
									}
								}}
							>
								<Icon class="h-4 w-4 flex-shrink-0 text-muted-foreground" aria-hidden="true" />
								<span class="min-w-0 flex-1 truncate">{entry.name}</span>
								<span class="text-[11px] text-muted-foreground">
									{entry.isCollection ? '' : fmtSize(entry.contentLength)}
								</span>
							</button>
						</li>
					{/each}
				</ul>
			{/if}
		</div>

		<!-- File detail strip -->
		{#if selected && !selected.isCollection}
			<div class="border-t border-border/40 bg-muted/20 px-4 py-3 text-xs">
				<div class="font-medium">{selected.name}</div>
				<dl class="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-muted-foreground">
					{#if selected.contentType}
						<dt>Type</dt>
						<dd class="font-mono">{selected.contentType}</dd>
					{/if}
					{#if selected.contentLength != null}
						<dt>Size</dt>
						<dd>{fmtSize(selected.contentLength)} ({selected.contentLength} bytes)</dd>
					{/if}
					{#if selected.lastModified}
						<dt>Modified</dt>
						<dd>{fmtDate(selected.lastModified)}</dd>
					{/if}
					{#if selected.etag}
						<dt>ETag</dt>
						<dd class="truncate font-mono">{selected.etag}</dd>
					{/if}
				</dl>
				<div class="mt-2 flex gap-2">
					<!-- Add-to-gallery is wired in step (c). Disabled stub
					     keeps the button visible so users see the planned
					     surface; tooltip explains. -->
					<Button size="sm" variant="default" disabled title="Wired up in the next commit">
						<Cloud class="h-3.5 w-3.5" />
						Add to gallery
					</Button>
					<a
						href={remoteUrl(selected)}
						target="_blank"
						rel="noopener noreferrer"
						class="inline-flex items-center gap-1 rounded-md border bg-background px-2 py-1 text-xs hover:bg-accent/40"
					>
						<ExternalLink class="h-3 w-3" />
						Open raw
					</a>
				</div>
			</div>
		{/if}

		<Sheet.Footer class="border-t border-border/40 px-4 py-2">
			<Button variant="ghost" size="sm" onclick={close}>Close</Button>
		</Sheet.Footer>
	</Sheet.Content>
</Sheet.Root>
