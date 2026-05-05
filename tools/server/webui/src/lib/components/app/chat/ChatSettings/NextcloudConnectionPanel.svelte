<script lang="ts">
	import { onMount } from 'svelte';
	import { Cloud, ExternalLink, Eye, EyeOff, Loader2, Check, X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Switch } from '$lib/components/ui/switch';
	import { Label } from '$lib/components/ui/label';
	import { DatabaseService } from '$lib/services/database.service';
	import { WebDavClient, WebDavError, WebDavNetworkError } from '$lib/services/webdav.service';
	import { SETTINGS_KEYS } from '$lib/constants';
	import type { SettingsConfigType } from '$lib/types';

	interface Props {
		localConfig: SettingsConfigType;
		// Matches `handleConfigChange` in ChatSettings.svelte exactly
		// — strings cover URL/username/path inputs, booleans cover the
		// auto-upload / mirror-deletes switches.
		onConfigChange: (key: string, value: string | boolean) => void;
	}

	let { localConfig, onConfigChange }: Props = $props();

	const PASSWORD_KEY = 'nextcloud-app-password';

	// Local state — the password lives in IndexedDB, not the prefs blob,
	// so we read it on mount and write it on Save below. URL/username/
	// remoteRoot/auto-upload/mirror-deletes flow through `onConfigChange`
	// the same way every other field does, so the dialog's Save button
	// persists them with everything else.
	let appPassword = $state('');
	let initialAppPassword = $state('');
	let showPassword = $state(false);
	let savingPassword = $state(false);

	let testStatus = $state<
		| { kind: 'idle' }
		| { kind: 'testing' }
		| { kind: 'ok'; message: string }
		| { kind: 'fail'; message: string; httpStatus?: number }
	>({ kind: 'idle' });

	let url = $derived(String(localConfig[SETTINGS_KEYS.NEXTCLOUD_URL] ?? ''));
	let username = $derived(String(localConfig[SETTINGS_KEYS.NEXTCLOUD_USERNAME] ?? ''));
	let remoteRoot = $derived(String(localConfig[SETTINGS_KEYS.NEXTCLOUD_REMOTE_ROOT] ?? '/AI/'));
	let autoUpload = $derived(Boolean(localConfig[SETTINGS_KEYS.NEXTCLOUD_AUTO_UPLOAD]));
	let mirrorDeletes = $derived(Boolean(localConfig[SETTINGS_KEYS.NEXTCLOUD_MIRROR_DELETES]));

	let canTest = $derived(Boolean(url.trim() && username.trim() && appPassword.trim()));
	let dirty = $derived(appPassword !== initialAppPassword);

	onMount(async () => {
		try {
			const stored = await DatabaseService.getSecret(PASSWORD_KEY);
			if (stored) {
				appPassword = stored;
				initialAppPassword = stored;
			}
		} catch (err) {
			console.warn('[nextcloud] failed to load app password', err);
		}
	});

	async function persistPassword(): Promise<void> {
		savingPassword = true;
		try {
			if (appPassword) {
				await DatabaseService.setSecret(PASSWORD_KEY, appPassword);
			} else {
				await DatabaseService.clearSecret(PASSWORD_KEY);
			}
			initialAppPassword = appPassword;
		} catch (err) {
			console.warn('[nextcloud] failed to persist app password', err);
		} finally {
			savingPassword = false;
		}
	}

	async function handleTest(): Promise<void> {
		if (!canTest) return;
		// Ensure the in-memory password matches what the test will use.
		// We don't wait for the user to hit Save — the test should reflect
		// what they just typed.
		await persistPassword();
		testStatus = { kind: 'testing' };
		try {
			const client = new WebDavClient({
				baseUrl: url.trim(),
				username: username.trim(),
				password: appPassword,
				remoteRoot: remoteRoot.trim() || '/'
			});
			const entries = await client.propfind('', 0);
			const root = entries[0];
			testStatus = {
				kind: 'ok',
				message: root?.isCollection
					? `Connected. Root collection found at ${remoteRoot.trim() || '/'}.`
					: `Connected, but ${remoteRoot.trim() || '/'} is not a folder.`
			};
		} catch (err) {
			if (err instanceof WebDavError) {
				const reason =
					err.status === 401
						? 'Authentication failed. Check the username and app password.'
						: err.status === 403
							? 'Forbidden. The app password may not have read access to the root folder.'
							: err.status === 404
								? `Root folder ${remoteRoot.trim() || '/'} not found. Create it on the server or pick a different path.`
								: err.status === 0
									? 'Could not parse the server response (is this URL really a Nextcloud / WebDAV endpoint?).'
									: `Server returned ${err.status} ${err.statusText}.`;
				testStatus = { kind: 'fail', message: reason, httpStatus: err.status };
			} else if (err instanceof WebDavNetworkError) {
				testStatus = {
					kind: 'fail',
					message:
						'Network error. The server may be offline, blocked by CORS, or unreachable from this device.'
				};
			} else {
				testStatus = {
					kind: 'fail',
					message: err instanceof Error ? err.message : String(err)
				};
			}
		}
	}

	// Toggle helpers exist mainly so the Switch's `onCheckedChange`
	// gets a typed function rather than an inline arrow that the
	// dialog's typed `onConfigChange` would balk at — the wider
	// SettingsConfigValue accepts undefined which the boolean
	// signature does not.
	function handleAutoUploadToggle(value: boolean): void {
		onConfigChange(SETTINGS_KEYS.NEXTCLOUD_AUTO_UPLOAD, value);
	}

	function handleMirrorDeletesToggle(value: boolean): void {
		onConfigChange(SETTINGS_KEYS.NEXTCLOUD_MIRROR_DELETES, value);
	}
</script>

<div class="space-y-5">
	<header class="flex items-center gap-2">
		<Cloud class="h-5 w-5 text-primary" aria-hidden="true" />
		<div>
			<h3 class="text-sm font-semibold">Nextcloud</h3>
			<p class="text-xs text-muted-foreground">
				Sync gallery artifacts to a WebDAV-backed Nextcloud server. Credentials stay on this device.
			</p>
		</div>
	</header>

	<div class="space-y-3 rounded-lg border bg-muted/20 p-4">
		<div class="space-y-1">
			<Label for="nc-url">Server URL</Label>
			<Input
				id="nc-url"
				type="url"
				placeholder="http://nextcloud.ht.local"
				value={url}
				oninput={(e) =>
					onConfigChange(SETTINGS_KEYS.NEXTCLOUD_URL, (e.currentTarget as HTMLInputElement).value)}
			/>
		</div>

		<div class="space-y-1">
			<Label for="nc-username">Username</Label>
			<Input
				id="nc-username"
				type="text"
				autocomplete="username"
				placeholder="admin"
				value={username}
				oninput={(e) =>
					onConfigChange(
						SETTINGS_KEYS.NEXTCLOUD_USERNAME,
						(e.currentTarget as HTMLInputElement).value
					)}
			/>
		</div>

		<div class="space-y-1">
			<Label for="nc-password">App password</Label>
			<div class="relative">
				<Input
					id="nc-password"
					type={showPassword ? 'text' : 'password'}
					autocomplete="off"
					spellcheck={false}
					placeholder="•••• •••• •••• ••••"
					bind:value={appPassword}
					onblur={() => persistPassword()}
					class="pr-9"
				/>
				<button
					type="button"
					class="absolute top-1/2 right-2 -translate-y-1/2 rounded p-1 text-muted-foreground hover:text-foreground"
					aria-label={showPassword ? 'Hide password' : 'Show password'}
					onclick={() => (showPassword = !showPassword)}
				>
					{#if showPassword}
						<EyeOff class="h-3.5 w-3.5" />
					{:else}
						<Eye class="h-3.5 w-3.5" />
					{/if}
				</button>
			</div>
			<p class="text-[11px] leading-snug text-muted-foreground">
				Use an <strong>app password</strong>, not your account password.
				{#if url.trim()}
					{@const settingsUrl = `${url.trim().replace(/\/+$/, '')}/settings/user/security`}
					Create one at
					<a
						href={settingsUrl}
						target="_blank"
						rel="noopener noreferrer"
						class="inline-flex items-center gap-1 underline"
					>
						{settingsUrl}
						<ExternalLink class="h-2.5 w-2.5" />
					</a>
				{:else}
					Open Nextcloud → Settings → Security → "Devices &amp; sessions" → "Create new app
					password".
				{/if}
				Stored in IndexedDB on this device only — never sent anywhere except the configured server.
			</p>
		</div>

		<div class="space-y-1">
			<Label for="nc-root">Remote folder root</Label>
			<Input
				id="nc-root"
				type="text"
				placeholder="/AI/"
				value={remoteRoot}
				oninput={(e) =>
					onConfigChange(
						SETTINGS_KEYS.NEXTCLOUD_REMOTE_ROOT,
						(e.currentTarget as HTMLInputElement).value
					)}
			/>
			<p class="text-[11px] leading-snug text-muted-foreground">
				Friendly path under your account. We resolve it to <code
					class="rounded bg-muted px-1 py-0.5"
					>{`/remote.php/dav/files/<user>/${(remoteRoot || '/AI/').replace(/^\/+/, '').replace(/\/+$/, '')}/`}</code
				> at request time, so you don't need to learn DAV paths.
			</p>
		</div>

		<div class="flex items-center gap-2 pt-1">
			<Button
				size="sm"
				variant="default"
				disabled={!canTest || testStatus.kind === 'testing'}
				onclick={handleTest}
			>
				{#if testStatus.kind === 'testing'}
					<Loader2 class="h-3.5 w-3.5 animate-spin" />
					Testing…
				{:else}
					<Cloud class="h-3.5 w-3.5" />
					Test connection
				{/if}
			</Button>

			{#if testStatus.kind === 'ok'}
				<span class="inline-flex items-center gap-1 text-xs text-emerald-600 dark:text-emerald-400">
					<Check class="h-3.5 w-3.5" />
					{testStatus.message}
				</span>
			{:else if testStatus.kind === 'fail'}
				<span class="inline-flex items-center gap-1 text-xs text-destructive">
					<X class="h-3.5 w-3.5" />
					{testStatus.message}
				</span>
			{/if}

			{#if dirty || savingPassword}
				<span class="ml-auto text-[11px] text-muted-foreground italic">
					{savingPassword ? 'Saving…' : 'Password edited — saves on blur or test'}
				</span>
			{/if}
		</div>
	</div>

	<div class="space-y-3 rounded-lg border bg-muted/20 p-4">
		<div class="flex items-start justify-between gap-3">
			<div>
				<Label class="text-sm">Auto-upload new artifacts</Label>
				<p class="text-[11px] text-muted-foreground">
					New gallery artifacts (images, videos, code, generated assets) sync to Nextcloud as soon
					as they're created. Existing artifacts are left alone — bulk backfill is a separate flow.
				</p>
			</div>
			<Switch checked={autoUpload} onCheckedChange={handleAutoUploadToggle} />
		</div>

		<div class="flex items-start justify-between gap-3 border-t pt-3">
			<div>
				<Label class="text-sm">Mirror deletes</Label>
				<p class="text-[11px] text-muted-foreground">
					When you delete an artifact locally, also remove the matching file on Nextcloud. Off by
					default — opt in if you want a single source of truth.
				</p>
			</div>
			<Switch checked={mirrorDeletes} onCheckedChange={handleMirrorDeletesToggle} />
		</div>
	</div>

	<p class="text-[11px] text-muted-foreground">
		Auto-upload and sync indicators light up only after a successful Test connection.
	</p>
</div>
