<script lang="ts">
	import { onDestroy } from 'svelte';
	import { goto } from '$app/navigation';
	import type { DatabaseArtifact } from '$lib/types/database';
	import { DatabaseService } from '$lib/services/database.service';
	import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';
	import { buildThumb, revokeThumb, type ArtifactThumb } from '$lib/utils/artifact-thumb';
	import {
		FileCode,
		FileText,
		FileImage,
		FileAudio,
		FileVideo,
		FileBadge,
		Trash2,
		Check,
		Cloud,
		CloudOff,
		Loader2,
		AlertTriangle
	} from '@lucide/svelte';
	import {
		getNextcloudSync,
		nextcloudSyncRuntime,
		isNextcloudConfigured,
		uploadArtifact
	} from '$lib/services/nextcloud-upload.service';
	import { toast } from 'svelte-sonner';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';

	interface Props {
		artifact: DatabaseArtifact;
		selectable?: boolean;
		selected?: boolean;
		onToggleSelect?: () => void;
	}
	let { artifact, selectable = false, selected = false, onToggleSelect }: Props = $props();

	let thumb = $state<ArtifactThumb | null>(null);
	let revisionCount = $state(1);

	$effect(() => {
		let cancelled = false;
		let localThumb: ArtifactThumb | null = null;
		(async () => {
			const revs = await DatabaseService.listArtifactRevisions(artifact.id);
			if (cancelled) return;
			revisionCount = revs.length;
			const current = revs.find((r) => r.id === artifact.currentRevisionId) ?? revs.at(-1);
			localThumb = buildThumb(artifact.kind, current);
			if (!cancelled) thumb = localThumb;
		})();
		return () => {
			cancelled = true;
			revokeThumb(localThumb);
		};
	});

	onDestroy(() => {
		revokeThumb(thumb);
	});

	let showDeleteConfirm = $state(false);

	function quickDelete(ev: Event) {
		ev.stopPropagation();
		showDeleteConfirm = true;
	}

	async function confirmDelete() {
		showDeleteConfirm = false;
		await artifactGalleryStore.remove(artifact.id);
	}

	function onClick() {
		if (selectable) {
			onToggleSelect?.();
		} else {
			goto(`#/artifacts/${artifact.id}`);
		}
	}

	const KIND_ICON = {
		html: FileCode,
		svg: FileCode,
		code: FileCode,
		markdown: FileText,
		image: FileImage,
		audio: FileAudio,
		video: FileVideo,
		pdf: FileBadge
	} as const;

	let Icon = $derived(KIND_ICON[artifact.kind]);

	// Sync state — derived directly from the artifact metadata + the
	// in-flight runtime store. Connection-not-configured suppresses the
	// badge entirely (no point yelling about "not synced" when the user
	// hasn't opted into the integration).
	let syncShown = $derived(isNextcloudConfigured());
	let isUploading = $derived(nextcloudSyncRuntime.isUploading(artifact.id));
	let sync = $derived(getNextcloudSync(artifact));
	let cameFromNextcloud = $derived(
		(artifact.metadata as Record<string, unknown> | undefined)?.source === 'nextcloud'
	);

	async function retryUpload(ev: Event) {
		ev.stopPropagation();
		const result = await uploadArtifact(artifact);
		if (result?.status === 'synced') toast.success('Re-synced to Nextcloud');
		else if (result?.status === 'failed') toast.error(`Re-sync failed — ${result.error ?? ''}`);
	}
</script>

<div
	class="group relative flex h-full flex-col overflow-hidden rounded-xl border bg-card text-left shadow-sm transition hover:border-primary hover:shadow-md {selectable &&
	selected
		? 'ring-2 ring-primary'
		: ''}"
>
	<button
		type="button"
		class="flex h-full w-full flex-col text-left"
		onclick={onClick}
		aria-label={selectable
			? (selected ? 'Deselect' : 'Select') + ` ${artifact.title}`
			: `Open ${artifact.title}`}
	>
		<div
			class="relative flex h-40 w-full items-center justify-center overflow-hidden bg-muted/50 text-muted-foreground"
		>
			{#if thumb?.kind === 'image'}
				<img src={thumb.url} alt={artifact.title} class="h-full w-full object-cover" />
			{:else if thumb?.kind === 'video'}
				<!-- preload="none": webkit2gtk's GStreamer pipeline crashes
				   the WebProcess on metadata probing for some clips. The
				   gallery card loses its poster frame on cold load, but
				   the detail page ships a full player so the user still
				   has a way to see the content. -->
				<video src={thumb.url} preload="none" muted class="h-full w-full object-cover"></video>
			{:else if thumb?.kind === 'audio'}
				<div class="flex flex-col items-center gap-2">
					<Icon class="h-8 w-8" />
					<span class="text-xs">audio</span>
				</div>
			{:else if thumb?.kind === 'pdf'}
				<div class="flex flex-col items-center gap-2">
					<Icon class="h-8 w-8" />
					<span class="text-xs">PDF</span>
				</div>
			{:else if thumb?.kind === 'svg'}
				<!-- eslint-disable-next-line svelte/no-at-html-tags -->
				<div class="flex h-full w-full items-center justify-center p-4">
					{@html thumb.markup}
				</div>
			{:else if thumb && (thumb.kind === 'code' || thumb.kind === 'html' || thumb.kind === 'markdown')}
				<pre
					class="w-full overflow-hidden p-3 font-mono text-[11px] leading-snug break-words whitespace-pre-wrap">{thumb.excerpt}</pre>
			{:else}
				<Icon class="h-8 w-8" />
			{/if}
		</div>
		<div class="flex flex-1 flex-col gap-1 border-t p-3">
			<div class="flex items-center justify-between gap-2">
				<span class="truncate text-sm font-medium">{artifact.title}</span>
				<span
					class="flex-shrink-0 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] text-primary uppercase"
					>{artifact.kind}</span
				>
			</div>
			<div class="flex items-center justify-between gap-2 text-xs text-muted-foreground">
				<span>
					{revisionCount}
					{revisionCount === 1 ? 'revision' : 'revisions'}
				</span>
				<time title={new Date(artifact.updatedAt).toLocaleString()}>
					{new Date(artifact.updatedAt).toLocaleDateString()}
				</time>
			</div>
		</div>
	</button>

	{#if syncShown}
		{#if isUploading}
			<div
				class="pointer-events-none absolute bottom-2 left-2 flex items-center gap-1 rounded-full border border-muted-foreground/30 bg-background/80 px-2 py-0.5 text-[10px] text-muted-foreground"
				title="Uploading to Nextcloud"
			>
				<Loader2 class="h-3 w-3 animate-spin" />
				syncing
			</div>
		{:else if cameFromNextcloud}
			<div
				class="pointer-events-none absolute bottom-2 left-2 flex items-center gap-1 rounded-full border border-muted-foreground/30 bg-background/80 px-2 py-0.5 text-[10px] text-muted-foreground"
				title="Imported from Nextcloud"
			>
				<Cloud class="h-3 w-3" />
				cloud
			</div>
		{:else if sync?.status === 'synced'}
			{#if sync.remoteUrl}
				<a
					href={sync.remoteUrl}
					target="_blank"
					rel="noopener noreferrer"
					onclick={(ev) => ev.stopPropagation()}
					class="absolute bottom-2 left-2 flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] text-emerald-600 transition hover:bg-emerald-500/20 dark:text-emerald-400"
					title={sync.lastSyncedAt
						? `Synced to Nextcloud at ${new Date(sync.lastSyncedAt).toLocaleString()} — open remote file`
						: 'Synced to Nextcloud — open remote file'}
				>
					<Cloud class="h-3 w-3" />
					synced
				</a>
			{:else}
				<div
					class="pointer-events-none absolute bottom-2 left-2 flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] text-emerald-600 dark:text-emerald-400"
					title={sync.lastSyncedAt
						? `Synced to Nextcloud at ${new Date(sync.lastSyncedAt).toLocaleString()}`
						: 'Synced to Nextcloud'}
				>
					<Cloud class="h-3 w-3" />
					synced
				</div>
			{/if}
		{:else if sync?.status === 'failed'}
			<button
				type="button"
				class="absolute bottom-2 left-2 flex items-center gap-1 rounded-full border border-destructive/40 bg-destructive/10 px-2 py-0.5 text-[10px] text-destructive transition hover:bg-destructive/20"
				onclick={retryUpload}
				title={sync.error
					? `Sync failed — ${sync.error}. Click to retry.`
					: 'Sync failed — click to retry'}
				aria-label="Retry Nextcloud sync"
			>
				<AlertTriangle class="h-3 w-3" />
				retry
			</button>
		{:else}
			<div
				class="pointer-events-none absolute bottom-2 left-2 flex items-center gap-1 rounded-full border border-muted-foreground/20 bg-background/80 px-2 py-0.5 text-[10px] text-muted-foreground/70"
				title="Not synced to Nextcloud yet"
			>
				<CloudOff class="h-3 w-3" />
				local
			</div>
		{/if}
	{/if}

	{#if selectable}
		<div
			class="pointer-events-none absolute top-2 left-2 flex h-6 w-6 items-center justify-center rounded-md border {selected
				? 'border-primary bg-primary text-primary-foreground'
				: 'border-muted-foreground/40 bg-background/80'}"
			aria-hidden="true"
		>
			{#if selected}
				<Check class="h-3.5 w-3.5" />
			{/if}
		</div>
	{:else}
		<!-- Quick-delete sits in the corner, opacity-0 until hover so it doesn't
			   compete with the thumbnail. Outside the main <button> so clicks don't
			   fall through to navigation. -->
		<button
			type="button"
			class="hover:text-destructive-foreground absolute top-2 right-2 rounded-md border border-destructive/40 bg-background/80 p-1 text-destructive opacity-0 transition group-hover:opacity-100 hover:bg-destructive"
			onclick={quickDelete}
			aria-label={`Delete ${artifact.title}`}
			title="Delete artifact"
		>
			<Trash2 class="h-3.5 w-3.5" />
		</button>
	{/if}
</div>

<AlertDialog.Root bind:open={showDeleteConfirm}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>Delete artifact?</AlertDialog.Title>
			<AlertDialog.Description>
				This removes <span class="font-mono">"{artifact.title}"</span> and all its revisions from the
				gallery. The action cannot be undone.
			</AlertDialog.Description>
		</AlertDialog.Header>
		<AlertDialog.Footer>
			<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>
			<AlertDialog.Action
				class="text-destructive-foreground bg-destructive hover:bg-destructive/90"
				onclick={confirmDelete}
			>
				Delete
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
