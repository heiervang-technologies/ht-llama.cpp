<script lang="ts">
	import { onDestroy } from 'svelte';
	import { goto } from '$app/navigation';
	import type { DatabaseArtifact } from '$lib/types/database';
	import { DatabaseService } from '$lib/services/database.service';
	import { buildThumb, revokeThumb, type ArtifactThumb } from '$lib/utils/artifact-thumb';
	import { FileCode, FileText, FileImage, FileAudio, FileVideo, FileBadge } from '@lucide/svelte';

	interface Props {
		artifact: DatabaseArtifact;
	}
	let { artifact }: Props = $props();

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
</script>

<button
	type="button"
	class="group flex h-full flex-col overflow-hidden rounded-xl border bg-card text-left shadow-sm transition hover:border-primary hover:shadow-md"
	onclick={() => goto(`/artifacts/${artifact.id}`)}
>
	<div
		class="relative flex h-40 w-full items-center justify-center overflow-hidden bg-muted/50 text-muted-foreground"
	>
		{#if thumb?.kind === 'image'}
			<img src={thumb.url} alt={artifact.title} class="h-full w-full object-cover" />
		{:else if thumb?.kind === 'video'}
			<!-- preload=metadata + no controls: we only want the poster frame here;
				   the detail page ships the full player. -->
			<video src={thumb.url} preload="metadata" muted class="h-full w-full object-cover"></video>
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
			<div class="flex h-full w-full items-center justify-center p-4">{@html thumb.markup}</div>
		{:else if thumb && (thumb.kind === 'code' || thumb.kind === 'html' || thumb.kind === 'markdown')}
			<pre
				class="w-full overflow-hidden whitespace-pre-wrap break-words p-3 font-mono text-[11px] leading-snug">{thumb.excerpt}</pre>
		{:else}
			<Icon class="h-8 w-8" />
		{/if}
	</div>
	<div class="flex flex-1 flex-col gap-1 border-t p-3">
		<div class="flex items-center justify-between gap-2">
			<span class="truncate text-sm font-medium">{artifact.title}</span>
			<span
				class="flex-shrink-0 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] uppercase text-primary"
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
