<script lang="ts">
	import { onDestroy } from 'svelte';
	import type { DatabaseArtifact, DatabaseArtifactRevision } from '$lib/types/database';
	import MarkdownContent from '$lib/components/app/content/MarkdownContent.svelte';
	import { artifactDrag, ARTIFACT_DRAG_MIME } from '$lib/stores/artifact-drag.svelte';

	interface Props {
		artifact: DatabaseArtifact;
		revision: DatabaseArtifactRevision;
	}
	let { artifact, revision }: Props = $props();

	let objectUrl = $state<string | null>(null);

	// Best-effort file name when the user drags the artifact out. Title may
	// contain characters the OS file pipeline dislikes, so strip the worst
	// offenders and append the MIME-derived extension when it's missing.
	function dragFileName(): string {
		const safe = artifact.title.replace(/[\\/:*?"<>|]+/g, '_').trim() || 'artifact';
		const mime = revision.blob?.type || revision.mimeType || '';
		const ext = mime.split('/')[1]?.split(';')[0];
		return ext && !safe.toLowerCase().endsWith('.' + ext.toLowerCase()) ? `${safe}.${ext}` : safe;
	}

	function handleDragStart(event: DragEvent) {
		if (!revision.blob || !event.dataTransfer) return;
		const file = new File([revision.blob], dragFileName(), {
			type: revision.blob.type || revision.mimeType
		});
		artifactDrag.begin(file);
		// Custom MIME signals "this is one of ours" to the drop target so the
		// existing 'Files'-only overlay path also lights up. The file payload
		// itself is handed off through the module-level holder — webkit2gtk's
		// support for dataTransfer.items.add(file) is too patchy to rely on.
		try {
			event.dataTransfer.setData(ARTIFACT_DRAG_MIME, file.name);
		} catch {
			/* setData is allowed to throw in some embedded contexts; harmless */
		}
		event.dataTransfer.effectAllowed = 'copy';
	}

	function handleDragEnd() {
		// If the drop landed somewhere unrelated, clear the holder so a later
		// native file drop doesn't accidentally pick up this stale artifact.
		artifactDrag.end();
	}

	// Defer mounting the <video> element until the user explicitly asks
	// to play. webkit2gtk's GStreamer pipeline can crash the WebProcess
	// during play / metadata probe for several common MP4 codecs, and
	// the failure is fatal-not-recoverable — the whole webview dies.
	// Showing a click-to-play poster keeps the detail page stable; the
	// user opts into playback intentionally and one crash takes the
	// player tab down but not the whole gallery view.
	let videoMounted = $state(false);

	$effect(() => {
		// Recreate the object URL whenever the revision changes so preview
		// playback starts from scratch rather than from a cached Blob URL.
		if (revision.blob) {
			const url = URL.createObjectURL(revision.blob);
			objectUrl = url;
			videoMounted = false; // new revision = fresh click-to-play
			return () => {
				URL.revokeObjectURL(url);
				objectUrl = null;
			};
		}
		objectUrl = null;
		videoMounted = false;
	});

	onDestroy(() => {
		if (objectUrl) URL.revokeObjectURL(objectUrl);
	});
</script>

<div class="flex h-full w-full flex-col overflow-hidden rounded-lg border bg-card">
	{#if artifact.kind === 'image' && objectUrl}
		<img
			src={objectUrl}
			alt={artifact.title}
			class="h-full w-full cursor-grab object-contain active:cursor-grabbing"
			loading="lazy"
			draggable="true"
			ondragstart={handleDragStart}
			ondragend={handleDragEnd}
		/>
	{:else if artifact.kind === 'video' && objectUrl}
		{#if videoMounted}
			<!-- svelte-ignore a11y_media_has_caption -->
			<video src={objectUrl} controls autoplay class="h-full w-full bg-black" preload="metadata"
			></video>
		{:else}
			<button
				type="button"
				onclick={() => (videoMounted = true)}
				class="group relative flex h-full w-full items-center justify-center bg-gradient-to-br from-black via-zinc-900 to-zinc-800 transition-colors"
				aria-label="Play {artifact.title}"
			>
				<div
					class="flex h-16 w-16 items-center justify-center rounded-full bg-white/90 text-black transition-transform group-hover:scale-105"
				>
					<svg viewBox="0 0 24 24" fill="currentColor" class="ml-1 h-8 w-8" aria-hidden="true">
						<path d="M8 5v14l11-7z" />
					</svg>
				</div>
				<a
					href={objectUrl}
					download={artifact.title}
					onclick={(e) => e.stopPropagation()}
					class="absolute right-3 bottom-3 rounded bg-black/60 px-2 py-1 text-xs text-white hover:bg-black/80"
				>
					Download
				</a>
			</button>
		{/if}
	{:else if artifact.kind === 'audio' && objectUrl}
		<div class="flex h-full w-full items-center justify-center p-6">
			<audio src={objectUrl} controls class="w-full max-w-lg"></audio>
		</div>
	{:else if artifact.kind === 'pdf' && objectUrl}
		<!-- Embedding the PDF via <object> keeps the browser's native viewer,
			   which is fine for a local Blob. We render it inline rather than
			   forcing a download. -->
		<object
			data={objectUrl}
			type="application/pdf"
			aria-label={artifact.title}
			class="h-full w-full"
		>
			<div class="p-4 text-sm">
				PDF preview unsupported in this webview.
				<a class="underline" href={objectUrl} target="_blank" rel="noopener">Open PDF</a>
			</div>
		</object>
	{:else if artifact.kind === 'html' && revision.text}
		<!-- Sandbox the HTML so a captured snippet can't touch the host app.
			   allow-scripts only: no same-origin, no top-nav, no forms. -->
		<iframe
			title={artifact.title}
			sandbox="allow-scripts"
			srcdoc={revision.text}
			class="h-full w-full border-0 bg-white"
		></iframe>
	{:else if artifact.kind === 'svg' && revision.text}
		<!-- eslint-disable-next-line svelte/no-at-html-tags -->
		<div class="flex h-full w-full items-center justify-center overflow-auto bg-muted/40 p-4">
			{@html revision.text}
		</div>
	{:else if artifact.kind === 'markdown' && revision.text}
		<div class="h-full overflow-auto p-4">
			<MarkdownContent content={revision.text} />
		</div>
	{:else if revision.text}
		<pre
			class="h-full overflow-auto p-4 font-mono text-xs leading-snug break-words whitespace-pre-wrap">{revision.text}</pre>
	{:else}
		<p class="p-4 text-sm text-muted-foreground">Empty revision.</p>
	{/if}
</div>
