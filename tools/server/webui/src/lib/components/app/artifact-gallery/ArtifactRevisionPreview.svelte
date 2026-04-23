<script lang="ts">
	import { onDestroy } from 'svelte';
	import type { DatabaseArtifact, DatabaseArtifactRevision } from '$lib/types/database';
	import MarkdownContent from '$lib/components/app/content/MarkdownContent.svelte';

	interface Props {
		artifact: DatabaseArtifact;
		revision: DatabaseArtifactRevision;
	}
	let { artifact, revision }: Props = $props();

	let objectUrl = $state<string | null>(null);

	$effect(() => {
		// Recreate the object URL whenever the revision changes so preview
		// playback starts from scratch rather than from a cached Blob URL.
		if (revision.blob) {
			const url = URL.createObjectURL(revision.blob);
			objectUrl = url;
			return () => {
				URL.revokeObjectURL(url);
				objectUrl = null;
			};
		}
		objectUrl = null;
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
			class="h-full w-full object-contain"
			loading="lazy"
		/>
	{:else if artifact.kind === 'video' && objectUrl}
		<!-- svelte-ignore a11y_media_has_caption -->
		<video src={objectUrl} controls class="h-full w-full bg-black" preload="metadata"></video>
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
			class="h-full overflow-auto whitespace-pre-wrap break-words p-4 font-mono text-xs leading-snug">{revision.text}</pre>
	{:else}
		<p class="p-4 text-sm text-muted-foreground">Empty revision.</p>
	{/if}
</div>
