<script lang="ts">
	/**
	 * Inline previews for artifacts referenced in a tool's JSON result.
	 *
	 * Tools like generate_image / edit_image / generate_video save the
	 * payload to the gallery and return only references — the JSON the
	 * model sees stays small (no base64 in the LLM context). Without
	 * an extra render path, the user would only see the JSON in the
	 * tool-call card and would have to navigate to /artifacts to view
	 * the actual image. This component closes that loop: it scans the
	 * tool result for artifact references and renders them as a small
	 * preview grid right inside the tool-call block.
	 *
	 * Recognised shapes — both `{ images: [...] }` and `{ videos: [...] }`
	 * arrays of `{ artifactId, revisionId, mimeType }`. Anything else is
	 * silently skipped (this is a non-essential UX nicety, not a hard
	 * contract — tool authors are not required to produce previews).
	 */
	import { SvelteMap } from 'svelte/reactivity';
	import { DatabaseService } from '$lib/services/database.service';

	interface Props {
		toolResult: string | undefined;
	}

	let { toolResult }: Props = $props();

	type Ref = { artifactId: string; revisionId: string; mimeType?: string; title?: string };

	let refs = $derived.by<Ref[]>(() => {
		if (!toolResult) return [];
		try {
			const parsed = JSON.parse(toolResult);
			if (!parsed || typeof parsed !== 'object') return [];
			const out: Ref[] = [];
			for (const key of ['images', 'videos', 'audio']) {
				const arr = (parsed as Record<string, unknown>)[key];
				if (Array.isArray(arr)) {
					for (const item of arr) {
						if (
							item &&
							typeof item === 'object' &&
							typeof (item as Record<string, unknown>).artifactId === 'string' &&
							typeof (item as Record<string, unknown>).revisionId === 'string'
						) {
							const r = item as Record<string, unknown>;
							out.push({
								artifactId: String(r.artifactId),
								revisionId: String(r.revisionId),
								mimeType: typeof r.mimeType === 'string' ? r.mimeType : undefined,
								title: typeof r.title === 'string' ? r.title : undefined
							});
						}
					}
				}
			}
			return out;
		} catch {
			return [];
		}
	});

	const cache = new SvelteMap<string, string>();

	async function loadRef(revisionId: string): Promise<string | null> {
		const cached = cache.get(revisionId);
		if (cached) return cached;
		const revision = await DatabaseService.getArtifactRevision(revisionId);
		if (!revision?.blob) return null;
		const dataUrl = await blobToDataUrl(revision.blob);
		cache.set(revisionId, dataUrl);
		return dataUrl;
	}

	function blobToDataUrl(blob: Blob): Promise<string> {
		return new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.onloadend = () => {
				if (typeof reader.result === 'string') resolve(reader.result);
				else reject(new Error('FileReader did not return a string'));
			};
			reader.onerror = () => reject(reader.error ?? new Error('FileReader failed'));
			reader.readAsDataURL(blob);
		});
	}
</script>

{#if refs.length > 0}
	<div class="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
		{#each refs as ref (ref.artifactId)}
			{#await loadRef(ref.revisionId) then dataUrl}
				{#if dataUrl}
					<a
						href="#/artifacts/{ref.artifactId}"
						class="block overflow-hidden rounded-md border bg-background transition-shadow hover:shadow-sm"
						title={ref.title ?? ref.artifactId}
					>
						{#if ref.mimeType?.startsWith('video/')}
							<!-- svelte-ignore a11y_media_has_caption -->
							<video src={dataUrl} controls class="h-auto w-full"></video>
						{:else if ref.mimeType?.startsWith('audio/')}
							<audio src={dataUrl} controls class="w-full"></audio>
						{:else}
							<img src={dataUrl} alt={ref.title ?? 'Generated artifact'} class="h-auto w-full" />
						{/if}
					</a>
				{/if}
			{/await}
		{/each}
	</div>
{/if}
