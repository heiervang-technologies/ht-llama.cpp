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
	import { onDestroy } from 'svelte';
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';
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
			// Recursive scan for any `{ artifactId, revisionId }` shape
			// anywhere in the tree. The agentic path's `generate_image`
			// returns `{ images: [...] }` at the top level, but
			// edit_image, generate_video, future tool wrappers, and
			// MCP servers that re-wrap our result might nest the refs
			// differently (e.g. `{ result: { images: [...] } }` or a
			// raw array). Walking the tree once means we don't have to
			// keep this list in sync with every producer.
			const out: Ref[] = [];
			// Plain dedup set — not a Svelte reactivity surface, just a
			// scratch array-of-keys swap to avoid the prefer-svelte-
			// reactivity lint flagging an inert Set used inside a
			// $derived computation.
			const seenKeys: string[] = [];
			const has = (k: string) => seenKeys.includes(k);
			const visit = (node: unknown, depth = 0) => {
				if (depth > 8) return; // cycle / runaway guard
				if (!node || typeof node !== 'object') return;
				if (Array.isArray(node)) {
					for (const child of node) visit(child, depth + 1);
					return;
				}
				const obj = node as Record<string, unknown>;
				if (typeof obj.artifactId === 'string' && typeof obj.revisionId === 'string') {
					const key = `${obj.artifactId}::${obj.revisionId}`;
					if (!has(key)) {
						seenKeys.push(key);
						out.push({
							artifactId: String(obj.artifactId),
							revisionId: String(obj.revisionId),
							mimeType: typeof obj.mimeType === 'string' ? obj.mimeType : undefined,
							title: typeof obj.title === 'string' ? obj.title : undefined
						});
					}
					// Don't recurse into a matched ref — its own fields
					// are flat strings, walking them is just overhead.
					return;
				}
				for (const value of Object.values(obj)) visit(value, depth + 1);
			};
			visit(parsed);
			return out;
		} catch {
			return [];
		}
	});

	const cache = new SvelteMap<string, string>();
	const objectUrls = new SvelteSet<string>();

	onDestroy(() => {
		for (const url of objectUrls) URL.revokeObjectURL(url);
		objectUrls.clear();
	});

	async function loadRef(revisionId: string): Promise<string | null> {
		const cached = cache.get(revisionId);
		if (cached) return cached;
		const revision = await DatabaseService.getArtifactRevision(revisionId);
		if (!revision?.blob) return null;
		const dataUrl = URL.createObjectURL(revision.blob);
		objectUrls.add(dataUrl);
		cache.set(revisionId, dataUrl);
		return dataUrl;
	}
</script>

{#if refs.length > 0}
	<div class="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
		{#each refs as ref (ref.artifactId)}
			{#await loadRef(ref.revisionId) then dataUrl}
				{#if dataUrl}
					<!--
						NOTE: media (video / audio with controls) cannot be wrapped
						in an <a> per the HTML spec — that nests interactive content
						inside an anchor and triggers browser console warnings. We
						use an <a> for the static <img> case and a plain container
						with a separate "Open" link for media.
					-->
					{#if ref.mimeType?.startsWith('video/')}
						<div
							class="overflow-hidden rounded-md border bg-background"
							title={ref.title ?? ref.artifactId}
						>
							<!-- svelte-ignore a11y_media_has_caption -->
							<video src={dataUrl} preload="none" controls class="h-auto w-full"></video>
							<a
								href="#/artifacts/{ref.artifactId}"
								class="block px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
							>
								Open in gallery →
							</a>
						</div>
					{:else if ref.mimeType?.startsWith('audio/')}
						<div
							class="overflow-hidden rounded-md border bg-background p-2"
							title={ref.title ?? ref.artifactId}
						>
							<audio src={dataUrl} controls class="w-full"></audio>
							<a
								href="#/artifacts/{ref.artifactId}"
								class="mt-1 block text-xs text-muted-foreground hover:text-foreground"
							>
								Open in gallery →
							</a>
						</div>
					{:else}
						<a
							href="#/artifacts/{ref.artifactId}"
							class="block overflow-hidden rounded-md border bg-background transition-shadow hover:shadow-sm"
							title={ref.title ?? ref.artifactId}
						>
							<img src={dataUrl} alt={ref.title ?? 'Generated artifact'} class="h-auto w-full" />
						</a>
					{/if}
				{/if}
			{/await}
		{/each}
	</div>
{/if}
