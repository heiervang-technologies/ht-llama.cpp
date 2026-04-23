<script lang="ts">
	import { onMount } from 'svelte';
	import { Input } from '$lib/components/ui/input';
	import { Button } from '$lib/components/ui/button';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';
	import type { DatabaseArtifactKind } from '$lib/types/database';
	import ArtifactGalleryCard from './ArtifactGalleryCard.svelte';
	import { Images as GalleryIcon, FilterX } from '@lucide/svelte';

	const KIND_FILTERS: { value: DatabaseArtifactKind | 'all'; label: string }[] = [
		{ value: 'all', label: 'All' },
		{ value: 'html', label: 'HTML' },
		{ value: 'svg', label: 'SVG' },
		{ value: 'image', label: 'Images' },
		{ value: 'code', label: 'Code' },
		{ value: 'markdown', label: 'Markdown' },
		{ value: 'audio', label: 'Audio' },
		{ value: 'video', label: 'Video' },
		{ value: 'pdf', label: 'PDF' }
	];

	let activeKind = $state<DatabaseArtifactKind | 'all'>('all');
	let query = $state('');

	onMount(() => {
		artifactGalleryStore.load();
	});

	let items = $derived(artifactGalleryStore.artifacts);

	let filtered = $derived.by(() => {
		const q = query.trim().toLowerCase();
		return items.filter((a) => {
			if (activeKind !== 'all' && a.kind !== activeKind) return false;
			if (!q) return true;
			return (
				a.title.toLowerCase().includes(q) ||
				a.summary?.toLowerCase().includes(q) ||
				a.tags.some((t) => t.toLowerCase().includes(q))
			);
		});
	});

	const sidebar = useSidebar();
</script>

<div
	class="flex h-full flex-col duration-200 ease-linear {sidebar.open
		? 'md:ml-[var(--sidebar-width)]'
		: ''}"
>
	<header
		class="sticky top-0 z-20 flex flex-col gap-3 border-b bg-background/80 p-4 backdrop-blur md:p-6"
	>
		<div class="flex items-center gap-2">
			<GalleryIcon class="h-5 w-5 text-primary" />
			<h1 class="text-lg font-semibold">Artifact gallery</h1>
			<span class="text-sm text-muted-foreground">
				{filtered.length} of {items.length}
			</span>
		</div>

		<div class="flex flex-col gap-2 md:flex-row md:items-center">
			<Input
				type="search"
				bind:value={query}
				placeholder="Search by title, summary, or tag…"
				class="md:max-w-sm"
			/>
			<div class="flex flex-wrap items-center gap-1">
				{#each KIND_FILTERS as f (f.value)}
					<Button
						type="button"
						size="sm"
						variant={activeKind === f.value ? 'default' : 'ghost'}
						class="h-7 rounded-full px-3 text-xs"
						onclick={() => (activeKind = f.value)}
					>
						{f.label}
					</Button>
				{/each}
				{#if activeKind !== 'all' || query}
					<Button
						type="button"
						size="sm"
						variant="ghost"
						class="h-7 rounded-full px-2 text-xs"
						onclick={() => {
							activeKind = 'all';
							query = '';
						}}
					>
						<FilterX class="h-3.5 w-3.5" />
					</Button>
				{/if}
			</div>
		</div>
	</header>

	<section class="flex-1 overflow-y-auto p-4 md:p-6">
		{#if !artifactGalleryStore.loaded && artifactGalleryStore.loading}
			<p class="text-sm text-muted-foreground">Loading artifacts…</p>
		{:else if items.length === 0}
			<div
				class="mx-auto flex max-w-md flex-col items-center gap-3 rounded-xl border border-dashed p-8 text-center"
			>
				<GalleryIcon class="h-10 w-10 text-muted-foreground" />
				<h2 class="text-base font-medium">Nothing saved yet</h2>
				<p class="text-sm text-muted-foreground">
					Artifacts auto-capture from model responses above the size thresholds, and pasted images
					or uploads can be saved manually. They'll show up here with revision history.
				</p>
			</div>
		{:else if filtered.length === 0}
			<p class="text-sm text-muted-foreground">No artifacts match this filter.</p>
		{:else}
			<div class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
				{#each filtered as artifact (artifact.id)}
					<ArtifactGalleryCard {artifact} />
				{/each}
			</div>
		{/if}
	</section>
</div>
