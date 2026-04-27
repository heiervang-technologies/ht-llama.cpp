<script lang="ts">
	import { onMount } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { toast } from 'svelte-sonner';
	import { Input } from '$lib/components/ui/input';
	import { Button } from '$lib/components/ui/button';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';
	import type { DatabaseArtifactKind } from '$lib/types/database';
	import type { CapturePayload } from '$lib/stores/artifact-gallery.svelte';
	import ArtifactGalleryCard from './ArtifactGalleryCard.svelte';
	import NextcloudBrowseDrawer from './NextcloudBrowseDrawer.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import {
		Images as GalleryIcon,
		Cloud,
		FilterX,
		Upload,
		Trash2,
		CheckSquare,
		Square
	} from '@lucide/svelte';

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
	let selectMode = $state(false);
	let selected = new SvelteSet<string>();
	let fileInput: HTMLInputElement | undefined = $state();
	let cloudDrawerOpen = $state(false);

	// "Browse Nextcloud" button only appears once a connection is
	// configured (URL + username minimum). Avoids dangling a button
	// that just throws "not configured" toasts when clicked.
	let nextcloudConfigured = $derived(
		Boolean(String(config().nextcloudUrl ?? '').trim()) &&
			Boolean(String(config().nextcloudUsername ?? '').trim())
	);

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

	function toggleSelectMode() {
		selectMode = !selectMode;
		if (!selectMode) selected.clear();
	}

	function toggleSelection(id: string) {
		if (selected.has(id)) selected.delete(id);
		else selected.add(id);
	}

	function selectAllFiltered() {
		for (const a of filtered) selected.add(a.id);
	}

	function clearSelection() {
		selected.clear();
	}

	async function deleteSelected() {
		const ids = [...selected];
		if (ids.length === 0) return;
		if (!confirm(`Delete ${ids.length} ${ids.length === 1 ? 'artifact' : 'artifacts'}?`)) return;
		await artifactGalleryStore.removeMany(ids);
		selected.clear();
		selectMode = false;
		toast.success(`Deleted ${ids.length} ${ids.length === 1 ? 'artifact' : 'artifacts'}`);
	}

	// Kind detection for uploaded files. Leans on MIME type first, then the
	// filename extension for the cases browsers don't sniff reliably (markdown,
	// svg served as text/plain). Everything unrecognised falls into `code` so
	// nothing is silently rejected.
	function kindFor(file: File): DatabaseArtifactKind {
		const mime = file.type || '';
		const name = file.name.toLowerCase();
		if (mime === 'image/svg+xml' || name.endsWith('.svg')) return 'svg';
		if (mime.startsWith('image/')) return 'image';
		if (mime.startsWith('audio/')) return 'audio';
		if (mime.startsWith('video/')) return 'video';
		if (mime === 'application/pdf' || name.endsWith('.pdf')) return 'pdf';
		if (mime === 'text/html' || name.endsWith('.html') || name.endsWith('.htm')) return 'html';
		if (
			mime === 'text/markdown' ||
			mime === 'text/x-markdown' ||
			name.endsWith('.md') ||
			name.endsWith('.markdown')
		)
			return 'markdown';
		// text/* and application/json and unknown MIMEs all go to `code`;
		// MarkdownContent + the code-excerpt renderer handle any text payload.
		return 'code';
	}

	function isTextKind(kind: DatabaseArtifactKind): boolean {
		return kind === 'html' || kind === 'svg' || kind === 'markdown' || kind === 'code';
	}

	async function handleFiles(files: FileList | File[]) {
		const arr = Array.from(files);
		if (arr.length === 0) return;
		let uploaded = 0;
		let failed = 0;
		for (const file of arr) {
			try {
				const kind = kindFor(file);
				const payload: CapturePayload = {
					kind,
					title: file.name,
					mimeType: file.type || 'application/octet-stream'
				};
				if (isTextKind(kind)) {
					payload.text = await file.text();
				} else {
					payload.blob = file;
				}
				await artifactGalleryStore.saveManual(payload);
				uploaded++;
			} catch (err) {
				failed++;
				console.warn('[artifact-gallery] upload failed', file.name, err);
			}
		}
		if (uploaded)
			toast.success(`Uploaded ${uploaded} ${uploaded === 1 ? 'artifact' : 'artifacts'}`);
		if (failed) toast.error(`${failed} upload${failed === 1 ? '' : 's'} failed — see console`);
	}

	function pickFiles() {
		fileInput?.click();
	}

	function onFileInput(ev: Event) {
		const input = ev.currentTarget as HTMLInputElement;
		if (input.files && input.files.length > 0) {
			void handleFiles(input.files);
			input.value = '';
		}
	}

	// Native drag-and-drop onto the gallery — skip parent handlers because the
	// chat scroll container's overlay isn't mounted here. The <main> in the
	// gallery has its own dropzone below.
	let dragDepth = $state(0);
	let isDragOver = $derived(dragDepth > 0);

	function onDragEnter(ev: DragEvent) {
		if (!ev.dataTransfer?.types.includes('Files')) return;
		ev.preventDefault();
		dragDepth++;
	}
	function onDragOver(ev: DragEvent) {
		if (ev.dataTransfer?.types.includes('Files')) ev.preventDefault();
	}
	function onDragLeave(ev: DragEvent) {
		ev.preventDefault();
		dragDepth = Math.max(0, dragDepth - 1);
	}
	function onDrop(ev: DragEvent) {
		if (!ev.dataTransfer?.files?.length) return;
		ev.preventDefault();
		dragDepth = 0;
		void handleFiles(ev.dataTransfer.files);
	}

	const sidebar = useSidebar();
</script>

<div
	class="flex h-full flex-col duration-200 ease-linear {sidebar.open
		? 'md:ml-[var(--sidebar-width)]'
		: ''}"
	ondragenter={onDragEnter}
	ondragover={onDragOver}
	ondragleave={onDragLeave}
	ondrop={onDrop}
	role="region"
	aria-label="Artifact gallery"
>
	<input bind:this={fileInput} type="file" multiple class="hidden" onchange={onFileInput} />

	<header
		class="sticky top-0 z-20 flex flex-col gap-3 border-b bg-background/80 p-4 backdrop-blur md:p-6"
	>
		<div class="flex items-center gap-2">
			<GalleryIcon class="h-5 w-5 text-primary" />
			<h1 class="text-lg font-semibold">Artifact gallery</h1>
			<span class="text-sm text-muted-foreground">
				{filtered.length} of {items.length}
			</span>
			<div class="ml-auto flex items-center gap-1">
				{#if nextcloudConfigured}
					<Button
						type="button"
						size="sm"
						variant="outline"
						onclick={() => (cloudDrawerOpen = true)}
						title="Browse Nextcloud"
					>
						<Cloud class="h-4 w-4" />
						<span class="hidden sm:inline">Cloud</span>
					</Button>
				{/if}
				<Button type="button" size="sm" variant="outline" onclick={pickFiles}>
					<Upload class="h-4 w-4" />
					<span class="hidden sm:inline">Upload</span>
				</Button>
				<Button
					type="button"
					size="sm"
					variant={selectMode ? 'default' : 'outline'}
					onclick={toggleSelectMode}
					title={selectMode ? 'Exit selection' : 'Select multiple'}
				>
					{#if selectMode}
						<CheckSquare class="h-4 w-4" />
					{:else}
						<Square class="h-4 w-4" />
					{/if}
					<span class="hidden sm:inline">Select</span>
				</Button>
			</div>
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

		{#if selectMode}
			<div
				class="flex items-center gap-2 rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-sm"
			>
				<span class="font-medium">
					{selected.size} selected
				</span>
				<Button type="button" size="sm" variant="ghost" onclick={selectAllFiltered}>
					Select all ({filtered.length})
				</Button>
				<Button type="button" size="sm" variant="ghost" onclick={clearSelection}>Clear</Button>
				<div class="ml-auto">
					<Button
						type="button"
						size="sm"
						variant="destructive"
						disabled={selected.size === 0}
						onclick={deleteSelected}
					>
						<Trash2 class="h-4 w-4" />
						Delete
					</Button>
				</div>
			</div>
		{/if}
	</header>

	<section class="relative flex-1 overflow-y-auto p-4 md:p-6">
		{#if isDragOver}
			<div
				class="pointer-events-none absolute inset-0 z-30 flex items-center justify-center rounded-lg border-2 border-dashed border-primary bg-primary/5"
			>
				<div class="flex flex-col items-center gap-2 text-primary">
					<Upload class="h-8 w-8" />
					<p class="text-sm font-medium">Drop files to add to the gallery</p>
				</div>
			</div>
		{/if}

		{#if !artifactGalleryStore.loaded && artifactGalleryStore.loading}
			<p class="text-sm text-muted-foreground">Loading artifacts…</p>
		{:else if items.length === 0}
			<div
				class="mx-auto flex max-w-md flex-col items-center gap-3 rounded-xl border border-dashed p-8 text-center"
			>
				<GalleryIcon class="h-10 w-10 text-muted-foreground" />
				<h2 class="text-base font-medium">Nothing saved yet</h2>
				<p class="text-sm text-muted-foreground">
					Artifacts auto-capture from model responses above the size thresholds. You can also drop
					files here or click <strong>Upload</strong> to add them manually.
				</p>
			</div>
		{:else if filtered.length === 0}
			<p class="text-sm text-muted-foreground">No artifacts match this filter.</p>
		{:else}
			<div class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
				{#each filtered as artifact (artifact.id)}
					<ArtifactGalleryCard
						{artifact}
						selectable={selectMode}
						selected={selected.has(artifact.id)}
						onToggleSelect={() => toggleSelection(artifact.id)}
					/>
				{/each}
			</div>
		{/if}
	</section>
</div>

<NextcloudBrowseDrawer open={cloudDrawerOpen} onOpenChange={(next) => (cloudDrawerOpen = next)} />
