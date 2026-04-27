<script lang="ts">
	/**
	 * Standalone image-generation playground.
	 *
	 * Mirrors the shape of HF Spaces / Krea / A1111: a dedicated surface
	 * with no chat thread, where the input is a prompt + a few knobs and
	 * the output appears immediately on the same page. Three-column
	 * desktop layout (input · canvas · history); on mobile the rails
	 * collapse into accordions above the canvas.
	 *
	 * Two modes share the same surface — Generate (text-to-image, calls
	 * `runImageGeneration`) and Edit (image-to-image, calls `runImageEdit`).
	 * The mode toggle is a top strip; the only column that changes
	 * between modes is the input rail. The canvas and history rail stay
	 * visually identical so the user is never disoriented.
	 *
	 * All four image-gen entry points (tool, slash, composer-toggle,
	 * playground) write into the same gallery — `metadata.source` plus
	 * the `playground` tag is what distinguishes them downstream.
	 */
	import { onMount } from 'svelte';
	import { SvelteMap } from 'svelte/reactivity';
	import { toast } from 'svelte-sonner';
	import {
		Image as ImageIcon,
		Pencil,
		PlayCircle,
		Loader2,
		Settings as SettingsIcon,
		History,
		Link as LinkIcon,
		Unlink as UnlinkIcon,
		X
	} from '@lucide/svelte';

	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Switch } from '$lib/components/ui/switch';
	import { Textarea } from '$lib/components/ui/textarea';
	import * as Select from '$lib/components/ui/select';

	import { runImageGeneration, runImageEdit } from '$lib/services/builtin-tools';
	import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';
	import { imagePlaygroundStore } from '$lib/stores/image-playground.svelte';
	import { DatabaseService } from '$lib/services/database.service';
	import { config } from '$lib/stores/settings.svelte';
	import { getChatSettingsDialogContext } from '$lib/contexts';
	import { SETTINGS_SECTION_TITLES } from '$lib/constants';
	import type { DatabaseArtifact } from '$lib/types/database';

	type Mode = 'generate' | 'edit';

	const GENERATE_MODELS = [
		{ id: 'z-image-turbo', label: 'z-image-turbo · ~52s · default' },
		{ id: 'newbie-image', label: 'newbie-image · ~22s · anime / manga' },
		{ id: 'qwen-image', label: 'qwen-image · ~10 min · slow but high quality' },
		{ id: 'flux2-klein', label: 'flux2-klein · broken · OOM on 24 GB' }
	];

	const EDIT_MODELS = [{ id: 'qwen-image-edit', label: 'qwen-image-edit · ~2.5 min @ 1024' }];

	const ASPECT_PRESETS: Array<{ id: string; label: string; w: number; h: number }> = [
		{ id: '1:1', label: '1 : 1 · square', w: 1024, h: 1024 },
		{ id: '4:3', label: '4 : 3', w: 1152, h: 864 },
		{ id: '3:2', label: '3 : 2', w: 1216, h: 832 },
		{ id: '16:9', label: '16 : 9 · widescreen', w: 1344, h: 768 },
		{ id: '2:3', label: '2 : 3', w: 832, h: 1216 },
		{ id: '3:4', label: '3 : 4 · portrait', w: 864, h: 1152 },
		{ id: '9:16', label: '9 : 16 · vertical', w: 768, h: 1344 }
	];

	const SIZE_STEP = 64; // ComfyUI VAEs prefer multiples of 64

	let mode = $state<Mode>('generate');
	let prompt = $state('');
	let model = $state(GENERATE_MODELS[0].id);
	let width = $state(1024);
	let height = $state(1024);
	// When the chain is locked, edits to width/height preserve the
	// aspect ratio captured at the moment the lock was clicked. We
	// hold the ratio as a frozen number rather than recomputing on
	// every edit so a tiny rounding drift can't slowly distort it.
	let ratioLocked = $state(false);
	let lockedRatio = $state(1);
	let nVariants = $state(1);

	// Advanced toggle. Most users only ever touch prompt + size + model;
	// power users want negative prompt and a fixed seed for reproducibility.
	// Default off keeps the surface uncluttered.
	let showAdvanced = $state(false);
	let negativePrompt = $state('');
	let seed = $state<number | null>(null);

	let editSourceDataUrl = $state<string | null>(null);
	let editSourceArtifactId = $state<string | null>(null);
	let editFileInputRef: HTMLInputElement | null = $state(null);

	// Run state lives in the imagePlaygroundStore (module-level), not
	// here, so a generation in flight survives navigating away from
	// /images and back. Local derived booleans below for ergonomics.
	let activeRun = $derived(imagePlaygroundStore.active);
	let isRunning = $derived(activeRun !== null);
	let lastResult = $derived(imagePlaygroundStore.lastFinished);

	// Elapsed-time ticker for the running banner. 1 Hz is fine; the
	// run is GPU-bound (20 s — 10 min) so sub-second precision is
	// noise. Interval is set up + torn down only while a run is
	// active so we don't burn cycles in the idle case.
	let elapsedMs = $state(0);
	$effect(() => {
		if (!activeRun) {
			elapsedMs = 0;
			return;
		}
		const startedAt = activeRun.startedAt;
		elapsedMs = Date.now() - startedAt;
		const id = setInterval(() => {
			elapsedMs = Date.now() - startedAt;
		}, 1000);
		return () => clearInterval(id);
	});

	function formatElapsed(ms: number): string {
		const total = Math.max(0, Math.floor(ms / 1000));
		const m = Math.floor(total / 60);
		const s = total % 60;
		return m > 0 ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`;
	}

	let imageGenEnabled = $derived(Boolean(config().imageGenEnabled));
	let imagesBaseUrl = $derived(String(config().imagesBaseUrl ?? '').trim());

	// Reactive history: every image artifact in the gallery, newest first.
	// Includes output from all four entry points; metadata.source is what
	// distinguishes them, but for browsing we want everything in one rail.
	let history = $derived.by<DatabaseArtifact[]>(() =>
		[...artifactGalleryStore.artifacts]
			.filter((a) => a.kind === 'image')
			.sort((a, b) => b.updatedAt - a.updatedAt)
	);

	// Resolved data-URL cache keyed by revisionId, for the history rail
	// thumbnails. Loaded lazily as artifacts come into view.
	let thumbnailCache = new SvelteMap<string, string>();

	onMount(async () => {
		// Refresh the gallery so the history rail is populated even if
		// the user lands here directly without visiting /artifacts.
		await artifactGalleryStore.load();
	});

	$effect(() => {
		// Switching modes resets the model selection so we never end up
		// with a Generate-only model id selected in Edit mode (or vice
		// versa). The size knob is shared.
		const allowed = mode === 'generate' ? GENERATE_MODELS : EDIT_MODELS;
		if (!allowed.some((m) => m.id === model)) {
			model = allowed[0].id;
		}
	});

	const chatSettingsDialog = getChatSettingsDialogContext();

	function openImagesSettings() {
		chatSettingsDialog.open(SETTINGS_SECTION_TITLES.IMAGES);
	}

	function snapToStep(value: number): number {
		// VAE alignment — round to nearest multiple of SIZE_STEP and clamp
		// into a sane range. Below 256 nothing trains well; above 2048 the
		// proxy starts OOMing on 24 GB.
		const stepped = Math.round(value / SIZE_STEP) * SIZE_STEP;
		return Math.max(256, Math.min(2048, stepped));
	}

	function applyPreset(w: number, h: number) {
		ratioLocked = false; // preset overrides any locked ratio
		width = w;
		height = h;
	}

	function toggleLock() {
		if (ratioLocked) {
			ratioLocked = false;
		} else {
			lockedRatio = width / height;
			ratioLocked = true;
		}
	}

	function handleWidthChange(value: number) {
		const next = snapToStep(value);
		width = next;
		if (ratioLocked) {
			height = snapToStep(next / lockedRatio);
		}
	}

	function handleHeightChange(value: number) {
		const next = snapToStep(value);
		height = next;
		if (ratioLocked) {
			width = snapToStep(next * lockedRatio);
		}
	}

	async function loadThumbnail(revisionId: string): Promise<string | null> {
		const cached = thumbnailCache.get(revisionId);
		if (cached) return cached;
		const revision = await DatabaseService.getArtifactRevision(revisionId);
		if (!revision?.blob) return null;
		const dataUrl = await blobToDataUrl(revision.blob);
		thumbnailCache.set(revisionId, dataUrl);
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

	function pickEditSource() {
		editFileInputRef?.click();
	}

	async function handleEditFileChange(event: Event) {
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;
		try {
			const dataUrl = await blobToDataUrl(file);
			editSourceDataUrl = dataUrl;
			editSourceArtifactId = null;
		} catch (e) {
			toast.error(`Failed to read file: ${e instanceof Error ? e.message : String(e)}`);
		} finally {
			input.value = '';
		}
	}

	async function pickFromHistory(artifact: DatabaseArtifact) {
		// Clicking a history card with no active result loads it in the
		// canvas. In Edit mode it ALSO becomes the source image so you
		// can iterate on it without a re-upload.
		const revision = await DatabaseService.getArtifactRevision(artifact.currentRevisionId);
		if (!revision?.blob) {
			toast.error('Could not load that artifact.');
			return;
		}
		const dataUrl = await blobToDataUrl(revision.blob);
		if (mode === 'edit') {
			editSourceDataUrl = dataUrl;
			editSourceArtifactId = artifact.id;
		}
		// Don't trample an in-flight run's spinner state. The user can
		// still pick a source for the *next* edit while a run is going
		// — the canvas just keeps showing the running indicator.
		if (imagePlaygroundStore.active) return;
		// Show in the canvas as a "viewing existing" preview by writing
		// to the playground store, same surface a real run uses.
		imagePlaygroundStore.finishRun({
			mode: 'generate',
			result: {
				model: String(artifact.tags.find((t) => t !== 'generated' && t !== 'playground') ?? '—'),
				size: null,
				prompt: artifact.title,
				images: [
					{
						artifactId: artifact.id,
						revisionId: artifact.currentRevisionId,
						title: artifact.title,
						mimeType: revision.mimeType ?? 'image/png'
					}
				]
			},
			dataUrls: [dataUrl],
			finishedAt: Date.now()
		});
	}

	function clearEditSource() {
		editSourceDataUrl = null;
		editSourceArtifactId = null;
	}

	async function run() {
		if (isRunning) return;
		const trimmedPrompt = prompt.trim();
		if (!trimmedPrompt) {
			toast.info('Enter a prompt first.');
			return;
		}
		if (!imageGenEnabled) {
			toast.error('Image generation is disabled. Enable it in Settings → Images.');
			openImagesSettings();
			return;
		}
		if (!imagesBaseUrl) {
			toast.error('Set Settings → Images → Base URL first.');
			openImagesSettings();
			return;
		}
		if (mode === 'edit' && !editSourceDataUrl) {
			toast.info('Drop a source image first, or click a history card to reuse one.');
			return;
		}

		const controller = new AbortController();
		const startedMode: Mode = mode;
		const startedModel = model;
		imagePlaygroundStore.beginRun({
			mode: startedMode,
			prompt: trimmedPrompt,
			model: startedModel,
			startedAt: Date.now(),
			abort: () => controller.abort()
		});

		const size = `${width}x${height}`;
		const trimmedNeg = negativePrompt.trim();
		const advancedExtras: { negativePrompt?: string; seed?: number } = {};
		if (showAdvanced && trimmedNeg) advancedExtras.negativePrompt = trimmedNeg;
		if (showAdvanced && typeof seed === 'number' && seed >= 0) advancedExtras.seed = seed;
		try {
			if (startedMode === 'generate') {
				const result = await runImageGeneration({
					source: 'playground',
					prompt: trimmedPrompt,
					model: startedModel,
					size,
					n: nVariants,
					...advancedExtras,
					signal: controller.signal
				});
				const dataUrls = await Promise.all(
					result.images.map(async (img) => {
						const rev = await DatabaseService.getArtifactRevision(img.revisionId);
						return rev?.blob ? blobToDataUrl(rev.blob) : Promise.resolve('');
					})
				);
				imagePlaygroundStore.finishRun({
					mode: 'generate',
					result,
					dataUrls,
					finishedAt: Date.now()
				});
			} else {
				const result = await runImageEdit({
					source: 'playground',
					prompt: trimmedPrompt,
					image: editSourceDataUrl as string,
					model: startedModel,
					size,
					n: nVariants,
					sourceArtifactId: editSourceArtifactId,
					...advancedExtras,
					signal: controller.signal
				});
				const dataUrls = await Promise.all(
					result.images.map(async (img) => {
						const rev = await DatabaseService.getArtifactRevision(img.revisionId);
						return rev?.blob ? blobToDataUrl(rev.blob) : Promise.resolve('');
					})
				);
				imagePlaygroundStore.finishRun({
					mode: 'edit',
					result,
					dataUrls,
					finishedAt: Date.now()
				});
			}
		} catch (e) {
			const message = e instanceof Error ? e.message : String(e);
			imagePlaygroundStore.failRun();
			if (controller.signal.aborted) {
				toast.info('Cancelled.');
			} else {
				toast.error(message);
			}
		}
	}

	function cancelRun() {
		imagePlaygroundStore.cancel();
	}
</script>

<div class="flex h-full w-full flex-col overflow-hidden">
	<!-- Top strip: mode toggle + page title -->
	<div class="flex flex-shrink-0 items-center gap-3 border-b px-4 py-3">
		<h1 class="text-lg font-semibold">Images</h1>

		<div class="ml-2 inline-flex rounded-full border bg-muted/30 p-0.5">
			<button
				type="button"
				class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors {mode ===
				'generate'
					? 'bg-primary text-primary-foreground'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (mode = 'generate')}
			>
				<ImageIcon class="h-3.5 w-3.5" />
				Generate
			</button>
			<button
				type="button"
				class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors {mode ===
				'edit'
					? 'bg-primary text-primary-foreground'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (mode = 'edit')}
			>
				<Pencil class="h-3.5 w-3.5" />
				Edit
			</button>
		</div>

		{#if !imageGenEnabled}
			<span class="ml-2 text-xs text-amber-600">Image generation is disabled.</span>
		{/if}

		<div class="ml-auto flex items-center gap-2">
			<Button variant="ghost" size="sm" onclick={openImagesSettings} class="gap-1.5">
				<SettingsIcon class="h-3.5 w-3.5" />
				Settings
			</Button>
		</div>
	</div>

	<!-- Body: 3-column on desktop, stacked on mobile -->
	<div
		class="grid min-h-0 flex-1 grid-cols-1 gap-0 overflow-hidden md:grid-cols-[20rem_minmax(0,1fr)_18rem]"
	>
		<!-- LEFT input rail -->
		<aside class="flex min-h-0 flex-col overflow-y-auto border-r bg-muted/10 p-4 md:bg-muted/20">
			<div class="flex flex-col gap-4">
				<div class="flex flex-col gap-1.5">
					<Label for="img-prompt" class="text-xs text-muted-foreground uppercase">Prompt</Label>
					<Textarea
						id="img-prompt"
						bind:value={prompt}
						placeholder={mode === 'generate'
							? 'A photo of a calico cat sitting on a sunlit windowsill…'
							: 'Make the sky a deep purple at sunset, keep everything else.'}
						rows={6}
						disabled={isRunning}
						class="resize-y"
					/>
				</div>

				{#if mode === 'edit'}
					<div class="flex flex-col gap-1.5">
						<Label class="text-xs text-muted-foreground uppercase">Source image</Label>
						{#if editSourceDataUrl}
							<div class="relative overflow-hidden rounded-md border bg-background">
								<img src={editSourceDataUrl} alt="Source" class="h-40 w-full object-cover" />
								<button
									type="button"
									onclick={clearEditSource}
									class="absolute top-1 right-1 rounded-full bg-background/80 p-1 hover:bg-background"
									aria-label="Clear source"
								>
									<X class="h-3 w-3" />
								</button>
							</div>
						{:else}
							<button
								type="button"
								onclick={pickEditSource}
								class="flex h-32 flex-col items-center justify-center rounded-md border border-dashed text-xs text-muted-foreground transition-colors hover:border-primary hover:text-foreground"
							>
								<ImageIcon class="mb-1 h-5 w-5 opacity-60" />
								Click to upload, or pick from history →
							</button>
						{/if}
						<input
							type="file"
							accept="image/*"
							class="hidden"
							bind:this={editFileInputRef}
							onchange={handleEditFileChange}
						/>
					</div>
				{/if}

				<div class="flex flex-col gap-1.5">
					<Label for="img-model" class="text-xs text-muted-foreground uppercase">Model</Label>
					<Select.Root type="single" bind:value={model} disabled={isRunning}>
						<Select.Trigger id="img-model" class="w-full">
							{(mode === 'generate' ? GENERATE_MODELS : EDIT_MODELS).find((m) => m.id === model)
								?.label ?? model}
						</Select.Trigger>
						<Select.Content>
							{#each mode === 'generate' ? GENERATE_MODELS : EDIT_MODELS as opt (opt.id)}
								<Select.Item value={opt.id} label={opt.label}>{opt.label}</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
				</div>

				<div class="flex flex-col gap-1.5">
					<Label class="text-xs text-muted-foreground uppercase">Size</Label>

					<!-- Width / lock / Height — same shape as Photoshop, Figma,
					     A1111. The lock toggle freezes the current ratio so
					     either input nudges the other proportionally. Snap to
					     SIZE_STEP (64) on every change to keep the VAEs happy. -->
					<div class="flex items-center gap-1.5">
						<Input
							id="img-width"
							type="number"
							min={256}
							max={2048}
							step={SIZE_STEP}
							value={width}
							oninput={(e) => handleWidthChange(Number((e.target as HTMLInputElement).value))}
							disabled={isRunning}
							class="w-full"
							aria-label="Width in pixels"
						/>
						<button
							type="button"
							onclick={toggleLock}
							disabled={isRunning}
							class="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-md border bg-background text-muted-foreground transition-colors hover:text-foreground disabled:opacity-40 {ratioLocked
								? 'border-primary text-primary'
								: ''}"
							title={ratioLocked
								? `Aspect ratio locked at ${lockedRatio.toFixed(2)} : 1`
								: 'Click to lock aspect ratio'}
							aria-pressed={ratioLocked}
							aria-label="Lock aspect ratio"
						>
							{#if ratioLocked}
								<LinkIcon class="h-4 w-4" />
							{:else}
								<UnlinkIcon class="h-4 w-4" />
							{/if}
						</button>
						<Input
							id="img-height"
							type="number"
							min={256}
							max={2048}
							step={SIZE_STEP}
							value={height}
							oninput={(e) => handleHeightChange(Number((e.target as HTMLInputElement).value))}
							disabled={isRunning}
							class="w-full"
							aria-label="Height in pixels"
						/>
					</div>

					<!-- Quick presets — clicking applies w / h and unlocks the
					     ratio so the next manual edit isn't constrained. -->
					<div class="mt-1 flex flex-wrap gap-1">
						{#each ASPECT_PRESETS as preset (preset.id)}
							<button
								type="button"
								onclick={() => applyPreset(preset.w, preset.h)}
								disabled={isRunning}
								class="rounded-full border px-2 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-40 {width ===
									preset.w && height === preset.h
									? 'border-primary text-primary'
									: ''}"
								title={preset.label}
							>
								{preset.id}
							</button>
						{/each}
					</div>
				</div>

				<div class="flex flex-col gap-1.5">
					<Label for="img-n" class="text-xs text-muted-foreground uppercase">Variants</Label>
					<Input
						id="img-n"
						type="number"
						min="1"
						max="4"
						bind:value={nVariants}
						disabled={isRunning}
						class="w-full"
					/>
				</div>

				<!-- Advanced settings toggle. Hides power-user knobs (negative
				     prompt, seed) behind a switch so first-time users see a
				     clean Prompt + Size + Variants surface. State doesn't
				     persist between sessions on purpose — most runs are
				     ad-hoc, not project-scoped. -->
				<div class="flex items-center justify-between border-t pt-3">
					<Label for="img-advanced" class="text-xs text-muted-foreground uppercase">Advanced</Label>
					<Switch id="img-advanced" bind:checked={showAdvanced} disabled={isRunning} />
				</div>

				{#if showAdvanced}
					<div class="flex flex-col gap-1.5">
						<Label for="img-negative" class="text-xs text-muted-foreground uppercase">
							Negative prompt
						</Label>
						<Textarea
							id="img-negative"
							bind:value={negativePrompt}
							placeholder="What you do NOT want — e.g. blurry, low quality, extra fingers"
							rows={3}
							disabled={isRunning}
							class="resize-y"
						/>
						<p class="text-[10px] text-muted-foreground">
							Models that don't read negative prompts ignore this silently.
						</p>
					</div>

					<div class="flex flex-col gap-1.5">
						<Label for="img-seed" class="text-xs text-muted-foreground uppercase">Seed</Label>
						<div class="flex gap-1.5">
							<Input
								id="img-seed"
								type="number"
								min={0}
								placeholder="random"
								value={seed ?? ''}
								oninput={(e) => {
									const v = (e.target as HTMLInputElement).value;
									seed = v.trim() === '' ? null : Number(v);
								}}
								disabled={isRunning}
								class="w-full"
							/>
							<Button
								type="button"
								variant="outline"
								size="sm"
								onclick={() => (seed = Math.floor(Math.random() * 2 ** 31))}
								disabled={isRunning}
								class="flex-shrink-0"
								title="Roll a new seed"
							>
								🎲
							</Button>
							<Button
								type="button"
								variant="ghost"
								size="sm"
								onclick={() => (seed = null)}
								disabled={isRunning || seed === null}
								class="flex-shrink-0"
								title="Clear seed (random per run)"
							>
								<X class="h-3 w-3" />
							</Button>
						</div>
					</div>
				{/if}

				<div class="flex flex-col gap-2">
					{#if isRunning}
						<Button variant="destructive" onclick={cancelRun} class="w-full gap-2">
							<X class="h-4 w-4" />
							Cancel
						</Button>
						<p class="text-center text-xs text-muted-foreground">
							{mode === 'generate' ? 'Generating…' : 'Editing…'} this can take 20 seconds to a few minutes
							depending on the model.
						</p>
					{:else}
						<Button onclick={run} class="w-full gap-2" disabled={!imageGenEnabled}>
							<PlayCircle class="h-4 w-4" />
							{mode === 'generate' ? 'Generate' : 'Edit'}
						</Button>
					{/if}
				</div>
			</div>
		</aside>

		<!-- CENTER canvas -->
		<main class="flex min-h-0 flex-col overflow-y-auto p-4">
			{#if activeRun}
				<!-- Running banner takes priority over a stale lastResult so
				     the user sees real-time state on re-entry to /images.
				     Survives navigation: state lives in the playground store. -->
				<div
					class="flex h-full min-h-64 flex-col items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground"
				>
					<Loader2 class="mb-2 h-6 w-6 animate-spin" />
					<p class="text-foreground">
						{activeRun.mode === 'generate' ? 'Generating' : 'Editing'} · {activeRun.model}
					</p>
					<p class="mt-1 max-w-md text-center text-xs opacity-70">
						{activeRun.prompt}
					</p>
					<p class="mt-2 font-mono text-xs opacity-60">
						{formatElapsed(elapsedMs)}
					</p>
				</div>
			{:else if !lastResult}
				<div
					class="flex h-full min-h-64 flex-col items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground"
				>
					<ImageIcon class="mb-2 h-8 w-8 opacity-40" />
					<p>
						{mode === 'generate'
							? 'Type a prompt and hit Generate.'
							: 'Drop a source image, type an instruction, hit Edit.'}
					</p>
					<p class="mt-1 text-xs opacity-70">
						Output also lands in the gallery; tagged <code>playground</code>.
					</p>
				</div>
			{:else}
				<div class="flex flex-col gap-4">
					<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
						{#each lastResult.dataUrls as dataUrl, i (lastResult.result.images[i].artifactId)}
							{#if dataUrl}
								<a
									href="#/artifacts/{lastResult.result.images[i].artifactId}"
									class="block overflow-hidden rounded-md border bg-background shadow-sm transition-shadow hover:shadow-md"
								>
									<img src={dataUrl} alt={lastResult.result.prompt} class="h-auto w-full" />
								</a>
							{/if}
						{/each}
					</div>
					<div class="flex flex-col gap-1 rounded-md bg-muted/40 p-3 text-xs">
						<p class="text-foreground"><strong>{lastResult.result.prompt}</strong></p>
						<p class="text-muted-foreground">
							{lastResult.mode === 'edit' ? 'Edited' : 'Generated'} · {lastResult.result.model}
							{lastResult.result.size ? `· ${lastResult.result.size}` : ''}
							· {lastResult.result.images.length} image{lastResult.result.images.length === 1
								? ''
								: 's'} · saved to gallery
						</p>
					</div>
				</div>
			{/if}
		</main>

		<!-- RIGHT history rail -->
		<aside
			class="hidden min-h-0 flex-col overflow-y-auto border-l bg-muted/10 p-3 md:flex md:bg-muted/20"
		>
			<div class="mb-2 flex items-center gap-1.5 text-xs text-muted-foreground uppercase">
				<History class="h-3 w-3" />
				History
			</div>
			{#if history.length === 0}
				<p class="text-xs text-muted-foreground">No images yet. Run something on the left.</p>
			{:else}
				<div class="grid grid-cols-2 gap-2">
					{#each history.slice(0, 60) as artifact (artifact.id)}
						{#await loadThumbnail(artifact.currentRevisionId) then dataUrl}
							<button
								type="button"
								onclick={() => pickFromHistory(artifact)}
								class="group block overflow-hidden rounded-md border bg-background transition-shadow hover:shadow-sm"
								title={artifact.title}
							>
								{#if dataUrl}
									<img
										src={dataUrl}
										alt={artifact.title}
										class="aspect-square w-full object-cover"
									/>
								{:else}
									<div class="aspect-square w-full bg-muted/40"></div>
								{/if}
							</button>
						{/await}
					{/each}
				</div>
			{/if}
		</aside>
	</div>
</div>
