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
	import { onDestroy, onMount } from 'svelte';
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';
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
		Film,
		Music,
		X
	} from '@lucide/svelte';

	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Switch } from '$lib/components/ui/switch';
	import { Textarea } from '$lib/components/ui/textarea';
	import * as Select from '$lib/components/ui/select';

	import {
		runImageGeneration,
		runImageEdit,
		runVideoGeneration,
		type RunImageGenerationResult,
		type RunImageEditResult,
		type RunVideoGenerationResult
	} from '$lib/services/builtin-tools';
	import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';
	import { imagePlaygroundStore } from '$lib/stores/image-playground.svelte';
	import { DatabaseService } from '$lib/services/database.service';
	import { config } from '$lib/stores/settings.svelte';
	import { getChatSettingsDialogContext } from '$lib/contexts';
	import { SETTINGS_SECTION_TITLES } from '$lib/constants';
	import type { DatabaseArtifact } from '$lib/types/database';

	// Task type is the unit of "what we're asking the proxy to do" —
	// the model picker, required inputs, and output modality all
	// derive from it. Picker is a flat strip rather than
	// modality→task because the only branching in the UI is
	// "which fields are visible", and that's already keyed off
	// taskType anyway. Adding a future t2v / v2v / inpaint slot is
	// just a new chip + a new filter case.
	type TaskType = 't2i' | 'i2i' | 'i2v' | 's2v' | 'flf';

	type ImageModel = { id: string; label: string };
	type VideoModel = { id: string; label: string; taskType: 'i2v' | 's2v' | 'flf' };

	const GENERATE_MODELS: ImageModel[] = [
		{ id: 'z-image-turbo', label: 'z-image-turbo · ~52s · default' },
		{ id: 'newbie-image', label: 'newbie-image · ~22s · anime / manga' },
		{ id: 'qwen-image', label: 'qwen-image · ~10 min · slow but high quality' },
		{ id: 'flux2-klein', label: 'flux2-klein · broken · OOM on 24 GB' }
	];

	const EDIT_MODELS: ImageModel[] = [
		{ id: 'qwen-image-edit', label: 'qwen-image-edit · ~2.5 min @ 1024' }
	];

	const VIDEO_MODELS: VideoModel[] = [
		{ id: 'wan22-i2v', label: 'wan22-i2v · ~60s — 3min · default', taskType: 'i2v' },
		{ id: 'wan22-i2v-hq', label: 'wan22-i2v-hq · ~5× slower · sharper', taskType: 'i2v' },
		{ id: 'ltx-2.3', label: 'ltx-2.3 · ~4 min · cinematic', taskType: 'i2v' },
		{ id: 'wan22-s2v', label: 'wan22-s2v · ~3.5 min · sound-driven', taskType: 's2v' },
		{ id: 'wan21-flf', label: 'wan21-flf · first ↔ last frame interpolation', taskType: 'flf' }
	];

	// Per-model native size — overriding crashes pipelines, so we
	// hint the user toward the trained resolution and let them go off
	// only via Advanced.
	const VIDEO_DEFAULT_SIZE: Record<string, string> = {
		'wan22-i2v': '832x480',
		'wan22-i2v-hq': '832x480',
		'ltx-2.3': '960x544',
		'wan22-s2v': '512x288',
		'wan21-flf': '832x480'
	};

	function modelsForTask(t: TaskType): Array<ImageModel | VideoModel> {
		switch (t) {
			case 't2i':
				return GENERATE_MODELS;
			case 'i2i':
				return EDIT_MODELS;
			default:
				return VIDEO_MODELS.filter((m) => m.taskType === t);
		}
	}

	function isVideoTask(t: TaskType): boolean {
		return t === 'i2v' || t === 's2v' || t === 'flf';
	}

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

	// Persist the input-rail knobs across reloads so the user's last
	// session shape (task type, model, size, prompt, advanced toggles,
	// frames) comes back when they revisit /images. Source images and
	// audio are not persisted — they're typically one-shot uploads and
	// stale data URLs in localStorage would bloat the bundle for no
	// real benefit. We do remember the prompt because most image-gen
	// UX (A1111, Krea, ComfyUI) keeps it sticky.
	const PERSIST_KEY = 'ht.images.lastParams.v1';
	type PersistedParams = {
		taskType?: TaskType;
		model?: string;
		prompt?: string;
		width?: number;
		height?: number;
		ratioLocked?: boolean;
		lockedRatio?: number;
		nVariants?: number;
		showAdvanced?: boolean;
		negativePrompt?: string;
		seed?: number | null;
		videoFrames?: number;
	};
	function loadPersisted(): PersistedParams {
		if (typeof localStorage === 'undefined') return {};
		try {
			const raw = localStorage.getItem(PERSIST_KEY);
			if (!raw) return {};
			const parsed = JSON.parse(raw) as PersistedParams;
			return parsed && typeof parsed === 'object' ? parsed : {};
		} catch {
			return {};
		}
	}
	const initial = loadPersisted();

	let taskType = $state<TaskType>(initial.taskType ?? 't2i');
	// Playground store still uses the modality-flavoured Mode union;
	// derive it from taskType at the boundary so we don't churn the
	// store/persistence layer for the new task-type axis.
	let mode = $derived<'generate' | 'edit' | 'video'>(
		taskType === 't2i' ? 'generate' : taskType === 'i2i' ? 'edit' : 'video'
	);
	let prompt = $state(initial.prompt ?? '');
	let model = $state(initial.model ?? GENERATE_MODELS[0].id);
	let width = $state(initial.width ?? 1024);
	let height = $state(initial.height ?? 1024);
	// When the chain is locked, edits to width/height preserve the
	// aspect ratio captured at the moment the lock was clicked. We
	// hold the ratio as a frozen number rather than recomputing on
	// every edit so a tiny rounding drift can't slowly distort it.
	let ratioLocked = $state(initial.ratioLocked ?? false);
	let lockedRatio = $state(initial.lockedRatio ?? 1);
	let nVariants = $state(initial.nVariants ?? 1);

	// Advanced toggle. Most users only ever touch prompt + size + model;
	// power users want negative prompt and a fixed seed for reproducibility.
	// Default off keeps the surface uncluttered.
	let showAdvanced = $state(initial.showAdvanced ?? false);
	let negativePrompt = $state(initial.negativePrompt ?? '');
	let seed = $state<number | null>(initial.seed ?? null);

	let editSourceDataUrl = $state<string | null>(null);
	let editSourceArtifactId = $state<string | null>(null);
	let editFileInputRef: HTMLInputElement | null = $state(null);

	// Video-specific knobs. The audio source is only used by s2v;
	// the last-frame source is only used by flf. Both fields render
	// conditionally on the active task type.
	let videoFrames = $state(initial.videoFrames ?? 17);
	let videoAudioDataUrl = $state<string | null>(null);
	let videoAudioFileInputRef: HTMLInputElement | null = $state(null);
	let lastFrameDataUrl = $state<string | null>(null);
	let lastFrameFileInputRef: HTMLInputElement | null = $state(null);

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

	// Persist the input-rail snapshot on every change. Reads are cheap
	// (one localStorage.setItem per state mutation) and the bundle
	// stays small because we never serialise large data URLs (source
	// images, audio).
	$effect(() => {
		if (typeof localStorage === 'undefined') return;
		const snapshot: PersistedParams = {
			taskType,
			model,
			prompt,
			width,
			height,
			ratioLocked,
			lockedRatio,
			nVariants,
			showAdvanced,
			negativePrompt,
			seed,
			videoFrames
		};
		try {
			localStorage.setItem(PERSIST_KEY, JSON.stringify(snapshot));
		} catch {
			/* quota exceeded or storage disabled — silently skip */
		}
	});

	function formatElapsed(ms: number): string {
		const total = Math.max(0, Math.floor(ms / 1000));
		const m = Math.floor(total / 60);
		const s = total % 60;
		return m > 0 ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`;
	}

	let imageGenEnabled = $derived(Boolean(config().imageGenEnabled));
	let videoGenEnabled = $derived(Boolean(config().videoGenEnabled));
	let imagesBaseUrl = $derived(String(config().imagesBaseUrl ?? '').trim());

	// Per-mode capability check used by the Run button. Image gen
	// flows share `imageGenEnabled` (Generate + Edit are the same
	// proxy); Video has its own toggle since it's an order of magnitude
	// slower and the user should opt in explicitly.
	let modeEnabled = $derived(isVideoTask(taskType) ? videoGenEnabled : imageGenEnabled);

	// Derived task-type booleans for cleaner UI conditionals.
	let isVideo = $derived(isVideoTask(taskType));
	let needsSourceImage = $derived(taskType !== 't2i');
	let needsAudio = $derived(taskType === 's2v');
	let needsLastFrame = $derived(taskType === 'flf');

	// Reactive history: every image artifact in the gallery, newest first.
	// Includes output from all four entry points; metadata.source is what
	// distinguishes them, but for browsing we want everything in one rail.
	let history = $derived.by<DatabaseArtifact[]>(() => {
		// Include images by default; widen to video when the user is on
		// a video-output task so they can drag an old generation back
		// into the canvas without flipping tabs.
		const wanted = isVideo ? ['image', 'video'] : ['image'];
		return [...artifactGalleryStore.artifacts]
			.filter((a) => wanted.includes(a.kind))
			.sort((a, b) => b.updatedAt - a.updatedAt);
	});

	// Resolved preview-url cache keyed by revisionId, for the history rail
	// thumbnails. Blob-backed artifact previews use object URLs to avoid
	// base64 inflation in the webview; user-selected source files still use
	// data URLs because the generation APIs send those in JSON.
	let thumbnailCache = new SvelteMap<string, string>();
	const objectUrls = new SvelteSet<string>();
	type PlaygroundFinishedRunArg = Parameters<typeof imagePlaygroundStore.finishRun>[0];

	onMount(async () => {
		// Refresh the gallery so the history rail is populated even if
		// the user lands here directly without visiting /artifacts.
		await artifactGalleryStore.load();
	});

	onDestroy(() => {
		for (const url of objectUrls) URL.revokeObjectURL(url);
		objectUrls.clear();
	});

	function objectUrlFor(blob: Blob): string {
		const url = URL.createObjectURL(blob);
		objectUrls.add(url);
		return url;
	}

	function releaseObjectUrls(urls: string[] | undefined) {
		for (const url of urls ?? []) {
			URL.revokeObjectURL(url);
			objectUrls.delete(url);
		}
	}

	function finishPreviewRun(args: PlaygroundFinishedRunArg) {
		releaseObjectUrls(imagePlaygroundStore.lastFinished?.revokeUrls);
		imagePlaygroundStore.finishRun(args);
	}

	$effect(() => {
		// Switching task types resets the model selection so we never
		// end up with a t2i-only model id selected in i2v mode (etc).
		// Each task type has its own model list — see modelsForTask().
		const allowed = modelsForTask(taskType);
		if (!allowed.some((m) => m.id === model)) {
			model = allowed[0]?.id ?? '';
		}
		// Snap size to the selected video model's native resolution so
		// the user doesn't waste 5 minutes finding out 1024x1024 OOMs.
		if (isVideo) {
			const native = VIDEO_DEFAULT_SIZE[model];
			if (native) {
				const [w, h] = native.split('x').map(Number);
				if (Number.isFinite(w) && Number.isFinite(h)) {
					width = w;
					height = h;
				}
			}
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

	async function loadThumbnail(artifact: DatabaseArtifact): Promise<string | null> {
		const revisionId = artifact.currentRevisionId;
		const cached = thumbnailCache.get(revisionId);
		if (cached) return cached;
		const revision = await DatabaseService.getArtifactRevision(revisionId);
		if (!revision?.blob) return null;
		const dataUrl = objectUrlFor(revision.blob);
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
		if (artifact.kind === 'video') {
			const objectUrl = objectUrlFor(revision.blob);
			const metadata = (artifact.metadata ?? {}) as Record<string, unknown>;
			finishPreviewRun({
				mode: 'video',
				result: {
					model: String(
						metadata.model ??
							artifact.tags.find((t) => t !== 'generated' && t !== 'playground') ??
							'video'
					),
					size: String(metadata.size ?? 'unknown size'),
					frames: Number(metadata.frames ?? 0),
					prompt: String(metadata.prompt ?? artifact.title),
					jobId: String(metadata.jobId ?? ''),
					video: {
						artifactId: artifact.id,
						revisionId: artifact.currentRevisionId,
						title: artifact.title,
						mimeType: revision.mimeType ?? 'video/mp4',
						bytes: revision.blob.size
					}
				},
				dataUrls: [objectUrl],
				revokeUrls: [objectUrl],
				finishedAt: Date.now()
			});
			return;
		}
		const previewUrl = objectUrlFor(revision.blob);
		const sourceDataUrl = needsSourceImage ? await blobToDataUrl(revision.blob) : null;
		// History click ALSO loads the artifact into the source-image
		// slot for tasks that consume one — saves a re-upload round-trip
		// when iterating on the same still. Skip for t2i (no source
		// slot) and flf (ambiguous: would the user mean first or last?
		// Force them to drag onto the explicit drop zone instead).
		if ((taskType === 'i2i' || taskType === 'i2v' || taskType === 's2v') && sourceDataUrl) {
			editSourceDataUrl = sourceDataUrl;
			editSourceArtifactId = artifact.id;
		}
		// Don't trample an in-flight run's spinner state. The user can
		// still pick a source for the *next* edit while a run is going
		// — the canvas just keeps showing the running indicator.
		if (imagePlaygroundStore.active) return;
		// Show in the canvas as a "viewing existing" preview by writing
		// to the playground store, same surface a real run uses.
		finishPreviewRun({
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
			dataUrls: [previewUrl],
			revokeUrls: [previewUrl],
			finishedAt: Date.now()
		});
	}

	function clearEditSource() {
		editSourceDataUrl = null;
		editSourceArtifactId = null;
	}

	// Drag-and-drop wiring for the Source image / Source audio drop zones.
	// The composer copy literally says "drop a source image first" — backing
	// it with real drop handlers so the affordance isn't a lie.
	let editDropActive = $state(false);
	let audioDropActive = $state(false);

	async function setEditSourceFromFile(file: File) {
		try {
			const dataUrl = await blobToDataUrl(file);
			editSourceDataUrl = dataUrl;
			editSourceArtifactId = null;
		} catch (e) {
			toast.error(`Failed to read file: ${e instanceof Error ? e.message : String(e)}`);
		}
	}

	function pickFileFromDrop(ev: DragEvent, predicate: (f: File) => boolean): File | null {
		const items = ev.dataTransfer?.files;
		if (!items?.length) return null;
		for (const f of Array.from(items)) {
			if (predicate(f)) return f;
		}
		return null;
	}

	function onEditDragEnter(ev: DragEvent) {
		if (!ev.dataTransfer?.types.includes('Files')) return;
		ev.preventDefault();
		editDropActive = true;
	}
	function onEditDragOver(ev: DragEvent) {
		if (ev.dataTransfer?.types.includes('Files')) ev.preventDefault();
	}
	function onEditDragLeave(ev: DragEvent) {
		ev.preventDefault();
		editDropActive = false;
	}
	async function onEditDrop(ev: DragEvent) {
		ev.preventDefault();
		editDropActive = false;
		const file = pickFileFromDrop(ev, (f) => f.type.startsWith('image/'));
		if (!file) {
			toast.info('Drop an image file (PNG, JPG, WebP, …)');
			return;
		}
		await setEditSourceFromFile(file);
	}

	function onAudioDragEnter(ev: DragEvent) {
		if (!ev.dataTransfer?.types.includes('Files')) return;
		ev.preventDefault();
		audioDropActive = true;
	}
	function onAudioDragOver(ev: DragEvent) {
		if (ev.dataTransfer?.types.includes('Files')) ev.preventDefault();
	}
	function onAudioDragLeave(ev: DragEvent) {
		ev.preventDefault();
		audioDropActive = false;
	}
	async function onAudioDrop(ev: DragEvent) {
		ev.preventDefault();
		audioDropActive = false;
		const file = pickFileFromDrop(ev, (f) => f.type.startsWith('audio/'));
		if (!file) {
			toast.info('Drop an audio file (wav, mp3, ogg, flac)');
			return;
		}
		try {
			videoAudioDataUrl = await blobToDataUrl(file);
		} catch (e) {
			toast.error(`Failed to read audio: ${e instanceof Error ? e.message : String(e)}`);
		}
	}

	async function handleVideoAudioFileChange(event: Event) {
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;
		try {
			videoAudioDataUrl = await blobToDataUrl(file);
		} catch (e) {
			toast.error(`Failed to read audio: ${e instanceof Error ? e.message : String(e)}`);
		} finally {
			input.value = '';
		}
	}

	function clearVideoAudio() {
		videoAudioDataUrl = null;
	}

	// ----- last-frame drop zone (FLF only) ---------------------------
	let lastFrameDropActive = $state(false);

	function pickLastFrame() {
		lastFrameFileInputRef?.click();
	}

	async function setLastFrameFromFile(file: File) {
		try {
			lastFrameDataUrl = await blobToDataUrl(file);
		} catch (e) {
			toast.error(`Failed to read file: ${e instanceof Error ? e.message : String(e)}`);
		}
	}

	async function handleLastFrameFileChange(event: Event) {
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;
		await setLastFrameFromFile(file);
		input.value = '';
	}

	function onLastFrameDragEnter(ev: DragEvent) {
		if (!ev.dataTransfer?.types.includes('Files')) return;
		ev.preventDefault();
		lastFrameDropActive = true;
	}
	function onLastFrameDragOver(ev: DragEvent) {
		if (ev.dataTransfer?.types.includes('Files')) ev.preventDefault();
	}
	function onLastFrameDragLeave(ev: DragEvent) {
		ev.preventDefault();
		lastFrameDropActive = false;
	}
	async function onLastFrameDrop(ev: DragEvent) {
		ev.preventDefault();
		lastFrameDropActive = false;
		const file = pickFileFromDrop(ev, (f) => f.type.startsWith('image/'));
		if (!file) {
			toast.info('Drop an image file (PNG, JPG, WebP, …)');
			return;
		}
		await setLastFrameFromFile(file);
	}

	function clearLastFrame() {
		lastFrameDataUrl = null;
	}

	async function run() {
		if (isRunning) return;
		const trimmedPrompt = prompt.trim();
		// Video tasks drive output from the visual / audio signal (the
		// still, the audio track, the first+last frame pair) — the
		// prompt is at most flavoring. runVideoGeneration substitutes a
		// neutral placeholder when empty so the proxy's minLength-1
		// schema passes. t2i and i2i are genuinely prompt-driven so the
		// empty guard still applies.
		if (!trimmedPrompt && !isVideo) {
			toast.info('Enter a prompt first.');
			return;
		}
		if (isVideo && !videoGenEnabled) {
			toast.error('Video generation is disabled. Enable it in Settings → Images.');
			openImagesSettings();
			return;
		}
		if (!isVideo && !imageGenEnabled) {
			toast.error('Image generation is disabled. Enable it in Settings → Images.');
			openImagesSettings();
			return;
		}
		if (!imagesBaseUrl) {
			toast.error('Set Settings → Images → Base URL first.');
			openImagesSettings();
			return;
		}
		if (needsSourceImage && !editSourceDataUrl) {
			toast.info(
				taskType === 'flf'
					? 'FLF needs a first frame. Drop one in the source slot.'
					: taskType === 'i2i'
						? 'Drop a source image first, or click a history card to reuse one.'
						: 'Drop a source image first.'
			);
			return;
		}
		if (needsAudio && !videoAudioDataUrl) {
			toast.info(`${model} is sound-driven — attach an audio clip below the source image.`);
			return;
		}
		if (needsLastFrame && !lastFrameDataUrl) {
			toast.info('FLF needs a last frame as well — drop one in the last-frame slot.');
			return;
		}

		const controller = new AbortController();
		const startedMode = mode;
		const startedTask = taskType;
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
						return rev?.blob ? objectUrlFor(rev.blob) : Promise.resolve('');
					})
				);
				finishPreviewRun({
					mode: 'generate',
					result,
					dataUrls,
					revokeUrls: dataUrls.filter(Boolean),
					finishedAt: Date.now()
				});
			} else if (startedMode === 'edit') {
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
						return rev?.blob ? objectUrlFor(rev.blob) : Promise.resolve('');
					})
				);
				finishPreviewRun({
					mode: 'edit',
					result,
					dataUrls,
					revokeUrls: dataUrls.filter(Boolean),
					finishedAt: Date.now()
				});
			} else {
				const result = await runVideoGeneration({
					source: 'playground',
					prompt: trimmedPrompt,
					model: startedModel,
					image: editSourceDataUrl as string,
					audio: startedTask === 's2v' ? (videoAudioDataUrl ?? undefined) : undefined,
					lastFrame: startedTask === 'flf' ? (lastFrameDataUrl ?? undefined) : undefined,
					size,
					frames: videoFrames,
					signal: controller.signal
				});
				const rev = await DatabaseService.getArtifactRevision(result.video.revisionId);
				const objectUrl = rev?.blob ? objectUrlFor(rev.blob) : '';
				finishPreviewRun({
					mode: 'video',
					result,
					dataUrls: [objectUrl],
					revokeUrls: objectUrl ? [objectUrl] : undefined,
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
	<!-- Top strip: task-type picker + page title.
		 Five chips, flat: t2i / i2i / i2v / s2v / flf. The model
		 dropdown filters to whichever task is active; required input
		 fields (source image / audio / last frame) appear conditionally
		 below. -->
	<div class="flex flex-shrink-0 flex-wrap items-center gap-3 border-b px-4 py-3">
		<h1 class="text-lg font-semibold">Images</h1>

		<div class="ml-2 inline-flex rounded-full border bg-muted/30 p-0.5">
			<button
				type="button"
				class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors {taskType ===
				't2i'
					? 'bg-primary text-primary-foreground'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (taskType = 't2i')}
				title="Text → image"
			>
				<ImageIcon class="h-3.5 w-3.5" />
				t2i
			</button>
			<button
				type="button"
				class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors {taskType ===
				'i2i'
					? 'bg-primary text-primary-foreground'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (taskType = 'i2i')}
				title="Image → image (edit)"
			>
				<Pencil class="h-3.5 w-3.5" />
				i2i
			</button>
			<button
				type="button"
				class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors {taskType ===
				'i2v'
					? 'bg-primary text-primary-foreground'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (taskType = 'i2v')}
				title="Image → video"
			>
				<Film class="h-3.5 w-3.5" />
				i2v
			</button>
			<button
				type="button"
				class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors {taskType ===
				's2v'
					? 'bg-primary text-primary-foreground'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (taskType = 's2v')}
				title="Sound → video"
			>
				<Music class="h-3.5 w-3.5" />
				s2v
			</button>
			<button
				type="button"
				class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors {taskType ===
				'flf'
					? 'bg-primary text-primary-foreground'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (taskType = 'flf')}
				title="First + last frame → video"
			>
				<Film class="h-3.5 w-3.5" />
				flf
			</button>
		</div>

		{#if isVideo && !videoGenEnabled}
			<span class="ml-2 text-xs text-amber-600">Video generation is disabled.</span>
		{:else if !isVideo && !imageGenEnabled}
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
						placeholder={taskType === 't2i'
							? 'A photo of a calico cat sitting on a sunlit windowsill…'
							: taskType === 'i2i'
								? 'Make the sky a deep purple at sunset, keep everything else.'
								: taskType === 'flf'
									? 'How the first frame should morph into the last frame…'
									: 'Describe how the still should animate…'}
						rows={6}
						disabled={isRunning}
						class="resize-y"
					/>
				</div>

				{#if needsSourceImage}
					<div
						class="flex flex-col gap-1.5"
						ondragenter={onEditDragEnter}
						ondragover={onEditDragOver}
						ondragleave={onEditDragLeave}
						ondrop={onEditDrop}
						role="presentation"
					>
						<Label class="text-xs text-muted-foreground uppercase">
							{taskType === 'flf'
								? 'First frame'
								: taskType === 'i2v' || taskType === 's2v'
									? 'Source image (still to animate)'
									: 'Source image'}
						</Label>
						{#if editSourceDataUrl}
							<div
								class="relative overflow-hidden rounded-md border bg-background transition-colors {editDropActive
									? 'border-primary ring-2 ring-primary/40'
									: ''}"
							>
								<img src={editSourceDataUrl} alt="Source" class="h-40 w-full object-cover" />
								<button
									type="button"
									onclick={clearEditSource}
									class="absolute top-1 right-1 rounded-full bg-background/80 p-1 hover:bg-background"
									aria-label="Clear source"
								>
									<X class="h-3 w-3" />
								</button>
								{#if editDropActive}
									<div
										class="pointer-events-none absolute inset-0 flex items-center justify-center bg-primary/15 text-xs font-medium text-primary"
									>
										Drop to replace
									</div>
								{/if}
							</div>
						{:else}
							<button
								type="button"
								onclick={pickEditSource}
								class="flex h-32 flex-col items-center justify-center rounded-md border border-dashed text-xs text-muted-foreground transition-colors hover:border-primary hover:text-foreground {editDropActive
									? 'border-primary bg-primary/5 text-primary'
									: ''}"
							>
								<ImageIcon class="mb-1 h-5 w-5 opacity-60" />
								{editDropActive
									? 'Drop image here'
									: 'Click to upload, drop a file, or pick from history →'}
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

				{#if needsLastFrame}
					<div
						class="flex flex-col gap-1.5"
						ondragenter={onLastFrameDragEnter}
						ondragover={onLastFrameDragOver}
						ondragleave={onLastFrameDragLeave}
						ondrop={onLastFrameDrop}
						role="presentation"
					>
						<Label class="text-xs text-muted-foreground uppercase">Last frame</Label>
						{#if lastFrameDataUrl}
							<div
								class="relative overflow-hidden rounded-md border bg-background transition-colors {lastFrameDropActive
									? 'border-primary ring-2 ring-primary/40'
									: ''}"
							>
								<img src={lastFrameDataUrl} alt="Last frame" class="h-40 w-full object-cover" />
								<button
									type="button"
									onclick={clearLastFrame}
									class="absolute top-1 right-1 rounded-full bg-background/80 p-1 hover:bg-background"
									aria-label="Clear last frame"
								>
									<X class="h-3 w-3" />
								</button>
								{#if lastFrameDropActive}
									<div
										class="pointer-events-none absolute inset-0 flex items-center justify-center bg-primary/15 text-xs font-medium text-primary"
									>
										Drop to replace
									</div>
								{/if}
							</div>
						{:else}
							<button
								type="button"
								onclick={pickLastFrame}
								class="flex h-32 flex-col items-center justify-center rounded-md border border-dashed text-xs text-muted-foreground transition-colors hover:border-primary hover:text-foreground {lastFrameDropActive
									? 'border-primary bg-primary/5 text-primary'
									: ''}"
							>
								<ImageIcon class="mb-1 h-5 w-5 opacity-60" />
								{lastFrameDropActive ? 'Drop image here' : 'Click to upload, or drop a file here'}
							</button>
						{/if}
						<input
							type="file"
							accept="image/*"
							class="hidden"
							bind:this={lastFrameFileInputRef}
							onchange={handleLastFrameFileChange}
						/>
					</div>
				{/if}

				{#if needsAudio}
					<div
						class="flex flex-col gap-1.5"
						ondragenter={onAudioDragEnter}
						ondragover={onAudioDragOver}
						ondragleave={onAudioDragLeave}
						ondrop={onAudioDrop}
						role="presentation"
					>
						<Label class="text-xs text-muted-foreground uppercase">Source audio (s2v)</Label>
						{#if videoAudioDataUrl}
							<div
								class="relative flex items-center gap-2 rounded-md border bg-background p-2 transition-colors {audioDropActive
									? 'border-primary ring-2 ring-primary/40'
									: ''}"
							>
								<Music class="h-4 w-4 text-muted-foreground" />
								<audio src={videoAudioDataUrl} controls class="min-w-0 flex-1"></audio>
								<button
									type="button"
									onclick={clearVideoAudio}
									class="rounded-full bg-background/80 p-1 hover:bg-background"
									aria-label="Clear audio"
								>
									<X class="h-3 w-3" />
								</button>
								{#if audioDropActive}
									<div
										class="pointer-events-none absolute inset-0 flex items-center justify-center rounded-md bg-primary/15 text-xs font-medium text-primary"
									>
										Drop to replace
									</div>
								{/if}
							</div>
						{:else}
							<button
								type="button"
								onclick={() => videoAudioFileInputRef?.click()}
								class="flex h-20 flex-col items-center justify-center rounded-md border border-dashed text-xs text-muted-foreground transition-colors hover:border-primary hover:text-foreground {audioDropActive
									? 'border-primary bg-primary/5 text-primary'
									: ''}"
							>
								<Music class="mb-1 h-4 w-4 opacity-60" />
								{audioDropActive ? 'Drop audio here' : 'wav · mp3 · ogg · flac · click or drop'}
							</button>
						{/if}
						<input
							type="file"
							accept="audio/*"
							class="hidden"
							bind:this={videoAudioFileInputRef}
							onchange={handleVideoAudioFileChange}
						/>
					</div>
				{/if}

				<div class="flex flex-col gap-1.5">
					<Label for="img-model" class="text-xs text-muted-foreground uppercase">Model</Label>
					<Select.Root type="single" bind:value={model} disabled={isRunning}>
						<Select.Trigger id="img-model" class="w-full">
							{modelsForTask(taskType).find((m) => m.id === model)?.label ?? model}
						</Select.Trigger>
						<Select.Content>
							{#each modelsForTask(taskType) as opt (opt.id)}
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

				{#if mode === 'video'}
					<div class="flex flex-col gap-1.5">
						<Label for="img-frames" class="text-xs text-muted-foreground uppercase">Frames</Label>
						<Input
							id="img-frames"
							type="number"
							min="1"
							max="121"
							bind:value={videoFrames}
							disabled={isRunning}
							class="w-full"
						/>
						<p class="text-[10px] text-muted-foreground">
							17 ≈ 1 s · 49 = balanced · 81 = long-form · runtime grows with frames.
						</p>
					</div>
				{:else}
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
				{/if}

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
							{mode === 'generate'
								? 'Generating…'
								: mode === 'edit'
									? 'Editing…'
									: 'Rendering video…'} can take 20 seconds to a few minutes depending on the model.
						</p>
					{:else}
						<Button onclick={run} class="w-full gap-2" disabled={!modeEnabled}>
							<PlayCircle class="h-4 w-4" />
							{mode === 'generate' ? 'Generate' : mode === 'edit' ? 'Edit' : 'Render video'}
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
					<Loader2 class="mb-2 h-6 w-6 animate-spin text-primary" />
					<p class="text-foreground">
						{activeRun.mode === 'generate'
							? 'Generating'
							: activeRun.mode === 'edit'
								? 'Editing'
								: 'Rendering video'} · {activeRun.model}
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
							: mode === 'edit'
								? 'Drop a source image, type an instruction, hit Edit.'
								: 'Drop a source image and a motion prompt to animate it.'}
					</p>
					<p class="mt-1 text-xs opacity-70">
						Output also lands in the gallery; tagged <code>playground</code>.
					</p>
				</div>
			{:else if lastResult.mode === 'video'}
				{@const videoResult = lastResult.result as RunVideoGenerationResult}
				<div class="flex flex-col gap-4">
					<a
						href="#/artifacts/{videoResult.video.artifactId}"
						class="block overflow-hidden rounded-md border bg-background shadow-sm transition-shadow hover:shadow-md"
						aria-label="Open video in gallery"
					>
						<!-- svelte-ignore a11y_media_has_caption -->
						<video src={lastResult.dataUrls[0]} preload="none" controls loop class="h-auto w-full"
						></video>
					</a>
					<div class="flex flex-col gap-1 rounded-md bg-muted/40 p-3 text-xs">
						<p class="text-foreground"><strong>{videoResult.prompt}</strong></p>
						<p class="text-muted-foreground">
							Rendered · {videoResult.model} · {videoResult.size} · {videoResult.frames} frames · saved
							to gallery
						</p>
					</div>
				</div>
			{:else}
				{@const imgResult = lastResult.result as RunImageGenerationResult | RunImageEditResult}
				<div class="flex flex-col gap-4">
					<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
						{#each lastResult.dataUrls as dataUrl, i (imgResult.images[i].artifactId)}
							{#if dataUrl}
								<a
									href="#/artifacts/{imgResult.images[i].artifactId}"
									class="block overflow-hidden rounded-md border bg-background shadow-sm transition-shadow hover:shadow-md"
								>
									<img src={dataUrl} alt={imgResult.prompt} class="h-auto w-full" />
								</a>
							{/if}
						{/each}
					</div>
					<div class="flex flex-col gap-1 rounded-md bg-muted/40 p-3 text-xs">
						<p class="text-foreground"><strong>{imgResult.prompt}</strong></p>
						<p class="text-muted-foreground">
							{lastResult.mode === 'edit' ? 'Edited' : 'Generated'} · {imgResult.model}
							{imgResult.size ? `· ${imgResult.size}` : ''}
							· {imgResult.images.length} image{imgResult.images.length === 1 ? '' : 's'} · saved to
							gallery
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
						{#await loadThumbnail(artifact) then dataUrl}
							<button
								type="button"
								onclick={() => pickFromHistory(artifact)}
								class="group block overflow-hidden rounded-md border bg-background transition-shadow hover:shadow-sm"
								title={artifact.kind === 'video'
									? `${artifact.title} (hover to play)`
									: artifact.title}
							>
								{#if dataUrl}
									{#if artifact.kind === 'video'}
										<!-- preload="none" — webkit2gtk's GStreamer pipeline
										     trips a `gst_value_collect_int_range` assertion on
										     metadata probing for some clips, which crashes the
										     entire WebProcess. Holding pipeline construction
										     until the user actually hovers (which calls
										     `play()`) keeps the gallery view stable, at the
										     cost of no poster frame on cold load — the empty
										     box fills in the moment the user reaches for it. -->
										<video
											src={dataUrl}
											preload="none"
											muted
											playsinline
											loop
											class="aspect-square w-full bg-muted/30 object-cover"
											onmouseenter={(ev) => {
												const v = ev.currentTarget as HTMLVideoElement;
												v.play().catch(() => {});
											}}
											onmouseleave={(ev) => {
												const v = ev.currentTarget as HTMLVideoElement;
												v.pause();
												v.currentTime = 0;
											}}
										>
											<track kind="captions" />
										</video>
									{:else}
										<img
											src={dataUrl}
											alt={artifact.title}
											class="aspect-square w-full object-cover"
										/>
									{/if}
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
