<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Mic, Upload, X, Play, Square } from '@lucide/svelte';

	interface Props {
		dataUri: string;
		fileName: string;
		onChange: (update: { dataUri: string; fileName: string }) => void;
	}

	let { dataUri, fileName, onChange }: Props = $props();

	let inputEl: HTMLInputElement | undefined = $state();
	let audioEl: HTMLAudioElement | undefined = $state();
	let isPlaying = $state(false);
	let error = $state<string | null>(null);
	let busy = $state(false);

	const MAX_BYTES = 5 * 1024 * 1024;

	async function readAsDataUri(file: File): Promise<string> {
		return new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.onerror = () => reject(reader.error ?? new Error('read error'));
			reader.onload = () => resolve(String(reader.result ?? ''));
			reader.readAsDataURL(file);
		});
	}

	async function handleSelect(ev: Event) {
		error = null;
		const input = ev.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		if (file.size > MAX_BYTES) {
			error = `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_BYTES / 1024 / 1024} MB.`;
			input.value = '';
			return;
		}

		busy = true;
		try {
			const uri = await readAsDataUri(file);
			onChange({ dataUri: uri, fileName: file.name });
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to read file';
		} finally {
			busy = false;
			input.value = '';
		}
	}

	function clear() {
		if (audioEl) {
			audioEl.pause();
			audioEl.currentTime = 0;
		}
		isPlaying = false;
		error = null;
		onChange({ dataUri: '', fileName: '' });
	}

	function togglePlayback() {
		if (!audioEl) return;
		if (audioEl.paused) {
			audioEl.play().then(() => {
				isPlaying = true;
			});
		} else {
			audioEl.pause();
			isPlaying = false;
		}
	}

	let sizeLabel = $derived.by(() => {
		if (!dataUri) return '';
		// Rough byte count from the base64 payload (length * 3/4 minus padding).
		const i = dataUri.indexOf('base64,');
		const b64 = i >= 0 ? dataUri.slice(i + 7) : dataUri;
		const bytes = Math.floor(b64.length * 0.75);
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
	});
</script>

<div class="space-y-2 rounded-md border border-border/40 p-3">
	<div class="flex items-center gap-2">
		<Mic class="h-4 w-4 text-muted-foreground" />
		<span class="text-sm font-medium">Reference audio (voice cloning)</span>
	</div>
	<p class="text-xs text-muted-foreground">
		Upload a short clean sample of the target voice (5–30s, any common format). When set, requests
		include <code>ref_audio</code> and
		<code>x_vector_only_mode=true</code>.
	</p>

	<input type="file" accept="audio/*" class="hidden" bind:this={inputEl} onchange={handleSelect} />

	<div class="flex flex-wrap items-center gap-2">
		<Button variant="outline" size="sm" onclick={() => inputEl?.click()} disabled={busy}>
			<Upload class="mr-1.5 h-3.5 w-3.5" />
			{dataUri ? 'Replace' : 'Upload audio'}
		</Button>

		{#if dataUri}
			<Button variant="ghost" size="sm" onclick={togglePlayback}>
				{#if isPlaying}
					<Square class="mr-1.5 h-3.5 w-3.5" /> Stop
				{:else}
					<Play class="mr-1.5 h-3.5 w-3.5" /> Preview
				{/if}
			</Button>
			<Button variant="ghost" size="sm" onclick={clear}>
				<X class="mr-1.5 h-3.5 w-3.5" /> Clear
			</Button>
		{/if}
	</div>

	{#if dataUri}
		<div class="flex items-center gap-2 text-xs text-muted-foreground">
			<span class="truncate font-mono" title={fileName}>{fileName || 'reference.audio'}</span>
			<span class="flex-shrink-0">·</span>
			<span class="flex-shrink-0">{sizeLabel}</span>
		</div>
		<audio bind:this={audioEl} src={dataUri} onended={() => (isPlaying = false)} class="hidden"
		></audio>
	{/if}

	{#if error}
		<p class="text-xs text-destructive">{error}</p>
	{/if}
</div>
