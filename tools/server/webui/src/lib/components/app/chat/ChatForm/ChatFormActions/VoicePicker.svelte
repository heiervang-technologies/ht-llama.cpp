<script lang="ts">
	import { untrack } from 'svelte';
	import { ChevronDown, Check, Loader2, Volume2, RefreshCw } from '@lucide/svelte';
	import * as Popover from '$lib/components/ui/popover';
	import { Input } from '$lib/components/ui/input';
	import { cn } from '$lib/components/ui/utils';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { voicesStore } from '$lib/stores/voices.svelte';
	import { TtsService } from '$lib/services/tts.service';
	import TtsRefAudioPicker from '$lib/components/app/chat/ChatSettings/TtsRefAudioPicker.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
	}

	let { class: className = '', disabled = false }: Props = $props();

	let isOpen = $state(false);
	let searchTerm = $state('');
	let lastFetchedBase: string | null = null;

	let currentConfig = $derived(config());
	let ttsReady = $derived(TtsService.isConfigured());
	let currentVoice = $derived(String(currentConfig.ttsVoice ?? '').trim());
	let refAudioName = $derived(String(currentConfig.ttsRefAudioName ?? '').trim());
	let refAudioUri = $derived(String(currentConfig.ttsRefAudio ?? '').trim());
	let hasCustomClone = $derived(Boolean(refAudioUri));

	// Re-fetch the voices list whenever the TTS base URL changes (e.g. user
	// edits it in Settings, or we switch environments). Graceful degradation
	// baked into the service — empty list on 404 / unreachable.
	$effect(() => {
		const baseUrl = String(currentConfig.ttsBaseUrl ?? '').trim();
		untrack(() => {
			if (baseUrl !== lastFetchedBase) {
				lastFetchedBase = baseUrl;
				if (baseUrl) {
					voicesStore.fetch().catch(() => {
						/* errors are swallowed in the service; empty list is the fallback */
					});
				} else {
					voicesStore.clear();
				}
			}
		});
	});

	let filteredVoices = $derived.by(() => {
		const q = searchTerm.trim().toLowerCase();
		if (!q) return voicesStore.voices;
		return voicesStore.voices.filter((v) => {
			const hay = `${v.id} ${v.name ?? ''} ${v.language ?? ''}`.toLowerCase();
			return hay.includes(q);
		});
	});

	let triggerLabel = $derived.by(() => {
		if (currentVoice) return currentVoice;
		if (hasCustomClone) return 'Custom clone';
		return 'Voice';
	});

	function pickVoice(voiceId: string) {
		settingsStore.updateConfig('ttsVoice', voiceId);
		isOpen = false;
	}

	function pickCustomClone() {
		// When the user explicitly selects the uploaded clone, clear the named
		// voice so the TTS server uses ref_audio only (Qwen3-TTS cloning).
		settingsStore.updateConfig('ttsVoice', '');
		isOpen = false;
	}

	function handleUpload(update: { dataUri: string; fileName: string }) {
		settingsStore.updateConfig('ttsRefAudio', update.dataUri);
		settingsStore.updateConfig('ttsRefAudioName', update.fileName);
	}

	function refetch() {
		voicesStore.fetch().catch(() => {});
	}
</script>

<Popover.Root bind:open={isOpen}>
	<Popover.Trigger
		{disabled}
		class={cn(
			'inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60',
			currentVoice || hasCustomClone ? 'text-foreground' : 'text-muted-foreground',
			isOpen ? 'text-foreground' : '',
			className
		)}
		aria-label="Pick voice"
	>
		<Volume2 class="h-3.5 w-3.5" />

		<span class="max-w-[8rem] truncate font-medium">{triggerLabel}</span>

		<ChevronDown class="h-3 w-3.5" />
	</Popover.Trigger>

	<Popover.Content align="end" class="w-80 p-1.5">
		<div class="flex items-center justify-between px-2 py-1.5">
			<span class="text-xs font-semibold text-muted-foreground/60 select-none">Voice</span>
			<button
				type="button"
				class="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
				onclick={refetch}
				disabled={!ttsReady || voicesStore.loading}
				title="Refresh voices"
			>
				{#if voicesStore.loading}
					<Loader2 class="h-3 w-3 animate-spin" />
				{:else}
					<RefreshCw class="h-3 w-3" />
				{/if}
				Refresh
			</button>
		</div>

		<div class="px-1">
			<Input type="text" placeholder="Search voices…" bind:value={searchTerm} class="h-7 text-xs" />
		</div>

		{#if !ttsReady}
			<div class="px-2 py-3 text-center text-xs text-muted-foreground">
				Configure TTS in Settings to pick a voice.
			</div>
		{:else}
			{#if voicesStore.lastError}
				<div
					class="mx-1 mt-1 rounded-sm border border-red-400/30 bg-red-400/10 px-2 py-1.5 text-[11px] text-red-400"
					role="alert"
				>
					<div class="font-semibold">Could not load voices</div>
					<div class="mt-0.5 break-words opacity-90">{voicesStore.lastError}</div>
				</div>
			{/if}

			<div class="mt-1.5 max-h-60 overflow-auto">
				{#if filteredVoices.length === 0 && !voicesStore.loading && !voicesStore.lastError}
					<div class="px-2 py-2 text-[11px] text-muted-foreground/70">
						{#if voicesStore.voices.length === 0}
							Server didn't expose a <code class="font-mono">/v1/audio/voices</code> list. Type a voice
							name below, or upload a reference clip.
						{:else}
							No matches.
						{/if}
					</div>
				{/if}

				{#each filteredVoices as voice (voice.id)}
					{@const isActive = currentVoice === voice.id}
					<button
						type="button"
						class={cn(
							'flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-xs hover:bg-accent',
							isActive ? 'text-foreground' : 'text-muted-foreground'
						)}
						onclick={() => pickVoice(voice.id)}
					>
						<Check class={cn('h-3.5 w-3.5 shrink-0', isActive ? 'opacity-100' : 'opacity-0')} />
						<span class="min-w-0 flex-1 truncate font-medium">
							{voice.name ?? voice.id}
						</span>
						{#if voice.language}
							<span class="shrink-0 text-[10px] text-muted-foreground/60">{voice.language}</span>
						{/if}
					</button>
				{/each}
			</div>

			<div class="mt-1 border-t border-border/50 pt-1.5">
				<label class="block px-2 py-1">
					<span class="text-[10px] font-semibold tracking-wider text-muted-foreground/60 uppercase">
						Free-text voice
					</span>
					<Input
						type="text"
						placeholder="e.g. Chelsie, af_bella"
						value={currentVoice}
						oninput={(e) =>
							settingsStore.updateConfig('ttsVoice', (e.currentTarget as HTMLInputElement).value)}
						class="mt-1 h-7 text-xs"
					/>
				</label>
			</div>

			<div class="mt-1 border-t border-border/50 pt-1.5">
				<div class="flex items-center justify-between px-2 py-1">
					<span class="text-[10px] font-semibold tracking-wider text-muted-foreground/60 uppercase">
						Voice cloning (ref audio)
					</span>
					{#if hasCustomClone && !currentVoice}
						<span class="text-[10px] text-foreground">Active</span>
					{/if}
				</div>
				<div class="px-2 pb-1.5">
					<TtsRefAudioPicker
						dataUri={refAudioUri}
						fileName={refAudioName}
						onChange={handleUpload}
					/>
					{#if hasCustomClone && currentVoice}
						<button
							type="button"
							class="mt-1.5 text-[11px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
							onclick={pickCustomClone}
						>
							Use uploaded clone instead of "{currentVoice}"
						</button>
					{/if}
				</div>
			</div>
		{/if}
	</Popover.Content>
</Popover.Root>
