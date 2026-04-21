<script lang="ts">
	import { Loader2, Mic, Square } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		sttReady?: boolean;
		isLoading?: boolean;
		isRecording?: boolean;
		isTranscribing?: boolean;
		onMicClick?: () => void;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		sttReady = false,
		isLoading = false,
		isRecording = false,
		isTranscribing = false,
		onMicClick
	}: Props = $props();

	// The mic is useful in two modes: (1) the active LLM accepts audio natively,
	// or (2) the user has STT/ASR configured, so we can record → transcribe →
	// drop the text into the textarea. Either path is valid.
	let canRecord = $derived(hasAudioModality || sttReady);

	let tooltipText = $derived.by(() => {
		if (isTranscribing) return 'Transcribing…';
		if (canRecord) return '';
		return 'Enable speech-to-text in Settings, or pick a model with audio modality.';
	});

	let srLabel = $derived.by(() => {
		if (isTranscribing) return 'Transcribing audio';
		if (isRecording) return 'Stop recording';
		return 'Start recording';
	});
</script>

<div class="flex items-center gap-1 {className}">
	<Tooltip.Root>
		<Tooltip.Trigger>
			<Button
				class="h-8 w-8 rounded-full p-0 {isTranscribing
					? 'bg-blue-500 text-white hover:bg-blue-600'
					: isRecording
						? 'animate-pulse bg-red-500 text-white hover:bg-red-600'
						: ''}"
				disabled={disabled || isLoading || isTranscribing || !canRecord}
				onclick={onMicClick}
				type="button"
			>
				<span class="sr-only">{srLabel}</span>

				{#if isTranscribing}
					<Loader2 class="h-4 w-4 animate-spin" />
				{:else if isRecording}
					<Square class="h-4 w-4 animate-pulse fill-white" />
				{:else}
					<Mic class="h-4 w-4" />
				{/if}
			</Button>
		</Tooltip.Trigger>

		{#if tooltipText}
			<Tooltip.Content>
				<p>{tooltipText}</p>
			</Tooltip.Content>
		{/if}
	</Tooltip.Root>
</div>
