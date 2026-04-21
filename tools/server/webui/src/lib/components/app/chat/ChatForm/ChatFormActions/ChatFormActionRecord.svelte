<script lang="ts">
	import { Mic, Square } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		class?: string;
		disabled?: boolean;
		hasAudioModality?: boolean;
		sttReady?: boolean;
		isLoading?: boolean;
		isRecording?: boolean;
		onMicClick?: () => void;
	}

	let {
		class: className = '',
		disabled = false,
		hasAudioModality = false,
		sttReady = false,
		isLoading = false,
		isRecording = false,
		onMicClick
	}: Props = $props();

	// The mic is useful in two modes: (1) the active LLM accepts audio natively,
	// or (2) the user has STT/ASR configured, so we can record → transcribe →
	// drop the text into the textarea. Either path is valid.
	let canRecord = $derived(hasAudioModality || sttReady);

	let tooltipText = $derived.by(() => {
		if (canRecord) return '';
		return 'Enable speech-to-text in Settings, or pick a model with audio modality.';
	});
</script>

<div class="flex items-center gap-1 {className}">
	<Tooltip.Root>
		<Tooltip.Trigger>
			<Button
				class="h-8 w-8 rounded-full p-0 {isRecording
					? 'animate-pulse bg-red-500 text-white hover:bg-red-600'
					: ''}"
				disabled={disabled || isLoading || !canRecord}
				onclick={onMicClick}
				type="button"
			>
				<span class="sr-only">{isRecording ? 'Stop recording' : 'Start recording'}</span>

				{#if isRecording}
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
