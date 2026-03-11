<script lang="ts">
	import { onMount } from 'svelte';
	import { ChevronDown, Layers, Loader2 } from '@lucide/svelte';
	import { Switch } from '$lib/components/ui/switch';
	import { Button } from '$lib/components/ui/button';
	import { loraStore } from '$lib/stores/lora.svelte';

	interface Props {
		class?: string;
		modelId?: string;
	}

	let { class: className = '', modelId }: Props = $props();

	let isOpen = $state(false);

	onMount(() => {
		loraStore.fetch(modelId);
	});

	$effect(() => {
		// Re-fetch when model changes (e.g. user switches model in router mode)
		if (modelId) {
			loraStore.fetch(modelId);
		}
	});
</script>

{#if loraStore.isAvailable}
	<div class={className}>
		<button
			type="button"
			class="inline-flex w-full cursor-pointer items-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-muted-foreground transition hover:bg-muted hover:text-foreground"
			onclick={() => (isOpen = !isOpen)}
		>
			<Layers class="h-3.5 w-3.5" />
			<span class="font-medium">LoRA Adapters</span>
			<span class="ml-auto text-[10px] tabular-nums text-muted-foreground/60">
				{loraStore.adapters.filter((a) => a.enabled).length}/{loraStore.adapters.length}
			</span>
			<ChevronDown
				class="h-3 w-3 transition-transform {isOpen ? 'rotate-180' : ''}"
			/>
		</button>

		{#if isOpen}
			<div class="mt-1 space-y-1.5 rounded-md border border-border/50 bg-muted/30 p-2">
				{#each loraStore.adapters as adapter (adapter.id)}
					<div class="flex items-center gap-2 rounded-sm px-1 py-1">
						<Switch
							checked={adapter.enabled}
							onCheckedChange={() => loraStore.toggle(adapter.id)}
						/>

						<div class="min-w-0 flex-1">
							<span
								class="block truncate text-xs font-medium {adapter.enabled
									? 'text-foreground'
									: 'text-muted-foreground'}"
								title={adapter.path}
							>
								{adapter.name}
							</span>
						</div>

						<div class="flex shrink-0 items-center gap-1.5">
							<input
								type="range"
								min="0"
								max="2"
								step="0.1"
								value={adapter.enabled ? adapter.scale : 0}
								disabled={!adapter.enabled}
								oninput={(e) => {
									const val = parseFloat(e.currentTarget.value);
									loraStore.setScale(adapter.id, val);
								}}
								class="h-1 w-16 cursor-pointer appearance-none rounded-full bg-border accent-primary disabled:cursor-not-allowed disabled:opacity-40"
							/>
							<span
								class="w-7 text-right font-mono text-[10px] tabular-nums {adapter.enabled
									? 'text-foreground'
									: 'text-muted-foreground/60'}"
							>
								{adapter.enabled ? adapter.scale.toFixed(1) : '0.0'}
							</span>
						</div>
					</div>
				{/each}

				{#if loraStore.hasChanges}
					<div class="flex justify-end pt-1">
						<Button
							size="sm"
							variant="outline"
							class="h-6 px-2 text-xs"
							disabled={loraStore.applying}
							onclick={() => loraStore.applyChanges()}
						>
							{#if loraStore.applying}
								<Loader2 class="mr-1 h-3 w-3 animate-spin" />
							{/if}
							Apply
						</Button>
					</div>
				{/if}
			</div>
		{/if}
	</div>
{/if}
