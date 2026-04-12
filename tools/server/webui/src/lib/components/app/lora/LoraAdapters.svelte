<script lang="ts">
	import { untrack } from 'svelte';
	import { ChevronDown, Layers, Loader2 } from '@lucide/svelte';
	import * as Popover from '$lib/components/ui/popover';
	import { Button } from '$lib/components/ui/button';
	import { cn } from '$lib/components/ui/utils';
	import { loraStore } from '$lib/stores/lora.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		modelId?: string;
	}

	let { class: className = '', disabled = false, modelId }: Props = $props();

	let isOpen = $state(false);
	let showOnlyMatching = $state(true);
	let lastFetchedModelId: string | undefined = undefined;

	let activeCount = $derived(loraStore.adapters.filter((a) => a.enabled).length);

	/** Paths of adapters loaded for the current model, for highlighting in "all" view */
	let loadedPaths = $derived(new Set(loraStore.adapters.map((a) => a.path)));

	// Only re-fetch when modelId actually changes
	$effect(() => {
		const currentModelId = modelId;
		untrack(() => {
			if (currentModelId !== lastFetchedModelId) {
				lastFetchedModelId = currentModelId;
				loraStore.fetch(currentModelId);
			}
		});
	});
</script>

{#if loraStore.isAvailable}
	<Popover.Root bind:open={isOpen}>
		<Popover.Trigger
			{disabled}
			class={cn(
				'inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60',
				activeCount > 0 ? 'text-foreground' : 'text-muted-foreground',
				isOpen ? 'text-foreground' : ''
			)}
		>
			<Layers class="h-3.5 w-3.5" />

			<span class="font-medium">
				{#if activeCount === 0}
					LoRA
				{:else}
					LoRA ({activeCount})
				{/if}
			</span>

			{#if loraStore.applying}
				<Loader2 class="h-3 w-3.5 animate-spin" />
			{:else}
				<ChevronDown class="h-3 w-3.5" />
			{/if}
		</Popover.Trigger>

		<Popover.Content align="end" class="w-72 p-1.5">
			<div class="flex items-center justify-between px-2 py-1.5">
				<span class="text-xs font-semibold text-muted-foreground/60 select-none">
					LoRA Adapters
				</span>
			</div>

			{#if showOnlyMatching}
				{#if loraStore.adapters.length === 0}
					<div class="px-2 py-3 text-center text-xs text-muted-foreground">
						No matching adapters for this model.
					</div>
				{:else}
					{#each loraStore.adapters as adapter (adapter.id)}
						<div
							class={cn(
								'flex items-center gap-2 rounded-sm px-2 py-1.5 hover:bg-accent',
								adapter.enabled ? 'text-foreground' : 'text-muted-foreground'
							)}
						>
							<button
								type="button"
								class="flex shrink-0 cursor-pointer items-center"
								onclick={() => loraStore.toggle(adapter.id)}
							>
								<span
									class={cn(
										'flex h-4 w-4 items-center justify-center rounded-sm border',
										adapter.enabled
											? 'border-primary bg-primary text-primary-foreground'
											: 'border-muted-foreground/40'
									)}
								>
									{#if adapter.enabled}
										<svg class="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
											<polyline points="20 6 9 17 4 12" />
										</svg>
									{/if}
								</span>
							</button>

							<button
								type="button"
								class="min-w-0 flex-1 cursor-pointer truncate text-left text-xs font-medium"
								title={adapter.path}
								onclick={() => loraStore.toggle(adapter.id)}
							>
								{adapter.name}
							</button>

							<input
								type="number"
								min="0"
								max="2"
								step="0.1"
								value={adapter.enabled ? adapter.scale.toFixed(1) : '0.0'}
								disabled={!adapter.enabled}
								oninput={(e) => {
									const val = parseFloat(e.currentTarget.value);
									if (!isNaN(val)) {
										loraStore.setScale(adapter.id, Math.max(0, Math.min(2, val)));
									}
								}}
								class={cn(
									'h-6 w-14 shrink-0 rounded border bg-background px-1.5 text-right font-mono text-[11px] tabular-nums outline-none focus:border-primary focus:ring-1 focus:ring-primary',
									adapter.enabled
										? 'border-border text-foreground'
										: 'border-border/50 text-muted-foreground/60 cursor-not-allowed'
								)}
							/>
						</div>
					{/each}
				{/if}

				{#if loraStore.hasChanges}
					<div class="mt-1 border-t border-border/50 pt-1.5">
						<div class="flex justify-end px-2 py-0.5">
							<Button
								size="sm"
								variant="outline"
								class="h-6 px-3 text-xs"
								disabled={loraStore.applying}
								onclick={() => loraStore.applyChanges()}
							>
								{#if loraStore.applying}
									<Loader2 class="mr-1 h-3 w-3 animate-spin" />
								{/if}
								Apply
							</Button>
						</div>
					</div>
				{/if}
			{:else}
				{#if loraStore.allDiscovered.length === 0}
					<div class="px-2 py-3 text-center text-xs text-muted-foreground">
						No LoRA adapters discovered.
					</div>
				{:else}
					{#each loraStore.allDiscovered as adapter (adapter.path)}
						<div
							class={cn(
								'flex items-center gap-2 rounded-sm px-2 py-1.5 text-xs',
								loadedPaths.has(adapter.path)
									? 'text-foreground'
									: 'text-muted-foreground'
							)}
						>
							<span
								class={cn(
									'h-2 w-2 shrink-0 rounded-full',
									loadedPaths.has(adapter.path) ? 'bg-green-500' : 'bg-muted-foreground/30'
								)}
							></span>

							<span class="min-w-0 flex-1 truncate font-medium" title={adapter.path}>
								{adapter.name}
							</span>

							<span class="shrink-0 text-[10px] text-muted-foreground/60">
								{adapter.architecture}
							</span>
						</div>
					{/each}
				{/if}
			{/if}

			<div class="mt-1 border-t border-border/50 pt-1.5">
				<label class="flex cursor-pointer items-center gap-2 px-2 py-1">
					<input
						type="checkbox"
						bind:checked={showOnlyMatching}
						class="h-3.5 w-3.5 rounded border-muted-foreground/40 accent-primary"
					/>
					<span class="text-[11px] text-muted-foreground">Show only matching</span>
				</label>
			</div>
		</Popover.Content>
	</Popover.Root>
{/if}
