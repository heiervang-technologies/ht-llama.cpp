<script lang="ts">
	import * as Sheet from '$lib/components/ui/sheet';
	import { Button } from '$lib/components/ui/button';
	import { Copy, Code, Eye, ExternalLink, Download } from '@lucide/svelte';
	import { artifactsStore, type ArtifactEntry } from '$lib/stores/artifacts.svelte';
	import { copyToClipboard } from '$lib/utils';

	let view = $state<'preview' | 'source'>('preview');
	let entries = $derived(artifactsStore.entries);
	let active = $derived(artifactsStore.active);

	// Reset to preview whenever the active artifact changes, so switching artifacts
	// does not strand the drawer in source view from a prior selection.
	$effect(() => {
		void active?.id;
		view = 'preview';
	});

	function handleOpenChange(open: boolean) {
		if (open) artifactsStore.show();
		else artifactsStore.close();
	}

	function srcdocFor(entry: ArtifactEntry): string {
		if (entry.kind === 'svg') {
			return `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;padding:0;height:100%;display:flex;align-items:center;justify-content:center;background:transparent}svg{max-width:100%;max-height:100%}</style></head><body>${entry.content}</body></html>`;
		}
		return entry.content;
	}

	function copySource() {
		if (!active) return;
		void copyToClipboard(active.content);
	}

	function openInNewTab() {
		if (!active) return;
		const blob = new Blob([srcdocFor(active)], { type: 'text/html;charset=utf-8' });
		const url = URL.createObjectURL(blob);
		window.open(url, '_blank', 'noopener');
		// Revoke later so the new tab has time to load.
		setTimeout(() => URL.revokeObjectURL(url), 60_000);
	}

	function download() {
		if (!active) return;
		const ext = active.kind === 'svg' ? 'svg' : 'html';
		const blob = new Blob([active.content], {
			type: active.kind === 'svg' ? 'image/svg+xml' : 'text/html;charset=utf-8'
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `${active.title.replace(/[^a-z0-9-_]+/gi, '_') || 'artifact'}.${ext}`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		setTimeout(() => URL.revokeObjectURL(url), 10_000);
	}
</script>

<Sheet.Root open={artifactsStore.open} onOpenChange={handleOpenChange}>
	<Sheet.Content side="right" class="flex w-full flex-col p-0 sm:max-w-2xl md:max-w-3xl">
		<Sheet.Header class="gap-1 border-b border-border/40 px-4 py-3">
			<Sheet.Title class="pr-8 text-base font-medium">
				{active?.title ?? 'Artifacts'}
			</Sheet.Title>
			<Sheet.Description class="text-xs text-muted-foreground">
				{#if active}
					{active.kind.toUpperCase()} · {active.content.length.toLocaleString()} chars
				{:else}
					No artifacts yet
				{/if}
			</Sheet.Description>
		</Sheet.Header>

		{#if entries.length > 1}
			<div class="flex gap-1 overflow-x-auto border-b border-border/40 px-4 py-2">
				{#each entries as entry (entry.id)}
					<button
						type="button"
						class="flex-shrink-0 rounded-md border border-border/40 px-2 py-1 text-xs transition-colors {active?.id ===
						entry.id
							? 'bg-accent text-accent-foreground'
							: 'text-muted-foreground hover:bg-accent/40'}"
						onclick={() => artifactsStore.activate(entry.id)}
					>
						{entry.title}
					</button>
				{/each}
			</div>
		{/if}

		<div class="flex items-center justify-between border-b border-border/40 px-4 py-2">
			<div class="flex gap-1">
				<Button
					variant={view === 'preview' ? 'secondary' : 'ghost'}
					size="sm"
					onclick={() => (view = 'preview')}
					disabled={!active}
				>
					<Eye class="mr-1 h-3.5 w-3.5" /> Preview
				</Button>
				<Button
					variant={view === 'source' ? 'secondary' : 'ghost'}
					size="sm"
					onclick={() => (view = 'source')}
					disabled={!active}
				>
					<Code class="mr-1 h-3.5 w-3.5" /> Source
				</Button>
			</div>
			<div class="flex gap-1">
				<Button
					variant="ghost"
					size="sm"
					onclick={copySource}
					disabled={!active}
					title="Copy source"
				>
					<Copy class="h-3.5 w-3.5" />
				</Button>
				<Button variant="ghost" size="sm" onclick={download} disabled={!active} title="Download">
					<Download class="h-3.5 w-3.5" />
				</Button>
				<Button
					variant="ghost"
					size="sm"
					onclick={openInNewTab}
					disabled={!active}
					title="Open in new tab"
				>
					<ExternalLink class="h-3.5 w-3.5" />
				</Button>
			</div>
		</div>

		<div class="min-h-0 flex-1 overflow-hidden bg-background">
			{#if !active}
				<div class="flex h-full items-center justify-center p-8 text-sm text-muted-foreground">
					Nothing to show yet. HTML and SVG blocks from assistant messages will appear here.
				</div>
			{:else if view === 'preview'}
				<iframe
					title={active.title}
					class="h-full w-full border-0 bg-white"
					sandbox="allow-scripts allow-forms allow-popups allow-modals"
					srcdoc={srcdocFor(active)}
				></iframe>
			{:else}
				<pre
					class="h-full w-full overflow-auto bg-muted/30 p-4 font-mono text-xs leading-relaxed"><code
						>{active.content}</code
					></pre>
			{/if}
		</div>
	</Sheet.Content>
</Sheet.Root>
