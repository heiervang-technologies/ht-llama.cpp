<script lang="ts">
	import type { DatabaseArtifactRevision } from '$lib/types/database';
	import { RefreshCw, PencilLine, GitBranch, Sparkles, Pin } from '@lucide/svelte';

	interface Props {
		revisions: DatabaseArtifactRevision[];
		activeRevisionId: string | null;
		currentRevisionId: string;
		onSelect: (revisionId: string) => void;
		onPin?: (revisionId: string) => void;
	}

	let { revisions, activeRevisionId, currentRevisionId, onSelect, onPin }: Props = $props();

	const ICON = {
		initial: Sparkles,
		regenerate: RefreshCw,
		edit: PencilLine,
		fork: GitBranch
	} as const;
</script>

<ol class="flex flex-col gap-1">
	{#each [...revisions].reverse() as rev (rev.id)}
		{@const Ico = ICON[rev.reason]}
		{@const isActive = rev.id === activeRevisionId}
		{@const isPinned = rev.id === currentRevisionId}
		<li>
			<button
				type="button"
				class="flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-left text-xs transition hover:border-primary/60 {isActive
					? 'border-primary bg-primary/10'
					: 'border-transparent'}"
				onclick={() => onSelect(rev.id)}
			>
				<Ico class="h-3.5 w-3.5 flex-shrink-0 text-muted-foreground" />
				<div class="flex min-w-0 flex-1 flex-col">
					<span class="flex items-center gap-1 font-medium">
						rev {rev.revisionNumber}
						{#if isPinned}
							<Pin class="h-3 w-3 text-primary" aria-label="Current default" />
						{/if}
					</span>
					<span class="truncate text-[10px] text-muted-foreground">
						{rev.reason} · {new Date(rev.createdAt).toLocaleString()}
					</span>
				</div>
				{#if onPin && !isPinned}
					<!-- The "make default" button lives inside the list item button; using
						   a nested <button> would break semantics, so it's a span with its own
						   click + keyboard handling. -->
					<span
						role="button"
						tabindex="0"
						aria-label="Make default"
						class="rounded-sm px-1 text-[10px] text-muted-foreground hover:bg-accent hover:text-foreground"
						onclick={(e) => {
							e.stopPropagation();
							onPin(rev.id);
						}}
						onkeydown={(e) => {
							if (e.key === 'Enter' || e.key === ' ') {
								e.preventDefault();
								e.stopPropagation();
								onPin(rev.id);
							}
						}}>pin</span
					>
				{/if}
			</button>
		</li>
	{/each}
</ol>
