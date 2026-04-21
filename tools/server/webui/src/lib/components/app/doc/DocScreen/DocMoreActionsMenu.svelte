<script lang="ts">
	import { Download, MoreHorizontal, Trash2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import {
		DropdownMenu,
		DropdownMenuContent,
		DropdownMenuItem,
		DropdownMenuSeparator,
		DropdownMenuTrigger
	} from '$lib/components/ui/dropdown-menu';

	interface Props {
		docName: string;
		docContent: string;
		onDelete: () => void;
		open?: boolean;
	}

	let { docName, docContent, onDelete, open = $bindable(false) }: Props = $props();

	// Strip path separators and other filename-hostile characters. The browser's
	// download attribute already enforces its own sanitization per-platform, but
	// keeping the offered name clean avoids accidental directory traversal on
	// uncommon browsers and keeps the suggested filename readable.
	function toFilename(name: string): string {
		const trimmed = (name ?? '').trim() || 'Untitled';
		const safe = trimmed.replace(/[\\/:*?"<>|]+/g, '-');
		return `${safe}.md`;
	}

	function downloadMarkdown() {
		const blob = new Blob([docContent ?? ''], { type: 'text/markdown;charset=utf-8' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = toFilename(docName);
		document.body.appendChild(a);
		a.click();
		a.remove();
		URL.revokeObjectURL(url);
		open = false;
	}

	function handleDelete() {
		open = false;
		onDelete();
	}
</script>

<DropdownMenu bind:open>
	<DropdownMenuTrigger>
		{#snippet child({ props })}
			<Button
				{...props}
				variant="ghost"
				size="icon-lg"
				class="rounded-full backdrop-blur-lg"
				title="More actions"
			>
				<MoreHorizontal class="h-4 w-4" />
				<span class="sr-only">More actions</span>
			</Button>
		{/snippet}
	</DropdownMenuTrigger>

	<DropdownMenuContent align="end" class="w-48">
		<DropdownMenuItem onclick={downloadMarkdown}>
			<Download class="h-4 w-4" />
			<span>Download .md</span>
		</DropdownMenuItem>
		<DropdownMenuSeparator />
		<DropdownMenuItem variant="destructive" onclick={handleDelete}>
			<Trash2 class="h-4 w-4" />
			<span>Delete document</span>
		</DropdownMenuItem>
	</DropdownMenuContent>
</DropdownMenu>
