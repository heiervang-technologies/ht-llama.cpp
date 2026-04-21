<script lang="ts">
	import { Copy, Download, MoreHorizontal, Trash2 } from '@lucide/svelte';
	import { toast } from 'svelte-sonner';
	import { Button } from '$lib/components/ui/button';
	import {
		DropdownMenu,
		DropdownMenuContent,
		DropdownMenuItem,
		DropdownMenuSeparator,
		DropdownMenuTrigger
	} from '$lib/components/ui/dropdown-menu';
	import { docsStore } from '$lib/stores/docs.svelte';

	interface Props {
		docId: string;
		docName: string;
		docContent: string;
		onDelete: () => void;
		open?: boolean;
	}

	let { docId, docName, docContent, onDelete, open = $bindable(false) }: Props = $props();

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

	async function duplicate() {
		open = false;
		try {
			// Navigates to the copy on success; the original stays in the sidebar.
			await docsStore.duplicateDoc(docId);
		} catch (err) {
			console.error('[doc] duplicate failed', err);
			const msg = err instanceof Error ? err.message : String(err);
			toast.error(`Could not duplicate: ${msg}`);
		}
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
		<DropdownMenuItem onclick={duplicate}>
			<Copy class="h-4 w-4" />
			<span>Duplicate</span>
		</DropdownMenuItem>
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
