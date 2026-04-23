<script lang="ts">
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import { Button } from '$lib/components/ui/button';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import { artifactGalleryStore } from '$lib/stores/artifact-gallery.svelte';
	import { DatabaseService } from '$lib/services/database.service';
	import type { DatabaseArtifact, DatabaseArtifactRevision } from '$lib/types/database';
	import {
		ArrowLeft,
		Copy,
		Download,
		Trash2,
		MessageSquarePlus,
		PencilLine,
		Save,
		X
	} from '@lucide/svelte';
	import ArtifactRevisionList from './ArtifactRevisionList.svelte';
	import ArtifactRevisionPreview from './ArtifactRevisionPreview.svelte';

	interface Props {
		artifactId: string;
	}
	let { artifactId }: Props = $props();

	let artifact = $state<DatabaseArtifact | null>(null);
	let revisions = $state<DatabaseArtifactRevision[]>([]);
	let activeRevision = $state<DatabaseArtifactRevision | null>(null);
	let loading = $state(true);

	let editing = $state(false);
	let editDraft = $state('');
	let saving = $state(false);
	let renaming = $state(false);
	let renameDraft = $state('');

	const sidebar = useSidebar();

	async function refresh() {
		loading = true;
		try {
			const a = await DatabaseService.getArtifact(artifactId);
			if (!a) {
				artifact = null;
				revisions = [];
				activeRevision = null;
				return;
			}
			artifact = a;
			revisions = await DatabaseService.listArtifactRevisions(artifactId);
			// Preserve the user's selection when just a new revision was appended;
			// fall back to the artifact's pinned default otherwise.
			const keep = activeRevision?.id
				? revisions.find((r) => r.id === activeRevision!.id)
				: undefined;
			activeRevision = keep ?? revisions.find((r) => r.id === a.currentRevisionId) ?? revisions.at(-1) ?? null;
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		void artifactId;
		refresh();
	});

	function selectRevision(revId: string) {
		const rev = revisions.find((r) => r.id === revId);
		if (rev) activeRevision = rev;
	}

	async function pinRevision(revId: string) {
		await artifactGalleryStore.setCurrentRevision(artifactId, revId);
		await refresh();
	}

	async function handleDelete() {
		if (!artifact) return;
		if (!confirm(`Delete "${artifact.title}" and all ${revisions.length} revisions?`)) return;
		await artifactGalleryStore.remove(artifactId);
		goto('/artifacts');
	}

	async function handleRename() {
		if (!artifact || !renameDraft.trim()) {
			renaming = false;
			return;
		}
		await artifactGalleryStore.rename(artifactId, renameDraft.trim());
		renaming = false;
		await refresh();
	}

	function handleCopy() {
		if (!activeRevision) return;
		if (activeRevision.text) {
			void navigator.clipboard.writeText(activeRevision.text);
			toast.success('Revision copied to clipboard');
		} else if (activeRevision.blob) {
			// Clipboard API for arbitrary blobs is spotty across webviews; a
			// download nudge is a more reliable escape hatch for binary data.
			toast.message('Binary artifact — use Download instead');
		}
	}

	function handleDownload() {
		if (!artifact || !activeRevision) return;
		const ext = (
			activeRevision.mimeType.split('/')[1] || inferExt(artifact.kind) || 'bin'
		).split(';')[0];
		const filename = `${artifact.title.replace(/[^\w.-]+/g, '_') || 'artifact'}.rev${activeRevision.revisionNumber}.${ext}`;
		const blob =
			activeRevision.blob ??
			(activeRevision.text
				? new Blob([activeRevision.text], { type: activeRevision.mimeType || 'text/plain' })
				: null);
		if (!blob) return;
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = filename;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	}

	function inferExt(kind: DatabaseArtifact['kind']): string {
		switch (kind) {
			case 'html':
				return 'html';
			case 'svg':
				return 'svg';
			case 'markdown':
				return 'md';
			case 'code':
				return 'txt';
			case 'image':
				return 'png';
			case 'audio':
				return 'wav';
			case 'video':
				return 'mp4';
			case 'pdf':
				return 'pdf';
		}
	}

	function beginEdit() {
		if (!activeRevision?.text) {
			toast.error('Binary artifacts are not editable here — download to edit externally.');
			return;
		}
		editDraft = activeRevision.text;
		editing = true;
	}

	async function saveEdit() {
		if (!artifact || !activeRevision) return;
		if (editDraft === activeRevision.text) {
			editing = false;
			return;
		}
		saving = true;
		try {
			await artifactGalleryStore.addUserEditRevision(artifactId, {
				kind: artifact.kind,
				title: artifact.title,
				mimeType: activeRevision.mimeType,
				text: editDraft
			});
			editing = false;
			await refresh();
			toast.success('New revision saved');
		} catch (err) {
			toast.error(`Save failed: ${err instanceof Error ? err.message : String(err)}`);
		} finally {
			saving = false;
		}
	}

	function openInChat() {
		if (!artifact?.sourceConversationId) return;
		goto(`/chat/${artifact.sourceConversationId}`);
	}
</script>

<div
	class="flex h-full flex-col duration-200 ease-linear {sidebar.open
		? 'md:ml-[var(--sidebar-width)]'
		: ''}"
>
	<header
		class="sticky top-0 z-20 flex items-center gap-2 border-b bg-background/80 p-3 backdrop-blur md:p-4"
	>
		<Button variant="ghost" size="sm" onclick={() => goto('/artifacts')}>
			<ArrowLeft class="h-4 w-4" />
			<span class="hidden md:inline">Gallery</span>
		</Button>

		<div class="min-w-0 flex-1">
			{#if renaming && artifact}
				<input
					type="text"
					class="w-full rounded-md border bg-background px-2 py-1 text-sm font-medium outline-none focus:ring-2 focus:ring-ring"
					bind:value={renameDraft}
					onblur={handleRename}
					onkeydown={(e) => {
						if (e.key === 'Enter') {
							e.preventDefault();
							(e.currentTarget as HTMLInputElement).blur();
						} else if (e.key === 'Escape') {
							e.preventDefault();
							renaming = false;
						}
					}}
				/>
			{:else if artifact}
				<button
					type="button"
					class="block w-full truncate rounded-md px-2 py-1 text-left text-sm font-medium hover:bg-muted/60"
					onclick={() => {
						renameDraft = artifact!.title;
						renaming = true;
					}}
					title="Rename"
				>
					{artifact.title}
				</button>
			{/if}
			{#if artifact}
				<p class="px-2 text-xs text-muted-foreground">
					{artifact.kind} · {revisions.length}
					{revisions.length === 1 ? 'revision' : 'revisions'}
					· updated {new Date(artifact.updatedAt).toLocaleString()}
				</p>
			{/if}
		</div>

		<div class="flex items-center gap-1">
			{#if artifact?.sourceConversationId}
				<Button variant="ghost" size="sm" onclick={openInChat} title="Open source chat">
					<MessageSquarePlus class="h-4 w-4" />
				</Button>
			{/if}
			<Button variant="ghost" size="sm" onclick={handleCopy} title="Copy text">
				<Copy class="h-4 w-4" />
			</Button>
			<Button variant="ghost" size="sm" onclick={handleDownload} title="Download revision">
				<Download class="h-4 w-4" />
			</Button>
			<Button variant="ghost" size="sm" onclick={beginEdit} title="Edit as new revision">
				<PencilLine class="h-4 w-4" />
			</Button>
			<Button
				variant="ghost"
				size="sm"
				onclick={handleDelete}
				class="text-destructive hover:text-destructive"
				title="Delete artifact"
			>
				<Trash2 class="h-4 w-4" />
			</Button>
		</div>
	</header>

	<section class="flex flex-1 overflow-hidden">
		<main class="flex min-w-0 flex-1 flex-col gap-3 p-3 md:p-5">
			{#if loading}
				<p class="text-sm text-muted-foreground">Loading…</p>
			{:else if !artifact || !activeRevision}
				<div class="flex h-full flex-col items-center justify-center gap-2 text-center">
					<p class="text-sm text-muted-foreground">Artifact not found.</p>
					<Button variant="outline" size="sm" onclick={() => goto('/artifacts')}>
						Back to gallery
					</Button>
				</div>
			{:else if editing}
				<textarea
					bind:value={editDraft}
					class="h-full w-full flex-1 rounded-lg border bg-card p-3 font-mono text-sm outline-none focus:ring-2 focus:ring-ring"
					spellcheck="false"
				></textarea>
				<div class="flex justify-end gap-2">
					<Button
						variant="ghost"
						size="sm"
						onclick={() => (editing = false)}
						disabled={saving}
					>
						<X class="h-4 w-4" />
						Cancel
					</Button>
					<Button size="sm" onclick={saveEdit} disabled={saving}>
						<Save class="h-4 w-4" />
						{saving ? 'Saving…' : 'Save as new revision'}
					</Button>
				</div>
			{:else}
				<ArtifactRevisionPreview {artifact} revision={activeRevision} />
			{/if}
		</main>

		<aside class="hidden w-60 flex-shrink-0 overflow-y-auto border-l p-3 md:block">
			<h2 class="mb-2 text-xs font-medium uppercase text-muted-foreground">Revisions</h2>
			{#if artifact}
				<ArtifactRevisionList
					{revisions}
					activeRevisionId={activeRevision?.id ?? null}
					currentRevisionId={artifact.currentRevisionId}
					onSelect={selectRevision}
					onPin={pinRevision}
				/>
			{/if}
		</aside>
	</section>
</div>
