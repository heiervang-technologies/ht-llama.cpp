/**
 * docsStore — reactive state for markdown documents.
 *
 * Mirrors conversationsStore in shape but for DatabaseDoc entities.
 * Docs are a separate workspace mode from chat conversations.
 */

import { goto } from '$app/navigation';
import { toast } from 'svelte-sonner';
import { SvelteMap } from 'svelte/reactivity';
import { DatabaseService } from '$lib/services/database.service';
import type { DocEditorApi } from '$lib/components/app/doc/DocScreen/DocEditor.svelte';

const DEFAULT_DOC_NAME = 'Untitled';

/**
 * Map of docId → live DocEditor api for currently-mounted editors. The
 * ai-patch dispatcher consults this so it can paint streaming edits onto
 * the real CM6 view when one is mounted, and fall back to headless
 * string-replace when none is.
 *
 * `SvelteMap` rather than a plain `Map` both to satisfy the project's
 * lint rule and because a future UI that shows "ai-patch is editing this
 * doc" can subscribe to the registry without an extra reactive wrapper.
 */
const activeViews = new SvelteMap<string, DocEditorApi>();

class DocsStore {
	docs = $state<DatabaseDoc[]>([]);
	activeDoc = $state<DatabaseDoc | null>(null);
	isInitialized = $state(false);

	async initialize(): Promise<void> {
		if (this.isInitialized) return;
		try {
			this.docs = await DatabaseService.listDocs();
			this.isInitialized = true;
		} catch (error) {
			console.error('[docs] failed to load', error);
			toast.error('Failed to load documents');
		}
	}

	async refresh(): Promise<void> {
		this.docs = await DatabaseService.listDocs();
	}

	async createDoc(options?: { name?: string; content?: string; navigate?: boolean }) {
		const doc = await DatabaseService.createDoc(
			options?.name ?? DEFAULT_DOC_NAME,
			options?.content ?? ''
		);
		await this.refresh();
		if (options?.navigate !== false) {
			await goto(`#/doc/${doc.id}`);
		}
		return doc;
	}

	async loadDoc(id: string): Promise<DatabaseDoc | null> {
		const doc = await DatabaseService.getDoc(id);
		if (!doc) {
			this.activeDoc = null;
			return null;
		}
		this.activeDoc = doc;
		return doc;
	}

	clearActive(): void {
		this.activeDoc = null;
	}

	/**
	 * Live in-memory content update (no DB write). Used during streaming
	 * operations like AI commands so the editor can show tokens as they
	 * arrive without thrashing IndexedDB. Persist with updateContent when
	 * streaming completes.
	 */
	setContentLive(id: string, content: string): void {
		if (this.activeDoc?.id === id) {
			this.activeDoc = { ...this.activeDoc, content };
		}
	}

	async updateContent(id: string, content: string): Promise<void> {
		await DatabaseService.updateDoc(id, { content });
		if (this.activeDoc?.id === id) {
			this.activeDoc = { ...this.activeDoc, content, lastModified: Date.now() };
		}
		// Keep the list entry fresh for sidebar ordering.
		const idx = this.docs.findIndex((d) => d.id === id);
		if (idx >= 0) {
			const next = [...this.docs];
			next[idx] = { ...next[idx], content, lastModified: Date.now() };
			next.sort((a, b) => b.lastModified - a.lastModified);
			this.docs = next;
		}
	}

	async renameDoc(id: string, name: string): Promise<void> {
		const trimmed = name.trim() || DEFAULT_DOC_NAME;
		await DatabaseService.updateDoc(id, { name: trimmed });
		if (this.activeDoc?.id === id) {
			this.activeDoc = { ...this.activeDoc, name: trimmed };
		}
		await this.refresh();
	}

	/**
	 * Fork an existing doc into a new one with identical content. Useful for
	 * trying a variation without losing the original. Navigates to the copy
	 * unless navigate === false.
	 */
	async duplicateDoc(id: string, options?: { navigate?: boolean }): Promise<DatabaseDoc | null> {
		const source = await DatabaseService.getDoc(id);
		if (!source) return null;
		const copyName = `${source.name?.trim() || DEFAULT_DOC_NAME} (copy)`;
		return this.createDoc({
			name: copyName,
			content: source.content ?? '',
			navigate: options?.navigate ?? true
		});
	}

	async deleteDoc(id: string): Promise<void> {
		await DatabaseService.deleteDoc(id);
		if (this.activeDoc?.id === id) {
			this.activeDoc = null;
			await goto('#/');
		}
		await this.refresh();
	}

	/**
	 * Register the DocEditor instance that is currently mounted for `docId`.
	 * Called from DocEditor's `onMount`. A later mount for the same id
	 * replaces the previous registration — Svelte's keyed `{#key doc.id}`
	 * block in DocScreen unmounts the old editor before mounting the new
	 * one so this is strictly sequential in practice.
	 */
	registerActiveView(docId: string, api: DocEditorApi): void {
		activeViews.set(docId, api);
	}

	/** Tear down the registration for `docId`. Called from `onDestroy`. */
	unregisterActiveView(docId: string): void {
		activeViews.delete(docId);
	}

	/**
	 * Look up the currently-mounted DocEditor for `docId`, if any. The
	 * ai-patch dispatcher calls this to decide whether to attach a CM6
	 * bridge (mounted view → attach + paint widget) or fall back to a
	 * headless string-replace commit (no view → just updateContent).
	 */
	getActiveView(docId: string): DocEditorApi | undefined {
		return activeViews.get(docId);
	}
}

export const docsStore = new DocsStore();

export const docs = () => docsStore.docs;
export const activeDoc = () => docsStore.activeDoc;
export const isDocsInitialized = () => docsStore.isInitialized;
