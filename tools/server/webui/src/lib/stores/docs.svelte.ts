/**
 * docsStore — reactive state for markdown documents.
 *
 * Mirrors conversationsStore in shape but for DatabaseDoc entities.
 * Docs are a separate workspace mode from chat conversations.
 */

import { goto } from '$app/navigation';
import { toast } from 'svelte-sonner';
import { DatabaseService } from '$lib/services/database.service';

const DEFAULT_DOC_NAME = 'Untitled';

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

	async deleteDoc(id: string): Promise<void> {
		await DatabaseService.deleteDoc(id);
		if (this.activeDoc?.id === id) {
			this.activeDoc = null;
			await goto('#/');
		}
		await this.refresh();
	}
}

export const docsStore = new DocsStore();

export const docs = () => docsStore.docs;
export const activeDoc = () => docsStore.activeDoc;
export const isDocsInitialized = () => docsStore.isInitialized;
