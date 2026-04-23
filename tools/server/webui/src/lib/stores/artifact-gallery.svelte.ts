/**
 * artifactGalleryStore — persisted cross-session artifact library.
 *
 * Separate from `artifactsStore` (the ephemeral right-drawer for the active
 * chat turn). This one is backed by IndexedDB via DatabaseService and drives
 * the /artifacts gallery + detail routes. Auto-capture code + explicit
 * "save to gallery" actions both funnel through `captureFromChat` /
 * `saveManual`.
 */

import { hashString } from '$lib/utils/artifacts';
import { DatabaseService } from '$lib/services/database.service';
import type {
	DatabaseArtifact,
	DatabaseArtifactKind,
	DatabaseArtifactRevision
} from '$lib/types/database';

export interface CapturePayload {
	kind: DatabaseArtifactKind;
	title: string;
	mimeType: string;
	/** Text-like payload (html / svg / code / markdown). */
	text?: string;
	/** Binary payload (image / audio / video / pdf). */
	blob?: Blob;
	summary?: string;
	tags?: string[];
	metadata?: Record<string, unknown>;
}

export interface CaptureSource {
	conversationId: string;
	/** Stable per-turn slot id — usually `${messageId}#${artifactIndex}` for
	 *  chat auto-capture. Used to find the existing artifact chain on
	 *  regeneration so we append a revision instead of making a twin. */
	slot: string;
	messageId?: string;
	reason: 'initial' | 'regenerate' | 'edit';
}

class ArtifactGalleryStore {
	artifacts = $state<DatabaseArtifact[]>([]);
	loaded = $state(false);
	loading = $state(false);

	async load(): Promise<void> {
		if (this.loading) return;
		this.loading = true;
		try {
			this.artifacts = await DatabaseService.listArtifacts();
			this.loaded = true;
		} finally {
			this.loading = false;
		}
	}

	/**
	 * Captures an artifact from a chat turn. If a prior artifact exists for
	 * the same (conversation, slot), appends a revision. Deduplicates on
	 * content hash so a no-op streaming update doesn't bloat the timeline.
	 */
	async captureFromChat(
		source: CaptureSource,
		payload: CapturePayload
	): Promise<DatabaseArtifact | null> {
		const contentHash = await hashPayload(payload);
		const existing = await DatabaseService.findArtifactBySlot(source.conversationId, source.slot);

		if (existing) {
			const revs = await DatabaseService.listArtifactRevisions(existing.id);
			const latest = revs.at(-1);
			if (latest?.contentHash === contentHash) {
				// Same bytes as the latest revision — nothing to record.
				return existing;
			}
			await DatabaseService.appendArtifactRevision(existing.id, {
				reason: source.reason === 'initial' ? 'regenerate' : source.reason,
				parentRevisionId: latest?.id,
				contentHash,
				mimeType: payload.mimeType,
				text: payload.text,
				blob: payload.blob,
				sourceMessageId: source.messageId,
				metadata: payload.metadata
			});
			// Title/summary only update on regenerate when the new revision
			// produced a different title — otherwise keep the first one so
			// the gallery label stays stable across retries.
			const patch: Partial<DatabaseArtifact> = {};
			if (payload.title && payload.title !== existing.title && source.reason === 'regenerate') {
				patch.title = payload.title;
			}
			if (Object.keys(patch).length > 0) {
				await DatabaseService.updateArtifact(existing.id, patch);
			}
			await this.load();
			return (await DatabaseService.getArtifact(existing.id)) ?? null;
		}

		const { artifact } = await DatabaseService.createArtifact(
			{
				title: payload.title,
				kind: payload.kind,
				tags: payload.tags ?? [],
				sourceConversationId: source.conversationId,
				sourceMessageSlot: source.slot,
				summary: payload.summary
			},
			{
				contentHash,
				mimeType: payload.mimeType,
				text: payload.text,
				blob: payload.blob,
				sourceMessageId: source.messageId,
				metadata: payload.metadata,
				reason: 'initial'
			}
		);
		await this.load();
		return artifact;
	}

	/**
	 * Saves an artifact with no chat source — e.g. a manual upload or a
	 * "save this image to gallery" action on a pasted attachment.
	 */
	async saveManual(payload: CapturePayload): Promise<DatabaseArtifact> {
		const contentHash = await hashPayload(payload);
		const { artifact } = await DatabaseService.createArtifact(
			{
				title: payload.title,
				kind: payload.kind,
				tags: payload.tags ?? [],
				summary: payload.summary
			},
			{
				contentHash,
				mimeType: payload.mimeType,
				text: payload.text,
				blob: payload.blob,
				metadata: payload.metadata,
				reason: 'initial'
			}
		);
		await this.load();
		return artifact;
	}

	/**
	 * Creates a new revision by hand — e.g. user opens an artifact, edits the
	 * HTML/SVG/code in place, saves.
	 */
	async addUserEditRevision(
		artifactId: string,
		payload: CapturePayload
	): Promise<DatabaseArtifactRevision> {
		const contentHash = await hashPayload(payload);
		const revs = await DatabaseService.listArtifactRevisions(artifactId);
		const latest = revs.at(-1);
		const rev = await DatabaseService.appendArtifactRevision(artifactId, {
			reason: 'edit',
			parentRevisionId: latest?.id,
			contentHash,
			mimeType: payload.mimeType,
			text: payload.text,
			blob: payload.blob,
			metadata: payload.metadata
		});
		await this.load();
		return rev;
	}

	async setCurrentRevision(artifactId: string, revisionId: string): Promise<void> {
		await DatabaseService.updateArtifact(artifactId, { currentRevisionId: revisionId });
		await this.load();
	}

	async rename(artifactId: string, title: string): Promise<void> {
		await DatabaseService.updateArtifact(artifactId, { title });
		await this.load();
	}

	async remove(artifactId: string): Promise<void> {
		await DatabaseService.deleteArtifact(artifactId);
		await this.load();
	}
}

async function hashPayload(p: CapturePayload): Promise<string> {
	if (p.text != null) return hashString(p.text);
	if (p.blob) {
		const buf = await p.blob.arrayBuffer();
		// SubtleCrypto is everywhere we ship. Using SHA-1 is fine — this is
		// a local dedup key, not a security property.
		const digest = await crypto.subtle.digest('SHA-1', buf);
		return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
	}
	return '';
}

export const artifactGalleryStore = new ArtifactGalleryStore();
