/**
 * artifactGalleryStore — persisted cross-session artifact library.
 *
 * Separate from `artifactsStore` (the ephemeral right-drawer for the active
 * chat turn). This one is backed by IndexedDB via DatabaseService and drives
 * the /artifacts gallery + detail routes. Auto-capture code + explicit
 * "save to gallery" actions both funnel through `captureFromChat` /
 * `saveManual`.
 */

import { SvelteMap, SvelteSet } from 'svelte/reactivity';
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
	/** Runs once per process to clean up same-slot duplicates left over from
	 *  the pre-lock race bug. Idempotent. */
	private dedupedOnce = false;

	/**
	 * Per-slot in-flight promise so concurrent `captureFromChat` calls for the
	 * same (conversation, slot) serialize. Without this, two racers both see
	 * `findArtifactBySlot → undefined` before either `createArtifact` commits,
	 * and end up inserting twin artifacts for the same content. The race is
	 * real: a remounting ChatMessageAssistant, an HMR reload mid-stream, or a
	 * scroll-virtualized re-render can all trigger overlapping captures.
	 */
	private slotLocks = new SvelteMap<string, Promise<DatabaseArtifact | null>>();

	async load(): Promise<void> {
		if (this.loading) return;
		this.loading = true;
		try {
			if (!this.dedupedOnce) {
				this.dedupedOnce = true;
				await this.dedupSameSlotArtifacts();
			}
			this.artifacts = await DatabaseService.listArtifacts();
			this.loaded = true;
		} finally {
			this.loading = false;
		}
	}

	/**
	 * One-shot cleanup for the pre-fix race bug that left multiple artifacts
	 * sharing the same `(sourceConversationId, sourceMessageSlot)` key.
	 *
	 * Strategy: group by slot, keep the earliest artifact in each group, fold
	 * every other sibling's revisions into the keeper as fresh revisions
	 * (reason=regenerate so they're visually distinct from genuine edits),
	 * then delete the siblings. Does nothing when no groups have duplicates,
	 * so re-running is cheap.
	 */
	async dedupSameSlotArtifacts(): Promise<number> {
		const all = await DatabaseService.listArtifacts();
		const groups = new SvelteMap<string, DatabaseArtifact[]>();
		for (const a of all) {
			if (!a.sourceConversationId || !a.sourceMessageSlot) continue;
			const key = `${a.sourceConversationId}::${a.sourceMessageSlot}`;
			const arr = groups.get(key);
			if (arr) arr.push(a);
			else groups.set(key, [a]);
		}
		let merged = 0;
		for (const group of groups.values()) {
			if (group.length < 2) continue;
			// Earliest createdAt is the keeper; stable within the group.
			group.sort((x, y) => x.createdAt - y.createdAt);
			const keeper = group[0];
			const dupes = group.slice(1);
			const keeperRevs = await DatabaseService.listArtifactRevisions(keeper.id);
			const seenHashes = new SvelteSet(keeperRevs.map((r) => r.contentHash));
			for (const dupe of dupes) {
				const dupeRevs = await DatabaseService.listArtifactRevisions(dupe.id);
				for (const rev of dupeRevs) {
					// Skip revisions we already have (deduped by content hash).
					if (seenHashes.has(rev.contentHash)) continue;
					seenHashes.add(rev.contentHash);
					await DatabaseService.appendArtifactRevision(keeper.id, {
						reason: 'regenerate',
						contentHash: rev.contentHash,
						mimeType: rev.mimeType,
						text: rev.text,
						blob: rev.blob,
						sourceMessageId: rev.sourceMessageId,
						metadata: { ...(rev.metadata ?? {}), mergedFromArtifactId: dupe.id }
					});
				}
				await DatabaseService.deleteArtifact(dupe.id);
				merged++;
			}
		}
		return merged;
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
		const lockKey = `${source.conversationId}::${source.slot}`;
		const inFlight = this.slotLocks.get(lockKey);
		// Chain onto any in-flight capture for this slot so the two calls see
		// each other's writes (the second call will hit the `existing` branch
		// once the first commits).
		const run = (inFlight ?? Promise.resolve(null)).then(() => this.#captureImpl(source, payload));
		this.slotLocks.set(lockKey, run);
		try {
			return await run;
		} finally {
			// Only clear if nobody else has taken the lock in the meantime.
			if (this.slotLocks.get(lockKey) === run) this.slotLocks.delete(lockKey);
		}
	}

	async #captureImpl(
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
				summary: payload.summary,
				metadata: payload.metadata
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
		// Fire-and-forget Nextcloud auto-upload — same gate as saveManual.
		const { maybeAutoUpload } = await import('$lib/services/nextcloud-upload.service');
		void maybeAutoUpload(artifact);
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
				summary: payload.summary,
				metadata: payload.metadata
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
		// Fire-and-forget Nextcloud auto-upload. The function short-circuits
		// when the connection isn't configured, when auto-upload is off,
		// or when the artifact came *from* Nextcloud — so it's safe to
		// always call.
		const { maybeAutoUpload } = await import('$lib/services/nextcloud-upload.service');
		void maybeAutoUpload(artifact);
		return artifact;
	}

	/**
	 * Creates a new revision by hand — e.g. user opens an artifact, edits the
	 * HTML/SVG/code in place, saves.
	 *
	 * `opts.parentRevisionId` pins the parent to a specific revision rather
	 * than the latest-at-write-time one. The ai-patch dispatcher uses this
	 * to snapshot the revision that was current when the patch *session
	 * started*, so concurrent edits during streaming don't silently re-parent
	 * the new revision onto whatever happens to be current at commit time.
	 * Callers that omit it keep the original behaviour.
	 *
	 * Dedup rule (must hold regardless of `opts.parentRevisionId`):
	 * - If the latest revision's contentHash equals the new hash AND the
	 *   caller's pinned parent is either omitted or equal to the latest
	 *   revision's id, we've produced a byte-for-byte duplicate of the
	 *   current tip — return the existing revision instead of appending.
	 * - If the caller pinned to an older revision and the new hash matches
	 *   the tip, we still append (the edit represents a divergent branch
	 *   that coincidentally arrived at the current tip text, and the
	 *   revision timeline should record that).
	 */
	async addUserEditRevision(
		artifactId: string,
		payload: CapturePayload,
		opts?: { parentRevisionId?: string }
	): Promise<DatabaseArtifactRevision> {
		const contentHash = await hashPayload(payload);
		const revs = await DatabaseService.listArtifactRevisions(artifactId);
		const latest = revs.at(-1);

		const pinnedParent = opts?.parentRevisionId;
		const effectiveParent = pinnedParent ?? latest?.id;

		// Short-circuit when the new content matches the tip and the caller
		// is parented off the tip (either by omission or by explicit pin).
		// Without this guard, an ai-patch session that "no-ops" — e.g. a
		// block matched but REPLACE was identical to SEARCH — would bloat
		// the revision list. The override path needed the same guard; see
		// commit message for context.
		if (latest && latest.contentHash === contentHash) {
			if (!pinnedParent || pinnedParent === latest.id) {
				return latest;
			}
		}

		const rev = await DatabaseService.appendArtifactRevision(artifactId, {
			reason: 'edit',
			parentRevisionId: effectiveParent,
			contentHash,
			mimeType: payload.mimeType,
			text: payload.text,
			blob: payload.blob,
			metadata: payload.metadata
		});
		await this.load();
		return rev;
	}

	/**
	 * ai-patch entry point for inline / autocapture targets. Thin wrapper
	 * over `captureFromChat` that:
	 *   - creates the artifact on first hit (delegates to the usual
	 *     create-on-slot-miss path);
	 *   - on subsequent hits threads `parentRevisionId` through
	 *     `addUserEditRevision` so the edit is parented off the revision
	 *     that was current when the patch session started, not the one at
	 *     commit time.
	 * Returns both ids so the dispatcher can flip an inline target in-place
	 * to an artifact handle after first commit.
	 */
	async captureFromChatForPatch(
		source: CaptureSource,
		payload: CapturePayload,
		opts?: { parentRevisionId?: string }
	): Promise<{ artifactId: string; revisionId: string }> {
		const existing = await DatabaseService.findArtifactBySlot(source.conversationId, source.slot);
		if (!existing) {
			// Slot has no persisted record yet — create + initial revision in
			// one transaction, matching `captureFromChat`'s miss path.
			const contentHash = await hashPayload(payload);
			const { artifact, revision } = await DatabaseService.createArtifact(
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
			return { artifactId: artifact.id, revisionId: revision.id };
		}

		// Existing slot — append a patch-derived edit. Reuse the dedup-aware
		// override path so a no-op edit against the current tip doesn't
		// bloat the revision list.
		const rev = await this.addUserEditRevision(existing.id, payload, opts);
		return { artifactId: existing.id, revisionId: rev.id };
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
		// Fetch BEFORE delete so the mirror-delete path has the
		// `nextcloudSync.remotePath` it needs to find the remote file.
		const artifact = await DatabaseService.getArtifact(artifactId);
		await DatabaseService.deleteArtifact(artifactId);
		await this.load();
		if (artifact) {
			const { maybeMirrorDelete } = await import('$lib/services/nextcloud-upload.service');
			void maybeMirrorDelete(artifact);
		}
	}

	/**
	 * Bulk delete. Single `load()` at the end so the gallery doesn't re-render
	 * N times during a multi-select wipe.
	 */
	async removeMany(artifactIds: string[]): Promise<void> {
		if (artifactIds.length === 0) return;
		// Pre-fetch every artifact so the mirror-delete pass below has
		// the remote-path metadata. The local delete clears it from
		// IndexedDB so we have to snapshot first.
		const snapshots = await Promise.all(artifactIds.map((id) => DatabaseService.getArtifact(id)));
		for (const id of artifactIds) {
			await DatabaseService.deleteArtifact(id);
		}
		await this.load();
		const { maybeMirrorDelete } = await import('$lib/services/nextcloud-upload.service');
		for (const artifact of snapshots) {
			if (artifact) void maybeMirrorDelete(artifact);
		}
	}

	/**
	 * Roll back an artifact to a prior revision by appending a new
	 * `reason: 'rollback'` revision that duplicates the target revision's
	 * payload. The artifact's `currentRevisionId` advances to the new
	 * revision so the rollback is the new tip.
	 *
	 * Dedup short-circuit: rolling back to the current tip is a no-op —
	 * we compare `contentHash` rather than id so rolling back to a
	 * different-id-but-identical-content revision also dedupes.
	 *
	 * Metadata threaded through:
	 *   - `metadata.rolledBackFrom` — the tip at time of rollback
	 *   - `metadata.rolledBackTo`   — the target revision
	 */
	async rollbackToRevision(
		artifactId: string,
		targetRevisionId: string
	): Promise<DatabaseArtifactRevision> {
		const [artifact, targetRev] = await Promise.all([
			DatabaseService.getArtifact(artifactId),
			DatabaseService.getArtifactRevision(targetRevisionId)
		]);
		if (!artifact) throw new Error(`Artifact ${artifactId} not found`);
		if (!targetRev || targetRev.artifactId !== artifactId) {
			throw new Error(`Revision ${targetRevisionId} not found on artifact ${artifactId}`);
		}
		const currentRevisionId = artifact.currentRevisionId;
		const allRevs = await DatabaseService.listArtifactRevisions(artifactId);
		const currentRev = allRevs.find((r) => r.id === currentRevisionId) ?? allRevs.at(-1);

		if (currentRev && currentRev.contentHash === targetRev.contentHash) {
			return currentRev;
		}

		const rev = await DatabaseService.appendArtifactRevision(artifactId, {
			reason: 'rollback',
			parentRevisionId: currentRevisionId,
			contentHash: targetRev.contentHash,
			mimeType: targetRev.mimeType,
			text: targetRev.text,
			blob: targetRev.blob,
			metadata: {
				...(targetRev.metadata ?? {}),
				rolledBackFrom: currentRevisionId,
				rolledBackTo: targetRevisionId
			}
		});
		await this.load();
		return rev;
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
