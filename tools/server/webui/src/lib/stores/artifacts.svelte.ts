/**
 * artifactsStore — right-side drawer state for assistant message artifacts.
 *
 * Flow: ChatMessageAssistant registers each finished assistant message's content.
 * The store extracts HTML/SVG artifacts, keeps them keyed by messageId+index,
 * and auto-opens the drawer to the freshest one the first time it shows up
 * (but not on subsequent registrations, so the user can close it and stay closed).
 */

import { extractArtifacts, hashString, type ExtractedArtifact } from '$lib/utils/artifacts';

export interface ArtifactEntry extends ExtractedArtifact {
	id: string;
	messageId: string;
	registeredAt: number;
}

class ArtifactsStore {
	entries = $state<ArtifactEntry[]>([]);
	activeId = $state<string | null>(null);
	open = $state(false);
	userClosed = $state(false);

	private lastHashByMessage = new Map<string, string>();

	register(messageId: string, content: string): void {
		if (!messageId || !content) return;
		const hash = hashString(content);
		if (this.lastHashByMessage.get(messageId) === hash) return;
		this.lastHashByMessage.set(messageId, hash);

		const extracted = extractArtifacts(content);
		if (extracted.length === 0) {
			// Nothing artifact-worthy for this message — remove any prior entries it owned.
			this.entries = this.entries.filter((e) => e.messageId !== messageId);
			if (this.activeId && !this.entries.find((e) => e.id === this.activeId)) {
				this.activeId = this.entries.at(-1)?.id ?? null;
				if (!this.activeId) this.open = false;
			}
			return;
		}

		const now = Date.now();
		const newEntries: ArtifactEntry[] = extracted.map((a) => ({
			...a,
			id: `${messageId}#${a.index}`,
			messageId,
			registeredAt: now
		}));

		// Replace any prior entries from this message with the fresh extraction.
		const others = this.entries.filter((e) => e.messageId !== messageId);
		this.entries = [...others, ...newEntries];

		const newest = newEntries.at(-1)!;
		this.activeId = newest.id;

		// Auto-open only if the user hasn't explicitly closed during this session.
		if (!this.userClosed) this.open = true;
	}

	activate(id: string): void {
		this.activeId = id;
		this.open = true;
		this.userClosed = false;
	}

	toggle(): void {
		this.open = !this.open;
		if (!this.open) this.userClosed = true;
		else this.userClosed = false;
	}

	show(): void {
		this.open = true;
		this.userClosed = false;
	}

	close(): void {
		this.open = false;
		this.userClosed = true;
	}

	get active(): ArtifactEntry | null {
		if (!this.activeId) return null;
		return this.entries.find((e) => e.id === this.activeId) ?? null;
	}
}

export const artifactsStore = new ArtifactsStore();
