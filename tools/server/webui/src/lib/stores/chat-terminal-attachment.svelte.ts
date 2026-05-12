/**
 * Chat-attached terminal — when the model executes `send_keys`
 * against a sandbox terminal we show the live PTY underneath the
 * chat composer so the user can either keep talking to the model
 * or pop into the terminal and type. The user can dismiss it; the
 * next `send_keys` call re-mounts whatever terminal it acted on.
 *
 * Process-local (not persisted). Reload = clean slate. The store
 * lives at the chat-layout level so a single drawer renders no
 * matter which conversation is active.
 */

class ChatTerminalAttachmentStore {
	/** Canonical terminal id the model most recently typed into. */
	terminalId = $state<string | null>(null);
	/** The user dismissed the drawer for this id; suppresses re-mount
	 *  until a *different* id arrives or the user re-opens it. */
	private dismissedId: string | null = null;

	/** Returns true if the drawer should be visible. */
	get visible(): boolean {
		return Boolean(this.terminalId) && this.terminalId !== this.dismissedId;
	}

	/** Called from the `send_keys` execute path. Re-attaches even if
	 *  the user previously dismissed the same id, on the assumption
	 *  that "the model is typing into it again" is interesting again. */
	attach(id: string): void {
		this.terminalId = id;
		this.dismissedId = null;
	}

	/** User pressed the close button on the drawer. Suppress re-mount
	 *  for this id only; a different terminal can still attach. */
	dismiss(): void {
		this.dismissedId = this.terminalId;
	}

	/** Hard reset — used on conversation switch if we ever decide the
	 *  attachment should be conversation-scoped. Today it's session-
	 *  global; calling this is optional. */
	clear(): void {
		this.terminalId = null;
		this.dismissedId = null;
	}
}

export const chatTerminalAttachment = new ChatTerminalAttachmentStore();
