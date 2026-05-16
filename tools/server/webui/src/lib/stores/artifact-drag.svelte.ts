/**
 * Shared state for dragging an artifact out of its preview and onto the chat
 * upload zone. The HTML5 drag-and-drop spec lets a source attach a File via
 * `dataTransfer.items.add(file)`, but that path is unreliable across the
 * webview surfaces we ship (webkit2gtk in particular). Routing the File via a
 * module-level holder keeps the handoff deterministic on every platform — the
 * drop target reads from here and falls back to `dataTransfer.files` only
 * when nothing artifact-shaped is in flight.
 */

export const ARTIFACT_DRAG_MIME = 'application/x-llama-artifact-pending';

let pendingFile: File | null = null;

export const artifactDrag = {
	begin(file: File) {
		pendingFile = file;
	},
	consume(): File | null {
		const file = pendingFile;
		pendingFile = null;
		return file;
	},
	end() {
		pendingFile = null;
	},
	get isPending(): boolean {
		return pendingFile !== null;
	}
};
