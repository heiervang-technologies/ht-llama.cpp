import { toast } from 'svelte-sonner';
import { LoraService } from '$lib/services/lora.service';
import type { LoraAdapter, LoraAdapterUpdate } from '$lib/services/lora.service';
import type { ApiDiscoveredLoraAdapter } from '$lib/types';

/**
 * LoRA adapter with UI state (loaded for the current model)
 */
export interface LoraAdapterState {
	id: number;
	path: string;
	name: string;
	scale: number;
	/** Scale value before the adapter was disabled (for toggle restore) */
	previousScale: number;
	enabled: boolean;
}

/**
 * Extract display name from adapter path.
 * Strips directory and .gguf extension.
 */
function extractAdapterName(path: string): string {
	const filename = path.split(/[/\\]/).pop() || path;
	return filename.replace(/\.gguf$/i, '');
}

/**
 * loraStore - Reactive store for LoRA adapter management
 *
 * Manages:
 * - Available LoRA adapters list (loaded for current model)
 * - All discovered adapters from models directory
 * - Per-adapter enable/disable toggle and scale
 * - Applying changes to the server
 */
class LoraStore {
	adapters = $state<LoraAdapterState[]>([]);
	allDiscovered = $state<ApiDiscoveredLoraAdapter[]>([]);
	loading = $state(false);
	applying = $state(false);

	/**
	 * Whether any LoRA adapters are available (loaded or discovered)
	 */
	get isAvailable(): boolean {
		return this.adapters.length > 0 || this.allDiscovered.length > 0;
	}

	/**
	 * Get adapters that are currently enabled (scale > 0)
	 */
	get activeAdapters(): LoraAdapterUpdate[] {
		return this.adapters
			.filter((a) => a.enabled)
			.map((a) => ({ id: a.id, scale: a.scale }));
	}

	/**
	 * Get all adapter scales for API submission (enabled use their scale, disabled use 0)
	 */
	get adapterUpdates(): LoraAdapterUpdate[] {
		return this.adapters.map((a) => ({
			id: a.id,
			scale: a.enabled ? a.scale : 0
		}));
	}

	/**
	 * Whether there are unsaved changes compared to what was last fetched
	 */
	private lastFetchedScales = $state<Map<number, number>>(new Map());

	get hasChanges(): boolean {
		return this.adapters.some((a) => {
			const effectiveScale = a.enabled ? a.scale : 0;
			const lastScale = this.lastFetchedScales.get(a.id) ?? 0;
			return Math.abs(effectiveScale - lastScale) > 0.001;
		});
	}

	/** Current model ID for router mode */
	private modelId: string | undefined = undefined;

	/**
	 * Fetch LoRA adapters from server.
	 * In router mode, pass modelId to route the request to the correct child.
	 * Also fetches all discovered adapters for the "show all" view.
	 */
	async fetch(modelId?: string): Promise<void> {
		if (this.loading) return;
		this.loading = true;
		this.modelId = modelId;

		try {
			const [adapters, allAdapters] = await Promise.all([
				LoraService.list(modelId).catch(() => [] as LoraAdapter[]),
				LoraService.listAll().catch(() => [] as ApiDiscoveredLoraAdapter[])
			]);

			this.allDiscovered = allAdapters;

			this.lastFetchedScales = new Map(adapters.map((a) => [a.id, a.scale]));

			this.adapters = adapters.map((a: LoraAdapter) => ({
				id: a.id,
				path: a.path,
				name: extractAdapterName(a.path),
				scale: 1.0,
				previousScale: 1.0,
				enabled: false
			}));
		} catch {
			// Server may not support LoRA — this is fine, leave empty
			this.adapters = [];
			this.allDiscovered = [];
		} finally {
			this.loading = false;
		}
	}

	/**
	 * Set the scale for a specific adapter
	 */
	setScale(id: number, scale: number): void {
		this.adapters = this.adapters.map((a) =>
			a.id === id ? { ...a, scale, previousScale: scale, enabled: scale > 0 } : a
		);
	}

	/**
	 * Toggle an adapter on/off.
	 * When disabling, remembers the previous scale for re-enable.
	 * When enabling, restores the previous scale (or 1.0 default).
	 */
	toggle(id: number): void {
		this.adapters = this.adapters.map((a) => {
			if (a.id !== id) return a;

			if (a.enabled) {
				return { ...a, enabled: false, previousScale: a.scale };
			} else {
				return { ...a, enabled: true, scale: a.previousScale || 1.0 };
			}
		});
	}

	/**
	 * Apply current adapter configuration to the server
	 */
	async applyChanges(): Promise<void> {
		if (this.applying) return;
		this.applying = true;

		try {
			await LoraService.update(this.adapterUpdates, this.modelId);

			// Update last fetched scales to reflect applied state
			this.lastFetchedScales = new Map(
				this.adapters.map((a) => [a.id, a.enabled ? a.scale : 0])
			);

			toast.success('LoRA adapters updated');
		} catch (error) {
			const msg = error instanceof Error ? error.message : 'Failed to update LoRA adapters';
			toast.error(msg);
		} finally {
			this.applying = false;
		}
	}

	/**
	 * Get the lora field for inclusion in chat completion requests.
	 * Returns undefined when no adapters are active (so the field is omitted).
	 */
	getRequestPayload(): LoraAdapterUpdate[] | undefined {
		const active = this.activeAdapters;
		return active.length > 0 ? active : undefined;
	}
}

export const loraStore = new LoraStore();
