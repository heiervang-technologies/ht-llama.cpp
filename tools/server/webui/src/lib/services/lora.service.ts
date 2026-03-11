import { apiFetch, apiFetchWithParams, apiPost } from '$lib/utils';

/**
 * LoRA adapter entry returned by the server
 */
export interface LoraAdapter {
	id: number;
	path: string;
	scale: number;
}

/**
 * Payload for updating adapter scales
 */
export interface LoraAdapterUpdate {
	id: number;
	scale: number;
}

/**
 * LoraService - Stateless service for LoRA adapter API communication
 *
 * In MODEL mode, calls go directly to the server.
 * In ROUTER mode, a `model` param is needed to proxy to the correct child process.
 */
export class LoraService {
	/**
	 * Fetch list of loaded LoRA adapters from the server.
	 * In router mode, pass modelId to route the request to the correct child.
	 */
	static async list(modelId?: string): Promise<LoraAdapter[]> {
		if (modelId) {
			return apiFetchWithParams<LoraAdapter[]>('./lora-adapters', { model: modelId });
		}
		return apiFetch<LoraAdapter[]>('/lora-adapters');
	}

	/**
	 * Update LoRA adapter scales on the server.
	 * In router mode, pass modelId as query param to route to the correct child.
	 */
	static async update(adapters: LoraAdapterUpdate[], modelId?: string): Promise<void> {
		const url = modelId ? `/lora-adapters?model=${encodeURIComponent(modelId)}` : '/lora-adapters';
		await apiPost<unknown, LoraAdapterUpdate[]>(url, adapters);
	}
}
