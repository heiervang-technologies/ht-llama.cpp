import { apiFetch, apiPost } from '$lib/utils';

/**
 * API endpoint paths for LoRA adapter operations
 */
const API_LORA = {
	LIST: '/lora-adapters',
	UPDATE: '/lora-adapters'
};

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
 * Handles communication with the `/lora-adapters` endpoint for listing
 * and updating LoRA adapter configurations.
 *
 * **Endpoints:**
 * - `GET /lora-adapters` — List all loaded LoRA adapters with current scales
 * - `POST /lora-adapters` — Update adapter scales
 */
export class LoraService {
	/**
	 * Fetch list of loaded LoRA adapters from the server.
	 *
	 * @returns Array of LoRA adapters with id, path, and scale
	 */
	static async list(): Promise<LoraAdapter[]> {
		return apiFetch<LoraAdapter[]>(API_LORA.LIST);
	}

	/**
	 * Update LoRA adapter scales on the server.
	 *
	 * @param adapters - Array of adapter id/scale pairs to update
	 */
	static async update(adapters: LoraAdapterUpdate[]): Promise<void> {
		await apiPost<unknown, LoraAdapterUpdate[]>(API_LORA.UPDATE, adapters);
	}
}
