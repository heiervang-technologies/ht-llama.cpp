import { toast } from 'svelte-sonner';
import { parseAiCommands, type AiCommand } from '$lib/constants/ai-commands';
import { AiCommandsService } from '$lib/services/ai-commands.service';
import { settingsStore } from '$lib/stores/settings.svelte';

class AiCommandsStore {
	#runningId = $state<string | null>(null);
	#controller: AbortController | null = null;

	get runningId(): string | null {
		return this.#runningId;
	}

	list(): AiCommand[] {
		return parseAiCommands(settingsStore.config.aiCommands);
	}

	stop(): void {
		if (this.#controller) {
			this.#controller.abort();
			this.#controller = null;
		}
		this.#runningId = null;
	}

	async run(
		commandId: string,
		document: string,
		onToken: (delta: string) => void,
		selection?: string
	): Promise<void> {
		const command = this.list().find((c) => c.id === commandId);
		if (!command) {
			toast.error(`Unknown AI command: ${commandId}`);
			return;
		}
		this.stop();
		const controller = new AbortController();
		this.#controller = controller;
		this.#runningId = commandId;
		try {
			await AiCommandsService.run({
				command,
				document,
				selection,
				signal: controller.signal,
				onToken
			});
		} catch (err) {
			if (err instanceof DOMException && err.name === 'AbortError') return;
			const msg = err instanceof Error ? err.message : String(err);
			console.error('[ai-commands]', err);
			toast.error(`Command failed: ${msg}`);
		} finally {
			if (this.#runningId === commandId) this.#runningId = null;
			if (this.#controller === controller) this.#controller = null;
		}
	}
}

export const aiCommandsStore = new AiCommandsStore();
