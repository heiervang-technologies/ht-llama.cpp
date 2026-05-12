import { CompletionService } from './completion.service';
import { fillTemplate, type AiCommand } from '$lib/constants/ai-commands';

export interface RunAiCommandOptions {
	command: AiCommand;
	document: string;
	selection?: string;
	signal?: AbortSignal;
	onToken?: (delta: string) => void;
}

/**
 * Runs a user-defined AI command against the current document.
 * Streams tokens via onToken; returns the full concatenated output.
 */
export class AiCommandsService {
	static async run(opts: RunAiCommandOptions): Promise<string> {
		const prompt = fillTemplate(opts.command.template, {
			document: opts.document,
			selection: opts.selection ?? ''
		});

		let full = '';
		for await (const delta of CompletionService.chatStream([{ role: 'user', content: prompt }], {
			signal: opts.signal
		})) {
			full += delta;
			opts.onToken?.(delta);
		}
		return full;
	}
}
