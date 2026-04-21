/**
 * User-invokable AI commands for the doc editor.
 *
 * A command takes the current document (and optional selection) and runs it
 * through the chat model with a filled template. Output is appended to the
 * document. Users can add/edit commands in Settings → AI Commands.
 */

export type AiCommand = {
	id: string;
	name: string;
	/** Template string; {{document}} and {{selection}} are substituted. */
	template: string;
	/** Where the AI output is placed in the document. */
	mode: 'append';
};

export const DEFAULT_AI_COMMANDS: AiCommand[] = [
	{
		id: 'builtin-summarize',
		name: 'Summarize',
		template: 'Summarize the following document in 3-5 concise bullet points.\n\n{{document}}',
		mode: 'append'
	},
	{
		id: 'builtin-outline',
		name: 'Outline',
		template: 'Produce a hierarchical markdown outline of the following document.\n\n{{document}}',
		mode: 'append'
	},
	{
		id: 'builtin-continue',
		name: 'Continue writing',
		template:
			'Continue writing the following document in the same style and voice. Output only the continuation — do not repeat what is already there.\n\n{{document}}',
		mode: 'append'
	}
];

export function parseAiCommands(raw: unknown): AiCommand[] {
	if (typeof raw !== 'string' || raw.trim().length === 0) return DEFAULT_AI_COMMANDS;
	try {
		const parsed = JSON.parse(raw);
		if (!Array.isArray(parsed)) return DEFAULT_AI_COMMANDS;
		return parsed.filter(
			(c): c is AiCommand =>
				c &&
				typeof c === 'object' &&
				typeof c.id === 'string' &&
				typeof c.name === 'string' &&
				typeof c.template === 'string'
		);
	} catch {
		return DEFAULT_AI_COMMANDS;
	}
}

export function fillTemplate(
	template: string,
	vars: { document: string; selection: string }
): string {
	return template
		.replaceAll('{{document}}', vars.document)
		.replaceAll('{{selection}}', vars.selection);
}
