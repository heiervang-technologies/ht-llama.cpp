/**
 * User-invokable AI commands for the doc editor.
 *
 * A command takes the current document (and optional selection) and runs it
 * through the chat model with a filled template. Output is appended to the
 * document. Users can add/edit commands in Settings → AI Commands.
 */

export type AiCommandMode = 'append' | 'replace';

export type AiCommand = {
	id: string;
	name: string;
	/** Template string; {{document}} and {{selection}} are substituted. */
	template: string;
	/** Where the AI output is placed in the document. */
	mode: AiCommandMode;
	/** If true, the command is skipped when there is no active selection. */
	requiresSelection?: boolean;
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
	},
	{
		id: 'builtin-rewrite-selection',
		name: 'Rewrite selection',
		template:
			'Rewrite the following passage to improve clarity and flow while preserving meaning and voice. Output only the rewritten passage — no preamble, no quotes, no explanation.\n\nPassage:\n{{selection}}',
		mode: 'replace',
		requiresSelection: true
	},
	{
		id: 'builtin-fix-grammar',
		name: 'Fix grammar & spelling',
		template:
			'Correct grammar, spelling, and punctuation in the following passage. Preserve the original voice, tone, and formatting. Output only the corrected passage — no preamble, no quotes, no explanation.\n\nPassage:\n{{selection}}',
		mode: 'replace',
		requiresSelection: true
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
				typeof c.template === 'string' &&
				(c.mode === 'append' || c.mode === 'replace')
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
