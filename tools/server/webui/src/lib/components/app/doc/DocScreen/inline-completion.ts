/**
 * CodeMirror 6 ghost-text inline completion extension.
 *
 * Shows a debounced AI suggestion after the cursor. Tab accepts, Esc dismisses,
 * any other edit or cursor move invalidates the current suggestion.
 *
 * Wired to llama-server via CompletionService. The toggle/delay/max-tokens are
 * read from the settings store each fetch, so changing them takes effect live.
 */

import { EditorState, Prec, StateEffect, StateField, type Extension } from '@codemirror/state';
import {
	Decoration,
	type DecorationSet,
	EditorView,
	ViewPlugin,
	type ViewUpdate,
	WidgetType,
	keymap
} from '@codemirror/view';
import { config } from '$lib/stores/settings.svelte';
import { CompletionService } from '$lib/services/completion.service';

interface Suggestion {
	text: string;
	from: number;
}

const setSuggestion = StateEffect.define<Suggestion | null>();

const suggestionField = StateField.define<Suggestion | null>({
	create: () => null,
	update(value, tr) {
		for (const effect of tr.effects) {
			if (effect.is(setSuggestion)) return effect.value;
		}
		// Any doc change that isn't the acceptance insertion invalidates.
		if (tr.docChanged) return null;
		// Selection moves away from the anchor → invalidate.
		if (value && tr.selection) {
			const head = tr.selection.main.head;
			if (head !== value.from) return null;
		}
		return value;
	},
	provide: (f) =>
		EditorView.decorations.from(f, (value) => {
			if (!value || !value.text) return Decoration.none;
			return Decoration.set([
				Decoration.widget({
					widget: new GhostWidget(value.text),
					side: 1
				}).range(value.from)
			]);
		})
});

class GhostWidget extends WidgetType {
	constructor(readonly text: string) {
		super();
	}
	eq(other: GhostWidget) {
		return other.text === this.text;
	}
	toDOM() {
		const span = document.createElement('span');
		span.className = 'cm-ghost-text';
		span.textContent = this.text;
		span.style.opacity = '0.45';
		span.style.pointerEvents = 'none';
		span.style.whiteSpace = 'pre-wrap';
		return span;
	}
	ignoreEvent() {
		return true;
	}
}

function currentSuggestion(state: EditorState): Suggestion | null {
	return state.field(suggestionField, false) ?? null;
}

function acceptSuggestion(view: EditorView): boolean {
	const s = currentSuggestion(view.state);
	if (!s || !s.text) return false;
	view.dispatch({
		changes: { from: s.from, to: s.from, insert: s.text },
		selection: { anchor: s.from + s.text.length },
		effects: setSuggestion.of(null),
		userEvent: 'input.complete'
	});
	return true;
}

function dismissSuggestion(view: EditorView): boolean {
	if (!currentSuggestion(view.state)) return false;
	view.dispatch({ effects: setSuggestion.of(null) });
	return true;
}

class InlineCompletionRunner {
	decorations: DecorationSet = Decoration.none;
	private timer: ReturnType<typeof setTimeout> | null = null;
	private controller: AbortController | null = null;
	private lastPrefix = '';

	constructor(readonly view: EditorView) {}

	update(u: ViewUpdate) {
		if (u.docChanged || u.selectionSet) {
			this.schedule();
		}
	}

	schedule() {
		this.cancel();
		const c = config();
		if (!c.inlineCompletionEnabled) return;
		const delay = Math.max(200, Number(c.inlineCompletionDelay ?? 800));
		this.timer = setTimeout(() => this.run(false), delay);
	}

	force() {
		this.cancel();
		// Skip the enabled check and cache on explicit trigger: the user asked for it.
		this.run(true);
	}

	private cancel() {
		if (this.timer) {
			clearTimeout(this.timer);
			this.timer = null;
		}
		if (this.controller) {
			this.controller.abort();
			this.controller = null;
		}
	}

	private async run(forced: boolean) {
		const view = this.view;
		const { state } = view;
		const head = state.selection.main.head;
		// Only suggest when selection is empty (just a caret).
		if (state.selection.main.from !== state.selection.main.to) return;

		const doc = state.doc.toString();
		const prefix = doc.slice(Math.max(0, head - 4096), head);
		if (!forced && !prefix.trim()) return;

		// Don't re-fire for the same prefix we already suggested against (unless forced).
		if (!forced && prefix === this.lastPrefix && currentSuggestion(state)) return;
		this.lastPrefix = prefix;

		this.controller = new AbortController();
		try {
			const result = await CompletionService.complete({
				prompt: prefix,
				signal: this.controller.signal
			});
			const text = result.content;
			if (!text) return;
			// Only apply if cursor hasn't moved since we started.
			const now = view.state.selection.main.head;
			if (now !== head) return;
			view.dispatch({ effects: setSuggestion.of({ text, from: head }) });
		} catch (err) {
			if ((err as { name?: string })?.name !== 'AbortError') {
				console.warn('[inline-completion]', err);
			}
		} finally {
			this.controller = null;
		}
	}

	destroy() {
		this.cancel();
	}
}

const inlineCompletionPlugin = ViewPlugin.fromClass(InlineCompletionRunner);

function forceTrigger(view: EditorView): boolean {
	const plugin = view.plugin(inlineCompletionPlugin);
	if (!plugin) return false;
	plugin.force();
	return true;
}

// Highest precedence so Tab wins over the editor's default indent handler when
// a suggestion is active. The handlers return false when no suggestion is
// present, which lets the default Tab-indent flow through.
const inlineCompletionKeymap = Prec.highest(
	keymap.of([
		{ key: 'Tab', run: acceptSuggestion },
		{ key: 'Escape', run: dismissSuggestion },
		{ key: 'Ctrl-Tab', run: forceTrigger, preventDefault: true },
		{ key: 'Mod-Space', run: forceTrigger, preventDefault: true }
	])
);

export function inlineCompletion(): Extension {
	return [suggestionField, inlineCompletionPlugin, inlineCompletionKeymap];
}
