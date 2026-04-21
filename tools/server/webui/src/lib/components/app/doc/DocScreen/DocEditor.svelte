<script module lang="ts">
	export interface DocEditorApi {
		getSelection(): { from: number; to: number; text: string };
		/**
		 * Replace the document range [from, to) with `text`. Pass
		 * { stream: true } while streaming tokens from an AI command so each
		 * dispatch is annotated as a single logical edit — CodeMirror's history
		 * will merge adjacent streaming transactions into one undo step instead
		 * of one-per-token.
		 */
		replaceRange(from: number, to: number, text: string, opts?: { stream?: boolean }): void;
		focus(): void;
	}
</script>

<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { EditorView, keymap, lineNumbers, highlightActiveLine } from '@codemirror/view';
	import { EditorState, Compartment } from '@codemirror/state';
	import { markdown, markdownLanguage } from '@codemirror/lang-markdown';
	import { languages } from '@codemirror/language-data';
	import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
	import {
		syntaxHighlighting,
		defaultHighlightStyle,
		indentOnInput,
		bracketMatching
	} from '@codemirror/language';
	import { mode } from 'mode-watcher';
	import { inlineCompletion } from './inline-completion';

	interface Props {
		content: string;
		onChange: (value: string) => void;
		onReady?: (api: DocEditorApi) => void;
	}

	let { content, onChange, onReady }: Props = $props();

	let hostEl: HTMLDivElement | undefined = $state();
	let view: EditorView | undefined;
	const themeCompartment = new Compartment();
	let lastExternalContent = '';

	function buildTheme(isDark: boolean) {
		return EditorView.theme(
			{
				'&': {
					height: '100%',
					fontSize: '14px',
					color: 'var(--foreground)',
					backgroundColor: 'transparent'
				},
				'.cm-scroller': {
					fontFamily: 'var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace)',
					lineHeight: '1.6'
				},
				'.cm-content': { padding: '1rem 0' },
				'.cm-gutters': {
					backgroundColor: 'transparent',
					color: 'var(--muted-foreground)',
					border: 'none'
				},
				'.cm-activeLine': { backgroundColor: 'transparent' },
				'.cm-activeLineGutter': { backgroundColor: 'transparent' },
				'.cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection': {
					backgroundColor: 'var(--accent) !important'
				},
				'.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--foreground)' },
				'&.cm-focused': { outline: 'none' },
				'.cm-line': { padding: '0 1rem' }
			},
			{ dark: isDark }
		);
	}

	function buildExtensions(isDark: boolean) {
		return [
			lineNumbers(),
			history(),
			indentOnInput(),
			bracketMatching(),
			highlightActiveLine(),
			syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
			markdown({ base: markdownLanguage, codeLanguages: languages }),
			inlineCompletion(),
			keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
			EditorView.lineWrapping,
			EditorView.updateListener.of((update) => {
				if (update.docChanged) {
					const value = update.state.doc.toString();
					if (value !== lastExternalContent) {
						onChange(value);
					}
				}
			}),
			themeCompartment.of(buildTheme(isDark))
		];
	}

	onMount(() => {
		if (!hostEl) return;
		const isDark = mode.current === 'dark';
		lastExternalContent = content;
		view = new EditorView({
			state: EditorState.create({
				doc: content,
				extensions: buildExtensions(isDark)
			}),
			parent: hostEl
		});
		if (onReady) {
			const v = view;
			onReady({
				getSelection: () => {
					const sel = v.state.selection.main;
					const from = Math.min(sel.from, sel.to);
					const to = Math.max(sel.from, sel.to);
					return { from, to, text: v.state.sliceDoc(from, to) };
				},
				replaceRange: (from, to, text, opts) => {
					const docLen = v.state.doc.length;
					const safeFrom = Math.max(0, Math.min(from, docLen));
					const safeTo = Math.max(safeFrom, Math.min(to, docLen));
					v.dispatch({
						changes: { from: safeFrom, to: safeTo, insert: text },
						// CM6 history merges consecutive transactions whose userEvent
						// shares a prefix ("input.*"). Tagging each streamed token
						// as "input.type.ai" collapses the entire run into one undo
						// step. Normal edits omit the annotation to preserve the
						// default per-keystroke behaviour for non-AI flows.
						userEvent: opts?.stream ? 'input.type.ai' : undefined
					});
				},
				focus: () => v.focus()
			});
		}
	});

	onDestroy(() => {
		view?.destroy();
		view = undefined;
	});

	// Push external content changes into the editor without fighting local edits.
	$effect(() => {
		if (!view) return;
		if (content === lastExternalContent) return;
		const current = view.state.doc.toString();
		if (current === content) {
			lastExternalContent = content;
			return;
		}
		lastExternalContent = content;
		view.dispatch({
			changes: { from: 0, to: current.length, insert: content }
		});
	});

	// Swap theme when light/dark toggles.
	$effect(() => {
		const isDark = mode.current === 'dark';
		if (view) {
			view.dispatch({ effects: themeCompartment.reconfigure(buildTheme(isDark)) });
		}
	});
</script>

<div bind:this={hostEl} class="doc-editor-host h-full w-full overflow-auto"></div>

<style>
	:global(.doc-editor-host .cm-editor) {
		height: 100%;
	}
</style>
