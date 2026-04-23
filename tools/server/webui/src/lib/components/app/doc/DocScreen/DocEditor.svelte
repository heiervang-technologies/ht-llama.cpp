<script module lang="ts">
	import type { EditorView as EditorViewType } from '@codemirror/view';

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
		/**
		 * Return the underlying CM6 view. Used by the ai-patch dispatcher so
		 * it can attach a shadow-doc bridge and paint streaming edits on
		 * top of the live view. Callers must not keep the reference past
		 * this editor's unmount — the docsStore registry is the stable
		 * handle for the *current* view.
		 */
		getEditorView(): EditorViewType;
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
	import { patchStateField } from '$lib/editor/ai-patch/cm6-bridge';
	import { docsStore } from '$lib/stores/docs.svelte';

	interface Props {
		content: string;
		onChange: (value: string) => void;
		onReady?: (api: DocEditorApi) => void;
		/**
		 * Stable Dexie id of the doc shown. Optional so existing callers
		 * that don't need ai-patch dispatcher integration can stay on the
		 * old signature; when provided, the editor registers itself in the
		 * docsStore active-view map so the dispatcher can paint streaming
		 * edits onto this view.
		 */
		docId?: string;
	}

	let { content, onChange, onReady, docId }: Props = $props();

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

	// Clipboard image → inline markdown image. CodeMirror's default paste
	// handler ignores non-text clipboard items, so we intercept the DOM event
	// before it bubbles up. Returning true tells CM we handled the paste and
	// keeps it from also pasting an empty string.
	function insertPastedImage(view: EditorView, blob: Blob) {
		const reader = new FileReader();
		reader.onload = () => {
			const dataUrl = typeof reader.result === 'string' ? reader.result : '';
			if (!dataUrl) return;
			const insert = `\n![pasted image](${dataUrl})\n`;
			const pos = view.state.selection.main.to;
			view.dispatch({
				changes: { from: pos, to: pos, insert },
				selection: { anchor: pos + insert.length },
				scrollIntoView: true,
				userEvent: 'input.paste'
			});
		};
		reader.readAsDataURL(blob);
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
			// ai-patch in-flight widget layer. Empty when no session is
			// active — the dispatcher installs decorations via effects at
			// streaming time. Included at construction so `appendConfig` is
			// a no-op; saves a reconfigure on the first patch.
			patchStateField,
			keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
			EditorView.lineWrapping,
			EditorView.domEventHandlers({
				paste(event, view) {
					const items = event.clipboardData?.items;
					if (!items) return false;
					for (const item of items) {
						if (item.kind === 'file' && item.type.startsWith('image/')) {
							const blob = item.getAsFile();
							if (blob) {
								event.preventDefault();
								insertPastedImage(view, blob);
								return true;
							}
						}
					}
					return false;
				}
			}),
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
		const v = view;
		const api: DocEditorApi = {
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
					userEvent: opts?.stream ? 'input.type.ai' : undefined,
					// Keep the growing tail visible as tokens arrive. Without
					// this, append-mode streaming scrolls off the bottom of
					// the visible region and the user sees a static view.
					effects: opts?.stream
						? EditorView.scrollIntoView(safeFrom + text.length, { y: 'nearest' })
						: undefined
				});
			},
			focus: () => v.focus(),
			getEditorView: () => v
		};
		if (docId) {
			docsStore.registerActiveView(docId, api);
		}
		if (onReady) onReady(api);
	});

	onDestroy(() => {
		if (docId) {
			docsStore.unregisterActiveView(docId);
		}
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

	/* ai-patch in-flight widget — subtle tint so the user sees where the
	   streaming edit is landing. No border, no layout shift. The widget
	   renders the shadow-doc contents on top of the real doc; when the
	   block closes the dispatcher swaps it out for a single real
	   transaction tagged "input.type.ai". */
	:global(.doc-editor-host .cm-ai-patch-inflight) {
		background-color: color-mix(in oklab, var(--primary) 12%, transparent);
		border-radius: 2px;
		padding: 0 1px;
	}
</style>
