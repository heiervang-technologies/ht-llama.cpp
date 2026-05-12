/**
 * CodeMirror 6 bridge — glues a live `EditorView` to a `ShadowDoc` streaming
 * buffer without touching the real document until the block closes.
 *
 * Design brief §4.3 ("simplest workable"): during streaming we render the
 * shadow's evolving text as a widget decoration over the anchor range. The
 * real doc is untouched, so there's no history pollution and no markdown
 * preview drift (because preview reads the real doc, not the shadow). At
 * block close we dispatch ONE transaction: clear the widget, apply the net
 * `ChangeSet` (anchor-range → finalText), tag `userEvent: 'input.type.ai'`
 * with `addToHistory: true`. Undo is one step per block.
 *
 * Public surface:
 *
 *     const attach = attachPatchView(view, shadow, { anchorFrom, anchorTo });
 *     // …chunks stream into shadow via appendChunk…
 *     attach.update();          // re-renders the widget with latest shadow text
 *     attach.commit(finalText); // one real transaction, one undo step
 *     // or
 *     attach.abort();           // drop the widget, no history entry
 *
 * State-field design:
 *
 *   - A `StateField<DecorationSet>` holds at most one replace-widget
 *     covering the block's anchor range. Empty when no block is in-flight.
 *   - Two `StateEffect`s: `setInflight` installs / replaces the widget;
 *     `clearInflight` removes it. The bridge dispatches these; the rest of
 *     the editor ignores them.
 *   - Mapping: while a block is in-flight we DO NOT touch the real doc, so
 *     no change-mapping is required for the widget's own range. Concurrent
 *     user edits outside the range are handled by the field's `map` hook
 *     so the widget doesn't drift if the user types above it (F4).
 *
 * Testability: the field + effects are exported so tests can exercise them
 * via `EditorState.update({ effects })` without constructing an `EditorView`
 * (which requires a real DOM).
 */

import { ChangeSet, StateEffect, StateField, type Text } from '@codemirror/state';
import {
	Decoration,
	EditorView,
	ViewPlugin,
	WidgetType,
	type DecorationSet,
	type ViewUpdate
} from '@codemirror/view';
import type { ShadowDoc } from './shadow-doc';
import { PatchFailureCode } from './types';

/* ------------------------------------------------------------------------- */
/* Effects                                                                   */
/* ------------------------------------------------------------------------- */

/**
 * Install or replace the in-flight replace-widget over `[from, to)`. Emitted
 * on every shadow update — the widget itself is the view into current
 * shadow state, so we rebuild it on every effect rather than mutating in
 * place.
 */
export const setInflight = StateEffect.define<{
	from: number;
	to: number;
	/** Snapshot of the current shadow text for the widget to render. */
	shadowText: string;
}>();

/** Remove the in-flight widget. Emitted on commit and on abort. */
export const clearInflight = StateEffect.define<void>();

/* ------------------------------------------------------------------------- */
/* Widget                                                                    */
/* ------------------------------------------------------------------------- */

class ShadowWidget extends WidgetType {
	constructor(readonly text: string) {
		super();
	}

	eq(other: WidgetType): boolean {
		return other instanceof ShadowWidget && other.text === this.text;
	}

	toDOM(): HTMLElement {
		const span = document.createElement('span');
		span.className = 'cm-ai-patch-inflight';
		// Preserve newlines — the widget replaces a line-span of the real
		// doc, so content must render with the same vertical rhythm.
		span.style.whiteSpace = 'pre-wrap';
		span.textContent = this.text;
		return span;
	}

	/** Keep the widget transparent to mouse / keyboard navigation. */
	ignoreEvent(): boolean {
		return false;
	}
}

/* ------------------------------------------------------------------------- */
/* State field                                                               */
/* ------------------------------------------------------------------------- */

/**
 * Field holding the current in-flight replace-decoration, if any. Exported
 * for tests and for callers that want to register it manually (e.g. an
 * editor that builds its own extension list).
 */
export const patchStateField = StateField.define<DecorationSet>({
	create: () => Decoration.none,
	update(decorations, tr) {
		// Map through any user edits so the widget range follows changes
		// happening outside it. CM6's ChangeDesc.mapPos does the right thing
		// when the user types above the widget (F4 §4.4 case "outside ⇒
		// shift anchor").
		let mapped = decorations.map(tr.changes);

		for (const effect of tr.effects) {
			if (effect.is(setInflight)) {
				const { from, to, shadowText } = effect.value;
				mapped = Decoration.set([
					Decoration.replace({
						widget: new ShadowWidget(shadowText),
						inclusive: true
					}).range(from, to)
				]);
			} else if (effect.is(clearInflight)) {
				mapped = Decoration.none;
			}
		}
		return mapped;
	},
	provide: (f) => EditorView.decorations.from(f)
});

/* ------------------------------------------------------------------------- */
/* Attachment                                                                */
/* ------------------------------------------------------------------------- */

export interface CM6AttachmentOptions {
	anchorFrom: number;
	anchorTo: number;
}

export interface CM6Attachment {
	/**
	 * Re-read the shadow doc and refresh the widget. Called by the
	 * dispatcher after every streamed chunk. Safe to call at high
	 * frequency — CM6 dedupes widget DOM when `ShadowWidget.eq` says so.
	 */
	update(): void;
	/**
	 * Dispatch the final transaction: clear the widget and replace
	 * `[anchorFrom, anchorTo)` with `finalText`. One undo step, tagged
	 * `input.type.ai` per design §4.3.
	 */
	commit(finalText: string): void;
	/**
	 * Clear the widget with no doc change. No history entry. Used when the
	 * block failed anchoring or the session aborted.
	 */
	abort(): void;
}

/**
 * Attach a shadow-doc to a live EditorView. Installs the state field if
 * the view doesn't have it yet (via `appendConfig`), emits the initial
 * widget, and returns the control handle.
 */
export function attachPatchView(
	view: EditorView,
	shadow: ShadowDoc,
	opts: CM6AttachmentOptions
): CM6Attachment {
	ensurePatchField(view);

	const { anchorFrom, anchorTo } = opts;

	// Initial widget with empty shadow text — subsequent `update()` calls
	// fill it. We set it immediately so the user sees the tinted background
	// ("AI is editing") as soon as the anchor locks, not only after the
	// first chunk arrives.
	dispatchInflight(view, anchorFrom, anchorTo, shadow.currentBlockText());

	return {
		update: () => {
			dispatchInflight(view, anchorFrom, anchorTo, shadow.currentBlockText());
		},
		commit: (finalText: string) => {
			view.dispatch({
				changes: { from: anchorFrom, to: anchorTo, insert: finalText },
				effects: clearInflight.of(undefined),
				userEvent: 'input.type.ai'
				// addToHistory defaults to true for user-dispatched
				// transactions; we do NOT pass `addToHistory: false`.
			});
		},
		abort: () => {
			view.dispatch({ effects: clearInflight.of(undefined) });
		}
	};
}

function dispatchInflight(view: EditorView, from: number, to: number, shadowText: string): void {
	view.dispatch({ effects: setInflight.of({ from, to, shadowText }) });
}

function ensurePatchField(view: EditorView): void {
	// If the field is already in the state, nothing to do. Otherwise add it
	// via `appendConfig` — a cheap no-op for editors that already installed
	// it at construction time.
	if (view.state.field(patchStateField, false) !== undefined) return;
	view.dispatch({
		effects: StateEffect.appendConfig.of([patchStateField])
	});
}

/* ------------------------------------------------------------------------- */
/* F4 — user-edit abort listener                                             */
/* ------------------------------------------------------------------------- */

/**
 * Contract the abort listener speaks to. The session implements this:
 * it tracks the set of in-flight anchor ranges (one per open block) and
 * reacts to user edits that either shift or cancel those ranges.
 *
 * We keep the interface narrow so a test harness can stand in for a
 * real `PatchSession` without dragging in the dispatcher.
 */
export interface PatchAbortTarget {
	/**
	 * Snapshot the currently-in-flight block anchor ranges, in
	 * stream-order. Returns an empty array if nothing is in flight.
	 */
	inflightAnchors(): InflightAnchor[];
	/**
	 * Called when a user edit touches or overlaps the anchor range of the
	 * block at `blockIndex`. The session should record an `E_USER_EDIT`
	 * error, drop the widget decoration for that block, and continue
	 * processing subsequent blocks. Re-entrancy: the same block never
	 * fires twice because `inflightAnchors` won't surface it again once
	 * aborted.
	 */
	abortBlock(blockIndex: number, reason: 'E_USER_EDIT'): void;
	/**
	 * Called for each outside-the-range user edit to let the session
	 * remap its tracked anchors through the change. `mapPos` is the CM6
	 * `ChangeDesc.mapPos` function bound to the transaction's changes.
	 */
	remapAnchors(mapPos: (pos: number, assoc?: number) => number): void;
}

/**
 * One tracked anchor range. Block indices are stream-order and stable
 * across the whole session so callers (repair loop, telemetry) can
 * correlate with `PatchSession.blocks[n]`.
 */
export interface InflightAnchor {
	blockIndex: number;
	from: number;
	to: number;
}

/**
 * Install a view plugin that watches transactions and drives the
 * session-side abort / remap logic on non-ai transactions.
 *
 * Behaviour per transaction:
 *
 *   - Skip if `userEvent === 'input.type.ai'` (our own inserts).
 *   - Skip if `!update.docChanged` (selection-only changes).
 *   - For each change range in `update.changes`:
 *       - If it overlaps an in-flight anchor range, mark that block for
 *         abort via `target.abortBlock(idx, 'E_USER_EDIT')`.
 *       - Otherwise ask `target.remapAnchors(mapPos)` to shift every
 *         still-active range through the transaction's changes.
 *
 * Boundary semantics: overlap is inclusive — an edit whose `fromA ===
 * anchor.to` or `toA === anchor.from` counts as touching, because the
 * user is clearly redirecting where the block lands. A strict deletion
 * that straddles the anchor edge (starts outside, ends inside) aborts;
 * likewise one that brackets the anchor entirely.
 *
 * Returns a CM6 `Extension` the caller can add via `Compartment` or
 * `appendConfig` at editor-construct time.
 */
export function attachAbortListener(view: EditorView, target: PatchAbortTarget): () => void {
	const plugin = ViewPlugin.define(() => ({
		update: (update: ViewUpdate) => {
			if (!update.docChanged) return;
			for (const tr of update.transactions) {
				if (tr.isUserEvent('input.type.ai')) continue;
				handleUserTransaction(tr.changes, target);
			}
		}
	}));
	view.dispatch({ effects: StateEffect.appendConfig.of([plugin]) });
	// We don't currently need a proper disposer — the plugin lifetime is
	// bound to the view's own teardown. Return a no-op so callers can keep
	// the handle-like symmetry with other attachers in this module.
	return () => {
		// No-op; installed via appendConfig, cleaned up when the view is
		// destroyed. See note in README on CM6 extension lifecycles.
	};
}

/**
 * Core decision step — shared between the view plugin and the headless
 * test harness. Walks the change ranges and dispatches abort / remap
 * based on overlap with the session's tracked anchors.
 *
 * Exported for the F4 test suite, which drives the logic directly
 * against an `EditorState.update` without constructing a view.
 */
export function handleUserTransaction(
	changes: import('@codemirror/state').ChangeSet,
	target: PatchAbortTarget
): void {
	const anchors = target.inflightAnchors();
	if (anchors.length === 0) return;

	const abortedBlocks = new Set<number>();
	changes.iterChanges((fromA, toA) => {
		for (const anchor of anchors) {
			if (abortedBlocks.has(anchor.blockIndex)) continue;
			// Inclusive overlap: edges touching count as a redirect.
			if (fromA <= anchor.to && toA >= anchor.from) {
				abortedBlocks.add(anchor.blockIndex);
			}
		}
	});

	for (const blockIndex of abortedBlocks) {
		target.abortBlock(blockIndex, 'E_USER_EDIT');
	}

	// Any non-aborted anchor still needs its coordinates remapped through
	// the change set so subsequent chunk appends land at the right offset
	// when the user typed above the anchor. The session tracks the live
	// anchor positions; we hand it the bound mapPos so it doesn't need a
	// reference to the change set itself.
	const stillActive = anchors.filter((a) => !abortedBlocks.has(a.blockIndex));
	if (stillActive.length > 0) {
		target.remapAnchors((pos, assoc = 1) => changes.mapPos(pos, assoc));
	}
}

/**
 * Convenience: export the failure-code for F4 at the bridge's top level
 * so a view-side caller doesn't have to reach into `./types` directly.
 */
export const E_USER_EDIT = PatchFailureCode.E_USER_EDIT;

/* ------------------------------------------------------------------------- */
/* Test support                                                              */
/* ------------------------------------------------------------------------- */

/**
 * Compute the single `ChangeSet` that the bridge will dispatch at commit
 * time, against a given base `Text`. Shared between the runtime path and
 * tests that want to assert the final change without a live view.
 */
export function buildCommitChange(
	baseDoc: Text,
	anchorFrom: number,
	anchorTo: number,
	finalText: string
): ChangeSet {
	return ChangeSet.of({ from: anchorFrom, to: anchorTo, insert: finalText }, baseDoc.length);
}
