/**
 * Public contracts for the streaming AI-patch tool.
 *
 * All types here are framework-agnostic: no CodeMirror, no Svelte, no Dexie.
 * See `../../../../docs/research/diff-edit-tool-design.md` — §2 (wire format),
 * §3 (state machine), §6 (failure modes), §7 (tool surface) — for the
 * design-level rationale.
 */

/**
 * Discriminated target handle for a patch session. Exactly one of the three
 * kinds carries the stable identity of the buffer being edited.
 *
 * See design brief §7.1: mixing handles is the primary bug class in
 * multi-surface edit tools; we force the caller to pick one, explicitly.
 */
export type PatchTarget =
	| { kind: 'artifact'; artifactId: string }
	| { kind: 'doc'; docId: string }
	| {
			kind: 'inline';
			conversationId: string;
			parentMessageId: string;
			artifactIndex: number;
			/** Initial kind / mime / title for the materialised artifact. Required
			 *  because the inline target represents an autocapture slot that has
			 *  no persisted record yet — we need to create one on first commit. */
			seed: {
				kind: 'html' | 'svg' | 'code' | 'markdown';
				title: string;
				mimeType: string;
				baseText: string;
			};
	  };

/**
 * A single SEARCH/REPLACE block as extracted from the assistant text stream.
 * `filename` is optional — present only when the model emitted a filename
 * line above the fence (required in the doc-editor context, optional in the
 * artifact context per design §2.3).
 */
export interface SearchReplaceBlock {
	search: string;
	replace: string;
	filename?: string;
}

/** Events emitted by the streaming parser. Listener must be fast / synchronous. */
export type ParserEvent =
	| { type: 'block-open'; filename?: string }
	/** One complete SEARCH line (without trailing newline). */
	| { type: 'search-line'; line: string }
	/** The `=======` separator has been seen; search buffer is frozen. */
	| { type: 'search-complete'; search: string }
	/** One complete REPLACE line (without trailing newline). */
	| { type: 'replace-line'; line: string }
	/** A partial (non-newline-terminated) chunk of REPLACE text — for per-char streaming. */
	| { type: 'replace-chunk'; chunk: string }
	/** The closing `>>>>>>> REPLACE` marker was recognised. */
	| { type: 'block-close'; block: SearchReplaceBlock }
	/** A malformed-but-recognisable marker was tolerated. Informational only. */
	| { type: 'parse-warning'; reason: string; line: string }
	/** An unrecoverable marker-grammar violation (failure bucket F11). */
	| { type: 'parse-error'; code: PatchFailureCode; reason: string };

/**
 * Numbered failure buckets from design brief §6 plus §8.1 F11.
 * The enum is a closed set so downstream code (repair loop, telemetry) can
 * switch on it exhaustively.
 */
export enum PatchFailureCode {
	/** F2: fuzz ladder returned zero matches. */
	E_NO_MATCH = 'E_NO_MATCH',
	/** F3: fuzz ladder returned multiple matches. */
	E_AMBIGUOUS = 'E_AMBIGUOUS',
	/** F6: REPLACE payload contains a lazy-elision marker. */
	E_ELISION = 'E_ELISION',
	/** F7: DOMParser flagged the post-patch text as syntactically invalid. */
	E_SYNTAX_INVALID = 'E_SYNTAX_INVALID',
	/** F11: streaming parser exhausted marker-recovery heuristics. */
	E_MARKER_GRAMMAR = 'E_MARKER_GRAMMAR',
	/** F14: per-block SEARCH+REPLACE byte budget exceeded (see limiter). */
	E_BYTE_BUDGET = 'E_BYTE_BUDGET',
	/**
	 * F4: the user typed or pasted inside an in-flight block's anchor range
	 * while the AI was still streaming. We abort that block (user-takeover
	 * beats model takeover) and do NOT auto-retry — the user already
	 * redirected the edit themselves.
	 */
	E_USER_EDIT = 'E_USER_EDIT',
	/**
	 * Commit 5: the live-chat bootstrap saw a SEARCH/REPLACE fence but could
	 * not resolve a target buffer — no filename line above the fence matched
	 * a known doc, no artifact was open for the turn, and no autocapture
	 * slot had been produced yet. The block is skipped and the session
	 * continues in case later blocks are targetable. Not currently in
	 * REPAIRABLE_CODES — a second model pass cannot invent a target out of
	 * thin air.
	 */
	E_NO_TARGET = 'E_NO_TARGET'
}

/**
 * Discriminated-union tag stored under `DatabaseMessage.metadata.source`
 * to mark conversation turns that did not originate from a human keystroke.
 *
 * Commit 4a introduces the `'patch-repair'` variant for the F2 reflection
 * loop. Future variants (tool-use, mcp, system-injected) should extend
 * this union rather than inventing parallel marker fields.
 */
export type MessageSource = {
	kind: 'patch-repair';
	/** The assistant-message id that produced the failed patch session. */
	parentSessionId: string;
	/** The first failure code on the session the loop is trying to repair. */
	failureCode: PatchFailureCode;
	/** Stream-order index of the failing block whose retry this turn requests. */
	blockIndex: number;
	/** 1-based reflection count — 1 on the first retry, 2 on the second, etc. */
	reflection: number;
};

/** Narrow guard for the patch-repair source variant. */
export function isPatchRepairSource(
	value: unknown
): value is Extract<MessageSource, { kind: 'patch-repair' }> {
	if (!value || typeof value !== 'object') return false;
	const v = value as { kind?: unknown };
	return v.kind === 'patch-repair';
}

/**
 * Minimum context required to open a patch session. The dispatcher enriches
 * this with resolved base text / EditorView before handing off to the
 * streaming parser.
 */
export interface PatchSessionOptions {
	messageId: string;
	modelId: string;
	/**
	 * Per-block SEARCH+REPLACE byte budget, in characters (UTF-16 code units).
	 * Defaults to 16 KB when omitted. When exceeded the limiter emits an
	 * E_BYTE_BUDGET parse-error and drops the remainder of the block.
	 */
	byteBudget?: number;
}

/**
 * Source attribution threaded through `commit` into the persisted revision's
 * metadata. Uniquely identifies an ai-patch session for debugging and audit.
 */
export interface PatchSource {
	kind: 'ai-patch';
	modelId: string;
	/** Stable id of the assistant message whose stream produced the patch. */
	sessionId: string;
}

/**
 * A single similar-line candidate surfaced by the fuzz-match ladder's
 * `{kind: 'none'}` branch. Stored verbatim on the failing block's error
 * record so the repair-loop prompt can quote them back to the model.
 */
export interface PatchSimilarRegion {
	from: number;
	to: number;
	similarity: number;
	/** The actual buffer text of the region — we snapshot this at failure
	 *  time so downstream consumers don't need the original buffer. */
	text: string;
}

/**
 * One failure entry on a patch session. `blockIndex` is the 0-based index
 * of the block in stream order; `search` and `similar` are populated for
 * anchor-level failures (F2/F3) so the repair loop can reference them.
 */
export interface PatchErrorRecord {
	code: PatchFailureCode;
	reason: string;
	blockIndex?: number;
	/** Verbatim SEARCH payload for the failing block (F2/F3). */
	search?: string;
	/** Top-N similar regions from the fuzz ladder (F2), or colliding match
	 *  regions for F3. `similarity` is 1 for F3 matches since every hit is
	 *  byte-identical to SEARCH. */
	similar?: PatchSimilarRegion[];
	/**
	 * Text against which the failure was evaluated — the shadow buffer at
	 * block-open time (F2/F3) or the committed post-patch text (F7). Used
	 * by the repair format to quote N lines of context around each match
	 * or around the reported parse position.
	 */
	targetText?: string;
	/** F6: the REPLACE text the block emitted, plus the detector's line index. */
	replace?: string;
	elisionLine?: number;
	/** F7: kind the DOMParser was asked to validate (html | svg). */
	artifactKind?: 'html' | 'svg' | 'code' | 'markdown';
	/** F7: parse position, if the browser surfaced one. 1-based. */
	syntaxLine?: number;
	syntaxColumn?: number;
	/** F11: the partial SEARCH / REPLACE text the parser managed to capture
	 *  before the marker grammar broke. Either may be an empty string. */
	partialSearch?: string;
	partialReplace?: string;
	/** F14: the budget (in UTF-16 code units) that was exceeded. */
	byteBudget?: number;
}

/** Result of committing a completed patch session. */
export interface CommitResult {
	/** New artifact-revision id (artifact target, or inline target after upgrade). */
	revisionId?: string;
	/** New artifact id (inline target — set only on the first commit that
	 *  materialises the autocapture slot into a gallery artifact). */
	newArtifactId?: string;
	/** Doc id (doc target — docs don't version). */
	docId?: string;
	/**
	 * `true` when the session ended without committing because every block
	 * failed (or the syntax gate rejected the final text). The caller can
	 * surface this to the repair loop without mistaking it for a successful
	 * no-op commit.
	 */
	committed?: false;
	/** Short code for the non-commit reason, for logging/telemetry. */
	reason?: 'no-blocks' | 'syntax-invalid' | 'aborted';
	/** Non-fatal diagnostics collected across the session. */
	errors?: PatchErrorRecord[];
	/**
	 * Set on non-committed sessions when at least one error is of a
	 * repair-format-supported code (today: F2 / E_NO_MATCH) *and* the
	 * caller-side reflection budget has not yet been exhausted. Callers use
	 * this to decide whether to drive the repair loop; the dispatcher
	 * itself is agnostic of the budget and always sets `true` if any
	 * repairable error exists.
	 */
	repairable?: boolean;
}

/**
 * A target resolved against the persistence layer. The dispatcher returns one
 * of these per `PatchTarget`; it owns the commit closure so the session loop
 * never has to know which store/service to talk to.
 */
export interface ResolvedTarget {
	kind: PatchTarget['kind'];
	/**
	 * Kind used by the syntax gate. `'doc'` maps to markdown (docs are
	 * always markdown in v1) — the gate will skip the DOMParser check for it.
	 */
	syntaxKind: 'html' | 'svg' | 'code' | 'markdown' | 'image' | 'audio' | 'video' | 'pdf' | 'doc';
	/** Snapshot of the buffer at session-start. Anchoring runs against this. */
	baseText: string;
	/**
	 * Revision id pinned at session-start (artifact target only). The commit
	 * closure threads this back into the store as `parentRevisionId` so
	 * concurrent edits during streaming don't silently re-parent the new
	 * revision onto whatever happens to be current at commit time.
	 */
	parentRevisionId?: string;
	/**
	 * Opaque reference to the downstream view (CM6 EditorView, Svelte
	 * component handle, etc.). Kept `unknown` here so the types module stays
	 * framework-agnostic. The dispatcher may populate this from the
	 * docsStore registry when a DocEditor is live for the target doc.
	 */
	viewRef?: unknown;
	commit: (finalText: string, opts: { source: PatchSource }) => Promise<CommitResult>;
}
