# Streaming Diff-Edit Tool — Design Brief

Design doc for an LLM-driven diff-edit tool that targets (a) CodeMirror 6 doc-editor
buffers persisted in Dexie and (b) text-kind artifacts (html / svg / code / markdown)
in the artifact-gallery. Grounded in four research reports:

- `docs/research/llm-diff-edit-literature.md` (academic / diff-format survey)
- `file-editing-research.md` (Anthropic text_editor + Claude Code Edit)
- `/tmp/aider_report.md` (Aider edit formats & fuzzy apply)
- `/tmp/edit-tools-research.md` (Cursor / Cline / Roo / Copilot / V4A / SWE-agent)

Out of scope: implementation. This doc defines contracts, state machines, and
integration points only.

---

## 1. Context & Hard Constraints

- **Local model backend.** llama.cpp serves small open models (7B–34B). Diff-XYZ
  ([arXiv:2510.12487](https://arxiv.org/abs/2510.12487)) shows these models
  generate **search/replace better than udiff**; udiff is stronger on *applying*
  a diff to a buffer, but we control the applier, not the generator.
- **Streaming is mandatory.** Patches arrive token-by-token from the completion
  stream. The target buffer must update incrementally — the user should watch
  text materialise, not see a post-hoc flash. This rules out "buffer the whole
  response, then apply atomically."
- **Two target buffer types**, but a single tool surface:
  1. Doc editor — a live CodeMirror 6 instance editing a markdown file in Dexie.
  2. Artifact revision — a text-kind artifact (`html` / `svg` / `code` /
     `markdown`) bound to `{artifactId, currentRevisionId}` with an append-only
     revision log via `DatabaseService.appendArtifactRevision`.
- **Cursor/selection must survive.** No unconditional `dispatch({selection})`;
  edits land as transactions that the user's caret naturally follows.
- **Undo must be a single step.** A 50-token streamed patch is one user-visible
  action in the editor history, not 50.
- **Cancellable mid-stream.** The user can abort; the buffer reverts cleanly.
- **No dedicated "apply model."** Cursor's two-model architecture
  ([fireworks.ai/blog/cursor](https://fireworks.ai/blog/cursor)) is out of
  scope — we have one local model, in one request.

---

## 2. Wire Format — Comparison and Recommendation

### 2.1 Candidates

| Format | Streamability | Local-model accuracy | Recovery story |
|---|---|---|---|
| **Whole-file rewrite** | Trivially streamable (append); but replaces buffer wholesale → flash, lost selection, huge token cost for small edits | High for < 400 lines (Cursor) | Trivial (replace) |
| **Anthropic `str_replace` tool call** | Atomic at call level; partial JSON not applicable until args complete — **no incremental buffer update** | Excellent on Claude; unknown on local models | Deterministic; uniqueness check |
| **Aider SEARCH/REPLACE blocks** in assistant text | Commit anchor at `=======`, stream REPLACE chars straight into the anchored range | Top of Aider leaderboard for nearly every model; Diff-XYZ says it wins *generation* for small models | Fuzzy-match ladder (exact → whitespace → difflib@0.8) |
| **Unified diff (Aider udiff / OpenAI V4A)** | Hunks commit only after full context seen; `+` lines can stream once anchored, but no per-token UI update during anchor search | udiff wins *application* in Diff-XYZ and cures GPT-4 Turbo laziness (20% → 61%); V4A requires the model be trained on V4A (Codex only) | `patch`-style fuzz=2, git `--3way` when blob known |
| **Structured JSON ops** (`{op:"insert_after_line", ...}`) | Per-op atomicity; similar to `str_replace` — no incremental update during args | Robust but verbose (3× tokens); needs grammar-constrained decoding to be reliable on small models | Trivial; each op validated |
| **Speculative apply model** (Cursor / Copilot Edits) | Excellent UX; requires a second model and draft-from-source inference stack | — | — (rewrites whole file) |

### 2.2 Recommendation: **Aider-style SEARCH/REPLACE blocks in assistant text**

Justified by **streamability** (the prime constraint) combined with local-model
performance:

1. **Incremental UI update is native.** Once the `=======` separator is seen,
   the SEARCH text is complete; we can anchor into the buffer *before the
   REPLACE text arrives*, then stream REPLACE tokens directly into that
   anchored range as they come off the wire. No format here achieves this as
   cleanly. Udiff anchors on context lines that are interleaved with `+` /
   `-` — we can't commit the anchor until we've seen enough context, and the
   `+` lines are mixed with `-` lines in arrival order.
2. **Strongest generation signal for the models we run.** Diff-XYZ (Glukhov et
   al., NeurIPS 2025 Workshop) isolates the format variable across 1000
   commits; for open / smaller models search/replace beats udiff on generation,
   and Aider's leaderboard confirms `diff` (SEARCH/REPLACE) as the top format
   for GPT-5, o3-pro, DeepSeek-V3.2, Claude — only Gemini prefers the
   `diff-fenced` variant.
3. **In assistant text, not a tool call.** OpenAI-style tool-calling streams
   tool arguments as JSON deltas, but most clients only surface them as a
   single structured object at the end. Emitting blocks inline in the
   assistant message — the shape Aider and Cline ship — gives us token-level
   stream access through the existing OpenAI-compatible chat endpoint, with no
   special tool plumbing. Agents that *want* structured calls get the
   programmatic wrapper (§7).
4. **Cheap error recovery.** If the SEARCH anchor fails the Aider fuzz ladder,
   we reject the block and inject a structured repair prompt with
   `find_similar_lines` hints — the same retry loop Aider runs with
   `max_reflections=3`.
5. **Aider's lazy-elision problem is mitigated.** Udiff's 3× win was
   specifically vs. GPT-4 Turbo's `# ... rest unchanged` habit. We guard
   against this in the parser (§6.6) — a cheap regex scan of REPLACE content.

Whole-file rewrite is retained as a **secondary path** for: (a) new artifact
creation, (b) refactors that touch > ~50% of the file. The tool picks the path;
the caller doesn't choose.

### 2.3 Block grammar

```
<<<<<<< SEARCH
<exact-lines-from-buffer>
=======
<replacement-lines>
>>>>>>> REPLACE
```

Multiple blocks per response, applied in **file order against the
progressively-mutated buffer** (each block sees the post-previous-block state).
No line numbers. No `@@` headers. Filename resolution:

- **Artifact context**: the target is implied — the active artifact. Filename
  line is optional; if present it must match `artifactId` or `name`.
- **Doc editor context**: filename line is required above the fence and must
  resolve to a Dexie-tracked markdown file. Aider-style fuzzy filename match
  (exact → basename → `difflib.get_close_matches(cutoff=0.8)`).

### 2.4 Why not structured tool calls?

A streaming client sees tool-call arguments as a series of JSON delta events
(`{"arguments": {"partial": "..."}}`). Parsing *partial* JSON to extract a
partial `new_string` is possible but fragile (quote escaping, unterminated
strings, UTF-8 split surrogates). Aider/Cline-style inline text blocks are
line-oriented and stream-parseable with a trivial state machine (§3). We ship
the programmatic wrapper (§7) for callers that need it.

---

## 3. Streaming Parser State Machine

### 3.1 States

```
IDLE
  ↓ (saw "^```" or "^<path>" line)
SAW_FENCE_OR_PATH
  ↓ (saw "^<<<<<<< SEARCH" within the fence)
COLLECTING_SEARCH
  ↓ (saw "^=======")
ANCHOR_LOCKED            ← buffer match resolved, provisional transaction opens
  ↓ (any REPLACE chars)
STREAMING_REPLACE        ← tokens flow into the buffer
  ↓ (saw "^>>>>>>> REPLACE")
BLOCK_COMMITTED          ← provisional → final; history entry merged
  ↓ (next block | end of message)
IDLE  or  DONE

+ ERROR state reachable from any of the above.
```

### 3.2 Per-state rules

- **IDLE**: scan each newly-arriving line for a filename line or fence opener.
  Buffer at most one non-fence line (may be a path) until we decide.
- **COLLECTING_SEARCH**: append each line to `search_buf`. Abort if we see
  `>>>>>>> REPLACE` without first seeing `=======` (malformed).
- **Transition to ANCHOR_LOCKED** (the only non-trivial transition):
  1. Freeze `search_buf`; run the **fuzz ladder** against the current buffer
     (Aider `replace_most_similar_chunk`):
     1. Exact match.
     2. Leading-whitespace-stripped match.
     3. Leading-blank-line tolerant.
     4. `...` elision expansion.
     5. `difflib.SequenceMatcher` ≥ 0.8 similarity.
  2. If 0 matches → enter ERROR with `E_NO_MATCH` + top 3
     `find_similar_lines` candidates.
  3. If ≥ 2 matches → enter ERROR with `E_AMBIGUOUS` + locations. (Aider's
     ambiguity guard: only enforce when the search target is ≥ 10
     non-whitespace chars; below that, require more context unconditionally.)
  4. Exactly 1 match → open a **provisional CM6 transaction** that:
     - Replaces `[anchorFrom, anchorTo)` with an empty string (visually deletes
       the SEARCH region; we'll re-materialize it as REPLACE streams in).
     - Marks the range with a `patchDecoration` (§4.2) to indicate "AI
       is writing here."
     - Is flagged `addToHistory: false` so the provisional state does not
       pollute undo.
- **STREAMING_REPLACE**: for each incoming chunk, dispatch a transaction that
  `insert`s the chunk at the end of the provisional range. `addToHistory:
  false`. Decoration range extends with each insert.
- **Transition to BLOCK_COMMITTED**: the closing `>>>>>>> REPLACE` is
  recognised. We:
  1. Strip the trailing partial line that was the marker (the parser only
     commits *complete* lines to REPLACE output; the marker is matched on a
     line prefix).
  2. Dispatch one **final "summary" transaction**: compute the net change
     vs. the original `anchorFrom..anchorTo` content, emit a single
     `ChangeSet` with `addToHistory: true` and `userEvent: "input.type.ai"`
     (matches the existing AI-streamed-edit userEvent in the doc editor).
  3. Revert all provisional inserts by dispatching their inverse **in the
     same tick** — so CM6 folds them into a single undo step. (See §4.3 for
     the exact recipe.)

### 3.3 Committability rule

A hunk is *committable* (i.e., safe to fold into history as a final
transaction) only when:

- The closing `>>>>>>> REPLACE` has been seen.
- The REPLACE content does **not** contain any elision marker
  (`E_ELISION` — see §6.6).
- Post-apply syntax gate passes for the artifact kind (§6.7).

If any check fails after streaming, we abort via the provisional-transaction
reverse path without touching history.

### 3.4 Backpressure & token boundaries

Tokens may split on byte boundaries (UTF-8) or mid-line. The parser operates
on a **line buffer**: accumulate bytes until a `\n`, then feed lines to the
state machine. Within STREAMING_REPLACE we relax this — partial lines are
inserted immediately (so the user sees per-token updates) but marker
recognition only happens on complete lines. This matches how Aider parses but
adds per-char rendering.

---

## 4. CodeMirror 6 Integration

### 4.1 Public surface

```ts
// New module: $lib/editor/ai-patch/
export interface PatchSession {
  view: EditorView;                 // the CM6 editor being mutated
  blocks: AppliedBlock[];           // committed + in-flight
  abort(): void;                    // reverts all in-flight, keeps committed? or all? see §5.3
  commit(): Promise<RevisionResult>;// ends streaming, finalizes revision
}
export interface AppliedBlock { search: string; replace: string; range: {from:number,to:number}; }

export function beginPatchSession(view: EditorView, opts: { messageId: string }): PatchSession;
```

### 4.2 State field + decorations

Single `StateField<DecorationSet>` holds all in-flight ranges.

```ts
const setPatchRange = StateEffect.define<{from:number,to:number,id:string}>();
const clearPatchRange = StateEffect.define<string>();   // id
const extendPatchRange = StateEffect.define<{id:string, delta:number}>();

const patchDecoration = Decoration.mark({class: "cm-ai-patch-inflight"});
// + a widget decoration at the head for a subtle "AI is editing" caret.
```

CSS: `cm-ai-patch-inflight` = low-opacity background tint + left-edge accent;
animated caret widget at the tail of the range. No layout changes — purely
visual, no reflow.

### 4.3 Undo-step merging — the recipe

CM6 merges adjacent transactions into one undo step when they share the same
`userEvent` and occur within `newGroupDelay` (default 500 ms). For a 10 s
streamed patch this is insufficient. Approach:

1. **All streaming transactions**: `addToHistory.of(false)`. They mutate the
   doc and the decoration set but do **not** push history entries.
2. **On commit**: dispatch two effects in one `transaction()`:
   - a `ChangeSet` that is `docBefore.replaceRange(anchorFrom, anchorTo,
     finalReplaceText)`  — i.e., the net delta as if the streaming never
     happened;
   - and a reverse `ChangeSet` cancelling the provisional in-memory state so
     the net-doc is unchanged.
   Because both are in the same transaction, they compose into a no-op for
   the *doc* but we then dispatch *another* transaction in the same tick
   with the real `ChangeSet` and `addToHistory.of(true)`,
   `userEvent: "input.type.ai"` (reuses the existing AI-streamed-edit userEvent
   from the doc editor so undo grouping stays consistent).
   
   *Simpler alternative* — skip provisional doc mutation entirely and use
   decorations to simulate the in-flight text via widgets. But this breaks
   markdown preview which reads from the real doc. Rejected.
   
   *Simplest workable* — buffer provisional state in a shadow doc
   (`Text.of(...)`) rendered via `decorations` as a replace-widget. At commit
   time, dispatch a single `addToHistory: true` transaction. This keeps the
   real doc untouched during streaming (no history pollution at all) and
   trivially merges to one undo step. **Recommended.**

### 4.4 Selection & cursor preservation

- At `beginPatchSession`: snapshot `view.state.selection`.
- Streaming uses widget decorations (shadow rendering) → real doc and
  selection unchanged.
- If the user types during streaming (§6.4), we face a choice — abort or
  pause. Recommended: the first user keystroke inside the patch region
  aborts; outside the region is allowed and shifts the anchor (`ChangeDesc`
  mapping applied to `anchorFrom`/`anchorTo`).
- At commit: the `ChangeSet` is mapped through any user-side changes via
  CM6's `ChangeDesc.mapPos` before being dispatched.

### 4.5 Virtualised artifacts (no live EditorView)

Artifacts in the gallery may not have a mounted CM6 view. For those, the
patch session runs against an in-memory `Text` (`@codemirror/state`), producing
the same `ChangeSet`; on commit we serialize `Text.toString()` into the new
revision. The shadow-doc strategy (§4.3) makes this trivial — the shadow
*is* the runtime buffer for virtualised artifacts.

---

## 5. Artifact-Revision Integration

### 5.1 Hook points

The schema is live — use it as-is:

```ts
// At session begin:
session.parentRevisionId = artifact.currentRevisionId; // snapshot at begin

// On successful commit — exact signature of the store method:
await DatabaseService.appendArtifactRevision(artifactId, {
  reason: 'edit',
  parentRevisionId: session.parentRevisionId, // session-start snapshot, NOT artifact.currentRevisionId at commit time
  mimeType: artifact.mimeType,
  text: finalContent,
  // contentHash intentionally omitted — store computes it (string hash for text,
  // SHA-1 via crypto.subtle for blobs)
  source: {
    kind: 'ai-patch',
    messageId: session.messageId,
    modelId: session.modelId,
  },
});
```

The store method transactionally appends and re-pins `currentRevisionId`; the
gallery's revision list re-renders with the new entry (pattern already
established by the artifact-gallery feature from `930322dbc` / `cbc808c40`).

**Autocapture slot keys.** Normal AI-generated artifacts are keyed by
`${parentMessageId}#${artifactIndex}`. If a patch stream is invoked on an
artifact that *also* has a live source chat turn (i.e., the user is editing
an artifact that was originally emitted by the current message), **route via
`artifactId`, not the slot key.** The slot-key path is for first-emission;
once an artifact exists, its id is the only stable handle.

### 5.2 One revision per session, not per block

Each `PatchSession` produces exactly one revision even if the response
contained multiple SEARCH/REPLACE blocks. Rationale:

- Blocks within one model turn are a single authorial intent.
- Revision diff remains readable in the revision list.
- Rollback semantics are clean: "undo this edit" = one click.

If this becomes coarse later (a 5-block response is really 5 unrelated
changes), we can add a `sessionGroupId` and keep per-block revisions grouped
under it. Ship the coarse version first.

### 5.3 Concurrent-edit conflict (parent drift)

If `artifact.currentRevisionId !== session.parentRevisionId` at commit time
(another edit — user or another agent — landed during streaming), *or* if
the user pinned a different revision mid-session:

- **Doc editor context**: the user typed into the same buffer. §4.4 handles
  this via ChangeDesc mapping; only remaining failure is if the mapping
  strips the patch entirely (user typed over the anchor), in which case
  abort cleanly.
- **Artifact context**: a foreign revision landed, or the pinned revision
  moved. Attempt a **3-way merge** (à la `git apply --3way`,
  [git-apply docs](https://git-scm.com/docs/git-apply)) using three known
  endpoints:
  - **base** = the revision whose id equals `session.parentRevisionId` (the
    session-start snapshot). Load its `text` explicitly; **do not** substitute
    `artifact.currentRevisionId` here — the two diverge when the user pins
    a different rev mid-session, and using `current` would treat the user's
    pin-change as part of the merge base.
  - **ours** = `session.parentRevisionId`'s text with our patch applied
    (what we would have committed without drift).
  - **theirs** = the text of `artifact.currentRevisionId` at commit time.

  If the merge is clean, append a new revision whose `parentRevisionId` is
  the current (drifted) revision — not the session-start one. If conflicting,
  surface a "conflict" toast with a diff and an "apply anyway / discard"
  prompt; do not silently overwrite.

### 5.4 Cancellation

Two cancellation paths:

1. **User abort** (presses Esc / cancel button): stop the completion stream,
   drop the shadow doc, dispatch no commit transaction. No revision.
2. **Stream terminates early** (EOS, network drop) mid-REPLACE: same as
   user abort — unfinished patches never commit. We do not emit a
   "partial revision."

---

## 6. Failure Modes & Recovery

Numbered for easy logging/metric bucketing.

| # | Failure | Detection | Handling |
|---|---|---|---|
| F1 | Partial buffer (stream ends mid-REPLACE) | State-machine in `STREAMING_REPLACE` when stream closes without `>>>>>>> REPLACE` | Drop shadow doc; no commit; no revision. Toast: "AI edit was interrupted." |
| F2 | Hunk mismatch (0 matches) | Fuzz ladder exhausted in COLLECTING_SEARCH → ANCHOR_LOCKED transition | Emit `E_NO_MATCH` with top-3 similar lines; inject a repair user-message with the error + hints; Aider-style `max_reflections=3`. |
| F3 | Ambiguous match (≥ 2) | Ladder returns multiple matches | `E_AMBIGUOUS` with locations; repair prompt asking for more context. |
| F4 | User edits mid-stream | Transaction in CM6 originates from a non-AI user event | If inside patch range → abort session; if outside → map anchor via ChangeDesc. |
| F5 | Contradictory later tokens (block overlaps a previously-committed block in the same session) | Anchor from block N lies inside block N−1's range | Apply against the post-N−1 buffer. If that fails, reject block N only; earlier blocks stand. |
| F6 | Lazy elision (`// ... rest unchanged`, `# ... existing code`, `/* ... */`) in REPLACE | Regex scan at BLOCK_COMMITTED on the REPLACE payload | Reject the block; repair prompt. Gemini CLI bug ([#4836](https://github.com/google-gemini/gemini-cli/issues/4836)) shows silent acceptance is catastrophic. |
| F7 | Post-apply syntax invalid | Tree-sitter / HTML parser / SVG parser depending on artifact kind | **Markdown**: warn, commit anyway. **Code / HTML / SVG**: reject the whole session; repair prompt with parser error. SWE-agent's +3pt improvement ([arXiv:2405.15793](https://arxiv.org/abs/2405.15793)) supports syntax-gated commits. |
| F8 | Parent drift (foreign revision during streaming) | `currentRevisionId !== parentRevisionId` at commit | 3-way merge (§5.3); fall back to conflict UI. |
| F9 | Filename resolution fail (doc-editor context) | No file matches filename line after fuzzy match | Reject session; surface "target file unknown." |
| F10 | Over-large block (SEARCH/REPLACE combined > some budget) | Byte count threshold | Reject with "split into smaller edits" prompt. Keeps per-block latency bounded. |
| F11 | Marker-grammar violation the parser can't recover | Streaming parser exhausted its malformed-marker recovery heuristics | Reject session; repair prompt with the literal byte offset of the confusing marker. Relevant on non-local backends where GBNF (§8.1) isn't active. |

### 6.1 Repair loop

After any of F2/F3/F6/F7/F10/F11, we **reflect** (Aider term): the error is
serialized as a user turn into the chat thread:

```
The patch could not be applied:
- Block 2: no match in artifact.svg. Closest lines:
    <circle cx="50" cy="50" r="45"/>
    <circle cx="50" cy="50" r="40"/>
Try again with more context, or rewrite the whole file.
```

Bounded retries (3). After that, surface to the user.

---

## 7. Built-in Tool Suite

Tools the model can call. Primary flow uses inline text blocks (§2.3); these
are the programmatic surface for agents and for non-patch operations.

| Tool | Shape | Notes |
|---|---|---|
| `view_artifact` | `{artifactId, revisionId?, viewRange?:[start,end]}` → `{content, lineCount, revisionId}` | Anthropic-style; returns line-numbered for reference; the model cites lines in its prose but **not** in SEARCH/REPLACE blocks. |
| `apply_patch` | `{artifactId, blocks: SRBlock[]}` — programmatic form of §2.3 | Non-streaming wrapper for non-chat agents. Internally runs the same parser + fuzz ladder. |
| `create_artifact` | `{kind, name, content}` → `{artifactId, revisionId}` | Whole-file creation; kinds = `html \| svg \| code \| markdown`. Streamable via a single "file fence" in assistant text. |
| `diff_artifacts` | `{artifactId, revA, revB}` → unified diff text | Read-only; for the model to reason about history. |
| `rollback_artifact` | `{artifactId, toRevisionId}` → `{newRevisionId}` | Appends a new revision whose content matches `toRevisionId`; `reason: 'rollback'`. |

Not included (deliberately):

- `insert_line` / `delete_line` — too line-number-dependent; drift problem.
- `str_replace` — covered by `apply_patch` with a single block.
- `undo_edit` — covered by `rollback_artifact` (revision log is the undo log).

### 7.1 Target identification contract

A single source of bugs in multi-surface edit tools is mixing target handles
at the call boundary. This tool recognises **exactly three** target handles,
and each tool-call carries exactly one, explicitly discriminated:

```ts
type PatchTarget =
  | { kind: 'artifact'; artifactId: string }       // persisted artifact in the gallery
  | { kind: 'doc';      docId: string }            // Dexie markdown doc in the doc editor
  | { kind: 'inline';   parentMessageId: string; artifactIndex: number }; // first-emission only
```

Rules:

1. **`artifact`** — the tool-call carries a concrete `artifactId` that the
   store knows about. All subsequent edits to that artifact **must** use this
   handle. This is the default post-emission path.
2. **`doc`** — the target is a Dexie-tracked markdown file mounted in the
   doc editor (a live CM6 `EditorView`). `docId` is the Dexie primary key;
   no artifact record exists. The filename line above the SEARCH fence is
   validated against the Dexie entry's filename (Aider-style fuzzy filename
   match with cutoff 0.8).
3. **`inline`** — **first-emission only**: the model is emitting into an
   autocapture slot keyed by `${parentMessageId}#${artifactIndex}` for an
   artifact that has not yet been persisted. On commit, the store materialises
   a new artifact record and returns its id; *subsequent* edits in the same
   session or any later session **must** switch to the `artifact` handle.
   Implementation: after first commit the session upgrades
   `target = {kind:'artifact', artifactId: newId}` in-place.

**Forbidden**: passing both `artifactId` and `docId` in the same call;
passing an `inline` handle when the artifact already has an id; passing an
`artifactId` that refers to a non-text-kind artifact (image / video / audio
cannot be patched).

**Dispatcher**: a single `applyPatchSession(target, blocks[])` resolves the
handle at the call boundary, loads the correct base text and
`parentRevisionId` (for artifact targets) or the `EditorView` (for doc
targets), and is the *only* place in the code that switches on `target.kind`.
All downstream code (parser, CM6 glue, revision writer) receives a uniform
resolved context.

### 7.2 Tool discovery / prompt surface

System prompt contribution (sketch, not final wording):

```
When editing an artifact, emit SEARCH/REPLACE blocks inline:

  <<<<<<< SEARCH
  ...exact lines from the current artifact...
  =======
  ...replacement lines...
  >>>>>>> REPLACE

Rules:
- Quote whole lines exactly; whitespace matters.
- Use multiple blocks for multiple changes (file-order).
- Do NOT write "..." or "unchanged" — the patch will be rejected.
- For new artifacts, call create_artifact instead.
```

This is ~80 tokens of system prompt; negligible next to the context cost of
the artifact itself.

---

## 8. Decisions & Open Questions

### 8.1 Decision: GBNF grammar-constrained markers — **go, as an optional mode**

**Decision**: support GBNF-constrained marker decoding as an **optional**
mode the session can turn on when the upstream is llama.cpp. The streaming
parser + commit-gate is the baseline and must remain fully sufficient on its
own; GBNF is a quality multiplier, never a correctness dependency.

**Why go**:

- llama.cpp's `/completion` and `/chat/completions` endpoints expose `grammar`
  (GBNF) and `json_schema` on the wire; no server-side change is needed.
- Diff-XYZ ([arXiv:2510.12487](https://arxiv.org/abs/2510.12487)) shows
  smaller open models benefit disproportionately from explicit format
  constraints — exactly the model size class we're serving.
- The grammar we need is skeletal — the `<<<<<<< SEARCH` / `=======` /
  `>>>>>>> REPLACE` boundary tokens, not the content between them. That keeps
  GBNF bounded and avoids the common trap of over-constraining the model into
  broken generations.

**Why only optional**:

1. **Non-local models bypass it silently.** Any OpenAI-compatible backend that
   isn't llama.cpp (hosted Claude, OpenAI, user-configured third parties) will
   ignore `grammar` fields. The tool must not *require* GBNF — the streaming
   parser has to handle malformed markers on its own.
2. **GBNF changes generation latency** (token sampling filters apply per
   token). Not every turn is worth the overhead; a short edit can skip it.
3. **It's easy to get the grammar wrong** and accidentally suppress valid
   output. Keeping it optional lets us measure the win and back off if the
   grammar rejects too much.

**Fallback contract** (mandatory for correctness, enforced at all times, GBNF
or not):

- The streaming parser (§3) **is the source of truth**. It must detect and
  recover from:
  - Malformed marker lines (e.g. `<<<<<<<<SEARCH` with extra `<`, missing
    space before `SEARCH`, wrong case, trailing whitespace).
  - Out-of-order markers (`>>>>>>> REPLACE` before `=======`).
  - Markers embedded inside code (false positives in the SEARCH or REPLACE
    content — we only recognise markers at start-of-line with correct
    bracketing, and we require fence context).
  - Missing filename lines (we use the session target as default).
- The commit-gate (§3.3) remains the final authority. A GBNF-passed block
  that fails elision regex or syntax check is still rejected.
- Failure bucket **F11** added (see below): "marker-grammar violation after
  streaming-parser recovery" — unrecoverable malformation. Handled via the
  §6.1 repair loop with `max_reflections=3`, same as F2/F3/F6.

**Is the fallback sufficient?** Yes, with one caveat. Aider runs production
at leaderboard quality on GPT-5, Claude, Gemini, DeepSeek with no
grammar-constrained decoding — a pure streaming parser with a fuzz ladder.
The caveat: Aider's own `replace_most_similar_chunk` was built over multiple
iterations of empirical failure modes. We inherit that design (§3.2) so we
inherit the track record. GBNF gets us the extra points on small local
models; it does not paper over a missing parser.

**Rollout**: ship the parser-only path first, add GBNF behind a
`ui.ai.patch.grammarConstrained: boolean` setting (default **on** when the
active endpoint identifies as llama.cpp, **off** otherwise), measure
edit-apply rate before/after on our own benchmark, promote to default if it
wins.

### 8.2 Open questions

1. **Apply-model escape hatch.** We explicitly exclude a Cursor-style
   two-model path for now, but the llama.cpp server supports speculative
   decoding with a draft model
   ([fireworks.ai/blog/cursor](https://fireworks.ai/blog/cursor)). Revisit if
   large refactors show quality problems.
2. **Shadow-doc vs. real-doc streaming.** §4.3 recommends widget decorations
   against a shadow doc. If CM6 widget rendering turns out to flash for
   large REPLACE payloads, fall back to real-doc + provisional-transaction
   reversal.
3. **Multi-file edits** (doc editor context). One SEARCH/REPLACE response
   can span files. We treat each file as a separate session internally but
   present "one edit" to the user. Aider already does this. Scope TBD —
   may punt to v2.
4. **Fuzz ceiling.** Aider's `difflib` threshold = 0.8. GNU patch caps fuzz
   at 2. We recommend 0.8 as default, with an experiment comparing it to
   0.85 on our own Aider-style polyglot benchmark for local models.
5. **Telemetry schema.** Failures F1–F11 should emit structured events so we
   can measure format reliability over time — the sole bucketed dataset
   that lets us tell "prompt tweak helped" from "it got lucky."

---

## 9. References

Primary sources cited above (see the four research docs for full URL lists):

- Aider, **Unified diffs make GPT-4 Turbo 3× less lazy** — https://aider.chat/docs/unified-diffs.html
- Aider, **Edit formats** — https://aider.chat/docs/more/edit-formats.html
- Aider, **Leaderboards** — https://aider.chat/docs/leaderboards/
- Aider source — `aider/coders/editblock_coder.py`, `udiff_coder.py`, `patch_coder.py`, `base_coder.py`
- Anthropic, **Text editor tool** — https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool.md
- Claude Code, **Tools reference** — https://code.claude.com/docs/en/tools-reference.md
- Fireworks, **How Cursor built Fast Apply** — https://fireworks.ai/blog/cursor
- OpenAI, **apply_patch / V4A** — https://developers.openai.com/api/docs/guides/tools-apply-patch
- OpenAI, **GPT-4.1 prompting guide** — https://developers.openai.com/cookbook/examples/gpt4-1_prompting_guide
- Cline system prompt — https://cline.bot/blog/system-prompt
- Roo Code apply_diff — https://docs.roocode.com/advanced-usage/available-tools/apply-diff
- SWE-agent (NeurIPS 2024) — https://arxiv.org/abs/2405.15793
- SWE-bench — https://arxiv.org/abs/2310.06770
- Agentless — https://arxiv.org/abs/2407.01489
- **Diff-XYZ** (NeurIPS 2025 Workshop) — https://arxiv.org/abs/2510.12487
- Moatless Tools — https://github.com/aorwall/moatless-tools
- GNU patch matching — https://www.gnu.org/s/diffutils/manual/html_node/Inexact.html
- git-apply --3way — https://git-scm.com/docs/git-apply
- Gemini CLI elision case — https://github.com/google-gemini/gemini-cli/issues/4836
