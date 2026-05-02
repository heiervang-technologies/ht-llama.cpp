# AI Patch — streaming diff-edit tool

Framework-agnostic TypeScript for the streaming LLM diff-edit tool described in
[`../../../../docs/research/diff-edit-tool-design.md`](../../../../docs/research/diff-edit-tool-design.md).
After commit 3, the module covers the full round-trip from a chat-stream
delta into a persisted edit on an artifact, doc, or inline autocapture
slot — with live CM6 rendering and undo-safe commits.

## Module map

| Module           | Responsibility                                                                                                                                                                                                                                                                                                             |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `types.ts`       | Public contracts: `PatchTarget` discriminated union, `SearchReplaceBlock`, `ParserEvent` variants, `PatchFailureCode` (F2 / F3 / F6 / F11 / F14), `PatchSessionOptions`, `CommitResult` (inc. non-commit sentinels).                                                                                                       |
| `parser.ts`      | `StreamingPatchParser` — design brief §3 state machine (IDLE → COLLECTING_SEARCH → COLLECTING_REPLACE → IDLE). Line-buffered with per-character chunk emission for live UI rendering and close-marker hold-back. Tolerates relaxed markers; emits `parse-warning` on recovery, `parse-error` (F11) on unrecoverable drift. |
| `fuzz-match.ts`  | `findAnchor(buffer, search)` — Aider-style ladder: exact → whitespace-stripped → leading-blank-tolerant → `...`-elision-expansion → LCS-similarity ≥ 0.8. Own LCS ratio implementation; no external dep. Ambiguity guard per design §3.2 step 3.                                                                           |
| `elision.ts`     | `detectElision(replaceText)` — F6 guard. Flags bare `...` lines and placeholder comments (`//`, `#`, `--`, `/* */`, `<!-- -->`) that combine an ellipsis with a placeholder keyword or are the sole body of a known placeholder phrase. Conservative: ignores ellipses embedded in real content.                           |
| `limiter.ts`     | `LimitedPatchStream` — F14 byte-budget enforcement around the parser. Per-block SEARCH+REPLACE cap (16 KB default); emits a synthetic `parse-error` with `E_BYTE_BUDGET` once tripped.                                                                                                                                     |
| `shadow-doc.ts`  | `ShadowDoc` — per-session buffer that stages anchor/append/close against an immutable `Text`. Emits `ChangeSet`s for the CM6 bridge and `getChangeSet(baseDoc)` for headless consumers.                                                                                                                                    |
| `syntax-gate.ts` | F7 `validateSyntax(kind, text)` — DOMParser-gated check for `html` / `svg`; markdown and `doc` always pass; code / binary kinds skipped in v1; SSR-safe when DOMParser is absent.                                                                                                                                          |
| `cm6-bridge.ts`  | `patchStateField` + `setInflight` / `clearInflight` effects + `attachPatchView(view, shadow, opts)` — widget-decoration layer that renders the evolving shadow on top of a live `EditorView`. On block close, one transaction replaces the anchor range with the final text, tagged `input.type.ai` for single-step undo.  |
| `dispatcher.ts`  | `PatchSession` + `resolveTarget` — orchestrates the round-trip. Wires F6 at block close, F7 + zero-block short-circuit at session commit, inline → artifact materialisation via `captureFromChatForPatch`, and auto-attach to a live DocEditor when the `docsStore` registry has one.                                      |
| `stream-hook.ts` | `createPatchStreamHook(session)` — chat-stream observer adapter (onChunk / onFinish / onAbort) for callers to plumb into `chat.service.ts` without patching the service itself.                                                                                                                                            |
| `index.ts`       | Barrel re-export.                                                                                                                                                                                                                                                                                                          |

## What's NOT yet implemented

Deferred to follow-up commits:

- GBNF grammar (§8.1) as an optional llama.cpp-only mode.
- Tree-sitter-based F7 checks for code kinds.
- Conflict-resolution UI for parent-revision drift (§5.3).
- Live reactivity for the `×` discard button — the registry is a plain
  Map, so the affordance only re-evaluates on component re-renders. A
  store-backed wrapper is the follow-up.
- Integration into the main chat-stream flow — `runPatchRepairLoop` is
  ready to plumb in, but `chatStore.sendMessage` does not yet call it.

## The repair loop (commit 4)

When the session ends with `{ committed: false }` and at least one
repairable error, the loop (§6.1):

1. Formats a `patch-repair` user turn quoting the failed SEARCH / REPLACE
   and any context the dispatcher captured (fuzz-ladder candidates for F2,
   match-context for F3, marked placeholder for F6, parser position for
   F7, partial capture for F11, budget for F14).
2. Persists the turn via `injectRepairTurn` (sets
   `metadata.source.kind === 'patch-repair'`; the renderer styles it
   muted + collapsed).
3. Re-opens a fresh `PatchSession` and re-invokes `runStream`. The caller
   owns the `ChatService.sendMessage` plumbing; the orchestrator only
   decides when to re-stream.
4. Repeats up to `MAX_REFLECTIONS` (= 3) attempts. On exhaustion: emits
   a toast and returns the final failure sentinel; committed blocks from
   earlier attempts stay on disk.

`E_USER_EDIT` (F4) is deliberately excluded from the repairable set:
when the user types over an in-flight block, we drop the block and do
NOT retry — the user has already redirected the edit.

Rollback lives in two places:

- **Per-patch**: one `input.type.ai` transaction per block, so Cmd-Z
  undoes one block at a time in the live editor.
- **Per-revision**: `artifactGalleryStore.rollbackToRevision` appends a
  `reason: 'rollback'` revision that duplicates the target revision's
  payload and threads `metadata.rolledBackFrom` / `rolledBackTo`. Dedup
  short-circuits rolling back to the current tip.

## Testing

This module is covered by the Vitest files under
`tests/unit/ai-patch-*.test.ts`:

| Test file                                         | Covers                                                                                                                                                                                  |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/ai-patch-parser.test.ts`              | `StreamingPatchParser` — event sequences under normal flow, random byte-boundary feed, malformed-marker recovery, contradictory-block scenarios.                                        |
| `tests/unit/ai-patch-fuzz-match.test.ts`          | `findAnchor` — each ladder rung (exact / ws / leading-blank / elision / similarity) and the `unique`/`none`/`ambiguous` outcomes.                                                       |
| `tests/unit/ai-patch-elision.test.ts`             | `detectElision` — positive cases (bare `...`, common placeholder comments across comment styles) and negative cases (real prose with `...`, legitimate comments mentioning "existing"). |
| `tests/unit/ai-patch-limiter.test.ts`             | `LimitedPatchStream` — byte-budget trip across SEARCH / REPLACE boundaries.                                                                                                             |
| `tests/unit/ai-patch-shadow-doc.test.ts`          | `ShadowDoc` — anchor / append / close round-trip, multi-block composition.                                                                                                              |
| `tests/unit/ai-patch-dispatcher.test.ts`          | `PatchSession` end-to-end on mocked stores; zero-block short-circuit; parse-error bucketing.                                                                                            |
| `tests/unit/ai-patch-syntax-gate.test.ts`         | F7 `validateSyntax` — html / svg parse-error path via a shimmed DOMParser, markdown always-pass, node-SSR skip fallback.                                                                |
| `tests/unit/ai-patch-elision-integration.test.ts` | F6 wired into the dispatcher event loop — elision blocks rewind, subsequent clean blocks still apply.                                                                                   |
| `tests/unit/ai-patch-dedup.test.ts`               | `addUserEditRevision` dedup matrix including the override-path hole fix.                                                                                                                |
| `tests/unit/ai-patch-inline-upgrade.test.ts`      | Inline-target materialisation via `captureFromChatForPatch`; in-place upgrade to an artifact handle after first commit.                                                                 |
| `tests/unit/ai-patch-e2e-docview.test.ts`         | Headless CM6 round-trip: streaming widget layer, single `input.type.ai` commit transaction, one-step undo reverts the whole patch.                                                      |

Run with:

```sh
cd tools/server/webui
npm install             # if node_modules is missing
npm run test:unit       # unit project only (no browser, no storybook)
npm run check           # svelte-check / tsc gate
```

For the full suite `npm test` runs UI / client / unit / e2e. The foundation
in this commit only needs `test:unit`; UI tests will appear alongside the CM6
glue commit.
