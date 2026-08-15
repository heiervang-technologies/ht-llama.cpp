# Adversarial review: PRs #153 and #156

Snapshots reviewed: PR #153 `49d0435f2cb34d65d1c77a8bea6fdb094176b81b`; PR #156 `f90bb3e28fbac60335a9d8590631ec9ea8b2c0bb`.

## PR #153 — alias reload reconciliation

The stale-entry removal for an alias deleted from a still-existing non-running model is correct. Alias moves are not order-independent as claimed. The reload loop validates and commits one model at a time. If new owner `a-new` sorts before old owner `z-old`, `a-new` still sees `shared` in `z-old.meta.aliases` and rejects it; `z-old` later removes it, leaving no owner. The opposite lexical order succeeds, so behavior depends on model names. No test was added.

The safe shape is a two-phase transaction: parse all proposed alias sets, resolve conflicts against that complete new snapshot, then atomically replace model alias sets and rebuild `alias_to_name`. Tests must move one alias in both lexical directions in a single reload and prove `has_model()`, `get_meta()`, load, and unload resolve identically.

**VERDICT: do-not-merge.** The primary advertised move case still fails in one deterministic ordering and the changed branch has no test.

**UPSTREAM: candidate.** Atomic alias-index reconciliation is generic router correctness work once implemented and tested as a focused change.

## PR #156 — tool-result adjacency

The converters previously emitted mixed user text before its tool result, separating that result from the preceding assistant tool call in the OpenAI intermediate representation. The patch emits collected tool results first and the remaining user message second. Its direct C++ test constructs the broken mixed text/result shape and asserts assistant/tool/user order for both Anthropic and Gemini, so the changed branches are reached. Multiple tool results retain their source order, and text-only/no-result behavior is unchanged.

**VERDICT: merge.** I found no new failure mode within this adjacency-only scope. The historical Gemini same-name call-ID collision is outside this patch and does not invalidate its ordering fix.

**UPSTREAM: candidate.** This is a small protocol-conversion correctness fix with focused regression coverage and no HT-specific policy.

## DISK

Neither PR makes a runtime disk-usage claim or adds persistent artifacts. PR #153 adds 19 source lines; PR #156 adds a small converter change and 49 test lines. Disk behavior is unchanged.
