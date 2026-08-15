# Adversarial review: PR #148 LOADING VRAM reserve

Snapshot reviewed: `165a11a9233d162fa3800a6857406b3097fc983f`, jointly with PR #146 RFC commit `44dbe6853129d74d8e43731d8f558e3088a828cd` and the production `--models-max 1` invariant.

## CLAIMS

- **REFUTED — PR #148 prevents a second same-device load from being spawned when the first load's reservation leaves insufficient VRAM.** The new calculation can set `mem_fits=false`, but `unload_lru()` has no failed-admit return. The patch excludes `LOADING` models from eviction; if no loaded/sleeping model can help, `lru_model_name` is empty and the function breaks. `load()` then continues to spawn the candidate. This is the exact two-concurrent-load case in the commit message.
- **REFUTED — the reservation equals demand not yet visible in physical free memory.** The full reserve remains until `LOADING` transitions to a terminal state. As child allocations progressively become visible, physical free already falls while the full reserve is still subtracted, double-counting the visible portion. This is conservative but can evict an unrelated loaded model or report false pressure.
- **UNPROVEN — the reserved plan is the plan the child will allocate.** `per_device_bytes` is produced by a state-dependent fit during discovery. The child independently runs fit later against a different free-memory snapshot. A discovery under pressure can record an offloaded/reduced plan while a later child selects more GPU memory, or vice versa. The known one-GPU context-reduction exit can also serialize a successful plan as system-GPU-sized zeroes, making the reserve structurally present but ineffective.
- **CONFIRMED — reserve mutation is serialized with status transition.** `reserve_apply()` runs under the model mutex before publishing the replacement instance, and child status handling must acquire the same mutex. The `old_status == LOADING` guard makes the ordinary duplicate-UNLOADED release idempotent. This local synchronization does not repair the failed admission postcondition or plan identity.
- **REFUTED — PR #148 is covered by a branch-reaching regression test.** The commit adds no tests. Existing router coverage loads models serially and the `models_max=2` LRU test waits for each load to finish, so it cannot observe `reserved_per_device`, a concurrent `LOADING` state, partial physical allocation, failed eviction, or reserve release after a failed child.

## PR #146 JOINT

The RFC correctly identifies per-device placement rather than aggregate free VRAM as the relevant constraint and reasonably prefers typed in-router parameter construction over stdout parsing. The implementation joint is incomplete: PR #148 records a mutable fit estimate, subtracts it from an independently changing physical counter, and never converts `mem_fits=false` into a hard admission result. The RFC's 50–200 ms subprocess figure is also an uncommitted estimate, not measured evidence, but it is not the blocker.

For a sound implementation, compute one stable candidate allocation plan, make it the child contract, reserve it atomically before spawn, and return a typed retryable failure when `free-after-committed-reservations` cannot satisfy it. If physical free is retained as the authority, reservations must be reconciled as allocations become visible rather than double-counted. Tests need injectable device-memory readings and a spawn sentinel so the no-spawn invariant is deterministic.

## `--models-max 1`

The new per-device reserve does not change the normal single-model admission rule: the under-lock active-count recheck already prevents a second concurrent model, and a later request still evicts the loaded model before replacement. The patch's exclusion of `LOADING` models does change contention behavior: a request arriving during the first load can no longer evict that load, but it falls through to plain `std::runtime_error("model limit reached, try again later")`, which is surfaced as HTTP 500 rather than the router's typed retryable 503. Thus the reserve is irrelevant to the production invariant, while the concurrency behavior remains an incorrect API failure mode.

## NEW-FAILURE-MODES

- Insufficient projected VRAM with no safe eviction is logged/calculated but ignored, and the candidate is spawned anyway.
- Partially visible allocations are counted once in physical free and again in the full outstanding reserve, causing false eviction pressure.
- A stale or zero discovery-time fit plan can under-reserve a child that independently chooses a larger runtime plan.
- With `--models-max 1`, concurrent callers receive an internal-server error for expected transient contention.
- Signed addition/subtraction in the reservation vector has no checked overflow invariant; the negative clamp hides over-release instead of exposing ledger corruption.

## DISK

PR #148 adds only source-level in-memory accounting and makes no runtime disk-usage claim. It adds no persistent files, caches, or artifacts. The relevant resource claim is VRAM, and that accounting is not established by committed raw evidence or a branch-reaching test.

## VERDICT

**do-not-merge.** The exact race remains: when reservations show the candidate does not fit and no non-loading model can be evicted, the code proceeds to spawn it. Re-review after failed admission is enforced, the plan is stable and shared with the child, partial physical allocation is reconciled, and concurrent success/failure/release branches are tested.

**UPSTREAM: candidate.** Per-device atomic load reservation is generally useful router correctness work and can be proposed as a focused upstream change once the hard-admit, stable-plan, and deterministic-test gaps are fixed. It should not be bundled with unrelated HT router policy.
