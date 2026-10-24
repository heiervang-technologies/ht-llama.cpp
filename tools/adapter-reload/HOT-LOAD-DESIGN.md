# Local adapter reload and a future hot-load endpoint

Verified on `origin/gems/b9634-responses-audio`, fetched 2026-10-24,
commit `6f41f735273b8d844d7e913058b0c320c9eb6612` (b9635).
No production requests were made. No hot-load endpoint was implemented.

## What exists

The brief's runtime API description is correct: adapters load with the model;
`GET /lora-adapters` lists them; `POST /lora-adapters` changes existing scales.
A request's `lora` array selects loaded adapter IDs. None of these loads a new
file. Local router `/models/unload` followed by `/models/load` respawns the
selected child and reads replacement bytes at the preset's existing adapter path.
The router process survives. In-flight requests to that child may fail; this is
near-hot replacement, not a zero-downtime operation. Schedule it when that model
is idle. Other models are not intentionally unloaded by this script.

The brief's claim that discovery remains only on the unmerged March branch is
outdated. The deployed revision already contains `load_from_models_dir_with_lora`,
architecture matching, child adapter injection, `/v1/models` adapter metadata,
and array-body POST routing via `?model=`. See `common/preset.cpp` and
`tools/server/server-models.cpp`; history includes downstream port `43197aedc`.
Directory discovery happens during router initialization, not each child reload.
Replacing an already registered path works; adding an unregistered path does not
make it discoverable without router restart or a future rescan API.

## Implemented local swap

`swap-adapter.sh` delegates to the standard-library Python helper. It:

1. Restricts the endpoint to explicit loopback HTTP addresses, bypasses proxies
   and rejects redirects; verifies a single registered adapter at ID 0.
2. Locks the target path, checks candidate SHA-256 and GGUF magic, and records
   the current registry, file hash and deterministic inference fingerprint.
3. Copies the old file to a recovery backup; writes and fsyncs a temporary file
   in the destination directory; atomically renames it and fsyncs that directory.
4. Unloads and reloads only the named child, restores prior global scales, and
   runs a deterministic request explicitly selecting adapter 0 at scale 1.
5. Requires the new output/probability fingerprint to match an independently
   validated candidate fingerprint and differ from the old adapter's fingerprint.
   A registry path alone cannot prove new weights were read.
6. On failure, restores the old file, reloads, restores scales and verifies old
   behavior. If recovery fails it retains the backup and records the failure.

The helper deliberately supports one adapter, not arbitrary multi-adapter stacks.
A candidate must be independently validated against the same model, binary,
hardware/backend and probe before invoking it. Floating point differences across
backends can change exact fingerprints. Probe quality matters: a candidate that
has no effect on the selected probe cannot pass this test. A request-scope active
adapter does not imply it should be globally enabled for other clients.
The lock coordinates this script's callers; it cannot stop unrelated writers or
administrators from modifying the file/router. Use with exclusive maintenance
ownership of the local model. It does not establish production authorization.

Usage (fill paths and expected hashes from independent local validation):

```sh
./swap-adapter.sh --url http://127.0.0.1:PORT --model MODEL \
  --adapter /absolute/active.gguf --replacement /absolute/candidate.gguf \
  --probe /absolute/probe.json --expected-file-sha FILE_SHA256 \
  --expected-probe-sha PROBE_SHA256 --receipt /absolute/receipt.json
```

`test-cpu-reload.py` is the executable example. It starts a temporary reference
server for candidate B and a loopback router with adapter A, verifies rollback on
an incorrect fingerprint, then verifies successful A→B replacement and that
B-at-scale-zero differs. All test processes are reaped, including router children.
The fixture is a 1.19 MB public `ggml-org/tiny-llamas` model with deterministic
rank-1 synthetic LoRAs (2,304 tensor bytes each). These are infrastructure fixtures,
not trained Jasonelle adapters. Actual GGUF files are 2,624 bytes each.
Source: https://huggingface.co/ggml-org/tiny-llamas ; immutable revision and SHA-256
are recorded in `cpu-test/source.json` on KC3.

Final test: PASS; successful replacement plus checks 0.049 seconds, rollback PASS,
router survived. This tiny-model timing does not predict 12B reload latency.
Evidence lives under `/run/media/me/storage/builds/jasonelle-bake/cpu-test/`.

## True hot loading: proposed API and lifetime rules

Add an authenticated management endpoint, for example
`POST /lora-adapters/load` with `{path, sha256, base_fingerprint, name}` and a
corresponding unload operation. Return an immutable adapter ID plus generation,
verified content hash and base identity. Keep the existing scale endpoint's
meaning unchanged. The router forwards an explicit model selection; future
persistence/rescan belongs to a separate router registry operation.

The underlying API exists in `include/llama.h`: `llama_adapter_lora_init(model,
path)`, `llama_set_adapters_lora(ctx, adapters, count, scales)` and
`llama_adapter_lora_free(adapter)`. The missing work is server ownership and
scheduling, not a new tensor format.

Start with a conservative implementation: enqueue a management task on the
inference owner thread, stop admissions for this child, drain active slots, then
load and validate the candidate. Reject requests with a bounded busy response
instead of allowing an unbounded queue. Do not mutate model/context ownership on
HTTP handler threads. Validate the canonical path beneath an allowlisted adapter
root, expected hash, exact base fingerprint, tensor names/shapes and memory budget.
Architecture-only matching (the current discovery policy) is insufficient to
establish base compatibility. Enforce management authorization independently of
normal inference access; do not accept arbitrary client filesystem paths.

Publish the new adapter atomically only after successful loading and a local
smoke test. Leave old adapters registered on failure. Assign IDs monotonically;
never reuse a numeric ID for different bytes while a request may reference it.
Changing an adapter generation must invalidate slot prompt/KV cache and saved
prompt checkpoints that depended on the old adapter. Flush/rebuild affected
compute graphs as required by the context API. Update per-slot adapter pointers,
global defaults, speculative/draft contexts and aLoRA invocation metadata
consistently; initially reject combinations not explicitly tested.

Unloading must wait until no in-flight request, slot, batch, cached graph or
context references the adapter. Detach it from all contexts before freeing it.
Use ownership/refcounts and deferred retirement, not a vector erase of raw
pointers. Initially allow one drain-and-switch transaction per model. Later,
versioned concurrent adapters can let old requests finish while new requests use
the new generation, provided memory admits both. Base weights remain resident.

Tests needed before shipping: live streaming/drain behavior, cancellation,
invalid files and wrong bases, OOM without losing the old adapter, concurrent
management requests, stale IDs, cache invalidation, default and request scales,
multiple adapters, router forwarding/auth, aLoRA and draft handling, and repeated
load/unload with memory-leak checks on CPU and the actual GPU backend.

## Rebase recommendation

Do not wholesale rebase/cherry-pick #13/#14 or the March branch onto gems: the
backend behavior is already ported and the branch includes old web UI generated
assets and unrelated ancestry. Verify the currently served UI separately before
claiming picker parity. If a picker is missing, port only the relevant UI behavior
to the current heierchat/upstream UI integration. For this task, stable preset
paths plus verified child reload already meet the near-hot requirement.
A future hot-load endpoint needs the lifecycle work above and would be a new,
small branch from freshly fetched gems, with CPU and GPU concurrency validation.
No PR or production deployment has been opened or performed.
