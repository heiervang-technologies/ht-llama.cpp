# RFC 001: Router Admit Decision for Same-Device-Pinned Models

## Context
Issue #66 identifies a router OOM condition that occurs when the router co-loads multiple large models pinned to the same device (e.g., via `tensor-split = 100,0` for `CUDA0`). The router currently checks if the aggregate available VRAM across all devices can fit the new model, failing to account for the physical footprint constraints imposed by `tensor-split` device pinning.

The proposed fix requires computing per-device memory usage (`compute_free_per_device(active_models)`) and testing the candidate model's per-device allocation plan against this per-device free memory.

To determine a candidate model's allocation plan, the router must invoke the fit calculation (`common_fit_params()`). However, `common_fit_params()` requires `llama_model_params` and `llama_context_params`, which are currently only bootstrapped inside the child model subprocess via `common_init_from_params()`.

## Architectural Decision: Approach 1 vs. Approach 2

As identified during the initial prototyping, there are two viable paths to implement the admit decision:

### Approach 1: In-router CLI Mini-Bootstrap (Recommended)
Refactor the preset parsing logic so that the router process can generate `mparams` and `cparams` from a model's preset directly.
- **Pros:**
  - Architecturally sound long-term solution.
  - Zero latency overhead during admit evaluation.
  - Robust (no string parsing or stdout serialization needed).
- **Cons:**
  - Requires a non-trivial refactor of `common_init_from_params` and `arg.cpp` to decouple parameter generation from the child-process boot path.

### Approach 2: Out-of-Process Dry-Run Subprocess
Shell out to the `tools/fit-params` binary, leveraging the `--print-per-device-bytes` (already partially implemented via `params.fit_params_print_plan`) for each admit decision.
- **Pros:**
  - Minimal refactoring required. Faster path-to-validation.
  - Keeps the router strictly isolated from `llama_model` dependencies.
- **Cons:**
  - Adds ~50-200ms latency per admit due to subprocess spawning and model header reading.
  - Brittle JSON/stdout parsing is required in the router process.

## Recommendation
We recommend **Approach 1**. 

While Approach 2 is faster to write, the 50-200ms latency hit per admit decision in the router's hot path (and the brittleness of parsing standard output across process boundaries for critical state management) introduces severe reliability risks for a production fleet. The router should be resilient and fast. 

We propose creating a `common_preset_to_params()` shared helper that can be called by both the child boot path and the router's admit loop. This avoids the subprocess overhead and maintains strong type safety during the admit decision.

## Next Steps
If the recommendation is accepted:
1. Extract `common_preset_to_params()` into `common/arg.cpp` or `common/common.cpp`.
2. Update the router's `pick_any_resident` and LRU eviction logic to respect per-device free memory calculations.
3. Remove the dry-run CLI code from `tools/fit-params` if it's no longer necessary.
