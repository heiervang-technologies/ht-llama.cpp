# DFlash Handoff — Gemma 4 31B + DFlash drafter

## Current status (2026-05-24, build with b0a828e8e)

End-to-end DFlash speculative decoding works. **Best acceptance crossed
double digits** (11.36% Q6_K best, 8.89% mean) after fixing the Gemma4
embedding-scale + softcap inheritance per vLLM PR #41703. Still
significantly under the published ~21% MT-Bench / 44% HumanEval acceptance.

| run                                          | gen t/s | accept (mean of 3) | best |
|---------------------------------------------|--------:|-------------------:|-----:|
| baseline (`llama-cli`, target alone)        |   29.1  |          —         |   —  |
| Round-3 (Q4_K_M drafter, pre-fix)            | 14.9   |          4.92%     | -    |
| Round-5 (Q6_K drafter, pre-fix)              | 10-11  |          6.88%     | 8.51% |
| **Round-6 (Q6_K, embed-scale+softcap fix)**  | 10-11  |        **8.89%**   | **11.36%** |
| Round-6 (Q4_K_M, embed-scale+softcap fix)    | 10-11  |          6.76%     | 8.16% |

Reference target for our prompt class (conversational, MT-Bench-like)
is **~21% acceptance per vLLM PR #41703**. We're at 8.89% mean — gap
of ~12pp remains. HumanEval-class prompts (code) would target ~44%.

DFlash is still slower than baseline because acceptance rate is below
the break-even point (block_size=16 means even 1/16 accept is "free
draft cost amortization"; need ~25% accept for net speedup vs target alone).

## What we ruled out

1. **Arch loading.** `llm_arch_from_string` strips `-draft` suffix, the
   model-loader passes `arch_name_override` so KV lookups hit
   `dflash-draft.*` keys.
2. **Tokenizer mismatch.** Vocab sha256 byte-identical between target and
   drafter (`7af66a9004b0dd94`), merges sha256 identical
   (`ea437aa17955e79c`), n_tokens=262144 both, special-token ids match
   except EOS (target=106 `<end_of_turn>`, drafter=1 `<eos>`) — EOS only
   matters at stream end, not mid-stream verification.
3. **Drafter graph.** `src/models/dflash.cpp` builds cross-attention over
   `inp_dflash->target_hidden`, with `pos_ctx` and `kq_mask` filled in
   `llm_graph_input_dflash::set_input`. Structure looks correct.
4. **Feature extraction.** `cb(cur, "dflash_extract_N", il)` hooks in
   both `src/models/llama.cpp` and `src/models/gemma4.cpp` tag the
   post-`l_out` hidden state at the layer ids listed in the drafter's
   `dflash.target_layer_ids = [1, 12, 23, 35, 46, 57]`. All ids are in
   range for the 60-layer target.

## Updated diagnosis (round 2)

The slice in `common/speculative.cpp:855-860` is actually **correct on
paper**:
- After verification, target's `extract_dflash_features` stores K+1
  features in ubatch order: `[id_last, draft0, draft1, ..., draftK-1]`
  at positions `[n_past_old, n_past_old+1, ..., n_past_old+K]`.
- Speculative algorithm always accepts drafts in prefix order: m accepts
  → first m+1 ubatch positions ARE the committed tokens.
- Taking `features[0..n_new]` with `n_new = m+1` aligns correctly with
  the m+1 newly-committed tokens.

So the alignment is right **IF** features and ubatch positions are in
the same order, which they are in the standard verification flow.

## Structural integration is consistent with the drafter GGUF

Compared our `src/models/dflash.cpp` graph against the `dflash-pr` POC
branch's older graph. Key insight: **the POC was written for a different
drafter variant**. POC uses `LLM_TENSOR_FFN_NORM` ("blk.N.ffn_norm");
our drafter GGUF has `LLM_TENSOR_ATTN_POST_NORM`
("blk.N.post_attention_norm") and no `ffn_norm`. Our graph uses
`layer.attn_post_norm` — correct for our GGUF.

POC also has no bucket-rounding/masking; it rebuilds the graph every
step. We bucket-round + mask padding for graph reuse — masking logic in
`llm_graph_input_dflash::set_input` looks correct (masks `[n_real,
ctx_len)`).

## Round-3 bench: extraction point

| run                                | accept |
|------------------------------------|-------:|
| Q6_K late (after l_out, default)   | 10.69% |
| Q6_K early (before per-layer-embd) |  6.22% |
| Q4_K_M late                        |  4.92% |
| Q4_K_M early                       |  5.56% |

Late extraction (current default, post-`l_out`) is correct. Toggle via
`LLAMA_DFLASH_EXTRACT=early` for ablation.

## Round-2 bench results (2026-05-23 ~20:06-20:09 UTC)

| run                                                | accept | gen t/s |
|----------------------------------------------------|-------:|--------:|
| Q4_K_M drafter, ctx_window=512 (baseline-dflash)   |  4.92% |  14.95  |
| Q4_K_M + `LLAMA_GRAPH_REUSE_DISABLE=1`             |  4.92% |  13.85  |
| Q4_K_M + `LLAMA_DFLASH_CTX_WINDOW=0`               |  6.22% |  14.44  |
| Q5_K_M drafter                                      |  3.70% |  13.23  |
| Q6_K drafter (same prompt)                          | 10.69% |  15.78  |
| Q6_K drafter (different longer prompt)              |  8.01% |  14.47  |
| Q8_0 drafter                                        |  7.60% |  14.81  |
| BF16 drafter (ctx=2048, q8_0 KV)                    |  9.42% |  14.30  |

Two conclusions land cleanly:

1. **Graph reuse is innocent.** Same accept rate to 4 sig figs with and
   without `LLAMA_GRAPH_REUSE_DISABLE=1`. The graph caching mechanism
   isn't corrupting input tensors across iterations.

2. **Drafter quantization has real but bounded effect.** Q6_K is ~2× Q4_K_M.
   But the absolute ceiling here is ~10% accept — vs published DFlash
   30-50%. So the drafter is fundamentally under-conditioned by the
   target features even at high precision.

Truncation (`ctx_window`) costs a few percent but is not the main bug.

## Round-4: hypothesis 2 (per-layer renorm) — RULED OUT (2026-05-24)

Implemented env-gated experiment in `src/models/dflash.cpp` to apply
`layer.attn_norm` to `fused_target` before each layer's wk/wv
(`LLAMA_DFLASH_PER_LAYER_RENORM=1`). Clean 3x3 A/B on Q6_K with VRAM
free (centurion-llm scaled to 0):

| run | renorm OFF | renorm ON |
|----:|-----------:|----------:|
| 1   | 5.56%      | 2.02%     |
| 2   | 6.22%      | 4.92%     |
| 3   | 6.22%      | 4.30%     |
| **mean** | **6.00%** | **3.75%** |

Per-layer renorm makes accept rate **~2.25pp WORSE** on Q6_K. Strong
signal that the drafter was NOT trained with per-layer ctx renorm —
current implementation (single `dflash_hidden_norm` at entry,
following POC design) matches what the drafter expects. The env-gate
stays in dflash.cpp for future ablation symmetry but defaults off.

**Variance caveat.** Q6_K baseline run-to-run variance is ±2pp on
same seed/prompt/code. The HANDOFF Round-3 table value of 10.69%
for Q6_K appears to be an outlier or stale-code state; reproducible
range under current HEAD (d74f7e1c6) is 4.3-6.2%. Update Round-3
table accordingly when next bench cycle happens.

## Round-6: Gemma4 embedding-scale + softcap fix (2026-05-24, b0a828e8e)

Root-cause find from vLLM PR #41703: drafter shares target's tok_embd
+ lm_head. For Gemma4 targets, the drafter must inherit two transforms
that target applies around the shared weights:

1. **`sqrt(n_embd)` noise embedding normalization** (Gemma4 pipeline).
   Without it, noise embeddings are ~73× too small.
2. **`final_logit_softcapping = 30.0`** on drafter's lm_head output.
   Monotonic; doesn't affect greedy argmax but matches training distribution.

Implementation: `llama-context.cpp:380-395` cross-binding inherits these
from `target_model->arch == LLM_ARCH_GEMMA4`. `dflash.cpp` consumes via
`hparams.f_embedding_scale` (applied automatically by `build_inp_embd`'s
Granite-arch code path — see footgun note below) and a manual softcap
block matching `gemma4.cpp:443-447`.

**Footgun for future arch ports:** `llama-graph.cpp:1827-1829`
auto-applies `hparams.f_embedding_scale` inside `build_inp_embd` (originally
added for Granite). If you also add a manual `ggml_scale(inpL, scale)`
in your model graph, you get DOUBLE scaling and a quietly broken model.
Grep for `f_embedding_scale` usages before adding new manual scales.
First attempt of this fix did double-scale and tanked Q6_K to 2.65%
mean. Removing manual scale fixed it.

Bench result (Q6_K drafter, 3 runs, q8_0 KV, same prompt/seed):

| pre-fix | with fix |
|--------:|---------:|
| 8.51%   | 6.80%    |
| 7.64%   | 11.36%   |
| 4.49%   | 8.51%    |
| **mean 6.88%** | **mean 8.89%** |

+2pp lift, first clean cross of 10% threshold. Confirms hypothesis but
doesn't close the gap to vLLM's ~21% MT-Bench reference.

## Round-5: correctness audit vs upstream PR #22105 + z-lab reference (2026-05-24)

Stopped chasing single-knob hypotheses on the bench and ran a full
implementation audit against authoritative sources (upstream PR
ggml-org/llama.cpp#22105, z-lab/dflash PyTorch reference, vLLM
qwen3_dflash, drafter GGUF metadata dump). Audit summary:

### What matches the reference cleanly

| Item | Reference | Ours | Status |
|------|-----------|------|--------|
| `fc` + `dflash_hidden_norm` location | Once outside layer loop | Once at dflash.cpp:72-74 | ✓ |
| No per-layer renorm of `fused_target` | Confirmed Round-4 | Default off | ✓ |
| K/V concat order | `[ctx, noise]` | `[ctx, noise]` (dim 2) | ✓ |
| K/V projection shares same `wk`/`wv` for ctx + noise | Yes | Yes | ✓ |
| `attn_norm` applied to noise only | Yes | Yes | ✓ |
| `attn_q_norm` on Q post-reshape | Yes | dflash.cpp:89 | ✓ |
| `attn_k_norm` on K post-reshape | Yes (post-concat) | Per-side pre-concat (mathematically equivalent for per-token RMSNorm) | ✓ |
| V not normed, not RoPE'd | Confirmed | dflash.cpp:118-128 | ✓ |
| Block content `[id_last, MASK×(K-1)]` | Confirmed | speculative.cpp:892-895 | ✓ |
| Drafts sampled from positions `[1..K-1]` | Confirmed | speculative.cpp:908 | ✓ |
| `attn_post_norm` as FFN-input norm | Gemma-specific (drafter tensor list) | dflash.cpp:148 | ✓ |
| FFN type SwiGLU | Confirmed | `LLM_FFN_SILU + PAR` | ✓ |
| lm_head shared with target | Confirmed | llama-context.cpp:377 binds `model.output` to target's | ✓ |
| `tok_embd` shared with target | Confirmed | llama-context.cpp:376 binds | ✓ |
| Non-causal attention | Drafter GGUF has `attention.causal = False` | Our `kq_mask` only masks bucket padding | ✓ |
| mask_token_id | Drafter GGUF: `4` (matches tokenizer `<mask>` at id 4) | Loaded from KV | ✓ |
| block_size | 16 (drafter KV) | Loaded from KV | ✓ |

### Divergences identified

1. **Sliding-window attention not implemented in drafter graph.**
   Drafter GGUF has `dflash-draft.attention.sliding_window = 2048`
   and `sliding_window_pattern = [True, True, True, True, False]` —
   layers blk.0..blk.3 use SWA-2048, blk.4 uses full. Our
   `src/models/dflash.cpp` decoder uses uniform full attention with
   only bucket-padding masking. Latent at our typical
   `ctx_window=512` (max ctx-to-noise distance ~528 < 2048 window).
   **Would matter the moment `LLAMA_DFLASH_CTX_WINDOW > 2048`** —
   ctx tokens beyond window 2048 would attend in our graph but be
   masked in the drafter's training distribution. Worth fixing for
   correctness even though bench-neutral today.

2. **Position scheme is RoPE-relative-only (acknowledged shortcut).**
   Reference PyTorch uses absolute target-sequence positions
   monotonically across iterations. We use `[0..n_ctx_used-1]` for
   ctx and `[n_ctx_used..n_ctx_used+15]` for noise — local positions
   reset each step. Equivalent under RoPE-relative attention. Upstream
   PR comment explicitly calls this out as "no draft KV cache" mode.

3. **Extraction point convention.** Tested in Round-5 below — not
   the bug.

### Round-5 bench: extraction-point ablation, 3x3, q8_0 KV

Added `LLAMA_DFLASH_EXTRACT=upstream` mode in `gemma4.cpp` that tags
`inpL` at layer start (matches upstream PR #22105's convention where
the converter applies `+1` to layer ids). A/B vs current default:

| run | mode=late (current) | mode=upstream (PR convention) |
|----:|--------------------:|------------------------------:|
| 1   | 8.51%               | 4.49%                          |
| 2   | 7.64%               | 7.64%                          |
| 3   | 4.49%               | 5.23%                          |
| **mean** | **6.88%**     | **5.79%**                      |

Means overlap within one standard deviation. Exact counts repeat
across modes (11/144 in late_2 and upstream_2; 7/156 in late_3 and
upstream_1) — there are ~3 distinct "states" the bench lands in and
extraction-point is not the decision boundary. Late wins by a hair
which weakly suggests Anbeeld's Gemma converter did NOT apply the
`+1` shift (i.e. GGUF `target_layer_ids` are raw Python indices).

### Conclusion of audit

Implementation is mostly correct on every architectural detail we
can verify. The 4-8% accept ceiling vs published 30-50% is not
explainable by any single structural bug at the llama.cpp level.

Remaining real candidates (in rough order of plausibility):

- **GGUF conversion fidelity vs Anbeeld safetensors.** Compare
  drafter logits between Anbeeld GGUF and the original z-lab
  safetensors → CPU fp32 reference on identical inputs. If the
  GGUF is malformed (missing tensor, wrong shape, miscalibrated
  scale), this is the most likely culprit. Requires ~6 GB HF
  download + reference Python inference. **Highest priority.**
- **Run-to-run variance (CUDA non-determinism).** ±2-3pp variance
  on same seed/prompt is real. Likely from float reduction order
  in CUDA flash-attention near tie-break thresholds in greedy
  sampling, possibly amplified by bidirectional attention. Not a
  correctness bug per se but pollutes all bench signal. Mitigation:
  bench at temp>0 with many samples, or move to CPU backend for
  determinism testing.
- **SWA implementation gap.** Add SWA-2048 mask to first 4 drafter
  layers. Low-priority at current ctx sizes but correctness fix.

### Concrete next experiments (revised priority order)

1. **GGUF↔safetensors drafter logit parity.** Download Anbeeld's
   z-lab/gemma-4-31B-it-DFlash safetensors. Run reference PyTorch
   forward (single layer at a time if needed) on a fixed input,
   compare logits to our drafter's logits on the same input. If they
   diverge beyond ~1% relative error per-position, the GGUF
   conversion is the bug. Largest single workpiece (~6 GB download +
   reference inference setup), highest-confidence root-cause signal.

2. **Reduce variance by averaging.** Run 10x same-seed bench at temp 0
   AND 10x at temp 0.7 with different seeds. Report mean ± std for
   each mode. Without variance reduction, sub-2pp deltas are noise.

3. **Add SWA mask for blk.0..blk.3 of drafter.** Drafter GGUF
   has `sliding_window=2048` + pattern `[T,T,T,T,F]`. Currently no
   SWA enforcement. Add windowing to `llm_graph_input_dflash::set_input`
   for layers where `is_swa(il)`. Correctness fix; likely
   bench-neutral at `ctx_window=512` but principled.

4. **Disable graph reuse via env**: `LLAMA_GRAPH_REUSE_DISABLE=1`.
   Already tested 2026-05-23 — identical accept (4.92%) with and
   without. Innocent. Re-run only if other variables move.

5. **Try IQ4_XS target** (`gemma-4-31B-it-IQ4_XS.gguf`): if the drafter
   was trained on hidden states extracted from a different target
   quant, swapping target quant could realign. Untested.

6. **Warmup interference**: the drafter ctx warmup runs the dflash graph
   with `cross.v_embd.empty()` → `ctx_len = n_ctx = 4096`. The first
   real call should reset via `sched_need_reserve` when bucket changes
   from 0 → small bucket. Verify by adding a log to `sched_reserve` to
   see if it's actually triggered between warmup and first real draft.

## What works

- Arch registration (`LLM_ARCH_DFLASH`) and KV namespace handling
- Drafter GGUF loads with arch `dflash-draft`
- Target hooks (`gemma4.cpp`, `llama.cpp`) fire on the right layers
- `llama-server` `/v1/chat/completions` `--dflash` plumbing
- 8 unit tests pass (`./build-dflash/bin/test-dflash`)
- New: `model: "any"` resolves to the most-recently-used resident model
  on the router (`server-models.cpp` — `pick_any_resident()`)

## Reproduce the bench

VRAM-clear required (~21 GB on this box was held by the centurion-llm
qwen pod; snoop-kube scales it down on request).

```bash
# Baseline
./build-cuda/bin/llama-cli \
  -m models/gemma-4-31B-it-Q4_K_M.gguf \
  -p "Write a 50-word paragraph about speculative decoding." \
  -n 128 -ngl 99 -fa on -no-cnv --single-turn --perf --temp 0 --seed 1

# DFlash
./build-cuda/bin/llama-speculative-simple \
  -m models/gemma-4-31B-it-Q4_K_M.gguf \
  -md models/dflash-gemma4-31b-gguf/gemma4-31b-it-dflash-Q4_K_M.gguf \
  --dflash \
  -p "Write a 50-word paragraph about speculative decoding." \
  -n 128 -c 4096 -ngl 99 -ngld 99 -fa on --temp 0 --seed 1
```

## Build

```bash
cd /home/me/ht/forks/ht-llama.cpp
cmake --build build-cuda --target llama-cli llama-speculative-simple llama-server -j8
```

`build-cuda` and `build-dflash` are both CUDA-enabled
(`GGML_CUDA=ON`); old note that build-dflash was CPU-only was wrong.

## Reference

- Drafter HF: `Anbeeld/gemma-4-31B-it-DFlash-GGUF`
- DFlash author repo: `z-lab/gemma-4-31B-it-DFlash`
- Upstream PR: https://github.com/ggml-org/llama.cpp/pull/22105
- `dflash-pr` local branch holds the older POC for comparison
- Snoop-kube's offer: titan CUDA1 (~24 GB free, no scale-down needed) —
  Job manifest pending
