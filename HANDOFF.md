# DFlash Handoff — Gemma 4 31B + DFlash drafter

## Current status (2026-05-23, build b9286-266a6f69d)

End-to-end DFlash speculative decoding **compiles, loads, and runs**, but the
**acceptance rate is 4.9%** — well below the floor where speculation pays off.
Net result on the centurion 3090 with `gemma-4-31B-it-Q4_K_M` (FA on, ngl 99,
temp 0):

| run                                         | gen t/s | accept |
|---------------------------------------------|--------:|-------:|
| baseline (`llama-cli`, target alone)        |   29.1  |   —    |
| DFlash spec (`llama-speculative-simple`, Q4_K_M drafter, `--dflash`) | 14.9 | 4.9% |

DFlash is ~2× **slower** than baseline. Functional integration works; perf
claim does not land yet.

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

## Remaining hypotheses

- **Extraction point is wrong.** `cb("dflash_extract_N", il)` currently
  tags `cur` right after `build_cvec(cur, il)` in both `llama.cpp` and
  `gemma4.cpp`. The drafter may have been trained on a different
  intermediate (pre-cvec, post-attn-residual, post-ffn-residual, or
  the pre-norm output before attention).
- **GGUF conversion fidelity.** Compare Anbeeld safetensors → CPU fp32
  reference drafter logits on identical inputs against our Q-quant
  drafter. If logits diverge beyond quantization noise, the conversion
  pipeline (HF → GGUF) dropped or misnamed a tensor. Requires HF
  download (~6 GB safetensors) + reference inference setup.
- **RoPE position scheme.** Drafter's `dflash.target_layer_ids =
  [1,12,23,35,46,57]` and `block_size=16`. Maybe the drafter trained
  with absolute target-position embeddings (raw sequence positions
  from the original conversation) rather than the local
  `[0..n_ctx_used-1]` scheme we use after ctx truncation.

## Concrete next experiments (in priority order)

1. **Disable graph reuse via env**: run with
   `LLAMA_GRAPH_REUSE_DISABLE=1 ./build-cuda/bin/llama-speculative-simple
   ...`. If accept jumps significantly, graph reuse is corrupting input
   tensors across iterations. Cost: zero code changes, one bench window.

2. **Drop bucket rounding** (or set `ctx_window = -1` to force full
   length): if accept improves, then bucket masking is the bug. Edit
   `common/speculative.cpp:862` `ctx_window = 512` → `0` (means use
   full n_ctx_total without truncation).

3. **Try the BF16 drafter** (`gemma4-31b-it-dflash-bf16.gguf`, 2.9 GB):
   if accept improves materially, Q4_K_M drafter quantization is
   degrading drafts more than expected. Q8_0 also worth trying as a
   middle ground.

4. **Try IQ4_XS target** (`gemma-4-31B-it-IQ4_XS.gguf`): if the drafter
   was trained on hidden states extracted from a different target
   quant, swapping target quant could realign.

5. **Instrument extraction tokens**: log the actual token ids per
   ubatch in `extract_dflash_features` and the token ids per features
   slice in `common_speculative_impl_dflash::draft()`. Confirm the
   alignment matches the committed sequence in `prompt_tgt`.

6. **Hidden state extraction point**: try moving the `cb("dflash_extract_N")`
   tag from after `build_cvec(cur, il)` to before — i.e., capture the
   pre-control-vector hidden state. Or to right after `attn_residual`
   (before FFN). Either could matter if the drafter was trained on a
   specific intermediate representation.

7. **Warmup interference**: the drafter ctx warmup runs the dflash graph
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
