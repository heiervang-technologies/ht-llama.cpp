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

## Prime suspect: feature/commit alignment in `common/speculative.cpp`

In `common_speculative_impl_dflash::draft()` (around line 855):
```cpp
const float * features = llama_get_dflash_target_features(ctx_tgt);
const size_t new_size = (size_t)n_target_features * (size_t)n_new;
accumulated_ctx.insert(accumulated_ctx.end(), features, features + new_size);
```

`llama_get_dflash_target_features(ctx_tgt)` returns features for the
**last ubatch the target processed**, which during verification is K+1
tokens (K drafts + 1 fall-through). But `n_new` is the number of tokens
just committed (typically 1 or 2). Taking the first `n_new` rows of that
buffer assumes the first `n_new` ubatch positions == the `n_new` committed
tokens. If acceptance order doesn't line up, the drafter gets fed features
for **discarded** draft tokens instead of committed ones, leading to
cascading misalignment.

**Suggested next step:** instrument `extract_dflash_features` to log token
ids per ubatch, and instrument `draft()` to log which features it picks
and what `dflash_n_past` value it advances to. Compare the captured token
ids against the committed sequence.

Secondary suspect: the warmup path runs the dflash decoder graph before
`cross.v_embd` is populated — verify warmup is skipped or handled.

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
