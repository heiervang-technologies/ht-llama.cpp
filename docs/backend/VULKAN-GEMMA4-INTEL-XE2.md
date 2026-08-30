# Gemma 4 on Intel Xe2 Vulkan: architecture, Flash Attention, and tuning

Status: engineering notes and local measurements, 2026-08-29.  The target used for
measurements is a Core Ultra 5 238V / Lunar Lake Arc 130V/140V iGPU, Linux `xe`,
Mesa 26.1.5 ANV, Vulkan 1.4, 32-wide subgroups, 48 KiB shared memory, FP16/BF16,
integer dot product, KHR cooperative matrix, and unified system memory.

This document is about making each usable Gemma 4 member perform well.  It does
not recommend replacing one family member with another: the dense, unified,
MoE, PLE, multimodal, and MTP variants serve different purposes.

## Bottom line for this Intel iGPU

1. Vulkan is the practical llama.cpp backend on this machine, but Flash Attention
   is not automatically a speed feature.  The right answer is shape-dependent:
   fused D=256 prompt attention is useful, while D=512 global prompts and small
   decode/speculative batches are faster as decomposed matmul/softmax graphs.
2. This branch implements that adaptive policy behind
   `LLAMA_VK_GEMMA4_HYBRID_FA=1` plus `--flash-attn on`.  At pp512 it measured
   roughly 40--59% faster than all-FA on 12B and 35--39% faster on 26B A4B.
   The selector also restores the faster decomposed decode path for batches below
   32 tokens.  Benchmark it against all-off for each family member; 12B all-off
   remains very competitive, while adaptive was the best measured 26B prefill.
3. Do not rely on `auto`: current llama.cpp auto selection checks support and
   device placement, not measured performance.
4. For the 12B dense target, its matching MTP drafter is the largest measured
   decode win: 12.29 to 18.69 token/s in a deterministic local test (+52%).
5. Q4_0 decode should use the Vulkan integer-dot MMVQ kernel on Lunar Lake Xe2.
   This branch now does so; the old exclusion was inherited from A770/Alchemist
   tuning.  An earlier forced-kernel A/B improved ordinary decode about 3.2%; a
   later warm matched run measured 10.7% on 12B and 10.1% on 26B A4B.  Treat the
   lower figure as the conservative expectation because iGPU package power moves.
6. For prompt processing, `--ubatch-size 512` was the best tested value.  At a
   2K prompt, 128/256/512/1024 gave 194.6/213.7/222.0/218.4 token/s with FA off.
7. Quantized KV is a capacity choice here, not a free speedup.  In the local
   2K+64 test, F16+FA was 66.9 token/s, Q8_0 was 58.9, and Q4_0 was 59.1.

Google describes five Gemma 4 sizes and publishes approximate Q4_0 footprints
of 2.9, 4.5, 6.7, 14.4, and 17.5 GB for E2B, E4B, 12B, 26B A4B, and 31B.  Those
figures include a 20% loading allowance but not the growing KV cache.  See the
[official overview](https://ai.google.dev/gemma/docs/core) and
[official model card](https://ai.google.dev/gemma/docs/core/model_card_4).

## Architectural quirks that affect runtimes

### The whole family

- Attention is hybrid: five sliding-window layers followed by one global layer,
  repeated, with the final layer global.  Global layers use unified K/V and
  proportional RoPE (p-RoPE).  A runtime cannot assume one attention geometry
  for every block.
- E2B and E4B use Per-Layer Embeddings (PLE).  Their effective compute parameter
  count is much lower than their stored embedding-inclusive count.  Embedding
  lookup locality and memory footprint therefore matter more than the nominal
  `E2B`/`E4B` labels suggest.
- 12B is encoder-free and projects raw image patches and audio waveforms into the
  decoder embedding space.  Multimodal token count directly increases decoder
  work; there is no separate encoder cost to optimize away.
- 26B A4B has 128 routed experts plus one shared expert and activates eight
  routed experts per token.  It has about 25.2B stored and 3.8B active parameters.
  Expert routing, gather/scatter, and reuse across tokens dominate more than its
  `A4B` label implies.
- Every family member has a matching MTP draft model.  The drafter shares the
  target embedding table and consumes the target's final-layer activations; it
  is not an independent generic draft model.  The
  [official MTP guide](https://ai.google.dev/gemma/docs/mtp/overview) warns that
  verifying several MoE tokens can load different experts, so 26B A4B MTP may
  fail to win at batch size one even when dense-model MTP wins strongly.
- The medium models use a 256K trained context; E2B/E4B use 128K.  Context support
  is not a claim of flat latency or that the full KV cache fits beside every
  quant on every device.

### Exact local text-model geometries

| Property | 12B dense | 26B A4B MoE |
|---|---:|---:|
| Layers | 48 | 30 |
| Hidden size | 3840 | 2816 |
| Query heads | 16 | 16 |
| Local/global pattern | 5 + 1, repeated | 5 + 1, repeated |
| Local Q/K/V head dimension | 256 | 256 |
| Local KV heads / GQA | 8 / 2:1 | 8 / 2:1 |
| Global Q/K/V head dimension | 512 | 512 |
| Global KV heads / GQA | 1 / 16:1 | 2 / 8:1 |
| Sliding window | 1024 | 1024 |
| Global/local RoPE base | 1,000,000 / 10,000 | 1,000,000 / 10,000 |
| Dense FFN | 15360 | 2112 shared path |
| Routed experts | none | 128 total, 8 active, FFN 704 |
| Final logit softcap | 30 | 30 |
| Vocabulary | 262144 | 262144 |

The dual D=256/D=512 attention geometry is the central kernel quirk.  In F16,
the approximate text KV footprints are:

- 12B: global cache grows by 16 KiB/token (about 4 GiB at 256K); the 40 local
  layers plateau around 160 MiB after their 1024-token window is full.
- 26B A4B: global cache grows by 20 KiB/token (about 5 GiB at 256K); the 25 local
  layers plateau around 200 MiB.

These are payload estimates before allocator/alignment overhead.  They explain
why KV quantization can be necessary for capacity even when it loses speed.

## Prompt, tokenizer, thinking, and tool-use quirks

Gemma 4 is not Gemma 3 with a renamed template.  The
[official prompt-format document](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
defines new turn, modality, reasoning, and tool tokens.

- Roles are `system`, `user`, and `model`, enclosed by `<|turn>` and `<turn|>`.
  Use the GGUF chat template rather than hand-building an older Gemma template.
- Thinking is conversation-level and enabled by `<|think|>` in the system turn.
  For 12B/26B/31B, the thinking-off template deliberately inserts an empty
  thought channel to suppress “ghost” thought output.  See the
  [official thinking guide](https://ai.google.dev/gemma/docs/capabilities/thinking).
- Do not put prior hidden thoughts back into ordinary multi-turn history.  Keep
  only the final response, except for tool-call turns where the protocol requires
  preserving the thought/tool lifecycle.
- `<|tool_response>` is an additional stop/EOG token.  Function strings use the
  special `<|"|>` delimiter, not ordinary JSON quoting alone.
- `<|image|>` and `<|audio|>` are placeholders replaced by soft embeddings.
  The model card recommends image before text and audio after text.  Image token
  budgets are 70/140/280/560/1120; this is a direct compute/quality dial.
- The shipped GGUF says `add_bos_token=false`, but llama.cpp intentionally
  overrides BOS insertion for Gemma 4.  Do not add a second BOS externally.
- The recommended sampling defaults are temperature 1.0, top-p 0.95, top-k 64.
  Greedy output is useful for correctness/performance A/B tests, not the general
  quality configuration.

## Why Flash Attention is weird here

“Flash Attention supported” only means a backend can execute the fused op.  It
does not mean that the selected shader beats the backend's separately optimized
QK matmul, softmax, and V matmul for this GPU and shape.

### What Vulkan currently dispatches

- KHR cooperative-matrix FA for prompt/chunk rows when the device and shared
  memory checks pass.
- A scalar/subgroup shader for one-token decode because the source already knows
  it is faster than cooperative matrices at `N == 1`.
- GQA row folding only when the query token count is at most eight.  For normal
  prompt chunks, Gemma's global 8:1 or 16:1 MQA/GQA shape uses token-row tiles.
- The cooperative-matrix-1 tuning is a fixed Br16/Bc64, four-subgroup,
  128-thread workgroup.  It is not specialized for Intel D=512.
- `auto` reserves a one-token graph and verifies that FA remains on the same
  device as its layer.  It has no timing model, so it enables a slow but supported
  Intel path.  A current upstream report shows an analogous 2.3x Intel A770 Vulkan
  decode regression when auto enables FA: [llama.cpp #27137](https://github.com/ggml-org/llama.cpp/issues/27137).

### Local operation profile

For a 128-token 12B prompt with F16 KV and FA enabled:

| Layer shape | Count | Warm time/op | Aggregate | Effective rate |
|---|---:|---:|---:|---:|
| local D=256, KV heads=8 | 40 | 0.876 ms | 35.1 ms | 613 GFLOP/s |
| global D=512, KV heads=1 | 8 | 11.084 ms | 88.7 ms | 97 GFLOP/s |

Eight global layers therefore cost about 2.5x all forty local layers combined.
One D=512 op is about 12.6x slower than one D=256 op despite having only about
twice the attention FLOPs.  This is consistent with register/shared-memory
pressure and poor occupancy in the large-head cooperative path, compounded by
the extreme global MQA geometry.

End-to-end local measurements (same 12B QAT-derived Q4_0 model):

| Test | FA on | FA off | Result |
|---|---:|---:|---:|
| pp512, earlier cool run | 154.2 | 238.7 | off is 1.55x |
| 2K prompt + 64 generation | 66.9 | 142.6 | off is 2.13x |
| pp512, later sustained run | 124.3 | 222.9 | off is 1.79x |
| tg64, later sustained run | 11.96 | 12.66 | off is 1.06x |

The retained adaptive graph selector is a stronger result than forcing either
setting over every layer and phase:

| Model/test | All FA | Adaptive | Adaptive result |
|---|---:|---:|---:|
| 12B pp128 | 185.5 | 209.0 | +12.7% |
| 12B pp512, paired sustained runs | 134.4 | 195.6 | +45.6% |
| 12B pp512, later adaptive run | -- | 213.6 | thermal-range confirmation |
| 12B tg64 | 11.0 | 12.45 | +13.6% |
| 26B A4B pp512 | 208.0 | 281.6--288.1 | +35.4--38.5% |

Rates are token/s.  A separate 26B decode-only comparison measured adaptive
28.14 versus all-off 28.54 token/s, effectively parity within iGPU variance.
The 12B adaptive prompt result is close enough to the all-off baseline that both
should remain in a deployment A/B; the 26B prompt result clearly favored adaptive.

### Adaptive attention retained in this branch

When `LLAMA_VK_GEMMA4_HYBRID_FA=1` is set and `--flash-attn on` is requested,
the graph builder applies the policy to both `gemma4` and `gemma4-assistant`:

- fewer than 32 query tokens: decomposed attention for decode and speculative
  verification;
- 32--63 query tokens: ordinary FA selection, avoiding first-request overhead
  around the crossover;
- 64 or more query tokens: FA for D=256 local layers and decomposed attention
  for D=512 global layers.

The environment gate is intentional: graph construction does not know which
backend will ultimately execute each node, so making this Xe2 result the default
would risk regressing CUDA, Metal, or a future faster Vulkan driver.  This is a
shape-aware scheduler win rather than a new mathematical attention algorithm.

Absolute iGPU rates move with package power, temperature, display activity, and
memory contention.  The paired direction is stable.  Similar Lunar Lake SYCL
results in the [Intel backend discussion](https://github.com/ggml-org/llama.cpp/discussions/23313)
show FA cutting pp512 from roughly 536 to 246 token/s while barely changing
decode, so this is not unique to one GGUF.

At long context, decode attention becomes a different problem: insufficient
memory-level parallelism.  An Intel Xe2 report measured an almost shape- and
backend-independent 21–25 ns per KV position/layer and obtained about 2.4x
aggregate throughput with four streams: [llama.cpp #26581](https://github.com/ggml-org/llama.cpp/issues/26581).
Another Gemma 4 26B Vulkan report measured tg falling from 26.8 near the start to
10.6 at 32K: [llama.cpp #24005](https://github.com/ggml-org/llama.cpp/issues/24005).

### Kernel experiments performed for this work

The HT CUDA backend already contains D=512 vector FA instances and an important
guard learned from measurements: direct vector K/V decoding can beat staging at
small GQA, but loses badly at Gemma's global GQA=16 because it rereads K/V for
every query head.  It therefore keeps the tiled/MMA path for large-GQA Gemma.

Three analogous Vulkan Xe2 specializations were implemented and measured, then
removed because they regressed the whole model:

1. Reinterpret a 16-row tile as the GQA heads of one prompt token.  This was
   correct in layout but did not reduce the number of workgroups or FLOPs and was
   slightly slower.
2. Route only D=512 FA to the scalar kernel, and separately try a two-subgroup
   Br16/Bc32 cooperative tile.  The scalar variant reduced pp512 to 146.4 token/s;
   the narrow cooperative variant reached 147.7, both below the stock kernel.
3. A true Br32/Bc32 two-subgroup cooperative kernel kept each K and V fragment
   live while updating two 16-row accumulators.  It passed 34/34 D=512 backend
   comparisons, including Gemma's GQA=16 and Q8_0/Q4_0 KV shapes, but tied stock
   at about 118.3 ms/op at pp512.  The saved K/V loads were canceled by register
   pressure from the second cooperative accumulator.  Isolating it as separate
   SPIR-V also proved the generic D=256 shader must not absorb this experiment.

No slower experimental shader is retained.  A worthwhile new D=512 kernel must
reuse one global K/V head across 8/16 query heads *and* multiple prompt tokens,
keep enough independent memory requests in flight, and avoid materializing the
score matrix.  Merely changing tile width or relabeling rows is insufficient.
Useful implementation references are the existing
[CUDA FA dispatcher](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu),
[Vulkan FA shader](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp),
and [WebGPU split vector path](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_vec_split.wgsl).

## Ranked optimization plan

Scores are expected benefit on this Xe2, implementation simplicity, and risk.
“High potential” is not the same as “already proven.”

| Rank | Optimization | Expected gain | Simplicity | Risk / scope |
|---:|---|---|---|---|
| 1 | Adaptive per-shape attention (implemented) | Very high prefill; +35--46% paired | High | Opt-in because graph is backend-agnostic |
| 2 | Matching MTP drafter for 12B dense | Very high; +52% measured greedy | Medium | Low quality risk; exact verification |
| 3 | F16 KV + all-off as the 12B baseline | Very high prefill; decode win | Very high | Loses fused/quantized-V path advantages |
| 4 | Keep model, KV, and active experts resident; avoid spill | Catastrophic loss avoided | High | Capacity planning per family member |
| 5 | `--ubatch-size 512` for prompt chunks | Medium; +14% vs 128 at p2048 | Very high | Recheck for multimodal/parallel loads |
| 6 | Xe2 Q4_0 MMVQ decode dispatch (implemented) | Low/medium; 10% warm A/B | High | Gated to Intel Xe2 + Q4_0 |
| 7 | Prompt/prefix cache and SWA-aware reuse | Workload-dependent, potentially huge | High | Server/application behavior |
| 8 | MoE expert batching/fused route+gather+matmul for 26B | High potential | Low | Major Vulkan kernel work |
| 9 | Multi-stream scheduling at deep context | High aggregate; ~2.4x reported at 4 streams | Medium | Raises single-request latency/memory |
| 10 | Quantized KV for longer resident context | Capacity gain, negative speed locally | High | FA correctness and quality validation |
| 11 | MTP for 26B A4B | Variable; can lose at batch 1 | Medium | Expert-weight verification traffic |
| 12 | Cross-head D=512 MQA/GQA-reuse kernel | High theoretical potential | Low | Must beat adaptive matmul path |
| 13 | SYCL oneDNN/OpenVINO graph paths | Medium future potential | Low today | Toolchain plus Gemma support gaps |

For 26B A4B specifically, prioritize expert matmul/routing and batch reuse before
micro-tuning attention.  Locally FA off measured 257.9 pp512 / 30.35 tg64 versus
210.9 / 27.79 with FA on.  For E2B/E4B, profile PLE lookup and mobile quantized
weights separately.  For 31B, preserve remaining memory for its global KV cache;
its Q4 weights fitting does not imply a useful 256K context fits.

The retained MMVQ dispatch was also checked with the Q4_0 Vulkan `MUL_MAT`
backend suite: all 29 supported comparisons passed.  The model-level warm A/Bs
were 12.77 versus 11.54 tg64 on 12B and 30.56 versus 27.75 tg32 on 26B A4B.

## Recommended commands for this machine

Adaptive 12B/26B serving or benchmarking with F16 KV (best measured 26B
prefill; compare with all-off for 12B):

```bash
LLAMA_VK_GEMMA4_HYBRID_FA=1 ./build-vulkan/bin/llama-server \
  --model MODEL.gguf --n-gpu-layers all \
  --flash-attn on --ubatch-size 512
```

All-off baseline:

```bash
./build-vulkan/bin/llama-server \
  --model MODEL.gguf --n-gpu-layers all \
  --flash-attn off --ubatch-size 512
```

12B MTP (use the matching 12B assistant only):

```bash
./build-vulkan/bin/llama-server \
  --model gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  --spec-draft-model mtp-gemma-4-12B-it-Q4_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 16 --spec-draft-p-min 0.9 \
  --n-gpu-layers all --n-gpu-layers-draft all \
  --flash-attn off --ubatch-size 512
```

Use FA and quantized KV only after measuring the capacity/throughput trade:

```bash
./build-vulkan/bin/llama-server \
  --model MODEL.gguf --n-gpu-layers all --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0
```

Do not set `GGML_VK_DISABLE_COOPMAT=1` as a general workaround.  It also disables
fast cooperative-matrix projection/FFN kernels and substantially reduces prompt
throughput.  `GGML_VK_FORCE_MMVQ=1` is no longer needed for Q4_0 on Xe2 in this
branch; normal dispatch selects it.

## Correctness gates before accepting any kernel gain

1. Compare deterministic token IDs/logits against CPU and stock Vulkan with a
   short prompt, >1024-token prompt, and context with freed/reused KV cells.
2. Test all four attention geometries represented locally: 12B local 2:1 D256,
   12B global 16:1 D512, 26B local 2:1 D256, 26B global 8:1 D512.
3. Cover F16/F16, Q8_0/Q8_0, and Q4_0/Q4_0 K/V; mixed types if advertised.
4. Cover N=1 decode, N=2–16 speculative verification, and N=128/512 prompt tiles.
5. Test mask boundaries, SWA wraparound, prefix reuse, split-K, multiple sequences,
   tool-response EOG, thinking on/off, and MTP accept/reject paths.
6. Measure end-to-end pp/tg and operation time.  Reject microkernel wins that
   slow graph setup, matmuls, copies, or the whole model.

This caution is not hypothetical.  Recent reports include stale freed K/V cells
in Vulkan FA ([#26744](https://github.com/ggml-org/llama.cpp/issues/26744)),
quantized-V corruption on D=256 Adreno ([#26195](https://github.com/ggml-org/llama.cpp/issues/26195)),
and an earlier Intel Arc Gemma 4/MTP garbled-output report
([#24560](https://github.com/ggml-org/llama.cpp/issues/24560)).  These reports do
not prove the current Xe2 build is wrong; they define regression cases a new
kernel must survive.

## Backend alternatives: present status, not model substitution

- SYCL exposes Lunar Lake, F16, oneDNN, and fused SDPA, but FA remains
  model/shape-dependent.  Current setup instructions are in the
  [llama.cpp SYCL guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md).
  A recent report also documents silently wrong SYCL FA for quantized,
  non-contiguous KV views ([#27769](https://github.com/ggml-org/llama.cpp/issues/27769)).
- OpenVINO's graph fusion is attractive in principle, but its llama.cpp backend
  remains a narrower work in progress; Gemma 4 has had conversion/support gaps
  ([OpenVINO guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md),
  [Gemma 4 issue #24415](https://github.com/ggml-org/llama.cpp/issues/24415)).
- On the tested 26B A4B class, community data in the Intel discussion shows
  Vulkan can outperform SYCL decode substantially.  Backend choice must therefore
  be benchmarked per family member and phase, not declared from feature lists.

The practical development order is now: use adaptive attention and FA-off/F16 as
measured baselines; keep the low-risk Xe2 MMVQ dispatch; then pursue cross-head
D=512 reuse or MoE fusion only when a kernel demonstrates an end-to-end win over
the adaptive decomposed path, not merely over the old fused shader.
