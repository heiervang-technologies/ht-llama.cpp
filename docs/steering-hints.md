# Steering Hints: Mid-Inference Context Injection

> Design document for injecting user text into active inference at a given context
> position, creating overlapping activations that steer model output without
> interrupting reasoning. **"Telepathy for LLMs."**

**Issue:** [#17](https://github.com/heiervang-technologies/ht-llama.cpp/issues/17)
**Branch:** `feature/steering-hints`
**Target:** `ht`

---

## Table of Contents

1. [Concept](#concept)
2. [Literature Review](#literature-review)
3. [llama.cpp Architecture Analysis](#llamacpp-architecture-analysis)
4. [Upstream Landscape](#upstream-landscape)
5. [Identified Gaps](#identified-gaps)
6. [Proposed Design](#proposed-design)
7. [Implementation Plan](#implementation-plan)
8. [Open Questions](#open-questions)
9. [References](#references)

---

## Concept

During active token generation, a user provides text input. Instead of waiting
for the current turn to finish (prompt-level injection) or applying a static
pre-computed vector (control vectors), we:

1. **Tokenize** the hint as a fully-wrapped chat message (proper open+close tags)
2. **Shift** existing KV cache positions to create a gap at the injection point
3. **Decode** the hint tokens into the gap via `llama_decode()`
4. **Resume** generation — the model now attends to the hint through standard
   causal attention

The model perceives the hint as if it had always been part of the context at that
position. Already-generated tokens' K vectors are re-rotated via K-shift (RoPE
update). New tokens attend to the hint naturally. This is not prompt replay — it
is context-level "telepathy."

### How this differs from existing approaches

| Approach | Level | Dynamic? | Mid-gen? | Cost |
|----------|-------|----------|----------|------|
| System prompt / CLAUDE.md | Prompt | No | No | Full context replay |
| Gemini CLI steering hints | Prompt (queued) | Per-turn | No | Next turn replay |
| Control vectors (llama.cpp) | Activation | Per-session | No | Zero (bias add) |
| **Steering hints (this)** | **Context/KV** | **Per-token** | **Yes** | **K-shift + hint decode** |

---

## Literature Review

### Foundational: Additive Activation Steering Works

**Contrastive Activation Addition (CAA)** (Rimsky et al., 2023) proved that
adding vectors to residual stream activations reliably steers behavior:

```
v_l = (1/N) * Σ [a_l(p⁺) - a_l(p⁻)]     # mean activation difference
h' = h + α * v_l                           # applied at inference
```

Effective at layers 15-17 in Llama 2. Minimal capability degradation. But
vectors are static and context-agnostic.

**Representation Engineering (RepE)** (Zou et al., 2023) provides the theoretical
foundation: high-level concepts are linearly encoded in activation space and
controllable via additive intervention.

### Dynamic & Context-Aware Methods

| Method | Innovation | Dynamic? | Relevance |
|--------|-----------|----------|-----------|
| **Steering Vector Fields** (Li et al., 2026) | Differentiable scoring function `f(h)` — gradient defines per-activation steering direction | Yes, per-token | Mathematical framework for context-dependent steering |
| **SADI** (Wang et al., 2025) | Element-wise masking + input-adaptive scaling of own activations | Per-input | Shows steering can adapt to current hidden state |
| **FASB** (2025) | Classifier-driven intervention + token backtracking | Yes, mid-gen | **Demonstrates stop-inject-resume pattern** |
| **CAST** (Lee et al., 2025) | Condition vectors + behavior vectors (if-then steering) | Conditional | Separates "when" from "what" of steering |
| **EasySteer** (ZJU, 2025) | Production framework on vLLM, pluggable methods, 5-11x speedup | Framework | Engineering patterns for modular steering |
| **AI Steerability 360** (IBM, 2026) | Four control surfaces: input, structural, state, output | Framework | Composable pipeline architecture |

### Text-to-Steering-Vector (Real-Time)

**Latent Steering Vectors** (Subramani et al., 2022) proved text can be faithfully
represented as additive activation-space vectors (>99 BLEU reconstruction).

**llm_steer** (Mihaiii) is the closest practical implementation: takes user text,
runs a forward pass to get activations, uses those as a steering direction. Works
with LLaMA, Mistral, Phi via HuggingFace hooks.

**Steer2Adapt** (2026) shows steering vectors can be dynamically composed from a
basis via Bayesian optimization at inference time.

### KV Cache Manipulation Research

- **Shadow in the Cache** (2025): Demonstrated crafted KV entries can be appended
  to hijack generation — proves the injection mechanism works
- **Stateful KV Cache Management** (2025): Selective retention/eviction/modification
  of position-specific KV entries
- **Activation Patching**: Overwrites activations at arbitrary layers/positions;
  generation continues coherently

### Key Insight

The building blocks exist but have never been combined:

1. Additive intervention works (CAA, RepE)
2. Context-dependent steering works (SVF, SADI)
3. Mid-generation intervention works (FASB backtracking)
4. Text → steering vectors works (Latent Steering, llm_steer)
5. KV cache manipulation at specific positions works (llama.cpp primitives)

**No published work combines these into "inject user text at position X during
active generation."** This is a novel contribution.

---

## llama.cpp Architecture Analysis

### KV Cache: Position-Based Attention Masking (The Key Insight)

The causal attention mask in `llama-kv-cache.cpp:set_input_kq_mask_impl()` is
**purely position-based**:

```cpp
// For each token i in batch, for each cell j in KV cache:
if (p0 > p1) { goto skip; }  // p0 = KV cell position, p1 = token position
```

The mask does not care *when* a KV entry was written — only what position it
claims. If we insert a KV entry at position P with sequence S, any future token
at position > P in sequence S **will attend to it**.

### Control Vector Application Point

Control vectors are applied via `build_cvec()` at the END of each transformer
layer, after attention and FFN (`llama-iswa.cpp:156`):

```cpp
cur = build_cvec(cur, il);  // ggml_add(ctx, cur, layer_dir)
```

Currently per-layer, position-independent. The tensor is `[n_embd]` — same vector
for all tokens in the batch.

### Batch System: Arbitrary Position Assignment

`llama_batch` allows explicit per-token positions:

```c
typedef struct llama_batch {
    llama_token  *  token;    // token ids
    llama_pos    *  pos;      // arbitrary positions per token
    llama_seq_id ** seq_id;   // sequence membership
} llama_batch;
```

Combined with KV cache sequence operations:

```c
llama_memory_seq_add(mem, seq_id, p0, p1, delta);  // shift positions
llama_memory_seq_rm(mem, seq_id, p0, p1);           // remove range
llama_memory_seq_cp(mem, src, dst, p0, p1);         // branch sequence
```

### K-Shift Mechanism

When positions are shifted via `seq_add`, `has_shift` is set to true. On the next
`update()`, a compute graph re-rotates all shifted K vectors with correct RoPE
embeddings. This is the same mechanism used for context-shift and is already
optimized.

**Important:** V vectors are NOT re-rotated (they are position-independent in
most architectures). Already-generated tokens' V vectors were computed without
the hint's influence. Only NEW tokens generated after injection will have their
computations influenced by the hint through attention.

### Position Continuity Invariant

`apply_ubatch()` enforces that all positions in `[pos_min, pos_max]` must exist
for each sequence. After a `seq_add` shift creates a gap, the gap positions
**MUST** be filled before the next decode of continuation tokens, or the invariant
violation will trigger a purge of existing entries.

### Chat Template Wrapping

To wrap a steering hint as a proper chat message:

```cpp
llama_chat_message msg = {"user", hint_text};
std::vector<const llama_chat_message *> chat = {&msg};
std::string formatted;
llm_chat_apply_template(detected_template, chat, formatted, false);

// Tokenize with special token parsing
auto tokens = common_tokenize(ctx, formatted, false, true);
```

Setting `parse_special=true` correctly tokenizes template special tokens
(`<|im_start|>`, `[INST]`, etc.).

### Server Architecture

The server's `update_slots()` loop in `server-context.cpp` runs synchronously:

1. Build batch from all generating slots
2. Decode batch
3. Sample next token per slot
4. Check stop conditions

**Best injection point:** Between sampling (step 3) and the next iteration's
batch building (step 1). A new task type (`SERVER_TASK_TYPE_INJECT_HINT`) can
be processed in the task queue between generation steps.

---

## Upstream Landscape

### What exists in upstream llama.cpp

| Feature | Status | Reference |
|---------|--------|-----------|
| Control vectors (static, per-session) | Merged | PR #5970 |
| Native cvector generation | Merged | Issue #6880, PR #7514 |
| KV cache seq operations (rm/cp/add) | Merged | PR #3228 |
| Abort callback for decode | Merged | PR #10571 |
| LoRA per-request (hot-swap) | Merged | Issue #10377 |
| Custom sampler chain | Merged | Discussion #3665 |
| Eval callback (`cb_eval`) | Merged | In `llama.h` |

### What is requested but NOT implemented upstream

| Feature | Status | Reference |
|---------|--------|-----------|
| Server support for control vectors | Open | Issue #6316 |
| Hot-swap control vectors via API | Open | Issue #10685 |
| Dynamic per-request control vectors | Not requested | — |
| Mid-generation context injection | Not requested | — |
| Steering hints | Not requested | — |

### Related external projects

- **vLLM Hook v0** (Ko & Chen, 2026): Plugin for programming model internals
  during inference in vLLM. Active + passive modes. Architecture reference for
  hook-based steering. Python/PyTorch only.
- **llm_steer**: Text-to-steering-vector via HuggingFace hooks
- **jukofyork/control-vectors**: GGUF control vector generation tool
- **antislop-sampler**: Token regeneration on pattern match (sampling-level steering)

---

## Identified Gaps

### Gap 1: No mid-generation injection API

There is no `llama_inject_hint()` or equivalent. Must be composed from existing
primitives: `seq_add` + batch construction + `llama_decode`.

### Gap 2: No per-position control vectors

`llama_adapter_cvec` applies the same vector to ALL tokens. Cannot apply stronger
steering to hint tokens vs. context tokens. Tensor is `[n_embd]` per layer; would
need `[n_embd, n_tokens]` or position-masked scaling.

### Gap 3: No server task type for hint injection

`server_task_type` enum has no injection variant. The server task queue and slot
state machine need extension.

### Gap 4: K-shift model compatibility

Models without RoPE (ALiBi, etc.) do not support K-shift. The `get_can_shift()`
method checks this. SWA (sliding window attention) caches also cannot support
position shifting or token removal.

### Gap 5: No text-to-steering-vector without forward pass

Converting arbitrary user text to a steering direction currently requires a full
forward pass (as in llm_steer). A lightweight encoder mapping text to activation
space would be needed for near-zero-cost steering, but this is a future
optimization.

---

## Proposed Design

### Phase 1: KV Cache Injection (Context-Level Steering)

The core mechanism — inject properly-wrapped user text into the KV cache at a
target position during active generation.

#### API

```c
// New public API
LLAMA_API int32_t llama_steering_hint_inject(
    struct llama_context * ctx,
    llama_seq_id           seq_id,
    llama_pos              inject_pos,    // -1 = current position
    const llama_token    * tokens,
    int32_t                n_tokens);
```

#### Algorithm

```
1. Pause generation (between decode steps)
2. Shift: llama_memory_seq_add(mem, seq_id, inject_pos, -1, n_tokens)
   → Opens gap at [inject_pos, inject_pos + n_tokens)
   → Triggers K-shift (RoPE re-rotation) on next update()
3. Build batch with hint tokens at gap positions:
   for i in 0..n_tokens:
     batch.token[i] = tokens[i]
     batch.pos[i]   = inject_pos + i
     batch.seq_id[i] = {seq_id}
4. llama_decode(ctx, batch)
   → Fills gap with hint KV entries
   → Position continuity invariant satisfied
5. Resume generation from shifted position
```

#### Chat Template Helper

```c
// Convenience: wrap text and tokenize with chat template
LLAMA_API int32_t llama_steering_hint_prepare(
    struct llama_context * ctx,
    const char           * role,          // "user" or "system"
    const char           * text,
    llama_token          * out_tokens,
    int32_t                max_tokens);
```

### Phase 2: Server Integration

#### New endpoint

```
POST /v1/steering/inject
{
  "id_slot": 0,          // or inferred from active generation
  "text": "focus on error handling",
  "role": "user",        // or "system"
  "position": -1         // -1 = current, or explicit position
}
```

#### Server task flow

```
1. Client sends POST /v1/steering/inject
2. Server creates SERVER_TASK_TYPE_INJECT_HINT task
3. In update_slots(), before building generation batch:
   a. Check for pending injection tasks
   b. For each: call llama_steering_hint_inject()
   c. Update slot's n_past to account for shifted positions
4. Continue normal generation loop
```

### Phase 3: Dynamic Control Vectors (Future)

Extend `llama_adapter_cvec` for per-position application:

```c
// Apply control vector only to tokens in position range
LLAMA_API int32_t llama_set_adapter_cvec_ranged(
    struct llama_context * ctx,
    const float * data,
    size_t len,
    int32_t n_embd,
    int32_t il_start,
    int32_t il_end,
    llama_pos pos_start,
    llama_pos pos_end);
```

And per-request hot-swap via server API (building on upstream issue #10685).

---

## Implementation Plan

### Phase 1: Core Mechanism

1. **`src/llama-steering.h`** — New header with `llama_steering_hint_inject()` and
   `llama_steering_hint_prepare()` declarations
2. **`src/llama-steering.cpp`** — Implementation using existing KV cache primitives
3. **`include/llama.h`** — Public API additions
4. **`common/arg.cpp`** — CLI flag `--steering-hints` to enable in interactive mode
5. **`tools/completion/completion.cpp`** — Interactive mode integration (type during
   generation)

### Phase 2: Server

6. **`tools/server/server.cpp`** — New task type, endpoint, slot integration
7. **`tools/server/public/`** — WebUI hint input during streaming

### Phase 3: Testing & Validation

8. **`tests/test-steering-hints.cpp`** — Unit tests for injection mechanics
9. **Perplexity benchmarks** — Measure quality impact of hint injection
10. **Latency benchmarks** — Measure K-shift + decode overhead

---

## Open Questions

1. **Optimal injection position:** Should hints go at current position (immediate
   effect) or at a position slightly behind (gives model "runway" to integrate)?

2. **Multiple hints:** If user sends several hints rapidly, should they be batched
   into a single injection or applied sequentially?

3. **Hint expiry:** Should injected hints be evicted after N tokens of generation
   to prevent context pollution? Or treated as permanent context?

4. **Role wrapping:** Should hints use `user` role (natural for user input) or
   `system` role (less likely to trigger conversational response patterns)?

5. **SWA compatibility:** Models with sliding window attention will naturally
   "forget" hints once they slide out of window. Is this acceptable or should
   hints be pinned?

6. **Causality semantics:** The model sees hints as if they were always there, but
   already-generated V vectors were computed without hint influence. Is this
   acceptable, or should we regenerate K tokens after injection (FASB-style
   backtracking)?

---

## References

### Research Papers

- Rimsky et al. (2023). [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681). ACL 2024.
- Zou et al. (2023). [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405).
- Subramani et al. (2022). [Extracting Latent Steering Vectors from Pretrained Language Models](https://arxiv.org/abs/2205.05124).
- Li et al. (2026). [Steering Vector Fields for Context-Aware Inference-Time Control](https://arxiv.org/abs/2602.01654).
- Wang et al. (2025). [Semantics-Adaptive Dynamic Intervention for LLMs via Dynamic Steering Vectors](https://arxiv.org/abs/2410.12299). ICLR 2025.
- (2025). [Steering When Necessary: Flexible Activation Steering with Backtracking](https://arxiv.org/abs/2508.17621) (FASB).
- Lee et al. (2025). [Programming Refusal with Conditional Activation Steering](https://arxiv.org/abs/2409.05907). ICLR 2025.
- ZJU-REAL (2025). [EasySteer: Unified Framework for High-Performance LLM Steering](https://arxiv.org/abs/2509.25175).
- IBM (2026). [AI Steerability 360: A Toolkit for Steering LLMs](https://arxiv.org/abs/2603.07837).
- Ko & Chen (2026). [vLLM Hook v0](https://arxiv.org/abs/2603.06588).
- (2026). [Steer2Adapt](https://arxiv.org/abs/2602.07276).

### Upstream llama.cpp

- [Issue #1460](https://github.com/ggml-org/llama.cpp/issues/1460) — Steering vectors investigation
- [PR #5970](https://github.com/ggml-org/llama.cpp/pull/5970) — Control vector support (merged)
- [Issue #6880](https://github.com/ggml-org/llama.cpp/issues/6880) — Native cvector generation
- [Issue #6316](https://github.com/ggml-org/llama.cpp/issues/6316) — Server control vector support (open)
- [Issue #10685](https://github.com/ggml-org/llama.cpp/issues/10685) — Hot-swap control vectors API (open)
- [PR #3228](https://github.com/ggml-org/llama.cpp/pull/3228) — KV cache redesign with seq operations
- [PR #10571](https://github.com/ggml-org/llama.cpp/pull/10571) — Generic abort callback
- [Issue #10377](https://github.com/ggml-org/llama.cpp/issues/10377) — LoRA per-request

### Agentic CLI Steering (Prompt-Level)

- [Gemini CLI #18782](https://github.com/google-gemini/gemini-cli/issues/18782) — Steering hints feature
- [Gemini CLI PR #19307](https://github.com/google-gemini/gemini-cli/pull/19307) — Implementation
- [Gemini CLI #17197](https://github.com/google-gemini/gemini-cli/issues/17197) — `/inject` command proposal

### Tools & Libraries

- [Mihaiii/llm_steer](https://github.com/Mihaiii/llm_steer) — Text-to-steering via HuggingFace
- [jukofyork/control-vectors](https://github.com/jukofyork/control-vectors) — GGUF control vector generation
- [nrimsky/CAA](https://github.com/nrimsky/CAA) — Contrastive Activation Addition reference
- [ZJU-REAL/EasySteer](https://github.com/ZJU-REAL/EasySteer) — Unified steering framework
- [IBM/AISteer360](https://github.com/IBM/AISteer360) — Steerability toolkit
