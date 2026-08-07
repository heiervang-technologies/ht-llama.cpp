# MTP speculative decoding on Pascal (Quadro P5200, sm_61)

Measured results for `--spec-type draft-mtp` on a compute-capability 6.1 GPU,
plus two behaviours that are easy to hit and are not obvious from the option
list.

All numbers below were recomputed from the raw per-request timings in
[`data/pascal-p5200-2026-08-02/`](data/pascal-p5200-2026-08-02/), and both
cautions were reproduced from scratch with the scripts in
[`data/repro-2026-08-07/`](data/repro-2026-08-07/) rather than carried over
from a summary.

## Test setup

| | |
| --- | --- |
| GPU | NVIDIA Quadro P5200 Mobile, 16 GiB, compute capability **6.1**, driver 580.159.04 |
| CPU | Intel Core i7-7820HQ, 4C/8T |
| Build | `b9862-798cf6cbe`, `CMAKE_CUDA_ARCHITECTURES=61`, `GGML_CUDA_FORCE_MMQ=ON`, `GGML_CUDA_F16=OFF` |
| Target | `gemma-4-12b-it-qat-q4_0.gguf` — 6,975,879,296 B, `93567e57a8fe10b2…` |
| Draft | `mtp-gemma-4-12B-it-Q4_0.gguf` — 253,708,960 B, `b894e614824dfc27…` |
| Server | one slot, 4096-token context, full offload, flash attention, greedy sampling |
| Samples | three independent 128-token requests per mode |

Both GGUFs were exact-size and SHA-256 verified before testing. The target is
`google/gemma-4-12B-it-qat-q4_0-gguf`; the assistant is the QAT-derived
`gemma4-assistant` GGUF from `ggml-org/gemma-4-12B-it-GGUF`.

## Results

| Mode | Generation | Prompt processing | Peak VRAM | Peak GPU | Peak power |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target only | 24.02 tok/s | 159.77 tok/s | 7,524 MiB | 98% | 149.10 W |
| MTP | **56.44 tok/s** | 156.16 tok/s | 7,824 MiB | 94% | 172.38 W |

**2.35x generation throughput** for **+300 MiB VRAM** and **+23 W**. Prompt
processing is unchanged within sample spread.

Per-sample generation rates were tight in both modes — baseline 24.04 / 24.01 /
24.01 tok/s, MTP 56.51 / 56.48 / 56.34 tok/s — so the ratio is not an artifact
of one fast run. Draft acceptance was **276 of 333 proposed tokens (82.88%)**,
i.e. 92 of 111 in each of the three runs.

On a 16 GiB card where VRAM is the binding constraint, +300 MiB for 2.35x is a
favourable trade; the assistant model itself is only 254 MB on disk.

## Caution 1 — `-md` without `--spec-type` makes a healthy server that 500s

Passing an MTP assistant via `-md` **without** `--spec-type draft-mtp` does not
fall back to plain generation. The server auto-selects a different speculative
implementation, which the MTP head model cannot satisfy:

```
W srv  load_model: [spec] failed to measure draft model memory: failed to create llama_context from model
I srv  load_model: loading draft model '.../mtp-gemma-4-12B-it-Q4_0.gguf'
W common_speculative_init: draft model is specified but 'draft' speculative type is not explicitly enabled - enabling it
I common_speculative_impl_draft_simple: adding speculative implementation 'draft-simple'
I common_speculative_impl_draft_simple: - n_max=3, n_min=0, p_min=0.000000
```

Startup then completes, and `GET /health` reports healthy. **Every completion
request fails:**

```
HTTP 500
{"error":{"code":500,"message":"decode() failed: failed to process speculative batch","type":"server_error"}}
```

So the tell is not slow generation — it is a server that passes its health
check and cannot serve a single request. The earlier
`failed to create llama_context from model` line is logged at warning level and
is not treated as fatal.

Reproduced with `data/repro-2026-08-07/mtp-g2.sh`; the response body above is
`data/repro-2026-08-07/G2.1.json`.

## Caution 2 — greedy output is not reproducible across requests by default

With `cache_prompt` at its default of `true`, repeated **identical greedy
requests** to the same server do not all return the same text. The first
request differs from the rest.

The cause is prompt cache reuse changing the prompt batch split, not sampling:

| Request | `cache_n` | `prompt_n` |
| --- | ---: | ---: |
| 1st | 0 | 34 |
| 2nd and later | 7 | 27 |

A different batch shape gives a different floating-point reduction order, which
can flip a greedy token, after which the continuations diverge.

Six conditions, five identical greedy requests each (`temperature: 0`,
`top_k: 1`, `seed: 42`):

| | Configuration | Distinct outputs / 5 |
| --- | --- | ---: |
| A | MTP, draft backend sampling **enabled** (default) | 2 |
| B | MTP, draft backend sampling **disabled** | 2 |
| C | target only, no draft model (control) | 2 |
| D | target only, `-bs` (main backend sampling on) | 2 |
| E | target only, `cache_prompt: false` | **1** |
| F | MTP, `cache_prompt: false` | **1** |

Three things worth reading off that table:

- **A and B produced byte-identical output sets.**
  `--no-spec-draft-backend-sampling` changed nothing. Backend sampling is not
  the cause.
- **The control diverges too.** C has no draft model at all, so MTP is not the
  cause either. C and D are likewise byte-identical to each other.
- **`cache_prompt: false` makes it fully repeatable**, in both the target-only
  and the MTP condition.

In every condition the split was the same: run 1 differed, runs 2–5 were
identical to one another — matching the cache-state table above exactly.

If you need reproducible greedy output across requests, send
`"cache_prompt": false`, or ensure every request starts from the same cache
state. Nothing here is specific to sm_61; the same reasoning applies wherever
batch shape affects reduction order.

## What "deterministic" does and does not mean here

Holding cache state fixed, MTP is repeatable: all five runs in condition F were
byte-identical.

The MTP output was **not** byte-identical to the target-only output (condition F
vs condition E). This is expected rather than a defect: the target still
verifies every proposed token, but batched target evaluation can select a
different greedy token than single-token evaluation because GPU floating-point
reduction order differs.

So this result demonstrates verified-token speculation and run-to-run
repeatability. It should **not** be described as bit-identical to
non-speculative generation.

## Reproducing

```sh
./build-cuda/bin/llama serve \
  -m /path/to/gemma-4-12b-it-qat-q4_0.gguf \
  -md /path/to/mtp-gemma-4-12B-it-Q4_0.gguf \
  -c 4096 -ngl all -ngld all -fa on --parallel 1 \
  --spec-type draft-mtp --spec-draft-n-max 16 --spec-draft-p-min 0.9 \
  --host 127.0.0.1 --port 8080 --no-webui --jinja
```

Gemma's embedded chat template requires `--jinja`; without it the legacy
template path reports that the custom template is unsupported.

## Scope

Single device, single model pair, `--parallel 1`. Speculative gains normally
shrink as concurrency rises, because the target batch fills with real work, so
**2.35x is an upper bound for a multi-slot server**, not a general figure. No
multi-slot measurement was taken.

## Raw data

[`data/pascal-p5200-2026-08-02/`](data/pascal-p5200-2026-08-02/) — the
throughput run, verbatim:

- `README.md` — the original run report, including a same-build `llama-bench`
  pass over Qwen3.6-35B-A3B Q4_K_M and Gemma 4 26B-A4B QAT Q4_0 on the same GPU
- `gemma12-baseline-bench.jsonl`, `gemma12-mtp-bench.jsonl` — three API
  responses per mode with llama.cpp timings and MTP acceptance counters
- `gemma12-baseline-telemetry.csv`, `gemma12-mtp-telemetry.csv` — 500 ms
  `nvidia-smi` samples
- `gemma12-*-smoke.json`, `gemma12-mtp-cpu-sampler-*.json` — smoke and
  determinism-isolation runs
- `qwen-*`, `gemma-*` — the `llama-bench` pass and its telemetry

[`data/repro-2026-08-07/`](data/repro-2026-08-07/) — the caution repros:

- `mtp-determinism.sh` — conditions A–D, including the no-draft control
- `mtp-determinism2.sh` — conditions E–F, the `cache_prompt: false` test
- `mtp-g2.sh` — caution 1, `-md` without `--spec-type`
- `{A..F}.{1..5}.json` — every response, with timings and cache counters
- `G2.1.json`, `G2.server.log.excerpt` — the 500 and the startup log
- `summary.txt` — condition-by-condition distinct-output counts
