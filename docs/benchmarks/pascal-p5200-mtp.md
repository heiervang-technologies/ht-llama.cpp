# MTP speculative decoding on Pascal (Quadro P5200, sm_61)

Measured results for `--spec-type draft-mtp` on a compute-capability 6.1 GPU,
plus two Pascal-specific behaviours that are easy to hit and are not obvious
from the option list.

All numbers below were recomputed from the raw per-request timings in
[`data/pascal-p5200-2026-08-02/`](data/pascal-p5200-2026-08-02/), not copied
from a summary.

## Test setup

| | |
| --- | --- |
| GPU | NVIDIA Quadro P5200 Mobile, 16 GiB, compute capability **6.1** |
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

## Pascal caution 1 — `-md` alone does not enable MTP

Supplying a draft model without an explicit `--spec-type draft-mtp` does not
turn MTP on. The server starts, loads both models and serves normally, so the
failure mode is silent: you get baseline throughput and no error.

Confirm from the response `timings`: if `draft_n` is `0`, MTP is not running.

## Pascal caution 2 — draft backend sampling is non-repeatable on sm_61

`--spec-draft-backend-sampling` defaults to **enabled**
(`common/common.h`: `backend_sampling = true`, "offload draft sampling to the
backend"). With it enabled on this P5200, repeated **greedy** requests with
identical input produced **different** outputs.

Passing `--no-spec-draft-backend-sampling` made repeated MTP runs byte-identical
to one another without materially reducing speed — the 56.44 tok/s above was
measured with backend sampling disabled.

This is reported as an observation on one sm_61 device, not as a diagnosis. It
has not been bisected, and no claim is made about other architectures. The
isolation runs are in `gemma12-mtp-cpu-sampler-1.json` and
`gemma12-mtp-cpu-sampler-2.json`.

Note that the flag is currently absent from the option list in
[`../speculative.md`](../speculative.md); it is added there by the same change
that adds this page.

## What "deterministic" does and does not mean here

With `--no-spec-draft-backend-sampling`, all three MTP responses were
byte-identical **to one another**, and all three target-only responses were
byte-identical to one another.

The MTP response was **not** byte-identical to the target-only response. This is
expected rather than a defect: the target still verifies every proposed token,
but batched target evaluation can select a different greedy token than
single-token evaluation because GPU floating-point reduction order differs.

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
  --no-spec-draft-backend-sampling \
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

[`data/pascal-p5200-2026-08-02/`](data/pascal-p5200-2026-08-02/) — verbatim, as
produced by the run:

- `README.md` — the original run report, including a same-build `llama-bench`
  pass over Qwen3.6-35B-A3B Q4_K_M and Gemma 4 26B-A4B QAT Q4_0 on the same GPU
- `gemma12-baseline-bench.jsonl`, `gemma12-mtp-bench.jsonl` — three API
  responses per mode with llama.cpp timings and MTP acceptance counters
- `gemma12-baseline-telemetry.csv`, `gemma12-mtp-telemetry.csv` — 500 ms
  `nvidia-smi` samples
- `gemma12-*-smoke.json`, `gemma12-mtp-cpu-sampler-*.json` — smoke and
  determinism-isolation runs
- `qwen-*`, `gemma-*` — the `llama-bench` pass and its telemetry
