# Pascal CUDA benchmark: Qwen3.6-35B-A3B and Gemma 4

Tested on 2026-08-02 with the `ht` branch of ht-llama.cpp.

## Runtime

- ht-llama.cpp commit: `798cf6cbe56440132df23eb2318f587b16e3c00e`
- build: `b9862-798cf6cbe`
- GPU: NVIDIA Quadro P5200, 16 GiB, compute capability 6.1
- CPU: Intel Core i7-7820HQ, 4 cores / 8 threads
- CUDA build settings:
  - `CMAKE_CUDA_ARCHITECTURES=61`
  - `GGML_CUDA_FORCE_MMQ=ON`
  - `GGML_CUDA_F16=OFF`
- benchmark settings: 4 CPU threads, flash attention on, automatic device fitting,
  1024 MiB VRAM safety target, 4096-token fit context, three measured repetitions

## Verified models

| Model | File size | SHA-256 |
| --- | ---: | --- |
| Qwen3.6-35B-A3B Q4_K_M | 20,419,565,568 bytes | `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7` |
| Gemma 4 26B-A4B IT QAT Q4_0 | 14,439,363,584 bytes | `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` |

Both files passed `sha256sum --check` before testing.

## Results

| Model | PP128 | PP512 | TG32 | TG128 | Warm TG128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B Q4_K_M | 225.72 tok/s | 445.85 tok/s | 30.35 tok/s | 27.50 tok/s | 33.73 tok/s |
| Gemma 4 26B-A4B Q4_0 | 503.47 tok/s | 861.37 tok/s | 52.23 tok/s | 51.52 tok/s | 51.80 tok/s |

`PP` is prompt processing and `TG` is token generation. The normal result columns
are arithmetic means of all three samples. Qwen's first TG128 sample was cold-page
limited at 15.06 tok/s; its next two samples were 33.12 and 34.33 tok/s, whose mean
is shown as Warm TG128. Gemma's TG128 samples were stable at 50.95, 51.84, and
51.76 tok/s.

## Device placement and telemetry

| Model | Placement | Peak VRAM | Peak GPU | Peak power | Peak temperature |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6 | 41/41 layers use CUDA; 13 layers partially overflow to host; 14,146 MiB CUDA model buffer | 15,036 MiB | 100% | 161.82 W | 64 C |
| Gemma 4 | 31/31 layers fully offloaded; 13,755 MiB CUDA model buffer; 577.5 MiB CPU-mapped model data | 14,652 MiB | 100% | 155.68 W | 66 C |

Across samples with more than 1 GiB VRAM allocated, average GPU utilization was
30.9% for Qwen and 83.9% for Gemma. Qwen's host overflow and file paging account
for the lower utilization and larger cold-run variance. Gemma fits almost entirely
in VRAM and is correspondingly steadier.

## Smoke tests

Both models loaded, generated tokens, and exited with status 0 using the same
Pascal CUDA build. Single-turn checks produced `QWEN_SMOKE_OK` and
`GEMMA_SMOKE_OK`. Gemma's embedded chat template requires `--jinja`; without it,
the legacy template path reports that the custom template is unsupported.

Representative command:

```sh
./build-cuda/bin/llama-bench \
  -m /home/me/Models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -p 128,512 -n 32,128 -r 3 -t 4 \
  -fa on -fitt 1024 -fitc 4096 --progress -o jsonl
```

For Gemma chat/completion, add `--jinja`:

```sh
./build-cuda/bin/llama completion \
  -m /home/me/Models/gemma-4-26B_q4_0-it.gguf \
  -c 2048 -ngl auto -fit on -fitt 1024 -fa on --jinja -st \
  -p 'Your prompt' -n 256
```

Raw data:

- `qwen-bench.jsonl` and `gemma-bench.jsonl`: llama-bench results and samples
- `qwen-telemetry.csv` and `gemma-telemetry.csv`: 500 ms `nvidia-smi` samples
- `qwen-bench.log` and `gemma-bench.log`: benchmark progress and device detection

## Gemma 4 12B official QAT + MTP

The official Google QAT target and the matching QAT-derived MTP assistant were
also installed and tested with the same `ht` Pascal build:

| Role | File | Size | SHA-256 |
| --- | --- | ---: | --- |
| Target | `gemma-4-12b-it-qat-q4_0.gguf` | 6,975,879,296 bytes | `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b` |
| MTP assistant | `mtp-gemma-4-12B-it-Q4_0.gguf` | 253,708,960 bytes | `b894e614824dfc2746b26d3c3ba78c50000a464382682502392b4325257b7602` |

Both files were exact-size checked and SHA-256 verified after download. The target
is from `google/gemma-4-12B-it-qat-q4_0-gguf`; the assistant is the QAT-derived
`gemma4-assistant` GGUF from `ggml-org/gemma-4-12B-it-GGUF`.

### Server generation benchmark

The test used one server slot, a 4096-token context, full GPU offload, flash
attention, greedy sampling, and three independent 128-token requests.

| Mode | Generation | Prompt processing | Peak VRAM | Peak GPU | Peak power |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target only | 24.02 tok/s | 159.77 tok/s | 7,524 MiB | 98% | 149.10 W |
| QAT MTP | 56.44 tok/s | 156.16 tok/s | 7,824 MiB | 94% | 172.38 W |

MTP improved generation throughput by **2.35x**. Every MTP run accepted 92 of
111 proposed draft tokens (82.88%), and all three MTP responses were byte-identical
to one another. All three target-only responses were also byte-identical to one
another.

The required MTP launch options are:

```sh
./build-cuda/bin/llama serve \
  -m /home/me/Models/gemma-4-12b-it-qat-q4_0.gguf \
  -md /home/me/Models/mtp-gemma-4-12B-it-Q4_0.gguf \
  -c 4096 -ngl all -ngld all -fa on --parallel 1 \
  --spec-type draft-mtp --spec-draft-n-max 16 --spec-draft-p-min 0.9 \
  --no-spec-draft-backend-sampling \
  --host 127.0.0.1 --port 8080 --no-webui --jinja
```

Two Pascal-specific cautions were confirmed. Supplying `-md` without the explicit
`--spec-type draft-mtp` does not enable MTP. Also, draft backend sampling produced
non-repeatable greedy output on this P5200; `--no-spec-draft-backend-sampling`
made repeated MTP runs deterministic without materially reducing speed.

The deterministic MTP response was not byte-identical to the target-only response.
The target still verifies proposed tokens, but batched target evaluation can choose
a different greedy token from single-token evaluation because GPU floating-point
evaluation order differs. Therefore this local result demonstrates verified-token
speculation and repeatability, but it should not be described as bit-identical to
non-speculative generation.

Additional raw data:

- `gemma12-baseline-bench.jsonl` and `gemma12-mtp-bench.jsonl`: three API responses
  per mode, including llama.cpp timing and MTP acceptance counters
- `gemma12-baseline-telemetry.csv` and `gemma12-mtp-telemetry.csv`: 500 ms GPU samples
- `gemma12-*-smoke.json` and `gemma12-mtp-cpu-sampler-*.json`: smoke and
  determinism-isolation runs
