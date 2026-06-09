# Pascal Quadro P5200 — ht-llama.cpp build notes

## Hardware
- GPU: NVIDIA Quadro P5200, 16 GB GDDR5, **CC 6.1 (sm_61)**, GP104 mobile (2560 cores, ~230 GB/s)
- Host: crystal (Arch Linux, Intel i7-7820HQ 4c/8t, 16 GB RAM, driver 580.159)

## Two facts that drive every decision (per primer)
1. **FP16 is 1/64 of FP32** on Pascal — never enable raw-FP16 fast paths. Use FP32 accumulation.
2. **Decode is bandwidth-bound** at ~230 GB/s — quantize to reduce bytes-per-token; INT8 DP4A is the fast path.

## Toolkit compatibility wall
- Arch `extra/cuda 13.2.1-3` **drops sm_61**: `nvcc fatal: Unsupported gpu architecture compute_61`.
- Driver 580 still **runs** sm_61 binaries (`nvidia-smi` works fine); it is the **toolkit** (nvcc + cuBLAS) that dropped support in CUDA 13.0.
- Resolution: install `aur/cuda-pascal 12.9.1-4` (last toolkit with Pascal support; pulls `aur/gcc14` as compatible host compiler).

## Build commands

### CUDA (preferred on Pascal)
```bash
export PATH=/opt/cuda-pascal/bin:$PATH        # verify in cuda-pascal PKGBUILD
cmake -B build-cuda -GNinja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_F16=OFF \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j 8 --target llama-bench llama-app
```
Rationale:
- `CMAKE_CUDA_ARCHITECTURES=61` — exact P5200 cap; faster compile.
- `GGML_CUDA_FORCE_MMQ=ON` — routes quantized matmul through `__dp4a` (the fast INT8 path on Pascal). Verify at startup: `ggml_cuda_init: GGML_CUDA_FORCE_MMQ: yes`.
- `GGML_CUDA_F16=OFF` — Pascal FP16 ALU is 1/64 of FP32; FP32 accumulators win.
- Flash-attention: `-fa on` works on Pascal via the no-tensor-core FA kernel with FP32 accumulation (issue #7055 fixed in #7188).

### Vulkan (baseline / portable fallback)
```bash
cmake -B build-vulkan -GNinja \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j 8 --target llama-bench llama-app
```
Pascal Vulkan startup reports `matrix cores: none` — no coopmat path. No usable Vulkan flash-attention on Pascal (requires coopmat2/Ampere+). Quantized KV (`-ctk q8_0`) also unavailable without FA.

## Benchmark
```bash
scripts/bench-pascal-p5200.sh both   # or `cuda` / `vulkan`
# Cross-product runs: pp 128/512/2048 × tg 32/128 × fa 0,(1 for CUDA), 3 reps.
```

## Results — Vulkan baseline (build f6feddb, Llama-3.1-8B-Instruct Q4_K_M, ngl=99, fa=0)

| test    | t/s    | stddev |
|---------|--------|--------|
| pp128   | 269.49 | 0.90   |
| pp512   | 277.78 | 0.10   |
| pp2048  | 250.92 | 0.62   |
| tg32    | 35.18  | 0.10   |
| tg128   | 35.22  | 0.03   |

Notes:
- `tg` flat across batch sizes — bandwidth-bound, as primer predicts.
- `pp` peaks at 512 then dips at 2048 — Pascal Vulkan has no FA so longer ctx suffers more from attention quadratic.

## Results — CUDA (pending cuda-pascal install)
TODO — fill in once `aur/cuda-pascal 12.9.1` finishes building. Primer predicts CUDA pp ≫ Vulkan pp on Pascal; tg comparable (decode is bandwidth-bound). `-fa 1` should give a small additional speedup on CUDA.

## Sources
- Primer (this repo, untracked): `quadro-p5200-llamacpp-primer.md`
- Issue #7055 / PR #7188 (Pascal FA fix)
- NVIDIA Pascal Tuning Guide (FP16 1/64, DP4A)
