# Quadro P5200 × llama.cpp — Support & Optimization Primer

A one-page field guide for running, supporting, or optimizing llama.cpp on the
**NVIDIA Quadro P5200** (16 GB, Pascal mobile workstation GPU). Compiled from
NVIDIA tuning docs, the llama.cpp CUDA/Vulkan perf discussions, and bug threads
(June 2026).

---

## 1. Know the silicon (it dictates everything)

| Property | Value | Why it matters for LLMs |
|---|---|---|
| Architecture | Pascal **GP104**, 16 nm | Pre-Volta → **no Tensor Cores** |
| Compute capability | **6.1 (`sm_61`)** | Build target; also the CUDA-13 cutoff |
| CUDA cores | 2560 | ~8.9 TFLOPS FP32 |
| VRAM | **16 GB GDDR5** (not GDDR5X) | Fits 7B–14B comfortably |
| Mem bandwidth | **~230 GB/s**, 256-bit | **The decode ceiling** (token-gen is bandwidth-bound) |
| **FP16 throughput** | **1/64 of FP32** | Native FP16 math is a trap — avoid |
| **INT8 (DP4A)** | Fast `__dp4a` path | **This is the fast path** for quantized models |
| Form factor / power | MXM, 100 W, Max-P & Max-Q | Mobile → sustained inference can thermal-throttle |
| Vulkan | 1.1 baseline (driver-dependent) | No coopmat / matrix-core path |

**The two facts that drive every decision:** (1) FP16 is crippled (1/64 FP32) but
INT8 DP4A is fast → **run quantized, not FP16**. (2) Decode speed is capped by
~230 GB/s memory bandwidth, not compute.

---

## 2. CUDA backend (the recommended path on this card)

- **Toolkit pin is mandatory.** **CUDA 13.0 dropped offline-compile + library
  support for Pascal/Volta/Maxwell.** Build with the **CUDA 12.x** toolkit
  (12.4–12.9). The driver still *runs* sm_61 binaries, but nvcc 13+/cuBLAS 13+
  won't emit or support code for it.
- **Build:**
  ```bash
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
  cmake --build build --config Release -j
  ```
- **Quantized kernels (MMQ + DP4A) are what you want.** Q4_K_M / Q5_K_M / Q4_0 /
  Q8_0 route through MMQ, which uses the `__dp4a` int8 dot-product. Optionally
  force it: `-DGGML_CUDA_FORCE_MMQ=ON` (lower VRAM, guarantees the Pascal
  integer path). Verify at startup: `ggml_cuda_init: GGML_CUDA_FORCE_MMQ: yes`.
- **Flash attention now works — but mind the history.** Early builds threw
  `flash_attn_ext_f16 has no device code compatible with CUDA arch 610`
  (issue #7055, fixed in #7188). Today `-fa on` runs on Pascal via the
  **no-tensor-core vector/tile FA kernel with FP32 accumulation**. It saves KV
  memory and gives a small speedup; correctness should be spot-checked.
- **Don't** load raw FP16/BF16 weights expecting speed — the 1/64 ratio dominates.

---

## 3. Vulkan backend (portable fallback)

- Works on Pascal, but startup reports **`matrix cores: none`** — coopmat
  (`KHR_cooperative_matrix`) needs Turing+, coopmat2 (`NV_cooperative_matrix2`)
  needs Ampere+. Pascal uses **scalar matmul shaders + `GL_EXT_integer_dot_product`
  (DP4A)** for quantized models.
- **No usable Vulkan flash attention on Pascal** (it requires coopmat2). `-fa`
  on Vulkan here is effectively a CPU fallback — leave it off.
- If a driver **falsely advertises coopmat** and you get garbage output or TDR
  hangs, force the safe path:
  `GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1`.
- **CUDA vs Vulkan:** CUDA wins prompt-processing (`pp`) clearly; Vulkan `tg`
  (token-gen) is roughly comparable, occasionally slightly higher. **Default to
  CUDA on this card**; reach for Vulkan only when the CUDA 12.x toolchain is
  unavailable.

---

## 4. What to expect (Pascal proxies, Llama-2-7B Q4_0, llama-bench)

| GPU (proxy) | Backend | pp512 t/s | tg128 t/s |
|---|---|---|---|
| **Quadro P5200 (GP104, GDDR5 230 GB/s) — MEASURED** | **CUDA** | **794.94** | **45.83** |
| **Quadro P5200 — MEASURED** | **Vulkan** | **418.05** | **42.99** |
| GTX 1080 (GP104, GDDR5X 320 GB/s) | CUDA | ~789 (FA ~825) | ~46 (FA ~47) |
| P104-100 (GP104 mining, GDDR5X) | Vulkan | ~312 | ~46 |
| Tesla P40 (GP102) | Vulkan | ~488 | ~59 |
| Tesla P100 (GP100, HBM2) | Vulkan | ~678 | ~63 |
| Quadro P2000 (small GP106) | Vulkan | ~170 | ~23 |

**P5200 measured (crystal, `pascal/p5200-build` @ `4d04cbc`, `-ngl 99 -fa 1`):**
it lands **on the GTX 1080 row**, not below it. The GDDR5-vs-GDDR5X gap is only
**~2–3%** (795/46 vs 789/46), *not* the ~25% an earlier draft of this section
feared — same GP104 die, and 7B Q4 decode isn't tight enough against 230 GB/s to
expose the bandwidth delta. **CUDA beats Vulkan 1.90× on pp, 1.07× on tg**
(Llama-3.1-8B Q4_K_M: 2.4× pp, tied tg). `FORCE_MMQ:yes` + CC 6.1 + no
cuBLAS-fallback line ⇒ the `__dp4a` path is confirmed by exclusion.

---

## 5. Sizing the 16 GB

| Model | Quant | Fit |
|---|---|---|
| 7B–8B | Q4_K_M / Q5_K_M / Q8_0 | Easy, long context |
| 13B–14B | Q4_K_M | Comfortable |
| ~20–32B | Q3_K / Q4_K_M | Tight, short context / partial offload |

- **Stretch context with a quantized KV cache** (`--cache-type-k q8_0
  --cache-type-v q8_0`) plus `-fa on` for the memory saving.
- Decode throughput ≈ bandwidth ÷ bytes-read-per-token → smaller quant = faster
  *and* roomier. Q4_K_M is the sweet spot.

---

## 6. If you're *implementing / optimizing* support for this GPU

1. **Ship an `sm_61` code path for every kernel.** The #7055 regression was a
   kernel with no Pascal device code. Any new attention/matmul kernel needs a
   non-tensor-core fallback or it hard-faults on launch.
2. **Optimize for bytes, not FLOPs.** Decode is bandwidth-bound at 230 GB/s —
   wins come from quantized weights/KV and reduced memory traffic, not from
   squeezing FP32 ALU.
3. **Treat FP16 as poison; lean on DP4A.** Keep accumulation in **FP32**; route
   quantized matmul through `__dp4a` (CUDA) / `GL_EXT_integer_dot_product`
   (Vulkan). Audit any FP16-accumulate fast path before enabling it on `sm_61`.
4. **Vulkan: assume coopmat is absent** and make the integer-dot-product shader
   the optimized path. Honor `GGML_VK_DISABLE_COOPMAT*`; consider a device-name
   guard if a driver lies about capability.
5. **Pin the toolchain.** CI/build that targets this card must use **CUDA 12.x**
   — a CUDA-13 build silently drops `sm_61`. Document it; don't assume `main`/13.
6. **Account for thermals.** Max-Q/100 W mobile parts downclock under sustained
   load — benchmark warm, not just cold.

---

### Sources
- [Running older Pascal GPU with llama.cpp — Discussion #19248](https://github.com/ggml-org/llama.cpp/discussions/19248)
- [Performance of llama.cpp on CUDA — Discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)
- [Performance of llama.cpp with Vulkan — Discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
- [Flash Attention broken on Quadro P3200 (CC 6.1) — Issue #7055](https://github.com/ggml-org/llama.cpp/issues/7055)
- [NVIDIA Pascal Tuning Guide (FP16 1/64, DP4A INT8)](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html)
- [NVIDIA drops CUDA support for Maxwell/Pascal/Volta in CUDA 13 — Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-to-drop-cuda-support-for-maxwell-pascal-and-volta-gpus-with-the-next-major-toolkit-release)
- [llama.cpp build docs (GGML_CUDA_FORCE_MMQ, CMAKE_CUDA_ARCHITECTURES)](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp Vulkan coopmat detection — Issue #6072](https://github.com/ggml-org/llama.cpp/issues/6072)
- [Quadro P5200 specs — NotebookCheck](https://www.notebookcheck.net/NVIDIA-Quadro-P5200-Workstation-GPU.239818.0.html)
