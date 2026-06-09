# Quadro P5200 → ht-llama.cpp — Omarchy Autoinstall Handoff

**For:** hai-os-dev (HP Z VR Backpack / Quadro P5200 fleet — crystal=unit1, amethyst=unit2, +~9).
**Verified on:** crystal, branch `pascal/p5200-build` @ `4d04cbc`, 2026-06-09.
**TL;DR for the ISO:** bake in the **CUDA 12.9 runfile `--extract` prefix + gcc-14 archive + patched math headers**, build `llama-bench`/`llama-app` for **`sm_61`** once, ship the binaries. **Zero AUR.** Pure pacman + 2 archive curls + 1 NVIDIA runfile extract + sed patches.

---

## Answers to your 7 questions

**1 · Extra packages to BUILD on Pascal** — all `extra/` pacman, **no AUR**:
`base-devel cmake ninja git curl jq zstd rsync` · driver `nvidia-580xx-dkms nvidia-580xx-utils` · Vulkan stack (for the Vulkan baseline target) `vulkan-headers vulkan-icd-loader vulkan-tools vulkan-validation-layers spirv-headers spirv-tools glslang shaderc`.
The **CUDA toolkit is NOT a package** — see #2. gcc-14 is NOT the AUR source build — it's a 51 MB archive binary (#5, step C).

**2 · Does 580xx ship a usable CUDA for sm_61? — No, and don't use Arch `cuda`.**
The `nvidia-580xx` driver gives the *runtime* (`libcuda`), not the *compiler*. Arch `extra/cuda` is **13.2 → dropped Pascal** (`nvcc fatal: Unsupported gpu architecture compute_61`). Pin **CUDA 12.9.1** via the NVIDIA **runfile, extracted (not installed)** into `/opt/cuda-pascal-runfile` (`nvcc` reports `12.9.86`, `--list-gpu-arch` includes `compute_61`). **No conflict** with the 580xx driver — toolkit and driver are independent; we `--extract` and never touch the runfile's bundled 575 driver. Any `nvidia-580xx-utils ≥ 575` satisfies the CUDA-12.9 runtime floor — **no driver pin needed**. (Arch `cuda` 13.2 may coexist at `/opt/cuda` for sm_75+ boxes; irrelevant here — omit it from the Pascal image.)

**3 · Build flags / env:**
```
-DGGML_CUDA=ON  -DCMAKE_CUDA_ARCHITECTURES=61  -DGGML_CUDA_FORCE_MMQ=ON  -DGGML_CUDA_F16=OFF
-DCMAKE_CUDA_COMPILER=/opt/cuda-pascal-runfile/bin/nvcc
-DCMAKE_CUDA_HOST_COMPILER=/opt/gcc-14/usr/bin/g++
```
`GGML_CUDA_F16=OFF` is deliberate — **Pascal FP16 = 1/64 FP32**, the int8 `__dp4a` MMQ path is the fast one. `FORCE_MMQ=ON` guarantees it and lowers VRAM. **Do not enable F16.** Runtime env (drop in `/etc/profile.d/cuda-pascal.sh`):
```
export CUDA_PATH=/opt/cuda-pascal-runfile
export PATH=/opt/cuda-pascal-runfile/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda-pascal-runfile/lib64:/opt/cuda-pascal-runfile/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

**4 · Non-package system config:** minimal. The 580xx DKMS module + `nvidia-smi` working is enough to bench. Recommended for a headless fleet node: enable **`nvidia-persistenced`** (`systemctl enable --now nvidia-persistenced`) to keep the GPU initialized and cut first-call latency. No special `modprobe.d`, kernel cmdline, or ulimit needed for inference. **Watch RAM at build time** — `nvcc` + 313 ninja objects peaks a few GB; on 16 GB boxes give it **zram or a swapfile** so the parallel build (`-j 8`) doesn't OOM. (No GPU-side swap concern — 16 GB VRAM fits 7B–8B Q4 with room.)

**5 · What FAILED / needs pre-handling** — five obstacles, all pre-solved in the recipe:
| # | Failure | Fix baked into recipe |
|---|---|---|
| 1 | Arch `cuda` 13.2 → `compute_61 unsupported` | pin CUDA 12.9.1 (runfile) |
| 2 | runfile `cuda-installer: libxml2.so.2: cannot open` (Arch ships `.so.16`) | `--extract` bypass — skips the installer + libxml2 entirely |
| 3 | nvcc 12.9 rejects host gcc-15/16 | **gcc-14 archive** binary → inside nvcc's supported range (NO `-allow-unsupported`, NO `host_config.h` strip) |
| 4 | AUR `cuda-pascal`/`gcc14` = 45-min source builds, `/opt/cuda` collision | replaced by runfile extract + gcc-14 archive curl (~30 s) |
| 5 | **glibc 2.43** added `cospi/sinpi/rsqrt/...` as `noexcept(true)`; CUDA 12.9 `math_functions.h/.hpp` declare them without → `cudafe++` exception-spec mismatch, build dies | **sed-patch the 3 decl groups to add `noexcept(true)`** (loops in recipe step E) |

**6 · Pre-build at image time, or post-install? — Pre-build once, ship the artifacts.**
The fleet is ~11 **identical sm_61 GP104** units, so one build is portable to all. Build **once on a reference unit at image-build time**, then ship as an overlay. **Two shipping gotchas, both solved by `cmake --install`:**

> **DO THIS BEFORE TARBALLING:** `cmake --install build-cuda --prefix /opt/ht-llama-cuda`. Raw build-dir binaries bake `RUNPATH=/home/.../build-cuda/bin` (absolute) — they won't find their `.so` once relocated. The install step **strips the build rpath** and produces a clean `bin/` + `lib/` layout. (Alternatives: `patchelf --set-rpath '$ORIGIN'`, or path-mirror the build dir — both worse.)

> **systemd services do NOT source `/etc/profile.d`.** A `profile.d` `LD_LIBRARY_PATH` only fixes interactive shells. Use **`/etc/ld.so.conf.d/cuda-pascal.conf`** + `ldconfig` — this is service-safe AND shell-safe. **Verified two-line content (both paths required — CUDA libs *and* the llama `.so`):**
> ```
> /opt/cuda-pascal-runfile/targets/x86_64-linux/lib
> /opt/ht-llama-cuda/lib
> ```
> After `ldconfig`: `ldd /opt/ht-llama-cuda/bin/llama-bench | grep "not found"` is empty, and `--help` prints `ggml_cuda_init: ... Quadro P5200, compute capability 6.1` — **no `LD_LIBRARY_PATH` anywhere.**

> **Verified reference tarball (cached fast-path):** `crystal:/home/me/pascal-cuda-artifacts.tar.zst` — **512 MB**, 110 members, `sha256 0efed65095d3da67713aa4344fcdb1c9e6f8faf397d5eaf1f24c9e8cd00fa339`, members rooted at `/` (`opt/ht-llama-cuda/` + `opt/cuda-pascal-runfile/`) → `tar -C / -x`. **Scope = bench + standalone CLI** (`llama-bench`, `llama-cli`, `llama-perplexity`, `llama-quantize`, `llama-imatrix`, `llama-tts`, `llama-mtmd-cli`, …). **Not included:** `llama-server` / unified `bin/llama` router (built `LLAMA_BUILD_SERVER=OFF`) — serving needs a v2 re-configure. Host-provided (correctly absent): `libstdc++.so.6`, `libgomp.so.1`, `libcuda.so.1` (from `gcc-libs` + `nvidia-580xx`).

**Tarball contents + measured sizes (crystal):**
| Path | Size | In runtime overlay? |
|---|---|---|
| `/opt/ht-llama-cuda/` (from `cmake --install`) | ~50 MB | **yes** — rpath-clean bin/ + lib/ |
| `/opt/cuda-pascal-runfile/` **runtime-pruned** (only `libcudart` + `libcublas` dirs) | **~150-200 MB** | yes |
| `/opt/cuda-pascal-runfile/` full toolkit | 9.5 GB | no (build-capable ISO only) |
| `/opt/gcc-14/` | 353 MB | **no — build-only**; rebuild-fallback tarball only |

The runtime `.so` set (~17 files, ship the whole `.so → .so.0 → .so.0.0.2` symlink chain via `cp -a`): `libggml-cuda.so` **(39 MB — the sm_61 kernels)**, `libllama-common.so` (6.1M), `libllama.so` (3.9M), `libggml-cpu.so` (1.1M), `libggml-base.so` (957K), `libggml.so` (55K), `libllama-bench-impl.so` (458K) + the thin shim binaries. **Build-capable ISO** (keeps recipe rebuild on-device) needs instead: `cuda_nvcc + cuda_cccl + cuda_cudart + cuda_nvrtc + libcublas` (~3-4 GB) + `/opt/gcc-14` (353 MB). Server was built `OFF`; flip `LLAMA_BUILD_SERVER=ON` for `llama-server`.

**7 · HF token / models at image-build? — No token; out of scope for the ISO.**
Both bench models are **public** (no gating): `TheBloke/Llama-2-7B-GGUF` (Q4_0, 3.83 GB) and `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (Q4_K_M, 4.92 GB). **Don't bake multi-GB GGUFs into the image** — pull them at **first-boot** with a plain `curl` of the `resolve/main` URL (curl avoids the `hf` xet-416 silent-truncation gotcha; byte-verify the size). Land them in a configurable models dir (`$GGUFS`/`$MODELS`, default `~/Models`) — never hard-code. Only wire an `HF_TOKEN` (from `usb/secrets/`) if you later add *gated* models.

---

## Canonical recipe (the autoinstall body, command-exact)

```bash
set -euo pipefail
# A. Packages (extra/ only, no AUR)
sudo pacman -S --needed base-devel cmake ninja git curl jq zstd rsync \
  nvidia-580xx-dkms nvidia-580xx-utils \
  vulkan-headers vulkan-icd-loader vulkan-tools vulkan-validation-layers \
  spirv-headers spirv-tools glslang shaderc

# B. CUDA 12.9.1 — download + EXTRACT (bypasses the libxml2-blocked installer)
mkdir -p ~/big-tmp/cuda-x
curl -L -o ~/big-tmp/cuda_12.9.1.run \
  https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sh ~/big-tmp/cuda_12.9.1.run --extract=$HOME/big-tmp/cuda-x --tmpdir=$HOME/big-tmp
sudo mkdir -p /opt/cuda-pascal-runfile
# MINIMAL set (verified via `ldd libggml-cuda.so` → libcudart, libcublas, libcublasLt only;
# libcuda comes from the 580xx driver). Saves ~1-2 GB vs the comprehensive set.
for d in ~/big-tmp/cuda-x/cuda_* ~/big-tmp/cuda-x/libcublas ~/big-tmp/cuda-x/libnvjitlink; do
  [[ -d "$d" ]] && sudo rsync -a "$d/" /opt/cuda-pascal-runfile/
done
# COMPREHENSIVE / future-proof (add if ggml-cuda later links cusparse/curand/cusolver/cufft):
#   ...also rsync libcusparse libcurand libcusolver libcufft libnvfatbin

# C. gcc-14 (51 MB archive binary — stays in nvcc 12.9's supported host range)
cd /tmp
curl -LO https://archive.archlinux.org/packages/g/gcc/gcc-14.2.1+r753+g1cd744a6828f-1-x86_64.pkg.tar.zst
curl -LO https://archive.archlinux.org/packages/g/gcc-libs/gcc-libs-14.2.1+r753+g1cd744a6828f-1-x86_64.pkg.tar.zst
# verify: 2f5d57f8...e5b8f  gcc ; 3353d8c2...91917  gcc-libs   (sha256)
sudo mkdir -p /opt/gcc-14
sudo tar -I zstd -xf gcc-14.2.1*.pkg.tar.zst      -C /opt/gcc-14
sudo tar -I zstd -xf gcc-libs-14.2.1*.pkg.tar.zst -C /opt/gcc-14

# D. (FALLBACK ONLY — skip on gcc-14 path) host_config.h guard strip for gcc-15/16:
# sudo sed -i '/unsupported GNU version/d' /opt/cuda-pascal-runfile/targets/x86_64-linux/include/crt/host_config.h

# E. REQUIRED — glibc-2.43 noexcept(true) patch on CUDA math headers
H=/opt/cuda-pascal-runfile/targets/x86_64-linux/include/crt
for n in rsqrt sinpi cospi sincospi rsqrtf sinpif cospif sincospif acospi asinpi atanpi tanpi; do
  sudo sed -i -E "s/(__MATH_FUNCTIONS_DECL__[ \t]+(float|double|void)[ \t]+$n[ \t]*\([^)]*\))$/\1 noexcept(true)/" "$H/math_functions.hpp"
  sudo sed -i -E "s/(__MATH_FUNCTIONS_DECL__[ \t]+(float|double|void)[ \t]+$n[ \t]*\([^)]*\));/\1 noexcept(true);/" "$H/math_functions.h"
done
for n in rsqrt sinpi cospi cospif sinpif rsqrtf acospi asinpi atanpi tanpi; do
  sudo sed -i -E "s/((float|double)[ \t]+$n[ \t]*\([^)]*\));/\1 noexcept(true);/g" "$H/math_functions.h"
done

# F. Build (sm_61). Vulkan target is optional (baseline comparison only).
cd ~/ht/forks/ht-llama.cpp
cmake -B build-cuda -GNinja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_F16=OFF \
  -DCMAKE_CUDA_COMPILER=/opt/cuda-pascal-runfile/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/opt/gcc-14/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build-cuda -j 8 --target llama-bench llama-app   # ~12 min on i7-7820HQ

# G. Verify (the proof the whole chain worked)
LD_LIBRARY_PATH=/opt/cuda-pascal-runfile/lib64:/opt/cuda-pascal-runfile/targets/x86_64-linux/lib \
  ./build-cuda/bin/llama-bench --version 2>&1 | grep ggml_cuda_init
# expect: found 1 CUDA devices ... Quadro P5200, compute capability 6.1, VMM: yes, VRAM 16257 MiB
```

---

## Where it plugs into HaiOS

| HaiOS file | Change |
|---|---|
| `config/packages/gpu-nvidia.txt` | **Pascal can't use `nvidia-open-dkms`** (open modules need Turing+). Add a `gpu-nvidia-pascal.txt` selecting `nvidia-580xx-dkms`/`-utils` + the vulkan/build stack above. |
| `scripts/helpers/detect-gpu.sh` | Today returns only `nvidia/amd/none`. Add a compute-cap probe (`nvidia-smi --query-gpu=compute_cap` → `6.1`) so the installer can branch the Pascal path. |
| `scripts/phases/` | New `05b-pascal-cuda.sh` (runs after `05-gpu` when nvidia+`cc 6.x`): unpacks the prebuilt `/opt/cuda-pascal-runfile` + `/opt/gcc-14` overlay, drops `/etc/profile.d/cuda-pascal.sh`, installs binaries, enables `nvidia-persistenced`. |
| `config/profiles/sentinel.env` | Add `INSTALL_PASCAL_CUDA=true` (the P5200 fleet role). |
| offline cache / `overlay/` | Ship the build-once artifacts (cuda prefix, gcc-14, binaries) as a tarball — alongside the existing `pascal-hypr-neutralize.sh`. |

## Measured baseline (Llama-2-7B Q4_0, `-ngl 99`, `fa=1`) — bake into health-check
| Backend | pp512 t/s | tg128 t/s |
|---|---|---|
| **CUDA** (sm_61, FORCE_MMQ) | **794.94** | **45.83** |
| Vulkan (`matrix cores: none`) | 418.05 | 42.99 |
| ratio | **1.90×** | 1.07× |

P5200 lands on the **GTX 1080 reference row** (pp 789 / tg 46) — same GP104 die; the GDDR5-vs-GDDR5X gap is only ~2-3%, not the ~25% the primer feared. Llama-3.1-8B Q4_K_M: CUDA **2.4×** pp, tied on tg. **dp4a path confirmed by exclusion** (no cuBLAS-fallback line + `FORCE_MMQ` + CC 6.1). JSON artifacts: `scripts/bench-pascal-p5200-*.json` @ `4d04cbc`.

> **Two traps for the installer to guard:** (a) the default `nvidia-open-dkms` is wrong on Pascal → force `nvidia-580xx-dkms`; (b) never `pacman -S cuda` on these boxes (13.2 silently drops sm_61). Full recipe: `scripts/build-pascal-p5200.md` on `pascal/p5200-build`.
