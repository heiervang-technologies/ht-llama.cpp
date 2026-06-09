# Pascal Quadro P5200 — ht-llama.cpp build notes (Arch Linux, 2026)

## Hardware
- GPU: NVIDIA Quadro P5200, 16 GB GDDR5, **CC 6.1 (sm_61)**, GP104 mobile (2560 cores, ~230 GB/s)
- Host: crystal (Arch Linux, Intel i7-7820HQ 4c/8t, 16 GB RAM, driver 580.159)

## Two facts that drive every decision (per primer)
1. **FP16 is 1/64 of FP32** on Pascal — never enable raw-FP16 fast paths. Use FP32 accumulation.
2. **Decode is bandwidth-bound** at ~230 GB/s — quantize to reduce bytes-per-token; INT8 DP4A is the fast path (Pascal lacks tensor cores; `__dp4a` int8 dot-product is what MMQ kernels use).

## Toolkit compatibility wall (and the climb)

Arch `extra/cuda 13.2.1-3` drops Pascal: `nvcc fatal: Unsupported gpu architecture compute_61`. Driver 580 still **runs** sm_61 binaries (`nvidia-smi` works); only the **toolkit** (`nvcc` + cuBLAS) dropped support in CUDA 13.0.

### The five obstacles on a stock Arch box
1. **CUDA 12.9 not in Arch repos.** AUR has `cuda-pascal 12.9.1` but it depends on `gcc14` (also AUR, source-build, ~45 min).
2. **NVIDIA runfile installer dies** with `error while loading shared libraries: libxml2.so.2`. Arch only ships `libxml2.so.16` (soname bumped in 2.15). Bypass with `--extract` rather than letting the installer binary run.
3. **gcc-15 / gcc-16 host compilers are too new for nvcc 12.9.** Both have `__is_pointer` / `__is_volatile` / `__array_rank` and `char8_t` built-ins that cudafe++ doesn't recognize. Need a real gcc-13 or gcc-14.
4. **gcc-14 isn't in Arch current repos** (Arch shipped gcc-15, then gcc-16). Source-build is slow. Solution: archlinux-archive — `gcc-14.2.1+r753-1` is a 51 MB `.pkg.tar.zst`; extract side-by-side to `/opt/gcc-14/`, point `CMAKE_CUDA_HOST_COMPILER` at it.
5. **glibc 2.43 `cospi` / `sinpi` / `rsqrt` collide with CUDA 12.9 math headers.** `bits/mathcalls.h` declares them with `noexcept(true)`; CUDA's `crt/math_functions.h` and `.hpp` decl/define them without. cudafe++ errors out on the exception-spec mismatch. Patch both CUDA headers to add `noexcept(true)` to all of `rsqrt` / `sinpi` / `cospi` / `sincospi` (and `f` variants, plus `acospi` / `asinpi` / `atanpi` / `tanpi`).

### Concrete install recipe on Arch (driver 580 already in place)

```bash
# 1) Extract CUDA 12.9 toolkit components — bypasses the libxml2.so.2-blocked installer binary.
sh ~/.cache/yay/cuda-pascal/cuda_12.9.1_575.57.08_linux.run \
    --extract=/home/me/big-tmp/cuda-x \
    --tmpdir=/home/me/big-tmp
sudo mkdir -p /opt/cuda-pascal-runfile
for d in /home/me/big-tmp/cuda-x/cuda_* \
         /home/me/big-tmp/cuda-x/libcublas \
         /home/me/big-tmp/cuda-x/libcusparse \
         /home/me/big-tmp/cuda-x/libcurand \
         /home/me/big-tmp/cuda-x/libcusolver \
         /home/me/big-tmp/cuda-x/libcufft \
         /home/me/big-tmp/cuda-x/libnvfatbin \
         /home/me/big-tmp/cuda-x/libnvjitlink; do
    [[ -d "$d" ]] && sudo rsync -a "$d/" /opt/cuda-pascal-runfile/
done
# Verify
/opt/cuda-pascal-runfile/bin/nvcc --version       # CUDA 12.9.86
/opt/cuda-pascal-runfile/bin/nvcc --list-gpu-arch | grep compute_61   # present

# 2) Install gcc-14 from archlinux-archive (binary, ~30s, no source build)
cd /tmp
curl -LO https://archive.archlinux.org/packages/g/gcc/gcc-14.2.1+r753+g1cd744a6828f-1-x86_64.pkg.tar.zst
curl -LO https://archive.archlinux.org/packages/g/gcc-libs/gcc-libs-14.2.1+r753+g1cd744a6828f-1-x86_64.pkg.tar.zst
sudo mkdir -p /opt/gcc-14
sudo tar -I zstd -xf gcc-14.2.1*.pkg.tar.zst -C /opt/gcc-14
sudo tar -I zstd -xf gcc-libs-14.2.1*.pkg.tar.zst -C /opt/gcc-14
/opt/gcc-14/usr/bin/g++ --version                 # GCC 14.2.1

# 3) Patch CUDA math headers for glibc-2.43 noexcept compatibility
for n in rsqrt sinpi cospi sincospi rsqrtf sinpif cospif sincospif acospi asinpi atanpi tanpi; do
    sudo sed -i -E "s/(__MATH_FUNCTIONS_DECL__[ \\t]+(float|double|void)[ \\t]+$n[ \\t]*\\([^)]*\\))$/\\1 noexcept(true)/" \
        /opt/cuda-pascal-runfile/targets/x86_64-linux/include/crt/math_functions.hpp
    sudo sed -i -E "s/(__MATH_FUNCTIONS_DECL__[ \\t]+(float|double|void)[ \\t]+$n[ \\t]*\\([^)]*\\));/\\1 noexcept(true);/" \
        /opt/cuda-pascal-runfile/targets/x86_64-linux/include/crt/math_functions.h
done
for n in rsqrt sinpi cospi cospif sinpif rsqrtf acospi asinpi atanpi tanpi; do
    sudo sed -i -E "s/((float|double)[ \\t]+$n[ \\t]*\\([^)]*\\));/\\1 noexcept(true);/g" \
        /opt/cuda-pascal-runfile/targets/x86_64-linux/include/crt/math_functions.h
done
```

## Build commands

### CUDA (preferred on Pascal)
```bash
cmake -B build-cuda -GNinja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_F16=OFF \
  -DCMAKE_CUDA_COMPILER=/opt/cuda-pascal-runfile/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/opt/gcc-14/usr/bin/g++ \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j 8 --target llama-bench llama-app
```

Rationale:
- `CMAKE_CUDA_ARCHITECTURES=61` — exactly the P5200 cap; faster compile.
- `GGML_CUDA_FORCE_MMQ=ON` — routes quantized matmul through `__dp4a` (the fast INT8 path on Pascal). Verify at startup: `ggml_cuda_init: GGML_CUDA_FORCE_MMQ: yes`.
- `GGML_CUDA_F16=OFF` — Pascal FP16 ALU is 1/64 of FP32; FP32 accumulators win.
- `LLAMA_BUILD_SERVER=ON` — required for the `llama-app` unified router (`bin/llama`) to link. The `llama-app` target depends on `libllama-server-impl.so` + `libllama-cli-impl.so`; without server-on, the link fails with `cannot find -lllama-server-impl`. Also required for Gemma4 MTP / spec-decode (`ctx_other` wiring for the `Gemma4Assistant` draft class lives only in `tools/server/server-context.cpp` — the standalone `llama-speculative-simple` binary segfaults with `Gemma4Assistant requires ctx_other to be set`).
- `LLAMA_BUILD_TESTS=OFF` — Pascal builds don't need the test suite; speeds up CI/image builds.
- Flash-attention `-fa on` works on Pascal via the no-tensor-core FA kernel with FP32 accumulation (issue #7055 fixed in #7188; reinforced by #7681 / #15769 / #22541).
- Spec-decode footgun: `--spec-type` defaults to `none`. Passing `-md <draft.gguf>` alone is silently ignored. Must pass `--spec-type draft-mtp` explicitly to engage MTP. `/props default_generation_settings.params["speculative.types"]` is the per-REQUEST sampler default, NOT the server engine state — the canonical engagement read is the server stderr (`draft acceptance = X.XXXXX (acc/gen)` + `statistics draft-mtp: ...`).

### Vulkan (baseline / portable fallback)
```bash
cmake -B build-vulkan -GNinja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j 8 --target llama-bench llama-app
```
Pascal Vulkan startup reports `matrix cores: none` — no coopmat path. No usable Vulkan flash-attention on Pascal (needs coopmat2 / Ampere+). Quantized KV (`-ctk q8_0`) also unavailable without FA.

## Benchmark
```bash
scripts/bench-pascal-p5200.sh both   # or `cuda` / `vulkan`
# Cross-product: pp 128/512/2048 × tg 32/128 × fa 0,(1 for CUDA), 3 reps.
```

## Results — Vulkan baseline (build f6feddb, Llama-3.1-8B-Instruct Q4_K_M, ngl=99, fa=0)

| test    | t/s    | stddev |
|---------|--------|--------|
| pp128   | 269.49 | 0.90   |
| pp512   | 277.78 | 0.10   |
| pp2048  | 250.92 | 0.62   |
| tg32    | 35.18  | 0.10   |
| tg128   | 35.22  | 0.03   |

- `tg` flat across batch sizes — bandwidth-bound, as primer predicts.
- `pp` peaks at 512, dips at 2048 — Vulkan Pascal has no FA, so attention quadratic shows.

## Results — CUDA (build 5159fee, /opt/cuda-pascal-runfile nvcc 12.9.86, /opt/gcc-14)

Init line confirms the path:
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16257 MiB):
  Device 0: Quadro P5200, compute capability 6.1, VMM: yes, VRAM: 16257 MiB
```
Build flags compile-time-baked: `GGML_CUDA_FORCE_MMQ=1` and `compute_61,sm_61` (visible in nvcc command line). CC 6.1 + MMQ-only + no cuBLAS fallback = INT8 dp4a path.

### Llama-2-7B Q4_0 (matches primer GTX 1080 / P40 / P100 reference table)

| test    | CUDA fa=0 | CUDA fa=1 | Vulkan fa=0 | CUDA/Vulkan |
|---------|----------:|----------:|------------:|------------:|
| pp128   | 701.09    | 749.79    | 393.53      | 1.91x       |
| pp512   | 764.73    | 794.94    | 418.05      | 1.90x       |
| pp2048  | 668.17    | 720.10    | 363.76      | 1.98x       |
| tg32    | 44.57     | 45.83     | 42.90       | 1.07x       |
| tg128   | 44.58     | 45.83     | 42.99       | 1.07x       |

P5200 vs the primer's GTX 1080 row (pp512=789, tg128=46): **basically the same die does basically the same numbers.** GDDR5 vs GDDR5X bandwidth difference (230 vs 320 GB/s) shows up in pp (-3%) and tg (-2%), but the gap is much smaller than the primer estimated (it guessed ~30-40 t/s tg; we get 44-46).

### Llama-3.1-8B Q4_K_M (matches Vulkan baseline above)

| test    | CUDA fa=0 | CUDA fa=1 | Vulkan fa=0 | CUDA/Vulkan |
|---------|----------:|----------:|------------:|------------:|
| pp128   | 602.50    | 629.32    | 264.45      | 2.38x       |
| pp512   | 650.51    | 664.46    | 276.30      | 2.40x       |
| pp2048  | 583.97    | 613.69    | 251.22      | 2.44x       |
| tg32    | 34.39     | 35.84     | 35.21       | 1.02x       |
| tg128   | 34.42     | 35.80     | 35.24       | 1.02x       |

### Observations
- **pp: CUDA ~2-2.4x Vulkan.** Pascal Vulkan lacks coopmat AND flash-attention, so attention scales quadratically and shaders are scalar. CUDA wins decisively on prompt processing.
- **tg: backends tied within ~2 t/s.** Confirms decode is bandwidth-bound (~230 GB/s ceiling). CUDA can't win on bytes-per-second over Vulkan; both saturate the bus.
- **FA on CUDA: +5-7% pp, +3-4% tg.** Real but small. Pascal's no-tensor-core FA kernel does its job; the gain is mostly KV-cache traffic reduction.
- **L2-7B Q4_0 ≈ GTX 1080 numbers.** Same GP104 die; the 28% bandwidth deficit (GDDR5 vs GDDR5X) costs ~2-3% in practice, not the ~25% the primer worst-cased.

### Bench artifacts (committed)
- `scripts/bench-pascal-p5200-vulkan.f6feddb.json` — initial Vulkan L3.1-8B baseline
- `scripts/bench-pascal-p5200-cuda-l31-8b.5159fee.json` — CUDA L3.1-8B Q4_K_M
- `scripts/bench-pascal-p5200-cuda-l2-7b.5159fee.json` — CUDA L2-7B Q4_0
- `scripts/bench-pascal-p5200-vulkan-l2-7b.5159fee.json` — Vulkan L2-7B Q4_0

## Packaging — producing a relocatable runtime tarball

Used to produce `pascal-cuda-artifacts.tar.zst` for the Omarchy ISO autoinstall (hai-os-dev consumes via `tar -C / -xf ...`). The end product is rpath-clean and resolves via `/etc/ld.so.conf.d/cuda-pascal.conf` (no `LD_LIBRARY_PATH` required at runtime).

```bash
# 1) cmake install — strips the build-dir RUNPATH from all installed targets
sudo cmake --install build-cuda --prefix /opt/ht-llama-cuda
# (verify: readelf -d /opt/ht-llama-cuda/bin/llama-bench should show no RPATH/RUNPATH)

# 2) cmake does NOT install libllama.so or libllama-common.so (they are intermediates,
#    not declared as install targets). Copy them in, then strip rpath with patchelf.
sudo cp -a build-cuda/bin/libllama.so.0.0.* /opt/ht-llama-cuda/lib/
sudo cp -a build-cuda/bin/libllama-common.so.0.0.* /opt/ht-llama-cuda/lib/
# Delete any stale older-version files left over from prior builds:
#   sudo rm /opt/ht-llama-cuda/lib/libllama{,-common}.so.0.0.<old>
sudo pacman -S --noconfirm patchelf   # required for the rpath strip below
sudo patchelf --remove-rpath /opt/ht-llama-cuda/lib/libllama.so.0.0.*
sudo patchelf --remove-rpath /opt/ht-llama-cuda/lib/libllama-common.so.0.0.*

# 3) Recreate the symlink chain (.so → .so.0 → .so.0.0.<X>)
cd /opt/ht-llama-cuda/lib
LV=$(basename libllama.so.0.0.*)
sudo ln -sfn "$LV" libllama.so.0 && sudo ln -sfn libllama.so.0 libllama.so
CV=$(basename libllama-common.so.0.0.*)
sudo ln -sfn "$CV" libllama-common.so.0 && sudo ln -sfn libllama-common.so.0 libllama-common.so

# 4) Stage with relative-from-/ layout. Members will be `opt/...`, never `/opt/...`
#    or `home/me/...` — so `tar -C / -xf` on the target expands back to /opt/.
rm -rf /tmp/stage && mkdir -p /tmp/stage/opt/cuda-pascal-runfile/targets/x86_64-linux/lib
cp -a /opt/cuda-pascal-runfile/targets/x86_64-linux/lib/libcudart.so*    /tmp/stage/opt/cuda-pascal-runfile/targets/x86_64-linux/lib/
cp -a /opt/cuda-pascal-runfile/targets/x86_64-linux/lib/libcublas.so*    /tmp/stage/opt/cuda-pascal-runfile/targets/x86_64-linux/lib/
cp -a /opt/cuda-pascal-runfile/targets/x86_64-linux/lib/libcublasLt.so*  /tmp/stage/opt/cuda-pascal-runfile/targets/x86_64-linux/lib/
sudo cp -a /opt/ht-llama-cuda /tmp/stage/opt/
sudo chown -R "$(id -un):$(id -gn)" /tmp/stage

# 5) Tarball
cd /tmp/stage && tar --zstd -cf ~/pascal-cuda-artifacts.tar.zst opt/
sha256sum ~/pascal-cuda-artifacts.tar.zst
tar --zstd -tf ~/pascal-cuda-artifacts.tar.zst | head    # confirm members start with opt/
```

Pruning notes:
- Static archives (`*.a`) and CUDA stubs are explicitly excluded from the rsync — `.so*` glob only.
- The full `/opt/cuda-pascal-runfile/` is 9.5 GB; the pruned runtime libset (libcudart + libcublas + libcublasLt) is ~816 MB unpacked. Together with the ~45 MB `/opt/ht-llama-cuda/` install prefix, the resulting zstd-compressed tarball is **512 MB**, 110 members. Reference hash from the first crystal build:
  ```
  sha256: 0efed65095d3da67713aa4344fcdb1c9e6f8faf397d5eaf1f24c9e8cd00fa339
  ```
  (zstd is non-deterministic across machines/versions; reproduce by re-running §7 and re-sha'ing locally.)
- libstdc++.so.6, libgomp.so.1, libcuda.so.1 are NOT in the tarball — they come from the host OS (`gcc-libs`, `nvidia-580xx-utils`). The runtime requires those packages installed on the target.

### Runtime setup on target (consumed by the ISO)

```bash
tar -C / -xf pascal-cuda-artifacts.tar.zst
cat > /etc/ld.so.conf.d/cuda-pascal.conf <<'EOF'
/opt/cuda-pascal-runfile/targets/x86_64-linux/lib
/opt/ht-llama-cuda/lib
EOF
ldconfig
# Sanity:
ldd /opt/ht-llama-cuda/bin/llama-bench | grep "not found"   # must be empty
/opt/ht-llama-cuda/bin/llama-bench --help                   # should print CUDA init + usage
```

Expected init line on a P5200 host:
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16257 MiB):
  Device 0: Quadro P5200, compute capability 6.1, VMM: yes, VRAM: 16257 MiB
```

### Heads-up: unified `bin/llama` (heierchat router) not in this tarball

The configure used here does not build `llama-cli-impl` / `llama-server-impl`, so the unified `bin/llama` router cannot link and is skipped during install. The shipped binaries cover bench / quantize / perplexity / imatrix / tts / mtmd-cli / completion / embedding / finetune / etc. — sufficient for standalone-CLI and benchmarking use. A v2 tarball with the router would need a re-configure that enables both impls.

## Sources
- Primer (this repo, untracked): `quadro-p5200-llamacpp-primer.md`
- Issue #7055 / PR #7188, #7681, #15769, #22541 (Pascal FA fix + tile-FA hardening)
- NVIDIA Pascal Tuning Guide (FP16 1/64, DP4A)
- archlinux-archive: https://archive.archlinux.org/packages/g/gcc/
