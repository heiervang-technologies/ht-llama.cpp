# Lunar Lake validation

Target: Ultra 5 238V, Intel `8086:64a0`, Linux/Mesa Vulkan. Hybrid attention
remains experimental and off by default. The exact value `1` enables its
eligibility check; `0` and unset disable it. It only applies with F16 K/V,
one sequence, full GPU layer/KV offload, and context at most 2048 tokens.
Other configurations retain stock attention selection. The Q4_0 MMVQ change
is limited to the validated PCI device and Mesa driver.

`models.json` pins fresh validation artifacts, including SHA256 and HF revision.
These match filenames in the historical notes, whose original Xe2 hashes were
not recorded. Do not compare new results to those tables as identical artifacts.
GGUFs remain outside the repository under `$GGUFS` or `--models-dir`.

Serving presets make context and cache choices explicit. The baseline uses
FA-off, F16 K/V, one slot, and 8K context; the hybrid experiment uses FA-on and
2K context. Both verify model hashes at startup. MTP is optional and limited
to the matching 12B assistant. No service is installed or deployed by these scripts.

```bash
python scripts/xe2/serve.py 12b --mtp
python scripts/xe2/serve.py 26b
python scripts/xe2/serve.py 26b --profile hybrid
```

Switch back to `--profile baseline` to disable hybrid selection. The baseline
accepts `--ctx-size` up to 32768; hybrid rejects sizes above 2048. The presets
remain candidates until the acceptance runs below have passed.

```bash
python scripts/xe2/fetch-models.py --models-dir "$GGUFS"
cmake --build build-vulkan -j 4 --target llama-server llama-bench \
  test-backend-ops test-cpu-quantized-copy test-gemma4-attention-policy test-gemma4-device
ctest --test-dir build-vulkan --output-on-failure \
  -R 'test-(cpu-quantized-copy|gemma4-attention-policy)$'
./build-vulkan/bin/test-backend-ops test -b Vulkan0 -o CPY,CONT
./build-vulkan/bin/test-backend-ops test -b Vulkan0 -o MUL_MAT -p 'type_a=q4_0'
./build-vulkan/bin/test-backend-ops test -b Vulkan0 -o FLASH_ATTN_EXT \
  -p 'hsk=(256|512),.*type_K=(f16|q8_0|q4_0),type_V=(f16|q8_0|q4_0)'
python scripts/xe2/validate.py parity --output /tmp/xe2-validation
python scripts/xe2/validate.py smoke --output /tmp/xe2-validation
python scripts/xe2/validate.py bench --output /tmp/xe2-validation
python scripts/xe2/summarize.py /tmp/xe2-validation
python scripts/xe2/validate.py soak --output /tmp/xe2-validation
```

Run GPU stages sequentially, on AC, with the same power profile and desktop
workload. Benchmarks rotate FA-off, FA-on, and hybrid ordering for five runs
at each of 2K/8K/16K/32K context, for both targets. Above 2K the hybrid command
intentionally follows stock FA; those runs measure fallback behavior, not a
long-context optimization. Retain only gains larger than observed variation.
Each benchmark command measures pp512, tg64, and combined pp512+tg64, starting
at depth `context - 576`. The combined test ends at the named context envelope;
standalone pp/tg finish slightly earlier. Raw JSON records the actual sizes.

Parity uses full CPU logits at 1/2/16, 31/32, 63/64, 128, and 1152 prompt tokens;
F16 GPU paths must satisfy the backend FA test tolerance (NMSE < 5e-4).
Top-token agreement is recorded separately for review. F16/Q8_0/Q4_0 caches
must remain finite and reproduce logits after dirtying, freeing, and reusing
a suffix beyond the sliding window. CPU reference files belong to the exact
manifest model and are generated afresh by the runner.

Smoke checks repeated-prefix greedy tokens, the unset/zero switch, quantized
cache fallback, thinking modes, a complete tool-call/result cycle, stream
cancellation and slot reuse, and the matching 12B MTP assistant. The default
soak lasts a total hour: 15 minutes per target/profile combination. Baseline
uses 8K context and one slot, with long prompts spanning multiple prefill batches;
hybrid uses 2K per slot, one slot for 12B and four concurrent slots for 26B.
Both 12B phases use MTP. Logs retain responses, timing, temperature, power profile, and RSS.
Inspect warm RSS trends and MTP engagement/acceptance in server logs before
accepting the soak; request success alone does not prove bounded memory or
that speculative decoding engaged.

`--model-ids`, `--depths`, `--repetitions`, and `--soak-seconds` permit targeted
debugging. Shortened runs are not the full acceptance suite. Outputs include
commands, model hashes, Git revision/diff, device descriptions, and subprocess
logs. Any missing model, checksum mismatch, failed assertion, invalid response,
or subprocess failure stops the relevant stage with a nonzero exit.

For CPU memory-safety validation, configure a separate Debug build with
`GGML_SANITIZE_ADDRESS=ON` and `GGML_SANITIZE_UNDEFINED=ON`, then run
`test-cpu-quantized-copy`. This test checks rejected operations through both
direct and planned execution, and verifies valid reshaped copies with canaries
and multiple threads.
