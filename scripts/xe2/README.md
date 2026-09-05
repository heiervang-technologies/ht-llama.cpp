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
2K context. Both set a 1 GiB RAM prompt-cache budget and a 512 MiB checkpoint budget
per slot, and verify model hashes at startup. These are cache-policy budgets,
not strict process-memory caps: the prompt cache retains at least one state,
and checkpoint eviction runs before a new state is appended. Measure actual
RSS and GPU memory, including any budget overshoot. MTP is optional and limited
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
python scripts/xe2/test_validate.py
python scripts/xe2/fetch-models.py --models-dir "$GGUFS"
cmake --build build-vulkan -j 4 --target llama-server llama-bench \
  test-backend-ops test-cpu-quantized-copy test-gemma4-attention-policy test-gemma4-device
ctest --test-dir build-vulkan --output-on-failure \
  -R 'test-(cpu-quantized-copy|gemma4-attention-policy|gemma4-logit-metrics)$'
./build-vulkan/bin/test-backend-ops test -b Vulkan0 -o CPY,CONT
./build-vulkan/bin/test-backend-ops test -b Vulkan0 -o MUL_MAT -p 'type_a=q4_0'
./build-vulkan/bin/test-backend-ops test -b Vulkan0 -o FLASH_ATTN_EXT \
  -p 'hsk=(256|512),.*type_K=(f16|q8_0|q4_0),type_V=(f16|q8_0|q4_0)'
python scripts/xe2/validate.py parity --output /tmp/xe2-validation
python scripts/xe2/validate.py smoke --output /tmp/xe2-validation
python scripts/xe2/validate.py bench --output /tmp/xe2-validation
python scripts/xe2/summarize.py /tmp/xe2-validation
python scripts/xe2/validate.py soak --output /tmp/xe2-validation
python scripts/xe2/summarize-soak.py /tmp/xe2-validation
```

Run GPU stages sequentially, on AC, with the same power profile and desktop
workload. Benchmarks rotate FA-off, FA-on, and hybrid ordering for five runs
at each of 2K/8K/16K/32K context, for both targets. Above 2K the hybrid command
intentionally follows stock FA; those runs measure fallback behavior, not a
long-context optimization. Retain only gains larger than observed variation.
Each benchmark command measures pp512, tg64, and combined pp512+tg64, starting
at depth `context - 576`. The combined test ends at the named context envelope;
standalone pp/tg finish slightly earlier. Raw JSON records the actual sizes.

Parity uses full CPU logits after continuation batches of 1/2/16, 31/32,
63/64, 128, and 1152 tokens following a GGUF-templated chat prefix;
F16 GPU paths must stay below 0.005 nats KL divergence and 0.05 total variation
against the CPU token distribution. A single-operation NMSE threshold is not
an appropriate full-model sampling bound: logits have an arbitrary common
offset, and vocabulary tails with negligible probability can dominate NMSE.
Raw NMSE is still reported for every comparison. KV reuse retains NMSE < 5e-4
in addition to the distribution checks. The metrics have model-free tests for
shift invariance, changed predictions, near ties, and non-finite rejection.
Declared suppressed tokens must retain their intentional negative-infinity
logits; all other logits must be finite. Suppressed entries are excluded from NMSE.
Top-token agreement, KL divergence, and total variation of token probabilities
are recorded separately for review. The test reports all numerical comparisons
before returning failure; it also compares F16 paths against GPU FA-off and
saves raw GPU logits beside the CPU reference. An optional final configuration
argument to `test-gemma4-device` (for example `off`) selects a diagnostic run;
it does not replace the full suite. F16/Q8_0/Q4_0 caches
must remain finite and reproduce logits after dirtying, freeing, and reusing
a suffix beyond the sliding window. CPU reference files belong to the exact
manifest model and are generated afresh by the runner.

Completion and soak requests use `/apply-template` with thinking disabled.
Raw, untemplated text is not a meaningful instruction-model lifecycle check.
Smoke requires correct capital/arithmetic answers and checks repeated-prefix greedy tokens, the unset/zero switch, quantized
cache fallback, thinking modes, a complete tool-call/result cycle, stream
cancellation and slot reuse, and the matching 12B MTP assistant. The default
soak lasts a total hour: 15 minutes per target/profile combination. Baseline
uses 8K context and one slot, with long prompts spanning multiple prefill batches;
hybrid uses 2K per slot, one slot for 12B and four concurrent slots for 26B.
Both 12B phases use MTP. Logs retain responses, timing, temperature, power profile, RSS, and per-client
DRM memory accounting. GPU allocations on this UMA device are not all reflected
in process RSS. Each MTP soak phase must actually draft and accept tokens. The soak also
checks the capital answer each time that prompt recurs.
Inspect warm RSS trends and MTP engagement/acceptance in server logs before
accepting the soak; request success alone does not prove bounded memory or
that speculative decoding engaged.

`--model-ids`, `--smoke-configs`, `--depths`, `--repetitions`, `--timeout-seconds`, and `--soak-seconds` permit targeted
debugging. Shortened runs are not the full acceptance suite. Outputs include
commands, model hashes, Git revision/diff, device descriptions, and subprocess
logs. Failed smoke comparisons retain their response evidence with `passed: false`.
Optional telemetry read failures are recorded rather than aborting the model.
Native/benchmark subprocesses have a configurable 30-minute deadline and retain
run metadata on failure or interruption.
Any missing model, checksum mismatch, failed assertion, invalid response,
or subprocess failure stops the relevant stage with a nonzero exit.

For CPU memory-safety validation, configure a separate Debug build with
`GGML_SANITIZE_ADDRESS=ON` and `GGML_SANITIZE_UNDEFINED=ON`, then run
`test-cpu-quantized-copy`. This test checks rejected operations through both
direct and planned execution, and verifies valid reshaped copies with canaries
and multiple threads.
