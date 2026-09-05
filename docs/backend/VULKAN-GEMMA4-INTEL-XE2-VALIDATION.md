# Lunar Lake acceptance record — 2026-09-05

Status: acceptance run in progress. Hybrid remains experimental and opt-in.
No fresh speedup or deployment-readiness claim is made until the results below
are complete.

## Scope and reproducibility

Device: Core Ultra 5 238V, 32 GB UMA, Intel PCI `8086:64a0`, Linux `xe`,
Mesa/vulkan-intel 26.1.5, GCC 16.1.1. Runs use AC power and the performance
platform profile. The [model manifest](../../scripts/xe2/models.json) pins the
12B target, matching MTP assistant, and 26B A4B target by revision and SHA256.
These establish a fresh baseline; historical Xe2 artifact hashes were absent.

Use the [validation commands](../../scripts/xe2/README.md) to reproduce the
checks. Raw outputs include commands, Git revision/diff, binary version,
model hashes, device information, power/temperature samples, and process/DRM
memory accounting. Server presets set a 1 GiB prompt-cache budget and a 512 MiB checkpoint
budget per slot. Cache policies can exceed those nominal budgets by a retained
or newly appended state; measured memory use is reported separately.

## Safety and local tests

- CPU quantized-copy regression tests and ASan/UBSan pass. Unsupported block
  transposes are rejected before direct/planned execution writes output.
  An oversized-buffer reproduction changed from writing 17,268,736 bytes beyond
  the logical destination to returning failure with zero writes beyond it.
- Vulkan comparisons pass: 280 CPY/CONT, 29 Q4_0 MUL_MAT, and 247 Gemma attention
  cases with 256/512 heads and F16/Q8_0/Q4_0 K/V.
- Local CPU CI passed Debug 45/45 and Release 47/47 using `GG_BUILD_LOW_PERF=1`.
  This selection excludes the large-model/high-performance jobs.
  `LLAMA_FATAL_WARNINGS=OFF` was required for a GCC 16 warning in unchanged
  vocabulary construction code. The later model-free logit-metric test passes in Debug and Release, and all
  three telemetry/process infrastructure regression tests pass.

## Corrections to the validation method

Initial tests fed untemplated text to instruction models. Some completions were
repetitive or nonsensical even while structured chat answered correctly. The
corrected lifecycle and soak tests apply the GGUF chat template, check capital
and arithmetic answers, verify the requested tool city, and compare MTP against
target-only capital and code responses. Stream cancellation waits for an actual
generated token before closing the connection.

The native test now prefills a templated conversation, then evaluates continuation
batches of 1, 2, 16, 31, 32, 63, 64, 128, and 1152 tokens. This exercises the
attention thresholds and sliding-window reuse with a valid instruction prefix.

A single-operation NMSE limit was also an unsuitable full-model sampling test.
Logits have an arbitrary common offset; negligible-probability vocabulary tails
can dominate their squared error. Full-model comparisons now require KL divergence
below 0.005 nats and total variation below 0.05. Raw NMSE and top-token agreement
remain in every report. KV-reuse checks additionally retain NMSE below `5e-4`.
Model-free checks cover shift invariance, changed predictions, near ties, and
non-finite rejection. Declared suppressed tokens must remain negative infinity;
other logits must be finite.

The original raw-text failures are retained as diagnostic evidence. An isolated
build of the original `ht` baseline (`06d9d42`) produced byte-identical full
logits at all nine lengths for both models with FA-off and hybrid. This establishes
that those numerical differences predate the safety hardening; it does not
establish universal CPU/GPU equivalence. A dequantized F16 CPU reference and
higher-precision Vulkan trial explained much of the short-prompt difference,
but the original long, repetitive raw prompt remained numerically sensitive.

## End-to-end results

| Check | 12B | 26B A4B |
|---|---|---|
| FA-off/hybrid distributions and KV reuse | Pass | Pass |
| Plain-FA-on comparison | Pass | **Fail**: KL at 16 tokens |
| Templated lifecycle, fallback, tools, cancellation | Pass (6 profiles) | Pass (5 profiles) |
| Baseline and hybrid soak, 15 minutes each | Pending | Pending |
| Five interleaved runs per mode at 2K/8K/16K/32K | Pending | Pending |

The complete 26B parity sweep returns failure: plain FA-on at 16 continuation
tokens has KL `0.00991865776` against the CPU reference, exceeding `0.005`.
Its top token still agrees. This is not waived or converted into a passing test.
Both offered presets pass their comparisons; plain FA-on is a benchmark/control
configuration, not a recommended serving preset.

| Model/mode | Maximum CPU-reference KL | Maximum total variation | Top-token agreement |
|---|---:|---:|---:|
| 12B FA-off | 0.00115015 | 0.0186292 | 9/9 |
| 12B hybrid | 0.000639504 | 0.0116276 | 9/9 |
| 26B FA-off | 0.00312896 | 0.0339058 | 9/9 |
| 26B hybrid | 0.00457187 | 0.0359762 | 9/9 |
| 26B plain FA-on control | **0.00991866** | 0.0359762 | 9/9 |

Automatic FA with hybrid enabled produces byte-identical logits to explicit
FA-on with hybrid enabled on both models. Worst hybrid KV-reuse NMSE is
`0.000430299` for 12B and `0.0000268791` for 26B. F16 FA-off/FA-on and Q8/Q4
reuse comparisons are exact. Quantized-cache checks establish finite outputs
and correct reuse, not quantization-quality equivalence to the F16 reference.

Above 2K, and with multiple configured sequences or quantized caches, the hybrid
switch intentionally retains stock attention. Benchmarks at those settings
measure fallback behavior. The four-slot 26B soak likewise exercises fallback,
not eligibility for hybrid attention.
