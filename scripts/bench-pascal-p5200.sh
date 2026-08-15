#!/usr/bin/env bash
# Representative bench for ht-llama.cpp on the Quadro P5200 (Pascal sm_61).
# Runs CUDA and/or Vulkan, captures JSON, prints a comparison table.
#
# Usage: scripts/bench-pascal-p5200.sh [cuda|vulkan|both]
# Defaults: both.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${PASCAL_BENCH_MODEL:-}"
OUTDIR="${PASCAL_BENCH_OUTDIR:-/tmp}"
WHICH="${1:-both}"
TS="$(date +%Y%m%d-%H%M%S)"
RESULT_FILES=()
CUDA_BIN="${PASCAL_BENCH_CUDA_BIN:-$ROOT/build-cuda/bin/llama-bench}"
VULKAN_BIN="${PASCAL_BENCH_VULKAN_BIN:-$ROOT/build-vulkan/bin/llama-bench}"

[[ -n "$MODEL" ]] || { echo "set PASCAL_BENCH_MODEL to an out-of-tree GGUF path"; exit 1; }
[[ -f "$MODEL" ]] || { echo "model missing: $MODEL"; exit 1; }
command -v jq >/dev/null || { echo "jq required"; exit 1; }

run_bench() {
    local backend="$1" bin="$2" fa_set="$3"; local out="$OUTDIR/bench-$backend-$TS.json"
    [[ -x "$bin" ]] || { echo "[$backend] binary missing: $bin — skipping"; return; }
    echo "[$backend] bench start — $(date +%H:%M:%S)"
    # Single invocation runs the cross-product of -p/-n/-fa/-ctk/-ctv values.
    "$bin" \
        -m "$MODEL" \
        -ngl 99 \
        -fa "$fa_set" \
        -p 128,512,2048 -n 32,128 \
        -ctk f16 -ctv f16 \
        -r 3 \
        -o json > "$out" 2> "${out%.json}.stderr" || {
            echo "[$backend] FAIL — stderr:"; tail -10 "${out%.json}.stderr"; return; }
    RESULT_FILES+=("$out")
    echo "[$backend] bench done — wrote $out"
}

case "$WHICH" in
    cuda)    run_bench cuda    "$CUDA_BIN"    "off,on" ;;
    vulkan)  run_bench vulkan  "$VULKAN_BIN"  "off" ;;
    both)
        run_bench cuda    "$CUDA_BIN"    "off,on"
        run_bench vulkan  "$VULKAN_BIN"  "off"
        ;;
    *) echo "usage: $0 [cuda|vulkan|both]"; exit 2 ;;
esac

# Combine + print
echo
echo "=== summary (P5200, Pascal sm_61, $(basename "$MODEL")) ==="
printf "%-8s %-7s %-7s %-3s %12s %10s\n" backend n_prompt n_gen fa avg_t/s stddev
for f in "${RESULT_FILES[@]}"; do
    backend=$(basename "$f" .json | sed -E 's/^bench-//; s/-[0-9]{8}-[0-9]{6}$//')
    jq -r --arg b "$backend" \
        ".[] | [\$b, (.n_prompt|tostring), (.n_gen|tostring), (.flash_attn|tostring), (.avg_ts|tostring), (.stddev_ts|tostring)] | @tsv" "$f" \
        | awk -F"\t" '{printf "%-8s %-7s %-7s %-3s %12.2f %10.2f\n", $1,$2,$3,$4,$5,$6}'
done
