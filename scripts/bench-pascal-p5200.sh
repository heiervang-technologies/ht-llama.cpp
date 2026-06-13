#!/usr/bin/env bash
# Representative bench for ht-llama.cpp on the Quadro P5200 (Pascal sm_61).
# Runs CUDA and/or Vulkan, captures JSON, prints a comparison table.
#
# Usage: scripts/bench-pascal-p5200.sh [cuda|vulkan|both]
# Defaults: both.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${PASCAL_BENCH_MODEL:-$HOME/Models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
OUTDIR="${PASCAL_BENCH_OUTDIR:-/tmp}"
WHICH="${1:-both}"

[[ -f "$MODEL" ]] || { echo "model missing: $MODEL"; exit 1; }
command -v jq >/dev/null || { echo "jq required"; exit 1; }

run_bench() {
    local backend="$1" bin="$2" fa_set="$3"; local out="$OUTDIR/bench-$backend.json"
    [[ -x "$bin" ]] || { echo "[$backend] binary missing: $bin — skipping"; return; }
    echo "[$backend] bench start — $(date +%H:%M:%S)"
    # Single invocation runs the cross-product of -p/-n/-fa/-ctk/-ctv values.
    "$bin" \
        -m "$MODEL" \
        -ngl 99 \
        -fa $fa_set \
        -p 128,512,2048 -n 32,128 \
        -ctk f16 -ctv f16 \
        -r 3 \
        -o json > "$out" 2> "${out%.json}.stderr" || {
            echo "[$backend] FAIL — stderr:"; tail -10 "${out%.json}.stderr"; return; }
    echo "[$backend] bench done — wrote $out"
}

case "$WHICH" in
    cuda)    run_bench cuda    "$ROOT/build-cuda/bin/llama-bench"    "0,1" ;;
    vulkan)  run_bench vulkan  "$ROOT/build-vulkan/bin/llama-bench"  "0" ;;
    both)
        run_bench cuda    "$ROOT/build-cuda/bin/llama-bench"    "0,1"
        run_bench vulkan  "$ROOT/build-vulkan/bin/llama-bench"  "0"
        ;;
    *) echo "usage: $0 [cuda|vulkan|both]"; exit 2 ;;
esac

# Combine + print
echo
echo "=== summary (P5200, Pascal sm_61, Llama-3.1-8B Q4_K_M) ==="
printf "%-8s %-7s %-7s %-3s %12s %10s\n" backend n_prompt n_gen fa avg_t/s stddev
for f in "$OUTDIR"/bench-cuda.json "$OUTDIR"/bench-vulkan.json; do
    [[ -f "$f" ]] || continue
    backend=$(basename "$f" .json | sed s/bench-//)
    jq -r --arg b "$backend" \
        ".[] | [\$b, (.n_prompt|tostring), (.n_gen|tostring), (.flash_attn|tostring), (.avg_ts|tostring), (.stddev_ts|tostring)] | @tsv" "$f" \
        | awk -F"\t" '{printf "%-8s %-7s %-7s %-3s %12.2f %10.2f\n", $1,$2,$3,$4,$5,$6}'
done
