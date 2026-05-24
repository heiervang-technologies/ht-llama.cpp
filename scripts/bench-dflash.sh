#!/usr/bin/env bash
# DFlash speculative decoding bench harness.
#
# Runs llama-speculative-simple across drafter quants and prompt classes,
# reporting acceptance rate and decode throughput. Each (drafter, prompt)
# pair runs N times so variance is visible (DFlash bench has ±2-3pp
# run-to-run variance even at temp=0 / fixed seed).
#
# VRAM requirement: ~22 GB free (target Q4_K_M ~18 GB + drafter ~1-3 GB +
# compute). Coordinate centurion-llm scale-down before running.
#
# Usage:
#   scripts/bench-dflash.sh [--quants Q4,Q6,Q8,BF16] [--runs 3] [--ctx 4096]
#
# Output goes to /tmp/dflash-bench-<timestamp>.md with a markdown summary
# table at the bottom.
#
# Reference acceptance per vLLM PR #41703 on Gemma4-26B-A4B-it (similar
# arch / Anbeeld-style drafter):
#   - HumanEval prompts: ~44.69% accept
#   - MT-Bench prompts:  ~21.68% accept
# Our prompt set covers both classes for direct comparison.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build-cuda/bin/llama-speculative-simple"
TARGET="$ROOT/models/gemma-4-31B-it-Q4_K_M.gguf"
DRAFTER_DIR="$ROOT/models/dflash-gemma4-31b-gguf"
TS=$(date +%Y%m%d-%H%M%S)
OUT="/tmp/dflash-bench-$TS.md"

QUANTS="Q4_K_M,Q6_K,Q8_0,bf16"
RUNS=3
CTX=4096

while (( $# )); do
    case "$1" in
        --quants) QUANTS="$2"; shift 2 ;;
        --runs)   RUNS="$2";   shift 2 ;;
        --ctx)    CTX="$2";    shift 2 ;;
        --help|-h)
            sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -E 's/^# ?//' | head -n -1
            exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# Prompts: one MT-Bench-class (conversational), one HumanEval-class (code).
MTBENCH_PROMPT="Write a 50-word paragraph about speculative decoding."
HUMANEVAL_PROMPT='Complete the Python function:
def fibonacci(n: int) -> list[int]:
    """Return the first n Fibonacci numbers as a list."""'

# Sanity: VRAM and binary.
if [[ ! -x "$BIN" ]]; then
    echo "missing or non-executable: $BIN" >&2
    echo "build with: cmake --build build-cuda --target llama-speculative-simple -j" >&2
    exit 1
fi
if [[ ! -f "$TARGET" ]]; then
    echo "missing target model: $TARGET" >&2
    exit 1
fi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if (( FREE_MIB < 20000 )); then
    echo "WARNING: only ${FREE_MIB} MiB free on GPU; need ~22000. Bench will likely OOM." >&2
    echo "Coordinate centurion-llm scale-down via snoop-kube first." >&2
    read -r -p "Continue anyway? [y/N] " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

run_one() {
    local label="$1" drafter="$2" prompt="$3" out_dir="$4"
    local err="$out_dir/$label.err"
    local out="$out_dir/$label.out"

    timeout 240 "$BIN" \
        -m "$TARGET" \
        -md "$drafter" \
        --dflash \
        -p "$prompt" \
        -n 128 -c "$CTX" -ngl 99 -ngld 99 -fa on \
        --temp 0 --seed 1 \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        > "$out" 2> "$err" || true

    local acc tok
    acc=$(grep 'accept    =' "$err" | tail -1 | awk -F'= ' '{print $2}' | tr -d ' ')
    tok=$(grep 'decoded ' "$err" | tail -1 | awk '{print $(NF-1), $NF}')
    echo "$acc|$tok"
}

OUT_DIR="/tmp/dflash-bench-$TS-runs"
mkdir -p "$OUT_DIR"

echo "# DFlash bench $TS" | tee "$OUT"
echo "" | tee -a "$OUT"
echo "Target: $(basename "$TARGET")" | tee -a "$OUT"
echo "Build: $(cd "$ROOT" && git rev-parse --short HEAD)" | tee -a "$OUT"
echo "Quants: $QUANTS" | tee -a "$OUT"
echo "Runs per condition: $RUNS" | tee -a "$OUT"
echo "ctx: $CTX, q8_0 KV cache" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "| drafter | prompt | run 1 | run 2 | run 3 | mean | tok/s (last run) |" | tee -a "$OUT"
echo "|---|---|---:|---:|---:|---:|---|" | tee -a "$OUT"

IFS=, read -ra QUANT_ARR <<< "$QUANTS"
for quant in "${QUANT_ARR[@]}"; do
    drafter="$DRAFTER_DIR/gemma4-31b-it-dflash-${quant}.gguf"
    [[ -f "$drafter" ]] || { echo "skip (missing): $drafter" >&2; continue; }

    for prompt_label in "MT-Bench" "HumanEval"; do
        if [[ "$prompt_label" == "MT-Bench" ]]; then
            prompt="$MTBENCH_PROMPT"
        else
            prompt="$HUMANEVAL_PROMPT"
        fi

        acc_sum=0
        results=()
        last_tok=""
        for ((r=1; r<=RUNS; r++)); do
            label="${quant}_${prompt_label}_${r}"
            result=$(run_one "$label" "$drafter" "$prompt" "$OUT_DIR")
            acc=$(echo "$result" | cut -d'|' -f1)
            tok=$(echo "$result" | cut -d'|' -f2)
            results+=("$acc")
            last_tok="$tok"

            num=$(echo "$acc" | tr -d '%')
            if [[ "$num" =~ ^[0-9.]+$ ]]; then
                acc_sum=$(awk -v s="$acc_sum" -v a="$num" 'BEGIN{print s+a}')
            fi
        done

        mean=$(awk -v s="$acc_sum" -v n="$RUNS" 'BEGIN{printf "%.2f%%", s/n}')
        row="| $quant | $prompt_label | ${results[0]:-—} | ${results[1]:-—} | ${results[2]:-—} | $mean | $last_tok |"
        echo "$row" | tee -a "$OUT"
    done
done

echo "" | tee -a "$OUT"
echo "Per-run stderr at $OUT_DIR/*.err" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "Done. Summary written to $OUT"
