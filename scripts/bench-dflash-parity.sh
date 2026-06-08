#!/usr/bin/env bash
# DFlash deployment-parity bench harness.
#
# Runs scripts/dflash-parity-prompts.json against llama-speculative-simple and
# emits per-prompt acceptance stats (tau, n_accept, n_drafted, decode_t/s) as
# JSON. snoop-kube runs the SAME prompts against vLLM/SGLang with
# z-lab/gemma-4-31B-it-DFlash on titan and emits the same JSON shape, so the
# two outputs can be diffed cell-by-cell.
#
# Apples-to-apples: greedy temp=0, --spec-draft-n-max = block_size - 1 = 15,
# fixed seed, --fa on. Target/drafter chosen via env (defaults to local Q-quant
# pair so it runs in 24 GB; titan should override with BF16 target).
#
# Usage:
#   scripts/bench-dflash-parity.sh [--out <file>]
#
# Env:
#   DFLASH_PARITY_TARGET     default: models/gemma-4-31B-it-IQ4_XS.gguf
#   DFLASH_PARITY_DRAFTER    default: models/dflash-gemma4-31b-gguf/gemma4-31b-it-dflash-Q6_K.gguf
#   DFLASH_PARITY_PROMPTS    default: scripts/dflash-parity-prompts.json
#   DFLASH_PARITY_BIN        default: build/bin/llama-speculative-simple
#
# Output (stdout + --out):
#   {
#     "build": "f6feddb49",
#     "target": "gemma-4-31B-it-IQ4_XS.gguf",
#     "drafter": "gemma4-31b-it-dflash-Q6_K.gguf",
#     "block_size": 16,
#     "spec_draft_n_max": 15,
#     "temp": 0.0,
#     "results": [
#       { "id": "...", "class": "mt_bench",
#         "n_predict": 33, "n_drafted": 480, "n_accept": 4,
#         "tau": 1.1379, "decode_tps": 18.7 },
#       ...
#     ]
#   }
# tau is computed as n_predict / (n_predict - n_accept), the same convention
# the z-lab reference reports.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${DFLASH_PARITY_BIN:-$ROOT/build/bin/llama-speculative-simple}"
TARGET="${DFLASH_PARITY_TARGET:-$ROOT/models/gemma-4-31B-it-IQ4_XS.gguf}"
DRAFTER="${DFLASH_PARITY_DRAFTER:-$ROOT/models/dflash-gemma4-31b-gguf/gemma4-31b-it-dflash-Q6_K.gguf}"
PROMPTS="${DFLASH_PARITY_PROMPTS:-$ROOT/scripts/dflash-parity-prompts.json}"
OUT="/tmp/dflash-parity-$(date +%Y%m%d-%H%M%S).json"

while (( $# )); do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -E 's/^# ?//' | head -n -1
            exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

for f in "$BIN" "$TARGET" "$DRAFTER" "$PROMPTS"; do
    [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

command -v jq >/dev/null || { echo "jq required" >&2; exit 1; }

BUILD=$(cd "$ROOT" && git rev-parse --short HEAD)
BLOCK=$(jq -r '.block_size' "$PROMPTS")
N_MAX=$(jq -r '.spec_draft_n_max' "$PROMPTS")
N_PROMPTS=$(jq -r '.prompts | length' "$PROMPTS")

mt_bench_tokens=$(jq -r '.max_new_tokens_per_class.mt_bench' "$PROMPTS")
humaneval_tokens=$(jq -r '.max_new_tokens_per_class.humaneval' "$PROMPTS")
gsm8k_tokens=$(jq -r '.max_new_tokens_per_class.gsm8k' "$PROMPTS")

tokens_for_class() {
    case "$1" in
        mt_bench)   echo "$mt_bench_tokens" ;;
        humaneval)  echo "$humaneval_tokens" ;;
        gsm8k)      echo "$gsm8k_tokens" ;;
        *)          echo 128 ;;
    esac
}

ERRDIR="${OUT%.json}-stderr"
mkdir -p "$ERRDIR"

results_json="[]"
for i in $(seq 0 $((N_PROMPTS - 1))); do
    id=$(jq -r ".prompts[$i].id" "$PROMPTS")
    class=$(jq -r ".prompts[$i].class" "$PROMPTS")
    text=$(jq -r ".prompts[$i].text" "$PROMPTS")
    n_predict=$(tokens_for_class "$class")
    err="$ERRDIR/$id.err"

    echo "[bench] $id ($class) n_predict=$n_predict" >&2
    timeout 240 "$BIN" \
        -m "$TARGET" -md "$DRAFTER" \
        --dflash --spec-draft-n-max "$N_MAX" \
        -p "$text" -n "$n_predict" \
        -c 8192 -ngl 99 -ngld 99 -fa on \
        --temp 0 --seed 1 \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        > /dev/null 2> "$err" || true

    n_drafted=$(grep -E 'n_drafted\s*=' "$err" | tail -1 | awk -F'=' '{print $2}' | tr -d ' ')
    n_accept=$(grep -E 'n_accept\s*=' "$err"  | tail -1 | awk -F'=' '{print $2}' | tr -d ' ')
    decoded_line=$(grep -E 'decoded\s+[0-9]+\s+tokens' "$err" | tail -1)
    n_pred=$(echo "$decoded_line" | awk '{print $(NF-7)}')
    decode_tps=$(echo "$decoded_line" | awk '{print $(NF-1)}')

    if [[ -z "${n_drafted:-}" || -z "${n_accept:-}" || -z "${n_pred:-}" ]]; then
        echo "[bench] $id: failed to parse counters (see $err)" >&2
        cell=$(jq -n --arg id "$id" --arg class "$class" \
            '{id:$id, class:$class, error:"parse_failed"}')
    else
        tau=$(awk -v p="$n_pred" -v a="$n_accept" 'BEGIN{ d=p-a; if (d<=0) print "inf"; else printf "%.4f", p/d }')
        cell=$(jq -n --arg id "$id" --arg class "$class" \
            --argjson npred  "$n_pred" \
            --argjson ndraft "$n_drafted" \
            --argjson nacc   "$n_accept" \
            --arg     tau    "$tau" \
            --arg     tps    "$decode_tps" \
            '{id:$id, class:$class, n_predict:$npred, n_drafted:$ndraft, n_accept:$nacc, tau:($tau|tonumber? // null), decode_tps:($tps|tonumber? // null)}')
    fi
    results_json=$(echo "$results_json" | jq --argjson c "$cell" '. + [$c]')
done

jq -n \
    --arg build "$BUILD" \
    --arg target "$(basename "$TARGET")" \
    --arg drafter "$(basename "$DRAFTER")" \
    --argjson block "$BLOCK" \
    --argjson nmax  "$N_MAX" \
    --argjson results "$results_json" \
    '{build:$build, target:$target, drafter:$drafter, block_size:$block, spec_draft_n_max:$nmax, temp:0.0, results:$results}' \
    | tee "$OUT"

echo "[bench] wrote $OUT (stderr per-prompt at $ERRDIR/)" >&2
