#!/usr/bin/env bash
# DFlash TARGET-precision sweep bench (Round-12).
#
# WHY THIS EXISTS (distinct from bench-dflash.sh):
#   bench-dflash.sh pins the target at Q4_K_M and sweeps the *drafter* quant.
#   But the DFlash drafter is trained on the target's BF16 hidden states, and
#   we feed it features extracted from a *quantized* target. So the most likely
#   cause of the 8% vs published ~21% accept gap is TARGET quant noise shifting
#   the hidden states off the distribution the drafter was trained on. This
#   harness sweeps the TARGET (-m) with the drafter (-md) FIXED, to test that.
#
# It also hardens the statistics vs the old harness (item 3 — trustworthy bench):
#   1. Accept is recomputed exactly from the binary's raw n_accept/n_drafted
#      counts (with the printed 'accept    =' percentage line as a fallback),
#      rather than from a 3-run mean of the rounded percentage.
#   2. It reports mean +/- sample stddev over N runs, not just a 3-run mean,
#      and flags cross-target deltas that fall within combined noise. DFlash
#      accept has ~2pp run-to-run variance, so sub-2pp deltas are noise.
#
# SAFETY: every target is validated with scripts/gguf-meta.py --check-instruct
#   before use. Base fine-tune (e.g. $GGUFS/gemma-4-31B.gguf) and truncated/stub
#   GGUFs (e.g. the 1.4GB gemma-4-31B-it-Q5_K_M.gguf) are REFUSED — benching the
#   instruct-trained drafter against a base target is a confounded experiment.
#
# Usage:
#   scripts/bench-dflash-target-sweep.sh \
#       [--targets <comma list of gguf paths or $GGUFS basenames>] \
#       [--drafter Q6_K] [--runs 10] [--temp 0] [--seed-base 1] [--ctx 4096] [-n 128]
#
# Default target list is the clean instruct targets currently on disk PLUS the
# high-precision targets that the Round-12 download/convert will produce; missing
# ones are skipped with a note (so this is the one command to re-run once F16/Q8
# land).
#
# Output: /tmp/dflash-target-sweep-<ts>.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build-cuda/bin/llama-speculative-simple"
META="$ROOT/scripts/gguf-meta.py"
GGUFS_DIR="${GGUFS:-$ROOT/models}"
DRAFTER_DIR="${DFLASH_DRAFTER_DIR:-$ROOT/models/dflash-gemma4-31b-gguf}"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="/tmp/dflash-target-sweep-$TS.md"

# Default sweep: low -> high precision instruct targets. F16/Q8 produced by the
# Round-12 convert step; listed here so this command "just works" once they exist.
TARGETS="gemma-4-31B-it-IQ4_XS.gguf,gemma-4-31B-it-Q4_K_M.gguf,gemma-4-31B-it-Q8_0.gguf,gemma-4-31B-it-BF16.gguf"
DRAFTER="Q6_K"
RUNS=10
TEMP=0
SEED_BASE=1
CTX=4096
NGEN=128

while (( $# )); do
    case "$1" in
        --targets)   TARGETS="$2"; shift 2 ;;
        --drafter)   DRAFTER="$2"; shift 2 ;;
        --runs)      RUNS="$2"; shift 2 ;;
        --temp)      TEMP="$2"; shift 2 ;;
        --seed-base) SEED_BASE="$2"; shift 2 ;;
        --ctx)       CTX="$2"; shift 2 ;;
        -n)          NGEN="$2"; shift 2 ;;
        --help|-h)   sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -E 's/^# ?//' | head -n -1; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

MTBENCH_PROMPT="Write a 50-word paragraph about speculative decoding."
HUMANEVAL_PROMPT='Complete the Python function:
def fibonacci(n: int) -> list[int]:
    """Return the first n Fibonacci numbers as a list."""'

[[ -x "$BIN" ]] || { echo "missing/non-executable: $BIN" >&2
    echo "build: cmake --build build-cuda --target llama-speculative-simple -j" >&2; exit 1; }
[[ -f "$META" ]] || { echo "missing gguf validator: $META" >&2; exit 1; }

drafter_path="$DRAFTER_DIR/gemma4-31b-it-dflash-${DRAFTER}.gguf"
[[ -f "$drafter_path" ]] || { echo "missing drafter: $drafter_path" >&2; exit 1; }

# Resolve a target token to an absolute path: accept abs path, else look in $GGUFS.
resolve_target() {
    local t="$1"
    if [[ -f "$t" ]]; then echo "$t"; return 0; fi
    if [[ -f "$GGUFS_DIR/$t" ]]; then echo "$GGUFS_DIR/$t"; return 0; fi
    return 1
}

# accept-rate parser. Each LOG_INF line in stderr is prefixed with a
# timestamp + level letter (e.g. "0.05.300.148 I n_drafted = 144"), so
# substring patterns are NOT line-anchored — anchored regex silently misses.
#
# Format strings (verified against build-cuda/bin/llama-speculative-simple):
#     <timestamp> I n_drafted = <int>
#     <timestamp> I n_accept  = <int>
#     <timestamp> I accept    = <float>%      (FOUR spaces before '=')
#
# Recompute the FRACTION from raw counts (n_accept / n_drafted) — exact and
# aggregates cleanly. Falls back to the printed percentage if counts absent;
# fallback requires actual digits in the percent field (skips '-nan%' from
# degenerate n_drafted=0 runs to avoid silent misparse).
parse_accept() {
    local err="$1" nd na
    nd="$(grep -E 'n_drafted = ' "$err" | tail -1 | sed -nE 's/.*n_drafted = ([0-9]+).*/\1/p')"
    na="$(grep -E 'n_accept  = ' "$err" | tail -1 | sed -nE 's/.*n_accept  = ([0-9]+).*/\1/p')"
    if [[ "$nd" =~ ^[0-9]+$ && "$na" =~ ^[0-9]+$ && "$nd" -gt 0 ]]; then
        awk -v a="$na" -v d="$nd" 'BEGIN{printf "%.6f", a/d}'
        return 0
    fi
    grep -E 'accept    = ' "$err" | tail -1 \
        | sed -nE 's/.*accept    = ([0-9]+\.[0-9]+)%.*/\1/p' \
        | awk '{if(NF==1 && $1 ~ /^[0-9.]+$/){printf "%.6f", $1/100}}'
}
parse_tps() {
    grep -E 'decoded .* t/s' "$1" | tail -1 | sed -E 's/.*speed: *([0-9.]+) t\/s/\1/'
}

run_one() {  # target_path drafter_path prompt seed out_dir label
    local tgt="$1" dft="$2" prompt="$3" seed="$4" out_dir="$5" label="$6"
    local err="$out_dir/$label.err"
    timeout 360 "$BIN" \
        -m "$tgt" -md "$dft" --dflash \
        -p "$prompt" \
        -n "$NGEN" -c "$CTX" -ngl 99 -ngld 99 -fa on \
        --temp "$TEMP" --seed "$seed" \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        > "$out_dir/$label.out" 2> "$err" || true
    local acc tps
    acc="$(parse_accept "$err")"; tps="$(parse_tps "$err")"
    echo "${acc:-NA}|${tps:-NA}"
}

# mean + sample stddev (percent) from a list of fractional accept rates.
stats() {  # space-separated fractions -> "mean_pct std_pct n_ok"
    awk '{
        n=0; s=0; ss=0;
        for (i=1;i<=NF;i++) if ($i+0==$i && $i!="") { v=$i*100; a[n++]=v; s+=v; ss+=v*v }
        if (n==0) { print "NA NA 0"; exit }
        m=s/n;
        sd=(n>1)?sqrt((ss-n*m*m)/(n-1)):0;
        printf "%.2f %.2f %d", m, sd, n
    }' <<< "$1"
}

OUT_DIR="/tmp/dflash-target-sweep-$TS-runs"; mkdir -p "$OUT_DIR"

{
    echo "# DFlash TARGET-precision sweep $TS"
    echo ""
    echo "- Build: $(cd "$ROOT" && git rev-parse --short HEAD)"
    echo "- Drafter (FIXED): gemma4-31b-it-dflash-${DRAFTER}.gguf"
    echo "- Runs/condition: $RUNS   temp: $TEMP   seed-base: $SEED_BASE   ctx: $CTX   n: $NGEN   KV: q8_0"
    echo "- Accept recomputed from raw n_accept/n_drafted counts (fallback: printed 'accept    =' line)."
    echo "- Reference (vLLM PR #41703, Gemma4 Anbeeld-style): MT-Bench ~21.7%, HumanEval ~44.7%."
    echo ""
    echo "| target | prompt | mean accept | stddev | n | tok/s | per-run |"
    echo "|---|---|---:|---:|---:|---:|---|"
} | tee "$OUT"

declare -A MEAN_MT MEAN_HE STD_MT STD_HE

IFS=, read -ra TGT_ARR <<< "$TARGETS"
for tname in "${TGT_ARR[@]}"; do
    tpath="$(resolve_target "$tname" || true)"
    if [[ -z "${tpath:-}" ]]; then
        echo "| $tname | — | SKIP (not on disk) | | | | |" | tee -a "$OUT"
        continue
    fi
    if ! python3 "$META" --check-instruct "$tpath" >/dev/null 2>/tmp/sweep_reject.$$; then
        reason="$(tr -s ' ' < /tmp/sweep_reject.$$)"
        echo "| $tname | — | **REFUSED** | | | | $reason |" | tee -a "$OUT"
        echo "REFUSED target $tname: $reason" >&2
        continue
    fi

    for plabel in "MT-Bench" "HumanEval"; do
        prompt="$MTBENCH_PROMPT"; [[ "$plabel" == "HumanEval" ]] && prompt="$HUMANEVAL_PROMPT"
        accs=""; last_tps="NA"
        for ((r=0; r<RUNS; r++)); do
            seed=$(( TEMP == 0 ? SEED_BASE : SEED_BASE + r ))  # temp0: fixed seed; temp>0: vary
            label="${tname%%.gguf}_${plabel}_${r}"
            res="$(run_one "$tpath" "$drafter_path" "$prompt" "$seed" "$OUT_DIR" "$label")"
            accs+=" $(cut -d'|' -f1 <<< "$res" | tr -d '%')"
            last_tps="$(cut -d'|' -f2 <<< "$res")"
        done
        read -r mean sd n <<< "$(stats "$accs")"
        per_run="$(echo "$accs" | sed -E 's/ /, /g; s/^, //')"
        echo "| $tname | $plabel | ${mean}% | ±${sd} | $n | ${last_tps} | ${per_run} |" | tee -a "$OUT"
        if [[ "$plabel" == "MT-Bench" ]]; then MEAN_MT[$tname]=$mean; STD_MT[$tname]=$sd
        else MEAN_HE[$tname]=$mean; STD_HE[$tname]=$sd; fi
    done
done

# Noise-aware comparison vs the Q4_K_M baseline, if present.
BASE="gemma-4-31B-it-Q4_K_M.gguf"
{
    echo ""
    echo "## Delta vs $BASE (is any precision gain real, or within noise?)"
    echo ""
    if [[ -n "${MEAN_MT[$BASE]:-}" ]]; then
        for tname in "${TGT_ARR[@]}"; do
            [[ "$tname" == "$BASE" || -z "${MEAN_MT[$tname]:-}" ]] && continue
            for cls in MT-Bench HumanEval; do
                if [[ "$cls" == "MT-Bench" ]]; then
                    m=${MEAN_MT[$tname]}; mb=${MEAN_MT[$BASE]}; s=${STD_MT[$tname]}; sb=${STD_MT[$BASE]}
                else
                    m=${MEAN_HE[$tname]}; mb=${MEAN_HE[$BASE]}; s=${STD_HE[$tname]}; sb=${STD_HE[$BASE]}
                fi
                read -r d verdict <<< "$(awk -v m="$m" -v mb="$mb" -v s="$s" -v sb="$sb" 'BEGIN{
                    d=m-mb; comb=sqrt(s*s+sb*sb);
                    printf "%.2f %s", d, (d>comb ? "REAL(>1sigma)" : "within-noise")
                }')"
                echo "- $tname [$cls]: ${d}pp vs baseline (combined sigma ~$(awk -v s="$s" -v sb="$sb" 'BEGIN{printf "%.2f", sqrt(s*s+sb*sb)}')) -> $verdict"
            done
        done
    else
        echo "(baseline $BASE not in this run — add it to --targets for delta analysis)"
    fi
    echo ""
    echo "Per-run stderr: $OUT_DIR/*.err"
} | tee -a "$OUT"

echo "Done -> $OUT"
