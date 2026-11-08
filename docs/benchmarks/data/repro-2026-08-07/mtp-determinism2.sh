#!/usr/bin/env bash
# Follow-up: test the prompt-cache hypothesis.
#
# Hypothesis: the run-1-vs-rest divergence is caused by prompt cache reuse
# changing the prompt batch split (run 1: cache_n=0 prompt_n=34;
# runs 2+: cache_n=7 prompt_n=27), not by backend sampling or MTP.
#
# Prediction, which can fail: with "cache_prompt": false every request
# evaluates the full prompt identically, so all N runs must be identical --
# in BOTH the target-only and the MTP condition.
#
#   E  target only, cache_prompt=false
#   F  MTP,         cache_prompt=false
#
# If E or F still diverges, the hypothesis is wrong.

set -uo pipefail

BIN="$HOME/ht/ht-llama.cpp/build-cuda/bin/llama"
TGT="$HOME/Models/gemma-4-12b-it-qat-q4_0.gguf"
DRAFT="$HOME/Models/mtp-gemma-4-12B-it-Q4_0.gguf"
PORT=8099
N=5
OUT="$HOME/mtp-determinism"
PROMPT="Explain in exactly four short paragraphs why RAID is not a backup. Include one practical example."

mkdir -p "$OUT"

start_server() {
    # shellcheck disable=SC2068
    "$BIN" serve $@ \
        -c 4096 -ngl all -fa on --parallel 1 \
        --host 127.0.0.1 --port "$PORT" --no-webui --jinja \
        > "$OUT/$COND.server.log" 2>&1 &
    SRV=$!
    for _ in $(seq 1 300); do
        curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$SRV" 2>/dev/null || { echo "  server died"; return 1; }
        sleep 1
    done
    echo "  never healthy"; return 1
}

run_condition() {
    COND=$1; shift
    DESC=$1; shift
    echo "=== $COND: $DESC"
    start_server "$@" || { echo "  SKIPPED"; return 1; }

    for i in $(seq 1 "$N"); do
        curl -sf -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$PROMPT" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')}],
                 \"max_tokens\":128,\"temperature\":0,\"top_k\":1,\"top_p\":1,\"seed\":42,
                 \"cache_prompt\":false,\"stream\":false}" \
            > "$OUT/$COND.$i.json"
    done
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 3

    python3 - "$COND" "$OUT" "$N" <<'PY'
import hashlib, json, sys, os
cond, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
digests = []
for i in range(1, n+1):
    o = json.load(open(os.path.join(out, f"{cond}.{i}.json")))
    m = o["choices"][0]["message"]
    text = (m.get("content") or "") + (m.get("reasoning_content") or "")
    t = o.get("timings", {})
    digests.append(hashlib.sha256(text.encode()).hexdigest()[:16])
    print(f"  run {i}: sha={digests[-1]}  len={len(text):4d}  "
          f"cache_n={t.get('cache_n')} prompt_n={t.get('prompt_n')} draft_n={t.get('draft_n')}")
uniq = len(set(digests))
print(f"  -> {uniq} distinct output(s) across {n} runs")
print(f"  -> VERDICT: {'REPEATABLE' if uniq==1 else 'NON-REPEATABLE'}")
open(os.path.join(out, "summary.txt"), "a").write(f"{cond}\tdistinct={uniq}/{n}\n")
PY
    echo
}

run_condition E "target only, cache_prompt=false" -m "$TGT"

run_condition F "MTP, cache_prompt=false" \
    -m "$TGT" -md "$DRAFT" -ngld all --spec-type draft-mtp \
    --spec-draft-n-max 16 --spec-draft-p-min 0.9

echo "=== SUMMARY (all conditions) ==="
cat "$OUT/summary.txt"
