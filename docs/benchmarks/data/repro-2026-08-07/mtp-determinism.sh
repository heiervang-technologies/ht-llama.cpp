#!/usr/bin/env bash
# Minimal repro: is greedy output repeatable on sm_61 with backend sampling?
#
# Four conditions, same prompt, same seed, greedy, N identical requests each.
#   A  MTP,         draft backend sampling ENABLED  (shipped default)
#   B  MTP,         draft backend sampling DISABLED
#   C  target only, main backend sampling DISABLED  (shipped default) [control]
#   D  target only, main backend sampling ENABLED   (-bs)             [discriminator]
#
# C is the control: if C diverges, the box/build is non-deterministic for
# reasons unrelated to backend sampling and the whole experiment is void.
# D discriminates "draft path is broken" from "backend sampling is broken".

set -uo pipefail

BIN="$HOME/ht/ht-llama.cpp/build-cuda/bin/llama"
TGT="$HOME/Models/gemma-4-12b-it-qat-q4_0.gguf"
DRAFT="$HOME/Models/mtp-gemma-4-12B-it-Q4_0.gguf"
PORT=8099
N=5
OUT="$HOME/mtp-determinism"
PROMPT="Explain in exactly four short paragraphs why RAID is not a backup. Include one practical example."

rm -rf "$OUT"; mkdir -p "$OUT"

echo "build: $("$BIN" --version 2>&1 | head -1)"
echo "gpu:   $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "target: $(sha256sum "$TGT" | cut -c1-16)  draft: $(sha256sum "$DRAFT" | cut -c1-16)"
echo

start_server() {
    # shellcheck disable=SC2068
    "$BIN" serve $@ \
        -c 4096 -ngl all -fa on --parallel 1 \
        --host 127.0.0.1 --port "$PORT" --no-webui --jinja \
        > "$OUT/$COND.server.log" 2>&1 &
    SRV=$!
    for _ in $(seq 1 300); do
        if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
        if ! kill -0 "$SRV" 2>/dev/null; then echo "  server died, see $OUT/$COND.server.log"; return 1; fi
        sleep 1
    done
    echo "  server never became healthy"; return 1
}

stop_server() {
    kill "$SRV" 2>/dev/null
    wait "$SRV" 2>/dev/null
    sleep 3
}

run_condition() {
    COND=$1; shift
    DESC=$1; shift
    echo "=== $COND: $DESC"
    if ! start_server "$@"; then echo "  SKIPPED"; return 1; fi

    for i in $(seq 1 "$N"); do
        curl -sf -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$PROMPT" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')}],
                 \"max_tokens\":128,\"temperature\":0,\"top_k\":1,\"top_p\":1,\"seed\":42,\"stream\":false}" \
            > "$OUT/$COND.$i.json"
    done
    stop_server

    python3 - "$COND" "$OUT" "$N" <<'PY'
import hashlib, json, sys, os
cond, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
digests, drafts = [], []
for i in range(1, n+1):
    p = os.path.join(out, f"{cond}.{i}.json")
    try:
        o = json.load(open(p))
    except Exception as e:
        print(f"  run {i}: UNREADABLE ({e})"); digests.append(f"err{i}"); continue
    m = o["choices"][0]["message"]
    text = (m.get("content") or "") + (m.get("reasoning_content") or "")
    t = o.get("timings", {})
    digests.append(hashlib.sha256(text.encode()).hexdigest()[:16])
    drafts.append((t.get("draft_n", 0), t.get("draft_n_accepted", 0)))
    print(f"  run {i}: sha={digests[-1]}  len={len(text):4d}  draft_n={t.get('draft_n',0)}")
uniq = len(set(digests))
mtp_active = any(d[0] > 0 for d in drafts)
print(f"  -> {uniq} distinct output(s) across {n} runs   MTP active: {mtp_active}")
print(f"  -> VERDICT: {'REPEATABLE' if uniq==1 else 'NON-REPEATABLE'}")
open(os.path.join(out, "summary.txt"), "a").write(
    f"{cond}\tdistinct={uniq}/{n}\tmtp_active={mtp_active}\n")
PY
    echo
}

run_condition A "MTP, draft backend sampling ENABLED (default)" \
    -m "$TGT" -md "$DRAFT" -ngld all --spec-type draft-mtp \
    --spec-draft-n-max 16 --spec-draft-p-min 0.9

run_condition B "MTP, draft backend sampling DISABLED" \
    -m "$TGT" -md "$DRAFT" -ngld all --spec-type draft-mtp \
    --spec-draft-n-max 16 --spec-draft-p-min 0.9 \
    --no-spec-draft-backend-sampling

run_condition C "target only, main backend sampling DISABLED (control)" \
    -m "$TGT"

run_condition D "target only, main backend sampling ENABLED (-bs)" \
    -m "$TGT" -bs

echo "=== SUMMARY ==="
cat "$OUT/summary.txt"
