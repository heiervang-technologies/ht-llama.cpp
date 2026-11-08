#!/usr/bin/env bash
set -uo pipefail
BIN="$HOME/ht/ht-llama.cpp/build-cuda/bin/llama"
TGT="$HOME/Models/gemma-4-12b-it-qat-q4_0.gguf"
DRAFT="$HOME/Models/mtp-gemma-4-12B-it-Q4_0.gguf"
PORT=8099; OUT="$HOME/mtp-determinism"
PROMPT="Explain in exactly four short paragraphs why RAID is not a backup. Include one practical example."
"$BIN" serve -m "$TGT" -md "$DRAFT" -ngld all \
  -c 4096 -ngl all -fa on --parallel 1 --host 127.0.0.1 --port $PORT --no-webui --jinja \
  > "$OUT/G2.server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 300); do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }; sleep 1; done
echo "server alive: $(kill -0 $SRV 2>/dev/null && echo yes || echo no)"
code=$(curl -s -o "$OUT/G2.1.json" -w "%{http_code}" -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":$(printf %s "$PROMPT" | python3 -c "import json,sys;print(json.dumps(sys.stdin.read()))")}],\"max_tokens\":128,\"temperature\":0,\"top_k\":1,\"top_p\":1,\"seed\":42,\"cache_prompt\":false,\"stream\":false}")
echo "http=$code  bytes=$(stat -c%s "$OUT/G2.1.json")"
head -c 300 "$OUT/G2.1.json"; echo
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
