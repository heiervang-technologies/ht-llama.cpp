#!/usr/bin/env bash
# Post-deploy smoke for DFlash speculative decoding via a router preset.
#
# Verifies:
#   1. Router exposes a dflash-enabled model (preset args contain draft-dflash + --model-draft)
#   2. POST /v1/chat/completions with that model id loads the drafter alongside the target
#      and produces a non-empty completion
#   3. (optional) /v1/models for that entry reports last_used_ms bumped post-request
#
# Usage:
#   scripts/smoke-dflash-deployed.sh <peer-url> <model-id>
#     e.g. scripts/smoke-dflash-deployed.sh http://127.0.0.1:8080 gemma-4-31b-dflash
#
# Exit 0 on full pass.

set -uo pipefail

if (( $# < 2 )); then
    echo "usage: $0 <peer-url> <model-id>" >&2
    exit 2
fi

PEER="$1"
MODEL="$2"

GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; RESET=$'\033[0m'
fail=0

step() {
    printf "\n${DIM}— %s${RESET}\n" "$1"
}

ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$1"; }
bad()  { printf "  ${RED}✗${RESET} %s\n" "$1"; fail=$((fail+1)); }
warn() { printf "  ${YELLOW}!${RESET} %s\n" "$1"; }

step "1. /v1/models — verify preset exposes dflash + drafter"
body=$(curl -sS --max-time 5 "$PEER/v1/models" 2>/dev/null)
[[ -z "$body" ]] && { bad "no response from /v1/models — peer unreachable"; exit 1; }

entry=$(echo "$body" | jq -e --arg model "$MODEL" '.data[] | select(.id == $model)' 2>/dev/null)
if [[ -z "$entry" ]]; then
    bad "model id '$MODEL' not present in /v1/models"
    echo "  available: $(echo "$body" | jq -r '.data[].id' | head -20 | tr '\n' ',' | sed 's/,$//')" >&2
    exit 1
fi
ok "model '$MODEL' present in router"

args=$(echo "$entry" | jq -r '.status.args | join(" ")')
if echo "$args" | grep -qE '(^| )--dflash( |$)|(^| )--spec-type draft-dflash( |$)'; then
    ok "preset args select draft-dflash"
else
    bad "preset args do NOT select draft-dflash"
    echo "  args: $args" >&2
fi

if echo "$args" | grep -qE '(^| )(--model-draft|-md|--spec-draft-model) '; then
    drafter=$(echo "$args" | grep -oE '(--model-draft|-md|--spec-draft-model) [^ ]+' | awk '{print $2}')
    ok "preset args carry drafter: $drafter"
else
    bad "preset args do NOT carry --model-draft / --spec-draft-model"
fi

step "2. POST /v1/chat/completions — drafter loads + non-empty completion"
last_used_before=$(echo "$entry" | jq -r '.last_used_ms // 0')

# enable_thinking:false to keep Gemma4-style targets from burning the small max_tokens
# budget on reasoning_content. max_tokens=64 gives enough headroom even for verbose templates.
t_start=$(date +%s%N)
request=$(jq -nc --arg model "$MODEL" \
    '{model:$model,messages:[{role:"user",content:"reply with exactly the word OK"}],max_tokens:64,stream:false,temperature:0,chat_template_kwargs:{enable_thinking:false}}')
response=$(curl -sS --max-time 120 -X POST "$PEER/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$request" 2>/dev/null)
t_end=$(date +%s%N)
elapsed_ms=$(( (t_end - t_start) / 1000000 ))

err=$(echo "$response" | jq -r '.error.message // empty' 2>/dev/null)
if [[ -n "$err" ]]; then
    bad "chat completion failed: $err"
    exit 1
fi

# content can land in .content (normal) or .reasoning_content (thinking-mode models like Gemma4).
# Either one means the model actually generated something.
content=$(echo "$response" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
reasoning=$(echo "$response" | jq -r '.choices[0].message.reasoning_content // empty' 2>/dev/null)
resolved_model=$(echo "$response" | jq -r '.model // empty' 2>/dev/null)
draft_n=$(echo "$response" | jq -r '.timings.draft_n // 0' 2>/dev/null)
draft_accepted=$(echo "$response" | jq -r '.timings.draft_n_accepted // 0' 2>/dev/null)

if [[ -n "$content" ]]; then
    ok "completion returned ($elapsed_ms ms): '${content:0:60}'"
elif [[ -n "$reasoning" ]]; then
    ok "completion in reasoning_content ($elapsed_ms ms): '${reasoning:0:60}'"
else
    bad "completion content AND reasoning_content are both empty"
fi

if (( draft_n > 0 )); then
    pct=$(awk "BEGIN{printf \"%.2f\", 100*$draft_accepted/$draft_n}")
    ok "drafter active: $draft_n drafted, $draft_accepted accepted (${pct}%)"
else
    bad "drafter inactive: timings.draft_n == 0 — dflash did not engage"
fi

if [[ "$resolved_model" == "$MODEL" ]]; then
    ok "response.model == request.model ($MODEL)"
else
    warn "response.model='$resolved_model' (differs from request — router rewrite?)"
fi

step "3. /v1/models — last_used_ms bumped after request"
body2=$(curl -sS --max-time 5 "$PEER/v1/models" 2>/dev/null)
last_used_after=$(echo "$body2" | jq -r --arg model "$MODEL" '.data[] | select(.id == $model) | .last_used_ms // 0')
if [[ "$last_used_after" -gt "$last_used_before" ]]; then
    ok "last_used_ms advanced: $last_used_before → $last_used_after"
else
    warn "last_used_ms did not advance ($last_used_before → $last_used_after) — router may not be tracking POST as a usage event"
fi

step "4. Sanity — second request to confirm drafter stays warm"
t_start=$(date +%s%N)
request2=$(jq -nc --arg model "$MODEL" \
    '{model:$model,messages:[{role:"user",content:"reply with just YES"}],max_tokens:64,stream:false,temperature:0,chat_template_kwargs:{enable_thinking:false}}')
response2=$(curl -sS --max-time 60 -X POST "$PEER/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$request2" 2>/dev/null)
t_end=$(date +%s%N)
elapsed_ms2=$(( (t_end - t_start) / 1000000 ))

content2=$(echo "$response2" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
reasoning2=$(echo "$response2" | jq -r '.choices[0].message.reasoning_content // empty' 2>/dev/null)
if [[ -n "$content2" || -n "$reasoning2" ]]; then
    ok "second request ($elapsed_ms2 ms): '${content2:-$reasoning2}'"
else
    bad "second request returned empty"
fi

echo
if (( fail > 0 )); then
    echo "${RED}FAIL${RESET}: $fail check(s) failed"
    exit 1
fi
echo "${GREEN}PASS${RESET}: dflash preset wired correctly on $PEER for $MODEL"
exit 0
