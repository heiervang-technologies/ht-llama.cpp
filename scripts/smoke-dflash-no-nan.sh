#!/usr/bin/env bash
# Regression smoke: catch the dflash NaN-logits bug class (mission m-20260527-103737).
#
# Bug profile: server-side spec_decode integration emits NaN drafter logits on the
# /v1/chat/completions path (jinja chat template). Drafter argmaxes to <pad> for every
# position, target rejects every draft, accept rate is 0%. dflash silently adds zero
# value while consuming GPU cycles.
#
# This script:
#   1. POSTs a chat completion to a deployed dflash router preset
#   2. Asserts the response had a non-zero acceptance count (drafter produced at least
#      ONE valid logit that target accepted)
#   3. Reports per-run + aggregate accept rate
#
# Usage:
#   scripts/smoke-dflash-no-nan.sh <peer-url> <dflash-model-id> [N_RUNS=3]
#   e.g. scripts/smoke-dflash-no-nan.sh http://127.0.0.1:8080 gemma-4-31b-dflash-Q6_K
#
# Override: SMOKE_RUNS=10 scripts/smoke-dflash-no-nan.sh ...
#
# Exit 0: no NaN signature, healthy accept count
# Exit 1: NaN-pattern detected (drafts proposed, zero accepts across all runs)
# Exit 2: bad request / unreachable peer
#
# CI integration suggestion:
#   - run as part of a deploy-verify step against a known dflash peer
#   - run multiple consecutive requests (SMOKE_RUNS=5+) to catch intermittent NaN
#   - target the /v1/chat/completions path specifically — the bug class does NOT
#     manifest on /v1/completions (jinja chat template is the trigger)

set -uo pipefail

if (( $# < 2 )); then
    echo "usage: $0 <peer-url> <dflash-model-id> [N_RUNS]" >&2
    exit 2
fi

PEER="$1"
MODEL="$2"
N_RUNS="${3:-${SMOKE_RUNS:-3}}"

GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; RESET=$'\033[0m'

total_drafted=0
total_accepted=0
fail_runs=0

for r in $(seq 1 "$N_RUNS"); do
    request=$(jq -nc --arg model "$MODEL" \
        '{model:$model,messages:[{role:"user",content:"Write five short haikus about the ocean."}],max_tokens:128,stream:false,temperature:0,chat_template_kwargs:{enable_thinking:false}}')
    body=$(curl -sS --max-time 60 -X POST "$PEER/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$request" 2>/dev/null)

    if [[ -z "$body" ]]; then
        echo "${RED}FAIL run $r${RESET}: empty response from $PEER"
        exit 2
    fi

    drafted=$(echo "$body" | jq -r '.timings.draft_n // 0')
    accepted=$(echo "$body" | jq -r '.timings.draft_n_accepted // 0')
    pred=$(echo "$body" | jq -r '.usage.completion_tokens // 0')
    finish=$(echo "$body" | jq -r '.choices[0].finish_reason // "?"')

    total_drafted=$((total_drafted + drafted))
    total_accepted=$((total_accepted + accepted))

    if (( drafted >= 50 && accepted == 0 )); then
        echo "${RED}FAIL run $r${RESET}: $drafted drafted, $accepted accepted, finish=$finish — NaN signature"
        fail_runs=$((fail_runs + 1))
    else
        local_rate=$(awk -v a="$accepted" -v d="$drafted" 'BEGIN { printf "%.2f", (d > 0 ? 100*a/d : 0) }')
        echo "${DIM}run $r${RESET}: $drafted drafted, $accepted accepted (${local_rate}%), $pred tokens, finish=$finish"
    fi
done

if (( total_drafted == 0 )); then
    echo "${YELLOW}WARN${RESET}: zero drafts proposed across $N_RUNS runs — dflash may not be engaged at all"
    exit 1
fi

overall_rate=$(awk -v a="$total_accepted" -v d="$total_drafted" 'BEGIN { printf "%.2f", 100*a/d }')

if (( fail_runs > 0 )); then
    echo
    echo "${RED}FAIL${RESET}: $fail_runs / $N_RUNS runs hit the NaN signature (drafts >=50, accepts 0)"
    echo "Overall: $total_accepted / $total_drafted = ${overall_rate}% accept across $N_RUNS runs"
    exit 1
fi

if (( total_accepted == 0 )); then
    echo
    echo "${RED}FAIL${RESET}: zero total accepts across $N_RUNS runs ($total_drafted drafted) — likely NaN"
    exit 1
fi

echo
echo "${GREEN}PASS${RESET}: $total_accepted / $total_drafted accepts = ${overall_rate}% across $N_RUNS runs (NaN signature absent)"
exit 0
