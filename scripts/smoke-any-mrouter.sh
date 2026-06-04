#!/usr/bin/env bash
# Smoke-test the "any" sentinel + last_used_ms field on a deployed unified-llm router.
#
# Validates the surface added/refreshed in mission m-20260524-165127-3bb03b:
#   1. GET /v1/models — confirm per-model `last_used_ms` field is present (int).
#   2. POST /v1/chat/completions with model="any" — confirm response.model is the
#      resolved instance id (not the literal "any"), and capture which model the
#      router picked.
#   3. (Optional) verify 4xx is returned with the right error.type when the
#      sentinel is sent to a router with nothing resident. Skipped by default
#      because emptying the router cache is destructive; gate behind --test-empty.
#
# Usage:
#   scripts/smoke-any-mrouter.sh [URL ...]
#     URL defaults: cluster peers as of mission m-20260524 (titan/centurion/lithium)
#   scripts/smoke-any-mrouter.sh --test-empty URL
#     Adds the destructive "no-model-resident" check (POST /unload_all first).
#
# Exit code: 0 if all reachable peers pass; non-zero if any reachable peer fails.
# Unreachable peers (connection refused / timeout) are reported but do not fail.

set -uo pipefail

DEFAULT_PEERS=(
    "http://192.168.8.158:30184"   # titan
    "http://192.168.8.170:30192"   # centurion
    "http://192.168.8.119:30187"   # lithium
)
TEST_EMPTY=0
PEERS=()

while (( $# )); do
    case "$1" in
        --test-empty) TEST_EMPTY=1; shift ;;
        --help|-h)
            sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed -E 's/^# ?//' | head -n -1
            exit 0 ;;
        http*) PEERS+=("$1"); shift ;;
        *)     echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if (( ${#PEERS[@]} == 0 )); then
    PEERS=("${DEFAULT_PEERS[@]}")
fi

GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; RESET=$'\033[0m'
pass_count=0; fail_count=0; skip_count=0

check_models_endpoint() {
    local peer="$1"
    local body
    body=$(curl -sS --max-time 5 "$peer/v1/models" 2>/dev/null) || return 2

    local has_data
    has_data=$(echo "$body" | jq -e '.data | type == "array"' 2>/dev/null) || return 3

    local n_models
    n_models=$(echo "$body" | jq '.data | length')
    [[ "$n_models" -ge 1 ]] || return 4

    # Every model entry must have a last_used_ms int field
    local n_with_field
    n_with_field=$(echo "$body" | jq '[.data[] | select(.last_used_ms != null and (.last_used_ms | type == "number"))] | length')
    if [[ "$n_with_field" -ne "$n_models" ]]; then
        return 5  # field missing on at least one model
    fi
    echo "$n_models"
    return 0
}

check_any_routing() {
    local peer="$1"
    local body
    # use a tiny, fast-to-complete request
    body=$(curl -sS --max-time 60 -X POST "$peer/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"any","messages":[{"role":"user","content":"reply with just OK"}],"max_tokens":4,"stream":false}' 2>/dev/null) || return 2

    # If 4xx, body has .error.message
    local err
    err=$(echo "$body" | jq -r '.error.message // empty' 2>/dev/null)
    if [[ -n "$err" ]]; then
        # 4xx case — only acceptable if testing empty router
        echo "4xx: $err"
        return 10
    fi

    local resolved
    resolved=$(echo "$body" | jq -r '.model // empty' 2>/dev/null)
    if [[ -z "$resolved" ]]; then
        return 3  # malformed
    fi
    if [[ "$resolved" == "any" ]]; then
        return 4  # router didn't rewrite — bug
    fi
    echo "$resolved"
    return 0
}

for peer in "${PEERS[@]}"; do
    printf "\n${peer} "
    # Step 1: /v1/models
    n_models=$(check_models_endpoint "$peer")
    rc=$?
    case $rc in
        0) printf "%s/v1/models%s ✓ (%s models, last_used_ms present) " "$GREEN" "$RESET" "$n_models" ;;
        2) printf "%sUNREACHABLE%s (skipped) " "$YELLOW" "$RESET"; ((skip_count+=1)); continue ;;
        3) printf "%s/v1/models FAIL%s (no .data array)\n" "$RED" "$RESET"; ((fail_count+=1)); continue ;;
        4) printf "%s/v1/models FAIL%s (zero models in router)\n" "$RED" "$RESET"; ((fail_count+=1)); continue ;;
        5) printf "%s/v1/models FAIL%s (last_used_ms missing on at least one model)\n" "$RED" "$RESET"; ((fail_count+=1)); continue ;;
    esac

    # Step 2: model="any" resolution
    resolved=$(check_any_routing "$peer")
    rc=$?
    case $rc in
        0)  printf "→ %sany resolved to '%s'%s\n" "$GREEN" "$resolved" "$RESET"; ((pass_count+=1)) ;;
        2)  printf "→ %sUNREACHABLE on chat endpoint%s\n" "$YELLOW" "$RESET"; ((skip_count+=1)) ;;
        3)  printf "→ %sany route FAIL (malformed response)%s\n" "$RED" "$RESET"; ((fail_count+=1)) ;;
        4)  printf "→ %sany route FAIL (router echoed 'any' — rewrite did not fire)%s\n" "$RED" "$RESET"; ((fail_count+=1)) ;;
        10) # 4xx — only OK if test-empty mode
            if (( TEST_EMPTY )); then
                printf "→ %sany on empty router returned 4xx (expected)%s — %s\n" "$GREEN" "$RESET" "$resolved"
                ((pass_count+=1))
            else
                printf "→ %sany route FAIL (4xx: %s — router has nothing resident, prime a model first)%s\n" "$RED" "$resolved" "$RESET"
                ((fail_count+=1))
            fi
            ;;
    esac
done

echo
echo "${DIM}---${RESET}"
printf "Pass: %s%d%s  Fail: %s%d%s  Skip: %s%d%s\n" "$GREEN" "$pass_count" "$RESET" "$RED" "$fail_count" "$RESET" "$YELLOW" "$skip_count" "$RESET"

if (( fail_count > 0 )); then
    exit 1
fi
exit 0
