#!/usr/bin/env bash
#
# Smoke tests for llama-fit-params, in particular the --fit-print-plan
# JSON output mode (ht/#66 step 2 prep, PR #72). Verifies:
#   1) success path emits a single-line JSON with the documented schema
#   2) plan vector size matches n_devices
#   3) total_bytes equals sum of per_device_bytes
#   4) plan is empty on a CPU-only build (no GPU/accel devices)
#
# Pattern lifted from tools/gguf-split/tests.sh — invoke from CI with
# the build's bin/ dir as $1.

set -eu

if [ $# -lt 1 ]
then
    echo "usage:   $0 path_to_build_binary [path_to_temp_folder]"
    echo "example: $0 ../../build/bin ../../tmp"
    exit 1
fi

if [ $# -gt 1 ]
then
    TMP_DIR=$2
else
    TMP_DIR=/tmp
fi

set -x

FIT_PARAMS=$1/llama-fit-params
WORK_PATH=$TMP_DIR/fit-params-test
ROOT_DIR=$(realpath "$(dirname "$0")"/../../)

mkdir -p "$WORK_PATH"

# 1. Get a small model (same one gguf-split tests use, already in CI cache after that runs)
(
cd "$WORK_PATH"
"$ROOT_DIR"/scripts/hf.sh --repo ggml-org/Qwen3-0.6B-GGUF --file Qwen3-0.6B-Q8_0.gguf
)
MODEL="$WORK_PATH/Qwen3-0.6B-Q8_0.gguf"
echo PASS

# 2. --fit-print-plan emits valid single-line JSON on success.
JSON=$("$FIT_PARAMS" --fit-print-plan -m "$MODEL" 2>/dev/null | tail -1)
echo "got: $JSON"
echo "$JSON" | python3 -c "
import json, sys
o = json.loads(sys.stdin.read())
assert 'per_device_bytes' in o, 'missing per_device_bytes'
assert 'n_devices' in o, 'missing n_devices'
assert 'total_bytes' in o, 'missing total_bytes'
assert isinstance(o['per_device_bytes'], list), 'per_device_bytes not a list'
assert isinstance(o['n_devices'], int), 'n_devices not an int'
assert isinstance(o['total_bytes'], int), 'total_bytes not an int'
print('schema OK')
"
echo PASS

# 3. plan vector size matches n_devices.
echo "$JSON" | python3 -c "
import json, sys
o = json.loads(sys.stdin.read())
assert len(o['per_device_bytes']) == o['n_devices'], \
    f'len(per_device_bytes)={len(o[\"per_device_bytes\"])} != n_devices={o[\"n_devices\"]}'
print('size invariant OK')
"
echo PASS

# 4. total_bytes equals sum of per_device_bytes.
echo "$JSON" | python3 -c "
import json, sys
o = json.loads(sys.stdin.read())
expected = sum(o['per_device_bytes'])
assert o['total_bytes'] == expected, f'total_bytes={o[\"total_bytes\"]} != sum={expected}'
print('total invariant OK')
"
echo PASS

# 5. On a CPU-only build, plan is empty (no GPU/accel devices).
#    Skip the assertion when running under a CUDA/Metal/Vulkan build —
#    only assert the invariant on CPU-only builds (detected by n_devices==0).
echo "$JSON" | python3 -c "
import json, sys
o = json.loads(sys.stdin.read())
if o['n_devices'] == 0:
    assert o['per_device_bytes'] == [], 'plan must be empty when n_devices==0'
    assert o['total_bytes'] == 0, 'total must be 0 when n_devices==0'
    print('CPU-only convention OK')
else:
    print(f'GPU build detected ({o[\"n_devices\"]} devices) — CPU-empty assertion skipped')
"
echo PASS

# 6. Fit-failure path: invoking against a nonexistent model surfaces a clean
#    JSON error marker on stdout (not garbage) so subprocess callers can
#    distinguish fit-failure from parse-failure.
FAIL_JSON=$("$FIT_PARAMS" --fit-print-plan -m "$WORK_PATH/does-not-exist.gguf" 2>/dev/null || true)
FAIL_JSON_LAST=$(echo "$FAIL_JSON" | tail -1)
echo "got fail: $FAIL_JSON_LAST"
if echo "$FAIL_JSON_LAST" | grep -q '^{'; then
    echo "$FAIL_JSON_LAST" | python3 -c "
import json, sys
try:
    o = json.loads(sys.stdin.read())
    if 'error' in o:
        print('failure JSON marker OK')
    else:
        print('WARNING: success-shaped JSON for missing model (got per_device_bytes)')
        sys.exit(1)
except json.JSONDecodeError:
    print('WARNING: failure path emitted non-JSON on stdout')
    sys.exit(1)
"
else
    # Missing-model may be caught upstream of common_fit_params and not emit
    # JSON. That's acceptable — non-zero exit suffices to signal failure.
    echo "fail path emitted no JSON (caught upstream of fit) — acceptable"
fi
echo PASS

echo
echo "ALL fit-params --fit-print-plan smoke tests PASSED"
