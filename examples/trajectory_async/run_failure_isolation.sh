#!/usr/bin/env bash
# trajectory-async | scenario: failure isolation | mock rollout | CPU simulation
# Two phases over the SAME seeded workload:
#   1. sanity  — no failures: trajectory- and group-level delivery must train on
#      identical data (exit code enforced).
#   2. contrast — per-attempt failures with an all-or-nothing group baseline
#      (--group-retry none): trajectory-level delivery saves the groups the
#      baseline loses whole (measurement — read the contrast table).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}

########################### user-adjustable ###########################
NUM_PROMPTS=${NUM_PROMPTS:-24}
N=${N:-8}
MINI_BATCH_GROUPS=${MINI_BATCH_GROUPS:-4}
SEED=${SEED:-5}
FAILURE_RATE=${FAILURE_RATE:-0.25}  # per-attempt failure probability (phase 2)
MAX_RETRIES=${MAX_RETRIES:-2}
REWARD_LATENCY=${REWARD_LATENCY:-0.2}
PREPROCESS_LATENCY=${PREPROCESS_LATENCY:-0.0}
########################### end user-adjustable ###########################

BASE=(
    --num-prompts "${NUM_PROMPTS}"
    --n "${N}"
    --mini-batch-groups "${MINI_BATCH_GROUPS}"
    --seed "${SEED}"
    --max-retries "${MAX_RETRIES}"
    --reward-latency-s "${REWARD_LATENCY}"
    --preprocess-latency-s "${PREPROCESS_LATENCY}"
)

echo "=== phase 1: no failures — delivery granularity must not change data ==="
"${PYTHON}" "${SCRIPT_DIR}/launch_demo.py" --mode trajectory --compare "${BASE[@]}"

echo
echo "=== phase 2: failure-rate=${FAILURE_RATE}, baseline all-or-nothing ==="
"${PYTHON}" "${SCRIPT_DIR}/launch_demo.py" --mode trajectory --compare \
    --failure-rate "${FAILURE_RATE}" --group-retry none "${BASE[@]}" "$@"
