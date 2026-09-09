#!/usr/bin/env bash
# trajectory-async | scenario: staleness control | mock multi-replica rollout | CPU simulation
# Freshness control: run the same workload unbounded vs with --staleness-drop 1.
# The bounded run must refuse groups older than the bound (trainer/oldest_staleness
# max <= bound, groups_dropped_stale > 0) at the cost of some dropped data — the
# freshness/throughput trade of fully-async training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}

########################### user-adjustable ###########################
NUM_PROMPTS=${NUM_PROMPTS:-12}
N=${N:-4}
MINI_BATCH_GROUPS=${MINI_BATCH_GROUPS:-2}
SEED=${SEED:-17}
STALENESS_DROP=${STALENESS_DROP:-1}  # max allowed version gap of trained groups

REPLICAS=${REPLICAS:-2}
BATCH_PER_REPLICA=${BATCH_PER_REPLICA:-6}
DECODE_RATE=${DECODE_RATE:-1200}
UPDATE_TIME=${UPDATE_TIME:-0.3}
WEIGHT_STORE=${WEIGHT_STORE:-p2p}
P2P_BACKEND=${P2P_BACKEND:-fake}
########################### end user-adjustable ###########################

BASE=(
    --mode trajectory
    --num-prompts "${NUM_PROMPTS}"
    --n "${N}"
    --mini-batch-groups "${MINI_BATCH_GROUPS}"
    --seed "${SEED}"
    --replicas "${REPLICAS}"
    --batch-per-replica "${BATCH_PER_REPLICA}"
    --decode-rate-tok-s "${DECODE_RATE}"
    --update-time-s "${UPDATE_TIME}"
    --weight-store "${WEIGHT_STORE}"
    --p2p-backend "${P2P_BACKEND}"
)

echo "=== run 1: no staleness bound (fully async, all data trained) ==="
"${PYTHON}" "${SCRIPT_DIR}/launch_demo.py" "${BASE[@]}"

echo
echo "=== run 2: staleness-drop=${STALENESS_DROP} (freshness control) ==="
"${PYTHON}" "${SCRIPT_DIR}/launch_demo.py" "${BASE[@]}" --staleness-drop "${STALENESS_DROP}" "$@"
