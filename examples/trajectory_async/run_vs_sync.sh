#!/usr/bin/env bash
# trajectory-async | scenario: sync vs trajectory-async | mock multi-replica rollout | CPU simulation
# The Laminar paper's headline comparison (Fig. 3(a) vs 3(e)): the same workload on
# the same replicas, synchronous RL (trainer waits for the slowest trajectory before
# its first update) vs trajectory-level async (updates start on complete groups while
# the long tail still generates). Key metrics: wall time, time-to-first-update,
# throughput, inherent staleness. Exits non-zero if the two train on different data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}

########################### user-adjustable ###########################
MODE=${MODE:-trajectory}
NUM_PROMPTS=${NUM_PROMPTS:-48}
N=${N:-8}
MINI_BATCH_GROUPS=${MINI_BATCH_GROUPS:-4}
SEED=${SEED:-7}
REPLICAS=${REPLICAS:-4}
BATCH_PER_REPLICA=${BATCH_PER_REPLICA:-12}
DECODE_RATE=${DECODE_RATE:-800}
UPDATE_TIME=${UPDATE_TIME:-0.5}
REPACK=${REPACK:-on}               # async side's active scheduling
WEIGHT_STORE=${WEIGHT_STORE:-p2p}
P2P_BACKEND=${P2P_BACKEND:-fake}
########################### end user-adjustable ###########################

FLAGS=(
    --mode "${MODE}"
    --num-prompts "${NUM_PROMPTS}"
    --n "${N}"
    --mini-batch-groups "${MINI_BATCH_GROUPS}"
    --seed "${SEED}"
    --replicas "${REPLICAS}"
    --batch-per-replica "${BATCH_PER_REPLICA}"
    --decode-rate-tok-s "${DECODE_RATE}"
    --update-time-s "${UPDATE_TIME}"
    --repack "${REPACK}"
    --weight-store "${WEIGHT_STORE}"
    --p2p-backend "${P2P_BACKEND}"
    --compare-vs-sync
)

"${PYTHON}" "${SCRIPT_DIR}/launch_demo.py" "${FLAGS[@]}" "$@"
