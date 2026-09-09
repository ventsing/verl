#!/usr/bin/env bash
# trajectory-async | scenario: multi-version weight store | mock rollout | CPU simulation
# The weight plane end to end: publish per actor update, per-replica pulls at batch
# boundaries, retention GC (keep_last). P2P_BACKEND=fake runs the full pipeline on
# CPU; kimi/mooncake route the same orchestration to the real checkpoint engines
# and exit with wiring instructions when the engine is absent (run on the cluster).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}

########################### user-adjustable ###########################
P2P_BACKEND=${P2P_BACKEND:-fake}    # fake | kimi | mooncake
REPLICAS=${REPLICAS:-4}
KEEP_LAST_VERSIONS=${KEEP_LAST_VERSIONS:-2}   # retention: versions kept staged
WEIGHT_MB=${WEIGHT_MB:-256}         # actor weight size (drives stage/pull bytes)

NUM_PROMPTS=${NUM_PROMPTS:-32}
N=${N:-4}
MINI_BATCH_GROUPS=${MINI_BATCH_GROUPS:-4}
SEED=${SEED:-3}
BATCH_PER_REPLICA=${BATCH_PER_REPLICA:-8}
ACTOR_STALL=${ACTOR_STALL:-0.5}     # publish path: offload + register (s)
########################### end user-adjustable ###########################

if [ "${P2P_BACKEND}" != "fake" ]; then
    echo "NOTE: P2P_BACKEND=${P2P_BACKEND} requires the real engine on a cluster;"
    echo "      on a bare machine the demo exits with the exact wiring instructions."
fi

FLAGS=(
    --mode trajectory
    --weight-store p2p
    --p2p-backend "${P2P_BACKEND}"
    --replicas "${REPLICAS}"
    --keep-last-versions "${KEEP_LAST_VERSIONS}"
    --weight-mb "${WEIGHT_MB}"
    --num-prompts "${NUM_PROMPTS}"
    --n "${N}"
    --mini-batch-groups "${MINI_BATCH_GROUPS}"
    --seed "${SEED}"
    --batch-per-replica "${BATCH_PER_REPLICA}"
    --actor-stall-s "${ACTOR_STALL}"
)

"${PYTHON}" "${SCRIPT_DIR}/launch_demo.py" "${FLAGS[@]}" "$@"
