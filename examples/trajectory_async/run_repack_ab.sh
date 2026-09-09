#!/usr/bin/env bash
# trajectory-async | scenario: repack A/B (active migration) | mock multi-replica rollout | CPU simulation
# Laminar-style active scheduling (arXiv 2510.12633 §5): identical workload with repack
# OFF vs ON. Key metrics: throughput, KVCache utilization, per-round migration KV
# effect (repack/rounds), inherent staleness. Exits non-zero if repack changes what
# is trained (data equivalence is a hard invariant).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}

########################### user-adjustable ###########################
MODE=${MODE:-trajectory}            # delivery granularity: trajectory | group
NUM_PROMPTS=${NUM_PROMPTS:-48}
N=${N:-8}                           # rollout.n — responses per prompt
MINI_BATCH_GROUPS=${MINI_BATCH_GROUPS:-4}
SEED=${SEED:-7}

REPLICAS=${REPLICAS:-4}             # rollout replicas (needs >= 2 to consolidate)
BATCH_PER_REPLICA=${BATCH_PER_REPLICA:-12}   # assignment quota per replica activation
MAX_RUNNING=${MAX_RUNNING:-24}      # B: roofline decode-batch bound
KV_CAPACITY=${KV_CAPACITY:-32768}   # C_max per replica (tokens)
DECODE_RATE=${DECODE_RATE:-800}     # per-request decode rate (tok/s)
REPACK_INTERVAL=${REPACK_INTERVAL:-0.5}      # periodic trigger (s)
REPACK_OVERHEAD=${REPACK_OVERHEAD:-0.5}      # cost of one migration round (s)
UPDATE_TIME=${UPDATE_TIME:-0.5}     # simulated actor update duration (s)

WEIGHT_STORE=${WEIGHT_STORE:-p2p}   # p2p (multi-version store) | relay (timing mock)
P2P_BACKEND=${P2P_BACKEND:-fake}    # fake | kimi | mooncake (real engines need a cluster)
KEEP_LAST_VERSIONS=${KEEP_LAST_VERSIONS:-2}
########################### end user-adjustable ###########################

FLAGS=(
    --mode "${MODE}"
    --num-prompts "${NUM_PROMPTS}"
    --n "${N}"
    --mini-batch-groups "${MINI_BATCH_GROUPS}"
    --seed "${SEED}"
    --replicas "${REPLICAS}"
    --batch-per-replica "${BATCH_PER_REPLICA}"
    --max-running "${MAX_RUNNING}"
    --kv-capacity-tokens "${KV_CAPACITY}"
    --decode-rate-tok-s "${DECODE_RATE}"
    --repack-interval-s "${REPACK_INTERVAL}"
    --repack-overhead-s "${REPACK_OVERHEAD}"
    --update-time-s "${UPDATE_TIME}"
    --weight-store "${WEIGHT_STORE}"
    --p2p-backend "${P2P_BACKEND}"
    --keep-last-versions "${KEEP_LAST_VERSIONS}"
    --compare-repack
)

"${PYTHON}" "${SCRIPT_DIR}/launch_demo.py" "${FLAGS[@]}" "$@"
