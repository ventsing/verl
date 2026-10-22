#!/usr/bin/env bash
# Per-node Mooncake CPU relays; each TP replica must remain within one node.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export RUN_NAME=${RUN_NAME:-qwen35_9b_laminar_mooncake_$(date +%Y%m%d_%H%M%S)}
export QWEN_OUTPUT_DIR=${QWEN_OUTPUT_DIR:-$REPO_DIR/outputs/$RUN_NAME}
MODE=()
case "${1:-}" in
    --prepare-only|--check-config) MODE=("$1"); shift ;;
esac
RELAY_ENV=()
if [[ -n ${MOONCAKE_RELAY_PYTHON_PATH:-} ]]; then
    RELAY_ENV+=("++trainer.v1.laminar.mooncake.runtime_env.env_vars.PYTHONPATH='${MOONCAKE_RELAY_PYTHON_PATH}:$REPO_DIR:${PYTHONPATH:-}'")
fi
if [[ -n ${MOONCAKE_RELAY_LIBRARY_PATH:-} ]]; then
    RELAY_ENV+=("++trainer.v1.laminar.mooncake.runtime_env.env_vars.LD_LIBRARY_PATH='${MOONCAKE_RELAY_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}'")
fi
exec bash "$SCRIPT_DIR/run_qwen35_9b_16npu_laminar_24k.sh" "${MODE[@]}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend=mooncake_relay \
    actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=verl.experimental.trajectory_async.mooncake_relay \
    "trainer.nnodes=${TRAINER_NNODES:-1}" \
    "trainer.n_gpus_per_node=${TRAINER_NPUS_PER_NODE:-8}" \
    "actor_rollout_ref.rollout.nnodes=${ROLLOUT_NNODES:-1}" \
    "actor_rollout_ref.rollout.n_gpus_per_node=${ROLLOUT_NPUS_PER_NODE:-8}" \
    "trainer.v1.laminar.mooncake.protocol=${MOONCAKE_RELAY_PROTOCOL:-rdma}" \
    "trainer.v1.laminar.mooncake.device_name='${MOONCAKE_RELAY_DEVICE_NAME:-}'" \
    "${RELAY_ENV[@]}" "$@"
