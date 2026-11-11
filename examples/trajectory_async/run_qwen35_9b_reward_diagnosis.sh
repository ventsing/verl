#!/usr/bin/env bash
# Fixed-evaluation reward diagnosis on an existing Ray cluster, without external plugins.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
unset VERL_USE_EXTERNAL_MODULES
export VERL_USE_EXTERNAL_PLUGINS=none
export TOTAL_STEPS=${TOTAL_STEPS:-50}
export EVALUATION_INTERVAL=${EVALUATION_INTERVAL:-10}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:--1}
export VAL_SAMPLES=${VAL_SAMPLES:-128}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
export TRAINER_NNODES=${TRAINER_NNODES:-2}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-2}
export RUN_NAME=${RUN_NAME:-qwen35_9b_reward_diagnosis_$(date +%Y%m%d_%H%M%S)}
export QWEN_OUTPUT_DIR=${QWEN_OUTPUT_DIR:-$REPO_DIR/outputs/$RUN_NAME}
export LAMINAR_RELAY_BACKEND=${LAMINAR_RELAY_BACKEND:-mooncake}
export MOONCAKE_RELAY_PROTOCOL=${MOONCAKE_RELAY_PROTOCOL:-tcp}
export MOONCAKE_RELAY_PYTHON_PATH=${MOONCAKE_RELAY_PYTHON_PATH:-$REPO_DIR/.runtime/mooncake-cpu/python}
export MOONCAKE_RELAY_LIBRARY_PATH=${MOONCAKE_RELAY_LIBRARY_PATH:-$MOONCAKE_RELAY_PYTHON_PATH/mooncake}
export VERL_FILE_LOGGER_PATH="$QWEN_OUTPUT_DIR/metrics.jsonl"
FSDP_MEMORY_FRACTION=${FSDP_MEMORY_FRACTION:-0.9}
MODE=()
case "${1:-}" in
    --prepare-only|--check-config) MODE=("$1"); shift ;;
esac
# Explicitly attach: never create a second cluster if discovery fails. Ray
# workers can inherit cluster-startup variables, so clear them in runtime_env too.
exec bash "$SCRIPT_DIR/run_qwen35_9b_16npu_laminar_reasoning.sh" "${MODE[@]}" \
    "+ray_kwargs.ray_init.address=${RAY_ADDRESS:-auto}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=''" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_PLUGINS=none \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BATCH_INVARIANT='1'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FULL_DETERMINISM='0'" \
    actor_rollout_ref.rollout.full_determinism=false \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$VERL_FILE_LOGGER_PATH'" \
    trainer.project_name=qwen35_reward_diagnosis \
    "data.val_batch_size=$VAL_SAMPLES" \
    trainer.logger='[console,file]' \
    "actor_rollout_ref.actor.fsdp_config.memory_fraction=$FSDP_MEMORY_FRACTION" \
    "actor_rollout_ref.ref.fsdp_config.memory_fraction=$FSDP_MEMORY_FRACTION" \
    actor_rollout_ref.model.enable_activation_offload=True \
    "trainer.test_freq=$EVALUATION_INTERVAL" \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=2 \
    "$@"
