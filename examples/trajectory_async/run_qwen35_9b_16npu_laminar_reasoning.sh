#!/usr/bin/env bash
# Dataset-aware mathematical reward and longer thinking for trajectory async RL.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
export VAL_SAMPLES=${VAL_SAMPLES:-16}
export RUN_NAME=${RUN_NAME:-qwen35_9b_laminar_reasoning_${MAX_RESPONSE_LENGTH}_$(date +%Y%m%d_%H%M%S)}
MODE=()
case "${1:-}" in
    --prepare-only|--check-config) MODE=("$1"); shift ;;
esac
case "${LAMINAR_RELAY_BACKEND:-mooncake}" in
    mooncake) BASE_LAUNCHER=run_qwen35_9b_laminar_mooncake.sh ;;
    cpu) BASE_LAUNCHER=run_qwen35_9b_16npu_laminar_24k.sh ;;
    *) printf 'LAMINAR_RELAY_BACKEND must be cpu or mooncake\n' >&2; exit 2 ;;
esac
exec bash "$SCRIPT_DIR/$BASE_LAUNCHER" "${MODE[@]}" \
    reward.custom_reward_function.path=pkg://verl.experimental.trajectory_async.reasoning_math_reward \
    reward.custom_reward_function.name=compute_score \
    +trainer.v1.sampler.sampler_kwargs.selection_strategy=mixed_ready \
    +trainer.v1.sampler.sampler_kwargs.selection_seed=42 \
    trainer.v1.laminar.weight_budget.capacity_timeout_s=3600.0 \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    "$@"
