#!/usr/bin/env bash
# Natural long-tail reasoning: DeepScaleR, thinking, 24K response budget, 8+8 NPUs.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
export DATASET=${DATASET:-deepscaler}
export ENABLE_THINKING=True
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-24576}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:--1}
export VAL_SAMPLES=${VAL_SAMPLES:-8}
export TOTAL_STEPS=${TOTAL_STEPS:-4}
export ROLLOUT_MEM_UTIL=${ROLLOUT_MEM_UTIL:-0.55}
export RUN_NAME=${RUN_NAME:-qwen3_4b_${ALGORITHM:-grpo}_trajectory_async_thinking24k_$(date +%Y%m%d_%H%M%S)}
export OUTPUT_DIR=${QWEN_OUTPUT_DIR:-$REPO_DIR/outputs/$RUN_NAME}
MODE=()
case "${1:-}" in
    --help|-h)
        printf 'Usage: bash %s [--prepare-only|--check-config] [Hydra overrides...]\n' "$0"
        printf 'Defaults: DeepScaleR, thinking=True, response=24576, prompt=2048, 8+8 NPUs, 4 steps.\n'
        exit 0 ;;
    --prepare-only) exec bash "$SCRIPT_DIR/run_qwen3_4b_16npu.sh" --prepare-only ;;
    --check-config) MODE=(--check-config); shift ;;
esac
exec bash "$SCRIPT_DIR/run_qwen3_4b_16npu_trajectory_async.sh" "${MODE[@]}" \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    trainer.val_before_train=False \
    trainer.test_freq=4 \
    "trainer.rollout_data_dir=$OUTPUT_DIR/rollouts" \
    "+trainer.v1.sampler.sampler_kwargs.trajectory_log_dir=$OUTPUT_DIR/trajectories" \
    "+trainer.v1.sampler.sampler_kwargs.response_limit=$MAX_RESPONSE_LENGTH" \
    "$@"
