#!/usr/bin/env bash
# Single-node Laminar semantics; portable CPU relay and recompute migration.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export RUN_NAME=${RUN_NAME:-qwen35_9b_${ALGORITHM:-grpo}_laminar_thinking24k_$(date +%Y%m%d_%H%M%S)}
export QWEN_OUTPUT_DIR=${QWEN_OUTPUT_DIR:-$REPO_DIR/outputs/$RUN_NAME}
MODE=()
case "${1:-}" in
    --prepare-only|--check-config) MODE=("$1"); shift ;;
esac
exec bash "$SCRIPT_DIR/run_qwen35_9b_16npu_trajectory_async_24k.sh" "${MODE[@]}" \
    trainer.v1.trainer_mode=laminar \
    "actor_rollout_ref.rollout.max_num_seqs=${LAMINAR_ENGINE_MAX_SEQS:-64}" \
    trainer.v1.laminar.run_migration_probe=True \
    data.dataloader_num_workers=0 \
    data.gen_batch_size=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=cpu_relay \
    actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=verl.experimental.trajectory_async.cpu_relay \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.rollout_correction.bypass_mode=True \
    actor_rollout_ref.rollout.logprobs_mode=processed_logprobs \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    trainer.v1.sampler.max_off_policy_strategy=none \
    "trainer.v1.laminar.event_dir=$QWEN_OUTPUT_DIR/trajectories" \
    "$@"
