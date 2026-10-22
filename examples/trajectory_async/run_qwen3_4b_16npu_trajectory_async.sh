#!/usr/bin/env bash
# Real Qwen3 trajectory-level async GRPO/GSPO: 8 trainer NPUs + 8 rollout NPUs.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"

case "${1:-}" in
    --help|-h)
        printf 'Usage: bash %s [--check-config|--prepare-only] [Hydra overrides...]\n' "$0"
        printf 'Defaults: 8 trainer + 8 rollout NPUs, GRPO, 2 training steps.\n'
        printf 'Use ALGORITHM=gspo for GSPO; TOTAL_STEPS controls training length.\n'
        exit 0
        ;;
    --prepare-only) exec bash "$SCRIPT_DIR/run_qwen3_4b_16npu.sh" --prepare-only ;;
esac
MODE=()
if [[ "${1:-}" == --check-config ]]; then
    MODE=(--check-config)
    shift
fi

export NDEVICES=16
export ROLLOUT_N=${ROLLOUT_N:-4}
export RUN_NAME=${RUN_NAME:-qwen3_4b_${ALGORITHM:-grpo}_trajectory_async_8_8_$(date +%Y%m%d_%H%M%S)}

# "nccl" selects HCCLCheckpointEngine on Ascend in this repository.
# The CPU mock/demo and the experimental P2P adapters are not used by this run.
exec bash "$SCRIPT_DIR/run_qwen3_4b_16npu.sh" "${MODE[@]}" \
    trainer.v1.trainer_mode=separate_async \
    trainer.n_gpus_per_node=8 \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    trainer.v1.separate_async.parameter_sync_step=1 \
    trainer.v1.separate_async.num_warmup_batches=2 \
    trainer.v1.separate_async.hybrid_rollout.enable_switch=False \
    trainer.v1.sampler.custom_sampler.path=pkg://verl.experimental.trajectory_async.tq_sampler \
    trainer.v1.sampler.custom_sampler.name=TrajectoryReplayBuffer \
    "+trainer.v1.sampler.sampler_kwargs.rollout_n=$ROLLOUT_N" \
    +trainer.v1.sampler.sampler_kwargs.val_rollout_n=1 \
    algorithm.filter_groups.enable=False \
    "$@"
