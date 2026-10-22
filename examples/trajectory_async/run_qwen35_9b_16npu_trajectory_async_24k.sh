#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export MODEL_PATH=${MODEL_PATH:-/opt/huawei/dataset/ccb_data/code/ventsing/downloaded_models/Qwen3.5-9B}
export UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-2048}
export VLLM_LOGGING_LEVEL=INFO
export RUN_NAME=${RUN_NAME:-qwen35_9b_${ALGORITHM:-grpo}_trajectory_async_thinking24k_$(date +%Y%m%d_%H%M%S)}
export QWEN_OUTPUT_DIR=${QWEN_OUTPUT_DIR:-$REPO_DIR/outputs/$RUN_NAME}
MODE=()
case "${1:-}" in
    --prepare-only|--check-config) MODE=("$1"); shift ;;
esac
if [[ ${#MODE[@]} == 0 && "${1:-}" != --help && "${1:-}" != -h ]]; then
    "${PYTHON_BIN:-python3}" - "$QWEN_OUTPUT_DIR" "$RUN_NAME" "$$" <<'PY'
import json
import os
import sys
from pathlib import Path
state_path = Path(os.getenv("QWEN_STATE_FILE", "/tmp/qwen35_longtail_run.json"))
state = json.loads(state_path.read_text()) if state_path.exists() else {}
state["previous_output_dir"] = state.get("output_dir")
state.update(output_dir=sys.argv[1], run_name=sys.argv[2], training_pid=int(sys.argv[3]),
             training_status="starting", response_budget=int(os.getenv("MAX_RESPONSE_LENGTH", "24576")))
state.pop("session_id", None)
for key, variable in [("ray_job_id", "QWEN_RAY_JOB_ID"), ("ray_dashboard", "QWEN_RAY_DASHBOARD")]:
    state.pop(key, None)
    if os.getenv(variable):
        state[key] = os.environ[variable]
state_path.write_text(json.dumps(state, indent=2) + "\n")
PY
fi
exec bash "$SCRIPT_DIR/run_qwen3_4b_16npu_trajectory_async_24k.sh" "${MODE[@]}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_dtype=bfloat16 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_ssm_cache_dtype=bfloat16 \
    "$@"
