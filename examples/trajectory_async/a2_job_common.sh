#!/usr/bin/env bash
# Shared A2 job bootstrap. Called by the two repository-root job entrypoints.
set -euo pipefail
_MODE=${1:?Expected trajectory_async or sync}
shift
case "$_MODE" in
    trajectory_async) _SYNCHRONOUS=False; _ADAPTIVE=True ;;
    sync) _SYNCHRONOUS=True; _ADAPTIVE=False ;;
    *) echo "Unknown execution mode: $_MODE" >&2; exit 2 ;;
esac
_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_REPO_DIR="$(cd -- "$_SCRIPT_DIR/../.." && pwd)"
cd "$_REPO_DIR"
_CHECK_CONFIG=()
_TRAIN_ARGS=()
for _arg in "$@"; do
    case "$_arg" in
        --help|-h)
            cat <<'HELP'
Run the same entrypoint on every A2 job node. VC_TASK_INDEX selects head/worker.
Defaults: Qwen3-8B, 64 NPUs = 16 train + 40 rollout + 8 evaluation, 200 steps.
GBS: 64 prompt groups x 8 responses; prompt 2048, response 32768 tokens.

  bash run_a2_job_trajectory_async.sh [Hydra overrides...]
  bash run_a2_job_sync.sh [Hydra overrides...]
  bash run_a2_job_trajectory_async.sh --check-config

Cluster: VC_TASK_INDEX, VC_WORKER_HOSTS (comma-separated), VC_WORKER_NUM.
Manual: VC_TASK_INDEX, MASTER_ADDR, RAY_NODE_IP, TOTAL_NODES.
Optional: RAY_PORT=6925, RAY_DASHBOARD_PORT=8960, NPUS_PER_NODE=8,
  TRAINER_NNODES=2, ROLLOUT_NNODES=5, EVAL_NNODES=1, SOCKET_IFNAME=eth0.
Training: MODEL_PATH, DATA_DIR, RUN_NAME, OUTPUT_ROOT, TOTAL_STEPS,
  TRAIN_BATCH_SIZE, ROLLOUT_N, ROLLOUT_TP, EVAL_TP, EVAL_NUM_WORKERS,
  BUFFER_MIN_BATCHES=3, BUFFER_MAX_BATCHES=5, KV_HORIZON_SECONDS=90.
Optional environment exports file: A2_ENV_FILE.
--check-config resolves configuration only; it does not start/connect to Ray.
Existing Ray processes are reused. Scripts never stop/restart a cluster.
HELP
            exit 0 ;;
        --check-config) _CHECK_CONFIG=(--check-config) ;;
        --cfg=*|--package=*|--config-path=*|--config-name=*|--config-dir=*|--experimental-rerun=*|--info=*)
            _TRAIN_ARGS+=("$_arg") ;;
        --*=*) ;;
        *) _TRAIN_ARGS+=("$_arg") ;;
    esac
done

if [[ -n ${A2_ENV_FILE:-} ]]; then
    # This optional file supplies exports; it must not start/stop Ray.
    source "$A2_ENV_FILE"
fi
export PYTHON_BIN=${PYTHON_BIN:-${PYTHON:-python3}}
command -v "$PYTHON_BIN" >/dev/null
# Preserve the image's native Ascend/vLLM installation and compiled custom ops.
if [[ ! -f ${ASCEND_HOME_PATH:-/nonexistent}/set_env.sh ]]; then
    for _candidate in /usr/local/Ascend/cann-9.1.1 /usr/local/Ascend/cann /usr/local/Ascend/ascend-toolkit/latest; do
        if [[ -f "$_candidate/set_env.sh" ]]; then export ASCEND_HOME_PATH=$_candidate; break; fi
    done
fi
for _env_file in "${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/set_env.sh" /usr/local/Ascend/nnal/atb/set_env.sh; do
    if [[ -f "$_env_file" ]]; then
        set +eu
        source "$_env_file"
        _source_status=$?
        set -euo pipefail
        if (( _source_status != 0 )); then
            echo "FATAL: could not source $_env_file (exit $_source_status)" >&2
            exit "$_source_status"
        fi
    fi
done
export CPATH="/usr/include${CPATH:+:$CPATH}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/cache/.torch_extension}
export PYTHONPATH="$_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset VERL_USE_EXTERNAL_MODULES PYTORCH_NPU_ALLOC_CONF
export VERL_USE_EXTERNAL_PLUGINS=none
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-7200}
export TORCH_DIST_TIMEOUT=${TORCH_DIST_TIMEOUT:-3600}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64000}
export SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME GLOO_SOCKET_IFNAME=$SOCKET_IFNAME TP_SOCKET_IFNAME=$SOCKET_IFNAME
export RAY_gcs_server_rpc_server_thread_num=${RAY_gcs_server_rpc_server_thread_num:-32}
export RAY_gcs_server_request_timeout_seconds=${RAY_gcs_server_request_timeout_seconds:-1200}
export RAY_timeout_ms=${RAY_timeout_ms:-120000}
export RAY_worker_register_timeout_seconds=${RAY_worker_register_timeout_seconds:-1200}
export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-60}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-30000}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 RAY_USAGE_STATS_ENABLED=0
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-1} PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export WANDB_MODE=${WANDB_MODE:-offline}
if [[ ${LD_PRELOAD:-} != *jemalloc* ]]; then
    _JEMALLOC=${A2_JEMALLOC_LIBRARY:-}
    if [[ -z "$_JEMALLOC" ]]; then
        for _candidate in /usr/lib/"$(uname -m)"-linux-gnu/libjemalloc.so.2 /usr/lib64/libjemalloc.so.2 /usr/lib/libjemalloc.so.2; do
            if [[ -r "$_candidate" ]]; then _JEMALLOC=$_candidate; break; fi
        done
    fi
    if [[ ! -r "$_JEMALLOC" ]]; then
        echo "FATAL: jemalloc is required; set A2_JEMALLOC_LIBRARY to its library path" >&2
        exit 1
    fi
    export LD_PRELOAD="$_JEMALLOC${LD_PRELOAD:+:$LD_PRELOAD}"
fi

export NPUS_PER_NODE=${NPUS_PER_NODE:-8}
export TRAINER_NNODES=${TRAINER_NNODES:-2} ROLLOUT_NNODES=${ROLLOUT_NNODES:-5} EVAL_NNODES=${EVAL_NNODES:-1}
for _name in NPUS_PER_NODE TRAINER_NNODES ROLLOUT_NNODES EVAL_NNODES; do
    if [[ ! ${!_name} =~ ^[1-9][0-9]*$ ]]; then echo "FATAL: $_name must be a positive integer" >&2; exit 2; fi
done
_REQUIRED_NODES=$((TRAINER_NNODES + ROLLOUT_NNODES + EVAL_NNODES))
export NDEVICES=$((_REQUIRED_NODES * NPUS_PER_NODE))
export MODEL_PATH=${MODEL_PATH:-/opt/huawei/dataset/multi_verl_share_resource/downloaded_models/Qwen3-8B}
export DATASET=deepscaler
export DATA_DIR=${DATA_DIR:-/opt/huawei/dataset/multi_verl_share_resource/datasets/trajectory_async_native}
export RUN_NAME=${RUN_NAME:-${EXPERIMENT_NAME:-qwen3_8b_${_MODE}_$(date -u +%Y%m%dT%H%M%SZ)}}
export OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT:-$_REPO_DIR/outputs}/$RUN_NAME}
export QWEN_OUTPUT_DIR=$OUTPUT_DIR VERL_FILE_LOGGER_PATH=$OUTPUT_DIR/metrics.jsonl
export TOTAL_STEPS=${TOTAL_STEPS:-200} TRAIN_SAMPLES=${TRAIN_SAMPLES:--1} VAL_SAMPLES=${VAL_SAMPLES:--1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
export ROLLOUT_N=${ROLLOUT_N:-8} ROLLOUT_TP=${ROLLOUT_TP:-2}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768} ENABLE_THINKING=True
export ROLLOUT_MEM_UTIL=${ROLLOUT_MEM_UTIL:-0.7} UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-2048}
_EVAL_TP=${EVAL_TP:-$ROLLOUT_TP}
for _value in "$ROLLOUT_TP" "$_EVAL_TP" "$TRAIN_BATCH_SIZE"; do
    if [[ ! $_value =~ ^[1-9][0-9]*$ ]]; then echo "FATAL: TP and batch sizes must be positive integers" >&2; exit 2; fi
done
if (( NPUS_PER_NODE % ROLLOUT_TP || NPUS_PER_NODE % _EVAL_TP )); then
    echo "FATAL: rollout/evaluation TP must divide NPUS_PER_NODE" >&2; exit 2
fi
_BUFFER_MIN=${BUFFER_MIN_BATCHES:-3}
_BUFFER_MAX=${BUFFER_MAX_BATCHES:-5}
_BUFFER_INITIAL=${BUFFER_INITIAL_BATCHES:-$_BUFFER_MIN}
_BUFFER_CAPACITY=$("$PYTHON_BIN" - "$TRAIN_BATCH_SIZE" "$_BUFFER_MIN" "$_BUFFER_MAX" "$_BUFFER_INITIAL" <<'PY'
import math
import sys
batch, minimum, maximum, initial = map(float, sys.argv[1:])
if not all(math.isfinite(x) for x in (batch, minimum, maximum, initial)) or not 1 <= minimum <= initial <= maximum:
    raise SystemExit('FATAL: buffer multipliers require 1 <= min <= initial <= max')
print(math.floor(batch * maximum))
PY
)
_WARMUP=1
if [[ $_MODE == sync ]]; then _BUFFER_CAPACITY=$TRAIN_BATCH_SIZE; _WARMUP=0; fi
_HISTORY_ARGS=()
if [[ -n ${RESPONSE_HISTORY_PATH:-} ]]; then
    _HISTORY_ARGS+=("trainer.v1.laminar.online_capacity.response_history_path=$RESPONSE_HISTORY_PATH")
fi

launch_training() {
    exec bash "$_SCRIPT_DIR/run_qwen3_4b_16npu_trajectory_async.sh" "${_CHECK_CONFIG[@]}" \
    trainer.v1.trainer_mode=laminar \
    "trainer.nnodes=$TRAINER_NNODES" "trainer.n_gpus_per_node=$NPUS_PER_NODE" \
    "actor_rollout_ref.rollout.nnodes=$ROLLOUT_NNODES" "actor_rollout_ref.rollout.n_gpus_per_node=$NPUS_PER_NODE" \
    actor_rollout_ref.rollout.checkpoint_engine.backend=mooncake_relay \
    actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=verl.experimental.trajectory_async.mooncake_relay \
    trainer.v1.laminar.mooncake.protocol=ascend \
    trainer.v1.laminar.mooncake.acquire_timeout_s=1800 \
    trainer.v1.laminar.weight_budget.max_versions=8 \
    trainer.v1.laminar.weight_budget.max_bytes=274877906944 \
    trainer.v1.laminar.weight_budget.capacity_timeout_s=3600 \
    trainer.v1.laminar.batch_per_replica=64 \
    "trainer.v1.laminar.prompt_buffer_groups=$_BUFFER_CAPACITY" \
    trainer.v1.laminar.online_capacity.enabled=True \
    trainer.v1.laminar.online_capacity.token_budget_admission=True \
    trainer.v1.laminar.online_capacity.rolling_kv_enabled=True \
    "trainer.v1.laminar.online_capacity.rolling_kv_horizon_s=${KV_HORIZON_SECONDS:-90.0}" \
    "trainer.v1.laminar.repack_interval_s=${REPACK_INTERVAL_SECONDS:-2.0}" \
    trainer.v1.laminar.online_capacity.kv_usage_limit=0.95 \
    trainer.v1.laminar.roofline.enabled=True \
    trainer.v1.laminar.roofline.continuous_sampling=True \
    trainer.v1.laminar.run_migration_probe=False \
    trainer.v1.laminar.event_dir="$OUTPUT_DIR/trajectories" \
    "trainer.v1.separate_async.num_warmup_batches=$_WARMUP" \
    trainer.v1.sampler.max_off_policy_strategy=none \
    +trainer.v1.sampler.sampler_kwargs.selection_strategy=mixed_ready \
    +trainer.v1.sampler.sampler_kwargs.selection_seed=42 \
    +trainer.v1.sampler.sampler_kwargs.trajectory_log_dir="$OUTPUT_DIR/trajectories" \
    "+trainer.v1.sampler.sampler_kwargs.response_limit=$MAX_RESPONSE_LENGTH" \
    trainer.v1.laminar.evaluation.enabled=True \
    "trainer.v1.laminar.evaluation.nnodes=$EVAL_NNODES" \
    "trainer.v1.laminar.evaluation.n_gpus_per_node=$NPUS_PER_NODE" \
    "trainer.v1.laminar.evaluation.tensor_model_parallel_size=$_EVAL_TP" \
    "trainer.v1.laminar.evaluation.num_workers=${EVAL_NUM_WORKERS:-$((EVAL_NNODES * NPUS_PER_NODE / _EVAL_TP))}" \
    trainer.v1.laminar.evaluation.max_num_seqs=128 \
    trainer.v1.laminar.evaluation.max_inflight_per_replica=64 \
    trainer.v1.laminar.evaluation.max_concurrent_versions=3 \
    trainer.v1.laminar.evaluation.overflow_policy=block \
    trainer.v1.laminar.evaluation.timeout_s=14400 \
    trainer.test_freq=10 trainer.val_before_train=True trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.project_name=qwen3_8b_trajectory_async \
    trainer.logger='[console,file]' \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.memory_fraction=0.9 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.logprobs_mode=processed_logprobs \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
    algorithm.rollout_correction.bypass_mode=True \
    data.dataloader_num_workers=0 data.gen_batch_size=1 \
    data.val_files="['$DATA_DIR/test.parquet']" \
    trainer.v1.laminar.evaluation.final_val_files="['$DATA_DIR/math_lighteval_test.parquet']" \
    reward.custom_reward_function.path=pkg://verl.experimental.trajectory_async.reasoning_math_reward \
    reward.custom_reward_function.name=compute_score \
    "+ray_kwargs.ray_init.address=${RAY_ADDRESS:-auto}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=''" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_PLUGINS=none \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$VERL_FILE_LOGGER_PATH'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE='1'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE='1'" \
    "trainer.v1.laminar.synchronous=$_SYNCHRONOUS" \
    "trainer.v1.laminar.adaptive_buffer.enabled=$_ADAPTIVE" \
    "trainer.v1.laminar.adaptive_buffer.min_batches=$_BUFFER_MIN" \
    "trainer.v1.laminar.adaptive_buffer.max_batches=$_BUFFER_MAX" \
    "trainer.v1.laminar.adaptive_buffer.initial_batches=$_BUFFER_INITIAL" \
    trainer.v1.laminar.adaptive_buffer.step_groups=16 \
    trainer.v1.laminar.adaptive_buffer.refill_groups=16 \
    "${_HISTORY_ARGS[@]}" \
    "${_TRAIN_ARGS[@]}"
}

# Config-only mode deliberately skips host resolution, readiness checks and Ray.
if (( ${#_CHECK_CONFIG[@]} )); then launch_training; fi
command -v ray >/dev/null
command -v pgrep >/dev/null
_RANK=${VC_TASK_INDEX:-0}
IFS=',' read -r -a _HOSTS <<< "${VC_WORKER_HOSTS:-}"
_TOTAL_NODES=${VC_WORKER_NUM:-${TOTAL_NODES:-$_REQUIRED_NODES}}
_TIMEOUT=${RAY_CLUSTER_WAIT_TIMEOUT:-600}
_INTERVAL=${RAY_CLUSTER_POLL_INTERVAL:-5}
if [[ ! $_RANK =~ ^[0-9]+$ || ! $_TOTAL_NODES =~ ^[1-9][0-9]*$ ]] || (( _TOTAL_NODES != _REQUIRED_NODES || _RANK >= _TOTAL_NODES )); then
    echo "FATAL: rank/node count invalid; placement requires $_REQUIRED_NODES nodes, received rank=$_RANK nodes=$_TOTAL_NODES" >&2
    exit 2
fi
_MASTER_HOST=${_HOSTS[0]:-${MASTER_ADDR:-}}
_NODE_HOST=${RAY_NODE_IP:-${_HOSTS[$_RANK]:-}}
if [[ -z $_MASTER_HOST || -z $_NODE_HOST ]]; then
    echo "FATAL: set VC_WORKER_HOSTS, or MASTER_ADDR + RAY_NODE_IP on every node" >&2
    exit 2
fi
resolve_ipv4() {
    "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys
print(socket.gethostbyname(sys.argv[1]))
PY
}
_MASTER_IP=$(resolve_ipv4 "$_MASTER_HOST")
_NODE_IP=$(resolve_ipv4 "$_NODE_HOST")
if (( _RANK == 0 )) && [[ $_MASTER_IP != "$_NODE_IP" ]]; then
    echo "FATAL: head IP $_NODE_IP differs from master $_MASTER_IP" >&2; exit 2
fi
export RAY_PORT=${RAY_PORT:-6925} RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8960}
export RAY_ADDRESS="$_MASTER_IP:$RAY_PORT"
_LOG_DIR=${A2_JOB_LOG_DIR:-${OUTPUT_ROOT:-$_REPO_DIR/outputs}/cluster/${VC_JOB_ID:-a2_$_MODE}}
mkdir -p "$_LOG_DIR"
exec > >(tee -a "$_LOG_DIR/rank_$_RANK.log") 2>&1
ulimit -s unlimited

wait_for_cluster() {
    "$PYTHON_BIN" - "$RAY_ADDRESS" "$_TOTAL_NODES" "$NPUS_PER_NODE" "$_NODE_IP" "$_TIMEOUT" "$_INTERVAL" "$1" <<'PY'
import sys
import time
import ray
address, expected, npus, local_ip, timeout, interval, mode = sys.argv[1:]
expected, npus = int(expected), int(npus)
deadline = time.monotonic() + float(timeout)
ray.init(address=address, log_to_driver=False)
try:
    previous = None
    while True:
        nodes = [n for n in ray.nodes() if n.get('Alive')]
        local = any(n.get('NodeManagerAddress') == local_ip for n in nodes)
        ready = [n for n in nodes if n.get('Resources', {}).get('NPU', 0) >= npus]
        available = ray.available_resources().get('NPU', 0)
        state = (len(nodes), len(ready), available, local)
        if state != previous:
            print(f'[a2-job] alive={len(nodes)}/{expected} NPU-ready={len(ready)}/{expected} '
                  f'available_NPU={available}/{expected*npus} local_joined={local}', flush=True)
            previous = state
        if local and (mode == 'local' or (len(ready) >= expected and available >= expected*npus)):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Timed out waiting for expected nodes/free NPUs; training was not started')
        time.sleep(float(interval))
finally:
    ray.shutdown()  # Disconnect the probe only; preserve the job cluster.
PY
}

if (( _RANK == 0 )); then
    if pgrep -x raylet >/dev/null; then
        echo "[a2-$_MODE] Reusing local Ray cluster at $RAY_ADDRESS"
        ray status --address="$RAY_ADDRESS" >/dev/null
    else
        ray start --head --node-ip-address="$_NODE_IP" --port="$RAY_PORT" \
            --dashboard-host=0.0.0.0 --dashboard-port="$RAY_DASHBOARD_PORT" \
            --resources="{\"NPU\": $NPUS_PER_NODE}"
    fi
    wait_for_cluster all
    echo "[a2-$_MODE] Starting $RUN_NAME: $((TRAINER_NNODES*NPUS_PER_NODE)) train + $((ROLLOUT_NNODES*NPUS_PER_NODE)) rollout + $((EVAL_NNODES*NPUS_PER_NODE)) evaluation NPUs"
    launch_training
else
    _DEADLINE=$((SECONDS + _TIMEOUT))
    until ray status --address="$RAY_ADDRESS" >/dev/null 2>&1; do
        if (( SECONDS >= _DEADLINE )); then echo "FATAL: timed out waiting for $RAY_ADDRESS" >&2; exit 1; fi
        sleep "$_INTERVAL"
    done
    if pgrep -x raylet >/dev/null; then
        wait_for_cluster local
        while pgrep -x raylet >/dev/null; do sleep 30; done
        echo "FATAL: worker raylet exited" >&2; exit 1
    else
        exec ray start --address="$RAY_ADDRESS" --node-ip-address="$_NODE_IP" \
            --resources="{\"NPU\": $NPUS_PER_NODE}" --block
    fi
fi
