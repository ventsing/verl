#!/usr/bin/env bash
# Single-node Ascend GRPO/GSPO smoke run using the existing NPU Python environment.
# Usage: bash examples/trajectory_async/run_qwen3_4b_16npu.sh [Hydra overrides...]
#        ALGORITHM=gspo bash examples/trajectory_async/run_qwen3_4b_16npu.sh
#        bash examples/trajectory_async/run_qwen3_4b_16npu.sh --prepare-only
#        bash examples/trajectory_async/run_qwen3_4b_16npu.sh --check-config
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-/opt/huawei/dataset/ccb_data/code/ventsing/downloaded_models/Qwen3-4B}
DATA_ROOT=${DATA_ROOT:-/opt/huawei/dataset/omni_training/datasets}
DATASET=${DATASET:-gsm8k}
DATA_DIR=${DATA_DIR:-$DATA_ROOT/$DATASET}
ALGORITHM=${ALGORITHM:-grpo}
NDEVICES=${NDEVICES:-16}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_N=${ROLLOUT_N:-4}
UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-1024}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
TOTAL_STEPS=${TOTAL_STEPS:-2}
VAL_SAMPLES=${VAL_SAMPLES:-64}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-128}
ENABLE_THINKING=${ENABLE_THINKING:-False}
RUN_NAME=${RUN_NAME:-qwen3_4b_${ALGORITHM}_16npu_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/outputs/$RUN_NAME}

MODE=run
case "${1:-}" in
    --help|-h)
        head -n 6 "$0"
        printf '\nDefaults: 16 NPUs, GRPO, 2 updates, 64 validation samples.\n'
        printf 'Set TOTAL_STEPS, VAL_SAMPLES, ALGORITHM=grpo|gspo, or pass Hydra overrides.\n'
        exit 0
        ;;
    --prepare-only) MODE=prepare; shift ;;
    --check-config) MODE=config; shift ;;
esac

POLICY=(actor_rollout_ref.actor.policy_loss.loss_mode=vanilla)
case "$ALGORITHM" in
    grpo) ;;
    gspo)
        POLICY=(
            actor_rollout_ref.actor.policy_loss.loss_mode=gspo
            actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
            actor_rollout_ref.actor.clip_ratio_low=0.0003
            actor_rollout_ref.actor.clip_ratio_high=0.0004
            actor_rollout_ref.actor.clip_ratio_c=10.0
            actor_rollout_ref.actor.use_kl_loss=False
        )
        ;;
    *) printf 'ALGORITHM must be grpo or gspo.\n' >&2; exit 2 ;;
esac

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_DISABLE_PROGRESS_BARS=0
export TOKENIZERS_PARALLELISM=false

prepare_data() {
    if [[ -s "$DATA_DIR/train.parquet" && -s "$DATA_DIR/test.parquet" ]]; then
        printf 'Using %s data: %s\n' "$DATASET" "$DATA_DIR"
        return
    fi
    if [[ "$DATASET" == deepscaler ]]; then
        mkdir -p "$DATA_DIR/raw"
        if [[ ! -s "$DATA_DIR/raw/deepscaler.json" ]]; then
            curl --fail --location --progress-bar --retry 3 --connect-timeout 20 --max-time 180 \
                https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29/deepscaler.json \
                --output "$DATA_DIR/raw/deepscaler.json.partial"
            mv -- "$DATA_DIR/raw/deepscaler.json.partial" "$DATA_DIR/raw/deepscaler.json"
        fi
        "$PYTHON_BIN" "$REPO_DIR/prepare_deepscaler.py" --data-dir "$DATA_DIR"
        return
    fi
    if [[ "$DATASET" != gsm8k ]]; then
        printf 'Unsupported DATASET: %s (choose gsm8k or deepscaler).\n' "$DATASET" >&2
        exit 2
    fi
    # Fetch OpenAI's original JSONL files without depending on Hub/Xet access.
    local raw_dir="$DATA_DIR/raw"
    local source_url="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
    local split
    mkdir -p "$raw_dir"
    for split in train test; do
        if [[ ! -s "$raw_dir/$split.jsonl" ]]; then
            printf 'Downloading GSM8K %s from OpenAI GitHub...\n' "$split"
            curl --fail --location --progress-bar --retry 3 --connect-timeout 20 --max-time 180 \
                "$source_url/$split.jsonl" --output "$raw_dir/$split.jsonl.partial"
            mv -- "$raw_dir/$split.jsonl.partial" "$raw_dir/$split.jsonl"
        fi
    done
    printf 'Preprocessing GSM8K into %s\n' "$DATA_DIR"
    "$PYTHON_BIN" - "$DATA_DIR" <<'PY'
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

data_dir = Path(sys.argv[1])
instruction = 'Let\'s think step by step and output the final answer after "####".'
for split in ["train", "test"]:
    records = []
    with (data_dir / "raw" / f"{split}.jsonl").open() as source:
        for idx, line in enumerate(source):
            example = json.loads(line)
            answer = re.search(r"#### (\-?[0-9\.\,]+)", example["answer"])
            if answer is None:
                raise ValueError(f"No final answer in GSM8K {split} row {idx}")
            records.append({
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": example["question"] + " " + instruction}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer.group(1).replace(",", "")},
                "extra_info": {"split": split, "index": idx, **example},
            })
    destination = data_dir / f"{split}.parquet"
    temporary = destination.with_suffix(".parquet.partial")
    pq.write_table(pa.Table.from_pylist(records), temporary)
    temporary.replace(destination)
    print(f"Saved {len(records)} rows: {destination}", flush=True)
PY
}

if [[ "$MODE" == prepare ]]; then
    prepare_data
    exit 0
fi

# The existing Ray cluster may have started before the shell's LD_PRELOAD was
# set. Pass it explicitly to Ray workers and their vLLM inspection subprocesses.
# Without jemalloc this machine's vLLM/Ascend native imports abort in malloc.
JEMALLOC_LIBRARY=${JEMALLOC_LIBRARY:-/usr/lib/aarch64-linux-gnu/libjemalloc.so.2}
if [[ ! -r "$JEMALLOC_LIBRARY" ]]; then
    printf 'Missing jemalloc library: %s (override JEMALLOC_LIBRARY if needed).\n' "$JEMALLOC_LIBRARY" >&2
    exit 1
fi
if [[ ":${LD_PRELOAD:-}:" != *":${JEMALLOC_LIBRARY}:"* ]]; then
    LD_PRELOAD="$JEMALLOC_LIBRARY${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export LD_PRELOAD
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    "data.train_files=$DATA_DIR/train.parquet"
    "data.val_files=$DATA_DIR/test.parquet"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.train_max_samples=$TRAIN_SAMPLES"
    "data.val_max_samples=$VAL_SAMPLES"
    data.val_batch_size=64
    data.seed=42
    data.dataloader_num_workers=2
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=error
    "+data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=1e-6
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    # Keep optimizer parameters in FP32; FSDP mixed precision computes in BF16.
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=$UPDATE_WEIGHTS_BUCKET_MB"
    actor_rollout_ref.rollout.mode=async
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_MEM_UTIL:-0.35}"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.enforce_eager=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    trainer.device=npu
    trainer.nnodes=1
    "trainer.n_gpus_per_node=$NDEVICES"
    trainer.logger='[console]'
    trainer.project_name=qwen3_4b_smoke
    "trainer.experiment_name=$RUN_NAME"
    "trainer.default_local_dir=$OUTPUT_DIR/checkpoints"
    "trainer.validation_data_dir=$OUTPUT_DIR/validation"
    trainer.log_val_generations=2
    trainer.val_before_train=True
    trainer.val_only=False
    trainer.test_freq=2
    trainer.save_freq=-1
    trainer.resume_mode=disable
    "trainer.total_epochs=${TOTAL_EPOCHS:-1}"
    "trainer.total_training_steps=$TOTAL_STEPS"
    ray_kwargs.ray_init.runtime_env.py_executable=null
)

# Existing Ray workers do not inherit the launching shell's CANN setup. Their
# vLLM subprocesses need the same Python modules, shared libraries, and toolkit.
for runtime_var in LD_PRELOAD PYTHONPATH LD_LIBRARY_PATH PATH \
    ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME VLLM_LOGGING_LEVEL; do
    if [[ -v "$runtime_var" ]]; then
        ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.$runtime_var='${!runtime_var}'")
    fi
done

if [[ "$MODE" == config ]]; then
    exec "$PYTHON_BIN" -m verl.trainer.main_ppo "${ARGS[@]}" "${POLICY[@]}" "$@" --cfg job --resolve
fi

# Preserve the installed torch_npu/vllm-ascend environment; do not sync CUDA extras.
"$PYTHON_BIN" - "$MODEL_PATH" "$NDEVICES" "$UPDATE_WEIGHTS_BUCKET_MB" <<'PY'
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

model_dir = Path(sys.argv[1])
required = ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"]
for name in required:
    if not (model_dir / name).is_file():
        raise SystemExit(f"Missing model file: {model_dir / name}")
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
largest_tensor = (0, "")
for name in set(index["weight_map"].values()):
    shard = model_dir / name
    if not shard.is_file() or shard.stat().st_size == 0:
        raise SystemExit(f"Missing or empty model shard: {shard}")
    # Read only the safetensors header, not the model's tensor payloads.
    with shard.open("rb") as source:
        header_length = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_length))
    for tensor_name, metadata in header.items():
        if tensor_name != "__metadata__":
            start, end = metadata["data_offsets"]
            largest_tensor = max(largest_tensor, (end - start, tensor_name))
bucket_bytes = int(sys.argv[3]) * 1024**2
if largest_tensor[0] > bucket_bytes:
    raise SystemExit(
        f"Weight transfer bucket ({sys.argv[3]} MiB) is too small for {largest_tensor[1]} "
        f"({largest_tensor[0] / 1024**2:.1f} MiB). Increase UPDATE_WEIGHTS_BUCKET_MB."
    )
print(
    f"Largest weight: {largest_tensor[1]} ({largest_tensor[0] / 1024**2:.1f} MiB); "
    f"transfer bucket: {sys.argv[3]} MiB.", flush=True,
)
for name in ["vllm", "vllm_ascend", "ray", "datasets", "hydra", "acl"]:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"Missing dependency in this Python environment: {name}")
if address := os.getenv("RAY_ADDRESS"):
    # Ray Jobs drivers have no assigned NPUs. Query the existing cluster;
    # worker placement will assign device visibility independently.
    import ray

    ray.init(address=address, log_to_driver=False)
    try:
        count = int(ray.cluster_resources().get("NPU", 0))
    finally:
        ray.shutdown()
else:
    count = torch.npu.device_count()
if count < int(sys.argv[2]):
    raise SystemExit(f"Only {count} NPUs available; requested {sys.argv[2]}")
print(f"Preflight passed: {count} NPUs available; model shards present.", flush=True)
PY

prepare_data
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-1}

mkdir -p "$OUTPUT_DIR"
printf 'Starting %s on %s NPUs, %s updates; log: %s/train.log\n' \
    "$ALGORITHM" "$NDEVICES" "$TOTAL_STEPS" "$OUTPUT_DIR"
"$PYTHON_BIN" -m verl.trainer.main_ppo "${ARGS[@]}" "${POLICY[@]}" "$@" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
