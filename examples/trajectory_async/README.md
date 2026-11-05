# Ascend trajectory asynchronous training examples

Run these launchers from the repository root, or use their absolute paths from
another directory. They resolve the repository root themselves. Default logs
and outputs stay under the repository's `outputs/` directory; `QWEN_OUTPUT_DIR`
can select another location for the trajectory launchers.

| Launcher | Purpose |
| --- | --- |
| `run_qwen3_4b_16npu.sh` | Qwen3-4B GRPO/GSPO smoke run on 16 NPUs |
| `run_qwen3_4b_16npu_trajectory_async.sh` | Qwen3-4B separate asynchronous training, 8 trainer + 8 rollout NPUs |
| `run_qwen3_4b_16npu_trajectory_async_24k.sh` | DeepScaleR and 24K thinking for Qwen3-4B |
| `run_qwen35_9b_16npu_trajectory_async_24k.sh` | Qwen3.5-9B model/kernel preset with 24K thinking |
| `run_qwen35_9b_16npu_laminar_24k.sh` | Laminar trajectory scheduling with the CPU relay |
| `run_qwen35_9b_laminar_mooncake.sh` | Laminar with the hierarchical Mooncake relay |
| `run_qwen35_9b_16npu_laminar_reasoning.sh` | 32K thinking, corrected mathematical reward and fixed validation |
| `run_qwen35_9b_16npu_laminar_sync_reasoning.sh` | Strict synchronous control on the same 8+8 NPU placement |
| `run_qwen35_9b_reward_diagnosis.sh` | 32-NPU, 50-update, 32K diagnosis with fixed validation and file metrics on an existing Ray cluster; external plugins disabled |

The reward diagnostic launcher keeps FP32 actor master parameters and sets
`actor_rollout_ref.{actor,ref}.fsdp_config.memory_fraction=0.9`, leaving device
memory for CANN allocations outside the PyTorch cache. It enables activation
offload and saves every five updates, retaining two checkpoints. The engine
option defaults to `None` for other launchers. Set `FSDP_MEMORY_FRACTION` to
change the diagnostic limit. Memory-probe success does not establish reward
improvement; see `verl/experimental/trajectory_async/REWARD_STABILITY.md`.

The reward diagnosis launcher requests `VLLM_BATCH_INVARIANT=1` in the Ray
runtime environment and keeps `full_determinism=false`, preserving stochastic
training while using the Ascend backend's invariant numerical path. Its worker
compatibility guard preserves native sums for unsupported boolean and nonlast-axis
reductions. Use fixed held-out validation to assess learning: completion-ordered
training reward can start high because shorter correct answers arrive first.

The Qwen3.5 preset uses FP32 SSM state cache, matching the model's
`text_config.mamba_ssm_dtype`, and BF16 convolution cache. Earlier runs forced
SSM state to BF16. This configuration correction still requires full-model
probability alignment and reward-learning verification; it is not a convergence
claim.

The scripts reuse the installed Ascend/CANN Python environment. Model and data
paths remain configurable through `MODEL_PATH`, `DATA_ROOT` and `DATA_DIR`.
Python submission, preparation and probe utilities remain at the repository root.

The Qwen3.5 model default is
`/opt/huawei/dataset/omni_training/downloaded_models/Qwen3.5-9B`; the dataset root
defaults to `/opt/huawei/dataset/omni_training/datasets`. The reasoning launcher
prepares the audited copy under `deepscaler/reasoning` unless `DATA_DIR` is set.
The restored DeepScaleR split holds out 512 exact, deduplicated questions;
the reasoning preset evaluates a fixed 128-question subset. This reconstructed
split does not reproduce the historical environment's missing split files.

The diagnosis launcher defaults to two nodes, with 16 trainer NPUs and 16 rollout
NPUs in total. `TRAINER_NNODES` and `ROLLOUT_NNODES` can override that placement.
Its global gradient batch remains 16 prompts × 4 responses. The launcher leaves
staleness unrestricted and retains the original
64-group prompt buffer. It records actual behavior versions, reward computation
time, producer-ready timestamps, and trainer selection timestamps. It unsets
`VERL_USE_EXTERNAL_MODULES`, disables external plugin discovery, and also clears
these settings in Ray's per-job environment. It attaches to `RAY_ADDRESS` (or
`auto`) without starting or restarting a cluster. Its isolated CPU Mooncake
dependency must already be deployed on every relay node; configure
`MOONCAKE_RELAY_PYTHON_PATH` and `MOONCAKE_RELAY_LIBRARY_PATH` when using a
different location. The installed NPU Mooncake package remains separate.

Run the asynchronous and strict synchronous controls **sequentially** on this
cluster. The installed TransferQueue uses a fixed controller namespace; a second
concurrent training job would reuse its queue. The synchronous control uses the
same launcher with `trainer.v1.laminar.synchronous=True`. Wait for the first job
and its queue to exit before starting the control; no Ray cluster restart is
required. Resolved configs should differ only in that flag and experiment paths.

```bash
# Resolve the CPU-relay configuration without starting training workers.
bash examples/trajectory_async/run_qwen35_9b_16npu_laminar_24k.sh --check-config

# Qwen3-4B asynchronous GRPO, or GSPO with ALGORITHM=gspo.
bash examples/trajectory_async/run_qwen3_4b_16npu_trajectory_async.sh

# Prepare an audited data copy for the mathematical reasoning reward.
python3 prepare_reasoning_math_data.py \
  --source /opt/huawei/dataset/ccb_data/code/package_install/datasets_for_train/deepscaler \
  --output /tmp/verl-reasoning-data

# Set the host's Mooncake Python/library environment first. Use TCP on hosts
# without RDMA; RDMA hosts can instead select rdma and their network device.
MOONCAKE_RELAY_PROTOCOL=tcp DATA_DIR=/tmp/verl-reasoning-data \
  bash examples/trajectory_async/run_qwen35_9b_16npu_laminar_reasoning.sh

# Submit the strict synchronous control to an existing Ray cluster.
MOONCAKE_RELAY_PROTOCOL=tcp DATA_DIR=/tmp/verl-reasoning-data \
  python3 submit_qwen35_laminar.py --reasoning --synchronous \
    --relay-backend mooncake --output-root /tmp/verl-laminar-runs
```

For the opt-in Ascend prefill path that shares decode primitives, set
`VERL_ASCEND_CONSISTENT_PREFILL=1` in the Ray worker runtime environment and
`VERL_ASCEND_GDN_PREFILL_FP32=0`. This path requires batch-invariant execution,
BF16 rollout weights, FP32 SSM state, eager execution, and PCP/DCP size 1.
Sliding-window attention, attention sinks, and speculative decoding are unsupported.
It uses native recurrent updates in windows of 16 tokens and cached one-query
attention in batches of 64, retaining each query's causal context length.
The native model constructor and KV-cache writer are preserved. The default is off.
Changing this mode changes the numerical serving path; compare training runs
from the same checkpoint and evaluate both with identical fixed-set settings.

Further details: [trajectory asynchronous training](../../verl/experimental/trajectory_async/README.md),
[mathematical reward and 32K reasoning](../../verl/experimental/trajectory_async/REWARD_STABILITY.md),
[synchronous comparison](../../verl/experimental/trajectory_async/SYNC_COMPARISON.md),
and [Mooncake setup](../../verl/experimental/trajectory_async/MOONCAKE_RELAY.md).
