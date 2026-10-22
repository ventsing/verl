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

The scripts reuse the installed Ascend/CANN Python environment. Model and data
paths remain configurable through `MODEL_PATH`, `DATA_ROOT` and `DATA_DIR`.
Python submission, preparation and probe utilities remain at the repository root.

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

Further details: [trajectory asynchronous training](../../verl/experimental/trajectory_async/README.md),
[mathematical reward and 32K reasoning](../../verl/experimental/trajectory_async/REWARD_STABILITY.md),
[synchronous comparison](../../verl/experimental/trajectory_async/SYNC_COMPARISON.md),
and [Mooncake setup](../../verl/experimental/trajectory_async/MOONCAKE_RELAY.md).
