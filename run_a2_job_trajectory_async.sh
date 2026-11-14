#!/usr/bin/env bash
# All job nodes run this entrypoint; VC_TASK_INDEX selects head/worker.
# Qwen3-8B trajectory async: 16 train + 40 rollout + 8 evaluation NPUs.
set -euo pipefail
_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$_ROOT/examples/trajectory_async/a2_job_common.sh" trajectory_async "$@"
