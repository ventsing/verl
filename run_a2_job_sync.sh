#!/usr/bin/env bash
# Strict synchronous control on the same 16 train + 40 rollout + 8 eval NPUs.
# One batch at a time; all rollout replicas cross the same weight-version barrier.
set -euo pipefail
_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$_ROOT/examples/trajectory_async/a2_job_common.sh" sync "$@"
