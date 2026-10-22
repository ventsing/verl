#!/usr/bin/env bash
# Strict synchronous control with the same 8 trainer + 8 rollout placement.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export RUN_NAME=${RUN_NAME:-qwen35_9b_sync_reasoning_32k_$(date +%Y%m%d_%H%M%S)}
exec bash "$SCRIPT_DIR/run_qwen35_9b_16npu_laminar_reasoning.sh" "$@" \
    trainer.v1.laminar.synchronous=True
