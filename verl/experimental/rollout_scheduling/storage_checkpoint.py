# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""P4a: register the multi-task-infer StorageCheckpointEngine into verl.

The multi-task-infer plugin (``vllm_external_executor``) ships a
``StorageCheckpointEngine`` that loads model weights from persistent storage
(NFS / Mooncake Store, RDMA ~9 GB/s) and implements verl's
``CheckpointEngineWithCache``-compatible interface. Registering it into
verl's ``CheckpointEngineRegistry`` makes the storage backend selectable
like any built-in engine — usable even for a single job (cold-start replicas
pull weights from storage; the trainer does not need to be online pushing).

Reference: rollout_layered_scheduling_design.md §10.4 (integration point 1).

Two enabling paths:

1. Config-only (recommended, uses verl's existing hook)::

       actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module = \\
           verl.experimental.rollout_scheduling.storage_checkpoint
       actor_rollout_ref.rollout.checkpoint_engine.backend = storage

   ``CheckpointEngineManager`` imports ``custom_backend_module`` right
   before resolving the backend, which triggers the module-level
   registration below.

2. Explicit, from the training script::

       from verl.experimental.rollout_scheduling import register_storage_checkpoint_engine
       register_storage_checkpoint_engine()

Registration is idempotent; the explicit call raises a descriptive
``ImportError`` when the dependency is missing (fail-fast), while the
import-time path only warns — the subsequent backend lookup then fails
with the registry's own "unknown backend" error anyway.
"""

import logging

logger = logging.getLogger(__name__)

STORAGE_BACKEND_NAME = "storage"


def register_storage_checkpoint_engine(backend_name: str = STORAGE_BACKEND_NAME) -> type:
    """Import and register ``StorageCheckpointEngine`` under ``backend_name``.

    Raises:
        ImportError: if ``vllm_external_executor`` (the multi-task-infer
            plugin) is not installed in the current environment.
    """
    try:
        from vllm_external_executor import StorageCheckpointEngine
    except ImportError as e:
        raise ImportError(
            "register_storage_checkpoint_engine() requires the multi-task-infer plugin "
            "(vllm_external_executor). Install it from the multi-task-infer checkout, e.g. "
            "`pip install -e /path/to/vllm/multi-task-infer`."
        ) from e

    from verl.checkpoint_engine.base import CheckpointEngineRegistry

    CheckpointEngineRegistry.register(backend_name)(StorageCheckpointEngine)
    logger.info("[rollout_scheduling] registered StorageCheckpointEngine as backend %r", backend_name)
    return StorageCheckpointEngine


# Import-time registration for the ``custom_backend_module`` config path.
# Registration is idempotent (the registry just overwrites the entry), so a
# later explicit call is harmless. Missing dependency only warns here: the
# backend lookup fails fast on its own when "storage" was actually requested.
try:
    register_storage_checkpoint_engine()
except ImportError as _e:
    logger.warning(
        "[rollout_scheduling] StorageCheckpointEngine not registered at import time (%s). "
        'If you configured checkpoint_engine.backend="storage", install the dependency first.',
        _e,
    )
