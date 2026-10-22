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
"""verl rollout scheduling plugin (L1 upgrade of the layered design).

Design: ``rollout_layered_scheduling_design.md`` (repo root).

This package is a *plugin*: all behaviour lives here, and the stock verl
modules only gained small default-off sockets (``llm_server.py``):

- ``GlobalRequestLoadBalancer.acquire_server/release_server`` accept (and
  ignore) the optional hint kwargs — making every balancer/client
  combination interop-safe;
- ``LLMServerManager.__init__`` resolves the balancer class from
  ``VERL_ROLLOUT_LOAD_BALANCER_CLS`` (FQN) when no explicit class is passed;
- ``LLMServerManager.get_client`` consults registered client-class upgraders.

Enabling (single env var — it imports this package, which registers the
client upgrader, and selects the token-weighted balancer)::

    export VERL_ROLLOUT_LOAD_BALANCER_CLS=\\
verl.experimental.rollout_scheduling.load_balancer.TokenWeightedLoadBalancer

Unset → behaviour is byte-identical to stock verl.

Components:

- :class:`TokenWeightedLoadBalancer` — L1 routing: token-weighted scoring,
  sticky sessions, drain set, bias, rebase sockets (design §3);
- :class:`HintedLLMServerClient` / :class:`HintedFullyAsyncLLMServerClient`
  — clients that pass ``kv_hint`` on acquire and ``request_id`` on release
  (design §3.3); auto-installed via the client-class upgrader;
- :class:`PooledLLMServerManager` / :class:`PooledvLLMReplica` /
  :class:`PooledvLLMHttpServer` — P4b: rollout replicas leased from a
  multi-task-infer actor pool (design §10.4, integration point 2); selected
  via ``rollout.llm_server_manager_class`` + ``rollout.shared_pool.*``;
- :func:`register_storage_checkpoint_engine` — P4a: register the
  multi-task-infer ``StorageCheckpointEngine`` into verl's checkpoint engine
  registry (design §10.4); opt-in, requires the ``vllm_external_executor``
  package.
"""

from verl.workers.rollout.llm_server import register_client_class_upgrader

from .client import HintedFullyAsyncLLMServerClient, HintedLLMServerClient, upgrade_client_class
from .load_balancer import TokenWeightedLoadBalancer
from .pooled_replica import (
    PooledLLMServerManager,
    PooledvLLMHttpServer,
    PooledvLLMReplica,
    check_pooled_prerequisites,
)
from .storage_checkpoint import register_storage_checkpoint_engine

# Importing this package installs the client upgrade. This is safe even when
# the balancer stays stock: the base balancer accepts-and-ignores the hint
# kwargs, so hinted clients degrade to exactly the current behaviour.
register_client_class_upgrader(upgrade_client_class)

__all__ = [
    "TokenWeightedLoadBalancer",
    "HintedLLMServerClient",
    "HintedFullyAsyncLLMServerClient",
    "upgrade_client_class",
    "register_storage_checkpoint_engine",
    "PooledLLMServerManager",
    "PooledvLLMReplica",
    "PooledvLLMHttpServer",
    "check_pooled_prerequisites",
]
