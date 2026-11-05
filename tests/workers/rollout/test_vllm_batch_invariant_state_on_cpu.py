"""Actual worker/server source contracts, without starting an inference engine."""

import __future__

import ast
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

REPO = Path(__file__).resolve().parents[3]
WORKER = REPO / "verl/workers/rollout/vllm_rollout/utils.py"
SERVER = REPO / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"


def source_method(path, class_name, name, scope, keep_class=False):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(
        node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    assert not node.decorator_list
    if keep_class:
        cls.body = [node]
        node = cls
    code = compile(
        ast.Module(body=[node], type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag
    )
    exec(code, scope)
    return scope[class_name if keep_class else name]


def test_server_collective_rpc_preserves_results_and_arguments():
    rpc = source_method(SERVER, "vLLMHttpServer", "collective_rpc", {})
    values = [{"rank": 0}, {"rank": 1}]
    engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=values))
    kwargs = {"read_only": True}
    result = asyncio.run(rpc(SimpleNamespace(engine=engine), "get_batch_invariant_state", 15, (), kwargs))
    assert result is values
    engine.collective_rpc.assert_awaited_once_with(
        method="get_batch_invariant_state", timeout=15, args=(), kwargs=kwargs
    )


def test_server_collective_rpc_preserves_worker_errors():
    rpc = source_method(SERVER, "vLLMHttpServer", "collective_rpc", {})
    engine = SimpleNamespace(collective_rpc=AsyncMock(side_effect=RuntimeError("worker failed")))
    with pytest.raises(RuntimeError, match="worker failed"):
        asyncio.run(rpc(SimpleNamespace(engine=engine), "get_batch_invariant_state"))


@pytest.mark.parametrize(
    "initialized,compatible,foreign,requested,expected",
    [
        (True, True, False, "1", True),
        (False, True, False, "1", False),
        (True, True, True, "1", False),
        (True, False, False, "1", False),
        (True, True, False, "0", True),
    ],
)
def test_metadata_reports_actual_dispatch_and_ownership_without_mutation(
    monkeypatch, initialized, compatible, foreign, requested, expected
):
    def native(*args):
        raise AssertionError("Metadata must not execute an operator")

    def reducer(*args):
        raise AssertionError("Metadata must not execute an operator")

    reducer._verl_ascend_sum_compat = compatible
    selected = reducer if initialized else native
    fake_torch = SimpleNamespace(sum=native if foreign else selected, Tensor=SimpleNamespace(sum=selected))
    backend = SimpleNamespace(reduce_sum=reducer, _batch_invariant_LIB=object() if initialized else None)
    monkeypatch.setitem(sys.modules, "vllm_ascend.batch_invariant", backend)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", requested)
    monkeypatch.setenv("VERL_FULL_DETERMINISM", "0")
    inspect = source_method(
        WORKER, "vLLMColocateWorkerExtension", "get_batch_invariant_state",
        {"os": os, "torch": fake_torch, "is_npu_available": True},
    )
    before = fake_torch.sum, fake_torch.Tensor.sum
    result = inspect(SimpleNamespace(rank=1, local_rank=0))
    assert result["rank"] == 1 and result["local_rank"] == 0
    assert result["VLLM_BATCH_INVARIANT"] == requested
    assert result["ascend_dispatch_initialized"] is initialized
    assert result["ascend_sum_compatibility_enabled"] is expected
    assert (fake_torch.sum, fake_torch.Tensor.sum) == before


def test_metadata_does_not_import_or_enable_an_unused_backend(monkeypatch):
    monkeypatch.delitem(sys.modules, "vllm_ascend.batch_invariant", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    inspect = source_method(
        WORKER, "vLLMColocateWorkerExtension", "get_batch_invariant_state",
        {"os": os, "torch": SimpleNamespace(), "is_npu_available": False},
    )
    result = inspect(SimpleNamespace(rank=0))
    assert not result["ascend_dispatch_initialized"] and not result["ascend_sum_compatibility_enabled"]
    assert result["VLLM_BATCH_INVARIANT"] == "0"
    assert "vllm_ascend.batch_invariant" not in sys.modules


@pytest.mark.parametrize("is_npu,requested,prepares", [(True, "1", True), (True, "0", False), (False, "1", False)])
def test_real_constructor_prepares_only_opted_in_ascend_workers_before_hijack(monkeypatch, is_npu, requested, prepares):
    from verl.utils.vllm import ascend_batch_invariant

    events = []
    monkeypatch.setattr(ascend_batch_invariant, "prepare_ascend_batch_invariant_sum", lambda: events.append("prepare"))
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", requested)
    monkeypatch.setenv("VERL_FULL_DETERMINISM", "0")
    monkeypatch.setenv("VERL_VLLM_FP8_QUANT_ENABLED", "0")
    scope = {
        "os": os,
        "is_npu_available": is_npu,
        "set_death_signal": lambda: events.append("death_signal"),
        "VLLMHijack": SimpleNamespace(hijack=lambda: events.append("hijack")),
        "is_fp8_model": lambda config: False,
        "VLLM_ASCEND_REQUIRED_ENV_VARS": {},
    }
    worker = source_method(WORKER, "vLLMColocateWorkerExtension", "__new__", scope, keep_class=True)
    instance = worker()
    assert ("prepare" in events) is prepares
    if prepares:
        assert events.index("prepare") < events.index("hijack")
    assert not instance._is_qat_model and not instance._is_modelopt_qat
