"""CPU tensor and installation contracts; no real device kernels are stubbed as proof."""

import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl.utils.ascend_gdn_prefill_fp32 import (
    get_ascend_gdn_prefill_precision_state,
    install_fp32_prefill,
    make_fp32_prefill_dispatch,
    prepare_ascend_fp32_gdn_prefill,
    select_native_chunk_metadata,
)


@pytest.fixture
def sample():
    q = torch.arange(10, dtype=torch.float32).reshape(1, 5, 1, 2).to(torch.bfloat16)
    state = (torch.arange(12, dtype=torch.float32) / 10000 + 1.0003).reshape(1, 2, 3, 2)
    values = dict(q=q, k=q.clone(), v=torch.zeros(1, 5, 2, 3, dtype=torch.bfloat16),
                  g=torch.zeros(1, 5, 2), beta=torch.ones(1, 5, 2, dtype=torch.bfloat16),
                  initial_state=state, cu_seqlens=torch.tensor([0, 5], dtype=torch.int32), scale=0.125)
    chunk = SimpleNamespace(chunk_indices_chunk64_host=(0, 0))
    meta = SimpleNamespace(prefill_query_start_loc=values["cu_seqlens"],
                           non_spec_prefill_metadata=SimpleNamespace(chunk=chunk))
    return values, chunk, meta


def config(dtype="float32", pcp=1):
    return SimpleNamespace(cache_config=SimpleNamespace(mamba_ssm_cache_dtype=dtype),
                           parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp))


def backend(pipeline):
    class Owner:
        _fused_chunk_available = None

        @staticmethod
        def _chunk_gated_delta_rule_fused(**kwargs):
            raise AssertionError("FP32 dispatch must not delegate to the BF16 state wrapper")

    return SimpleNamespace(AscendGatedDeltaNetAttention=Owner, chunk_gated_delta_rule=pipeline)


def test_default_off_imports_no_device_dependencies_in_fresh_process():
    code = '''
import importlib.abc, importlib.util, os, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'vllm_ascend'}:
            raise AssertionError('Default-off imported device dependency: ' + fullname)
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location('isolated_gdn_precision', os.environ['GDN_PREFILL_MODULE_PATH'])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prepare_ascend_fp32_gdn_prefill = module.prepare_ascend_fp32_gdn_prefill
get_ascend_gdn_prefill_precision_state = module.get_ascend_gdn_prefill_precision_state
assert prepare_ascend_fp32_gdn_prefill(None) == {'installed': False, 'reason': 'not_requested'}
assert get_ascend_gdn_prefill_precision_state() == {'installed': False}
'''
    env = dict(os.environ)
    env.pop("VERL_ASCEND_GDN_PREFILL_FP32", None)
    env.pop("VERL_USE_EXTERNAL_MODULES", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["GDN_PREFILL_MODULE_PATH"] = inspect.getsourcefile(prepare_ascend_fp32_gdn_prefill)
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=20)


def test_fp32_non_square_states_metadata_and_input_identity_preserved(sample):
    values, chunk, meta = sample
    before = values["initial_state"].clone()
    calls = []
    output = torch.zeros_like(values["v"])

    def pipeline(**kwargs):
        calls.append(kwargs)
        for name in ["q", "k", "v", "g", "beta", "cu_seqlens"]:
            assert kwargs[name] is values[name]
        assert kwargs["prebuilt_meta"] is chunk
        assert kwargs["initial_state"].dtype == torch.float32
        assert torch.equal(kwargs["initial_state"], before.transpose(-1, -2))
        assert not torch.equal(kwargs["initial_state"], before.bfloat16().float().transpose(-1, -2))
        assert kwargs["output_final_state"] and kwargs["use_qk_l2norm_in_kernel"]
        assert not kwargs["head_first"] and kwargs["scale"] == 0.125
        return output, kwargs["initial_state"] + 0.0002

    state = dict(calls=0)
    dispatch = make_fp32_prefill_dispatch(torch, pipeline, lambda: {"unrelated": None, "real": meta}, state)
    actual, final = dispatch(**values)
    assert actual is output and final.dtype == torch.float32
    assert tuple(final.shape) == (1, 2, 3, 2)
    assert torch.equal(final, before + 0.0002) and torch.equal(values["initial_state"], before)
    assert len(calls) == state["calls"] == 1
    assert state["exact_native_chunk_metadata_used"]
    assert state["last_initial_dtype"] == state["last_final_dtype"] == "torch.float32"


@pytest.mark.parametrize("bad", [None, "bf16", "wrong_rank", "wrong_layout", "wrong_heads"])
def test_invalid_incoming_state_rejected_before_backend(sample, bad):
    values, _, meta = sample
    values = dict(values)
    if bad is None:
        values["initial_state"] = None
    elif bad == "bf16":
        values["initial_state"] = values["initial_state"].bfloat16()
    elif bad == "wrong_rank":
        values["initial_state"] = values["initial_state"].squeeze(0)
    elif bad == "wrong_layout":
        values["initial_state"] = values["initial_state"].transpose(-1, -2)
    else:
        values["initial_state"] = values["initial_state"][:, :1]

    def forbidden(**kwargs):
        raise AssertionError("Invalid input reached backend")

    dispatch = make_fp32_prefill_dispatch(torch, forbidden, lambda: meta, dict(calls=0))
    with pytest.raises(ValueError):
        dispatch(**values)


@pytest.mark.parametrize("kind", ["missing", "bf16", "wrong_shape"])
def test_invalid_final_state_is_never_silently_promoted(sample, kind):
    values, _, meta = sample

    def pipeline(**kwargs):
        result = kwargs["initial_state"]
        if kind == "missing":
            result = None
        elif kind == "bf16":
            result = result.bfloat16()
        else:
            result = result[:, :, :, :1]
        return values["v"], result

    state = dict(calls=0)
    dispatch = make_fp32_prefill_dispatch(torch, pipeline, lambda: meta, state)
    with pytest.raises(RuntimeError):
        dispatch(**values)
    assert state["calls"] == 0


def test_backend_exception_propagates(sample):
    values, _, meta = sample
    error = RuntimeError("actual backend failure")

    def pipeline(**kwargs):
        raise error

    dispatch = make_fp32_prefill_dispatch(torch, pipeline, lambda: meta, dict(calls=0))
    with pytest.raises(RuntimeError) as caught:
        dispatch(**values)
    assert caught.value is error


def test_metadata_matches_request_identity_not_first_or_equal_valued_tensor(sample):
    values, chunk, meta = sample
    different_tensor = SimpleNamespace(prefill_query_start_loc=values["cu_seqlens"].clone(),
                                      non_spec_prefill_metadata=SimpleNamespace(chunk=object()))
    assert select_native_chunk_metadata({"first": different_tensor, "right": meta}, values["cu_seqlens"]) is chunk
    assert select_native_chunk_metadata({"alias1": meta, "alias2": meta}, values["cu_seqlens"]) is chunk
    with pytest.raises(RuntimeError):
        select_native_chunk_metadata({"equal_values_only": different_tensor}, values["cu_seqlens"])


def test_missing_ambiguous_or_empty_metadata_rejected(sample):
    values, _, meta = sample
    cu = values["cu_seqlens"]
    with pytest.raises(ValueError):
        select_native_chunk_metadata(meta, None)
    with pytest.raises(RuntimeError):
        select_native_chunk_metadata({}, cu)
    other = SimpleNamespace(prefill_query_start_loc=cu,
                            non_spec_prefill_metadata=SimpleNamespace(chunk=SimpleNamespace(
                                chunk_indices_chunk64_host=(0, 0))))
    with pytest.raises(RuntimeError):
        select_native_chunk_metadata({"a": meta, "b": other}, cu)
    other.non_spec_prefill_metadata.chunk.chunk_indices_chunk64_host = ()
    with pytest.raises(RuntimeError):
        select_native_chunk_metadata(other, cu)


@pytest.mark.parametrize("cfg", [config("bfloat16"), config(pcp=2), config(pcp=0)])
def test_incompatible_config_rejected_before_backend_mutation(cfg):
    b = backend(lambda **kwargs: None)
    previous = inspect.getattr_static(b.AscendGatedDeltaNetAttention, "_chunk_gated_delta_rule_fused")
    with pytest.raises(ValueError):
        install_fp32_prefill(cfg, torch, b, lambda: None)
    assert inspect.getattr_static(b.AscendGatedDeltaNetAttention, "_chunk_gated_delta_rule_fused") is previous


def test_installation_idempotent_constructor_and_probe_untouched_and_trace_hook_retained(sample):
    values, _, meta = sample
    calls = []
    b = backend(lambda **kwargs: (values["v"], kwargs["initial_state"]))
    owner = b.AscendGatedDeltaNetAttention
    constructor = owner.__new__
    result = install_fp32_prefill(config(), torch, b, lambda: meta)
    installed = inspect.getattr_static(owner, "_chunk_gated_delta_rule_fused")
    assert result["installed"] and not result["already_installed"]
    again = install_fp32_prefill(config(), torch, b, lambda: meta)
    assert again["already_installed"]
    assert inspect.getattr_static(owner, "_chunk_gated_delta_rule_fused") is installed
    assert owner.__new__ is constructor and owner._fused_chunk_available is None

    def observed(**kwargs):
        calls.append(kwargs)
        return values["v"], kwargs["initial_state"]

    b.chunk_gated_delta_rule = observed
    owner._chunk_gated_delta_rule_fused(**values)
    assert len(calls) == 1


def test_existing_experimental_dispatch_is_not_overwritten():
    b = backend(lambda **kwargs: None)
    owner = b.AscendGatedDeltaNetAttention
    original = inspect.getattr_static(owner, "_chunk_gated_delta_rule_fused")
    original.__func__._verl_prefix_precision_experiment = True
    with pytest.raises(RuntimeError):
        install_fp32_prefill(config(), torch, b, lambda: None)
    assert inspect.getattr_static(owner, "_chunk_gated_delta_rule_fused") is original


def test_metadata_observation_reads_copy_of_loaded_state_without_device_calls(monkeypatch, sample):
    values, _, meta = sample
    b = backend(lambda **kwargs: (values["v"], kwargs["initial_state"]))
    install_fp32_prefill(config(), torch, b, lambda: meta)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.gdn", b)
    result = get_ascend_gdn_prefill_precision_state()
    assert result["installed"] and result["calls"] == 0
    result["calls"] = 100
    assert get_ascend_gdn_prefill_precision_state()["calls"] == 0


def test_public_prepare_disabled_and_enabled_config_errors(monkeypatch):
    monkeypatch.delenv("VERL_ASCEND_GDN_PREFILL_FP32", raising=False)
    assert not prepare_ascend_fp32_gdn_prefill(None)["installed"]
    monkeypatch.setenv("VERL_ASCEND_GDN_PREFILL_FP32", "1")
    with pytest.raises(ValueError):
        prepare_ascend_fp32_gdn_prefill(config("bfloat16"))
    assert not torch.npu.is_initialized()
