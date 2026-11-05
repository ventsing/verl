# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in compatibility for Ascend's restricted batch-invariant sum kernel.

Some vllm-ascend versions replace all torch/Tensor sums with a floating-point,
last-axis-only CANN operator. This breaks visual profiling's nonlast-axis sum
and the sampler's boolean rank count. Preserve native torch.sum semantics for
unsupported calls; retain the invariant kernel for supported floating sums.
This guard alone does not establish whole-model batch invariance.

Nothing is installed on import. Call prepare_ascend_batch_invariant_sum in a
new worker before vllm-ascend initializes its invariant dispatch, or afterward
while its original sum replacement is still installed.
"""

import os


def make_batch_invariant_sum(torch_module, original_sum, invariant_sum):
    """Route only the CANN kernel's supported subset; preserve all other calls."""
    supported_dtypes = (torch_module.float16, torch_module.float32, torch_module.bfloat16)

    def compatible_sum(input, *args, **kwargs):
        # Let torch validate uncommon/invalid overloads and preserve its errors,
        # dtype promotion, explicit accumulation dtype, and out= behavior.
        if (
            not isinstance(input, torch_module.Tensor)
            or len(args) > 2
            or set(kwargs) - {"dim", "keepdim", "dtype", "out"}
            or (args and "dim" in kwargs)
            or (len(args) > 1 and "keepdim" in kwargs)
            or kwargs.get("dtype") is not None
            or kwargs.get("out") is not None
        ):
            return original_sum(input, *args, **kwargs)
        dim = args[0] if args else kwargs.get("dim")
        keepdim = args[1] if len(args) > 1 else kwargs.get("keepdim", False)
        if dim is None and input.ndim == 1:
            dim = -1
        if (
            input.device.type in {"npu", "privateuseone"}
            and input.dtype in supported_dtypes
            and input.ndim > 0
            and input.numel() > 0
            and isinstance(dim, int)
            and not isinstance(dim, bool)
            and dim in {-1, input.ndim - 1}
            and isinstance(keepdim, bool)
        ):
            return invariant_sum(input, -1, keepdim)
        return original_sum(input, *args, **kwargs)

    compatible_sum.__name__ = "verl_ascend_batch_invariant_sum"
    compatible_sum._verl_ascend_sum_compat = True
    return compatible_sum


def prepare_ascend_batch_invariant_sum():
    """Prepare only an explicitly requested Ascend invariant worker.

    All mutations are process-local. Dependency files and default workers are
    untouched. The native reduction fallback does not promise batch-invariant
    floating reductions outside the supported last-axis kernel subset.
    """
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "1":
        return {"installed": False, "reason": "batch_invariant_not_requested"}
    import torch
    from vllm_ascend import batch_invariant as backend

    if getattr(backend.reduce_sum, "_verl_ascend_sum_compat", False):
        if backend._batch_invariant_LIB is not None and (
            torch.sum is not backend.reduce_sum or torch.Tensor.sum is not backend.reduce_sum
        ):
            raise RuntimeError("Prepared invariant sum was replaced by another owner")
        return {"installed": True, "already_prepared": True}
    if not backend.HAS_ASCENDC_BATCH_INVARIANT:
        return {"installed": False, "reason": "ascendc_sum_replacement_not_used"}
    previous = backend.reduce_sum
    wrapper = make_batch_invariant_sum(
        torch, backend.torch_sum, torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant
    )
    # Native enable_batch_invariant_mode subsequently reads this module global.
    backend.reduce_sum = wrapper
    if backend._batch_invariant_LIB is not None:
        # Fail before changing dispatch if another owner has replaced either sum.
        if torch.sum is not previous or torch.Tensor.sum is not previous:
            backend.reduce_sum = previous
            raise RuntimeError("Cannot replace an independently modified torch sum implementation")
        torch.sum = wrapper
        torch.Tensor.sum = wrapper
    return {"installed": True, "already_prepared": False}
