# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Compare patched decoder outputs and gradients with the installed HF model."""

import copy

import pytest
import torch

modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
config_module = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
patch = pytest.importorskip("verl.models.transformers.qwen3_5")


@pytest.mark.parametrize("block_type", ["linear_attention", "full_attention"])
@pytest.mark.parametrize("legacy_attribute", [False, True])
def test_decoder_outputs_and_gradients_match_hf(block_type, legacy_attribute, monkeypatch):
    # Keep the CPU test independent of optional accelerator-only FLA kernels.
    for name in ["FusedRMSNormGated", "causal_conv1d_fn", "chunk_gated_delta_rule"]:
        monkeypatch.setattr(modeling, name, None)
    config = config_module.Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=[block_type], attn_implementation="eager",
    )
    torch.manual_seed(42)
    reference = modeling.Qwen3_5DecoderLayer(config, 0)
    actual = copy.deepcopy(reference)
    if legacy_attribute:
        actual.layer_type = block_type
        if hasattr(actual, "block_type"):
            del actual.block_type
    actual.forward = patch.qwen3_5_decoder_layer_forward.__get__(actual)
    if block_type == "linear_attention":
        actual.linear_attn.forward = patch.qwen3_5_gated_delta_net_forward.__get__(actual.linear_attn)
    x = torch.randn(1, 11, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    embeddings = (torch.ones(1, 11, 16), torch.zeros(1, 11, 16))
    expected = reference(x, position_embeddings=embeddings)
    observed = actual(y, position_embeddings=embeddings)
    torch.testing.assert_close(observed, expected)
    expected.square().mean().backward()
    observed.square().mean().backward()
    torch.testing.assert_close(y.grad, x.grad)
    for (name, param), (other_name, other) in zip(reference.named_parameters(), actual.named_parameters(), strict=True):
        assert name == other_name
        if param.grad is not None:
            torch.testing.assert_close(other.grad, param.grad)
