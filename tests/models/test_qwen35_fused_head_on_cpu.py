"""Check chunked Qwen3.5 output-head values, gradients and module hooks."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from verl.models.transformers.monkey_patch import patch_forward_with_backends


class Backbone(nn.Embedding):
    def forward(self, input_ids, **kwargs):
        return BaseModelOutput(last_hidden_state=super().forward(input_ids))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_5")
        self.model = Backbone(19, 8)
        self.lm_head = nn.Linear(8, 19, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids)[0])


@pytest.mark.parametrize("temperature", [0.6, 1.0, 1.5])
@pytest.mark.parametrize("shift_labels", [False, True])
def test_fused_head_uses_module_hooks_and_matches_dense(temperature, shift_labels):
    # Use a fresh subclass: backend patching changes the model class forward.
    class Actual(TinyModel):
        pass

    torch.manual_seed(42)
    actual = Actual()
    expected = TinyModel()
    expected.load_state_dict(copy.deepcopy(actual.state_dict()))
    patch_forward_with_backends(actual, use_fused_kernels=True, fused_kernels_backend="torch")
    hooks = []
    handle = actual.lm_head.register_forward_hook(lambda *args: hooks.append(True))
    inputs = torch.tensor([[2, 3, 4, 5, 6]])
    targets = torch.tensor([[7, 8, 9, 10, 11]]) if shift_labels else inputs.roll(-1, -1)
    kwargs = {"shift_labels": targets} if shift_labels else {}
    observed = actual(inputs, temperature=temperature, **kwargs)
    handle.remove()
    assert hooks == [True]
    logits = expected(inputs) / temperature
    logprobs = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    torch.testing.assert_close(observed.log_probs, logprobs)
    torch.testing.assert_close(observed.entropy, entropy)
    (-observed.log_probs.mean() + 0.1 * observed.entropy.mean()).backward()
    (-logprobs.mean() + 0.1 * entropy.mean()).backward()
    for name, param in actual.named_parameters():
        torch.testing.assert_close(param.grad, dict(expected.named_parameters())[name].grad)
    # Calling the head without PPO labels retains its usual linear behavior.
    hidden = torch.randn(1, 3, 8)
    torch.testing.assert_close(actual.lm_head(hidden), expected.lm_head(hidden))
