from __future__ import annotations

import pytest
import torch

from prismaquant.kl_measurement import (
    _OverrideSetTargetHooks,
    _specs_by_canonical_name,
)


class _Cache:
    def __init__(self, tensor: torch.Tensor | None):
        self.tensor = tensor
        self.calls: list[tuple[str, str]] = []

    def get(self, name: str, fmt: str):
        self.calls.append((name, fmt))
        return self.tensor


def test_override_set_hook_uses_production_cache_without_prequant_cache():
    linear = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 1.0]]))
    cached_weight = torch.tensor([[10.0, 20.0]])
    cache = _Cache(cached_weight)
    hooks = _OverrideSetTargetHooks(
        torch.nn.Module(),
        {"linear": "FP8_E4M3"},
        _specs_by_canonical_name({"FP8_E4M3"}),
        [{"linear": "FP8_E4M3"}],
        base_batch=1,
        include_activation_quant=False,
        production_weight_cache=cache,
        strict_production_weight_cache=True,
    )

    x = torch.tensor([[2.0, 3.0]])
    y = linear(x)
    out = hooks._make_hook("linear")(linear, (x,), {}, y)

    assert torch.equal(out, torch.tensor([[80.0]]))
    assert cache.calls == [("linear", "FP8_E4M3")]


def test_override_set_hook_strict_production_cache_miss_raises():
    linear = torch.nn.Linear(2, 1, bias=False)
    cache = _Cache(None)
    hooks = _OverrideSetTargetHooks(
        torch.nn.Module(),
        {"linear": "FP8_E4M3"},
        _specs_by_canonical_name({"FP8_E4M3"}),
        [{"linear": "FP8_E4M3"}],
        base_batch=1,
        include_activation_quant=False,
        production_weight_cache=cache,
        strict_production_weight_cache=True,
    )

    x = torch.tensor([[2.0, 3.0]])
    y = linear(x)
    with pytest.raises(RuntimeError, match="production_weight_cache miss"):
        hooks._make_hook("linear")(linear, (x,), {}, y)
