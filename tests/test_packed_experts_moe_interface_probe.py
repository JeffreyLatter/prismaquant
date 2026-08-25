"""Regression: transformers 5.x dispatches packed-experts forward through an
MoE interface (`config._experts_implementation` -> batched_mm / grouped_mm)
that gathers `packed[expert_ids]` and runs a batched matmul — there is no
per-expert `F.linear(x, packed[e])` boundary, so the probe's per-token Fisher
interception can't see the weight and the M3 fail-fast guard trips.

`install_packed_expert_hooks` forces the reference "eager" per-expert
implementation for the probe forward (same math, interceptable kernel), so
the packed-expert Fisher is captured faithfully and the guard stays silent.

This is the Ornith-1.0-35B (Qwen3.5-MoE, transformers 5.5.4) case that the
Qwen3.6-35B-A3B runs never hit (older per-expert layout).
"""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

import prismaquant.sensitivity_probe as sp

moe_mod = pytest.importorskip(
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    reason="qwen3_5_moe modeling not available in this transformers build",
)
Qwen3_5MoeExperts = getattr(moe_mod, "Qwen3_5MoeExperts", None)


def _make_experts(impl):
    if Qwen3_5MoeExperts is None:
        pytest.skip("Qwen3_5MoeExperts not present")
    cfg = SimpleNamespace(
        num_experts=4, hidden_size=8, moe_intermediate_size=5,
        hidden_act="silu", _experts_implementation=impl)
    experts = Qwen3_5MoeExperts(cfg)
    experts.config = cfg
    # __init__ allocates via torch.empty (uninitialized) — fill with real values.
    with torch.no_grad():
        for pn in ("gate_up_proj", "down_proj"):
            p = getattr(experts, pn, None)
            if p is not None:
                p.copy_(torch.randn_like(p) * 0.1)
    return experts, cfg


def test_moe_interface_experts_probe_faithfully():
    """batched_mm default would trip the guard; the hook forces eager and
    captures nonzero per-token Fisher for every routed expert."""
    experts, cfg = _make_experts("batched_mm")
    parent = nn.Module()
    parent.add_module("experts", experts)

    acc, ch, fu = {}, {}, {}
    sp.install_packed_expert_hooks(parent, acc, ch, fu, profile=None)
    # the hook must have forced the interceptable implementation
    assert cfg._experts_implementation == "eager"

    torch.manual_seed(0)
    hs = torch.randn(6, 8, requires_grad=True)
    top_k_index = torch.randint(0, 4, (6, 2))
    top_k_weights = torch.rand(6, 2)

    out = experts.forward(hs, top_k_index, top_k_weights)  # must NOT raise the guard
    out.pow(2).sum().backward()

    # faithful per-token Fisher accumulated for both packed params, all experts
    assert set(ch) == {"experts.gate_up_proj", "experts.down_proj"}
    for name, tens in ch.items():
        assert tens.shape[0] == 4
        assert float(tens.sum()) > 0.0, f"{name} accumulated zero Fisher"
        assert int((tens.sum(dim=1) > 0).sum()) == 4, f"{name} missing experts"
    # no dense [E, M, N] weight gradient materialized (memory guard)
    assert experts.gate_up_proj.grad is None
    assert experts.down_proj.grad is None
