"""Unit tests for block-wise output matching (#12)."""
from __future__ import annotations

import torch
import torch.nn as nn

from prismaquant.block_output_match import (
    BlockSpec,
    block_output_mse,
    refine_block_scales,
)


class _TwoLinearBlock(nn.Module):
    """A trivial 'block' = two Linears in series. Stand-in for an
    attention or MLP block. Used to test the refiner mechanics without
    full transformer plumbing."""

    def __init__(self, d: int):
        super().__init__()
        self.l1 = nn.Linear(d, d, bias=False)
        self.l2 = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def _scale_perturbed_weight(orig: torch.Tensor, mult: float) -> torch.Tensor:
    """Return orig × `mult` per-group. Standin for 'this scale candidate
    produces this rounded weight'."""
    return orig * mult


def test_refine_picks_better_scale():
    torch.manual_seed(0)
    d = 16
    block = _TwoLinearBlock(d).eval()
    x = torch.randn(8, d)

    # FP16 reference output.
    with torch.no_grad():
        ref = block(x).clone()

    # Save original weights.
    l1_orig = block.l1.weight.data.clone()
    l2_orig = block.l2.weight.data.clone()

    # Set up the spec. "scale" candidates here are just multiplicative
    # perturbations of the original weight (a stand-in for swapping in
    # a candidate-quantized weight). Candidate 1.0 = identity (best);
    # candidate 1.5 = perturbed (worse).
    def scale_setter(qname, scale):
        if qname == "l1":
            block.l1.weight.data = _scale_perturbed_weight(l1_orig, scale)
        elif qname == "l2":
            block.l2.weight.data = _scale_perturbed_weight(l2_orig, scale)

    def scale_getter(qname):
        if qname == "l1":
            # Recover the multiplier from current weight.
            return torch.tensor(
                (block.l1.weight.data / l1_orig).mean().item())
        elif qname == "l2":
            return torch.tensor(
                (block.l2.weight.data / l2_orig).mean().item())

    spec = BlockSpec(
        linears=["l1", "l2"],
        forward_fn=lambda x: block(x),
        scale_setter=scale_setter,
        scale_getter=scale_getter,
    )

    # Start with both Linears perturbed (mult=1.5 → wrong).
    scale_setter("l1", 1.5)
    scale_setter("l2", 1.5)
    initial_mse = block_output_mse(spec, x, ref)
    assert initial_mse > 0

    # Candidates include the right scale (1.0).
    candidates = {
        "l1": [torch.tensor(0.5), torch.tensor(1.0), torch.tensor(1.5)],
        "l2": [torch.tensor(0.5), torch.tensor(1.0), torch.tensor(1.5)],
    }
    final_mse = refine_block_scales(spec, x, ref, candidates,
                                    max_passes=3)

    # Greedy coordinate descent finds the best 1-coord-reachable state,
    # not necessarily the global optimum (which would need joint moves).
    # Concrete check: refiner improves materially over initial.
    assert final_mse < initial_mse, (
        f"refiner regressed: initial={initial_mse}, final={final_mse}")
    # And the improvement is substantial (>10×) on this problem.
    assert final_mse < initial_mse / 10, (
        f"refiner didn't deliver expected gain: "
        f"initial={initial_mse}, final={final_mse}")


def test_make_attention_block_spec():
    """make_attention_block_spec produces a working BlockSpec for a
    standard transformer-style attention module."""
    torch.manual_seed(0)
    d = 16

    class _SelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False)
            self.k_proj = nn.Linear(d, d, bias=False)
            self.v_proj = nn.Linear(d, d, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _SelfAttn()

    from prismaquant.block_output_match import (
        make_attention_block_spec, block_output_mse, refine_block_scales)

    layer = _Layer().eval()
    x = torch.randn(4, 8, d)
    with torch.no_grad():
        ref = layer.self_attn.o_proj(
            torch.softmax(layer.self_attn.q_proj(x)
                          @ layer.self_attn.k_proj(x).transpose(-2, -1)
                          / (d ** 0.5), dim=-1)
            @ layer.self_attn.v_proj(x))

    spec = make_attention_block_spec(layer, "model.layers.0")
    assert spec is not None
    assert set(spec.linears) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    # At identity scales, MSE should be ~0.
    mse_id = block_output_mse(spec, x, ref)
    assert mse_id < 1e-6

    # Perturb q_proj, verify refiner restores it.
    spec.scale_setter("q_proj", torch.tensor(1.5))
    mse_perturbed = block_output_mse(spec, x, ref)
    assert mse_perturbed > mse_id

    candidates = {qn: [torch.tensor(0.85), torch.tensor(1.0),
                       torch.tensor(1.15)] for qn in spec.linears}
    final = refine_block_scales(spec, x, ref, candidates, max_passes=2)
    assert final < mse_perturbed


def test_refine_handles_no_improvement():
    """If no candidate improves over the current state, refine_block_scales
    should converge in <= max_passes and not regress."""
    torch.manual_seed(0)
    d = 8
    block = _TwoLinearBlock(d).eval()
    x = torch.randn(4, d)
    with torch.no_grad():
        ref = block(x).clone()

    l1_orig = block.l1.weight.data.clone()
    l2_orig = block.l2.weight.data.clone()

    def scale_setter(qname, scale):
        if qname == "l1":
            block.l1.weight.data = _scale_perturbed_weight(l1_orig, scale)
        elif qname == "l2":
            block.l2.weight.data = _scale_perturbed_weight(l2_orig, scale)

    def scale_getter(qname):
        if qname == "l1":
            return torch.tensor(
                (block.l1.weight.data / l1_orig).mean().item())
        elif qname == "l2":
            return torch.tensor(
                (block.l2.weight.data / l2_orig).mean().item())

    spec = BlockSpec(
        linears=["l1", "l2"],
        forward_fn=lambda x: block(x),
        scale_setter=scale_setter,
        scale_getter=scale_getter,
    )

    # Start at the optimum (mult=1.0). All candidates are worse.
    scale_setter("l1", 1.0)
    scale_setter("l2", 1.0)
    initial_mse = block_output_mse(spec, x, ref)
    assert initial_mse < 1e-6

    candidates = {
        "l1": [torch.tensor(0.5), torch.tensor(1.5)],  # both worse
        "l2": [torch.tensor(0.5), torch.tensor(1.5)],
    }
    final_mse = refine_block_scales(spec, x, ref, candidates,
                                    max_passes=2)
    # Final state shouldn't have regressed.
    assert final_mse <= initial_mse + 1e-9
