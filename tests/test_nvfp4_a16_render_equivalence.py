"""NVFP4 and NVFP4A16 are ONE serialization and must render identically.

WHY
---
The two formats ship the same bytes: identical packed 4-bit weights, identical
group scales, identical per-tensor global scale. They differ only in whether the
exported config group declares ``input_activations``, which vLLM reads at LOAD
time to choose W4A4 (CUTLASS) or W4A16 (Marlin). Nothing about the weight plane
differs.

The render gate was literally ``fmt == "NVFP4"``, so NVFP4A16 fell through to
the registry RTN path and got weights that had never seen GPTQ, static_act_order
or JSO -- measured max|diff| 0.0217 on a 512x1024 Linear. Two consequences, both
bad:

  * An A16-vs-A4 A/B would have measured the RENDERING difference, not the
    activation contract. That is a rendering confound (principle 8), the same
    class of error that made the grouped-KL and staged-render results invert.
  * It silently biased the allocator AGAINST A16, because A16 looked worse on
    the weight plane for a reason that had nothing to do with activations.

So this is not a performance tidy-up; it is what makes the activation axis a
free and honest choice, and what lets ONE production cache entry serve both
contracts.

The second test is the one that stops a lazy fix: making both formats render
RTN would also make them equal, and would be a silent quality regression on the
shipping NVFP4 path. Equality is necessary, not sufficient.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prismaquant import format_registry as fr  # noqa: E402
from prismaquant.production_weight_cache import (  # noqa: E402
    NVFP4_RENDER_EQUIVALENT, render_production_weight,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the production render is a GPU path")

QNAME = "probe.linear"
LEVERS = {"gptq": True, "static_act_order": True, "joint_scale_opt": True,
          "gptq_damp_sweep": False, "gptq_fixed_damp": 1.0,
          "nvfp4_scale_rule": "joint_mse", "scale_sweep": False}


def _fixture(o=512, i=1024):
    torch.manual_seed(0)
    w = torch.randn(o, i, device="cuda", dtype=torch.bfloat16) * 0.02
    acts = {QNAME: torch.randn(128, i, device="cuda", dtype=torch.float32)}
    return w, acts


def _render(w, acts, fmt):
    return render_production_weight(
        w, fmt, qname=QNAME, activations=acts, levers=LEVERS)


def test_a16_renders_bit_identically_to_nvfp4():
    w, acts = _fixture()
    a = _render(w, acts, "NVFP4")
    b = _render(w, acts, "NVFP4A16")
    assert torch.equal(a, b), (
        "NVFP4 and NVFP4A16 are the same weight artifact; a difference here "
        "means an A16-vs-A4 comparison would measure the render, not the "
        f"activation contract (max|diff| "
        f"{(a.float() - b.float()).abs().max().item():.4g})")


def test_the_shared_render_is_the_activation_aware_one():
    """Equality must come from BOTH getting GPTQ+JSO, not both getting RTN.

    Making the two formats agree by dropping NVFP4 to the registry RTN path
    would satisfy the test above while silently regressing the shipping NVFP4
    render. This pins the direction of the fix.
    """
    w, acts = _fixture()
    rendered = _render(w, acts, "NVFP4A16")
    rtn = fr.get_format("NVFP4").quantize_dequantize(w)
    assert not torch.equal(rendered, rtn), (
        "NVFP4A16 rendered identically to plain RTN -- the activation-aware "
        "passes did not run, so the formats were equalized the wrong way")


def test_render_equivalence_set_is_explicit():
    """The set is a serialization fact and should be stated, not inferred."""
    assert NVFP4_RENDER_EQUIVALENT == {"NVFP4", "NVFP4A16"}
