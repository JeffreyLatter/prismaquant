"""Oracle-agreement spot-check: AQUA's materialized W vs the streaming loader.

Runs only on a box that has the DSv4-Flash-0731 source checkpoint and its
sensitivity card; skips everywhere else. For units spanning BOTH source
encodings (MXFP4 routed experts, block-FP8 attention projections) it proves
that `materialize_source_weight` produces exactly the weights
`_apply_fp8_dequant_inplace` -- the literal function the probe, the cost stage
and the exporter loaded this model through -- installs for the same tensors.

Agreement is BIT-EXACT with zero tolerance, at bf16 AND at fp32: E2M1 and
e4m3 elements times power-of-two E8M0 scales are all exactly representable in
bf16 (2- and 3-bit significands), so the loader's bf16 storage downcast is
lossless and there is no rounding anywhere to hide behind.
"""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

MODEL = "/home/rob/dq-runs/dsv4-flash-0731/source"
CARD = ("/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p7-marginals/"
        "artifacts/dsv4-flash-sensitivity-card.npz")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(MODEL) and os.path.isfile(CARD)),
    reason="DSv4-Flash-0731 source checkpoint + card not on this box")

UNITS = [
    "model.layers.0.mlp.experts.0.gate_proj",   # routed expert w1 (MXFP4)
    "model.layers.0.mlp.experts.0.down_proj",   # routed expert w2 (MXFP4)
    "model.layers.41.self_attn.wq_a",           # attention proj (block-FP8)
    "model.layers.0.self_attn.wkv",             # attention proj (block-FP8)
]


@pytest.fixture(scope="module")
def env():
    from safetensors import safe_open  # noqa: F401  (import check)

    from prismaquant.aqua_activation_cost import build_weight_resolver
    from prismaquant.layer_streaming import _build_fp8_scale_inv_map
    from prismaquant.model_profiles.registry import detect_profile
    from prismaquant.sensitivity_card import SensitivityCard

    with open(os.path.join(MODEL, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    profile = detect_profile(MODEL)
    return {
        "card": SensitivityCard.from_npz(CARD),
        "fp8_map": _build_fp8_scale_inv_map(MODEL),
        "weight_map": weight_map,
        "resolver": build_weight_resolver(weight_map, profile=profile),
    }


@pytest.mark.parametrize("name", UNITS)
def test_materialized_w_equals_the_streaming_loaders_install(name, env):
    from safetensors import safe_open

    from prismaquant.aqua_activation_cost import materialize_source_weight
    from prismaquant.layer_streaming import _apply_fp8_dequant_inplace

    card, fp8_map = env["card"], env["fp8_map"]
    unit = card[name]
    ck_key = env["resolver"][name]
    with safe_open(os.path.join(MODEL, env["weight_map"][ck_key]),
                   framework="pt", device="cpu") as h:
        raw = h.get_tensor(ck_key)
        scale = h.get_tensor(fp8_map[name + ".weight"][1])

    mine = materialize_source_weight(name, raw, scale, fp8_map)
    assert mine.dtype == torch.float32
    assert tuple(mine.shape) == (unit.out_features, unit.in_features), (
        "the materialized W must match the probe stats' logical geometry")
    assert torch.isfinite(mine).all()

    # The oracle: the streaming loader's own dequant pass on the raw tensor,
    # reading its scales from the shards itself.
    live = {name + ".weight": raw.clone()}
    n = _apply_fp8_dequant_inplace(live, fp8_map, torch.device("cpu"))
    loader = live[name + ".weight"]
    assert n == 1 and loader.dtype == torch.bfloat16

    assert torch.equal(mine.to(torch.bfloat16), loader), (
        "bf16 downcast must be bit-identical to the loader's installed tensor")
    assert torch.equal(mine, loader.to(torch.float32)), (
        "and the fp32 values themselves are exactly bf16-representable, so "
        "even the fp32 comparison is zero-tolerance")

    # Magnitude sanity: dequanted weights, not fp8 code-range values (±448).
    assert float(mine.abs().max()) < 4.0
