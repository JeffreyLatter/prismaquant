"""Tests for the BlockOrtho-G exporter helpers."""

from __future__ import annotations

import math

import pytest
import torch

from prismaquant.block_rotation import sylvester_hadamard
from prismaquant.export_native_compressed import (
    _build_block_rotation_transform_export,
    _production_cache_block_rotation,
)
from prismaquant.production_weight_cache import ProductionWeightCache


class _IdentityProfile:
    def to_vllm_internal_name(self, name: str) -> str:
        return name


def _make_cache(rotations: dict[str, torch.Tensor]) -> ProductionWeightCache:
    return ProductionWeightCache(
        weights={},
        levers={"block_rotation": True},
        block_rotations=rotations,
    )


def test_build_transform_export_returns_none_when_no_rotations():
    cache = ProductionWeightCache(weights={}, levers={"block_rotation": False})
    tc, tensors = _build_block_rotation_transform_export(cache, _IdentityProfile())
    assert tc is None
    assert tensors == {}


def test_build_transform_export_single_cluster_one_scheme():
    g = 16
    R = sylvester_hadamard(g)
    rotations = {
        "model.layers.0.self_attn.q_proj": R,
        "model.layers.0.self_attn.k_proj": R,
        "model.layers.0.self_attn.v_proj": R,
    }
    tc, tensors = _build_block_rotation_transform_export(
        _make_cache(rotations), _IdentityProfile()
    )
    assert tc is not None
    assert set(tc["config_groups"].keys()) == {"block_ortho_g_0"}
    scheme = tc["config_groups"]["block_ortho_g_0"]
    assert scheme["type"] == "random-matrix"
    assert scheme["head_dim"] == g
    assert scheme["precision"] == "torch.float32"
    targets = scheme["apply"][0]["targets"]
    assert sorted(targets) == sorted(rotations.keys())
    assert scheme["apply"][0]["location"] == "input"

    # one transform tensor per cluster member, all referencing the same scheme
    expected_keys = {
        f"{qname}.block_ortho_g_0_input.weight" for qname in rotations
    }
    assert set(tensors.keys()) == expected_keys
    for tensor in tensors.values():
        assert tensor.shape == (g, g)
        # stored matrix is R * sqrt(G)
        recovered = tensor / math.sqrt(g)
        assert torch.allclose(recovered, R, atol=1e-6)


def test_build_transform_export_distinct_clusters_get_distinct_schemes():
    g = 16
    R_qkv = sylvester_hadamard(g)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(11)
    from prismaquant.block_rotation import random_orthogonal

    R_mlp = random_orthogonal(g, generator=gen, device="cpu")
    qkv_qnames = (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
    )
    mlp_qnames = (
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    )
    rotations = {q: R_qkv for q in qkv_qnames} | {q: R_mlp for q in mlp_qnames}
    tc, tensors = _build_block_rotation_transform_export(
        _make_cache(rotations), _IdentityProfile()
    )
    assert tc is not None
    assert set(tc["config_groups"].keys()) == {"block_ortho_g_0", "block_ortho_g_1"}

    # Look up which scheme name each cluster ended up under (order-agnostic).
    scheme_targets = {
        name: set(scheme["apply"][0]["targets"])
        for name, scheme in tc["config_groups"].items()
    }
    qkv_scheme = next(
        n for n, t in scheme_targets.items() if t == set(qkv_qnames)
    )
    mlp_scheme = next(
        n for n, t in scheme_targets.items() if t == set(mlp_qnames)
    )
    assert qkv_scheme != mlp_scheme

    expected_keys = (
        {f"{q}.{qkv_scheme}_input.weight" for q in qkv_qnames}
        | {f"{q}.{mlp_scheme}_input.weight" for q in mlp_qnames}
    )
    assert set(tensors.keys()) == expected_keys


def test_production_cache_block_rotation_lookup_validates_width():
    g = 16
    R = sylvester_hadamard(g)
    cache = _make_cache({"foo.q_proj": R})
    looked_up = _production_cache_block_rotation(
        cache, "foo.q_proj", width=64, device=torch.device("cpu")
    )
    assert looked_up is not None
    assert torch.allclose(looked_up, R.to(torch.float32), atol=1e-6)

    # Width not divisible by group size → not eligible, return None.
    assert (
        _production_cache_block_rotation(
            cache, "foo.q_proj", width=33, device=torch.device("cpu")
        )
        is None
    )

    # Width zero → not eligible.
    assert (
        _production_cache_block_rotation(
            cache, "foo.q_proj", width=0, device=torch.device("cpu")
        )
        is None
    )


def test_transform_config_roundtrips_through_compressed_tensors():
    """The dict we emit must be parseable by compressed_tensors.TransformConfig."""
    g = 16
    R = sylvester_hadamard(g)
    rotations = {
        "model.layers.0.self_attn.q_proj": R,
        "model.layers.0.self_attn.v_proj": R,
    }
    tc, _ = _build_block_rotation_transform_export(
        _make_cache(rotations), _IdentityProfile()
    )
    from compressed_tensors.transform import TransformConfig

    parsed = TransformConfig.model_validate(tc)
    assert set(parsed.config_groups.keys()) == set(tc["config_groups"].keys())
    scheme = parsed.config_groups["block_ortho_g_0"]
    assert scheme.head_dim == g
    assert scheme.type == "random-matrix"
    assert sorted(scheme.apply[0].targets) == sorted(rotations.keys())
