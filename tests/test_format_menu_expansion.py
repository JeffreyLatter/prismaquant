from __future__ import annotations

import pytest

from prismaquant import format_registry as fr
from prismaquant.allocator import filter_candidates_for_profile
from prismaquant.allocator_candidates import check_format_applicability
from prismaquant.allocator_solver import Candidate
from prismaquant.export_native_compressed import (
    FORMAT_SCHEME,
    canonicalize_format,
)
from prismaquant.kl_sensitivity_probe import _production_cache_formats


VLLM_PROFILE = "vllm_qwen3_5_packed_moe"


def test_production_cache_formats_include_all_non_bf16_registry_picks():
    assert _production_cache_formats(
        ["NVFP4", "MXFP8_E4M3", "FP8_E4M3", "BF16"],
        "NVFP4",
    ) == ["FP8_E4M3", "MXFP8_E4M3", "NVFP4"]

    assert _production_cache_formats(
        ["MXFP8_E5M2", "FP8_E5M2", "BF16"],
        "NVFP4",
    ) == ["FP8_E5M2", "MXFP8_E5M2", "NVFP4"]


def test_vllm_profile_allows_dense_fp8_e4m3_but_not_e5m2():
    dense = "model.layers.0.self_attn.q_proj"
    shape = (5120, 17408)

    assert check_format_applicability(
        shape,
        "FP8_E4M3",
        qname=dense,
        source_kind="bf16",
        target_profile=VLLM_PROFILE,
    ).legal

    for fmt in ("MXFP8_E5M2", "FP8_E5M2"):
        verdict = check_format_applicability(
            shape,
            fmt,
            qname=dense,
            source_kind="bf16",
            target_profile=VLLM_PROFILE,
        )
        assert not verdict.legal
        assert verdict.reason == "profile_mismatch"


def test_vllm_profile_keeps_packed_moe_menu_conservative():
    expert = "model.layers.0.mlp.experts.gate_up_proj"
    shape = (5120, 17408)

    assert check_format_applicability(
        shape,
        "MXFP8_E4M3",
        qname=expert,
        source_kind="bf16",
        target_profile=VLLM_PROFILE,
    ).legal

    for fmt in ("FP8_E4M3", "MXFP8_E5M2", "FP8_E5M2"):
        verdict = check_format_applicability(
            shape,
            fmt,
            qname=expert,
            source_kind="bf16",
            target_profile=VLLM_PROFILE,
        )
        assert not verdict.legal
        assert verdict.reason == "profile_mismatch"


def test_allocator_profile_filter_keeps_only_vllm_backed_fp8_choices():
    dense = "model.layers.0.self_attn.q_proj"
    expert = "model.layers.0.mlp.experts.gate_up_proj"
    candidates = {
        dense: [
            Candidate("NVFP4", 4.5, 100, 1.0),
            Candidate("FP8_E4M3", 8.5, 190, 0.1),
            Candidate("FP8_E5M2", 8.5, 190, 0.1),
            Candidate("MXFP8_E5M2", 8.25, 180, 0.2),
            Candidate("BF16", 16.0, 512, 0.0),
        ],
        expert: [
            Candidate("NVFP4", 4.5, 100, 1.0),
            Candidate("MXFP4", 4.25, 96, 0.9),
            Candidate("FP8_E4M3", 8.5, 190, 0.1),
            Candidate("MXFP8_E4M3", 8.25, 180, 0.2),
            Candidate("BF16", 16.0, 512, 0.0),
        ],
    }

    filtered = filter_candidates_for_profile(candidates, VLLM_PROFILE)

    assert [c.fmt for c in filtered[dense]] == [
        "NVFP4",
        "FP8_E4M3",
        "BF16",
    ]
    assert [c.fmt for c in filtered[expert]] == [
        "NVFP4",
        "MXFP4",
        "MXFP8_E4M3",
        "BF16",
    ]


def test_export_canonicalizes_and_configures_dense_fp8_e4m3():
    assert canonicalize_format("FP8_E4M3") == "FP8_E4M3"
    assert canonicalize_format(
        fr.get_format("FP8_E4M3").autoround_config()
    ) == "FP8_E4M3"
    assert FORMAT_SCHEME["FP8_E4M3"]["format"] == "float-quantized"
    assert FORMAT_SCHEME["FP8_E4M3"]["weights"]["strategy"] == "channel"
    assert FORMAT_SCHEME["FP8_E4M3"]["input_activations"]["strategy"] == "token"


def test_export_rejects_e5m2_until_vllm_weight_dispatch_is_validated():
    with pytest.raises(ValueError, match="E5M2"):
        canonicalize_format("FP8_E5M2")
    with pytest.raises(ValueError, match="E5M2"):
        canonicalize_format(fr.get_format("MXFP8_E5M2").autoround_config())
