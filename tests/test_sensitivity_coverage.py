from __future__ import annotations

from prismaquant.sensitivity_coverage import summarize_sensitivity_coverage


def test_sensitivity_coverage_counts_top_units_without_double_counting_members():
    report = {
        "schema": "prismaquant.propagated_group_sensitivity.v1",
        "rows": [
            {
                "key": "fused:a.qkv",
                "category": "self_attn",
                "layer": "0",
                "members": ["a.q", "a.k", "a.v"],
                "propagated_kl": 0.3,
                "propagated_kl_per_added_bit": 3.0,
            },
            {
                "key": "tensor:b.down",
                "category": "shared_expert",
                "layer": "1",
                "members": ["b.down"],
                "propagated_kl": 0.4,
                "propagated_kl_per_added_bit": 2.0,
            },
            {
                "key": "tensor:c.out",
                "category": "linear_attn",
                "layer": "2",
                "members": ["c.out"],
                "propagated_kl": 0.5,
                "propagated_kl_per_added_bit": 1.0,
            },
        ],
    }
    assignment = {
        "a.q": "MXFP8_E4M3",
        "a.k": "MXFP8_E4M3",
        "a.v": "MXFP8_E4M3",
        "b.down": "BF16",
        "c.out": "NVFP4",
    }

    summary = summarize_sensitivity_coverage(report, assignment, top_ns=(1, 2, 3))

    assert summary["top"]["1"]["format_units"] == {"MXFP8_E4M3": 1}
    assert summary["top"]["1"]["no_nvfp4_units"] == 1
    assert summary["top"]["2"]["all_bf16_units"] == 1
    assert summary["top"]["2"]["no_nvfp4_units"] == 2
    assert summary["top"]["3"]["nvfp4_units"] == 1
    assert summary["units"][0]["member_count"] == 3
    assert summary["units"][0]["format_key"] == "MXFP8_E4M3"


def test_sensitivity_coverage_surfaces_mixed_or_missing_fused_units():
    report = {
        "rows": [
            {
                "key": "fused:a.qkv",
                "members": ["a.q", "a.k", "a.v"],
                "propagated_kl_per_added_bit": 1.0,
            },
        ],
    }
    assignment = {
        "a.q": "MXFP8_E4M3",
        "a.k": "NVFP4",
    }

    summary = summarize_sensitivity_coverage(report, assignment, top_ns=(1,))

    unit = summary["units"][0]
    assert unit["format_key"] == "mixed:MISSING=1,MXFP8_E4M3=1,NVFP4=1"
    assert unit["has_nvfp4"] is True
    assert unit["no_nvfp4"] is False
    assert unit["has_missing"] is True
    assert summary["top"]["1"]["missing_units"] == 1
