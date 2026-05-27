from __future__ import annotations

from prismaquant import format_registry as fr
from prismaquant.validate_assignments_kl import (
    _assignment_bpp_details,
    _kl_repeat_summary,
)


class _Profile:
    def is_pinned_name(self, qname: str) -> bool:
        return qname == "lm_head"

    def source_passthrough_prefixes(self) -> tuple[str, ...]:
        return ("mtp.", "model.visual.")


def _stats(n_params: int = 256) -> dict:
    return {
        "n_params": n_params,
        "in_features": 16,
        "out_features": 16,
    }


def test_assignment_bpp_excludes_pinned_and_auxiliary_entries():
    stats = {
        "model.layers.0.mlp.down_proj": _stats(),
        "mtp.layers.0.mlp.down_proj": _stats(),
        "model.visual.blocks.0.mlp.fc1": _stats(),
        "lm_head": _stats(),
    }
    assignment = {
        "model.layers.0.mlp.down_proj": "NVFP4",
        "mtp.layers.0.mlp.down_proj": "BF16",
        "model.visual.blocks.0.mlp.fc1": "BF16",
        "lm_head": "BF16",
    }
    specs = {name: fr.get_format(name) for name in ("NVFP4", "BF16")}

    details = _assignment_bpp_details(
        stats,
        assignment,
        specs,
        profile=_Profile(),
    )

    expected = (
        8.0
        * fr.get_format("NVFP4").memory_bytes_for_shape((16, 16))
        / 256.0
    )
    assert details["bpp"] == expected
    assert details["quantizable_entries"] == 1
    assert details["excluded_entries"] == 3


def test_assignment_bpp_excludes_auxiliary_entries_even_when_quantized():
    stats = {
        "model.layers.0.mlp.down_proj": _stats(),
        "model.visual.blocks.0.mlp.fc1": _stats(),
    }
    assignment = {
        "model.layers.0.mlp.down_proj": "NVFP4",
        "model.visual.blocks.0.mlp.fc1": "NVFP4",
    }
    specs = {name: fr.get_format(name) for name in ("NVFP4", "BF16")}

    details = _assignment_bpp_details(
        stats,
        assignment,
        specs,
        profile=_Profile(),
    )

    expected = (
        8.0
        * fr.get_format("NVFP4").memory_bytes_for_shape((16, 16))
        / 256.0
    )
    assert details["bpp"] == expected
    assert details["quantizable_entries"] == 1
    assert details["excluded_entries"] == 1


def test_kl_repeat_summary_reports_stderr_and_ucb():
    summary = _kl_repeat_summary([0.10, 0.20, 0.30], ucb_z=2.0)

    assert abs(summary["last_token_kl"] - 0.20) < 1e-12
    assert summary["kl_repeat_count"] == 3
    assert summary["kl_std"] > 0
    assert summary["kl_stderr"] > 0
    assert summary["kl_ucb"] > summary["last_token_kl"]
