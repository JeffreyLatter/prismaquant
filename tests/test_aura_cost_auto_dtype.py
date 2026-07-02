"""--dtype auto sizing for aura_cost: fp32 only when the resident model
fits with headroom (the 35B fp32 OOM-kill of 2026-07-01, prevented)."""
from __future__ import annotations

import json

from prismaquant.aura_cost import _resolve_auto_dtype

GIB = 1024**3


def _mk_checkpoint(tmp_path, total_size_bytes, fp8=False):
    weight_map = {"model.layers.0.self_attn.q_proj.weight": "model.safetensors"}
    if fp8:
        weight_map["model.layers.0.self_attn.q_proj.weight_scale_inv"] = (
            "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size_bytes},
        "weight_map": weight_map,
    }))
    return tmp_path


def test_auto_picks_fp32_when_it_fits(tmp_path):
    # 27B-class bf16 checkpoint: 54 GiB -> fp32 needs ~108 GiB; 121 GiB
    # available with 10 GiB headroom -> fits.
    _mk_checkpoint(tmp_path, 54 * GIB)
    assert _resolve_auto_dtype(
        tmp_path, 10.0, available_bytes=121 * GIB) == "float32"


def test_auto_picks_bf16_when_fp32_would_oom(tmp_path):
    # 35B-class bf16 checkpoint: 67 GiB -> fp32 needs ~134 GiB > 121 GiB.
    _mk_checkpoint(tmp_path, 67 * GIB)
    assert _resolve_auto_dtype(
        tmp_path, 18.0, available_bytes=121 * GIB) == "bfloat16"


def test_auto_accounts_for_fp8_sources(tmp_path):
    # fp8 source: 1 byte/param, so a 60 GiB checkpoint is ~60B params ->
    # fp32 needs ~240 GiB even though the file is "small".
    _mk_checkpoint(tmp_path, 60 * GIB, fp8=True)
    assert _resolve_auto_dtype(
        tmp_path, 18.0, available_bytes=121 * GIB) == "bfloat16"


def test_auto_unsizable_keeps_historical_default(tmp_path):
    assert _resolve_auto_dtype(
        tmp_path, 18.0, available_bytes=121 * GIB) == "float32"
