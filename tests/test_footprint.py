"""Unit tests for prismaquant.footprint (exact artifact-GB accounting).

Validates the residual-floor identity
    artifact = (source_total - Σ_reencoded n_params·src_bpp)
               + Σ_reencoded memory_bytes_for_shape(shape, fmt)
against a hand-computed synthetic checkpoint, and the safetensors header reader
against a synthetic shard (header-only; no torch). The end-to-end 0.00% match vs
real 27B exports is covered by the verification pass, not here.
"""
from __future__ import annotations

import json
import struct

import pytest

from prismaquant import footprint as fp
from prismaquant import format_registry as fr


def _write_safetensors(path, tensors):
    """Write a minimal valid .safetensors file. tensors: {name: (dtype, shape)}.

    Data is zero-filled; only the header (dtype/shape/data_offsets) matters for
    the byte accounting, which reads spans, not values.
    """
    header = {}
    off = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = fp._ST_DTYPE_BYTES[dtype]
        for d in shape:
            nbytes *= d
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * off)


def test_source_checkpoint_bytes_reads_spans(tmp_path):
    _write_safetensors(tmp_path / "model-00001.safetensors", {
        "embed.weight": ("BF16", (100, 8)),      # 100*8*2 = 1600
        "layer.w.weight": ("BF16", (4, 8)),      # 4*8*2 = 64
    })
    _write_safetensors(tmp_path / "model-00002.safetensors", {
        "lm_head.weight": ("BF16", (100, 8)),    # 1600
    })
    total, by_dtype = fp.source_checkpoint_bytes(str(tmp_path))
    assert total == 1600 + 64 + 1600
    assert by_dtype == {"BF16": 1600 + 64 + 1600}
    assert fp.dominant_source_bytes_per_param(by_dtype) == 2


def test_source_checkpoint_bytes_no_shards(tmp_path):
    with pytest.raises(FileNotFoundError):
        fp.source_checkpoint_bytes(str(tmp_path))


def test_dominant_source_bytes_per_param_fp8():
    # native-fp8 source dominated by F8_E4M3 -> 1 byte/param
    assert fp.dominant_source_bytes_per_param({"F8_E4M3": 80, "BF16": 5}) == 1
    assert fp.dominant_source_bytes_per_param({}) == 2          # default bf16
    assert fp.dominant_source_bytes_per_param({"WEIRD": 9}) == 2  # unknown -> 2


def test_source_regime_robust_to_large_vocab_fp8():
    # The whole point: a large-vocab fp8 model where bf16 embed+lm_head OUTMASS
    # the fp8 body fools dominant-by-mass (-> bf16) but source_regime keys off
    # the *presence* of fp8 (which only the body has) -> correctly 'fp8'.
    by = {"BF16": 16000, "F8_E4M3": 10000, "F32": 4}
    assert fp.dominant_source_bytes_per_param(by) == 2   # mass says bf16 (wrong)
    assert fp.source_regime(by) == "fp8"                 # presence says fp8 (right)
    assert fp.source_regime({"BF16": 999}) == "bf16"
    assert fp.source_regime({}) == "bf16"


def test_assignment_artifact_bytes_residual_floor():
    # One body Linear (4x8, 32 params) re-encoded NVFP4; everything else is the
    # floor (kept at source precision). source_total carries a 1600-byte embed +
    # 1600-byte lm_head + the body's own 64 source bytes = 3264.
    stats = {"layer.w": {"n_params": 32, "in_features": 8, "out_features": 4}}
    source_total = 3264
    r = fp.assignment_artifact_bytes(
        {"layer.w": "NVFP4"}, stats,
        source_total_bytes=source_total, regime="bf16",
    )
    body_q = fr.get_format("NVFP4").memory_bytes_for_shape((4, 8))
    # floor = source_total - reencoded_source = 3264 - 32*2 = 3200 (embed+lm_head)
    assert r["floor_bytes"] == 3200
    assert r["body_quant_bytes"] == body_q
    assert r["artifact_bytes"] == 3200 + body_q
    assert r["reencoded_source_bytes"] == 64
    assert r["n_reencoded"] == 1
    assert r["n_missing_stats"] == 0
    assert r["regime"] == "bf16"


def test_assignment_artifact_bytes_fp8_source_removes_scale_inv():
    # fp8-native source: each re-encoded Linear ships fp8 weight + fp32 128x128
    # weight_scale_inv. The floor must remove BOTH (regime='fp8'), else the
    # source scale_inv is double-counted (the old scalar-src_bpp bug).
    stats = {"layer.w": {"n_params": 65536, "in_features": 256, "out_features": 256}}
    src_weight = 65536                                    # fp8 weight bytes
    src_scale_inv = fr.get_format("FP8_SOURCE").memory_bytes_for_shape((256, 256)) - src_weight
    embed = 1600
    source_total = src_weight + src_scale_inv + embed
    r = fp.assignment_artifact_bytes(
        {"layer.w": "NVFP4"}, stats,
        source_total_bytes=source_total, regime="fp8",
    )
    # floor must be exactly the embed; the fp8 weight AND its scale_inv are removed
    assert r["floor_bytes"] == embed
    assert r["reencoded_source_bytes"] == src_weight + src_scale_inv
    assert r["body_quant_bytes"] == fr.get_format("NVFP4").memory_bytes_for_shape((256, 256))
    assert r["artifact_bytes"] == embed + r["body_quant_bytes"]
    # the old scalar (n_params*1) would have left src_scale_inv in the floor:
    old_floor_bug = source_total - src_weight  # = embed + src_scale_inv
    assert old_floor_bug == embed + src_scale_inv and r["floor_bytes"] < old_floor_bug


def test_assignment_artifact_bytes_bf16_passthrough_is_floor_equivalent():
    # Re-encoding a tensor to BF16 must equal leaving it in the (bf16) floor:
    # body_quant(BF16) == source bytes, so artifact == source_total exactly.
    stats = {"layer.w": {"n_params": 32, "in_features": 8, "out_features": 4}}
    r = fp.assignment_artifact_bytes(
        {"layer.w": "BF16"}, stats,
        source_total_bytes=3264, regime="bf16",
    )
    assert r["artifact_bytes"] == 3264


def test_assignment_artifact_bytes_missing_stats_stay_in_floor():
    # A name absent from stats is not subtracted from the floor (stays at source
    # precision) and is counted as missing — the total is still well-defined.
    stats = {"layer.w": {"n_params": 32, "in_features": 8, "out_features": 4}}
    r = fp.assignment_artifact_bytes(
        {"layer.w": "NVFP4", "ghost.w": "NVFP4"}, stats,
        source_total_bytes=3264, regime="bf16",
    )
    assert r["n_missing_stats"] == 1
    assert r["n_reencoded"] == 1


def test_assignment_artifact_gb_matches_bytes():
    stats = {"layer.w": {"n_params": 32, "in_features": 8, "out_features": 4}}
    kw = dict(source_total_bytes=3264, regime="bf16")
    gb = fp.assignment_artifact_gb({"layer.w": "NVFP4"}, stats, **kw)
    b = fp.assignment_artifact_bytes({"layer.w": "NVFP4"}, stats, **kw)["artifact_bytes"]
    assert gb == pytest.approx(b / fp.GB)


def test_floor_bytes_for_model(tmp_path):
    _write_safetensors(tmp_path / "m.safetensors", {
        "embed.weight": ("BF16", (100, 8)),      # 1600 floor
        "layer.w.weight": ("BF16", (4, 8)),      # 64 reencoded
    })
    stats = {"layer.w": {"n_params": 32, "in_features": 8, "out_features": 4}}
    info = fp.floor_bytes_for_model(str(tmp_path), ["layer.w"], stats)
    assert info["source_total_bytes"] == 1664
    assert info["regime"] == "bf16"
    assert info["source_bytes_per_param"] == 2
    assert info["reencoded_source_bytes"] == 64
    assert info["floor_bytes"] == 1600
