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
    # + 8 B fp32 NVFP4 global sidecars (weight_global_scale +
    # input_global_scale) the export emits per 2-D Linear (§3.14 fix).
    body_q = fr.get_format("NVFP4").memory_bytes_for_shape((4, 8)) + 8
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
    assert r["body_quant_bytes"] == (
        fr.get_format("NVFP4").memory_bytes_for_shape((256, 256)) + 8)
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


def test_nvfp4_global_sidecar_bytes_dense_and_packed():
    """§3.14 (2026-07-02 audit): the export emits fp32 weight_global_scale +
    input_global_scale per NVFP4 2-D Linear (8 B, verified against shipped
    safetensors headers), and per expert × on-disk projection for packed 3-D
    tensors (gate_up_proj splits into gate_proj + up_proj per expert)."""
    assert fp.nvfp4_global_sidecar_bytes("model.layers.0.self_attn.q_proj",
                                         (128, 64)) == 8
    # down_proj: one projection per expert -> 8·E
    assert fp.nvfp4_global_sidecar_bytes(
        "model.layers.0.mlp.experts.down_proj", (256, 32, 64)) == 8 * 256
    # gate_up_proj: two on-disk projections per expert -> 8·E·2
    assert fp.nvfp4_global_sidecar_bytes(
        "model.layers.0.mlp.experts.gate_up_proj", (256, 128, 32)) == 16 * 256


def test_assignment_artifact_bytes_packed_nvfp4_counts_per_expert_globals():
    stats = {
        "layer.experts.gate_up_proj": {
            "n_params": 4 * 128 * 32, "in_features": 32,
            "out_features": 128, "num_experts": 4,
        },
    }
    r = fp.assignment_artifact_bytes(
        {"layer.experts.gate_up_proj": "NVFP4"}, stats,
        source_total_bytes=4 * 128 * 32 * 2, regime="bf16",
    )
    expected = (
        fr.get_format("NVFP4").memory_bytes_for_shape((4, 128, 32))
        + 8 * 4 * 2  # per-expert weight_global + input_global, gate+up
    )
    assert r["body_quant_bytes"] == expected


# ---------------------------------------------------------------------------
# Per-tensor source-byte manifest (mixed-precision sources)
# ---------------------------------------------------------------------------

def _mixed_source_checkpoint(tmp_path):
    """Two re-encoded tensors on a mixed source: an MXFP4-packed expert
    (I8 nibble weights, 0.5 B/param on disk, + E8M0 group scales) and an
    fp8 attention Linear (+ fp32 weight_scale_inv), plus a BF16 floor
    tensor. The fp8 dtype flips the regime detector to "fp8", so the
    regime path charges the packed expert 1 B/LOGICAL-param — about twice
    its actual on-disk bytes."""
    _write_safetensors(tmp_path / "m.safetensors", {
        # logical (64, 256) packed 2-per-byte -> stored (64, 128) I8 = 8192 B
        "layers.0.experts.0.down_proj.weight": ("I8", (64, 128)),
        # E8M0 scale per 32-element group: (64, 8) U8 = 512 B
        "layers.0.experts.0.down_proj.scale": ("U8", (64, 8)),
        "layers.0.self_attn.q_proj.weight": ("F8_E4M3", (32, 32)),   # 1024 B
        "layers.0.self_attn.q_proj.weight_scale_inv": ("F32", (1, 1)),  # 4 B
        "embed.weight": ("BF16", (100, 8)),                         # 1600 B
    })
    stats = {
        "layers.0.experts.0.down_proj": {
            "n_params": 64 * 256, "in_features": 256, "out_features": 64},
        "layers.0.self_attn.q_proj": {
            "n_params": 32 * 32, "in_features": 32, "out_features": 32},
    }
    reencoded = list(stats)
    return stats, reencoded


def test_source_tensor_bytes_manifest_charges_actual_spans(tmp_path):
    _mixed_source_checkpoint(tmp_path)
    manifest = fp.source_tensor_bytes_manifest(str(tmp_path))
    assert manifest["layers.0.experts.0.down_proj"] == 8192 + 512
    assert manifest["layers.0.self_attn.q_proj"] == 1024 + 4
    assert manifest["embed"] == 1600


def test_manifest_floor_non_negative_where_regime_path_goes_negative(tmp_path):
    stats, reencoded = _mixed_source_checkpoint(tmp_path)
    total, by_dtype = fp.source_checkpoint_bytes(str(tmp_path))
    regime = fp.source_regime(by_dtype)
    assert regime == "fp8"

    # Regime-wide accounting: the packed expert's LOGICAL params are charged
    # 1 B each -> more bytes removed than the tensor (or checkpoint) holds.
    regime_removed = {
        q: fp.reencoded_source_bytes_for_shape(
            (stats[q]["out_features"], stats[q]["in_features"]), regime)
        for q in reencoded
    }
    regime_floor = total - sum(regime_removed.values())
    assert regime_floor < 0
    with pytest.raises(ValueError, match="negative non-quantizable floor"):
        fp.check_floor_non_negative(
            regime_floor, total, regime_removed, context="unit")

    # Manifest accounting: actual header spans can never exceed the total.
    manifest = fp.source_tensor_bytes_manifest(str(tmp_path))
    manifest_floor = total - sum(manifest[q] for q in reencoded)
    assert manifest_floor == 1600  # exactly the BF16 floor tensor
    fp.check_floor_non_negative(
        manifest_floor, total, {q: manifest[q] for q in reencoded},
        context="unit")  # does not raise


def test_floor_bytes_for_model_uses_manifest_on_mixed_source(tmp_path):
    stats, reencoded = _mixed_source_checkpoint(tmp_path)
    info = fp.floor_bytes_for_model(str(tmp_path), reencoded, stats)
    assert info["regime"] == "fp8"
    assert info["reencoded_source_bytes"] == (8192 + 512) + (1024 + 4)
    assert info["floor_bytes"] == 1600
