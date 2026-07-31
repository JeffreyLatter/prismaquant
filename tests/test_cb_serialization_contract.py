"""Cross-stage tests for the authoritative CB serialization contract."""

from __future__ import annotations

import itertools
import json
import math
import pickle
import struct

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.allocator import (
    _serialized_format_rates,
    _sort_specs_by_serialized_rate,
)
from prismaquant.kl_measurement import assignment_bit_total
from prismaquant.measure_quant_cost import _cb_cost_quantize_dequantize
from prismaquant.incremental_measure_quant_cost import merge_cost_pickles
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    _safetensors_data_spans,
    cb_assignment_payload_breakdown,
    cb_assignment_serialization_stamps,
    cb_export_artifact_inventory,
    cb_serialization_context_stamp,
    finalize_cb_export_artifact_inventory,
    validate_cb_cost_provenance,
)
from prismaquant.validate_assignments_kl import _assignment_bpp_details


@pytest.mark.parametrize("mode,k", [("product", 12), ("signed", 13)])
@pytest.mark.parametrize("shape", [(2, 256), (2, 2, 256)])
@pytest.mark.parametrize("scale_coding", ["v1", "two_tier"])
def test_cost_qdq_matches_serialized_pack_unpack(
    monkeypatch, mode, k, shape, scale_coding,
):
    """Cost reconstruction is exactly what the selected writer serializes."""
    monkeypatch.setenv("CB_SCALE_CODING", scale_coding)
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "fast")
    torch.manual_seed(1000 + k)
    weight = torch.randn(*shape) * 0.2
    col_weights = torch.rand(*shape[:-2], 1, shape[-1]) + 0.05
    if len(shape) == 2:
        col_weights = col_weights.reshape(-1)
    spec = fr.get_format(f"NVFP4_CB_{'S' if mode == 'signed' else 'K'}{k}")

    measured = _cb_cost_quantize_dequantize(
        spec,
        weight.clone(),
        col_weights=col_weights,
    )
    packed, fields = cb.nvfp4_cb_pack(
        weight.clone(),
        k,
        grid="fp4",
        mode=mode,
        col_weights=col_weights,
        scale_coding=scale_coding,
        encode_tier="fast",
    )
    unpacked = cb.nvfp4_cb_unpack(
        packed,
        k,
        "fp4",
        mode,
        tuple(weight.shape),
        codebook=fields["codebook"],
        scale_coding=scale_coding,
    )
    serialized = cb.nvfp4_cb_reconstruct(
        unpacked,
        k,
        grid="fp4",
        mode=mode,
    ).to(weight.dtype)
    assert torch.equal(measured, serialized)


@pytest.mark.parametrize("shape", [(2, 256), (2, 2, 256)])
def test_fp8_cost_qdq_matches_serialized_pack_unpack(monkeypatch, shape):
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "fast")
    torch.manual_seed(1044)
    weight = torch.randn(*shape) * 0.2
    col_weights = torch.rand(*shape[:-2], 1, shape[-1]) + 0.05
    if len(shape) == 2:
        col_weights = col_weights.reshape(-1)
    spec = fr.get_format("FP8_CB_K36")

    measured = _cb_cost_quantize_dequantize(
        spec, weight.clone(), col_weights=col_weights
    )
    packed, fields = cb.nvfp4_cb_pack(
        weight.clone(),
        36,
        grid="fp8",
        mode="product",
        col_weights=col_weights,
        encode_tier="fast",
    )
    unpacked = cb.nvfp4_cb_unpack(
        packed,
        36,
        "fp8",
        "product",
        tuple(weight.shape),
        codebook=fields["codebook"],
        scales=fields["scales"],
    )
    serialized = cb.nvfp4_cb_reconstruct(
        unpacked, 36, grid="fp8", mode="product"
    ).to(weight.dtype)
    assert torch.equal(measured, serialized)


def test_cost_cache_identity_missing_or_mismatched_fails_closed():
    production = CBSerializationContext.production(codebook_source="learned")
    formats = ["NVFP4_CB_K16", "BF16"]
    with pytest.raises(ValueError, match="no serialized-payload identity"):
        validate_cb_cost_provenance(
            {"provenance": {}},
            formats,
            context=production,
            where="unit cost",
        )
    stale = {
        "provenance": {
            "cb_serialized_payload": cb_serialization_context_stamp(
                CBSerializationContext.legacy_v1(codebook_source="learned")
            )
        }
    }
    with pytest.raises(ValueError, match="differs from allocator recipe"):
        validate_cb_cost_provenance(
            stale,
            formats,
            context=production,
            where="unit cost",
        )


def test_incremental_merge_rejects_a_stale_cb_shard(tmp_path, monkeypatch):
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    fresh = tmp_path / "fresh.pkl"
    stale = tmp_path / "stale.pkl"
    output = tmp_path / "merged.pkl"
    common = {"formats": ["NVFP4_CB_K16"], "meta": {}}
    fresh.write_bytes(pickle.dumps({
        **common,
        "costs": {"layer.0": {}},
        "provenance": {
            "cb_serialized_payload": cb_serialization_context_stamp(
                CBSerializationContext.production()
            ),
        },
    }))
    stale.write_bytes(pickle.dumps({
        **common,
        "costs": {"layer.1": {}},
        "provenance": {
            "cb_serialized_payload": cb_serialization_context_stamp(
                CBSerializationContext.legacy_v1()
            ),
        },
    }))
    with pytest.raises(ValueError, match="differs from allocator recipe"):
        merge_cost_pickles([fresh, stale], output)
    assert not output.exists()


def _stats(shape):
    return {
        "n_params": math.prod(shape),
        "out_features": shape[-2],
        "in_features": shape[-1],
        **({"num_experts": shape[0]} if len(shape) == 3 else {}),
    }


@pytest.mark.parametrize("scale_coding", ["v1", "two_tier"])
def test_assignment_bits_prices_fp4_layout_and_shared_role_once(scale_coding):
    names = ("model.layers.0.q_proj", "model.layers.1.q_proj")
    shape = (4, 512)
    fmt = "NVFP4_CB_K16"
    assignment = {name: fmt for name in names}
    shapes = {name: shape for name in names}
    stats = {name: _stats(shape) for name in names}
    context = CBSerializationContext(
        scale_coding=scale_coding,
        codebook_source="learned",
    )
    stamps = cb_assignment_serialization_stamps(
        assignment, shapes, context=context
    )
    breakdown = cb_assignment_payload_breakdown(
        assignment, shapes, context=context
    )
    assert len(breakdown["sidecars"]) == 1
    bits = assignment_bit_total(
        stats,
        assignment,
        {fmt: fr.get_format(fmt)},
        cb_serialization_context=context,
        cb_serialization_stamps=stamps,
    )
    assert bits == 8 * breakdown["total_bytes"]
    bytes_per_superblock = 4 * 16 + (16 if scale_coding == "v1" else 9)
    assert breakdown["tensor_payload_bytes"] == 2 * 4 * 2 * bytes_per_superblock


@pytest.mark.parametrize("in_features", [512, 5120])
def test_assignment_bits_prices_fp8_row_scales_at_real_shape(in_features):
    shape = (3, in_features)
    fmt = "FP8_CB_K36"
    assignment = {"model.layers.0.q_proj": fmt}
    shapes = {name: shape for name in assignment}
    stats = {name: _stats(shape) for name in assignment}
    context = CBSerializationContext.production()
    stamps = cb_assignment_serialization_stamps(
        assignment, shapes, context=context
    )
    breakdown = cb_assignment_payload_breakdown(
        assignment, shapes, context=context
    )
    assert breakdown["fp8_row_scale_bytes"] == 4 * shape[0]
    assert assignment_bit_total(
        stats,
        assignment,
        {fmt: fr.get_format(fmt)},
        cb_serialization_context=context,
        cb_serialization_stamps=stamps,
    ) == 8 * breakdown["total_bytes"]


def test_assignment_bpp_details_uses_exact_cb_assignment_payload():
    fmt = "FP8_CB_K36"
    assignment = {
        "model.layers.0.q_proj": fmt,
        "model.layers.1.q_proj": fmt,
    }
    shapes = {name: (3, 512) for name in assignment}
    stats = {name: _stats(shape) for name, shape in shapes.items()}
    context = CBSerializationContext.production(codebook_source="learned")
    stamps = cb_assignment_serialization_stamps(
        assignment, shapes, context=context
    )
    payload = cb_assignment_payload_breakdown(
        assignment, shapes, context=context
    )
    details = _assignment_bpp_details(
        stats,
        assignment,
        {fmt: fr.get_format(fmt)},
        cb_serialization_context=context,
        cb_serialization_stamps=stamps,
        where="direct bpp contract test",
    )
    assert details["bpp"] == pytest.approx(
        8 * payload["total_bytes"]
        / sum(item["n_params"] for item in stats.values())
    )


def test_assignment_bits_requires_every_matching_per_layer_stamp():
    fmt = "NVFP4_CB_K16"
    shape = (4, 256)
    assignment = {"model.layers.0.q_proj": fmt}
    stats = {name: _stats(shape) for name in assignment}
    context = CBSerializationContext.production()
    specs = {fmt: fr.get_format(fmt)}
    with pytest.raises(ValueError, match="missing per-layer"):
        assignment_bit_total(
            stats,
            assignment,
            specs,
            cb_serialization_context=context,
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        assignment_bit_total(
            stats,
            assignment,
            specs,
            cb_serialization_context=context,
            cb_serialization_stamps={next(iter(assignment)): "stale"},
        )


def test_serialized_format_order_is_input_order_independent():
    stats = {
        f"model.layers.{index}.q_proj": _stats((512, 512))
        for index in range(64)
    }
    names = ["NVFP4", "NVFP4_CB_K24", "FP8_CB_K36", "BF16"]
    context = CBSerializationContext.production()
    expected = None
    for permutation in itertools.permutations(names):
        ordered, rates = _sort_specs_by_serialized_rate(
            [fr.get_format(name) for name in permutation],
            stats,
            context,
        )
        observed = [spec.name for spec in ordered]
        expected = observed if expected is None else expected
        assert observed == expected
        # Production K24 uses 4k+9, not FormatSpec's stale 4k+16 rate.
        assert rates["NVFP4_CB_K24"] < fr.get_format(
            "NVFP4_CB_K24"
        ).effective_bits


def _write_safetensors(path, entries, *, data_bytes=None):
    header = {
        name: {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": list(offsets),
        }
        for name, (dtype, shape, offsets) in entries.items()
    }
    raw = json.dumps(header).encode()
    extent = max((offsets[1] for _dtype, _shape, offsets in entries.values()),
                 default=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<Q", len(raw)) + raw
        + (b"\0" * extent if data_bytes is None else data_bytes)
    )


@pytest.mark.parametrize(
    "entries,match",
    [
        (
            {
                "a": ("U8", (2,), (0, 2)),
                "b": ("U8", (2,), (3, 5)),
            },
            "leaves a gap",
        ),
        ({"a": ("F16", (2,), (0, 2))}, "requires 4B"),
    ],
)
def test_safetensors_span_parser_rejects_malformed_layout(tmp_path, entries, match):
    path = tmp_path / "bad.safetensors"
    _write_safetensors(path, entries)
    with pytest.raises(AssertionError, match=match):
        _safetensors_data_spans(path)


def test_recursive_inventory_reaches_a_stable_quant_config_fixed_point(tmp_path):
    context = CBSerializationContext.production()
    assignment = {"layer.q_proj": "NVFP4_CB_K16"}
    shapes = {"layer.q_proj": (1, 256)}
    payload = cb_assignment_payload_breakdown(
        assignment, shapes, context=context
    )
    tensor_bytes = int(payload["tensor_payload_bytes"])
    sidecar_bytes = int(payload["codebook_sidecar_bytes"])
    _write_safetensors(
        tmp_path / "nested" / "model.safetensors",
        {"layer.q_proj.cb_qweight": ("U8", (tensor_bytes,), (0, tensor_bytes))},
    )
    _write_safetensors(
        tmp_path / "cb_codebooks.pqcb",
        {"tables": ("F16", (sidecar_bytes // 2,), (0, sidecar_bytes))},
    )
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "extra.txt").write_text("abc")
    config = {"provenance": {}}
    inventory = finalize_cb_export_artifact_inventory(
        tmp_path,
        config,
        serialized_payload=payload,
        cb_tensor_names=["layer.q_proj.cb_qweight"],
        codebook_file="cb_codebooks.pqcb",
    )
    assert "nested/model.safetensors" in inventory["file_bytes"]
    assert "tokenizer/extra.txt" in inventory["file_bytes"]
    on_disk = json.loads((tmp_path / "quant_config.json").read_text())
    assert on_disk["provenance"]["artifact_inventory"] == inventory
    assert cb_export_artifact_inventory(
        tmp_path,
        serialized_payload=payload,
        cb_tensor_names=["layer.q_proj.cb_qweight"],
        codebook_file="cb_codebooks.pqcb",
    ) == inventory


def test_fp8_rate_helper_reflects_in_features_row_scale_amortization():
    spec = fr.get_format("FP8_CB_K36")
    context = CBSerializationContext.production()
    narrow = _serialized_format_rates(
        [spec], {"w": _stats((512, 512))}, context
    )[spec.name]
    wide = _serialized_format_rates(
        [spec], {"w": _stats((512, 5120))}, context
    )[spec.name]
    assert narrow > wide
