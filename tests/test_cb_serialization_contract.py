"""Cross-stage tests for the authoritative CB serialization contract."""

from __future__ import annotations

import hashlib
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
    cb_tensor_serialization_stamp,
    codebook_subtable_shapes,
    cb_export_artifact_inventory,
    cb_serialization_context_stamp,
    enforce_whole_artifact_budget,
    finalize_cb_export_artifact_inventory,
    load_cb_codebook_digest_manifest,
    whole_artifact_budget_stamp,
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
    digests = {"materialized": "a" * 64}
    production = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests=digests,
    )
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
                CBSerializationContext.legacy_v1(
                    codebook_source="learned",
                    codebook_content_digests=digests,
                )
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


def test_learned_context_validation_allows_unused_menu_digest_superset():
    selected = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests={"selected": "a" * 64},
    )
    menu = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests={
            "selected": "a" * 64,
            "unused_candidate": "b" * 64,
        },
    )
    stamp = cb_serialization_context_stamp(menu)
    from prismaquant.nvfp4_cb_footprint import (
        validate_cb_serialization_context_stamp,
    )

    validate_cb_serialization_context_stamp(stamp, selected, where="export")
    wrong = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests={"selected": "c" * 64},
    )
    with pytest.raises(ValueError, match="mismatched"):
        validate_cb_serialization_context_stamp(stamp, wrong, where="export")


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


def _learned_digests(fmt, *roles):
    count = len(codebook_subtable_shapes(fmt))
    return {
        f"cb_codebook.{role}.{fmt}.sub{index}": f"{index + 1:064x}"
        for role in roles
        for index in range(count)
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
        codebook_content_digests=_learned_digests(fmt, "q_proj"),
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


def test_assignment_bits_includes_native_nvfp4_global_scale_tensors():
    name = "model.layers.0.self_attn.o_proj"
    shape = (4, 256)
    spec = fr.get_format("NVFP4")
    assert assignment_bit_total(
        {name: _stats(shape)},
        {name: "NVFP4"},
        {"NVFP4": spec},
    ) == 8 * (spec.memory_bytes_for_shape(shape) + 8)


def test_assignment_bpp_details_uses_exact_cb_assignment_payload():
    fmt = "FP8_CB_K36"
    assignment = {
        "model.layers.0.q_proj": fmt,
        "model.layers.1.q_proj": fmt,
    }
    shapes = {name: (3, 512) for name in assignment}
    stats = {name: _stats(shape) for name, shape in shapes.items()}
    context = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests=_learned_digests(fmt, "q_proj"),
    )
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


def test_tensor_stamp_binds_shape_and_every_byte_component():
    context = CBSerializationContext.production()
    a = cb_tensor_serialization_stamp(
        "FP8_CB_K36", (3, 512), qname="layer.q_proj", context=context
    )
    b = cb_tensor_serialization_stamp(
        "FP8_CB_K36", (6, 256), qname="layer.q_proj", context=context
    )
    assert a != b, "equal n_params must not hide row/block/scale shape drift"
    parsed = json.loads(a)
    assert parsed["shape"] == [3, 512]
    assert parsed["output_rows"] == 3
    assert parsed["superblocks_per_row"] == 2
    assert parsed["fp8_row_scale_bytes"] == 12
    assert parsed["tensor_payload_bytes"] == (
        parsed["packed_weight_bytes"] + parsed["fp8_row_scale_bytes"]
    )


def test_learned_identity_requires_materialized_content_digests():
    context = CBSerializationContext.production(codebook_source="learned")
    with pytest.raises(ValueError, match="materialized SHA-256"):
        cb_assignment_payload_breakdown(
            {"layer.q_proj": "NVFP4_CB_K16"},
            {"layer.q_proj": (2, 256)},
            context=context,
        )
    with pytest.raises(ValueError, match="codebook_content_digests"):
        cb_serialization_context_stamp(context)


def test_codebook_digest_manifest_accepts_inline_json_and_rejects_duplicates():
    assert load_cb_codebook_digest_manifest(
        '{"sidecar":"' + ("a" * 64) + '"}', where="unit"
    ) == {"sidecar": "a" * 64}
    with pytest.raises(AssertionError, match="duplicate JSON object key"):
        load_cb_codebook_digest_manifest(
            '{"sidecar":"' + ("a" * 64) + '","sidecar":"'
            + ("b" * 64) + '"}',
            where="unit",
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


def _write_raw_safetensors(path, raw_header: str, data: bytes = b""):
    raw = raw_header.encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


@pytest.mark.parametrize(
    "dtype,shape,span",
    [
        ("F8_E8M0", (3,), 3),
        ("F4", (3,), 2),
        ("F4_E2M1", (4,), 2),
    ],
)
def test_safetensors_span_parser_understands_packed_float_widths(
    tmp_path, dtype, shape, span,
):
    path = tmp_path / "packed.safetensors"
    _write_safetensors(path, {"a": (dtype, shape, (0, span))})
    assert _safetensors_data_spans(path) == {"a": span}


@pytest.mark.parametrize(
    "raw_header,match",
    [
        (
            '{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
            '"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}',
            "duplicate JSON object key",
        ),
        ('{"__metadata__":{"bad":NaN}}', "non-finite JSON constant"),
    ],
)
def test_safetensors_span_parser_rejects_ambiguous_json(
    tmp_path, raw_header, match,
):
    path = tmp_path / "bad-json.safetensors"
    _write_raw_safetensors(path, raw_header, b"\0")
    with pytest.raises(AssertionError, match=match):
        _safetensors_data_spans(path)


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
        {
            ref: ("F16", tuple(shape), (offset, offset + math.prod(shape) * 2))
            for sidecar in payload["sidecars"]
            for ref, shape, offset in zip(
                sidecar["codebook_ref"],
                sidecar["subtable_shapes"],
                itertools.accumulate(
                    [0] + [math.prod(s) * 2 for s in sidecar["subtable_shapes"][:-1]]
                ),
            )
        },
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


def _materialize_minimal_cb_export(tmp_path, *, context=None):
    context = context or CBSerializationContext.production()
    assignment = {"layer.q_proj": "NVFP4_CB_K16"}
    payload = cb_assignment_payload_breakdown(
        assignment, {"layer.q_proj": (1, 256)}, context=context
    )
    tensor_bytes = int(payload["tensor_payload_bytes"])
    _write_safetensors(
        tmp_path / "model.safetensors",
        {"layer.q_proj.cb_qweight": (
            "U8", (tensor_bytes,), (0, tensor_bytes)
        )},
    )
    sidecar_entries = {}
    offset = 0
    for sidecar in payload["sidecars"]:
        for ref, shape in zip(
            sidecar["codebook_ref"], sidecar["subtable_shapes"]
        ):
            nbytes = math.prod(shape) * 2
            sidecar_entries[ref] = ("F16", tuple(shape), (offset, offset + nbytes))
            offset += nbytes
    _write_safetensors(tmp_path / "cb_codebooks.pqcb", sidecar_entries)
    return payload


def test_inventory_verifies_learned_codebook_content_digest(tmp_path):
    fmt = "NVFP4_CB_K16"
    refs = [f"cb_codebook.q_proj.{fmt}.sub{index}" for index in range(2)]
    digest = hashlib.sha256(b"\0" * (256 * 4 * 2)).hexdigest()
    context = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests={ref: digest for ref in refs},
    )
    payload = _materialize_minimal_cb_export(tmp_path, context=context)
    inventory = cb_export_artifact_inventory(
        tmp_path,
        serialized_payload=payload,
        cb_tensor_names=["layer.q_proj.cb_qweight"],
        codebook_file="cb_codebooks.pqcb",
        expected_model_files=["model.safetensors"],
    )
    assert inventory["cb_codebook_content_sha256"] == {
        ref: digest for ref in refs
    }

    codebook = tmp_path / "cb_codebooks.pqcb"
    raw = bytearray(codebook.read_bytes())
    raw[-1] ^= 1
    codebook.write_bytes(raw)
    with pytest.raises(AssertionError, match="differ from their content identity"):
        cb_export_artifact_inventory(
            tmp_path,
            serialized_payload=payload,
            cb_tensor_names=["layer.q_proj.cb_qweight"],
            codebook_file="cb_codebooks.pqcb",
            expected_model_files=["model.safetensors"],
        )


def test_inventory_rejects_stale_model_shards(tmp_path):
    payload = _materialize_minimal_cb_export(tmp_path)
    _write_safetensors(tmp_path / "stale-00002.safetensors", {})
    with pytest.raises(AssertionError, match="fresh export plan"):
        cb_export_artifact_inventory(
            tmp_path,
            serialized_payload=payload,
            cb_tensor_names=["layer.q_proj.cb_qweight"],
            codebook_file="cb_codebooks.pqcb",
            expected_model_files=["model.safetensors"],
        )


def test_inventory_rejects_stale_codebook_sidecars(tmp_path):
    payload = _materialize_minimal_cb_export(tmp_path)
    _write_safetensors(tmp_path / "stale-codebooks.pqcb", {})
    with pytest.raises(AssertionError, match="stale CB codebook sidecar"):
        cb_export_artifact_inventory(
            tmp_path,
            serialized_payload=payload,
            cb_tensor_names=["layer.q_proj.cb_qweight"],
            codebook_file="cb_codebooks.pqcb",
            expected_model_files=["model.safetensors"],
        )


def test_inventory_rejects_unexpected_cb_suffix_tensors(tmp_path):
    payload = _materialize_minimal_cb_export(tmp_path)
    tensor_bytes = int(payload["tensor_payload_bytes"])
    _write_safetensors(
        tmp_path / "model.safetensors",
        {
            "layer.q_proj.cb_qweight": (
                "U8", (tensor_bytes,), (0, tensor_bytes)
            ),
            "layer.q_proj.weight_scale": (
                "F32", (1,), (tensor_bytes, tensor_bytes + 4)
            ),
        },
    )
    with pytest.raises(AssertionError, match="unexpected/stale CB tensors"):
        cb_export_artifact_inventory(
            tmp_path,
            serialized_payload=payload,
            cb_tensor_names=["layer.q_proj.cb_qweight"],
            codebook_file="cb_codebooks.pqcb",
            expected_model_files=["model.safetensors"],
        )


def test_final_inventory_hard_fails_actual_recursive_size_over_budget(tmp_path):
    payload = _materialize_minimal_cb_export(tmp_path)
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(RuntimeError, match="exact recursive export size"):
        finalize_cb_export_artifact_inventory(
            tmp_path,
            {"provenance": {}},
            serialized_payload=payload,
            cb_tensor_names=["layer.q_proj.cb_qweight"],
            codebook_file="cb_codebooks.pqcb",
            expected_model_files=["model.safetensors"],
            whole_artifact_budget_bytes=1,
        )


def test_generic_export_budget_gate_measures_files_recursively(tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "nested").mkdir(parents=True)
    (artifact / "a.bin").write_bytes(b"a" * 7)
    (artifact / "nested" / "b.bin").write_bytes(b"b" * 5)
    payload = {
        "whole_artifact_budget": whole_artifact_budget_stamp(
            budget_bytes=12,
            selection_tensor_payload_bytes=8,
            selection_non_tensor_reserve_bytes=4,
        ),
    }
    attestation = enforce_whole_artifact_budget(
        artifact, payload, where="unit export"
    )
    assert attestation["actual_bytes"] == 12
    assert attestation["within_budget"]

    payload["whole_artifact_budget"] = whole_artifact_budget_stamp(
        budget_bytes=11,
        selection_tensor_payload_bytes=8,
        selection_non_tensor_reserve_bytes=3,
    )
    with pytest.raises(RuntimeError, match="exact completed artifact size"):
        enforce_whole_artifact_budget(artifact, payload, where="unit export")


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
