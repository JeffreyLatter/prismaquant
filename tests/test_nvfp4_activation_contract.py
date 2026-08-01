from __future__ import annotations

import json
import re

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant.cb_export_config import build_cb_scheme, build_quant_config
from prismaquant.nvfp4_activation_contract import (
    FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
    LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
    NVFP4_ACTIVATION_CONTRACT_KEY,
    NVFP4_ACTIVATION_CONTRACT_SCHEMA,
    NVFP4_ACTIVATION_EXECUTION,
    NVFP4_INPUT_GLOBAL_SCALE_SUFFIX,
    UNCALIBRATED_INPUT_GLOBAL_SCALE,
    build_execution_contract,
    calibrated_input_global_scales,
    fused_dense_group,
    fused_sibling_group_key,
    group_fused_sibling_targets,
    input_global_scale_from_max_abs,
    nvfp4_activation_qdq_served,
    resolve_input_global_scale_value,
    select_mse_grid_input_global_scale,
    target_values_sha256,
    unify_fused_sibling_input_global_scales,
    unify_fused_sibling_max_abs,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_serialization_context_stamp,
    cb_tensor_payload_breakdown,
    cb_tensor_serialization_stamp,
)


class _FusedProfile:
    @staticmethod
    def fused_sibling_group(name):
        if name in {"layer.q_proj", "layer.k_proj", "layer.v_proj"}:
            return "layer.qkv"
        return None


def _write_activation(cache_dir, name, inputs, row_indices=None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
    blob = {"name": name, "inputs": inputs}
    if row_indices is not None:
        blob["row_indices"] = row_indices
    torch.save(blob, cache_dir / filename)


def test_formula_policies_are_explicit_and_f32_rounded():
    legacy = input_global_scale_from_max_abs(
        3.0,
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    )
    full = input_global_scale_from_max_abs(
        3.0,
        policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
    )
    assert legacy == 2.0
    assert full == 896.0
    with pytest.raises(ValueError, match="activation samples"):
        input_global_scale_from_max_abs(
            3.0,
            policy=MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
        )
    with pytest.raises(ValueError, match="finite and > 0"):
        input_global_scale_from_max_abs(
            0.0,
            policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        )


def test_uncalibrated_fallback_is_explicit_and_legacy_only():
    assert input_global_scale_from_max_abs(
        0.0,
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
        nonpositive_fallback=UNCALIBRATED_INPUT_GLOBAL_SCALE,
    ) == 1.0
    with pytest.raises(ValueError, match="finite and > 0"):
        input_global_scale_from_max_abs(
            float("nan"),
            policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
            nonpositive_fallback=UNCALIBRATED_INPUT_GLOBAL_SCALE,
        )

    with pytest.raises(ValueError, match="no calibrated"):
        resolve_input_global_scale_value(target="layer.q_proj")
    assert resolve_input_global_scale_value(
        target="layer.q_proj",
        allow_uncalibrated_fallback=True,
    ) == UNCALIBRATED_INPUT_GLOBAL_SCALE
    assert resolve_input_global_scale_value(
        3.0,
        target="layer.q_proj",
        calibrated_scales={"layer.q_proj": 2.0},
        allow_uncalibrated_fallback=True,
    ) == 3.0
    assert resolve_input_global_scale_value(
        target="layer.q_proj",
        calibrated_scales={"layer.q_proj": 2.0},
        allow_uncalibrated_fallback=True,
    ) == 2.0


def test_one_grouping_primitive_drives_max_and_reciprocal_joins():
    q = "model.layers.2.self_attn.q_proj"
    k = "model.layers.2.self_attn.k_proj"
    down = "model.layers.2.mlp.down_proj"

    fallback = fused_dense_group(q)
    assert fallback == (
        "model.layers.2",
        ("q_proj", "k_proj", "v_proj"),
    )
    assert fused_sibling_group_key(q) == fused_sibling_group_key(k)
    groups = group_fused_sibling_targets([q, k, down])
    assert sorted(map(tuple, groups.values())) == sorted([(q, k), (down,)])

    max_abs = unify_fused_sibling_max_abs({q: 1.0, k: 4.0, down: 3.0})
    scales = unify_fused_sibling_input_global_scales(
        {q: 0.5, k: 0.25, down: 0.75}
    )
    assert max_abs == {q: 4.0, k: 4.0, down: 3.0}
    assert scales == {q: 0.25, k: 0.25, down: 0.75}


def test_grouping_profile_errors_are_strict_unless_legacy_opts_in():
    class BrokenProfile:
        @staticmethod
        def fused_sibling_group(_name):
            raise RuntimeError("profile unavailable")

    name = "model.layers.0.self_attn.q_proj"
    with pytest.raises(RuntimeError, match="profile unavailable"):
        fused_sibling_group_key(name, profile=BrokenProfile())
    assert fused_sibling_group_key(
        name,
        profile=BrokenProfile(),
        tolerate_profile_errors=True,
    ) == "model.layers.0::__fused__:q_proj,k_proj,v_proj"


def test_profile_leaf_mapping_and_direct_group_share_key_api():
    class LeafProfile:
        @staticmethod
        def fused_sibling_group(_name):
            return None

        @staticmethod
        def fused_sibling_leaf_mapping():
            return {"ab_proj": ("a_proj", "b_proj")}

    profile = LeafProfile()
    assert fused_sibling_group_key(
        "layer.a_proj", profile=profile
    ) == fused_sibling_group_key("layer.b_proj", profile=profile)


def test_e2m1_midpoints_use_encoded_index_even_rne():
    # Include 6 so the stored group scale is exactly 1 at G=1.
    values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
    x = torch.tensor(values + [-value for value in values]).reshape(1, 16)
    expected_positive = torch.tensor(
        [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]
    )
    expected = torch.cat((expected_positive, -expected_positive)).reshape(1, 16)
    torch.testing.assert_close(
        nvfp4_activation_qdq_served(x, 1.0),
        expected,
        rtol=0,
        atol=0,
    )


def test_ue4m3_scale_underflow_has_no_minimum_clamp():
    midpoint_to_first_subnormal = 6.0 * (2.0 ** -10)
    at_tie = torch.full((1, 16), midpoint_to_first_subnormal)
    just_above = torch.full(
        (1, 16),
        torch.nextafter(
            torch.tensor(midpoint_to_first_subnormal),
            torch.tensor(float("inf")),
        ).item(),
    )
    assert torch.count_nonzero(nvfp4_activation_qdq_served(at_tie, 1.0)) == 0
    assert torch.count_nonzero(
        nvfp4_activation_qdq_served(just_above, 1.0)
    ) == 16


def test_static_qdq_is_chunk_independent():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(17, 32, generator=generator) * 0.7
    whole = nvfp4_activation_qdq_served(x, 128.0)
    chunked = torch.cat([
        nvfp4_activation_qdq_served(x[:3], 128.0),
        nvfp4_activation_qdq_served(x[3:11], 128.0),
        nvfp4_activation_qdq_served(x[11:], 128.0),
    ])
    assert torch.equal(whole, chunked)


def test_mse_grid_contains_both_formula_endpoints():
    generator = torch.Generator().manual_seed(11)
    sample = torch.randn(23, 32, generator=generator)
    max_abs = float(sample.abs().max())
    legacy = input_global_scale_from_max_abs(
        max_abs, policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY
    )
    full = input_global_scale_from_max_abs(
        max_abs, policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
    )
    selected = select_mse_grid_input_global_scale([sample])

    def mse(scale):
        return float((nvfp4_activation_qdq_served(sample, scale) - sample)
                     .square().mean())

    assert mse(selected) <= min(mse(legacy), mse(full)) + 1e-12


@pytest.mark.parametrize(
    "policy",
    [FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY, MSE_GRID_INPUT_GLOBAL_SCALE_POLICY],
)
def test_fused_siblings_fit_union_and_emit_identical_f32(tmp_path, policy):
    cache = tmp_path / "act"
    q = torch.linspace(-1.0, 1.0, 64).reshape(2, 32)
    k = torch.linspace(-4.0, 4.0, 96).reshape(3, 32)
    # Deliberately different reservoirs/row identities: runtime still merges.
    _write_activation(cache, "layer.q_proj", q, torch.tensor([1, 9]))
    _write_activation(cache, "layer.k_proj", k, torch.tensor([2, 3, 7]))
    scales = calibrated_input_global_scales(
        ["layer.q_proj", "layer.k_proj"],
        activation_cache_dir=cache,
        policy=policy,
        profile=_FusedProfile(),
    )
    assert scales["layer.q_proj"] == scales["layer.k_proj"]
    if policy == FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY:
        assert scales["layer.q_proj"] == input_global_scale_from_max_abs(
            4.0,
            policy=policy,
        )
    else:
        assert scales["layer.q_proj"] == select_mse_grid_input_global_scale(
            [q, k]
        )


def test_missing_calibration_fails_closed(tmp_path):
    cache = tmp_path / "act"
    _write_activation(cache, "layer.q_proj", torch.ones(2, 32))
    with pytest.raises(ValueError, match="no calibrated input"):
        calibrated_input_global_scales(
            ["layer.q_proj", "layer.k_proj"],
            activation_cache_dir=cache,
            policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
            profile=_FusedProfile(),
        )


def test_contract_digest_framing_has_pinned_vector():
    assert target_values_sha256(
        {"a": 1.0, "b": 2.0},
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    ) == "5207c30737409ae6d16586f1f169efc8f56948bee51031e1610683f0fee08d0f"


def test_contract_uses_the_canonical_tensor_suffix_api():
    record, _ = build_execution_contract(
        {"layer.q_proj": 2.0},
        policy=LEGACY_INPUT_GLOBAL_SCALE_POLICY,
    )
    assert record["tensor_suffix"] == NVFP4_INPUT_GLOBAL_SCALE_SUFFIX


def test_accounting_is_keyed_to_static_contract_variant():
    static = CBSerializationContext.production()
    old = CBSerializationContext(
        scale_coding="two_tier",
        codebook_source="lattice",
    )
    fp4_static = cb_tensor_payload_breakdown(
        "NVFP4_CB_K16", (8, 256), qname="w", context=static
    )
    fp4_old = cb_tensor_payload_breakdown(
        "NVFP4_CB_K16", (8, 256), qname="w", context=old
    )
    fp8_static = cb_tensor_payload_breakdown(
        "FP8_CB_K36", (8, 256), qname="w", context=static
    )
    assert fp4_static["input_global_scale_bytes"] == 4
    assert fp4_static["tensor_payload_bytes"] == (
        fp4_old["tensor_payload_bytes"] + 4
    )
    assert fp8_static["input_global_scale_bytes"] == 0


def test_config_has_one_top_level_contract_and_fp4_only_reference():
    codebooks = {
        ("lattice", "NVFP4_CB_K12"): (
            torch.zeros(64, 4), torch.zeros(64, 4)
        ),
        ("lattice", "FP8_CB_K28"): tuple(
            torch.zeros(128, 2) for _ in range(4)
        ),
    }
    record, _ = build_execution_contract(
        {"fp4": 8.0},
        policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
    )
    config = build_quant_config(
        assignment={"fp4": "NVFP4_CB_K12", "fp8": "FP8_CB_K28"},
        cb_targets={
            "fp4": ("fp4", "product", 12),
            "fp8": ("fp8", "product", 28),
        },
        source_targets=[],
        stock_targets={},
        by_group={
            ("lattice", "NVFP4_CB_K12"): ["fp4"],
            ("lattice", "FP8_CB_K28"): ["fp8"],
        },
        codebooks=codebooks,
        col_weights={},
        codebook_tensors_by_name={},
        ignore=[],
        codebook_file=None,
        scale_coding="two_tier",
        codebook_source="lattice",
        serialized_payload_summary={"total_bytes": 0},
        serialization_context=CBSerializationContext.production(),
        cb_render_identity=None,
        activation_execution_contract=record,
        git_commit="test",
    )
    assert config["execution_contracts"] == {
        NVFP4_ACTIVATION_CONTRACT_KEY: record
    }
    assert "nvfp4_activation_contract" not in config["provenance"]
    schemes = {
        group["scheme"]["grid"]: group["scheme"]
        for group in config["config_groups"].values()
    }
    assert schemes["fp4"]["activation_contract"] == (
        NVFP4_ACTIVATION_CONTRACT_KEY
    )
    assert "activation_contract" not in schemes["fp8"]


def _production_layer_config(
    path,
    qname,
    weight,
    col_weights,
    *,
    extra_assignment=None,
    context=None,
):
    from prismaquant.production_weight_cache import (
        bind_cb_render_identity_source_weights,
        build_production_cache_cb_render_identity,
    )

    context = context or CBSerializationContext.production()
    identity = build_production_cache_cb_render_identity(
        {qname: "NVFP4_CB_K16"},
        cb_serialization_context=context,
        col_weights={qname: col_weights},
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    identity = bind_cb_render_identity_source_weights(
        identity,
        {qname: weight},
    )
    stamp = cb_serialization_context_stamp(
        context,
        formats=["NVFP4_CB_K16"],
    )
    payload = {
        qname: {
            "data_type": "nvfp4_cb",
            "cb_k": 16,
            "cb_serialized_identity": cb_tensor_serialization_stamp(
                "NVFP4_CB_K16",
                tuple(weight.shape),
                qname=qname,
                context=context,
            ),
        },
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "cb_serialized_payload": stamp,
            "cb_render_identity": identity,
        },
    }
    payload.update(extra_assignment or {})
    path.write_text(json.dumps(payload))


def test_resident_and_streaming_export_same_static_scalar_and_contract(tmp_path):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb
    from prismaquant.export_nvfp4_cb_streaming import export_nvfp4_cb_streaming

    source = tmp_path / "source"
    source.mkdir()
    qname = "model.layers.0.self_attn.o_proj"
    stock_name = "model.layers.0.mlp.down_proj"
    generator = torch.Generator().manual_seed(123)
    weight = (torch.randn(8, 256, generator=generator) * 0.2).to(
        torch.bfloat16
    )
    stock_weight = (
        torch.randn(8, 256, generator=generator) * 0.2
    ).to(torch.bfloat16)
    save_file(
        {
            qname + ".weight": weight,
            stock_name + ".weight": stock_weight,
        },
        str(source / "model.safetensors"),
    )
    (source / "config.json").write_text(json.dumps({
        "architectures": ["ContractTiny"],
        "hidden_size": 256,
    }))
    col_weights = torch.linspace(0.5, 1.5, 256)
    assignment = tmp_path / "assignment.json"
    _production_layer_config(
        assignment,
        qname,
        weight,
        col_weights,
        extra_assignment={stock_name: "NVFP4"},
    )
    activation_cache = tmp_path / "act"
    _write_activation(
        activation_cache,
        qname,
        torch.randn(13, 256, generator=generator) * 0.4,
    )
    _write_activation(
        activation_cache,
        stock_name,
        torch.randn(11, 256, generator=generator) * 0.6,
    )

    resident = tmp_path / "resident"
    streaming = tmp_path / "streaming"
    common = dict(
        activation_cache_dir=activation_cache,
        activation_scale_policy=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY,
        device="cpu",
    )
    export_nvfp4_cb(
        source,
        assignment,
        resident,
        {qname: col_weights},
        **common,
    )
    export_nvfp4_cb_streaming(
        source,
        assignment,
        streaming,
        {qname: col_weights},
        **common,
    )

    resident_tensors = load_file(str(resident / "model.safetensors"))
    streaming_tensors = load_file(str(streaming / "model.safetensors"))
    for target in (qname, stock_name):
        scalar_name = target + ".input_global_scale"
        assert resident_tensors[scalar_name].dtype == torch.float32
        assert tuple(resident_tensors[scalar_name].shape) == (1,)
        assert torch.equal(
            resident_tensors[scalar_name],
            streaming_tensors[scalar_name],
        )
    resident_config = json.loads((resident / "quant_config.json").read_text())
    streaming_config = json.loads((streaming / "quant_config.json").read_text())
    assert resident_config["execution_contracts"] == (
        streaming_config["execution_contracts"]
    )
    record = resident_config["execution_contracts"][
        NVFP4_ACTIVATION_CONTRACT_KEY
    ]
    assert record["schema"] == NVFP4_ACTIVATION_CONTRACT_SCHEMA
    assert record["contract"] == NVFP4_ACTIVATION_EXECUTION
    assert record["target_count"] == 2
    assert record["target_names"] == sorted((qname, stock_name))
    emitted_scalar_targets = sorted(
        name.removesuffix(".input_global_scale")
        for name in resident_tensors
        if name.endswith(".input_global_scale")
    )
    assert emitted_scalar_targets == record["target_names"]
    assert resident_config["provenance"]["serialized_payload"][
        "input_global_scale_bytes"
    ] == 4


def test_research_export_omits_static_contract_and_scalar(tmp_path):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    source = tmp_path / "source"
    source.mkdir()
    qname = "model.layers.0.self_attn.o_proj"
    weight = torch.randn(4, 256).to(torch.bfloat16)
    save_file({qname + ".weight": weight}, str(source / "model.safetensors"))
    (source / "config.json").write_text("{}")
    assignment = tmp_path / "assignment.json"
    assignment.write_text(json.dumps({qname: "NVFP4_CB_K16"}))
    out = tmp_path / "out"
    export_nvfp4_cb(
        source,
        assignment,
        out,
        {qname: torch.ones(256)},
        device="cpu",
        allow_unstamped_research=True,
    )
    config = json.loads((out / "quant_config.json").read_text())
    tensors = load_file(str(out / "model.safetensors"))
    assert "execution_contracts" not in config
    assert qname + ".input_global_scale" not in tensors
    group = next(iter(config["config_groups"].values()))
    assert "activation_contract" not in group["scheme"]


def test_v2_stamped_export_remains_fused_ineligible(tmp_path):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    source = tmp_path / "source"
    source.mkdir()
    qname = "model.layers.0.self_attn.o_proj"
    weight = torch.randn(4, 256).to(torch.bfloat16)
    save_file({qname + ".weight": weight}, str(source / "model.safetensors"))
    (source / "config.json").write_text("{}")
    assignment = tmp_path / "assignment.json"
    old_context = CBSerializationContext(
        scale_coding="two_tier",
        codebook_source="lattice",
    )
    _production_layer_config(
        assignment,
        qname,
        weight,
        torch.ones(256),
        context=old_context,
    )
    out = tmp_path / "out"
    export_nvfp4_cb(
        source,
        assignment,
        out,
        {qname: torch.ones(256)},
        device="cpu",
    )
    config = json.loads((out / "quant_config.json").read_text())
    tensors = load_file(str(out / "model.safetensors"))
    assert config["provenance"]["serialized_payload"]["schema"].endswith(
        ".v2"
    )
    assert "execution_contracts" not in config
    assert qname + ".input_global_scale" not in tensors
    group = next(iter(config["config_groups"].values()))
    assert "activation_contract" not in group["scheme"]
