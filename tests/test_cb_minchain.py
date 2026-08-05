from __future__ import annotations

import pytest
import torch

from prismaquant.cb_minchain import (
    MINCHAIN_FLAG,
    MinChainInterpolationConfig,
    chain_identity,
    chain_identity_from_digest,
    epsilon_le,
    embed_predecessor,
    guarantee_accounting,
    interpolation_acceptance_v2,
    minchain_enabled,
    pchip_monotone,
    refine_one_entry,
    recipe_solution_digest,
    relative_epsilon,
    select_arm,
    select_reconstruction_slices,
    solution_digest,
)
from prismaquant.nvfp4_cb_formats import (
    nvfp4_cb_assemble_bytes,
    nvfp4_cb_fields,
    nvfp4_cb_pack,
    nvfp4_cb_reconstruct,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_fields_for_context,
    cb_serialization_context_from_env,
    cb_serialization_context_stamp,
    validate_cb_serialization_context_stamp,
)
from prismaquant.production_weight_cache import (
    bind_cb_render_identity_source_weights,
    build_production_cache_cb_render_identity,
    validate_cb_render_identity_metadata,
)
from prismaquant import format_registry as fr


def _fields() -> dict:
    torch.manual_seed(7)
    weight = torch.randn(4, 256)
    return nvfp4_cb_fields(
        weight, 28, grid="fp8", mode="product", scale_sweep=False
    )


def test_embed_requires_explicit_production_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(MINCHAIN_FLAG, raising=False)
    with pytest.raises(RuntimeError, match=MINCHAIN_FLAG):
        embed_predecessor(_fields(), 29)


def test_flag_defaults_off_and_rejects_ambiguous_values():
    assert minchain_enabled({}) is False
    assert minchain_enabled({MINCHAIN_FLAG: "1"}) is True
    with pytest.raises(ValueError, match=MINCHAIN_FLAG):
        minchain_enabled({MINCHAIN_FLAG: "perhaps"})


def test_embed_is_reconstruction_exact(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(MINCHAIN_FLAG, "1")
    predecessor = _fields()
    embedded = embed_predecessor(predecessor, 29)
    before = nvfp4_cb_reconstruct(
        predecessor, 28, grid="fp8", mode="product"
    )
    after = nvfp4_cb_reconstruct(
        embedded, 29, grid="fp8", mode="product"
    )
    assert torch.equal(before, after)
    assert torch.equal(embedded["scales"], predecessor["scales"])
    for old, new in zip(predecessor["codebook"], embedded["codebook"]):
        assert torch.equal(old, new[: old.shape[0]])


def test_digest_and_chain_identity_cover_arm_and_predecessor():
    fields = _fields()
    digest = solution_digest(fields)
    assert len(digest) == 64
    identity = chain_identity(
        winning_arm="embed", solution=fields, predecessor_digest=digest
    )
    assert identity["winning_arm"] == "embed"
    assert identity["predecessor_digest"] == digest
    with pytest.raises(ValueError, match="requires"):
        chain_identity(
            winning_arm="refine", solution=fields, predecessor_digest=None
        )
    recipe_digest = recipe_solution_digest({"qname": "x", "rung": 29})
    recipe_identity = chain_identity_from_digest(
        winning_arm="refine", solution_digest_value=recipe_digest,
        predecessor_digest=digest,
    )
    assert recipe_identity["digest_basis"] == "deterministic_content_gated_recipe"


def test_selection_is_deterministic_and_zero_tax():
    arm, error = select_arm({"free": 2.0, "embed": 1.0, "refine": 1.0})
    assert (arm, error) == ("embed", 1.0)
    arm, error = select_arm({"free": 0.5, "embed": 1.0, "refine": 0.75})
    assert (arm, error) == ("free", 0.5)


def test_slice_chain_property_is_monotone_and_zero_tax():
    generator = torch.Generator().manual_seed(919)
    source = torch.randn(17, 5, 7, generator=generator)
    predecessor = source + 0.1 * torch.randn(
        source.shape, generator=generator
    )
    predecessor_errors = (
        (source - predecessor).square().mean(dim=(-2, -1)).tolist()
    )
    predecessor_identities = [
        chain_identity_from_digest(
            winning_arm="free",
            solution_digest_value=recipe_solution_digest({"slice": index}),
            predecessor_digest=None,
        )
        for index in range(source.shape[0])
    ]
    for rung in range(29, 36):
        free = source + 0.12 * torch.randn(
            source.shape, generator=generator
        )
        free_errors = (
            (source - free).square().mean(dim=(-2, -1)).tolist()
        )
        selected = select_reconstruction_slices(
            weight=source,
            free_reconstruction=free,
            qname="model.layers.14.experts.down_proj",
            rung=rung,
            format_name=f"FP8_CB_K{rung}",
            content_guard={"fixture": "property"},
            predecessor_reconstruction=predecessor,
            predecessor_errors=predecessor_errors,
            predecessor_identities=predecessor_identities,
        )
        for old, free_error, chosen in zip(
            predecessor_errors, free_errors, selected["selected_errors"]
        ):
            assert epsilon_le(chosen, old)
            assert epsilon_le(chosen, free_error)
        predecessor = selected["reconstruction"]
        predecessor_errors = selected["selected_errors"]
        predecessor_identities = selected["identities"]


def test_optimized_two_arm_selection_uses_registered_epsilon():
    free = 1.0
    inside = free - 0.5e-12
    outside = free - 2.0e-12
    assert relative_epsilon(free, inside) == pytest.approx(1e-12)
    assert epsilon_le(free, inside)
    assert select_arm({"free": free, "embed": inside}) == ("free", free)
    assert select_arm({"free": free, "embed": outside}) == ("embed", outside)


def test_add_one_refine_freezes_prefix_and_scales(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(MINCHAIN_FLAG, "1")
    torch.manual_seed(11)
    weight = torch.randn(2, 256)
    col_weights = torch.rand(1, 256).add_(0.1)
    predecessor = nvfp4_cb_fields(
        weight, 28, grid="fp8", mode="product",
        col_weights=col_weights, scale_sweep=False,
    )
    refined = refine_one_entry(
        weight, predecessor, 29,
        col_weights=col_weights,
        activation_rows=torch.randn(4, 256),
        iterations=1,
    )
    assert torch.equal(refined["scales"], predecessor["scales"])
    for old, new in zip(predecessor["codebook"], refined["codebook"]):
        assert torch.equal(old, new[: old.shape[0]])


def test_five_anchor_pchip_and_accept_all_backstop():
    config = MinChainInterpolationConfig(
        anchors=(28, 33, 38, 43, 48),
        holdbacks=(33, 43),
    )
    anchor_errors = {
        rung: [float(100 - rung), float(100 - rung)]
        for rung in config.anchors
    }
    result = interpolation_acceptance_v2(
        anchor_errors,
        config=config,
        target_rungs=(29, 34, 46),
        audit_rung=34,
        audit_errors=(66.0, 66.0),
    )
    assert result["backstop_failed"] == []
    assert result["audit"]["pass"] is True
    assert result["full_measure_layer"] is False
    assert result["predictions"][34] == pytest.approx([66.0, 66.0])
    predicted = pchip_monotone(
        config.anchors,
        tuple(anchor_errors[rung][0] for rung in config.anchors),
        tuple(range(28, 49)),
    )
    assert all(left >= right for left, right in zip(predicted, predicted[1:]))


def test_gross_cv_outlier_and_audit_failure_actions():
    config = MinChainInterpolationConfig(
        anchors=(28, 33, 38, 43, 48),
        holdbacks=(33, 43),
    )
    anchor_errors = {
        rung: [float(100 - rung), float(100 - rung)]
        for rung in config.anchors
    }
    anchor_errors[33][0] = 1.0
    result = interpolation_acceptance_v2(
        anchor_errors,
        config=config,
        target_rungs=(29, 34),
        audit_rung=34,
        audit_errors=(1.0, 66.0),
    )
    assert result["backstop_failed"] == [0]
    assert result["audit"]["pass"] is False
    assert result["full_measure_layer"] is True


def test_audit_draw_is_layer_deterministic():
    config = MinChainInterpolationConfig(
        anchors=(28, 33, 38, 43, 48),
        holdbacks=(33, 43),
    )
    rungs = tuple(range(28, 49))
    assert config.audit_rung(14, rungs) == config.audit_rung(14, rungs)
    assert config.audit_rung(14, rungs) not in config.anchors


def test_abort_is_not_counted_as_a_guarantee_violation():
    assert guarantee_accounting(
        aborted=True,
        monotone_violations=99,
        zero_tax_violations=99,
    ) == {
        "status": "ABORT",
        "monotone_violations": None,
        "zero_tax_violations": None,
    }


def test_flag_off_serialization_context_is_byte_identical_to_pre_mode():
    before = cb_serialization_context_stamp(CBSerializationContext.production())
    after = cb_serialization_context_stamp(cb_serialization_context_from_env({}))
    assert after == before
    assert after["schema"] == "prismaquant.cb_serialized_payload.v3"
    assert "minchain" not in after
    assert "minchain_version" not in after


def test_flag_off_is_byte_identical_to_existing_fake_encoder_idiom(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(23)
    weight = torch.randn(8, 256)
    col_weights = torch.rand(256).add_(0.1)
    spec = fr.get_format("NVFP4_CB_K12")
    monkeypatch.delenv(MINCHAIN_FLAG, raising=False)
    context = CBSerializationContext.production(minchain=False)

    baseline, _ = nvfp4_cb_pack(
        weight,
        12,
        grid="fp4",
        mode="product",
        col_weights=col_weights,
        scale_coding="two_tier",
        encode_tier=context.encode_tier,
    )
    fields = cb_fields_for_context(
        spec,
        weight,
        context=context,
        col_weights=col_weights,
    )
    flagged_off = nvfp4_cb_assemble_bytes(
        fields, 12, grid="fp4", mode="product"
    )
    assert torch.equal(flagged_off, baseline)


def test_minchain_context_mismatch_is_refused():
    plain = CBSerializationContext.production(minchain=False)
    chain = CBSerializationContext.production(minchain=True)
    stamp = cb_serialization_context_stamp(chain)
    assert stamp["minchain"] is True
    assert stamp["minchain_version"] == "minchain-v1"
    with pytest.raises(ValueError, match="differs"):
        validate_cb_serialization_context_stamp(stamp, plain, where="export")


def test_render_identity_refuses_missing_or_changed_per_cell_stamp():
    fmt = "NVFP4_CB_K16"
    context = CBSerializationContext.production(minchain=True)
    col_weights = {"layer.0": torch.ones(256)}
    source = {"layer.0": torch.ones(2, 256)}
    identity = build_production_cache_cb_render_identity(
        {"layer.0": [fmt]},
        cb_serialization_context=context,
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    identity = bind_cb_render_identity_source_weights(identity, source)
    with pytest.raises(ValueError, match="missing per-cell"):
        validate_cb_render_identity_metadata(
            identity,
            require_minchain_cells=True,
            where="export",
        )

    cell = chain_identity_from_digest(
        winning_arm="free",
        solution_digest_value=recipe_solution_digest({"cell": 0}),
        predecessor_digest=None,
    )
    identity["cb_minchain_cells"] = {"layer.0": {fmt: [cell]}}
    validate_cb_render_identity_metadata(
        identity,
        require_minchain_cells=True,
        where="export",
    )
    identity["cb_minchain_cells"]["layer.0"][fmt][0][
        "solution_digest"
    ] = "0" * 63
    with pytest.raises(ValueError, match="SHA-256"):
        validate_cb_render_identity_metadata(
            identity,
            require_minchain_cells=True,
            where="export",
        )
