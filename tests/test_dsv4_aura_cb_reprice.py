from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant.anchored_cost import (
    AnchoredCostError,
    RenderRequest,
    SegmentKey,
    candidates_by_segment,
    plan_anchor_requests,
)
from prismaquant.cb_anchored_cost import (
    CBPanelPolicy,
    CBUnitDeclaration,
    CodebookAnchoredFormatPlugin,
    LATTICE_BASIS,
    LEARNED_BASIS,
    ROUTE_FLIP_LIMITATION,
    anchors_from_streamed_payload,
    basis_segment_dict,
    build_cb_units,
    build_cb_extrapolation_input_identity,
    build_streamed_cb_render_plan,
    fit_all_cb_segments,
    heldout_validation_report,
    observations_from_streamed_payload,
    plan_cb_panel_and_validation,
    require_allocator_supersurrogate_support,
    write_exportable_artifacts,
)
from prismaquant.dsv4_aura_cb_reprice import (
    DSV4_EXPECTED_ANCHORS,
    DSV4_NONEXPERT_UNITS,
    DSV4_PANEL_POLICY,
    DSV4_TOTAL_UNITS,
    DSv4CampaignError,
    FP8_EXPERT_FORMATS,
    FP8_NONEXPERT_FORMATS,
    NVFP4_FORMATS,
    _recipe_format,
    run_dsv4_anchor_campaign,
    _validate_routed_bundle_selection_identity,
    _safe_work_dir,
    _validated_cold_expert_provenance,
    render_economics_report,
)


_ROLES = (
    "gate_proj", "up_proj", "down_proj",
    "wq_a", "wq_b", "wkv", "wo_b",
)


def _source_map() -> dict[str, str]:
    return {
        **{name: LATTICE_BASIS for name in NVFP4_FORMATS},
        **{
            name: (
                LEARNED_BASIS
                if int(name.rsplit("K", 1)[1]) <= 46
                else LATTICE_BASIS
            )
            for name in FP8_NONEXPERT_FORMATS
        },
    }


def _plugin(*, source_map: dict[str, str] | None = None):
    return CodebookAnchoredFormatPlugin(
        codebook_source_by_format=(
            _source_map() if source_map is None else source_map
        ),
        arm_identity={
            "production_arm": "gptq-static-act-order-scale-sweep",
            "rtn_anchor_allowed": False,
        },
    )


def _declaration(
    qname: str,
    *,
    role: str,
    nonexpert: bool,
) -> CBUnitDeclaration:
    fp8_formats = (
        FP8_NONEXPERT_FORMATS if nonexpert else FP8_EXPERT_FORMATS
    )
    terminal = (
        "FP8_BLOCK_UE8M0_SOURCE" if nonexpert else "MXFP4_SOURCE"
    )
    formats = (*NVFP4_FORMATS, *fp8_formats)
    payloads = {
        name: 1_000 + index * 100
        for index, name in enumerate(formats)
    }
    payloads[terminal] = 9_000
    return CBUnitDeclaration(
        qname=qname,
        role=role,
        unit_class="nonexpert" if nonexpert else "routed_expert",
        n_params=8_192,
        payload_bytes_by_format=payloads,
        terminal_format=terminal,
    )


def _nonexpert_panel_units(count_per_role: int = 36):
    declarations = tuple(
        _declaration(
            f"layers.{unit_index}.{role}",
            role=role,
            nonexpert=True,
        )
        for role in _ROLES
        for unit_index in range(count_per_role)
    )
    plugin = _plugin()
    return build_cb_units(declarations, plugin), plugin


def _streamed_payload(requests, costs, plugin):
    formats: dict[str, list[str]] = {}
    purposes: dict[str, dict[str, list[str]]] = {}
    for request in requests:
        per_unit = formats.setdefault(request.qname, [])
        if request.format_name not in per_unit:
            per_unit.append(request.format_name)
        per_format = purposes.setdefault(request.qname, {}).setdefault(
            request.format_name, []
        )
        if request.purpose not in per_format:
            per_format.append(request.purpose)
    return {
        "costs": costs,
        "provenance": {
            "production_anchor_renderer": {
                "schema": "test-production-renderer.v1",
                "arm_identity": plugin.arm_identity,
                "formats_by_qname": formats,
                "retention": "one_layer_in_memory",
            },
            "production_anchor_render_purposes": purposes,
            "cb_render_identity": {
                "schema": "test-cb-render-identity.v1",
                "outputs_materialized": False,
            },
        },
    }


def test_cb_mapping_uses_authoritative_basis_and_source_gated_ladders():
    plugin = _plugin()
    units = build_cb_units((
        _declaration("expert", role="gate_proj", nonexpert=False),
        _declaration("dense", role="wq_a", nonexpert=True),
    ), plugin)
    by_name = {unit.qname: unit for unit in units}
    expert, dense = by_name["expert"], by_name["dense"]

    expert_segments = candidates_by_segment(expert, plugin)
    dense_segments = candidates_by_segment(dense, plugin)
    assert {
        (segment.family, segment.equivalence_class)
        for segment in expert_segments
    } == {
        ("nvfp4_cb", LATTICE_BASIS),
        ("fp8_cb", LEARNED_BASIS),
    }
    assert {
        (segment.family, segment.equivalence_class)
        for segment in dense_segments
    } == {
        ("nvfp4_cb", LATTICE_BASIS),
        ("fp8_cb", LEARNED_BASIS),
        ("fp8_cb", LATTICE_BASIS),
    }

    expert_fp8 = [
        candidate for candidate in expert.candidates
        if candidate.family == "fp8_cb"
    ]
    dense_fp8 = [
        candidate for candidate in dense.candidates
        if candidate.family == "fp8_cb"
    ]
    assert max(candidate.coordinate for candidate in expert_fp8) == 33
    assert max(candidate.coordinate for candidate in dense_fp8) == 48
    assert not any(
        candidate.equivalence_class == LATTICE_BASIS
        for candidate in expert_fp8
    )

    # The segment-specific feature declaration is identifiable at every seam.
    for candidate in dense.candidates:
        if candidate.terminal:
            continue
        expected_width = (
            1
            if (
                candidate.family,
                candidate.equivalence_class,
            ) == ("fp8_cb", LATTICE_BASIS)
            else 2
        )
        assert len(candidate.shape_features) == expected_width

    incomplete = _source_map()
    incomplete.pop("FP8_CB_K47")
    with pytest.raises(AnchoredCostError, match="lacks exact candidate"):
        build_cb_units((
            _declaration("dense", role="wq_a", nonexpert=True),
        ), _plugin(source_map=incomplete))


def test_anchor_count_scales_by_unit_equivalence_segment_not_rung():
    plugin = _plugin()
    units = build_cb_units((
        _declaration("expert", role="gate_proj", nonexpert=False),
        _declaration("dense", role="wq_a", nonexpert=True),
    ), plugin)
    anchors = plan_anchor_requests(units, plugin)

    # Expert: NV-lattice + FP8-learned. Dense adds FP8-lattice.
    assert len(anchors) == 2 + 3 == 5
    assert sum(
        not candidate.terminal
        for unit in units
        for candidate in unit.candidates
    ) == 41
    assert len({
        (request.qname, request.segment)
        for request in anchors
    }) == len(anchors)
    assert all(request.purpose == "anchor" for request in anchors)

    formats, purposes, report = build_streamed_cb_render_plan(
        units, plugin, anchors, (), (),
    )
    assert report["physical_union_render_cells"] == 5
    assert report["no_full_menu_materialization"] is True
    assert report["rendered_weight_persisted"] is False
    assert all(len(purposes[unit.qname]) in {2, 3} for unit in units)
    assert all(
        len(formats[unit.qname]) == len(purposes[unit.qname]) + 1
        for unit in units
    )  # the extra entry is the unsynthesized passthrough terminal

    for request in anchors:
        serialized = basis_segment_dict(request.segment)
        assert serialized["equivalence_class"] == serialized["basis"]
        assert set(serialized) == {
            "family", "role", "equivalence_class", "basis",
        }
    assert DSV4_EXPECTED_ANCHORS == (
        2 * DSV4_TOTAL_UNITS + DSV4_NONEXPERT_UNITS
    ) == 66_951


def test_panel_policy_is_exact_segment_keyed_ranked_and_held_out():
    units, plugin = _nonexpert_panel_units()
    panel, validation, report = plan_cb_panel_and_validation(
        units, plugin, DSV4_PANEL_POLICY,
    )

    assert len(panel) == 7 * 32 * (4 + 4 + 2) == 2_240
    assert len(validation) == 7 * 4 * 2 == 56
    assert report["panel_render_cells"] == 2_240
    assert report["validation_render_cells"] == 56
    assert report["segment_count"] == 7 * 3
    assert {request.qname for request in panel}.isdisjoint(
        request.qname for request in validation
    )

    for role in _ROLES:
        nv = report["segments"][f"nvfp4_cb|{role}|lattice"]
        learned = report["segments"][f"fp8_cb|{role}|learned"]
        lattice = report["segments"][f"fp8_cb|{role}|lattice"]
        assert (nv["design_rank"], nv["design_rank_required"]) == (2, 2)
        assert (
            learned["design_rank"], learned["design_rank_required"]
        ) == (2, 2)
        assert (
            lattice["design_rank"], lattice["design_rank_required"]
        ) == (1, 1)
        assert all(
            row["segment"]["equivalence_class"]
            == row["segment"]["basis"]
            for row in (nv, learned, lattice)
        )

    assert all(len(key) == 3 for key in DSV4_PANEL_POLICY.panel_rungs_by_segment)
    assert all(
        len(key) == 3
        for key in DSV4_PANEL_POLICY.validation_rungs_by_segment
    )

    anchors = plan_anchor_requests(units, plugin)
    _formats, _purposes, union = build_streamed_cb_render_plan(
        units, plugin, anchors, panel, validation,
    )
    assert len(anchors) == 7 * 36 * 3 == 756
    assert union["physical_union_render_cells"] == 2_380
    assert union["logical_total"] == 756 + 2_240 + 56


def test_panel_fits_are_separate_and_currency_invariance_is_diagnostic():
    units, plugin = _nonexpert_panel_units()
    panel, _validation, _report = plan_cb_panel_and_validation(
        units, plugin, DSV4_PANEL_POLICY,
    )
    costs: dict[str, dict[str, object]] = {}
    for request in panel:
        rung = int(request.format_name.rsplit("K", 1)[1])
        level = 1.0 + int(request.qname.split(".")[1])
        parity = rung % 2
        aura = level * 10.0 ** (-0.025 * rung + 0.03 * parity)
        costs.setdefault(request.qname, {})[request.format_name] = {
            "predicted_dloss": aura,
            "weight_mse_diagnostic": 7.0 * level * 10.0 ** (
                -0.025 * rung + 0.03 * parity
            ),
            "dw_source": "production_render",
            "production_anchor_measured": True,
        }
    observations = observations_from_streamed_payload(
        panel, _streamed_payload(panel, costs, plugin),
    )
    fits = fit_all_cb_segments(observations, units, plugin)

    assert len(fits) == 21
    for segment, fit in fits.items():
        assert fit.segment == segment
        assert fit.design_rank == fit.design_rank_required
        assert set(fit.g_by_format) == {
            candidate.format_name
            for unit in units
            if unit.role == segment.role
            for candidate in unit.candidates
            if not candidate.terminal
            and candidate.family == segment.family
            and candidate.equivalence_class == segment.equivalence_class
        }
        diagnostic = fit.aura_vs_weight_diagnostic
        assert diagnostic is not None
        assert diagnostic["currency_invariance_test_only"] is True
        assert diagnostic["max_abs_dex"] == pytest.approx(0.0, abs=1e-10)

    learned = SegmentKey("fp8_cb", "wq_a", LEARNED_BASIS)
    lattice = SegmentKey("fp8_cb", "wq_a", LATTICE_BASIS)
    assert fits[learned].design_rank_required == 2
    assert fits[lattice].design_rank_required == 1
    assert set(fits[learned].g_by_format).isdisjoint(
        fits[lattice].g_by_format
    )


def test_streamed_rows_must_be_real_production_arm_measurements():
    plugin = _plugin()
    unit = build_cb_units((
        _declaration("dense", role="wq_a", nonexpert=True),
    ), plugin)[0]
    request = plan_anchor_requests((unit,), plugin)[0]
    good = _streamed_payload((request,), {
            request.qname: {
                request.format_name: {
                    "predicted_dloss": 1.25,
                    "dw_source": "production_render",
                    "production_anchor_measured": True,
                },
            },
        }, plugin)
    anchor = anchors_from_streamed_payload((request,), good)[
        (request.qname, request.segment)
    ]
    assert anchor.predicted_dloss == 1.25
    assert anchor.receipt.request == request

    for update in (
        {"dw_source": "rtn"},
        {"production_anchor_measured": False},
    ):
        bad = json.loads(json.dumps(good))
        bad["costs"][request.qname][request.format_name].update(update)
        with pytest.raises(AnchoredCostError, match="not a real production"):
            anchors_from_streamed_payload((request,), bad)


def test_streamed_receipts_hash_global_identity_once(monkeypatch):
    import prismaquant.cb_anchored_cost as cb_adapter

    plugin = _plugin()
    units = build_cb_units((
        _declaration("expert", role="gate_proj", nonexpert=False),
        _declaration("dense", role="wq_a", nonexpert=True),
    ), plugin)
    requests = plan_anchor_requests(units, plugin)
    costs = {
        request.qname: {
            candidate.format_name: {
                "predicted_dloss": 1.0,
                "dw_source": "production_render",
                "production_anchor_measured": True,
            }
            for candidate in requests
            if candidate.qname == request.qname
        }
        for request in requests
    }
    payload = _streamed_payload(requests, costs, plugin)
    original = cb_adapter.canonical_json_sha256
    calls: list[str] = []

    def counted(value, *, where):
        calls.append(where)
        return original(value, where=where)

    monkeypatch.setattr(cb_adapter, "canonical_json_sha256", counted)
    anchors = anchors_from_streamed_payload(requests, payload)

    assert len(anchors) == len(requests) == 5
    assert calls == [
        "streamed production render arm identity",
        "streamed production render payload identity",
    ]


def test_full_extrapolation_identity_expands_inputs_not_measurements():
    import torch
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext
    from prismaquant.production_weight_cache import (
        bind_cb_render_identity_source_weights,
        build_production_cache_cb_render_identity,
        project_cb_render_identity,
    )

    qname = "model.layers.0.self_attn.wq_a"
    context = CBSerializationContext.production()
    col_weights = {qname: torch.ones(256)}
    sparse = build_production_cache_cb_render_identity(
        {qname: ["NVFP4_CB_K15"]},
        cb_serialization_context=context,
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    sparse = bind_cb_render_identity_source_weights(
        sparse, {qname: torch.zeros(2, 256)},
    )

    expanded = build_cb_extrapolation_input_identity(
        sparse,
        legal_formats_by_qname={
            qname: ("NVFP4_CB_K12", "NVFP4_CB_K15"),
        },
        col_weights=col_weights,
        cb_serialization_context=context,
    )

    assert sparse["cb_formats_by_qname"][qname] == ["NVFP4_CB_K15"]
    assert expanded["cb_formats_by_qname"][qname] == [
        "NVFP4_CB_K12", "NVFP4_CB_K15",
    ]
    assert expanded["source_weights_sha256"] == (
        sparse["source_weights_sha256"]
    )
    selected = project_cb_render_identity(
        expanded,
        {qname: ["NVFP4_CB_K12"]},
        col_weights=col_weights,
        where="anchored selected rung",
    )
    assert selected["cb_formats_by_qname"] == {
        qname: ["NVFP4_CB_K12"],
    }


def test_recipe_parser_recovers_actual_dsv4_fp8_source_terminal():
    actual_track_a_recipe = {
        "bits": 8,
        "group_size": 128,
        "data_type": "fp8_e4m3",
        "sym": True,
        "scale_fmt": "ue8m0",
        "act_bits": 16,
        "act_data_type": "float",
    }
    assert _recipe_format(actual_track_a_recipe) == (
        "FP8_BLOCK_UE8M0_SOURCE"
    )
    neighbour = dict(actual_track_a_recipe)
    neighbour.pop("scale_fmt")
    assert _recipe_format(neighbour) == "FP8_SOURCE"


def test_heldout_zero_is_strict_json_bad_signal_not_a_gate():
    plugin = _plugin()
    units = build_cb_units(tuple(
        _declaration(
            f"layers.{index}.wq_a", role="wq_a", nonexpert=True,
        )
        for index in range(36)
    ), plugin)
    anchors = plan_anchor_requests(units, plugin)
    panel, validation, _report = plan_cb_panel_and_validation(
        units, plugin, DSV4_PANEL_POLICY,
    )
    all_requests = (*anchors, *panel, *validation)
    costs: dict[str, dict[str, object]] = {}
    zero_cell = (validation[0].qname, validation[0].format_name)
    for request in all_requests:
        rung = int(request.format_name.rsplit("K", 1)[1])
        level = 1.0 + int(request.qname.split(".")[1])
        value = level * 10.0 ** (-0.025 * rung + 0.03 * (rung % 2))
        if (request.qname, request.format_name) == zero_cell:
            value = 0.0
        costs.setdefault(request.qname, {})[request.format_name] = {
            "predicted_dloss": value,
            "weight_mse_diagnostic": (
                None if value == 0.0 else 7.0 * value
            ),
            "dw_source": "production_render",
            "production_anchor_measured": True,
        }
    payload = _streamed_payload(all_requests, costs, plugin)
    anchor_values = anchors_from_streamed_payload(anchors, payload)
    panel_observations = observations_from_streamed_payload(panel, payload)
    fits = fit_all_cb_segments(panel_observations, units, plugin)
    validation_observations = observations_from_streamed_payload(
        validation, payload,
    )
    report = heldout_validation_report(
        validation_observations, anchor_values, fits,
    )

    assert report["reported_not_gated"] is True
    assert report["status"] == "BAD_FACTORISATION_SIGNAL"
    assert report["n_nonfinite_dex"] == 1
    zero_row = next(
        row for row in report["rows"]
        if row["measured_predicted_dloss"] == 0.0
    )
    assert zero_row["absolute_dex_error"] is None
    assert zero_row["dex_error_nonfinite_reason"] == (
        "measured_predicted_dloss_is_nonpositive"
    )
    json.dumps(report, allow_nan=False)


def test_cb_adapter_is_model_agnostic_and_provenance_names_generic_segment():
    source = (
        Path(__file__).parents[1] / "prismaquant" / "cb_anchored_cost.py"
    ).read_text()
    for dsv4_specific in (
        "DSV4", "33_325", "66_951", "112_690_000_000", "43 * 256",
        '"gate_proj", "up_proj", "down_proj"',
    ):
        assert dsv4_specific not in source

    provenance = _plugin().provenance_identity_fields()
    assert provenance["segment_key_fields"] == [
        "family", "role", "equivalence_class",
    ]
    assert provenance["cb_segment_alias_fields"] == [
        "family", "role", "basis",
    ]
    assert provenance["aura_is_only_cost_currency"] is True
    assert provenance["cb_col_weights_role"] == (
        "production_render_input_only"
    )
    assert provenance["route_flip_limitation"] == ROUTE_FLIP_LIMITATION


def test_exportable_artifacts_preserve_render_imatrix_and_refuse_overwrite(
    tmp_path,
):
    allocator = tmp_path / "allocator"
    allocator.mkdir()
    (allocator / "layer_config.json").write_text(json.dumps({
        "unit": {"data_type": "fp8_cb", "cb_k": 33},
    }))
    (allocator / "selection.json").write_text(json.dumps({
        "feasible": True,
        "chosen_achieved_bits": 2.75,
        "predicted_dloss": 12.5,
        "budget_bytes": 112_690_000_000,
    }))
    col_weights = tmp_path / "cb_col_weights.pkl"
    col_weights.write_bytes(b"render-input-only")
    destination = tmp_path / "new-artifacts"

    output = write_exportable_artifacts(
        destination,
        allocator_output_dir=allocator,
        cb_col_weights_path=col_weights,
        provenance={"campaign": "test"},
    )
    assert output == destination
    assert (destination / "cb_col_weights.pkl").read_bytes() == (
        b"render-input-only"
    )
    layer_config = json.loads(
        (destination / "layer_config.json").read_text()
    )
    selection = json.loads((destination / "selection.json").read_text())
    stamp = layer_config["__prismaquant__"]["aura_cb_reprice"]
    assert stamp["cost_currency"] == "aura_predicted_dloss"
    assert stamp["fisher_application_count"] == 1
    assert stamp["route_flip_limitation"] == ROUTE_FLIP_LIMITATION
    assert selection["aura_cb_reprice"] == stamp
    assert selection["feasible"] is True

    before = {
        name: (destination / name).stat().st_mtime_ns
        for name in (
            "layer_config.json", "selection.json", "cb_col_weights.pkl",
            ".anchored_publish.json",
        )
    }
    assert write_exportable_artifacts(
        destination,
        allocator_output_dir=allocator,
        cb_col_weights_path=col_weights,
        provenance={"campaign": "test"},
        resume=True,
    ) == destination
    assert {
        name: (destination / name).stat().st_mtime_ns
        for name in before
    } == before

    with pytest.raises(
        AnchoredCostError, match="artifact publish identity mismatch",
    ):
        write_exportable_artifacts(
            destination,
            allocator_output_dir=allocator,
            cb_col_weights_path=col_weights,
            provenance={"campaign": "different"},
            resume=True,
        )

    with pytest.raises(AnchoredCostError, match="refusing overwrite"):
        write_exportable_artifacts(
            destination,
            allocator_output_dir=allocator,
            cb_col_weights_path=col_weights,
            provenance={"campaign": "test"},
        )


def test_exportable_artifact_resume_safely_finishes_interrupted_publish(
    tmp_path, monkeypatch,
):
    import prismaquant.cb_anchored_cost as cb_anchored_cost

    allocator = tmp_path / "allocator"
    allocator.mkdir()
    (allocator / "layer_config.json").write_text(json.dumps({
        "unit": {"data_type": "fp8_cb", "cb_k": 33},
    }))
    (allocator / "selection.json").write_text(json.dumps({
        "feasible": True,
        "chosen_achieved_bits": 2.75,
        "predicted_dloss": 12.5,
        "budget_bytes": 112_690_000_000,
    }))
    col_weights = tmp_path / "cb_col_weights.pkl"
    col_weights.write_bytes(b"render-input-only")
    destination = tmp_path / "interrupted-artifacts"
    provenance = {"campaign": "identity-bound"}
    real_atomic_write = cb_anchored_cost.atomic_write_bytes

    def interrupt_selection(path, payload):
        if path.name == "selection.json":
            raise RuntimeError("synthetic publish interruption")
        real_atomic_write(path, payload)

    monkeypatch.setattr(
        cb_anchored_cost, "atomic_write_bytes", interrupt_selection,
    )
    with pytest.raises(RuntimeError, match="synthetic publish interruption"):
        write_exportable_artifacts(
            destination,
            allocator_output_dir=allocator,
            cb_col_weights_path=col_weights,
            provenance=provenance,
            resume=False,
        )
    layer_stat = (destination / "layer_config.json").stat()
    manifest = json.loads(
        (destination / ".anchored_publish.json").read_text()
    )
    assert manifest["complete"] is False
    assert not (destination / "selection.json").exists()

    monkeypatch.setattr(
        cb_anchored_cost, "atomic_write_bytes", real_atomic_write,
    )
    with pytest.raises(
        AnchoredCostError, match="artifact publish identity mismatch",
    ):
        write_exportable_artifacts(
            destination,
            allocator_output_dir=allocator,
            cb_col_weights_path=col_weights,
            provenance={"campaign": "wrong"},
            resume=True,
        )
    assert not (destination / "selection.json").exists()

    assert write_exportable_artifacts(
        destination,
        allocator_output_dir=allocator,
        cb_col_weights_path=col_weights,
        provenance=provenance,
        resume=True,
    ) == destination
    assert (destination / "layer_config.json").stat() == layer_stat
    assert (destination / "selection.json").is_file()
    assert (destination / "cb_col_weights.pkl").read_bytes() == (
        b"render-input-only"
    )
    assert json.loads(
        (destination / ".anchored_publish.json").read_text()
    )["complete"] is True


def test_dsv4_driver_wires_resume_to_allocator_and_artifact_publish():
    tree = ast.parse(inspect.getsource(run_dsv4_anchor_campaign))
    calls = {
        node.func.id: node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {
            "run_allocator_once", "write_exportable_artifacts",
        }
    }
    assert set(calls) == {
        "run_allocator_once", "write_exportable_artifacts",
    }
    for call in calls.values():
        resume = next(
            keyword.value for keyword in call.keywords
            if keyword.arg == "resume"
        )
        assert ast.dump(resume) == ast.dump(ast.parse(
            "bool(args.resume)", mode="eval",
        ).body)


def test_allocator_supersurrogate_gap_refuses_before_any_campaign_render():
    with pytest.raises(
        AnchoredCostError,
        match="allocator lacks explicit AURA supersurrogate",
    ):
        require_allocator_supersurrogate_support()


def test_panel_policy_requires_exact_role_segment_key():
    units, plugin = _nonexpert_panel_units(count_per_role=4)
    bad_policy = CBPanelPolicy(
        panel_rungs_by_segment={
            ("fp8_cb", "wrong_role", LEARNED_BASIS): (
                "FP8_CB_K28", "FP8_CB_K33",
            ),
        },
        validation_rungs_by_segment={},
        panel_units_per_role=4,
        validation_units_per_role=0,
    )
    with pytest.raises(AnchoredCostError, match="panel policy lacks"):
        plan_cb_panel_and_validation(units, plugin, bad_policy)


def test_economics_scales_measured_timings_by_params_and_deduplicates(
    tmp_path,
):
    reference = 2048 * 4096
    nv = SegmentKey("nvfp4_cb", "gate_proj", LATTICE_BASIS)
    learned = SegmentKey("fp8_cb", "gate_proj", LEARNED_BASIS)
    lattice = SegmentKey("fp8_cb", "gate_proj", LATTICE_BASIS)
    anchors = (
        RenderRequest("q0", nv, "NVFP4_CB_K15", "anchor"),
        RenderRequest("q0", learned, "FP8_CB_K33", "anchor"),
        RenderRequest("q1", lattice, "FP8_CB_K47", "anchor"),
    )
    panel = (
        RenderRequest("q0", nv, "NVFP4_CB_K15", "panel"),
        RenderRequest("q0", nv, "NVFP4_CB_K12", "panel"),
        RenderRequest("q0", learned, "FP8_CB_K33", "panel"),
        RenderRequest("q0", learned, "FP8_CB_K41", "panel"),
        RenderRequest("q1", lattice, "FP8_CB_K47", "panel"),
        RenderRequest("q1", lattice, "FP8_CB_K48", "panel"),
    )
    validation = (
        RenderRequest("q2", learned, "FP8_CB_K28", "validation"),
        RenderRequest("q2", learned, "FP8_CB_K46", "validation"),
    )
    purposes: dict[str, dict[str, list[str]]] = {}
    for request in (*anchors, *panel, *validation):
        rows = purposes.setdefault(request.qname, {})
        rows.setdefault(request.format_name, []).append(request.purpose)
    col_weights = tmp_path / "cb_col_weights.pkl"
    col_weights.write_bytes(b"render-input")
    prepared = SimpleNamespace(
        args=SimpleNamespace(
            n_probes=32,
            work_dir=str(tmp_path),
            col_weights=str(col_weights),
            activation_cache_dir="/read-only/activation-cache",
        ),
        probe_stats={
            "q0": {"n_params": reference},
            "q1": {"n_params": 4 * reference},
            "q2": {"n_params": reference // 2},
        },
        anchor_requests=anchors,
        panel_requests=panel,
        validation_requests=validation,
        purposes_by_qname=purposes,
        units=(
            SimpleNamespace(candidates=(1, 2)),
            SimpleNamespace(candidates=(1, 2, 3)),
        ),
        plan_report={
            "anchor_renders_by_family_basis": {
                "nvfp4_cb|lattice": 1,
                "fp8_cb|learned": 1,
                "fp8_cb|lattice": 1,
            },
            "anchor_renders_by_segment": {
                nv.stamp: 1,
                learned.stamp: 1,
                lattice.stamp: 1,
            },
        },
    )

    report = render_economics_report(prepared)

    assert report["physical_union_render_cells"] == 8
    assert report["physical_render_cells_charged_by_purpose"] == {
        "anchor": 3,
        "panel": 3,
        "validation": 2,
    }
    assert report["reference_tensor_equivalents_by_purpose"] == {
        "anchor": 6.0,
        "panel": 6.0,
        "validation": 1.0,
    }
    expected_encode_seconds = (
        0.069821 + 0.144363 + 4 * 3.8868322437820098
        + 0.069821 + 1.3973947751555897 + 4 * 4.442083168687532
        + 0.5 * 0.075109 + 0.5 * 3.335986553436669
    )
    assert report["encode_projection_seconds"] == pytest.approx(
        expected_encode_seconds
    )
    assert report["p0_projection_seconds"] == pytest.approx(3_788.0)
    assert report["total_projected_gpu_hours"] == pytest.approx(
        (expected_encode_seconds + 3_788.0) / 3_600.0
    )
    block = report["disk_projection_block_bytes"]
    assert report["projected_peak_new_disk_bytes"] == (
        (8 + 5 + 2) * block + col_weights.stat().st_size
    )
    assert "not separately bounded" in report[
        "projected_peak_new_disk_assumptions"
    ]
    assert report["persistent_rendered_weight_bytes"] == 0


def test_cold_expert_render_is_bound_to_probe_and_imatrix_evidence(tmp_path):
    cold = "model.layers.0.mlp.experts.7.gate_proj"
    warm = "model.layers.0.mlp.experts.8.gate_proj"
    stats = {
        cold: {"n_tokens_seen": 0},
        warm: {"n_tokens_seen": 8},
    }
    unit_classes = {
        cold: "profile_declared_routed_expert",
        warm: "profile_declared_routed_expert",
    }
    col_weights = tmp_path / "cb_col_weights.pkl"
    col_weights.write_bytes(b"fixture")
    sidecar = Path(f"{col_weights}.provenance.json")
    sidecar.write_text(json.dumps({
        "rule": "unrouted_expert_neutral_prior:layer_routed_mean",
        "basis": "probe n_tokens_seen == 0",
        "names": [cold],
    }))

    provenance = _validated_cold_expert_provenance(
        stats=stats,
        unit_classes=unit_classes,
        missing_activations=[cold],
        col_weights_path=col_weights,
    )
    assert provenance["names"] == [cold]
    assert provenance["count"] == 1
    assert len(provenance["imatrix_provenance_sha256"]) == 64

    with pytest.raises(DSv4CampaignError, match="misses differ"):
        _validated_cold_expert_provenance(
            stats=stats,
            unit_classes=unit_classes,
            missing_activations=[],
            col_weights_path=col_weights,
        )

    sidecar.write_text(json.dumps({
        "rule": "unrouted_expert_neutral_prior:layer_routed_mean",
        "basis": "probe n_tokens_seen == 0",
        "names": [warm],
    }))
    with pytest.raises(DSv4CampaignError, match="imatrix provenance differs"):
        _validated_cold_expert_provenance(
            stats=stats,
            unit_classes=unit_classes,
            missing_activations=[cold],
            col_weights_path=col_weights,
        )


def test_campaign_paths_refuse_track_a_baseline_before_writing():
    baseline = Path(
        "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p7/artifacts"
    )
    with pytest.raises(DSv4CampaignError, match="Track A baseline"):
        _safe_work_dir(baseline.parent)

    with pytest.raises(AnchoredCostError, match="baseline overwrite"):
        write_exportable_artifacts(
            baseline / "nested-new-output",
            allocator_output_dir=Path("/does/not/matter"),
            cb_col_weights_path=Path("/does/not/matter"),
            provenance={},
        )


def test_routed_bundle_origins_are_bound_to_supplied_selection(monkeypatch):
    import prismaquant.cb_learned_bundle as learned_bundle
    from prismaquant.cb_banked_books import BANKED_CBL_ORIGIN_SCHEMA

    selected_sha = "a" * 64
    digest = "b" * 64
    cells: dict[str, dict[str, object]] = {}
    for layer in range(43):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            qname = f"model.layers.{layer}.mlp.experts.{projection}"
            cells[qname] = {}
            for rung in range(28, 34):
                cells[qname][f"FP8_CB_K{rung}"] = {
                    "source": "learned",
                    "pretrained_origin": {
                        "schema": BANKED_CBL_ORIGIN_SCHEMA,
                        "selection_sha256": selected_sha,
                        "selection_path": "/selection.json",
                        "burn_shard": "/burn.pkl",
                        "burn_content_key": digest,
                        "burn_pass_tag": "pass",
                        "book_path": "/book.pqcb",
                        "book_key": digest,
                        "book_file_sha256": digest,
                        "pooled_book_sha256": digest,
                        "subtable_content_sha256": [digest],
                        "layer": layer,
                        "projection": projection,
                        "rung": rung,
                        "source_digest": digest,
                        "col_weights_digest": digest,
                    },
                }
    fake_bundle = SimpleNamespace(
        manifest={"cells": cells},
        bundle_content_sha256="c" * 64,
    )
    monkeypatch.setattr(
        learned_bundle, "load_bundle_cached", lambda _path: fake_bundle,
    )

    report = _validate_routed_bundle_selection_identity(
        "/bundle.pqcb", selected_sha,
    )
    assert report["routed_learned_origin_cells"] == 43 * 3 * 6
    assert report["selection_sha256"] == selected_sha

    first_qname = next(iter(cells))
    cells[first_qname]["FP8_CB_K28"]["pretrained_origin"][
        "selection_sha256"
    ] = "d" * 64
    with pytest.raises(DSv4CampaignError, match="not bound"):
        _validate_routed_bundle_selection_identity(
            "/bundle.pqcb", selected_sha,
        )
