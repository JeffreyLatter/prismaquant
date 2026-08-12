from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from prismaquant.anchored_cost import CandidateSpec, UnitSpec
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
import prismaquant.dsv4_aura_cb_reprice as campaign


def _write_minimal_completed_replay(tmp_path: Path):
    from prismaquant.aura_cost import (
        AURA_CHECKPOINT_IDENTITY_SCHEMA,
        AURA_CHECKPOINT_MANIFEST_SCHEMA,
        AURA_CHECKPOINT_UNIT_SCHEMA,
    )
    from prismaquant.production_weight_cache import (
        _combined_source_weights_sha256,
    )

    qname = "model.layers.0.self_attn.wq_a"
    measured_format = "FP8_CB_K32"
    terminal_format = "FP8_BLOCK_UE8M0_SOURCE"
    unit = UnitSpec(
        qname=qname,
        role="wq_a",
        unit_class="nonexpert",
        n_params=16,
        candidates=(
            CandidateSpec(
                measured_format, 4.0, 8, "fp8_cb", "learned",
                (32.0,), 32.0,
            ),
            CandidateSpec(
                terminal_format, 8.0, 16, "source_terminal",
                "passthrough", (), 0.0,
                terminal=True,
                allocator_selectable=False,
            ),
        ),
    )
    work_dir = tmp_path / "campaign"
    checkpoint_root = work_dir / "checkpoints" / "aura"
    artifact_path = work_dir / "artifacts" / "streamed_anchor_aura.pkl"
    args = SimpleNamespace(
        work_dir=str(work_dir),
        checkpoint_dir=str(work_dir / "checkpoints"),
        n_probes=2,
    )
    arm_identity = {"test_arm": "production"}
    purposes = {qname: {measured_format: ["anchor"]}}
    prepared = SimpleNamespace(
        args=args,
        units=(unit,),
        probe_stats={qname: {
            "out_features": 4,
            "in_features": 4,
            "n_params": 16,
        }},
        probe_meta={"calib_hash": "calibration-hash"},
        purposes_by_qname=purposes,
        formats_by_qname={qname: (terminal_format, measured_format)},
        format_plan=SimpleNamespace(identity_sha256="f" * 64),
        routed_selection_sha256="e" * 64,
        arm_identity=arm_identity,
    )

    base_cb = {
        "schema": "test.cb.render.v1",
        "cb_formats_by_qname": {qname: [measured_format]},
        "immutable_render_input": "bound",
    }
    base_renderer = {
        "schema": "test.production.renderer.v1",
        "arm_identity": arm_identity,
        "formats_by_qname": {qname: [measured_format]},
        "cb_render_identity": base_cb,
        "retention": "one_layer_in_memory",
    }
    source_record = {"shape": [4, 4], "sha256": "a" * 64}
    source_records = {qname: source_record}
    cb_shapes = {qname: [4, 4]}
    cb_content = {qname: "a" * 64}
    completed_cb = {
        **base_cb,
        "source_weights_complete": True,
        "source_weights_shapes": cb_shapes,
        "source_weights_content_sha256": cb_content,
        "source_weights_sha256": _combined_source_weights_sha256(
            cb_shapes, cb_content,
        ),
        "render_scope": "sparse_production_anchors",
    }
    completed_renderer = {
        **base_renderer,
        "cb_render_identity": completed_cb,
        "source_weights": {
            "complete": True,
            "scope": "sparse_anchor_plan",
            "records": source_records,
            "identity_sha256": canonical_json_sha256(
                source_records, where="test source records",
            ),
        },
    }

    state_row = {
        "s2": 13.0,
        "s4": 97.0,
        "x2_probe": [4.0, 9.0],
        "dw_src": "production_render",
    }
    state = {
        "g_trace": 6.0,
        "rows": {measured_format: state_row},
        "col_energy": None,
        "source_weight_identity": source_record,
    }
    measured_row = campaign._expected_checkpoint_cost_row(
        qname,
        measured_format,
        state_row,
        n_probes=2,
        diagnostic_expected=False,
    )
    legacy_zero = {
        "predicted_dloss": 0.0,
        "output_mse_measured": False,
        "cost_source": "aura_passthrough_zero",
    }
    raw_payload = {
        "schema": "prismaquant.aura_cost.v1",
        "n_probes": 2,
        "formats": [measured_format, terminal_format, "MXFP4_SOURCE"],
        "stats": {qname: {
            "h_trace": 3.0,
            "n_params": 16,
            "in_features": 4,
            "out_features": 4,
            "n_probes": 2,
        }},
        "costs": {qname: {
            measured_format: measured_row,
            # Exact historical rows: block-FP8 is now activation-unmeasured;
            # MXFP4 is the old global-zero cross-pollution on this dense unit.
            terminal_format: legacy_zero,
            "MXFP4_SOURCE": legacy_zero,
        }},
        "provenance": {
            "calib_hash": "calibration-hash",
            "production_anchor_render_purposes": purposes,
            "production_anchor_renderer": completed_renderer,
            "production_anchor_sparse_render_identity": completed_cb,
            "cb_render_identity": completed_cb,
            "dw_production_anchor_rows": 1,
        },
    }

    identity = {
        "schema": AURA_CHECKPOINT_IDENTITY_SCHEMA,
        "calibration": {"calib_hash": "calibration-hash"},
        "units": [{
            "qname": qname,
            "shape": [4, 4],
            "dtype": "torch.bfloat16",
            "n_params": 16,
        }],
        "chunks": [[qname]],
        "n_probes": 2,
        "collect_col_energy": False,
        "require_production_cache": True,
        "extra": {
            "campaign_schema": campaign.DSV4_CAMPAIGN_SCHEMA,
            "source_format_plan_identity_sha256": "f" * 64,
            "routed_book_selection_sha256": "e" * 64,
            "include_routed_experts": True,
            "production_anchor_render_purposes": purposes,
            "production_anchor_renderer": base_renderer,
            # The live campaign was launched before the global-zero plan bug
            # was fixed, so both legacy terminal names appear in this identity.
            "streamed_formats_by_qname": {qname: [
                terminal_format, measured_format, "MXFP4_SOURCE",
            ]},
        },
    }
    identity_sha256 = canonical_json_sha256(
        identity, where="test AURA checkpoint identity",
    )
    unit_payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {
        "schema": AURA_CHECKPOINT_UNIT_SCHEMA,
        "qname": qname,
        "identity_sha256": identity_sha256,
        "payload_sha256": hashlib.sha256(unit_payload).hexdigest(),
        "payload": unit_payload,
    }
    unit_name = hashlib.sha256(qname.encode()).hexdigest() + ".pkl"
    manifest = {
        "schema": AURA_CHECKPOINT_MANIFEST_SCHEMA,
        "identity_sha256": identity_sha256,
        "identity": identity,
        "units": [{"qname": qname, "file": f"units/{unit_name}"}],
    }

    (checkpoint_root / "units").mkdir(parents=True)
    (checkpoint_root / "units" / unit_name).write_bytes(
        pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    )
    (checkpoint_root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True)
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(
        pickle.dumps(raw_payload, protocol=pickle.HIGHEST_PROTOCOL)
    )
    return prepared, artifact_path, raw_payload, qname, measured_format


def test_cpu_replay_quarantines_legacy_terminal_zeros(tmp_path):
    prepared, artifact, _raw, qname, measured = (
        _write_minimal_completed_replay(tmp_path)
    )
    sanitized, provenance = (
        campaign._load_and_audit_completed_streamed_payload(
            prepared, artifact,
        )
    )

    assert set(sanitized["costs"][qname]) == {measured}
    assert provenance["measurement_invoked"] is False
    assert provenance["legacy_fp8_terminal_zero_rows_quarantined"] == 1
    assert provenance["legacy_cross_terminal_zero_rows_quarantined"] == 1
    assert provenance["unit_checkpoint_count"] == 1


def test_cpu_replay_refuses_monolithic_scalar_tamper(tmp_path):
    prepared, artifact, raw, qname, measured = (
        _write_minimal_completed_replay(tmp_path)
    )
    raw["costs"][qname][measured]["predicted_dloss"] += 1.0
    artifact.write_bytes(pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL))

    with pytest.raises(
        campaign.DSv4CampaignError, match="scalar differs from journal",
    ):
        campaign._load_and_audit_completed_streamed_payload(
            prepared, artifact,
        )


def test_replay_control_plane_never_invokes_measurement(monkeypatch, tmp_path):
    args = SimpleNamespace(
        replay_streamed_payload=str(tmp_path / "streamed_anchor_aura.pkl"),
        work_dir=str(tmp_path / "work"),
    )
    prepared = SimpleNamespace(args=args)
    def prepare(_args, *, publish_format_plan=True):
        assert publish_format_plan is False
        return prepared

    monkeypatch.setattr(campaign, "prepare_dsv4_campaign", prepare)
    monkeypatch.setattr(
        campaign, "require_allocator_supersurrogate_support", lambda: None,
    )
    monkeypatch.setattr(
        campaign,
        "_load_and_audit_completed_streamed_payload",
        lambda _prepared, _path: ({"costs": {}}, {"measurement_invoked": False}),
    )

    def forbidden(_prepared):
        raise AssertionError("GPU measurement must not run during replay")

    monkeypatch.setattr(campaign, "_measure_streamed", forbidden)
    observed = {}

    def finish(_prepared, _payload, **kwargs):
        observed.update(kwargs)
        return 0

    monkeypatch.setattr(campaign, "_finish_dsv4_campaign", finish)
    assert campaign.run_dsv4_anchor_replay(
        args, control_plane=campaign.__name__,
    ) == 0
    assert observed["replay_provenance"]["measurement_invoked"] is False
    assert observed["allocator_output"].name == (
        "allocator-aura-activation-safe"
    )
