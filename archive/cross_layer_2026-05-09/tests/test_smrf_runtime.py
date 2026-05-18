from __future__ import annotations

import json
from pathlib import Path

from prismaquant import decision_units as du
from prismaquant.research_components import smrf_runtime as smrf


def _option(omega_ii: float, bits_per_param: float, n_params: int) -> dict:
    return {
        "omega_ii": float(omega_ii),
        "bits_per_param": float(bits_per_param),
        "memory_bytes": int(round(n_params * bits_per_param / 8.0)),
    }


def _unit(
    members: list[str],
    *,
    n_params: int,
    nvfp4_cost: float,
    mxfp8_cost: float,
) -> dict:
    return {
        "members": members,
        "options": {
            "NVFP4": _option(nvfp4_cost, 4.0, n_params),
            "MXFP8": _option(mxfp8_cost, 8.0, n_params),
            "BF16": _option(0.0, 16.0, n_params),
        },
    }


def _payload() -> dict:
    return {
        "schema": du.SCHEMA,
        "blocks": {
            "model.layers.0": {
                "units": {
                    "model.layers.0.self_attn.qkv_proj": _unit(
                        [
                            "model.layers.0.self_attn.q_proj",
                            "model.layers.0.self_attn.k_proj",
                            "model.layers.0.self_attn.v_proj",
                        ],
                        n_params=96,
                        nvfp4_cost=0.75,
                        mxfp8_cost=0.18,
                    ),
                    "model.layers.0.self_attn.o_proj": _unit(
                        ["model.layers.0.self_attn.o_proj"],
                        n_params=64,
                        nvfp4_cost=0.22,
                        mxfp8_cost=0.06,
                    ),
                    "model.layers.0.mlp.up_gate_proj": _unit(
                        [
                            "model.layers.0.mlp.up_proj",
                            "model.layers.0.mlp.gate_proj",
                        ],
                        n_params=128,
                        nvfp4_cost=1.05,
                        mxfp8_cost=0.24,
                    ),
                },
                "pairs": [],
            }
        },
        "singletons": {
            "model.layers.1.self_attn.o_proj": {
                "block_id": "model.layers.1",
                "members": ["model.layers.1.self_attn.o_proj"],
                "options": {
                    "NVFP4": _option(0.35, 4.0, 64),
                    "MXFP8": _option(0.09, 8.0, 64),
                    "BF16": _option(0.0, 16.0, 64),
                },
            }
        },
    }


def test_smrf_generates_archive_candidates_and_expands_fused_members():
    archive = smrf.generate_archive_payload(
        _payload(),
        bpp_min=4.0,
        bpp_max=16.0,
        n_lambdas=7,
        bit_precision_bpp=0.25,
        beam_per_bin=3,
        validation_candidates=5,
    )

    assert archive["schema"] == smrf.ARCHIVE_SCHEMA
    assert archive["meta"]["dp"]["beam_per_bin"] == 3
    assert archive["meta"]["n_generated"] > 0
    assert 0 < archive["meta"]["n_selected"] <= 5
    assert archive["surrogate_frontier"]

    generated = archive["generated"]
    assert generated == sorted(
        generated,
        key=lambda row: (row["achieved_bpp"], row["surrogate_loss"], row["assignment_hash"]),
    )
    for row in generated:
        assert 4.0 <= row["achieved_bpp"] <= 16.0
        assert row["block_format_counts"]
        assert row["assignment_entries"] == len(row["assignment"])
        assert "model.layers.0.self_attn.qkv_proj" in row["unit_assignment"]
        assert "model.layers.0.self_attn.qkv_proj" not in row["assignment"]
        qkv_formats = {
            row["assignment"]["model.layers.0.self_attn.q_proj"],
            row["assignment"]["model.layers.0.self_attn.k_proj"],
            row["assignment"]["model.layers.0.self_attn.v_proj"],
        }
        assert len(qkv_formats) == 1


def test_smrf_lambda_grid_uses_unit_bpp_contribution():
    payload = {
        "schema": du.SCHEMA,
        "blocks": {
            "model.layers.0": {
                "units": {
                    "model.layers.0.small": {
                        "members": ["model.layers.0.small"],
                        "options": {
                            "NVFP4": _option(1.0, 4.0, 1),
                            "BF16": _option(0.0, 16.0, 1),
                        },
                    },
                    "model.layers.0.large": {
                        "members": ["model.layers.0.large"],
                        "options": {
                            "NVFP4": _option(1.0, 4.0, 1000),
                            "BF16": _option(0.0, 16.0, 1000),
                        },
                    },
                },
                "pairs": [],
            },
        },
        "singletons": {},
    }
    units = smrf._all_units(payload)
    grid = smrf._lambda_grid(units, 5, total_params=smrf._total_params(units))

    assert grid[0] == 0.0
    assert max(grid[1:]) / min(value for value in grid[1:] if value > 0) > 100.0


def test_smrf_builds_decision_unit_payload_from_probe_and_costs():
    stats = {}
    costs = {}
    for suffix in ("q_proj", "k_proj", "v_proj"):
        name = f"model.layers.0.self_attn.{suffix}"
        stats[name] = {
            "h_trace": 10.0,
            "n_params": 256,
            "in_features": 16,
            "out_features": 16,
        }
        costs[name] = {
            "NVFP4": {"output_mse": 0.1, "weight_mse": 0.01},
            "FP8_E4M3": {"output_mse": 0.02, "weight_mse": 0.002},
            "BF16": {"output_mse": 0.0, "weight_mse": 0.0},
        }

    payload = smrf.decision_unit_payload_from_probe_costs(
        {"stats": stats},
        {"costs": costs, "formats": ["NVFP4", "FP8_E4M3", "BF16"]},
        formats=["NVFP4", "FP8_E4M3", "BF16"],
        aggregate_siblings=True,
    )

    units = payload["blocks"]["model.layers.0"]["units"]
    assert list(units) == [
        "model.layers.0.self_attn.__siblings__.model__layers__0__self_attn__qkv_proj"
    ]
    unit = next(iter(units.values()))
    assert set(unit["members"]) == {
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    }
    assert set(unit["options"]) == {"NVFP4", "FP8_E4M3", "BF16"}
    assert unit["options"]["NVFP4"]["omega_ii"] > unit["options"]["FP8_E4M3"]["omega_ii"]


def test_smrf_probe_cost_bridge_excludes_profile_passthrough(monkeypatch):
    class Profile:
        def pinned_names(self):
            return ("lm_head",)

        def is_pinned_name(self, qname):
            return qname == "lm_head"

        def source_passthrough_prefixes(self):
            return ("mtp.",)

    monkeypatch.setattr(smrf, "_profile_for_model", lambda _model_path: Profile())

    stats = {}
    costs = {}
    for name in (
        "model.layers.0.self_attn.o_proj",
        "mtp.layers.0.self_attn.o_proj",
        "lm_head",
    ):
        stats[name] = {
            "h_trace": 10.0,
            "n_params": 256,
            "in_features": 16,
            "out_features": 16,
        }
        costs[name] = {
            "NVFP4": {"output_mse": 0.1, "weight_mse": 0.01},
            "BF16": {"output_mse": 0.0, "weight_mse": 0.0},
        }

    payload = smrf.decision_unit_payload_from_probe_costs(
        {"stats": stats},
        {"costs": costs, "formats": ["NVFP4", "BF16"]},
        model_path="dummy",
        formats=["NVFP4", "BF16"],
        aggregate_siblings=False,
    )

    units = payload["blocks"]["model.layers.0"]["units"]
    assert list(units) == ["model.layers.0.self_attn.o_proj"]
    assert "mtp.layers.0.self_attn.o_proj" not in payload["singletons"]
    assert "lm_head" not in payload["singletons"]
    assert payload["meta"]["profile_excluded_qnames"] == 2
    assert payload["meta"]["profile_excluded_sample"] == [
        "lm_head",
        "mtp.layers.0.self_attn.o_proj",
    ]


def test_smrf_builds_full_bpp_payload_from_l3_costs_with_frozen_baseline():
    stats = {}
    baseline = {}
    for suffix in ("q_proj", "k_proj", "v_proj", "o_proj"):
        name = f"model.layers.0.self_attn.{suffix}"
        stats[name] = {
            "h_trace": 10.0,
            "n_params": 256,
            "in_features": 16,
            "out_features": 16,
        }
        baseline[name] = "NVFP4"

    l3_costs = {}
    for suffix in ("q_proj", "k_proj", "v_proj"):
        name = f"model.layers.0.self_attn.{suffix}"
        l3_costs[name] = {
            "BF16": {"propagated_end_kl": 0.0},
            "NVFP4": {"propagated_end_kl": 0.12},
            "FP8_E4M3": {"propagated_end_kl": 0.03},
        }

    payload = smrf.decision_unit_payload_from_l3_costs(
        {"stats": stats},
        {"costs": l3_costs, "formats": ["NVFP4", "FP8_E4M3", "BF16"]},
        baseline,
        formats=["NVFP4", "FP8_E4M3", "BF16"],
        aggregate_siblings=True,
    )

    assert payload["meta"]["source"] == "l3_propagated_cost_bridge"
    assert payload["meta"]["measured_units_pre_aggregation"] == 3
    assert payload["meta"]["frozen_units_pre_aggregation"] == 1

    units = payload["blocks"]["model.layers.0"]["units"]
    qkv = units[
        "model.layers.0.self_attn.__siblings__.model__layers__0__self_attn__qkv_proj"
    ]
    assert set(qkv["members"]) == {
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    }
    assert set(qkv["options"]) == {"BF16", "NVFP4", "FP8_E4M3"}
    assert round(qkv["options"]["NVFP4"]["omega_ii"], 6) == 0.36
    assert set(units["model.layers.0.self_attn.o_proj"]["options"]) == {"NVFP4"}


def test_smrf_writes_manifest_and_assignment_files(tmp_path):
    archive = smrf.generate_archive_candidates(
        _payload(),
        output_dir=tmp_path,
        bpp_min=4.0,
        bpp_max=16.0,
        n_lambdas=5,
        validation_candidates=3,
    )

    manifest = archive["manifest"]
    assert manifest["schema"] == smrf.CANDIDATE_MANIFEST_SCHEMA
    assert Path(manifest["archive"]).exists()
    assert 0 < len(manifest["candidates"]) <= 3

    first = manifest["candidates"][0]
    assignment_payload = json.loads(Path(first["path"]).read_text())
    assert assignment_payload["schema"] == "prismaquant.smrf.assignment.v1"
    assert assignment_payload["assignment_hash"] == first["assignment_hash"]
    assert "model.layers.0.mlp.up_proj" in assignment_payload["assignment"]


def test_smrf_includes_baseline_assignment_in_validation_manifest(tmp_path):
    payload = _payload()
    baseline = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.self_attn.k_proj": "NVFP4",
        "model.layers.0.self_attn.v_proj": "NVFP4",
        "model.layers.0.self_attn.o_proj": "NVFP4",
        "model.layers.0.mlp.up_proj": "NVFP4",
        "model.layers.0.mlp.gate_proj": "NVFP4",
        "model.layers.1.self_attn.o_proj": "NVFP4",
    }

    archive = smrf.generate_archive_candidates(
        payload,
        output_dir=tmp_path,
        bpp_min=4.0,
        bpp_max=16.0,
        validation_candidates=2,
        include_assignments={"matched_pq": baseline},
    )

    manifest = archive["manifest"]
    included = [
        row for row in manifest["candidates"]
        if row["source"] == "included_assignment"
    ]
    assert len(included) == 1
    assert Path(included[0]["path"]).name.startswith("matched_pq_bpp_")
    assert included[0]["hamming_from_anchor"] == 0


def test_smrf_cli_generates_candidate_manifest(tmp_path):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_payload()))
    out_dir = tmp_path / "out"

    rc = smrf.main([
        "--payload",
        str(payload_path),
        "--output-dir",
        str(out_dir),
        "--bpp-min",
        "4.0",
        "--bpp-max",
        "16.0",
        "--n-lambdas",
        "5",
        "--validation-candidates",
        "2",
    ])

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert rc == 0
    assert manifest["schema"] == smrf.CANDIDATE_MANIFEST_SCHEMA
    assert len(manifest["candidates"]) == 2
