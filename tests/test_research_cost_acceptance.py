from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from prismaquant.research_cost_acceptance import (
    RESEARCH_COST_PROVENANCE,
    accepted_cost_provenance,
    assemble_research_cost_table,
    enforce_research_export_acknowledgement,
    propagated_cost_provenance,
)


def _entry(value: float, **extra) -> dict:
    return {"output_mse": value, "weight_mse": value * 2, **extra}


def _write(path: Path, payload: dict) -> None:
    path.write_bytes(pickle.dumps(payload))


def test_assembly_checks_rows_layer_keys_and_base_precedence(tmp_path):
    formats = tuple(f"NVFP4_CB_K{k}" for k in range(12, 19))
    base_costs = {}
    segments = tmp_path / "segments"
    segments.mkdir()
    for layer in range(2):
        costs = {}
        for row in range(2):
            name = f"model.layers.{layer}.linear_{row}"
            costs[name] = {fmt: _entry(layer * 10 + row + k) for k, fmt in enumerate(formats)}
            base_costs[name] = {
                "NVFP4_CB_K14": _entry(100 + layer * 10 + row),
                "NVFP4_CB_K15": _entry(layer * 10 + row + 3),
                "FP8_CB_K36": _entry(0.5),
            }
        _write(segments / f"layer_{layer:03d}.pkl", {
            "costs": costs, "formats": list(formats), "provenance": {}, "meta": {}
        })
    base = tmp_path / "base.pkl"
    _write(base, {
        "costs": base_costs,
        "formats": ["NVFP4_CB_K14", "NVFP4_CB_K15", "FP8_CB_K36"],
        "provenance": {"production": True},
        "meta": {},
    })
    out = tmp_path / "accepted.pkl"
    assembled, manifest = assemble_research_cost_table(
        base, segments, output_path=out,
        expected_layers=2, expected_rows_per_layer=2,
    )

    assert len(assembled["costs"]) == 4
    assert manifest["assembled_row_count"] == 4
    assert [row["layer"] for row in manifest["layers"]] == [0, 1]
    assert all(row["row_count"] == 2 for row in manifest["layers"])
    assert assembled["costs"]["model.layers.1.linear_1"]["NVFP4_CB_K14"] == _entry(111)
    assert manifest["precedence"] == "production_v2_base_wins_every_overlapping_cell"
    assert manifest["k14_k15_cross_run_bit_equality"]["NVFP4_CB_K15"][
        "all_full_entries_bit_equal"
    ] is True
    assert manifest["k14_k15_cross_run_bit_equality"]["NVFP4_CB_K14"][
        "all_full_entries_bit_equal"
    ] is False
    assert accepted_cost_provenance(pickle.loads(out.read_bytes()))[
        "cost_provenance"
    ] == RESEARCH_COST_PROVENANCE


def test_assembly_rejects_wrong_layer_keying(tmp_path):
    segments = tmp_path / "segments"
    segments.mkdir()
    formats = [f"NVFP4_CB_K{k}" for k in range(12, 19)]
    row = {fmt: _entry(1.0) for fmt in formats}
    _write(segments / "layer_000.pkl", {
        "costs": {"model.layers.1.linear": row}, "formats": formats
    })
    base = tmp_path / "base.pkl"
    _write(base, {"costs": {"model.layers.1.linear": {}}, "formats": []})
    with pytest.raises(ValueError, match="does not match row keying"):
        assemble_research_cost_table(
            base, segments, expected_layers=1, expected_rows_per_layer=1
        )


def test_stamp_propagation_and_export_gate_refusal():
    manifest = {
        "schema": "prismaquant.research_cost_manifest.v1",
        "cost_provenance": RESEARCH_COST_PROVENANCE,
        "assembled_row_count": 1,
    }
    selection = propagated_cost_provenance(manifest)
    layer_config = {
        "model.layers.0.linear": "NVFP4_CB_K12",
        "__prismaquant__": {"cost_provenance": selection["cost_provenance"]},
    }
    round_trip = json.loads(json.dumps(layer_config))
    with pytest.raises(ValueError, match="refusing to export"):
        enforce_research_export_acknowledgement(
            round_trip, acknowledged=False, where="test exporter"
        )
    accepted = enforce_research_export_acknowledgement(
        round_trip, acknowledged=True, where="test exporter"
    )
    assert accepted == manifest


def test_export_gate_is_inert_for_unstamped_selection():
    assert propagated_cost_provenance(None) == {}
    assert enforce_research_export_acknowledgement(
        {"model.layers.0.linear": "BF16"},
        acknowledged=False,
        where="test exporter",
    ) is None
