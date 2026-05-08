from __future__ import annotations

import json

import pytest

from prismaquant import block_clado as bc
from prismaquant import iterate_block_clado as ibc
from prismaquant.iterate_block_clado import assignment_for_units, load_assignment_json


def test_load_assignment_json_accepts_allocator_seed_payload(tmp_path):
    path = tmp_path / "seed.json"
    path.write_text(json.dumps({
        "schema": "prismaquant.allocator.pareto_assignment.v1",
        "assignment": {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.self_attn.k_proj": "BF16",
        },
    }))

    assert load_assignment_json(path) == {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.self_attn.k_proj": "BF16",
    }


def test_load_assignment_json_accepts_probe_selection_payload(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps({
        "selection": {
            "chosen": {
                "assignment": {
                    "model.layers.0.mlp.down_proj": "MXFP8_E4M3",
                },
            },
        },
    }))

    assert load_assignment_json(path) == {
        "model.layers.0.mlp.down_proj": "MXFP8_E4M3",
    }


def test_load_assignment_json_rejects_unknown_shape(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(["not", "an", "assignment"]))

    with pytest.raises(ValueError, match="unsupported assignment JSON shape"):
        load_assignment_json(path)


def test_assignment_for_units_normalizes_missing_and_illegal_formats():
    unit_a = bc.DecisionUnit(
        name="a",
        block_id="block",
        member_qnames=("a.q", "a.k"),
        options=(
            bc.FormatCost("NVFP4", 0.0, 4.0, 4),
            bc.FormatCost("BF16", 0.0, 16.0, 16),
        ),
    )
    unit_b = bc.DecisionUnit(
        name="lm_head",
        block_id="lm_head",
        member_qnames=("lm_head",),
        options=(bc.FormatCost("BF16", 0.0, 16.0, 16),),
    )

    assert assignment_for_units({"a.k": "mxfp8_e4m3", "lm_head": "NVFP4"}, [unit_a, unit_b]) == {
        "a.q": "BF16",
        "a.k": "BF16",
        "lm_head": "BF16",
    }
    assert assignment_for_units({"a.q": "nvfp4"}, [unit_a, unit_b]) == {
        "a.q": "NVFP4",
        "a.k": "NVFP4",
        "lm_head": "BF16",
    }


def test_run_iteration_keeps_center_when_validated_candidates_regress(tmp_path, monkeypatch):
    unit = bc.DecisionUnit(
        name="a",
        block_id="model.layers.0",
        member_qnames=("a.weight",),
        options=(
            bc.FormatCost("NVFP4", 0.0, 4.0, 50),
            bc.FormatCost("MXFP8_E4M3", 0.01, 8.0, 100),
            bc.FormatCost("BF16", -0.02, 16.0, 200),
        ),
    )
    payload = bc.units_and_pairs_to_payload(
        blocks={"model.layers.0": [unit]},
        singletons=[],
        pairs_by_block={"model.layers.0": []},
        meta={"elapsed_seconds": 0.0, "center_kl": 0.05, "centered": True},
    )

    monkeypatch.setattr(ibc, "collect_output_fisher", lambda *args, **kwargs: payload)
    monkeypatch.setattr(ibc, "measure_assignment_kl", lambda *args, **kwargs: 0.2)

    result = ibc.run_iteration(
        model=None,
        calib_ids=None,
        ref_log_probs=None,
        profile=None,
        formats=[],
        work_root=tmp_path / "work",
        iter_idx=0,
        center_assignment={"a.weight": "NVFP4"},
        center_label="seed",
        output_root=tmp_path / "out",
        n_neighbors_validate=4,
        skip_polish=True,
        measure_method="output_fisher",
    )

    assert result.best_validated_kl == pytest.approx(0.05)
    assert result.polished_kl == pytest.approx(0.05)
    assert result.best_validated_assignment == {"a.weight": "NVFP4"}

    validation = json.loads((tmp_path / "out" / "iter_0" / "validation.json").read_text())
    center_rows = [row for row in validation["rows"] if row.get("is_center_baseline")]
    assert len(center_rows) == 1
    assert center_rows[0]["real_kl"] == pytest.approx(0.05)
