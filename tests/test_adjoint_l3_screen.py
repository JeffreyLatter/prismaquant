import json

import pytest

from prismaquant import adjoint_l3 as l3a
from prismaquant import adjoint_l3_screen as screen


def _units():
    return [
        l3a.AdjointUnit(
            name="model.layers.1.mlp.gate_up_proj",
            options=(
                l3a.AdjointFormatOption(
                    name="model.layers.1.mlp.gate_up_proj",
                    fmt="NVFP4",
                    sketch=(1.0, 0.0),
                    diagonal_cost=0.1,
                    bits_per_param=4.0,
                    memory_bytes=4,
                ),
                l3a.AdjointFormatOption(
                    name="model.layers.1.mlp.gate_up_proj",
                    fmt="BF16",
                    sketch=(0.0, 0.0),
                    diagonal_cost=0.0,
                    bits_per_param=16.0,
                    memory_bytes=16,
                ),
            ),
        ),
        l3a.AdjointUnit(
            name="model.layers.3.linear_attn.in_proj_qkvz",
            options=(
                l3a.AdjointFormatOption(
                    name="model.layers.3.linear_attn.in_proj_qkvz",
                    fmt="NVFP4",
                    sketch=(0.0, 1.0),
                    diagonal_cost=0.1,
                    bits_per_param=4.0,
                    memory_bytes=4,
                ),
                l3a.AdjointFormatOption(
                    name="model.layers.3.linear_attn.in_proj_qkvz",
                    fmt="MXFP8_E4M3",
                    sketch=(0.0, 0.5),
                    diagonal_cost=0.05,
                    bits_per_param=8.0,
                    memory_bytes=8,
                ),
            ),
        ),
    ]


def test_reference_upgrade_rows_scores_one_unit_moves():
    rows = screen.reference_upgrade_rows(
        _units(),
        2,
        {
            "model.layers.1.mlp.gate_up_proj": "NVFP4",
            "model.layers.3.linear_attn.in_proj_qkvz": "NVFP4",
        },
    )

    assert len(rows) == 2
    assert rows[0]["layer_index"] == 1
    assert rows[0]["module_kind"] == "mlp.gate_up"
    assert {row["to_format"] for row in rows} == {"BF16", "MXFP8_E4M3"}
    assert all(row["delta_bits"] > 0 for row in rows)


def test_select_diverse_upgrade_rows_keeps_unique_moves():
    rows = [
        {
            "name": f"model.layers.{idx}.mlp.gate_up_proj",
            "members": [f"model.layers.{idx}.mlp.gate_up_proj"],
            "module_kind": "mlp.gate_up",
            "layer_index": idx,
            "from_format": "NVFP4",
            "to_format": "BF16",
            "delta_objective": -float(idx),
            "delta_diagonal": -0.1 * float(idx),
            "delta_low_rank": -0.2 * float(idx),
            "delta_bits": 10.0,
        }
        for idx in range(4)
    ]

    selected = screen.select_diverse_upgrade_rows(rows, max_candidates=3, per_bucket=2)

    assert len(selected) == 3
    assert len({(row["name"], row["to_format"]) for row in selected}) == 3
    assert all("selection_bucket" in row for row in selected)


def test_combo_writer_uses_only_validated_positive_moves(tmp_path):
    base = {
        "model.layers.1.mlp.gate_up_proj": "NVFP4",
        "model.layers.3.linear_attn.in_proj_qkvz": "NVFP4",
    }
    proposals = [
        {
            "label": "good_a",
            "unit": "model.layers.1.mlp.gate_up_proj",
            "members": ["model.layers.1.mlp.gate_up_proj"],
            "to": "BF16",
            "delta_bits": 12.0,
        },
        {
            "label": "bad_b",
            "name": "model.layers.3.linear_attn.in_proj_qkvz",
            "members": ["model.layers.3.linear_attn.in_proj_qkvz"],
            "to_format": "MXFP8_E4M3",
            "delta_bits": 4.0,
        },
    ]
    validation = {
        "old": {"last_token_kl": 0.5, "bpp": 5.0},
        "good_a": {"last_token_kl": 0.4, "bpp": 5.1},
        "bad_b": {"last_token_kl": 0.6, "bpp": 5.0},
    }

    combos = screen.write_combo_assignments_from_validation(
        output_dir=tmp_path,
        base_assignment=base,
        proposals=proposals,
        validation_results=validation,
        base_label="old",
    )

    assert len(combos) == 1
    assert combos[0]["sum_measured_delta_bpp"] == pytest.approx(0.1)
    assignment = json.loads(open(combos[0]["assignment_path"]).read())
    assert assignment["model.layers.1.mlp.gate_up_proj"] == "BF16"
    assert assignment["model.layers.3.linear_attn.in_proj_qkvz"] == "NVFP4"


def test_combo_writer_can_cap_measured_bpp_increases(tmp_path):
    base = {
        "model.layers.1.mlp.gate_up_proj": "NVFP4",
        "model.layers.3.linear_attn.in_proj_qkvz": "NVFP4",
    }
    proposals = [
        {
            "label": "small_gain",
            "name": "model.layers.1.mlp.gate_up_proj",
            "members": ["model.layers.1.mlp.gate_up_proj"],
            "to_format": "MXFP8_E4M3",
            "delta_bits": 4.0,
        },
        {
            "label": "large_gain",
            "name": "model.layers.3.linear_attn.in_proj_qkvz",
            "members": ["model.layers.3.linear_attn.in_proj_qkvz"],
            "to_format": "BF16",
            "delta_bits": 12.0,
        },
    ]
    validation = {
        "old": {"last_token_kl": 1.0, "bpp": 5.0},
        "small_gain": {"last_token_kl": 0.9, "bpp": 5.04},
        "large_gain": {"last_token_kl": 0.7, "bpp": 5.4},
    }

    combos = screen.write_combo_assignments_from_validation(
        output_dir=tmp_path,
        base_assignment=base,
        proposals=proposals,
        validation_results=validation,
        base_label="old",
        max_move_delta_bpp=0.1,
    )

    assert [combo["moves"][0]["label"] for combo in combos] == ["small_gain"]

    combos = screen.write_combo_assignments_from_validation(
        output_dir=tmp_path / "combo_cap",
        base_assignment=base,
        proposals=proposals,
        validation_results=validation,
        base_label="old",
        max_combo_delta_bpp=0.1,
    )

    assert [combo["moves"][0]["label"] for combo in combos] == ["small_gain"]
