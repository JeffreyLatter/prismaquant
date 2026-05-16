"""Unit tests for prismaquant.hadamard_duquant_allocator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant.hadamard_duquant_allocator import (
    ClusterCandidate,
    ClusterOverride,
    JointSearchSidecar,
    apply_cost_overrides,
    build_cluster_overrides,
    derive_picks,
    emit_picks,
    load_sidecar,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _write_test_sidecar(
    tmp_path: Path,
    *,
    cluster_key: str = "model.layers.0.attn.residual",
    consumer_qnames: list[str] | None = None,
    no_rot_nvfp4: float = 0.020,
    rot_nvfp4: float = 0.011,
    no_rot_mxfp8: float = 0.009,
    rot_mxfp8: float = 0.008,
    no_rot_bf16: float = 0.000,
    online: bool = True,
) -> Path:
    consumer_qnames = consumer_qnames or [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    payload = {
        "version": "1",
        "clusters": {
            cluster_key: {
                "insertion_kind": "residual",
                "group_size": 16,
                "input_dim": 4096,
                "online": online,
                "consumer_qnames": consumer_qnames,
                "producer_qnames": [],
                "candidates": {
                    "no_rot+NVFP4":     {"fisher_mse": no_rot_nvfp4, "bpp": 4.5},
                    "rot+NVFP4":        {"fisher_mse": rot_nvfp4,    "bpp": 4.5,
                                         "rotation_key": f"{cluster_key}/NVFP4/composed_matrix",
                                         "runtime_transform_type": "hadamard",
                                         "runtime_head_dim": 16},
                    "no_rot+MXFP8_E4M3":{"fisher_mse": no_rot_mxfp8, "bpp": 8.25},
                    "rot+MXFP8_E4M3":   {"fisher_mse": rot_mxfp8,    "bpp": 8.25,
                                         "rotation_key": f"{cluster_key}/MXFP8_E4M3/composed_matrix"},
                    "no_rot+BF16":      {"fisher_mse": no_rot_bf16,  "bpp": 16.0},
                },
            },
        },
    }
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(json.dumps(payload, indent=2))
    return sidecar_path


# ---------------------------------------------------------------------------
# Sidecar parsing
# ---------------------------------------------------------------------------


def test_load_sidecar_parses_version_and_clusters(tmp_path: Path):
    sidecar_path = _write_test_sidecar(tmp_path)
    parsed = load_sidecar(sidecar_path)
    assert parsed.version == "1"
    assert "model.layers.0.attn.residual" in parsed.clusters


def test_load_sidecar_parses_consumer_qnames(tmp_path: Path):
    sidecar_path = _write_test_sidecar(tmp_path)
    parsed = load_sidecar(sidecar_path)
    consumers = parsed.consumer_qnames("model.layers.0.attn.residual")
    assert consumers == (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    )


def test_load_sidecar_parses_candidates(tmp_path: Path):
    sidecar_path = _write_test_sidecar(tmp_path)
    parsed = load_sidecar(sidecar_path)
    candidates = parsed.candidates("model.layers.0.attn.residual")
    assert "rot+NVFP4" in candidates
    assert candidates["rot+NVFP4"].rotation == "rot"
    assert candidates["rot+NVFP4"].format_label == "NVFP4"
    assert candidates["rot+NVFP4"].fisher_mse == 0.011
    assert candidates["rot+NVFP4"].runtime_transform_type == "hadamard"
    assert candidates["rot+NVFP4"].runtime_head_dim == 16
    assert candidates["no_rot+BF16"].rotation == "no_rot"


def test_load_sidecar_skips_invalid_labels(tmp_path: Path):
    payload = {
        "version": "1",
        "clusters": {
            "c1": {
                "candidates": {
                    "garbage_label": {"fisher_mse": 0.1, "bpp": 4.0},
                    "rot+NVFP4": {"fisher_mse": 0.05, "bpp": 4.5},
                },
            },
        },
    }
    sidecar_path = tmp_path / "bad.json"
    sidecar_path.write_text(json.dumps(payload))
    parsed = load_sidecar(sidecar_path)
    cands = parsed.candidates("c1")
    assert "rot+NVFP4" in cands
    assert "garbage_label" not in cands


# ---------------------------------------------------------------------------
# build_cluster_overrides
# ---------------------------------------------------------------------------


def test_build_overrides_picks_min_per_format(tmp_path: Path):
    """For each format, override.best_cost_by_format is min(no_rot, rot)."""
    sidecar = load_sidecar(_write_test_sidecar(
        tmp_path,
        no_rot_nvfp4=0.020, rot_nvfp4=0.011,    # rot wins
        no_rot_mxfp8=0.009, rot_mxfp8=0.008,    # rot wins (barely)
    ))
    overrides = build_cluster_overrides(sidecar)
    ov = overrides["model.layers.0.attn.residual"]
    assert ov.best_cost_by_format["NVFP4"] == 0.011
    assert ov.virtual_picks["NVFP4"] == "rot"
    assert ov.best_cost_by_format["MXFP8_E4M3"] == 0.008
    assert ov.virtual_picks["MXFP8_E4M3"] == "rot"
    # BF16 has only no_rot
    assert ov.best_cost_by_format["BF16"] == 0.000
    assert ov.virtual_picks["BF16"] == "no_rot"


def test_build_overrides_chooses_no_rot_when_cheaper(tmp_path: Path):
    """If no_rot's cost is lower, virtual_pick is no_rot."""
    sidecar = load_sidecar(_write_test_sidecar(
        tmp_path,
        no_rot_nvfp4=0.005, rot_nvfp4=0.020,    # no_rot wins
    ))
    overrides = build_cluster_overrides(sidecar)
    ov = overrides["model.layers.0.attn.residual"]
    assert ov.best_cost_by_format["NVFP4"] == 0.005
    assert ov.virtual_picks["NVFP4"] == "no_rot"


def test_build_overrides_ignores_non_finite_format_scores(tmp_path: Path):
    """NaN/Inf sidecar scores should not poison allocator costs."""
    sidecar = load_sidecar(_write_test_sidecar(
        tmp_path,
        no_rot_nvfp4=float("nan"),
        rot_nvfp4=float("inf"),
    ))
    overrides = build_cluster_overrides(sidecar)
    ov = overrides["model.layers.0.attn.residual"]
    assert "NVFP4" not in ov.best_cost_by_format
    assert "NVFP4" not in ov.virtual_picks
    assert ov.best_cost_by_format["MXFP8_E4M3"] == 0.008


def test_build_overrides_carries_consumer_qnames(tmp_path: Path):
    sidecar = load_sidecar(_write_test_sidecar(tmp_path))
    overrides = build_cluster_overrides(sidecar)
    ov = overrides["model.layers.0.attn.residual"]
    assert "model.layers.0.self_attn.q_proj" in ov.consumer_qnames


def test_build_overrides_skips_clusters_without_candidates(tmp_path: Path):
    payload = {
        "version": "1",
        "clusters": {
            "empty": {"candidates": {}},
        },
    }
    sidecar_path = tmp_path / "empty.json"
    sidecar_path.write_text(json.dumps(payload))
    sidecar = load_sidecar(sidecar_path)
    assert build_cluster_overrides(sidecar) == {}


# ---------------------------------------------------------------------------
# apply_cost_overrides
# ---------------------------------------------------------------------------


def test_apply_cost_overrides_rewrites_per_qname_per_fmt(tmp_path: Path):
    """Each consumer gets an equal share of the cluster's min cost."""
    sidecar = load_sidecar(_write_test_sidecar(
        tmp_path,
        no_rot_nvfp4=0.020, rot_nvfp4=0.011,
    ))
    overrides = build_cluster_overrides(sidecar)

    # Build a cost table the allocator would normally load
    cost_table: dict[str, dict[str, dict]] = {
        qname: {
            "NVFP4": {"output_mse": 0.999, "bpp": 4.5},
            "MXFP8_E4M3": {"output_mse": 0.888, "bpp": 8.25},
            "BF16": {"output_mse": 0.0, "bpp": 16.0},
        }
        for qname in (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        )
    }

    n_modified = apply_cost_overrides(cost_table, overrides)
    # Three qnames × three formats with overrides each = 9 cells
    assert n_modified == 9
    for qname in cost_table:
        # Cluster scores are split across q/k/v so fused aggregation sums
        # back to the sidecar value rather than tripling it.
        assert cost_table[qname]["NVFP4"]["output_mse"] == pytest.approx(0.011 / 3)
        assert cost_table[qname]["MXFP8_E4M3"]["output_mse"] == pytest.approx(0.008 / 3)
        # BF16 has only no_rot (0.000)
        assert cost_table[qname]["BF16"]["output_mse"] == 0.000


def test_apply_cost_overrides_skips_missing_qname(tmp_path: Path):
    """qnames in the sidecar but absent from the cost table are skipped silently."""
    sidecar = load_sidecar(_write_test_sidecar(tmp_path))
    overrides = build_cluster_overrides(sidecar)
    cost_table: dict[str, dict[str, dict]] = {}
    n_modified = apply_cost_overrides(cost_table, overrides)
    assert n_modified == 0


def test_apply_cost_overrides_custom_cost_key(tmp_path: Path):
    """Caller can pick a different cost_key (e.g., 'fisher_output_mse')."""
    sidecar = load_sidecar(_write_test_sidecar(tmp_path))
    overrides = build_cluster_overrides(sidecar)
    cost_table: dict[str, dict[str, dict]] = {
        "model.layers.0.self_attn.q_proj": {
            "NVFP4": {"fisher_output_mse": 9.9, "output_mse": 8.8, "bpp": 4.5},
        },
    }
    apply_cost_overrides(cost_table, overrides, cost_key="fisher_output_mse")
    cell = cost_table["model.layers.0.self_attn.q_proj"]["NVFP4"]
    assert cell["fisher_output_mse"] == 0.011  # overridden
    assert cell["output_mse"] == 8.8  # untouched


# ---------------------------------------------------------------------------
# derive_picks
# ---------------------------------------------------------------------------


def test_derive_picks_recovers_chosen_rotation(tmp_path: Path):
    """Given assignment ⇒ cluster_format ⇒ rotation pick."""
    sidecar = load_sidecar(_write_test_sidecar(
        tmp_path,
        no_rot_nvfp4=0.020, rot_nvfp4=0.011,    # rot wins for NVFP4
    ))
    overrides = build_cluster_overrides(sidecar)
    # Allocator picks NVFP4 for all q/k/v
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.self_attn.k_proj": "NVFP4",
        "model.layers.0.self_attn.v_proj": "NVFP4",
    }
    picks = derive_picks(sidecar, overrides, assignment)
    assert picks["model.layers.0.attn.residual"] == "rot+NVFP4"


def test_derive_picks_no_rotation_when_no_rot_wins(tmp_path: Path):
    sidecar = load_sidecar(_write_test_sidecar(
        tmp_path,
        no_rot_nvfp4=0.005, rot_nvfp4=0.020,    # no_rot wins
    ))
    overrides = build_cluster_overrides(sidecar)
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.self_attn.k_proj": "NVFP4",
        "model.layers.0.self_attn.v_proj": "NVFP4",
    }
    picks = derive_picks(sidecar, overrides, assignment)
    assert picks["model.layers.0.attn.residual"] == "no_rot+NVFP4"


def test_derive_picks_no_rotation_when_format_is_bf16(tmp_path: Path):
    """BF16 has no rotation alternative ⇒ always no_rot+BF16."""
    sidecar = load_sidecar(_write_test_sidecar(tmp_path))
    overrides = build_cluster_overrides(sidecar)
    assignment = {
        "model.layers.0.self_attn.q_proj": "BF16",
        "model.layers.0.self_attn.k_proj": "BF16",
        "model.layers.0.self_attn.v_proj": "BF16",
    }
    picks = derive_picks(sidecar, overrides, assignment)
    assert picks["model.layers.0.attn.residual"] == "no_rot+BF16"


def test_derive_picks_skips_clusters_without_assignment(tmp_path: Path):
    """Clusters whose consumers aren't in the assignment are skipped."""
    sidecar = load_sidecar(_write_test_sidecar(tmp_path))
    overrides = build_cluster_overrides(sidecar)
    picks = derive_picks(sidecar, overrides, assignment={})
    assert picks == {}


def test_derive_picks_uses_first_available_consumer(tmp_path: Path):
    """Pick from the first consumer whose assignment is known."""
    sidecar = load_sidecar(_write_test_sidecar(tmp_path))
    overrides = build_cluster_overrides(sidecar)
    # Only k_proj is in the assignment
    assignment = {
        "model.layers.0.self_attn.k_proj": "MXFP8_E4M3",
    }
    picks = derive_picks(sidecar, overrides, assignment)
    assert picks["model.layers.0.attn.residual"].endswith("+MXFP8_E4M3")


# ---------------------------------------------------------------------------
# emit_picks
# ---------------------------------------------------------------------------


def test_emit_picks_writes_sorted_json(tmp_path: Path):
    picks = {
        "model.layers.5.attn.residual": "rot+NVFP4",
        "model.layers.0.attn.residual": "no_rot+BF16",
        "model.layers.3.mlp.down": "rot+MXFP8_E4M3",
    }
    out = tmp_path / "picks.json"
    emit_picks(picks, out)
    obj = json.loads(out.read_text())
    assert obj["version"] == "1"
    assert obj["picks"] == picks
    # Sorted JSON ⇒ deterministic across runs
    assert list(obj["picks"].keys()) == sorted(picks.keys())


def test_emit_picks_creates_parent_directory(tmp_path: Path):
    out = tmp_path / "nested" / "dir" / "picks.json"
    emit_picks({"c": "rot+NVFP4"}, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# End-to-end flow
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 6 helpers: load_picks, specs_from_sidecar, load_state_from_artifacts
# ---------------------------------------------------------------------------


def test_load_picks_round_trips_emit_picks(tmp_path: Path):
    from prismaquant.hadamard_duquant_allocator import load_picks
    original = {
        "model.layers.0.attn.residual": "rot+NVFP4",
        "model.layers.3.mlp.down": "no_rot+BF16",
    }
    out = tmp_path / "picks.json"
    emit_picks(original, out)
    loaded = load_picks(out)
    assert loaded == original


def test_specs_from_sidecar_reconstructs_consumer_qnames(tmp_path: Path):
    from prismaquant.hadamard_duquant_allocator import specs_from_sidecar
    sidecar = load_sidecar(_write_test_sidecar(tmp_path))
    specs = specs_from_sidecar(sidecar)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.cluster_key == "model.layers.0.attn.residual"
    assert spec.input_dim == 4096
    assert spec.group_size == 16
    assert spec.online is True
    assert "model.layers.0.self_attn.q_proj" in spec.consumer_qnames


def test_specs_from_sidecar_skips_invalid_kind(tmp_path: Path):
    from prismaquant.hadamard_duquant_allocator import specs_from_sidecar
    payload = {
        "version": "1",
        "clusters": {
            "c1": {
                "insertion_kind": "unknown_kind",
                "group_size": 16,
                "input_dim": 64,
                "online": False,
                "consumer_qnames": ["q"],
                "producer_qnames": [],
                "candidates": {"no_rot+NVFP4": {"fisher_mse": 0.1, "bpp": 4.5}},
            },
        },
    }
    p = tmp_path / "bad_kind.json"
    p.write_text(json.dumps(payload))
    sidecar = load_sidecar(p)
    assert specs_from_sidecar(sidecar) == []


def test_load_state_from_artifacts_builds_cache_state(tmp_path: Path):
    """End-to-end Phase 5→3 bridge: sidecar+rotations+picks ⇒ cache state."""
    from safetensors.torch import save_file as save_safetensors
    import torch

    from prismaquant.hadamard_duquant import (
        NVFP4_GROUP_SIZE,
        sylvester_hadamard,
    )
    from prismaquant.hadamard_duquant_allocator import load_state_from_artifacts

    sidecar_path = _write_test_sidecar(tmp_path)
    # Write a rotation matrix at the expected safetensors key
    rotations_path = tmp_path / "rotations.safetensors"
    save_safetensors(
        {
            "model.layers.0.attn.residual/NVFP4/composed_matrix":
                sylvester_hadamard(NVFP4_GROUP_SIZE).contiguous(),
        },
        str(rotations_path),
    )
    # Emit picks that select rotation at NVFP4
    picks_path = tmp_path / "picks.json"
    emit_picks({"model.layers.0.attn.residual": "rot+NVFP4"}, picks_path)

    state = load_state_from_artifacts(sidecar_path, rotations_path, picks_path)
    assert not state.is_empty()
    rotation = state.rotation_for_consumer("model.layers.0.self_attn.q_proj")
    assert rotation is not None
    assert rotation.format_label == "NVFP4"
    assert rotation.group_size == NVFP4_GROUP_SIZE


def test_load_state_from_artifacts_empty_when_no_rotation_picked(tmp_path: Path):
    """If picks select no rotations, the state is empty (no safetensors needed)."""
    from prismaquant.hadamard_duquant_allocator import load_state_from_artifacts

    sidecar_path = _write_test_sidecar(tmp_path)
    picks_path = tmp_path / "picks.json"
    emit_picks({"model.layers.0.attn.residual": "no_rot+BF16"}, picks_path)
    state = load_state_from_artifacts(
        sidecar_path,
        rotation_safetensors_path=tmp_path / "does_not_exist.safetensors",
        picks_path=picks_path,
    )
    assert state.is_empty()


def test_full_pipeline_rot_wins_at_nvfp4(tmp_path: Path):
    """Sidecar → overrides → apply to cost table → DP decision → derive picks."""
    sidecar = load_sidecar(_write_test_sidecar(
        tmp_path,
        no_rot_nvfp4=0.020, rot_nvfp4=0.011,
    ))
    overrides = build_cluster_overrides(sidecar)
    cost_table: dict[str, dict[str, dict]] = {
        qname: {
            "NVFP4": {"output_mse": 999.0, "bpp": 4.5},
            "MXFP8_E4M3": {"output_mse": 999.0, "bpp": 8.25},
            "BF16": {"output_mse": 0.0, "bpp": 16.0},
        }
        for qname in (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        )
    }
    apply_cost_overrides(cost_table, overrides)
    # Simulate DP picking NVFP4 (which now has the override cost = 0.011)
    assignment = {qname: "NVFP4" for qname in cost_table}
    picks = derive_picks(sidecar, overrides, assignment)
    assert picks["model.layers.0.attn.residual"] == "rot+NVFP4"
