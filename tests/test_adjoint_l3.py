import pytest
import json
import pickle

from prismaquant import adjoint_l3 as l3a
from prismaquant import format_registry as fr


def _payload() -> dict:
    return {
        "schema": l3a.SCHEMA,
        "rank": 2,
        "units": {
            "layer.a": {
                "formats": {
                    "NVFP4": {"sketch": [1.0, 3.0], "diagonal_cost": 0.5},
                    "BF16": {"sketch": [0.0, 0.0], "diagonal_cost": 0.0},
                }
            }
        },
    }


def test_validate_rejects_rank_mismatch():
    payload = _payload()
    payload["units"]["layer.a"]["formats"]["NVFP4"]["sketch"] = [1.0]

    with pytest.raises(ValueError, match="sketch rank"):
        l3a.validate_adjoint_l3_payload(payload)


def test_load_probe_stats_accepts_raw_and_wrapped_probe(tmp_path):
    import pickle

    raw_path = tmp_path / "raw.pkl"
    wrapped_path = tmp_path / "wrapped.pkl"
    stats = {"layer.a": {"n_params": 1}}
    with raw_path.open("wb") as fh:
        pickle.dump(stats, fh)
    with wrapped_path.open("wb") as fh:
        pickle.dump({"stats": stats, "meta": {}}, fh)

    assert l3a._load_probe_stats(raw_path) == stats
    assert l3a._load_probe_stats(wrapped_path) == stats


def test_load_assignment_json_accepts_layer_config_entries(tmp_path):
    path = tmp_path / "layer_config.json"
    path.write_text(json.dumps({
        "layer.fp16": {"bits": 16, "data_type": "float"},
        "layer.nvfp4": {
            "bits": 4,
            "data_type": "fp4_e2m1",
            "act_bits": 8,
            "act_data_type": "fp8_e4m3",
        },
        "layer.mxfp8": {
            "bits": 8,
            "data_type": "fp8_e4m3",
            "group_size": 32,
            "act_bits": 8,
        },
    }))

    assignment = l3a._load_assignment_json(path)

    assert assignment["layer.fp16"] == "BF16"
    assert assignment["layer.nvfp4"] == "NVFP4"
    assert assignment["layer.mxfp8"] == "MXFP8_E4M3"


def test_unary_adjoint_cost_builds_legacy_candidates():
    payload = _payload()
    stats = {
        "layer.a": {
            "n_params": 256,
            "in_features": 16,
            "out_features": 16,
        }
    }
    formats = [fr.get_format("NVFP4"), fr.get_format("BF16")]

    candidates = l3a.build_adjoint_l3_candidates(stats, payload, formats)

    nvfp4 = next(c for c in candidates["layer.a"] if c.fmt == "NVFP4")
    bf16 = next(c for c in candidates["layer.a"] if c.fmt == "BF16")
    assert nvfp4.predicted_dloss == pytest.approx(3.0)
    assert bf16.predicted_dloss == pytest.approx(0.0)
    assert nvfp4.memory_bytes == fr.get_format("NVFP4").memory_bytes_for_shape((16, 16))
    assert bf16.bits_per_param == pytest.approx(
        fr.get_format("BF16").effective_bits_for_shape((16, 16))
    )


def test_payload_converts_to_legacy_propagated_costs_without_memory_fields():
    propagated = l3a.adjoint_payload_to_propagated_costs(_payload())

    assert propagated["layer.a"]["NVFP4"]["propagated_end_kl"] == pytest.approx(3.0)
    assert propagated["layer.a"]["NVFP4"]["adjoint_l3_diagonal_cost"] == pytest.approx(
        0.5
    )
    assert propagated["layer.a"]["BF16"]["propagated_end_kl"] == pytest.approx(0.0)


def test_payload_converts_to_resume_l3_pickle_shape():
    resume = l3a.adjoint_payload_to_l3_resume_payload(
        _payload(),
        formats=["NVFP4", "BF16"],
        meta={"anchor_bpp": 5.5},
    )

    assert "costs" in resume
    assert resume["cost_history"] == [resume["costs"]]
    assert resume["formats"] == ["NVFP4", "BF16"]
    assert resume["meta"]["source_schema"] == l3a.SCHEMA
    assert resume["meta"]["anchor_bpp"] == pytest.approx(5.5)


def test_retune_adjoint_diagonal_costs_uses_mse_floor_components():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "meta": {"mse_floor_scale": 10.0},
        "units": {
            "layer.a": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "adjoint_self_cost": 0.0,
                        "output_delta_mse": 0.0,
                    },
                    "NVFP4": {
                        "sketch": [2.0],
                        "diagonal_cost": 2.0,
                        "adjoint_self_cost": 2.0,
                        "output_delta_mse": 5.0,
                    },
                }
            }
        },
    }

    retuned = l3a.retune_adjoint_diagonal_costs(
        payload,
        diagonal_floor_frac=1.5,
        mse_diagonal_floor_frac=0.2,
    )

    formats = retuned["units"]["layer.a"]["formats"]
    assert formats["NVFP4"]["diagonal_cost"] == pytest.approx(13.0)
    assert formats["NVFP4"]["mse_floor_cost"] == pytest.approx(10.0)
    assert formats["BF16"]["diagonal_cost"] == pytest.approx(0.0)
    assert payload["units"]["layer.a"]["formats"]["NVFP4"]["diagonal_cost"] == 2.0


def test_solver_accepts_mxfp8_alias_and_can_choose_it():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "layer.a": {
                "formats": {
                    "NVFP4": {
                        "sketch": [3.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                    "MXFP8": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.1,
                        "bits_per_param": 8.0,
                        "memory_bytes": 8,
                    },
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                }
            }
        },
    }
    units = l3a.adjoint_units_from_payload(payload)

    result = l3a.solve_low_rank_lagrangian(units, 1)

    assert result.assignment == {"layer.a": "MXFP8_E4M3"}
    assert result.objective == pytest.approx(0.1)


def test_collapse_assignment_to_solve_units_majority_seeds_fused_groups():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "group": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            }
        },
    }
    solve_units = l3a.adjoint_units_from_payload(payload)

    collapsed = l3a.collapse_assignment_to_solve_units(
        {
            "raw.a": "BF16",
            "raw.b": "NVFP4",
            "raw.c": "NVFP4",
        },
        solve_units,
        {"group": ("raw.a", "raw.b", "raw.c")},
    )

    assert collapsed == {"group": "NVFP4"}


def test_score_assignment_keeps_psd_cross_terms():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "a": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [2.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
            "b": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [-2.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
        },
    }
    units = l3a.adjoint_units_from_payload(payload)

    both_nvfp4 = l3a.score_adjoint_assignment(
        units,
        1,
        {"a": "NVFP4", "b": "NVFP4"},
    )
    one_nvfp4 = l3a.score_adjoint_assignment(
        units,
        1,
        {"a": "NVFP4", "b": "BF16"},
    )

    assert both_nvfp4[0] == pytest.approx(0.0)
    assert one_nvfp4[0] == pytest.approx(3.0)
    assert both_nvfp4[4] == pytest.approx((0.0,))


def test_build_move_report_scores_revert_delta_in_final_context():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "a": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [2.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
            "b": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [-2.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
        },
    }
    units = l3a.adjoint_units_from_payload(payload)

    report = l3a.build_move_report(
        units,
        1,
        {"a": "NVFP4", "b": "NVFP4"},
        {"a": "BF16", "b": "BF16"},
    )

    assert len(report) == 2
    assert {row["name"] for row in report} == {"a", "b"}
    assert all(row["delta_objective_vs_revert"] == pytest.approx(-3.0) for row in report)
    assert all(row["delta_bits"] == pytest.approx(-96.0) for row in report)


def test_low_rank_solver_can_take_cancellation_move_from_seed():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "a": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [2.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
            "b": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [-2.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
        },
    }
    units = l3a.adjoint_units_from_payload(payload)

    result = l3a.solve_low_rank_lagrangian(
        units,
        1,
        initial_assignment={"a": "NVFP4", "b": "BF16"},
    )

    assert result.assignment == {"a": "NVFP4", "b": "NVFP4"}
    assert result.objective == pytest.approx(0.0)
    assert result.moves == 1


def test_budget_sweep_can_use_seed_assignment_multistart():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "a": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [2.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
            "b": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [-2.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
        },
    }
    units = l3a.adjoint_units_from_payload(payload)

    result = l3a.solve_low_rank_budget_sweep(
        units,
        1,
        target_bits_total=10_000.0,
        lambdas=(0.0,),
        initial_assignments=(None, {"a": "NVFP4", "b": "NVFP4"}),
        max_passes=0,
    )

    assert result.assignment == {"a": "NVFP4", "b": "NVFP4"}
    assert result.objective == pytest.approx(0.0)


def test_low_rank_solver_uses_pair_move_for_cancellation_from_unary_seed():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "a": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [2.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
            "b": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [-2.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
        },
    }
    units = l3a.adjoint_units_from_payload(payload)

    result = l3a.solve_low_rank_lagrangian(units, 1)

    assert result.assignment == {"a": "NVFP4", "b": "NVFP4"}
    assert result.objective == pytest.approx(0.0)
    assert result.moves == 1


def test_low_rank_solver_respects_changed_unit_trust_region():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "a": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
            "b": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
        },
    }
    units = l3a.adjoint_units_from_payload(payload)

    result = l3a.solve_low_rank_lagrangian(
        units,
        1,
        reference_assignment={"a": "BF16", "b": "BF16"},
        max_changed_units=1,
    )

    assert result.changed_units == 1
    assert list(result.assignment.values()).count("NVFP4") == 1
    assert result.objective == pytest.approx(1.0)


def test_low_rank_solver_can_forbid_reference_downgrades():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "protected": {
                "formats": {
                    "BF16": {
                        "sketch": [10.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
            "upgrade": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [0.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
        },
    }
    units = l3a.adjoint_units_from_payload(payload)

    result = l3a.solve_low_rank_lagrangian(
        units,
        1,
        reference_assignment={"protected": "BF16", "upgrade": "NVFP4"},
        forbid_reference_downgrades=True,
    )

    assert result.assignment["protected"] == "BF16"
    assert result.assignment["upgrade"] == "BF16"
    assert result.changed_units == 1


def test_budget_polish_repairs_over_budget_solution_with_lowest_damage():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "a": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.2,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
            "b": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [0.0],
                        "diagonal_cost": 2.0,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
        },
    }
    units = l3a.adjoint_units_from_payload(payload)
    seed = l3a.solve_low_rank_lagrangian(units, 1, lambda_penalty=0.0)

    repaired = l3a.polish_low_rank_to_budget(
        units,
        1,
        seed,
        target_bits_total=160.0,
    )

    assert repaired.bits_total <= 160.0
    assert repaired.assignment == {"a": "NVFP4", "b": "BF16"}
    assert repaired.objective == pytest.approx(0.2)


def test_budget_polish_prefers_non_crossing_move_over_huge_undershoot():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "huge": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 100,
                    },
                    "NVFP4": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.1,
                        "bits_per_param": 4.5,
                        "memory_bytes": 1,
                    },
                }
            },
            "small": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 100,
                    },
                    "NVFP4": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.2,
                        "bits_per_param": 4.5,
                        "memory_bytes": 80,
                    },
                }
            },
        },
    }
    units = l3a.adjoint_units_from_payload(payload)
    seed = l3a.solve_low_rank_lagrangian(units, 1, lambda_penalty=0.0)

    repaired = l3a.polish_low_rank_to_budget(
        units,
        1,
        seed,
        target_bits_total=1440.0,
    )

    assert repaired.bits_total == pytest.approx(1440.0)
    assert repaired.assignment == {"huge": "BF16", "small": "NVFP4"}


def test_group_adjoint_units_by_profile_sums_fused_sibling_options():
    payload = {
        "schema": l3a.SCHEMA,
        "rank": 2,
        "units": {
            "layer.q_proj": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0, 0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [1.0, 2.0],
                        "diagonal_cost": 0.1,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
            "layer.k_proj": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0, 0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [-0.25, 3.0],
                        "diagonal_cost": 0.2,
                        "bits_per_param": 4.5,
                        "memory_bytes": 4,
                    },
                }
            },
        },
    }

    class _Profile:
        def fused_sibling_group(self, name):
            if name in {"layer.q_proj", "layer.k_proj"}:
                return "layer.qkv_proj"
            return None

    units = l3a.adjoint_units_from_payload(payload)
    grouped, members = l3a.group_adjoint_units_by_profile(units, _Profile())

    assert tuple(unit.name for unit in grouped) == ("layer.qkv_proj",)
    assert members == {"layer.qkv_proj": ("layer.k_proj", "layer.q_proj")}
    nvfp4 = {opt.fmt: opt for opt in grouped[0].options}["NVFP4"]
    assert nvfp4.sketch == pytest.approx((0.75, 5.0))
    assert nvfp4.diagonal_cost == pytest.approx(0.3)
    assert nvfp4.memory_bytes == 8

    expanded = l3a.expand_grouped_assignment(
        {"layer.qkv_proj": "NVFP4"},
        members,
    )
    assert expanded == {"layer.k_proj": "NVFP4", "layer.q_proj": "NVFP4"}


def test_cli_target_full_bpp_subtracts_fixed_base_assignment_bits(tmp_path):
    payload_path = tmp_path / "adjoint.json"
    probe_path = tmp_path / "probe.pkl"
    base_path = tmp_path / "base.json"
    output_path = tmp_path / "solve.json"
    full_output_path = tmp_path / "solve_full.json"

    payload_path.write_text(json.dumps({
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "layer.a": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.1,
                        "bits_per_param": 4.0,
                        "memory_bytes": 4,
                    },
                }
            }
        },
    }))
    stats = {
        "layer.a": {
            "n_params": 8,
            "_memory_bytes_by_format": {"BF16": 16, "NVFP4": 4},
        },
        "layer.fixed": {
            "n_params": 8,
            "_memory_bytes_by_format": {"BF16": 16, "NVFP4": 4},
        },
    }
    with probe_path.open("wb") as fh:
        pickle.dump({"stats": stats}, fh)
    base_path.write_text(json.dumps({
        "assignment": {
            "layer.a": "BF16",
            "layer.fixed": "BF16",
        }
    }))

    rc = l3a.main([
        "--adjoint-costs", str(payload_path),
        "--probe", str(probe_path),
        "--formats", "NVFP4,BF16",
        "--target-full-bpp", "10.0",
        "--base-assignment", str(base_path),
        "--output", str(output_path),
        "--full-assignment-output", str(full_output_path),
    ])

    assert rc == 0
    solved = json.loads(output_path.read_text())
    assert solved["assignment"] == {"layer.a": "NVFP4"}
    assert solved["bits_total"] == pytest.approx(32.0)
    assert solved["meta"]["fixed_entry_count"] == 1
    assert solved["meta"]["fixed_bits"] == pytest.approx(128.0)
    assert solved["meta"]["computed_target_total_bits"] == pytest.approx(32.0)
    assert solved["meta"]["solved_full_bits_total"] == pytest.approx(160.0)
    assert solved["meta"]["solved_full_bpp"] == pytest.approx(10.0)
    assert solved["meta"]["target_feasible"] is True
    assert solved["meta"]["achieved_solved_bits_total"] == pytest.approx(32.0)

    full_solved = json.loads(full_output_path.read_text())
    assert full_solved["assignment"] == {
        "layer.a": "NVFP4",
        "layer.fixed": "BF16",
    }
    assert full_solved["meta"]["assignment_scope"] == "base_plus_adjoint_overlay"
