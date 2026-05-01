import pytest

from prismaquant.iterate_perturbed_allocation import (
    assignment_hash,
    build_l3_polish_summary,
    resolve_two_cycle,
    smooth_cost_history,
    weighted_hamming_fraction,
)
from prismaquant.propagated_cost import L3NeighborhoodEntry


def test_cost_ema_uses_geometric_decay_and_skips_errors():
    history = [
        {
            "layer": {
                "NVFP4": {"predicted_dloss": 4.0, "output_mse": 8.0},
                "MXFP8": {"error": "failed"},
            }
        },
        {
            "layer": {
                "NVFP4": {
                    "predicted_dloss": 2.0,
                    "output_mse": 6.0,
                    "output_mse_measured": False,
                },
                "MXFP8": {"predicted_dloss": 1.0},
            }
        },
    ]

    smoothed = smooth_cost_history(history, decay=0.5)

    assert smoothed["layer"]["NVFP4"]["predicted_dloss"] == pytest.approx(
        (2.0 + 0.5 * 4.0) / 1.5
    )
    assert smoothed["layer"]["NVFP4"]["output_mse"] == pytest.approx(
        (6.0 + 0.5 * 8.0) / 1.5
    )
    assert smoothed["layer"]["NVFP4"]["output_mse_measured"] is False
    assert smoothed["layer"]["MXFP8"]["predicted_dloss"] == pytest.approx(1.0)


def test_weighted_hamming_uses_predicted_dloss_delta():
    old = {"a": "NVFP4", "b": "NVFP4"}
    new = {"a": "MXFP8", "b": "NVFP4"}
    costs = {
        "a": {
            "NVFP4": {"predicted_dloss": 10.0},
            "MXFP8": {"predicted_dloss": 4.0},
        },
        "b": {"NVFP4": {"predicted_dloss": 6.0}},
    }

    got = weighted_hamming_fraction(old, new, costs)

    assert got == pytest.approx(0.6)


def test_cycle_detection_re_solves_on_averaged_costs():
    a = {"layer": "NVFP4"}
    b = {"layer": "MXFP8"}
    c = {"layer": "BF16"}

    resolved, mode = resolve_two_cycle(
        a,
        b,
        a,
        {"layer": {"NVFP4": {"predicted_dloss": 1.0}}},
        {"layer": {"MXFP8": {"predicted_dloss": 1.0}}},
        lambda _costs: c,
        lambda _assignment: 0.0,
    )

    assert resolved == c
    assert mode == "averaged-costs"


def test_cycle_detection_kl_tie_breaks_if_average_stays_endpoint():
    a = {"layer": "NVFP4"}
    b = {"layer": "MXFP8"}
    kl = {assignment_hash(a): 3.0, assignment_hash(b): 1.0}

    resolved, mode = resolve_two_cycle(
        a,
        b,
        a,
        {"layer": {"NVFP4": {"predicted_dloss": 1.0}}},
        {"layer": {"MXFP8": {"predicted_dloss": 1.0}}},
        lambda _costs: a,
        lambda assignment: kl[assignment_hash(assignment)],
    )

    assert resolved == b
    assert mode == "kl-prev"


def test_l3_polish_summary_reports_flips_and_regression():
    selected = [
        L3NeighborhoodEntry(
            name="layer",
            current_format="MXFP8",
            formats=("NVFP4", "MXFP8", "BF16"),
            margin=0.03,
            l2_current_cost=2.0,
            reasons=("uncertain",),
        )
    ]
    summary = build_l3_polish_summary(
        selected=selected,
        l3_costs={
            "layer": {
                "MXFP8": {"propagated_end_kl": 3.0},
                "BF16": {"propagated_end_kl": 0.0},
            }
        },
        before_assignment={"layer": "MXFP8"},
        after_assignment={"layer": "BF16"},
        kl_before=1.0,
        kl_after=1.2,
    )

    assert summary["regression"] is True
    assert summary["flip_count"] == 1
    assert summary["flips"] == [
        {
            "name": "layer",
            "from": "MXFP8",
            "to": "BF16",
            "from_l3_cost": 3.0,
            "to_l3_cost": 0.0,
        }
    ]
    assert summary["selected"][0]["reasons"] == ["uncertain"]
