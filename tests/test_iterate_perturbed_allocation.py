import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant import iterate_perturbed_allocation as ipa
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
        elapsed_seconds=2.5,
    )

    assert summary["l3_enabled"] is True
    assert isinstance(summary["selected_count"], int)
    assert summary["regression"] is True
    assert isinstance(summary["kl_before"], float)
    assert isinstance(summary["kl_after"], float)
    assert summary["elapsed_seconds"] == pytest.approx(2.5)
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


class _TinyLogitsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.layer.weight.copy_(torch.eye(2))

    def forward(self, input_ids):
        if input_ids.dtype in (torch.int32, torch.int64, torch.long):
            x = torch.nn.functional.one_hot(input_ids % 2, num_classes=2).float()
        else:
            x = input_ids.float()
        return SimpleNamespace(logits=self.layer(x))


def _tiny_stat():
    return {
        "n_params": 4,
        "in_features": 2,
        "out_features": 2,
        "h_trace": 1.0,
        "_memory_bytes_by_format": {
            "INT8_W8A16": 8,
            "BF16": 8,
        },
    }


def test_iterate_main_emits_observability_traces(tmp_path, monkeypatch, capsys):
    probe_path = tmp_path / "probe.pkl"
    costs_path = tmp_path / "costs.pkl"
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "out"
    stats = {"layer": _tiny_stat()}
    initial_costs = {
        "layer": {
            "INT8_W8A16": {"predicted_dloss": 1.0},
            "BF16": {"predicted_dloss": 0.0},
        }
    }
    with open(probe_path, "wb") as f:
        pickle.dump({"stats": stats}, f)
    with open(costs_path, "wb") as f:
        pickle.dump({"costs": initial_costs}, f)

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_Tokenizer),
    )
    monkeypatch.setattr(ipa, "validate_probe_payload", lambda *_args: None)
    monkeypatch.setattr(ipa, "validate_cost_payload", lambda *_args: None)
    monkeypatch.setattr(
        ipa,
        "load_text_model_under_work_root",
        lambda *_args, **_kwargs: _TinyLogitsModel(),
    )
    monkeypatch.setattr(
        ipa,
        "load_wikitext_calibration",
        lambda *_args, **_kwargs: torch.tensor([[0, 1], [1, 0]]),
    )
    monkeypatch.setattr(ipa, "cache_reference_log_probs", lambda *_args: [])
    monkeypatch.setattr(
        ipa,
        "measure_assignment_kl",
        lambda _model, assignment, *_args, **_kwargs: (
            0.9 if assignment.get("layer") == "BF16" else 1.0
        ),
    )

    def _capture(_model, _assignment, _calib_ids, cache_dir, **_kwargs):
        cache_dir = tmp_path / Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "cache.bin").write_bytes(b"x")
        return {"cache_dir": str(cache_dir)}

    class _ActivationIndex:
        def __init__(self, *_args, **_kwargs):
            pass

        def __contains__(self, _name):
            return True

    calls = {"n": 0}

    def _run_cost_pass(*_args, **_kwargs):
        calls["n"] += 1
        value = 1.0 + 0.1 * calls["n"]
        return {
            "layer": {
                "INT8_W8A16": {"predicted_dloss": value},
                "BF16": {"predicted_dloss": 0.0},
            }
        }

    monkeypatch.setattr(ipa, "capture_perturbed_activation_cache", _capture)
    monkeypatch.setattr(ipa, "ActivationIndex", _ActivationIndex)
    monkeypatch.setattr(ipa, "run_cost_pass", _run_cost_pass)

    rc = ipa.main([
        "--model", "tiny",
        "--probe", str(probe_path),
        "--initial-costs", str(costs_path),
        "--formats", "INT8_W8A16,BF16",
        "--target-bits", "16",
        "--work-dir", str(work_dir),
        "--output-dir", str(output_dir),
        "--max-iters", "2",
        "--convergence-frac", "-1",
        "--l3-polish",
    ])

    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("[l2] === iteration") == 2
    assert "[l3] === polish ===" in out
    assert "format histogram" in out

    iteration_trace = output_dir / "iteration_trace.jsonl"
    l3_trace = output_dir / "l3_polish_trace.jsonl"
    assert iteration_trace.exists()
    assert l3_trace.exists()
    iter_lines = [json.loads(line) for line in iteration_trace.read_text().splitlines()]
    l3_lines = [json.loads(line) for line in l3_trace.read_text().splitlines()]
    assert len(iter_lines) == 2
    assert len(l3_lines) == 1
