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
        frozen_dp_precision_used=0.05,
    )

    assert summary["l3_enabled"] is True
    assert summary["accepted"] is False
    assert isinstance(summary["selected_count"], int)
    assert summary["regression"] is True
    assert isinstance(summary["kl_before"], float)
    assert isinstance(summary["kl_after"], float)
    assert summary["elapsed_seconds"] == pytest.approx(2.5)
    assert summary["frozen_dp_precision_used"] == pytest.approx(0.05)
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


def _run_tiny_l3_regression_case(tmp_path, monkeypatch, *, tolerance, kl_after):
    probe_path = tmp_path / "probe.pkl"
    costs_path = tmp_path / "costs.pkl"
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "out"
    stats = {"layer": _tiny_stat()}
    costs = {
        "layer": {
            "INT8_W8A16": {"predicted_dloss": 0.0},
            "BF16": {"predicted_dloss": 10.0},
        }
    }
    with open(probe_path, "wb") as f:
        pickle.dump({"stats": stats}, f)
    with open(costs_path, "wb") as f:
        pickle.dump({"costs": costs}, f)

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
        lambda _tokenizer, n_samples, seqlen: torch.zeros(
            (n_samples, seqlen),
            dtype=torch.long,
        ),
    )
    monkeypatch.setattr(ipa, "cache_reference_log_probs", lambda *_args: [])
    monkeypatch.setattr(
        ipa,
        "measure_assignment_kl",
        lambda _model, assignment, *_args, **_kwargs: (
            float(kl_after) if assignment.get("layer") == "BF16" else 1.0
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

    monkeypatch.setattr(ipa, "capture_perturbed_activation_cache", _capture)
    monkeypatch.setattr(ipa, "ActivationIndex", _ActivationIndex)
    monkeypatch.setattr(ipa, "run_cost_pass", lambda *_args, **_kwargs: costs)
    monkeypatch.setattr(
        ipa,
        "measure_propagated_costs",
        lambda *_args, **_kwargs: {
            "layer": {
                "INT8_W8A16": {"propagated_end_kl": 1.0, "downstream_output_mse": 0.0},
                "BF16": {"propagated_end_kl": 0.0, "downstream_output_mse": 0.0},
            }
        },
    )
    monkeypatch.setattr(
        ipa,
        "solve_frozen_l3_neighborhood",
        lambda *_args, **_kwargs: (
            {"layer": "BF16"},
            {},
            {"frozen_dp_precision_used": 0.001},
        ),
    )

    argv = [
        "--model", "tiny",
        "--probe", str(probe_path),
        "--initial-costs", str(costs_path),
        "--formats", "INT8_W8A16,BF16",
        "--target-bits", "16",
        "--work-dir", str(work_dir),
        "--output-dir", str(output_dir),
        "--max-iters", "1",
        "--convergence-frac", "1",
        "--l3-polish",
        "--no-l3-tail-only",
    ]
    if tolerance is not None:
        argv.extend(["--l3-regression-tolerance", str(tolerance)])

    rc = ipa.main(argv)
    with open(output_dir / "final_assignment.json") as f:
        final_assignment = json.load(f)
    with open(output_dir / "l3_polish_summary.json") as f:
        l3_summary = json.load(f)
    return rc, final_assignment, l3_summary


def test_l3_polish_rejects_assignment_on_regression(tmp_path, monkeypatch):
    rc, final_assignment, l3_summary = _run_tiny_l3_regression_case(
        tmp_path,
        monkeypatch,
        tolerance=None,
        kl_after=1.2,
    )

    assert rc == 0
    assert final_assignment == {"layer": "INT8_W8A16"}
    assert l3_summary["regression"] is True
    assert l3_summary["accepted"] is False
    assert l3_summary["accepted_assignment"] == "l2"
    assert l3_summary["accepted_flip_count"] == 0


def test_l3_polish_accepts_within_tolerance(tmp_path, monkeypatch):
    rc, final_assignment, l3_summary = _run_tiny_l3_regression_case(
        tmp_path,
        monkeypatch,
        tolerance=0.10,
        kl_after=1.05,
    )

    assert rc == 0
    assert final_assignment == {"layer": "BF16"}
    assert l3_summary["regression"] is True
    assert l3_summary["accepted"] is True
    assert l3_summary["accepted_assignment"] == "polished"
    assert l3_summary["accepted_flip_count"] == 1


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
    calib_calls = []

    def _load_calib(_tokenizer, n_samples, seqlen):
        calib_calls.append((n_samples, seqlen))
        return torch.zeros((n_samples, seqlen), dtype=torch.long)

    monkeypatch.setattr(ipa, "load_wikitext_calibration", _load_calib)
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
        "--l3-max-lanes-per-batch", "3",
        "--no-l3-tail-only",
        "--l3-n-calib-samples", "1",
        "--l3-calib-seqlen", "2",
    ])

    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("[l2] === iteration") == 2
    assert "[l3] === polish ===" in out
    assert "format histogram" in out
    assert (
        "[l2] iteration 2: format histogram delta:" in out
        or "[l2] iteration 2: format histogram: unchanged" in out
    )
    assert "[l2] iteration 1: weighted_hamming " in out
    assert "[l2] iteration 2: weighted_hamming " not in out
    assert "== marginal cost ==" in out
    assert "total wall:" in out

    iteration_trace = output_dir / "iteration_trace.jsonl"
    l3_trace = output_dir / "l3_polish_trace.jsonl"
    assert iteration_trace.exists()
    assert l3_trace.exists()
    iter_lines = [json.loads(line) for line in iteration_trace.read_text().splitlines()]
    l3_lines = [json.loads(line) for line in l3_trace.read_text().splitlines()]
    assert len(iter_lines) == 2
    assert len(l3_lines) == 1
    assert (8, 512) in calib_calls
    assert (1, 2) in calib_calls
    with open(output_dir / "l3_propagated_costs.pkl", "rb") as f:
        l3_payload = pickle.load(f)
    assert l3_payload["meta"]["l3_max_lanes_per_batch"] == 3
    assert l3_payload["meta"]["tail_only"] is False
    assert l3_payload["meta"]["l3_n_calib_samples"] == 1
    assert l3_payload["meta"]["l3_calib_seqlen"] == 2


def _run_global_l3_case(tmp_path, monkeypatch, *, l3_costs, kl_after):
    probe_path = tmp_path / "probe.pkl"
    costs_path = tmp_path / "costs.pkl"
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "out"
    stats = {f"layer{i}": _tiny_stat() for i in range(4)}
    l2_costs = {
        name: {
            "INT8_W8A16": {"predicted_dloss": 0.0},
            "BF16": {"predicted_dloss": 10.0},
        }
        for name in stats
    }
    with open(probe_path, "wb") as f:
        pickle.dump({"stats": stats}, f)
    with open(costs_path, "wb") as f:
        pickle.dump({"costs": l2_costs}, f)

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
        lambda _tokenizer, n_samples, seqlen: torch.zeros(
            (n_samples, seqlen),
            dtype=torch.long,
        ),
    )
    monkeypatch.setattr(ipa, "cache_reference_log_probs", lambda *_args: [])

    l2_assignment = {name: "INT8_W8A16" for name in stats}

    def _kl(_model, assignment, *_args, **_kwargs):
        return float(kl_after) if dict(assignment) != l2_assignment else 1.0

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

    measured_names = []

    def _measure(_model, _assignment, selected, *_args, **_kwargs):
        measured_names.extend(entry.name for entry in selected)
        return l3_costs

    monkeypatch.setattr(ipa, "measure_assignment_kl", _kl)
    monkeypatch.setattr(ipa, "capture_perturbed_activation_cache", _capture)
    monkeypatch.setattr(ipa, "ActivationIndex", _ActivationIndex)
    monkeypatch.setattr(ipa, "run_cost_pass", lambda *_args, **_kwargs: l2_costs)
    monkeypatch.setattr(ipa, "measure_propagated_costs", _measure)

    rc = ipa.main([
        "--model", "tiny",
        "--probe", str(probe_path),
        "--initial-costs", str(costs_path),
        "--formats", "INT8_W8A16,BF16",
        "--target-bits", "16",
        "--work-dir", str(work_dir),
        "--output-dir", str(output_dir),
        "--max-iters", "1",
        "--convergence-frac", "1",
        "--l3-polish",
        "--l3-mode", "global",
        "--no-l3-tail-only",
    ])
    with open(output_dir / "final_assignment.json") as f:
        final_assignment = json.load(f)
    with open(output_dir / "l3_polish_summary.json") as f:
        l3_summary = json.load(f)
    with open(output_dir / "l3_propagated_costs.pkl", "rb") as f:
        l3_payload = pickle.load(f)
    return rc, final_assignment, l3_summary, l3_payload, measured_names


def _global_l3_costs_for_joint_choice():
    return {
        "layer0": {
            "INT8_W8A16": {"propagated_end_kl": 1.0, "downstream_output_mse": 0.0},
            "BF16": {"propagated_end_kl": 0.0, "downstream_output_mse": 0.0},
        },
        "layer1": {
            "INT8_W8A16": {"propagated_end_kl": 0.0, "downstream_output_mse": 0.0},
            "BF16": {"propagated_end_kl": 1.0, "downstream_output_mse": 0.0},
        },
        "layer2": {
            "INT8_W8A16": {"propagated_end_kl": 1.0, "downstream_output_mse": 0.0},
            "BF16": {"propagated_end_kl": 0.0, "downstream_output_mse": 0.0},
        },
        "layer3": {
            "INT8_W8A16": {"propagated_end_kl": 0.0, "downstream_output_mse": 0.0},
            "BF16": {"propagated_end_kl": 1.0, "downstream_output_mse": 0.0},
        },
    }


def test_global_l3_measures_all_linears(tmp_path, monkeypatch):
    rc, _final, summary, payload, measured_names = _run_global_l3_case(
        tmp_path,
        monkeypatch,
        l3_costs=_global_l3_costs_for_joint_choice(),
        kl_after=0.5,
    )

    assert rc == 0
    assert sorted(measured_names) == [f"layer{i}" for i in range(4)]
    assert sorted(payload["costs"]) == [f"layer{i}" for i in range(4)]
    assert payload["meta"]["l3_mode"] == "global"
    assert payload["meta"]["total_count"] == 4
    assert payload["meta"]["model_linear_count"] == 4
    assert summary["l3_mode"] == "global"
    assert summary["total_count"] == 4
    assert summary["model_linear_count"] == 4


def test_global_l3_dp_optimizes_jointly(tmp_path, monkeypatch):
    rc, final_assignment, summary, _payload, _measured = _run_global_l3_case(
        tmp_path,
        monkeypatch,
        l3_costs=_global_l3_costs_for_joint_choice(),
        kl_after=0.5,
    )

    assert rc == 0
    assert final_assignment == {
        "layer0": "BF16",
        "layer1": "INT8_W8A16",
        "layer2": "BF16",
        "layer3": "INT8_W8A16",
    }
    assert summary["accepted"] is True
    assert summary["accepted_flip_count"] == 2


def test_global_l3_respects_regression_rollback(tmp_path, monkeypatch):
    rc, final_assignment, summary, _payload, _measured = _run_global_l3_case(
        tmp_path,
        monkeypatch,
        l3_costs=_global_l3_costs_for_joint_choice(),
        kl_after=1.2,
    )

    assert rc == 0
    assert final_assignment == {f"layer{i}": "INT8_W8A16" for i in range(4)}
    assert summary["l3_mode"] == "global"
    assert summary["regression"] is True
    assert summary["accepted"] is False
    assert summary["accepted_assignment"] == "l2"


def test_multi_budget_clusters_by_tolerance():
    targets = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]

    clusters = ipa.plan_target_bit_clusters(targets, 0.25)

    assert len(clusters) == 6
    assert [cluster["targets"] for cluster in clusters] == [[t] for t in targets]


def _dummy_budget_result(target_bpp, anchor_bpp, tmp_path):
    return ipa.BudgetResult(
        target_bpp=float(target_bpp),
        anchor_bpp=float(anchor_bpp),
        distance_from_anchor=abs(float(target_bpp) - float(anchor_bpp)),
        anchor_stale=target_bpp != anchor_bpp,
        achieved_bpp=float(target_bpp),
        predicted_dloss=0.1,
        validation_kl=1.0 / float(target_bpp),
        accepted=True,
        regression=False,
        flips_accepted=1,
        format_histogram={"counts": {"BF16": 1}, "total": 1},
        assignment={"layer": "BF16"},
        layer_config_path=str(tmp_path / f"final_layer_config_bpp_{target_bpp}.json"),
    )


def test_multi_budget_reanchor_runs_full_l2_l3(tmp_path, monkeypatch):
    calls = {"l2": 0, "l3": 0}

    def _anchor(_args, _runtime, anchor_bpp, *, measure_all_formats=False):
        calls["l2"] += 1
        calls["l3"] += 1
        return SimpleNamespace(anchor_bpp=float(anchor_bpp))

    def _single(_args, target_bits, reusable_anchor=None):
        return _dummy_budget_result(target_bits, reusable_anchor.anchor_bpp, tmp_path)

    monkeypatch.setattr(ipa, "run_anchor_budget", _anchor)
    monkeypatch.setattr(ipa, "run_single_budget", _single)
    args = SimpleNamespace(
        target_bits_list="4.0,4.5,5.0",
        target_bits_share_tolerance=0.25,
        target_bits_anchor=None,
    )
    runtime = SimpleNamespace(output_root=tmp_path)

    assert ipa.run_multi_budget(args, runtime) == 0
    assert calls == {"l2": 3, "l3": 3}


def test_multi_budget_widens_format_filter(tmp_path, monkeypatch):
    specs = [
        ipa.fr.get_format(name)
        for name in [
            "NVFP4",
            "MXFP4",
            "MXFP6_E3M2",
            "MXFP6_E2M3",
            "MXFP8",
            "MXFP8_E5M2",
            "BF16",
        ]
    ]
    entry = L3NeighborhoodEntry(
        name="layer",
        current_format="MXFP6_E3M2",
        formats=("NVFP4", "MXFP6_E3M2", "BF16"),
        margin=0.1,
        l2_current_cost=1.0,
        reasons=("global",),
    )

    widened = ipa.widen_l3_neighborhood_formats([entry], specs)

    assert widened[0].formats == tuple(spec.name for spec in specs)
    assert "all_formats" in widened[0].reasons

    flags = []

    def _anchor(_args, _runtime, anchor_bpp, *, measure_all_formats=False):
        flags.append(bool(measure_all_formats))
        return SimpleNamespace(anchor_bpp=float(anchor_bpp))

    def _single(_args, target_bits, reusable_anchor=None):
        return _dummy_budget_result(target_bits, reusable_anchor.anchor_bpp, tmp_path)

    monkeypatch.setattr(ipa, "run_anchor_budget", _anchor)
    monkeypatch.setattr(ipa, "run_single_budget", _single)
    args = SimpleNamespace(
        target_bits_list="4.0,5.5",
        target_bits_share_tolerance=0.25,
        target_bits_anchor=None,
    )

    assert ipa.run_multi_budget(args, SimpleNamespace(output_root=tmp_path)) == 0
    assert flags and all(flags)


def test_knee_search_segmented_kneedle():
    values = {
        4.0: 1.0,
        5.0: 0.5,
        5.5: 0.5,
        5.75: 0.45,
        6.0: 0.2,
        6.5: 0.3,
        7.0: 0.18,
        8.0: 0.17,
    }

    def _evaluate(bpp):
        return values.get(round(float(bpp), 2), 1.0 / float(bpp))

    knee, points, meta = ipa.adaptive_segmented_kneedle(
        _evaluate,
        4.0,
        8.0,
        tolerance=0.25,
        max_evaluations=8,
    )

    assert knee == pytest.approx(6.0)
    assert len(points) <= 8
    assert meta["mode"] == "kneedle"


def test_knee_threshold_mode_finds_lowest_bpp():
    knee, points, meta = ipa.threshold_knee_search(
        lambda bpp: 10.0 - float(bpp),
        4.0,
        8.0,
        threshold_kl=4.0,
        tolerance=0.1,
        max_evaluations=20,
    )

    assert knee == pytest.approx(6.0, abs=0.1)
    assert len(points) <= 20
    assert meta["mode"] == "threshold"


def test_knee_search_max_evaluations_stop():
    calls = {"n": 0}

    def _evaluate(bpp):
        calls["n"] += 1
        return 1.0 / float(bpp)

    _knee, points, _meta = ipa.adaptive_segmented_kneedle(
        _evaluate,
        4.0,
        8.0,
        tolerance=0.001,
        max_evaluations=4,
    )

    assert calls["n"] == 4
    assert len(points) == 4


def test_main_dispatches_target_bits_list_with_one_model_load(tmp_path, monkeypatch):
    probe_path = tmp_path / "probe.pkl"
    costs_path = tmp_path / "costs.pkl"
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "out"
    stats = {"layer": _tiny_stat()}
    costs = {
        "layer": {
            "INT8_W8A16": {"predicted_dloss": 1.0},
            "BF16": {"predicted_dloss": 0.0},
        }
    }
    with open(probe_path, "wb") as f:
        pickle.dump({"stats": stats}, f)
    with open(costs_path, "wb") as f:
        pickle.dump({"costs": costs}, f)

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    load_calls = {"n": 0}
    captured = {}

    def _load_model(*_args, **_kwargs):
        load_calls["n"] += 1
        return _TinyLogitsModel()

    def _run_multi(args, runtime):
        captured["l3_mode"] = args.l3_mode
        captured["model"] = runtime.model
        captured["targets"] = ipa.parse_target_bits_list(args.target_bits_list)
        return 0

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_Tokenizer),
    )
    monkeypatch.setattr(ipa, "validate_probe_payload", lambda *_args: None)
    monkeypatch.setattr(ipa, "validate_cost_payload", lambda *_args: None)
    monkeypatch.setattr(ipa, "load_text_model_under_work_root", _load_model)
    monkeypatch.setattr(
        ipa,
        "load_wikitext_calibration",
        lambda _tokenizer, n_samples, seqlen: torch.zeros(
            (n_samples, seqlen),
            dtype=torch.long,
        ),
    )
    monkeypatch.setattr(ipa, "cache_reference_log_probs", lambda *_args: [])
    monkeypatch.setattr(ipa, "run_multi_budget", _run_multi)

    rc = ipa.main([
        "--model", "tiny",
        "--probe", str(probe_path),
        "--initial-costs", str(costs_path),
        "--formats", "INT8_W8A16,BF16",
        "--target-bits-list", "4.5,5.0",
        "--work-dir", str(work_dir),
        "--output-dir", str(output_dir),
        "--l3-polish",
    ])

    assert rc == 0
    assert load_calls["n"] == 1
    assert captured["l3_mode"] == "global"
    assert captured["targets"] == [4.5, 5.0]
    assert isinstance(captured["model"], _TinyLogitsModel)
