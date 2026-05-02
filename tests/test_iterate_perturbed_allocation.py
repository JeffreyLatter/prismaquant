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
        "--no-l3-coord-descent-fallback",
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


def test_solve_target_from_anchor_rejects_to_l2_at_target(tmp_path, monkeypatch):
    stats = {
        "layer": {
            "n_params": 16,
            "in_features": 4,
            "out_features": 4,
            "h_trace": 1.0,
        }
    }
    costs = {
        "layer": {
            "INT8_W8A16": {"predicted_dloss": 1.0},
            "BF16": {"predicted_dloss": 0.0},
        }
    }
    specs = [ipa.fr.get_format("INT8_W8A16"), ipa.fr.get_format("BF16")]
    anchor = ipa.AnchorResult(
        anchor_bpp=16.0,
        output_dir=tmp_path / "anchor_bpp_16.00",
        l2_assignment={"layer": "BF16"},
        l2_kl=1.0,
        l3_selected=[],
        l3_candidates={},
        l3_costs={},
        latest_smoothed_costs=costs,
    )
    runtime = ipa.BudgetRuntime(
        work_root=tmp_path / "work",
        output_root=tmp_path / "out",
        stats=stats,
        current_costs=costs,
        specs=specs,
        profile=None,
        model=_TinyLogitsModel(),
        calib_ids=torch.zeros((1, 2), dtype=torch.long),
        l3_calib_ids=torch.zeros((1, 2), dtype=torch.long),
        ref_log_probs=[],
        dtype=torch.bfloat16,
        probe_load_timing={},
    )
    args = SimpleNamespace(
        bit_precision=0.001,
        l3_regression_tolerance=0.0,
    )
    monkeypatch.setattr(ipa, "measure_assignment_kl", lambda *_args, **_kwargs: 2.0)

    result = ipa.solve_target_from_anchor(args, runtime, anchor, 12.0)

    assert result.accepted is False
    assert result.assignment == {"layer": "INT8_W8A16"}
    assert result.achieved_bpp == pytest.approx(12.0)
    assert result.anchor_bpp == pytest.approx(16.0)
    assert result.target_bpp == pytest.approx(12.0)


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


class _WideStackLogitsModel(nn.Module):
    def __init__(self, width=33, layers=3):
        super().__init__()
        self.layer_names = [f"layer{i}" for i in range(layers)]
        for idx, name in enumerate(self.layer_names):
            layer = nn.Linear(width, width, bias=False)
            with torch.no_grad():
                layer.weight.copy_(torch.eye(width) * (1.0 + idx * 0.01))
            setattr(self, name, layer)

    def forward(self, input_ids):
        x = input_ids.float()
        for name in self.layer_names:
            x = getattr(self, name)(x)
        return SimpleNamespace(logits=x)


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


def _wide_stat(width=33, formats=("COUNTED_FP4_A", "COUNTED_FP4_B", "BF16")):
    n_params = width * width
    return {
        "n_params": n_params,
        "in_features": width,
        "out_features": width,
        "h_trace": 1.0,
        "_memory_bytes_by_format": {
            fmt: 2 * n_params for fmt in formats
        },
    }


def _counted_fp_spec(name: str):
    codebook = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    return ipa.fr.FormatSpec(
        name=name,
        weight_bits=4,
        group_size=0,
        scale_bits=0,
        scale_dtype_name="none",
        weight_element_dtype="test_counted_fp",
        quantize_dequantize=lambda w: ipa.fr._rtn_fp_codebook(w, codebook, 0),
        activation_quantize_dequantize=lambda x: x.clone(),
    )


def _scaled_weight_spec(name: str, scale: float):
    return ipa.fr.FormatSpec(
        name=name,
        weight_bits=8,
        group_size=0,
        scale_bits=0,
        scale_dtype_name="none",
        weight_element_dtype="test_scaled",
        quantize_dequantize=lambda w: w.detach().clone() * float(scale),
        activation_quantize_dequantize=lambda x: x.clone(),
    )


class _CountingWideStackLogitsModel(_WideStackLogitsModel):
    def __init__(self, width=33, layers=3):
        super().__init__(width=width, layers=layers)
        self.forward_calls = 0

    def forward(self, input_ids):
        self.forward_calls += 1
        return super().forward(input_ids)


class _ReplayDecoderBlock(nn.Module):
    def __init__(self, width: int, idx: int):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        self.forward_calls = 0
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(width) * (1.0 + idx * 0.01))

    def forward(self, hidden_states):
        self.forward_calls += 1
        return self.proj(hidden_states)


class _ReplayDecoder(nn.Module):
    def __init__(self, width: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_ReplayDecoderBlock(width, idx) for idx in range(layers)]
        )
        self.norm = nn.Identity()


class _ReplayStackLogitsModel(nn.Module):
    def __init__(self, width=33, layers=4):
        super().__init__()
        self.model = _ReplayDecoder(width, layers)
        self.lm_head = nn.Identity()
        self.layer_names = [
            f"model.layers.{idx}.proj"
            for idx in range(layers)
        ]

    def forward(self, input_ids):
        hidden = input_ids.float()
        for layer in self.model.layers:
            hidden = layer(hidden)
        hidden = self.model.norm(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))

    def reset_layer_forward_calls(self):
        for layer in self.model.layers:
            layer.forward_calls = 0

    def layer_forward_calls(self) -> int:
        return sum(layer.forward_calls for layer in self.model.layers)


def test_iterate_main_emits_observability_traces(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "0")
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
        "--l3-iter-max", "1",
        "--no-l3-coord-descent-fallback",
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


def _iterated_l3_args(**overrides):
    values = dict(
        l3_mode="global",
        l3_iter_max=3,
        l3_hamming_cap_init=1,
        l3_hamming_cap_max=4,
        l3_coord_descent_fallback=False,
        l3_coord_descent_max_passes=1,
        l3_max_lanes_per_batch=4,
        l3_tail_only=False,
        l3_regression_tolerance=0.0,
        ema_decay=0.5,
        bit_precision=0.001,
        frozen_dp_budget_tolerance=0.0,
        l3_uncertainty_rel_tol=0.10,
        l3_min_fraction=0.05,
        l3_max_fraction=1.0,
        l3_safety_fraction=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_l3_iteration_monotone(tmp_path, monkeypatch):
    stats = {f"layer{i}": _tiny_stat() for i in range(2)}
    specs = [ipa.fr.get_format("INT8_W8A16"), ipa.fr.get_format("BF16")]
    assignment = {name: "INT8_W8A16" for name in stats}
    l2_costs = {
        name: {
            "INT8_W8A16": {"predicted_dloss": 1.0},
            "BF16": {"predicted_dloss": 0.0},
        }
        for name in stats
    }

    def _measure(_model, current_assignment, selected, *_args, **_kwargs):
        del selected
        if current_assignment["layer0"] == "INT8_W8A16":
            return {
                "layer0": {
                    "INT8_W8A16": {"propagated_end_kl": 2.0},
                    "BF16": {"propagated_end_kl": 0.0},
                },
                "layer1": {
                    "INT8_W8A16": {"propagated_end_kl": 2.0},
                    "BF16": {"propagated_end_kl": 3.0},
                },
            }
        return {
            name: {
                "INT8_W8A16": {"propagated_end_kl": 2.0},
                "BF16": {"propagated_end_kl": 0.0},
            }
            for name in stats
        }

    def _kl(_model, trial_assignment, *_args, **_kwargs):
        bf16_count = sum(fmt == "BF16" for fmt in trial_assignment.values())
        return 3.0 - float(bf16_count)

    monkeypatch.setattr(ipa, "measure_propagated_costs", _measure)
    monkeypatch.setattr(ipa, "measure_assignment_kl", _kl)

    result = ipa.run_iterated_l3_polish(
        _iterated_l3_args(l3_hamming_cap_init=1, l3_hamming_cap_max=2),
        _TinyLogitsModel(),
        assignment,
        3.0,
        stats,
        l2_costs,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.long),
        [],
        work_root=tmp_path,
    )

    accepted = [
        row for row in result.iterations
        if row["accepted"] and row["hamming"] > 0
    ]
    assert len(accepted) >= 2
    kl_sequence = [3.0] + [row["candidate_kl"] for row in accepted]
    assert all(
        after <= before
        for before, after in zip(kl_sequence, kl_sequence[1:])
    )
    assert result.final_kl <= 3.0


def test_l3_hamming_cap_respected():
    stats = {f"layer{i}": _tiny_stat() for i in range(4)}
    specs = [ipa.fr.get_format("INT8_W8A16"), ipa.fr.get_format("BF16")]
    costs = {
        name: {
            "INT8_W8A16": {"predicted_dloss": 10.0},
            "BF16": {"predicted_dloss": 0.0},
        }
        for name in stats
    }
    baseline = {name: "INT8_W8A16" for name in stats}

    solved = ipa.solve_from_costs_with_cap(
        stats,
        costs,
        specs,
        target_bits=16.0,
        bit_precision=0.001,
        baseline_assignment=baseline,
        max_flips=2,
    )

    assert ipa._hamming_count(baseline, solved) <= 2


def test_coord_descent_non_regressive(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "0")
    stats = {"layer": _tiny_stat()}
    specs = [ipa.fr.get_format("INT8_W8A16"), ipa.fr.get_format("BF16")]
    l3_costs = {
        "layer": {
            "INT8_W8A16": {"propagated_end_kl": 1.0},
            "BF16": {"propagated_end_kl": 0.0},
        }
    }

    def _kl(_model, assignment, *_args, **_kwargs):
        return 0.5 if assignment["layer"] == "BF16" else 1.0

    monkeypatch.setattr(ipa, "measure_assignment_kl", _kl)

    polished, final_kl, meta = ipa.coordinate_descent_polish(
        _TinyLogitsModel(),
        {"layer": "INT8_W8A16"},
        l3_costs,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        [],
        stats=stats,
        work_root=tmp_path,
        current_kl=1.0,
        return_metadata=True,
    )

    assert final_kl <= 1.0
    assert polished == {"layer": "BF16"}
    assert meta["flips_committed"] == 1


def test_coord_descent_uses_frozen_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "0")
    spec_a = _counted_fp_spec("COUNTED_FP4_A")
    spec_b = _counted_fp_spec("COUNTED_FP4_B")
    monkeypatch.setitem(ipa.fr.REGISTRY, spec_a.name, spec_a)
    monkeypatch.setitem(ipa.fr.REGISTRY, spec_b.name, spec_b)

    n_layers = 3
    model = _WideStackLogitsModel(layers=n_layers).eval()
    stats = {
        name: _wide_stat(formats=(spec_a.name, spec_b.name, "BF16"))
        for name in model.layer_names
    }
    specs = [spec_a, spec_b, ipa.fr.get_format("BF16")]
    assignment = {name: spec_a.name for name in model.layer_names}
    l3_costs = {
        name: {
            spec_a.name: {"propagated_end_kl": 0.0},
            spec_b.name: {"propagated_end_kl": 1.0},
            "BF16": {"propagated_end_kl": 2.0},
        }
        for name in model.layer_names
    }
    calib_ids = torch.randn(2, 4, 33)
    ref_log_probs = ipa.cache_reference_log_probs(
        model,
        calib_ids,
        next(model.parameters()).device,
    )

    calls = {"count": 0}
    original = ipa.fr._rtn_fp_codebook

    def _counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ipa.fr, "_rtn_fp_codebook", _counted)

    _polished, _final_kl, meta = ipa.coordinate_descent_polish(
        model,
        assignment,
        l3_costs,
        specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=stats,
        work_root=tmp_path,
        current_kl=0.0,
        return_metadata=True,
        early_stop_streak=100,
    )

    assert meta["measurements"] == n_layers * 2
    assert calls["count"] == n_layers * 2


def test_coord_descent_frozen_cache_spans_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "0")
    spec_a = _counted_fp_spec("COUNTED_COORD_PASS_FP4_A")
    spec_b = _counted_fp_spec("COUNTED_COORD_PASS_FP4_B")
    monkeypatch.setitem(ipa.fr.REGISTRY, spec_a.name, spec_a)
    monkeypatch.setitem(ipa.fr.REGISTRY, spec_b.name, spec_b)

    n_layers = 5
    model = _WideStackLogitsModel(layers=n_layers).eval()
    stats = {
        name: _wide_stat(formats=(spec_a.name, spec_b.name))
        for name in model.layer_names
    }
    specs = [spec_a, spec_b]
    assignment = {name: spec_a.name for name in model.layer_names}
    l3_costs = {
        name: {
            spec_a.name: {"propagated_end_kl": 1.0},
            spec_b.name: {"propagated_end_kl": 0.0},
        }
        for name in model.layer_names
    }
    calib_ids = torch.randn(1, 4, 33)
    ref_log_probs = ipa.cache_reference_log_probs(
        model,
        calib_ids,
        next(model.parameters()).device,
    )

    calls = {"count": 0}
    original = ipa.fr._rtn_fp_codebook

    def _counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    kl_sequence = {"next": 1000.0}

    def _decreasing_kl(*_args, **_kwargs):
        kl_sequence["next"] -= 1.0
        return torch.tensor(kl_sequence["next"], dtype=torch.float32)

    monkeypatch.setattr(ipa.fr, "_rtn_fp_codebook", _counted)
    monkeypatch.setattr(ipa, "kl_divergence", _decreasing_kl)

    _polished, _final_kl, meta = ipa.coordinate_descent_polish(
        model,
        assignment,
        l3_costs,
        specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=stats,
        work_root=tmp_path,
        current_kl=1000.0,
        return_metadata=True,
        early_stop_streak=100,
        max_passes=2,
    )

    expected_trials = 10
    max_unique_weight_rounds = n_layers * len(specs)
    assert meta["measurements"] == expected_trials
    assert calls["count"] <= max_unique_weight_rounds
    assert calls["count"] <= max_unique_weight_rounds + expected_trials


def test_coord_descent_l3_ranked_order(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "0")
    stats = {f"layer{i}": _tiny_stat() for i in range(2)}
    specs = [ipa.fr.get_format("INT8_W8A16"), ipa.fr.get_format("BF16")]
    assignment = {name: "INT8_W8A16" for name in stats}
    l3_costs = {
        "layer0": {
            "INT8_W8A16": {"propagated_end_kl": 10.0},
            "BF16": {"propagated_end_kl": -5.0},
        },
        "layer1": {
            "INT8_W8A16": {"propagated_end_kl": 10.0},
            "BF16": {"propagated_end_kl": 9.0},
        },
    }
    tried = []

    def _kl(_model, trial_assignment, *_args, **_kwargs):
        tried.append(dict(trial_assignment))
        return 1.0

    monkeypatch.setattr(ipa, "measure_assignment_kl", _kl)

    ipa.coordinate_descent_polish(
        _TinyLogitsModel(),
        assignment,
        l3_costs,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        [],
        stats=stats,
        work_root=tmp_path,
        current_kl=0.0,
        early_stop_streak=10,
    )

    assert tried[0]["layer0"] == "BF16"
    assert tried[0]["layer1"] == "INT8_W8A16"


def test_coord_descent_early_stop_streak(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "0")
    stats = {f"layer{i}": _tiny_stat() for i in range(5)}
    specs = [ipa.fr.get_format("INT8_W8A16"), ipa.fr.get_format("BF16")]
    assignment = {name: "INT8_W8A16" for name in stats}
    l3_costs = {
        name: {
            "INT8_W8A16": {"propagated_end_kl": 1.0},
            "BF16": {"propagated_end_kl": float(idx)},
        }
        for idx, name in enumerate(stats)
    }
    tried = []

    def _kl(_model, trial_assignment, *_args, **_kwargs):
        tried.append(dict(trial_assignment))
        return 1.0

    monkeypatch.setattr(ipa, "measure_assignment_kl", _kl)

    _polished, _final_kl, meta = ipa.coordinate_descent_polish(
        _TinyLogitsModel(),
        assignment,
        l3_costs,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        [],
        stats=stats,
        work_root=tmp_path,
        current_kl=0.0,
        return_metadata=True,
        early_stop_streak=3,
    )

    assert len(tried) == 3
    assert meta["measurements"] == 3
    assert meta["halted"] == "streak"


def test_coord_descent_lane_batched_matches_sequential(tmp_path, monkeypatch):
    scale_spec = _scaled_weight_spec("COORD_SCALE90_MATCH", 0.9)
    monkeypatch.setitem(ipa.fr.REGISTRY, scale_spec.name, scale_spec)
    specs = [scale_spec, ipa.fr.get_format("BF16")]
    layers = 3
    stats = {
        f"layer{i}": _wide_stat(width=4, formats=(scale_spec.name, "BF16"))
        for i in range(layers)
    }
    assignment = {name: scale_spec.name for name in stats}
    l3_costs = {
        name: {
            scale_spec.name: {"propagated_end_kl": 1.0},
            "BF16": {"propagated_end_kl": 0.0},
        }
        for name in stats
    }
    calib_ids = torch.tensor(
        [[[1.0, -1.0, 0.5, 0.25], [0.25, 0.5, -0.75, 1.0]]],
        dtype=torch.float32,
    )

    seq_model = _WideStackLogitsModel(width=4, layers=layers).eval()
    ref_log_probs = ipa.cache_reference_log_probs(
        seq_model,
        calib_ids,
        next(seq_model.parameters()).device,
    )
    current_kl = ipa.measure_assignment_kl(
        seq_model,
        assignment,
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
    )

    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "0")
    sequential_assignment, sequential_kl, sequential_meta = ipa.coordinate_descent_polish(
        seq_model,
        assignment,
        l3_costs,
        specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=stats,
        work_root=tmp_path,
        current_kl=current_kl,
        return_metadata=True,
        early_stop_streak=100,
        max_lanes_per_batch=2,
    )

    lane_model = _WideStackLogitsModel(width=4, layers=layers).eval()
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "1")
    lane_assignment, lane_kl, lane_meta = ipa.coordinate_descent_polish(
        lane_model,
        assignment,
        l3_costs,
        specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=stats,
        work_root=tmp_path,
        current_kl=current_kl,
        return_metadata=True,
        early_stop_streak=100,
        max_lanes_per_batch=2,
    )

    assert lane_meta["lane_batched"] is True
    assert sequential_meta["lane_batched"] is False
    assert lane_assignment == sequential_assignment
    assert lane_kl == pytest.approx(sequential_kl, abs=1e-9, rel=0.0)


def test_coord_descent_replay_cache_matches_full_forward(tmp_path, monkeypatch):
    scale_spec = _scaled_weight_spec("COORD_REPLAY_SCALE90_MATCH", 0.9)
    monkeypatch.setitem(ipa.fr.REGISTRY, scale_spec.name, scale_spec)
    specs = [scale_spec, ipa.fr.get_format("BF16")]
    layers = 4
    width = 33
    stats = {
        f"model.layers.{idx}.proj": _wide_stat(
            width=width,
            formats=(scale_spec.name, "BF16"),
        )
        for idx in range(layers)
    }
    assignment = {name: scale_spec.name for name in stats}
    l3_costs = {
        name: {
            scale_spec.name: {"propagated_end_kl": 1.0},
            "BF16": {"propagated_end_kl": 0.0},
        }
        for name in stats
    }
    calib_ids = torch.linspace(-1.0, 1.0, steps=2 * 3 * width).reshape(2, 3, width)

    full_model = _ReplayStackLogitsModel(width=width, layers=layers).eval()
    ref_log_probs = ipa.cache_reference_log_probs(
        full_model,
        calib_ids,
        next(full_model.parameters()).device,
    )
    current_kl = ipa.measure_assignment_kl(
        full_model,
        assignment,
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
    )

    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "1")
    monkeypatch.setenv("PRISMAQUANT_COORD_REPLAY_CACHE", "0")
    full_assignment, full_kl, full_meta = ipa.coordinate_descent_polish(
        full_model,
        assignment,
        l3_costs,
        specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=stats,
        work_root=tmp_path,
        current_kl=current_kl,
        return_metadata=True,
        early_stop_streak=100,
        max_lanes_per_batch=2,
    )

    replay_model = _ReplayStackLogitsModel(width=width, layers=layers).eval()
    monkeypatch.setenv("PRISMAQUANT_COORD_REPLAY_CACHE", "1")
    replay_assignment, replay_kl, replay_meta = ipa.coordinate_descent_polish(
        replay_model,
        assignment,
        l3_costs,
        specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=stats,
        work_root=tmp_path,
        current_kl=current_kl,
        return_metadata=True,
        early_stop_streak=100,
        max_lanes_per_batch=2,
    )

    assert full_meta["replay_cache_active"] is False
    assert replay_meta["replay_cache_active"] is True
    assert replay_assignment == full_assignment
    assert replay_kl == pytest.approx(full_kl, abs=1e-9, rel=0.0)


def test_coord_descent_replay_cache_invalidated_on_commit(tmp_path, monkeypatch):
    scale_spec = _scaled_weight_spec("COORD_REPLAY_SCALE90_INVALIDATE", 0.9)
    monkeypatch.setitem(ipa.fr.REGISTRY, scale_spec.name, scale_spec)
    specs = [scale_spec, ipa.fr.get_format("BF16")]
    layers = 3
    width = 33
    stats = {
        f"model.layers.{idx}.proj": _wide_stat(
            width=width,
            formats=(scale_spec.name, "BF16"),
        )
        for idx in range(layers)
    }
    assignment = {name: scale_spec.name for name in stats}
    l3_costs = {
        name: {
            scale_spec.name: {"propagated_end_kl": 1.0},
            "BF16": {"propagated_end_kl": 0.0},
        }
        for name in stats
    }
    calib_ids = torch.linspace(-0.5, 0.5, steps=2 * 2 * width).reshape(2, 2, width)
    model = _ReplayStackLogitsModel(width=width, layers=layers).eval()
    ref_log_probs = ipa.cache_reference_log_probs(
        model,
        calib_ids,
        next(model.parameters()).device,
    )
    current_kl = ipa.measure_assignment_kl(
        model,
        assignment,
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
    )

    populate_calls = []
    real_cache_cls = ipa.LayerHiddenStateCache

    class _CountingLayerHiddenStateCache(real_cache_cls):
        def populate(self, baseline_assignment, *args, **kwargs):
            populate_calls.append(dict(baseline_assignment))
            return super().populate(baseline_assignment, *args, **kwargs)

    monkeypatch.setattr(ipa, "LayerHiddenStateCache", _CountingLayerHiddenStateCache)
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "1")
    monkeypatch.setenv("PRISMAQUANT_COORD_REPLAY_CACHE", "1")

    _assignment, _kl, meta = ipa.coordinate_descent_polish(
        model,
        assignment,
        l3_costs,
        specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=stats,
        work_root=tmp_path,
        current_kl=current_kl,
        return_metadata=True,
        early_stop_streak=100,
        max_lanes_per_batch=1,
    )

    assert meta["flips_committed"] >= 1
    assert meta["replay_cache_populates"] == len(populate_calls)
    assert len(populate_calls) >= 2


def test_coord_descent_replay_cache_emits_fewer_layer_forwards(tmp_path, monkeypatch):
    scale_spec = _scaled_weight_spec("COORD_REPLAY_SCALE90_FORWARD_COUNT", 0.9)
    monkeypatch.setitem(ipa.fr.REGISTRY, scale_spec.name, scale_spec)
    specs = [scale_spec, ipa.fr.get_format("BF16")]
    layers = 14
    width = 33
    candidate_depths = list(range(7, layers))
    stats = {
        f"model.layers.{idx}.proj": _wide_stat(
            width=width,
            formats=("BF16", scale_spec.name),
        )
        for idx in candidate_depths
    }
    assignment = {name: "BF16" for name in stats}
    l3_costs = {
        f"model.layers.{idx}.proj": {
            "BF16": {"propagated_end_kl": 0.0},
            scale_spec.name: {"propagated_end_kl": float(idx)},
        }
        for idx in candidate_depths
    }
    calib_ids = torch.linspace(-1.0, 1.0, steps=1 * 3 * width).reshape(1, 3, width)

    full_model = _ReplayStackLogitsModel(width=width, layers=layers).eval()
    ref_log_probs = ipa.cache_reference_log_probs(
        full_model,
        calib_ids,
        next(full_model.parameters()).device,
    )
    full_model.reset_layer_forward_calls()
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "1")
    monkeypatch.setenv("PRISMAQUANT_COORD_REPLAY_CACHE", "0")
    _full_assignment, _full_kl, full_meta = ipa.coordinate_descent_polish(
        full_model,
        assignment,
        l3_costs,
        specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=stats,
        work_root=tmp_path,
        current_kl=0.0,
        return_metadata=True,
        early_stop_streak=100,
        max_lanes_per_batch=2,
    )
    full_layer_forwards = full_model.layer_forward_calls()

    replay_model = _ReplayStackLogitsModel(width=width, layers=layers).eval()
    replay_model.reset_layer_forward_calls()
    monkeypatch.setenv("PRISMAQUANT_COORD_REPLAY_CACHE", "1")
    _replay_assignment, _replay_kl, replay_meta = ipa.coordinate_descent_polish(
        replay_model,
        assignment,
        l3_costs,
        specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=stats,
        work_root=tmp_path,
        current_kl=0.0,
        return_metadata=True,
        early_stop_streak=100,
        max_lanes_per_batch=2,
    )
    replay_layer_forwards = replay_model.layer_forward_calls()

    assert full_meta["measurements"] == len(candidate_depths)
    assert replay_meta["measurements"] == len(candidate_depths)
    assert replay_meta["replay_cache_active"] is True
    assert full_layer_forwards / replay_layer_forwards >= 1.5


def test_coord_descent_lane_batched_emits_fewer_forwards(tmp_path, monkeypatch):
    scale_spec = _scaled_weight_spec("COORD_SCALE90_FORWARD_COUNT", 0.9)
    monkeypatch.setitem(ipa.fr.REGISTRY, scale_spec.name, scale_spec)
    specs = [scale_spec, ipa.fr.get_format("BF16")]
    layers = 5
    lane_batch_size = 2
    stats = {
        f"layer{i}": _wide_stat(width=4, formats=(scale_spec.name, "BF16"))
        for i in range(layers)
    }
    assignment = {name: "BF16" for name in stats}
    l3_costs = {
        name: {
            "BF16": {"propagated_end_kl": 0.0},
            scale_spec.name: {"propagated_end_kl": 1.0},
        }
        for name in stats
    }
    calib_ids = torch.tensor(
        [[[1.0, -1.0, 0.5, 0.25], [0.25, 0.5, -0.75, 1.0]]],
        dtype=torch.float32,
    )

    seq_model = _CountingWideStackLogitsModel(width=4, layers=layers).eval()
    ref_log_probs = ipa.cache_reference_log_probs(
        seq_model,
        calib_ids,
        next(seq_model.parameters()).device,
    )
    seq_model.forward_calls = 0
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "0")
    _seq_assignment, _seq_kl, seq_meta = ipa.coordinate_descent_polish(
        seq_model,
        assignment,
        l3_costs,
        specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=stats,
        work_root=tmp_path,
        current_kl=0.0,
        return_metadata=True,
        early_stop_streak=100,
        max_lanes_per_batch=lane_batch_size,
    )
    sequential_forwards = seq_model.forward_calls

    lane_model = _CountingWideStackLogitsModel(width=4, layers=layers).eval()
    lane_model.forward_calls = 0
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "1")
    emitted = []
    _lane_assignment, _lane_kl, lane_meta = ipa.coordinate_descent_polish(
        lane_model,
        assignment,
        l3_costs,
        specs,
        16.0,
        calib_ids,
        ref_log_probs,
        stats=stats,
        work_root=tmp_path,
        current_kl=0.0,
        return_metadata=True,
        early_stop_streak=100,
        max_lanes_per_batch=lane_batch_size,
        emit=emitted.append,
        anchor_label="4.00",
    )
    lane_forwards = lane_model.forward_calls

    expected_batches = (layers + lane_batch_size - 1) // lane_batch_size
    expected_lane_counts = [
        min(lane_batch_size, layers - idx * lane_batch_size)
        for idx in range(expected_batches)
    ]
    batch_lines = [
        line for line in emitted
        if line.startswith("[coord] anchor 4.00 pass 1 batch ")
    ]

    assert seq_meta["measurements"] == layers
    assert lane_meta["measurements"] == layers
    assert sequential_forwards == layers
    assert lane_forwards <= (layers + lane_batch_size - 1) // lane_batch_size
    assert len(batch_lines) == expected_batches
    assert [
        f"batch {idx}/{expected_batches}" in line
        for idx, line in enumerate(batch_lines, 1)
    ] == [True] * expected_batches
    assert [
        f"n_lanes={expected_lanes}" in line
        for expected_lanes, line in zip(expected_lane_counts, batch_lines)
    ] == [True] * expected_batches
    assert all("best_in_batch=layer" in line for line in batch_lines)
    assert all("cumul_accepted=0" in line for line in batch_lines)
    assert all("cumul_best_kl=0.0000e+00" in line for line in batch_lines)


def test_coord_descent_lane_batched_handles_commit_correctly(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_COORD_LANE_BATCH", "1")
    stats = {f"layer{i}": _tiny_stat() for i in range(2)}
    specs = [ipa.fr.get_format("INT8_W8A16"), ipa.fr.get_format("BF16")]
    assignment = {name: "INT8_W8A16" for name in stats}
    l3_costs = {
        "layer0": {
            "INT8_W8A16": {"propagated_end_kl": 0.0},
            "BF16": {"propagated_end_kl": -2.0},
        },
        "layer1": {
            "INT8_W8A16": {"propagated_end_kl": 0.0},
            "BF16": {"propagated_end_kl": -1.0},
        },
    }
    calls = []

    def _measure(_model, baseline, candidate_flips, *_args, **_kwargs):
        calls.append((dict(baseline), list(candidate_flips)))
        if baseline["layer0"] == "INT8_W8A16":
            return [0.5, 0.4]
        return [0.8]

    monkeypatch.setattr(ipa, "measure_lane_batched_kl_deltas", _measure)
    emitted = []

    polished, final_kl, meta = ipa.coordinate_descent_polish(
        _TinyLogitsModel(),
        assignment,
        l3_costs,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        [],
        stats=stats,
        work_root=tmp_path,
        current_kl=1.0,
        return_metadata=True,
        early_stop_streak=100,
        max_lanes_per_batch=2,
        emit=emitted.append,
        anchor_label="4.00",
    )

    assert calls == [
        (
            {"layer0": "INT8_W8A16", "layer1": "INT8_W8A16"},
            [("layer0", "BF16"), ("layer1", "BF16")],
        ),
        (
            {"layer0": "BF16", "layer1": "INT8_W8A16"},
            [("layer1", "BF16")],
        ),
    ]
    assert polished == {"layer0": "BF16", "layer1": "INT8_W8A16"}
    assert final_kl == pytest.approx(0.5)
    assert meta["flips_committed"] == 1
    commit_lines = [line for line in emitted if " COMMIT: " in line]
    assert len(commit_lines) == 1
    assert commit_lines[0].startswith(
        "[coord] anchor 4.00 COMMIT: layer0.INT8_W8A16 -> BF16"
    )
    assert "delta=-5.0000e-01" in commit_lines[0]
    assert "pass 1 batch 1/2" in commit_lines[0]


def test_measure_assignment_kl_deterministic_same_inputs(tmp_path):
    model = _WideStackLogitsModel(layers=2).eval()
    assignment = {name: "BF16" for name in model.layer_names}
    calib_ids = torch.randn(2, 4, 33)
    ref_log_probs = ipa.cache_reference_log_probs(
        model,
        calib_ids,
        next(model.parameters()).device,
    )

    first = ipa.measure_assignment_kl(
        model,
        assignment,
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
    )
    second = ipa.measure_assignment_kl(
        model,
        assignment,
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
    )

    assert first == second


def test_l3_iteration_terminates_on_cycle(tmp_path, monkeypatch):
    stats = {"layer": _tiny_stat()}
    specs = [ipa.fr.get_format("INT8_W8A16"), ipa.fr.get_format("BF16")]
    l2_costs = {
        "layer": {
            "INT8_W8A16": {"predicted_dloss": 1.0},
            "BF16": {"predicted_dloss": 0.0},
        }
    }
    initial = {"layer": "INT8_W8A16"}

    monkeypatch.setattr(
        ipa,
        "measure_propagated_costs",
        lambda *_args, **_kwargs: {
            "layer": {
                "INT8_W8A16": {"propagated_end_kl": 0.0},
                "BF16": {"propagated_end_kl": 0.0},
            }
        },
    )
    monkeypatch.setattr(ipa, "measure_assignment_kl", lambda *_args, **_kwargs: 1.0)

    def _solve_cycle(_stats, assignment, *_args, **_kwargs):
        next_fmt = "BF16" if assignment["layer"] == "INT8_W8A16" else "INT8_W8A16"
        return {"layer": next_fmt}, {}, {"frozen_dp_precision_used": 0.001}

    monkeypatch.setattr(ipa, "_solve_l3_candidates_with_hamming_cap", _solve_cycle)

    result = ipa.run_iterated_l3_polish(
        _iterated_l3_args(l3_iter_max=4),
        _TinyLogitsModel(),
        initial,
        1.0,
        stats,
        l2_costs,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.long),
        [],
        work_root=tmp_path,
    )

    assert result.cycle_detected is True
    assert len(result.iterations) == 2


def test_multi_budget_clusters_by_tolerance():
    targets = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]

    clusters = ipa.plan_target_bit_clusters(targets, 0.25)

    assert len(clusters) == 6
    assert [cluster["targets"] for cluster in clusters] == [[t] for t in targets]


def _dummy_budget_result(
    target_bpp,
    anchor_bpp,
    tmp_path,
    *,
    l2_kl=0.25,
    validation_kl=None,
    accepted=True,
):
    if validation_kl is None:
        validation_kl = 1.0 / float(target_bpp)
    return ipa.BudgetResult(
        target_bpp=float(target_bpp),
        anchor_bpp=float(anchor_bpp),
        distance_from_anchor=abs(float(target_bpp) - float(anchor_bpp)),
        anchor_stale=target_bpp != anchor_bpp,
        achieved_bpp=float(target_bpp),
        predicted_dloss=0.1,
        l2_kl=float(l2_kl),
        validation_kl=float(validation_kl),
        accepted=bool(accepted),
        regression=bool(float(validation_kl) > float(l2_kl)),
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


def test_multi_budget_emit_and_json_include_l2_kl_on_regression(
    tmp_path,
    monkeypatch,
    capsys,
):
    result = _dummy_budget_result(
        4.0,
        4.0,
        tmp_path,
        l2_kl=1.0,
        validation_kl=1.2,
        accepted=False,
    )

    def _anchor(_args, _runtime, anchor_bpp, *, measure_all_formats=False):
        return SimpleNamespace(anchor_bpp=float(anchor_bpp))

    def _single(_args, _target_bits, reusable_anchor=None):
        return result

    monkeypatch.setattr(ipa, "run_anchor_budget", _anchor)
    monkeypatch.setattr(ipa, "run_single_budget", _single)
    args = SimpleNamespace(
        target_bits_list="4.0",
        target_bits_share_tolerance=0.25,
        target_bits_anchor=None,
    )

    assert result.l2_kl == pytest.approx(1.0)
    assert ipa.run_multi_budget(args, SimpleNamespace(output_root=tmp_path)) == 0

    out = capsys.readouterr().out
    assert "L2_KL=1" in out
    assert "L3_KL=1.2" in out
    assert "delta=+0.2" in out
    assert (
        "[multi] WARNING: L3 polish regressed L2 by >5% — "
        "likely non-additive cost interaction; consider --l3-mode selective"
    ) in out

    with open(tmp_path / "pareto_curve.json") as f:
        pareto = json.load(f)
    with open(tmp_path / "summary.json") as f:
        summary = json.load(f)

    assert pareto["points"][0]["l2_kl"] == pytest.approx(1.0)
    assert summary["pareto"]["points"][0]["l2_kl"] == pytest.approx(1.0)


def test_multi_budget_anchor_l3_passes_progress_callback(
    tmp_path,
    monkeypatch,
    capsys,
):
    stats = {"layer": _tiny_stat()}
    costs = {
        "layer": {
            "INT8_W8A16": {"predicted_dloss": 0.0},
            "BF16": {"predicted_dloss": 1.0},
        }
    }
    specs = [ipa.fr.get_format("INT8_W8A16"), ipa.fr.get_format("BF16")]
    runtime = ipa.BudgetRuntime(
        work_root=tmp_path / "work",
        output_root=tmp_path / "out",
        stats=stats,
        current_costs=costs,
        specs=specs,
        profile=None,
        model=_TinyLogitsModel(),
        calib_ids=torch.zeros((1, 2), dtype=torch.long),
        l3_calib_ids=torch.zeros((1, 2), dtype=torch.long),
        ref_log_probs=[],
        dtype=torch.bfloat16,
        probe_load_timing={},
    )
    args = SimpleNamespace(
        initial_config=None,
        max_iters=1,
        input_rows=None,
        model="tiny",
        probe="probe",
        device="cpu",
        cost_mode="mse",
        chunk_size=1,
        h_detail_dir=None,
        ema_decay=0.5,
        convergence_frac=1.0,
        bit_precision=0.001,
        l3_max_lanes_per_batch=14,
        l3_tail_only=False,
    )
    entry = L3NeighborhoodEntry(
        name="layer",
        current_format="INT8_W8A16",
        formats=("INT8_W8A16", "BF16"),
        margin=0.0,
        l2_current_cost=0.0,
        reasons=("global",),
    )
    captured = {"callback_present": False}

    def _capture(_model, _assignment, _calib_ids, cache_dir, **_kwargs):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return {"cache_dir": str(cache_dir)}

    class _ActivationIndex:
        def __init__(self, *_args, **_kwargs):
            pass

        def __contains__(self, _name):
            return True

    def _measure(_model, _assignment, _selected, *_args, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        captured["callback_present"] = progress_callback is not None
        progress_callback({
            "event": "depth_group_start",
            "group": "layers.20.self_attn.q_proj",
            "group_index": 5,
            "group_count": 36,
            "entry_count": 7,
            "lane_count": 14,
            "mode": "batched",
        })
        progress_callback({
            "event": "depth_group_end",
            "group": "layers.20.self_attn.q_proj",
            "group_index": 5,
            "group_count": 36,
            "entry_count": 7,
            "lane_count": 14,
            "elapsed_seconds": 12.34,
        })
        return {"layer": {"INT8_W8A16": {"propagated_end_kl": 0.0}}}

    monkeypatch.setattr(ipa, "capture_perturbed_activation_cache", _capture)
    monkeypatch.setattr(ipa, "ActivationIndex", _ActivationIndex)
    monkeypatch.setattr(ipa, "run_cost_pass", lambda *_args, **_kwargs: costs)
    monkeypatch.setattr(ipa, "solve_from_costs", lambda *_args, **_kwargs: {
        "layer": "INT8_W8A16",
    })
    monkeypatch.setattr(ipa, "measure_assignment_kl", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(ipa, "build_global_l3_neighborhood", lambda *_args: [entry])
    monkeypatch.setattr(ipa, "build_l3_candidates", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(ipa, "measure_propagated_costs", _measure)

    ipa.run_anchor_budget(args, runtime, 4.5)

    out = capsys.readouterr().out
    assert captured["callback_present"] is True
    assert (
        "[multi][l3] anchor 4.50: depth group 5/36 "
        "layers.20.self_attn.q_proj: start entries=7 lanes=14 mode=batched"
    ) in out
    assert (
        "[multi][l3] anchor 4.50: depth group 5/36 "
        "layers.20.self_attn.q_proj: done in 12.3s"
    ) in out


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
