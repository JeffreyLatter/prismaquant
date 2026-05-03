import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant import iterate_perturbed_allocation as ipa
from prismaquant import propagated_cost as pc
from prismaquant.iterate_perturbed_allocation import (
    assignment_hash,
    build_l3_polish_summary,
    resolve_two_cycle,
    smooth_cost_history,
    weighted_hamming_fraction,
)
from prismaquant.propagated_cost import L3NeighborhoodEntry


class _FakeQkvProfile:
    def fused_sibling_group(self, name: str) -> str | None:
        if name.endswith((".q_proj", ".k_proj", ".v_proj")):
            return name.rsplit(".", 1)[0] + ".qkv_proj"
        return None


def test_candidate_trial_changes_fused_siblings_atomically():
    assignment = {
        "model.layers.0.self_attn.q_proj": "BF16",
        "model.layers.0.self_attn.k_proj": "BF16",
        "model.layers.0.self_attn.v_proj": "BF16",
        "model.layers.0.self_attn.o_proj": "BF16",
    }

    trial = ipa._candidate_trial_assignment(
        assignment,
        "model.layers.0.self_attn.q_proj",
        "NVFP4",
        profile=_FakeQkvProfile(),
    )

    assert trial["model.layers.0.self_attn.q_proj"] == "NVFP4"
    assert trial["model.layers.0.self_attn.k_proj"] == "NVFP4"
    assert trial["model.layers.0.self_attn.v_proj"] == "NVFP4"
    assert trial["model.layers.0.self_attn.o_proj"] == "BF16"


def test_fused_assignment_coherence_promotes_mixed_group():
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.self_attn.k_proj": "BF16",
        "model.layers.0.self_attn.v_proj": "BF16",
    }
    specs = [ipa.fr.get_format("NVFP4"), ipa.fr.get_format("BF16")]

    coherent = ipa._enforce_fused_assignment_coherence(
        assignment,
        specs,
        profile=_FakeQkvProfile(),
    )

    assert set(coherent.values()) == {"BF16"}


def test_coord_replay_cache_default_is_opt_in(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_COORD_REPLAY_CACHE", raising=False)
    assert ipa._coord_replay_cache_enabled() is False

    monkeypatch.setenv("PRISMAQUANT_COORD_REPLAY_CACHE", "1")
    assert ipa._coord_replay_cache_enabled() is True

    monkeypatch.setenv("PRISMAQUANT_COORD_REPLAY_CACHE", "0")
    assert ipa._coord_replay_cache_enabled() is False


def test_l3_output_mse_default_is_opt_in():
    assert ipa._l3_output_mse_names(SimpleNamespace()) == []
    assert ipa._l3_output_mse_names(SimpleNamespace(l3_output_mse=False)) == []
    assert ipa._l3_output_mse_names(SimpleNamespace(l3_output_mse=True)) is None


def test_memory_aware_policy_keeps_resume_l3_calibration(monkeypatch):
    args = SimpleNamespace(
        memory_aware_l3=True,
        resume_l3_costs="anchor_bpp_4.50=/tmp/l3.pkl",
        resume_l3_costs_dir=None,
        n_calib_samples=8,
        calib_seqlen=512,
        l3_n_calib_samples=8,
        l3_calib_seqlen=512,
        input_rows=256,
        chunk_size=256,
        l3_max_lanes_per_batch=32,
        l3_interaction_max_lanes_per_batch=32,
        l3_validation_scout_max_candidates=64,
    )
    stats = {"layer": {"n_params": 4_000_000_000}}
    monkeypatch.setattr(ipa, "_gpu_memory_gb_for_model", lambda _model: (100.0, 128.0))
    monkeypatch.setattr(ipa, "_host_available_memory_gb", lambda: 120.0)
    monkeypatch.setattr(ipa, "_emit", lambda _msg: None)

    ipa._apply_memory_aware_runtime_policy(args, stats, model=None)

    assert args.n_calib_samples == 2
    assert args.calib_seqlen == 128
    assert args.l3_n_calib_samples == 8
    assert args.l3_calib_seqlen == 512
    assert args.input_rows == 64
    assert args.chunk_size == 64
    assert args.l3_max_lanes_per_batch == 8
    assert args.l3_interaction_max_lanes_per_batch == 6
    assert args.l3_validation_scout_max_candidates == 16


def test_memory_aware_policy_clamps_fresh_large_calibration(monkeypatch):
    args = SimpleNamespace(
        memory_aware_l3=True,
        resume_l3_costs=None,
        resume_l3_costs_dir=None,
        n_calib_samples=8,
        calib_seqlen=512,
        l3_n_calib_samples=8,
        l3_calib_seqlen=512,
        input_rows=256,
        chunk_size=256,
        l3_max_lanes_per_batch=32,
        l3_interaction_max_lanes_per_batch=32,
        l3_validation_scout_max_candidates=64,
    )
    stats = {"layer": {"n_params": 4_000_000_000}}
    monkeypatch.setattr(ipa, "_gpu_memory_gb_for_model", lambda _model: (100.0, 128.0))
    monkeypatch.setattr(ipa, "_host_available_memory_gb", lambda: 120.0)
    monkeypatch.setattr(ipa, "_emit", lambda _msg: None)

    ipa._apply_memory_aware_runtime_policy(args, stats, model=None)

    assert args.n_calib_samples == 2
    assert args.calib_seqlen == 128
    assert args.l3_n_calib_samples == 2
    assert args.l3_calib_seqlen == 128
    assert args.l3_interaction_max_lanes_per_batch == 6


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
        lambda _tokenizer, n_samples, seqlen, **_kwargs: torch.zeros(
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
    assert result.validation_kl == pytest.approx(2.0)
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

    def _load_calib(_tokenizer, n_samples, seqlen, **kwargs):
        calib_calls.append((n_samples, seqlen, kwargs))
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
    assert any(
        n_samples == 8
        and seqlen == 512
        and kwargs["split"] == "validation"
        and kwargs["seed"] == 4242
        for n_samples, seqlen, kwargs in calib_calls
    )
    assert any(
        n_samples == 1
        and seqlen == 2
        and kwargs["split"] == "train"
        and kwargs["seed"] == 42
        for n_samples, seqlen, kwargs in calib_calls
    )
    with open(output_dir / "l3_propagated_costs.pkl", "rb") as f:
        l3_payload = pickle.load(f)
    assert l3_payload["meta"]["l3_max_lanes_per_batch"] == 3
    assert l3_payload["meta"]["tail_only"] is False
    assert l3_payload["meta"]["kl_calib_split"] == "validation"
    assert l3_payload["meta"]["kl_calib_seed"] == 4242
    assert l3_payload["meta"]["l3_calib_split"] == "train"
    assert l3_payload["meta"]["l3_calib_seed"] == 42
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
        lambda _tokenizer, n_samples, seqlen, **_kwargs: torch.zeros(
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
        l3_interaction_refine=False,
        l3_interaction_top_units=8,
        l3_interaction_neighbor_radius=1,
        l3_interaction_max_pairs=0,
        l3_interaction_max_passes=4,
        l3_interaction_max_seconds=600.0,
        l3_interaction_exact=True,
        l3_interaction_exact_max_states=2_000_000,
        l3_validation_scout=False,
        l3_validation_scout_rounds=1,
        l3_validation_scout_max_candidates=64,
        l3_validation_scout_sample_per_bucket=0,
        l3_validation_scout_seed=0,
        l3_validation_scout_bpp_slack=0.25,
        l3_validation_scout_max_predicted_delta=0.0,
        l3_validation_scout_min_improvement=0.0,
        l3_validation_scout_commit_best=True,
        search_telemetry_jsonl=None,
        _search_telemetry_path=None,
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


def test_l3_interaction_refine_repairs_bad_pairwise_choice(tmp_path, monkeypatch):
    stats = {f"layer{i}": _tiny_stat() for i in range(2)}
    specs = [
        ipa.fr.get_format("NVFP4"),
        ipa.fr.get_format("MXFP8"),
        ipa.fr.get_format("BF16"),
    ]
    base_assignment = {name: "MXFP8" for name in stats}
    additive_candidate = {name: "NVFP4" for name in stats}
    l3_candidates = {
        name: [
            ipa.Candidate("NVFP4", bits_per_param=4.0, memory_bytes=2, predicted_dloss=0.0),
            ipa.Candidate("MXFP8", bits_per_param=8.0, memory_bytes=4, predicted_dloss=0.3),
        ]
        for name in stats
    }

    measured_overrides = []

    def _measure(_model, _assignment, overrides, *_args, **_kwargs):
        values = []
        for override in overrides:
            measured_overrides.append(dict(override))
            fmts = (override["layer0"], override["layer1"])
            if fmts == ("MXFP8", "MXFP8"):
                values.append(0.6)
            elif fmts == ("NVFP4", "NVFP4"):
                values.append(2.0)
            elif set(fmts) == {"MXFP8", "NVFP4"}:
                values.append(0.3)
            else:
                values.append(0.0)
        return values

    monkeypatch.setattr(ipa, "measure_override_paired_kl_deltas", _measure)

    refined, meta = ipa.run_l3_interaction_refine(
        _iterated_l3_args(
            l3_interaction_refine=True,
            l3_interaction_top_units=2,
            l3_interaction_neighbor_radius=2,
        ),
        _TinyLogitsModel(),
        base_assignment,
        additive_candidate,
        l3_candidates,
        stats,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        work_root=tmp_path,
    )

    assert meta["attempted"] is True
    assert meta["pair_measurements"] == len(measured_overrides)
    assert refined != additive_candidate
    assert sum(fmt == "NVFP4" for fmt in refined.values()) == 1


def test_l3_interaction_refine_preserves_non_selected_l3_flips(tmp_path, monkeypatch):
    stats = {f"layer{i}": _tiny_stat() for i in range(3)}
    specs = [
        ipa.fr.get_format("NVFP4"),
        ipa.fr.get_format("MXFP8"),
        ipa.fr.get_format("BF16"),
    ]
    base_assignment = {name: "MXFP8" for name in stats}
    additive_candidate = {name: "NVFP4" for name in stats}
    l3_candidates = {
        "layer0": [
            ipa.Candidate("NVFP4", bits_per_param=4.0, memory_bytes=2, predicted_dloss=0.0),
            ipa.Candidate("MXFP8", bits_per_param=8.0, memory_bytes=4, predicted_dloss=0.3),
        ],
        "layer1": [
            ipa.Candidate("NVFP4", bits_per_param=4.0, memory_bytes=2, predicted_dloss=0.0),
            ipa.Candidate("MXFP8", bits_per_param=8.0, memory_bytes=4, predicted_dloss=0.3),
        ],
        "layer2": [
            ipa.Candidate("NVFP4", bits_per_param=4.0, memory_bytes=2, predicted_dloss=0.1),
            ipa.Candidate("MXFP8", bits_per_param=8.0, memory_bytes=4, predicted_dloss=0.15),
        ],
    }

    def _measure(_model, _assignment, overrides, *_args, **_kwargs):
        values = []
        for override in overrides:
            assert set(override) == {"layer0", "layer1"}
            fmts = (override["layer0"], override["layer1"])
            if fmts == ("MXFP8", "MXFP8"):
                values.append(0.6)
            elif fmts == ("NVFP4", "NVFP4"):
                values.append(2.0)
            elif set(fmts) == {"MXFP8", "NVFP4"}:
                values.append(0.3)
            else:
                values.append(0.0)
        return values

    monkeypatch.setattr(ipa, "measure_override_paired_kl_deltas", _measure)

    refined, meta = ipa.run_l3_interaction_refine(
        _iterated_l3_args(
            l3_interaction_refine=True,
            l3_interaction_top_units=2,
            l3_interaction_neighbor_radius=2,
        ),
        _TinyLogitsModel(),
        base_assignment,
        additive_candidate,
        l3_candidates,
        stats,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        work_root=tmp_path,
    )

    assert meta["attempted"] is True
    assert refined["layer2"] == "NVFP4"
    assert sum(refined[name] == "NVFP4" for name in ("layer0", "layer1")) == 1


def test_l3_interaction_refine_starts_from_feasible_candidate(tmp_path, monkeypatch):
    stats = {f"layer{i}": _tiny_stat() for i in range(3)}
    specs = [
        ipa.fr.get_format("NVFP4"),
        ipa.fr.get_format("MXFP8"),
        ipa.fr.get_format("BF16"),
    ]
    base_assignment = {
        "layer0": "BF16",
        "layer1": "MXFP8",
        "layer2": "MXFP8",
    }
    additive_candidate = {
        "layer0": "NVFP4",
        "layer1": "BF16",
        "layer2": "MXFP8",
    }
    l3_candidates = {
        name: [
            ipa.Candidate("NVFP4", bits_per_param=4.0, memory_bytes=2, predicted_dloss=0.2),
            ipa.Candidate("MXFP8", bits_per_param=8.0, memory_bytes=4, predicted_dloss=0.1),
            ipa.Candidate("BF16", bits_per_param=16.0, memory_bytes=8, predicted_dloss=0.0),
        ]
        for name in stats
    }

    def _measure(_model, _assignment, overrides, *_args, **_kwargs):
        return [0.0 for _override in overrides]

    monkeypatch.setattr(ipa, "measure_override_paired_kl_deltas", _measure)

    refined, meta = ipa.run_l3_interaction_refine(
        _iterated_l3_args(
            l3_interaction_refine=True,
            l3_interaction_top_units=2,
            l3_interaction_neighbor_radius=2,
        ),
        _TinyLogitsModel(),
        base_assignment,
        additive_candidate,
        l3_candidates,
        stats,
        specs,
        8.0,
        torch.zeros((1, 2), dtype=torch.long),
        work_root=tmp_path,
    )

    assert meta["attempted"] is True
    assert meta["skipped_reason"] is None
    assert refined["layer0"] == "NVFP4"


def test_l3_interaction_refine_honors_time_budget(tmp_path, monkeypatch):
    stats = {f"layer{i}": _tiny_stat() for i in range(2)}
    specs = [
        ipa.fr.get_format("NVFP4"),
        ipa.fr.get_format("MXFP8"),
        ipa.fr.get_format("BF16"),
    ]
    base_assignment = {name: "MXFP8" for name in stats}
    additive_candidate = {name: "NVFP4" for name in stats}
    l3_candidates = {
        name: [
            ipa.Candidate("NVFP4", bits_per_param=4.0, memory_bytes=2, predicted_dloss=0.0),
            ipa.Candidate("MXFP8", bits_per_param=8.0, memory_bytes=4, predicted_dloss=0.3),
        ]
        for name in stats
    }

    def _measure(_model, _assignment, overrides, *_args, **kwargs):
        progress_callback = kwargs["progress_callback"]
        progress_callback({
            "event": "paired_override_chunk_start",
            "chunk_index": 1,
            "chunk_count": 1,
            "override_count": len(overrides),
            "lane_count": 1,
        })
        return [0.0 for _override in overrides]

    times = iter([0.0, 1.0, 1.0])
    monkeypatch.setattr(ipa.time, "monotonic", lambda: next(times, 1.0))
    monkeypatch.setattr(ipa, "measure_override_paired_kl_deltas", _measure)

    refined, meta = ipa.run_l3_interaction_refine(
        _iterated_l3_args(
            l3_interaction_refine=True,
            l3_interaction_top_units=2,
            l3_interaction_neighbor_radius=2,
            l3_interaction_max_seconds=0.5,
        ),
        _TinyLogitsModel(),
        base_assignment,
        additive_candidate,
        l3_candidates,
        stats,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        work_root=tmp_path,
    )

    assert refined == additive_candidate
    assert meta["attempted"] is True
    assert meta["stopped_reason"] == "time_budget_exceeded"
    assert meta["skipped_reason"] == "time_budget_exceeded"
    assert meta["pair_measurement_seconds"] == pytest.approx(1.0)


def test_l3_interaction_validation_keeps_better_additive_candidate(tmp_path, monkeypatch):
    stats = {f"layer{i}": _tiny_stat() for i in range(2)}
    specs = [ipa.fr.get_format("MXFP8"), ipa.fr.get_format("BF16")]
    initial = {name: "MXFP8" for name in stats}
    additive = {"layer0": "BF16", "layer1": "MXFP8"}
    interaction = {"layer0": "BF16", "layer1": "BF16"}
    l3_costs = {
        name: {
            "MXFP8": {"propagated_end_kl": 1.0},
            "BF16": {"propagated_end_kl": 0.0},
        }
        for name in stats
    }

    def _solve_l3(*_args, **_kwargs):
        return dict(additive), {}, {"frozen_dp_precision_used": 0.001}

    def _interaction_refine(*_args, **_kwargs):
        return dict(interaction), {
            "enabled": True,
            "attempted": True,
            "changed": True,
        }

    def _kl(_model, assignment, *_args, **_kwargs):
        if assignment == additive:
            return 1.0
        if assignment == interaction:
            return 2.0
        return 3.0

    monkeypatch.setattr(ipa, "_solve_l3_candidates_with_hamming_cap", _solve_l3)
    monkeypatch.setattr(ipa, "run_l3_interaction_refine", _interaction_refine)
    monkeypatch.setattr(ipa, "measure_assignment_kl", _kl)

    result = ipa.run_iterated_l3_polish(
        _iterated_l3_args(l3_iter_max=1),
        _TinyLogitsModel(),
        initial,
        3.0,
        stats,
        l3_costs,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.long),
        [],
        work_root=tmp_path,
        initial_l3_costs=l3_costs,
    )

    assert result.assignment == additive
    assert result.final_kl == pytest.approx(1.0)
    meta = result.iterations[0]["interaction_refine"]
    assert meta["validation_rollback_to_additive"] is True
    assert meta["additive_candidate_kl"] == pytest.approx(1.0)
    assert meta["interaction_candidate_kl"] == pytest.approx(2.0)


def test_l3_validation_scout_commits_best_validated_flip(tmp_path, monkeypatch):
    stats = {f"layer{i}": _tiny_stat() for i in range(2)}
    specs = [ipa.fr.get_format("NVFP4"), ipa.fr.get_format("MXFP8"), ipa.fr.get_format("BF16")]
    assignment = {"layer0": "NVFP4", "layer1": "NVFP4"}
    l3_costs = {
        "layer0": {
            "NVFP4": {"propagated_end_kl": 1.0},
            "MXFP8": {"propagated_end_kl": 0.5},
            "BF16": {"propagated_end_kl": 0.0},
        },
        "layer1": {
            "NVFP4": {"propagated_end_kl": 1.0},
            "MXFP8": {"propagated_end_kl": 0.8},
            "BF16": {"propagated_end_kl": 0.7},
        },
    }
    measured_batches = []

    def _measure(_model, _assignment, flips, *_args, **_kwargs):
        measured_batches.append(list(flips))
        values = []
        for name, fmt in flips:
            values.append(0.4 if (name, fmt) == ("layer0", "BF16") else 0.9)
        return values

    monkeypatch.setattr(ipa, "measure_lane_batched_kl_deltas", _measure)
    telemetry_path = tmp_path / "search_telemetry.jsonl"

    refined, final_kl, meta = ipa.run_l3_validation_scout(
        _iterated_l3_args(
            l3_validation_scout=True,
            l3_validation_scout_max_candidates=2,
            _search_telemetry_path=telemetry_path,
        ),
        _TinyLogitsModel(),
        assignment,
        1.0,
        l3_costs,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        [],
        stats=stats,
        work_root=tmp_path,
    )

    assert measured_batches
    assert refined["layer0"] == "BF16"
    assert final_kl == pytest.approx(0.4)
    assert meta["candidates_evaluated"] == 2
    assert meta["hits"] == 2
    assert meta["flips_committed"] == 1
    records = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert {record["event"] for record in records} == {
        "validation_scout_candidate",
        "validation_scout_commit",
        "validation_scout_round_start",
    }
    assert all(record["candidate_source"] == "l3_ranked_unary" for record in records)
    assert all(record["target_bpp"] == pytest.approx(16.0) for record in records)


def test_l3_fixed_point_remeasurement_gates_validation_scout(tmp_path, monkeypatch):
    stats = {f"layer{i}": _tiny_stat() for i in range(2)}
    specs = [ipa.fr.get_format("INT8_W8A16"), ipa.fr.get_format("BF16")]
    assignment = {"layer0": "INT8_W8A16", "layer1": "INT8_W8A16"}
    l3_costs = {
        "layer0": {
            "INT8_W8A16": {"propagated_end_kl": 1.0},
            "BF16": {"propagated_end_kl": 0.0},
        },
        "layer1": {
            "INT8_W8A16": {"propagated_end_kl": 1.0},
            "BF16": {"propagated_end_kl": 0.5},
        },
    }

    def _same_assignment(*_args, **_kwargs):
        return dict(assignment), [], {"frozen_dp_precision_used": 0.001}

    def _remeasured_kl(_model, measured_assignment, *_args, **_kwargs):
        assert measured_assignment == assignment
        return 0.5

    def _scout_measure(_model, _assignment, flips, *_args, **_kwargs):
        assert flips == [("layer0", "BF16")]
        return [0.75]

    monkeypatch.setattr(ipa, "_solve_l3_candidates_with_hamming_cap", _same_assignment)
    monkeypatch.setattr(ipa, "measure_assignment_kl", _remeasured_kl)
    monkeypatch.setattr(ipa, "measure_lane_batched_kl_deltas", _scout_measure)

    result = ipa.run_iterated_l3_polish(
        _iterated_l3_args(
            l3_iter_max=1,
            l3_validation_scout=True,
            l3_validation_scout_max_candidates=1,
        ),
        _TinyLogitsModel(),
        assignment,
        1.0,
        stats,
        l3_costs,
        specs,
        16.0,
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.long),
        [],
        work_root=tmp_path,
        initial_l3_costs=l3_costs,
    )

    assert result.final_kl == pytest.approx(0.5)
    assert result.assignment == assignment
    assert result.validation_scout["flips_committed"] == 0
    assert result.validation_scout["stopped_reason"] == "no_validated_improvement"
    assert result.iterations[0]["termination"] == "fixed_point"


def test_l3_validation_scout_samples_unrepresented_rank_buckets():
    rows = [
        {
            "rank": rank,
            "name": f"layer{rank}",
            "to_format": "BF16",
        }
        for rank in [0, 1, 2, 3, 4, 5, 40, 41, 140, 141, 600, 601]
    ]

    selected = ipa._select_validation_scout_candidates(
        _iterated_l3_args(
            l3_validation_scout_max_candidates=6,
            l3_validation_scout_sample_per_bucket=1,
            l3_validation_scout_seed=0,
        ),
        rows,
    )

    selected_buckets = {ipa._rank_bucket(row["rank"]) for row in selected}
    assert selected_buckets == {
        "rank_000_031",
        "rank_032_127",
        "rank_128_511",
        "rank_512_plus",
    }
    assert [row["rank"] for row in selected[:3]] == [0, 1, 2]


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kl_cuda_graphs_replay_matches_eager(tmp_path, monkeypatch):
    ipa._KL_CUDA_GRAPH_REGISTRY.clear()
    pc._CUDA_GRAPH_WARNED_LABELS.clear()
    scale50 = _scaled_weight_spec("KL_GRAPH_SCALE50", 0.5)
    scale90 = _scaled_weight_spec("KL_GRAPH_SCALE90", 0.9)
    monkeypatch.setitem(ipa.fr.REGISTRY, scale50.name, scale50)
    monkeypatch.setitem(ipa.fr.REGISTRY, scale90.name, scale90)
    calib_ids = torch.linspace(-1.0, 1.0, steps=3 * 2 * 33).reshape(3, 2, 33)
    assignments = [
        {"layer0": scale50.name, "layer1": "BF16"},
        {"layer0": scale90.name, "layer1": "BF16"},
    ]

    def _measure(graphs_enabled: bool) -> list[float]:
        monkeypatch.setenv(
            "PRISMAQUANT_KL_CUDA_GRAPHS",
            "1" if graphs_enabled else "0",
        )
        model = _WideStackLogitsModel(layers=2).eval().cuda()
        ref_log_probs = ipa.cache_reference_log_probs(
            model,
            calib_ids,
            next(model.parameters()).device,
        )
        return [
            ipa.measure_assignment_kl(
                model,
                assignment,
                calib_ids,
                ref_log_probs,
                work_root=tmp_path,
            )
            for assignment in assignments
        ]

    eager = _measure(False)
    warnings = []
    monkeypatch.setattr(
        pc,
        "_warn_cuda_graph_fallback_once",
        lambda label, exc: warnings.append((label, exc)),
    )
    graphed = _measure(True)
    assert graphed == pytest.approx(eager, abs=1e-9, rel=0.0)
    assert abs(graphed[0] - graphed[1]) > 1e-8
    assert ipa._KL_CUDA_GRAPH_REGISTRY.entries
    assert not ipa._KL_CUDA_GRAPH_REGISTRY.disabled_keys
    assert not warnings


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_safety_fallback_on_capture_failure(tmp_path, monkeypatch):
    scale75 = _scaled_weight_spec("KL_GRAPH_FALLBACK_SCALE75", 0.75)
    monkeypatch.setitem(ipa.fr.REGISTRY, scale75.name, scale75)
    assignment = {"layer0": scale75.name, "layer1": "BF16"}
    calib_ids = torch.linspace(-1.0, 1.0, steps=2 * 2 * 33).reshape(2, 2, 33)
    model = _WideStackLogitsModel(layers=2).eval().cuda()
    ref_log_probs = ipa.cache_reference_log_probs(
        model,
        calib_ids,
        next(model.parameters()).device,
    )
    monkeypatch.setenv("PRISMAQUANT_KL_CUDA_GRAPHS", "0")
    eager = ipa.measure_assignment_kl(
        model,
        assignment,
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
    )

    def _raise_capture(self, *args, **kwargs):
        raise RuntimeError("forced graph capture failure")

    monkeypatch.setattr(pc.CUDAGraphRegistry, "_capture", _raise_capture)
    monkeypatch.setenv("PRISMAQUANT_KL_CUDA_GRAPHS", "1")
    fallback = ipa.measure_assignment_kl(
        model,
        assignment,
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
    )

    assert fallback == pytest.approx(eager, abs=1e-9, rel=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kl_cuda_graphs_capture_succeeds(tmp_path, monkeypatch):
    ipa._KL_CUDA_GRAPH_REGISTRY.clear()
    pc._CUDA_GRAPH_WARNED_LABELS.clear()
    monkeypatch.setenv("PRISMAQUANT_KL_CUDA_GRAPHS", "1")
    model = _WideStackLogitsModel(layers=2).eval().cuda()
    calib_ids = torch.linspace(-1.0, 1.0, steps=2 * 2 * 33).reshape(2, 2, 33)
    ref_log_probs = ipa.cache_reference_log_probs(
        model,
        calib_ids,
        next(model.parameters()).device,
    )
    warnings = []
    captures = []
    original_capture = ipa._KL_CUDA_GRAPH_REGISTRY._capture

    def _record_warning(label, exc):
        warnings.append((label, exc))

    def _count_capture(*args, **kwargs):
        captures.append(True)
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(pc, "_warn_cuda_graph_fallback_once", _record_warning)
    monkeypatch.setattr(ipa._KL_CUDA_GRAPH_REGISTRY, "_capture", _count_capture)

    value = ipa.measure_assignment_kl(
        model,
        {"layer0": "FP8_E4M3", "layer1": "BF16"},
        calib_ids,
        ref_log_probs,
        work_root=tmp_path,
    )

    assert value >= 0.0
    assert captures
    assert ipa._KL_CUDA_GRAPH_REGISTRY.entries
    assert not ipa._KL_CUDA_GRAPH_REGISTRY.disabled_keys
    assert not warnings


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


def test_surrogate_grid_points_fixed_step():
    args = SimpleNamespace(
        knee_bpp_min=4.5,
        knee_bpp_max=5.0,
        knee_surrogate_bpp_step=0.25,
        knee_surrogate_points=99,
    )

    assert ipa._surrogate_grid_points(args) == [4.5, 4.75, 5.0]


def test_surrogate_knee_search_writes_knee_without_model(tmp_path):
    stats = {
        f"layer{i}": {
            "n_params": 4096,
            "in_features": 64,
            "out_features": 64,
            "h_trace": 1.0,
        }
        for i in range(4)
    }
    current_costs = {
        name: {
            "NVFP4": {"predicted_dloss": 1.0},
            "BF16": {"predicted_dloss": 0.0},
        }
        for name in stats
    }
    l3_costs = {
        "layer0": {
            "NVFP4": {"propagated_end_kl": 0.8},
            "BF16": {"propagated_end_kl": 0.0},
        },
        "layer1": {
            "NVFP4": {"propagated_end_kl": 0.5},
            "BF16": {"propagated_end_kl": 0.0},
        },
        "layer2": {
            "NVFP4": {"propagated_end_kl": 0.2},
            "BF16": {"propagated_end_kl": 0.0},
        },
        "layer3": {
            "NVFP4": {"propagated_end_kl": 0.1},
            "BF16": {"propagated_end_kl": 0.0},
        },
    }
    resume_path = tmp_path / "l3_propagated_costs.pkl"
    with open(resume_path, "wb") as f:
        pickle.dump(
            {
                "costs": l3_costs,
                "formats": ["NVFP4", "BF16"],
                "meta": {"anchor_bpp": 5.5},
            },
            f,
        )
    initial_config = tmp_path / "initial_assignment.json"
    initial_config.write_text(
        json.dumps({name: "NVFP4" for name in stats})
    )
    args = SimpleNamespace(
        target_bits_anchor=5.5,
        knee_bpp_min=4.5,
        knee_bpp_max=16.0,
        knee_surrogate_bpp_step=2.5,
        knee_surrogate_points=9,
        knee_surrogate_neighbors=1,
        _resume_l3_costs_single=resume_path,
        _resume_l3_costs_by_anchor={},
        resume_l3_costs_dir=None,
        initial_config=str(initial_config),
        bit_precision=0.25,
        frozen_dp_budget_tolerance=0.0,
    )
    specs = [ipa.fr.get_format("NVFP4"), ipa.fr.get_format("BF16")]

    assert ipa.run_surrogate_knee_search(
        args,
        stats=stats,
        current_costs=current_costs,
        specs=specs,
        output_root=tmp_path / "out",
    ) == 0

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["pareto"]["mode"] == "surrogate_knee_search"
    assert Path(summary["knee"]["layer_config"]).exists()
    assert Path(summary["knee"]["assignment"]).exists()


def test_surrogate_knee_cli_requires_validation_or_unsafe_flag(tmp_path, monkeypatch):
    probe_path = tmp_path / "probe.pkl"
    costs_path = tmp_path / "costs.pkl"
    stats = {"layer": _tiny_stat()}
    costs = {
        "layer": {
            "NVFP4": {"predicted_dloss": 1.0},
            "BF16": {"predicted_dloss": 0.0},
        }
    }
    with open(probe_path, "wb") as f:
        pickle.dump({"stats": stats}, f)
    with open(costs_path, "wb") as f:
        pickle.dump({"costs": costs}, f)
    monkeypatch.setattr(ipa, "validate_probe_payload", lambda *_args: None)
    monkeypatch.setattr(ipa, "validate_cost_payload", lambda *_args: None)

    with pytest.raises(RuntimeError, match="--knee-surrogate writes"):
        ipa.main([
            "--model", "tiny",
            "--probe", str(probe_path),
            "--initial-costs", str(costs_path),
            "--formats", "NVFP4,BF16",
            "--knee-search",
            "--knee-surrogate",
            "--l3-polish",
            "--work-dir", str(tmp_path / "work"),
            "--output-dir", str(tmp_path / "out"),
        ])


def test_endpoint_fallback_warning_emits(capsys):
    ipa._warn_endpoint_fallback("knee", {"endpoint_fallback": True})

    out = capsys.readouterr().out
    assert "[knee] warning:" in out
    assert "endpoint fallback" in out


def test_validated_surrogate_knee_uses_real_kl(tmp_path, monkeypatch):
    stats = {
        f"layer{i}": {
            "n_params": 4096,
            "in_features": 64,
            "out_features": 64,
            "h_trace": 1.0,
        }
        for i in range(3)
    }
    specs = [ipa.fr.get_format("NVFP4"), ipa.fr.get_format("BF16")]
    assignments = [
        {"layer0": "NVFP4", "layer1": "NVFP4", "layer2": "NVFP4"},
        {"layer0": "BF16", "layer1": "NVFP4", "layer2": "NVFP4"},
        {"layer0": "BF16", "layer1": "BF16", "layer2": "NVFP4"},
    ]
    frontier = []
    for idx, assignment in enumerate(assignments):
        assignment_path = tmp_path / f"a{idx}.json"
        layer_config_path = tmp_path / f"c{idx}.json"
        assignment_path.write_text(json.dumps(assignment))
        ipa.write_layer_config(assignment, layer_config_path)
        histogram = ipa._format_histogram(stats, assignment, specs, 0.0)
        frontier.append({
            "target_bpp": float(idx),
            "achieved_bpp": histogram["achieved_bpp"],
            "surrogate_loss": float(idx),
            "assignment_path": str(assignment_path),
            "layer_config_path": str(layer_config_path),
            "format_histogram": histogram,
        })
    out = tmp_path / "out"
    out.mkdir()
    (out / "surrogate_frontier.json").write_text(json.dumps({
        "knee": {"assignment": frontier[1]["assignment_path"]},
        "frontier": frontier,
    }))
    kl_by_hash = {
        ipa._assignment_digest(assignments[0]): 0.05,
        ipa._assignment_digest(assignments[1]): 0.01,
        ipa._assignment_digest(assignments[2]): 0.02,
    }

    def _fake_kl(_model, assignment, *_args, **_kwargs):
        return kl_by_hash[ipa._assignment_digest(assignment)]

    monkeypatch.setattr(ipa, "measure_assignment_kl", _fake_kl)
    runtime = SimpleNamespace(
        output_root=out,
        work_root=tmp_path / "work",
        stats=stats,
        specs=specs,
        model=object(),
        calib_ids=torch.zeros((1, 2), dtype=torch.long),
        ref_log_probs=[],
        profile=None,
    )
    args = SimpleNamespace(
        knee_include_assignment=[],
        knee_surrogate_validation_candidates=9,
        n_calib_samples=1,
        calib_seqlen=2,
    )

    assert ipa.run_validated_surrogate_knee_search(args, runtime) == 0
    summary = json.loads((out / "validated_summary.json").read_text())
    assert summary["knee"]["validation_kl"] == pytest.approx(0.01)
    assert Path(summary["knee"]["layer_config"]).exists()


def test_validated_surrogate_knee_caps_generated_candidates_but_keeps_includes(
    tmp_path, monkeypatch
):
    stats = {
        f"layer{i}": {
            "n_params": 4096,
            "in_features": 64,
            "out_features": 64,
            "h_trace": 1.0,
        }
        for i in range(3)
    }
    specs = [ipa.fr.get_format("NVFP4"), ipa.fr.get_format("BF16")]
    frontier_assignments = [
        {"layer0": "NVFP4", "layer1": "NVFP4", "layer2": "NVFP4"},
        {"layer0": "BF16", "layer1": "NVFP4", "layer2": "NVFP4"},
        {"layer0": "BF16", "layer1": "BF16", "layer2": "NVFP4"},
        {"layer0": "BF16", "layer1": "BF16", "layer2": "BF16"},
    ]
    frontier = []
    for idx, assignment in enumerate(frontier_assignments):
        assignment_path = tmp_path / f"frontier_{idx}.json"
        layer_config_path = tmp_path / f"frontier_{idx}_config.json"
        assignment_path.write_text(json.dumps(assignment))
        ipa.write_layer_config(assignment, layer_config_path)
        frontier.append({
            "target_bpp": float(idx),
            "achieved_bpp": ipa._format_histogram(
                stats, assignment, specs, 0.0
            )["achieved_bpp"],
            "surrogate_loss": float(idx),
            "assignment_path": str(assignment_path),
            "layer_config_path": str(layer_config_path),
            "format_histogram": ipa._format_histogram(stats, assignment, specs, 0.0),
        })
    included_assignment = {
        "layer0": "NVFP4",
        "layer1": "BF16",
        "layer2": "NVFP4",
    }
    included_path = tmp_path / "included.json"
    included_path.write_text(json.dumps(included_assignment))
    out = tmp_path / "out"
    out.mkdir()
    (out / "surrogate_frontier.json").write_text(json.dumps({
        "knee": {"assignment": frontier[1]["assignment_path"]},
        "frontier": frontier,
    }))
    calls = []

    def _fake_kl(_model, assignment, *_args, **_kwargs):
        calls.append(ipa._assignment_digest(assignment))
        return float(len(calls)) / 100.0

    monkeypatch.setattr(ipa, "measure_assignment_kl", _fake_kl)
    runtime = SimpleNamespace(
        output_root=out,
        work_root=tmp_path / "work",
        stats=stats,
        specs=specs,
        model=object(),
        calib_ids=torch.zeros((1, 2), dtype=torch.long),
        ref_log_probs=[],
        profile=None,
    )
    args = SimpleNamespace(
        knee_include_assignment=[str(included_path)],
        knee_surrogate_validation_candidates=2,
        n_calib_samples=1,
        calib_seqlen=2,
    )

    assert ipa.run_validated_surrogate_knee_search(args, runtime) == 0
    assert len(calls) == 3
    assert ipa._assignment_digest(frontier_assignments[2]) not in calls
    assert ipa._assignment_digest(included_assignment) in calls


def _dummy_budget_result(
    target_bpp,
    anchor_bpp,
    tmp_path,
    *,
    l2_kl=0.25,
    validation_kl=None,
    achieved_bpp=None,
    accepted=True,
):
    if validation_kl is None:
        validation_kl = 1.0 / float(target_bpp)
    if achieved_bpp is None:
        achieved_bpp = target_bpp
    return ipa.BudgetResult(
        target_bpp=float(target_bpp),
        anchor_bpp=float(anchor_bpp),
        distance_from_anchor=abs(float(target_bpp) - float(anchor_bpp)),
        anchor_stale=target_bpp != anchor_bpp,
        achieved_bpp=float(achieved_bpp),
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
        "likely non-additive cost interaction; consider tighter L3 scope"
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


def test_multi_budget_does_not_widen_format_filter_by_default(tmp_path, monkeypatch):
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
    assert flags == [False, False]


def test_multi_budget_l3_measure_all_formats_override(tmp_path, monkeypatch):
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
        l3_measure_all_formats=False,
    )
    assert ipa.run_multi_budget(args, SimpleNamespace(output_root=tmp_path)) == 0
    assert flags == [False, False]

    flags.clear()
    args.target_bits_list = "4.5"
    args.l3_measure_all_formats = True
    assert ipa.run_multi_budget(args, SimpleNamespace(output_root=tmp_path)) == 0
    assert flags == [True]


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


def test_nondominated_budget_results_respects_kl_noise_floor(tmp_path):
    results = [
        _dummy_budget_result(4.0, 4.0, tmp_path, validation_kl=0.100000),
        _dummy_budget_result(4.5, 4.5, tmp_path, validation_kl=0.099995),
        _dummy_budget_result(5.0, 5.0, tmp_path, validation_kl=0.099000),
    ]

    frontier = ipa._nondominated_budget_results(
        results,
        kl_noise_floor=0.000010,
    )

    assert [row.target_bpp for row in frontier] == [4.0, 5.0]


def test_kneedle_leave_one_out_diagnostic_flags_unstable():
    points = [
        (4.0, 0.20),
        (5.0, 0.10),
        (6.0, 0.05),
        (7.0, 0.049),
        (8.0, 0.048),
    ]
    knee_bpp, knee_kl, _score, _endpoint = ipa.segmented_kneedle_point(points)

    diagnostic = ipa.kneedle_leave_one_out_diagnostic(
        points,
        knee_bpp,
        knee_kl,
        tolerance_bpp=0.1,
        kl_noise_floor=0.000010,
    )

    assert diagnostic["enabled"] is True
    assert diagnostic["stable"] is False
    assert diagnostic["max_bpp_shift"] == pytest.approx(1.0)
    assert diagnostic["max_kl_shift"] == pytest.approx(0.05)


def test_validated_frontier_kneedle_quality_guard_selects_best_kl(tmp_path):
    initial_results = [
        _dummy_budget_result(
            5.0625,
            5.31,
            tmp_path,
            validation_kl=0.020272708032280207,
            achieved_bpp=5.076189995966115,
            accepted=False,
        ),
        _dummy_budget_result(
            5.1,
            5.31,
            tmp_path,
            validation_kl=0.020065429194801254,
            achieved_bpp=5.116327148043566,
            accepted=False,
        ),
        _dummy_budget_result(
            5.175,
            5.17,
            tmp_path,
            validation_kl=0.01953181641874835,
            achieved_bpp=5.277027027027027,
        ),
        _dummy_budget_result(
            5.3117184348527635,
            5.3117184348527635,
            tmp_path,
            validation_kl=0.015127555539947934,
            achieved_bpp=5.3117184348527635,
        ),
        _dummy_budget_result(
            5.3,
            5.3117184348527635,
            tmp_path,
            validation_kl=0.013117771479301155,
            achieved_bpp=5.323416700282372,
        ),
    ]

    def _unexpected(_bpp):
        raise AssertionError("seeded frontier should be sufficient")

    chosen, _results, meta = ipa.adaptive_validated_frontier_kneedle(
        _unexpected,
        5.05,
        5.55,
        tolerance=0.02,
        max_evaluations=len(initial_results),
        initial_points=len(initial_results),
        initial_results=initial_results,
    )

    assert chosen.target_bpp == pytest.approx(5.3)
    assert chosen.validation_kl == pytest.approx(0.013117771479301155)
    assert meta["selection_guard"]["applied"] is True
    assert meta["selection_guard"]["geometric_chosen_target_bpp"] == pytest.approx(
        5.175
    )


def test_validated_frontier_kneedle_warn_policy_keeps_geometric_knee(tmp_path):
    initial_results = [
        _dummy_budget_result(
            5.0625,
            5.31,
            tmp_path,
            validation_kl=0.020272708032280207,
            achieved_bpp=5.076189995966115,
        ),
        _dummy_budget_result(
            5.1,
            5.31,
            tmp_path,
            validation_kl=0.020065429194801254,
            achieved_bpp=5.116327148043566,
        ),
        _dummy_budget_result(
            5.175,
            5.17,
            tmp_path,
            validation_kl=0.01953181641874835,
            achieved_bpp=5.277027027027027,
        ),
        _dummy_budget_result(
            5.3117184348527635,
            5.3117184348527635,
            tmp_path,
            validation_kl=0.015127555539947934,
            achieved_bpp=5.3117184348527635,
        ),
        _dummy_budget_result(
            5.3,
            5.3117184348527635,
            tmp_path,
            validation_kl=0.013117771479301155,
            achieved_bpp=5.323416700282372,
        ),
    ]

    chosen, _results, meta = ipa.adaptive_validated_frontier_kneedle(
        lambda _bpp: (_ for _ in ()).throw(AssertionError("unexpected eval")),
        5.05,
        5.55,
        tolerance=0.02,
        max_evaluations=len(initial_results),
        initial_points=len(initial_results),
        initial_results=initial_results,
        unstable_policy="warn",
    )

    assert chosen.target_bpp == pytest.approx(5.175)
    assert meta["selection_guard"]["applied"] is False


def test_validated_frontier_kneedle_discards_empirically_dominated_points(tmp_path):
    # Higher-bpp probes can be worse when the allocator lands in a bad local
    # solution. The validated frontier search must not let those points define
    # the knee.
    kl_by_target = {
        4.0: 0.20,
        4.5: 0.15,
        5.0: 0.08,
        5.5: 0.015,
        6.0: 0.060,
        6.5: 0.050,
        7.0: 0.040,
        7.5: 0.035,
        8.0: 0.030,
    }

    def _evaluate(bpp):
        target = round(float(bpp), 1)
        achieved = 5.31 if target == 5.5 else target
        return _dummy_budget_result(
            target,
            target,
            tmp_path,
            validation_kl=kl_by_target[target],
            achieved_bpp=achieved,
        )

    chosen, _results, meta = ipa.adaptive_validated_frontier_kneedle(
        _evaluate,
        4.0,
        8.0,
        tolerance=0.25,
        max_evaluations=9,
        initial_points=9,
    )

    assert meta["mode"] == "validated_frontier_kneedle"
    assert meta["knee_kl_noise_floor"] == pytest.approx(
        ipa.DEFAULT_KNEE_KL_NOISE_FLOOR
    )
    assert meta["leave_one_out"]["enabled"] is True
    assert chosen.achieved_bpp <= 5.31
    assert all(
        point["achieved_bpp"] <= 5.31
        for point in meta["frontier_points"]
    )


def test_knee_search_uses_validated_frontier_metadata(tmp_path, monkeypatch):
    def _anchor(_args, _runtime, anchor_bpp, *, measure_all_formats=False):
        return SimpleNamespace(anchor_bpp=float(anchor_bpp))

    def _single(_args, target_bits, reusable_anchor=None):
        kl_by_target = {
            4.0: 0.20,
            5.0: 0.08,
            6.0: 0.06,
            7.0: 0.04,
            8.0: 0.03,
        }
        target = round(float(target_bits), 1)
        return _dummy_budget_result(
            target,
            reusable_anchor.anchor_bpp,
            tmp_path,
            validation_kl=kl_by_target[target],
        )

    monkeypatch.setattr(ipa, "run_anchor_budget", _anchor)
    monkeypatch.setattr(ipa, "run_single_budget", _single)
    args = SimpleNamespace(
        knee_mode="kneedle",
        knee_bpp_min=4.0,
        knee_bpp_max=8.0,
        knee_tolerance=0.25,
        knee_max_evaluations=5,
        knee_initial_points=5,
        target_bits_share_tolerance=0.25,
        l3_measure_all_formats=False,
    )

    assert ipa.run_knee_search(args, SimpleNamespace(output_root=tmp_path)) == 0

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["knee"]["mode"] == "validated_frontier_kneedle"
    assert summary["knee"]["chosen_achieved_bpp"] == pytest.approx(
        summary["knee"]["chosen_bpp"]
    )
    assert summary["knee"]["frontier_points"]


def test_knee_search_reuses_seed_frontier_without_remeasurement(tmp_path, monkeypatch):
    stats = {
        f"layer{i}": {
            "n_params": 4096,
            "in_features": 64,
            "out_features": 64,
            "h_trace": 1.0,
        }
        for i in range(3)
    }
    specs = [ipa.fr.get_format("NVFP4"), ipa.fr.get_format("BF16")]
    assignments = [
        {"layer0": "NVFP4", "layer1": "NVFP4", "layer2": "NVFP4"},
        {"layer0": "BF16", "layer1": "NVFP4", "layer2": "NVFP4"},
        {"layer0": "BF16", "layer1": "BF16", "layer2": "NVFP4"},
    ]
    rows = []
    for idx, (target, kl, assignment) in enumerate(
        zip((4.0, 6.0, 8.0), (0.20, 0.05, 0.03), assignments)
    ):
        assignment_path = tmp_path / f"seed_assignment_{idx}.json"
        assignment_path.write_text(json.dumps(assignment))
        rows.append(
            {
                "target_bpp": target,
                "achieved_bpp": target,
                "validation_kl": kl,
                "assignment_path": str(assignment_path),
            }
        )
    seed_path = tmp_path / "validated_frontier.json"
    seed_path.write_text(json.dumps({"validated_candidates": rows}))

    def _unexpected_anchor(*_args, **_kwargs):
        raise AssertionError("seeded knee search should not measure new points")

    monkeypatch.setattr(ipa, "run_anchor_budget", _unexpected_anchor)
    args = SimpleNamespace(
        knee_mode="kneedle",
        knee_bpp_min=4.0,
        knee_bpp_max=8.0,
        knee_tolerance=0.25,
        knee_max_evaluations=3,
        knee_initial_points=3,
        knee_seed_frontier=[str(seed_path)],
        target_bits_share_tolerance=0.25,
        l3_measure_all_formats=False,
    )

    assert ipa.run_knee_search(
        args,
        SimpleNamespace(output_root=tmp_path / "out", stats=stats, specs=specs),
    ) == 0

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["knee"]["seeded_evaluations"] == 3
    assert summary["knee"]["evaluations"] == 3
    assert summary["pareto"]["knee_seed_frontier"]["loaded"] == 3


def test_knee_checkpoint_frontier_can_seed_search(tmp_path):
    stats = {"layer": _tiny_stat()}
    specs = [ipa.fr.get_format("BF16")]
    assignment = {"layer": "BF16"}
    assignment_path, layer_config_path = ipa._write_budget_artifacts(
        tmp_path,
        "5.00",
        assignment,
    )
    histogram = ipa._format_histogram(stats, assignment, specs, 5.0)
    result = ipa.BudgetResult(
        target_bpp=5.0,
        anchor_bpp=5.0,
        distance_from_anchor=0.0,
        anchor_stale=False,
        achieved_bpp=float(histogram["achieved_bpp"]),
        predicted_dloss=0.0,
        l2_kl=0.02,
        validation_kl=0.01,
        accepted=True,
        regression=False,
        flips_accepted=1,
        format_histogram=histogram,
        assignment=assignment,
        assignment_path=str(assignment_path),
        layer_config_path=str(layer_config_path),
    )
    checkpoint_path = ipa._write_knee_checkpoint_point(
        tmp_path / "out",
        result,
        source="unit_test",
        metadata={"round": 1},
    )

    seeds, meta = ipa._load_knee_seed_results(
        SimpleNamespace(knee_seed_frontier=[str(checkpoint_path)]),
        SimpleNamespace(output_root=tmp_path / "fresh", stats=stats, specs=specs),
    )

    assert meta["loaded"] == 1
    assert len(seeds) == 1
    assert seeds[0].validation_kl == pytest.approx(0.01)
    assert seeds[0].assignment == assignment


def test_knee_seed_frontier_rejects_incoherent_fused_assignment(tmp_path):
    stats = {
        "model.layers.0.self_attn.q_proj": _tiny_stat(),
        "model.layers.0.self_attn.k_proj": _tiny_stat(),
        "model.layers.0.self_attn.v_proj": _tiny_stat(),
    }
    specs = [ipa.fr.get_format("NVFP4"), ipa.fr.get_format("BF16")]
    incoherent_assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4",
        "model.layers.0.self_attn.k_proj": "BF16",
        "model.layers.0.self_attn.v_proj": "BF16",
    }
    coherent_assignment = {
        "model.layers.0.self_attn.q_proj": "BF16",
        "model.layers.0.self_attn.k_proj": "BF16",
        "model.layers.0.self_attn.v_proj": "BF16",
    }
    incoherent_path = tmp_path / "incoherent_assignment.json"
    coherent_path = tmp_path / "coherent_assignment.json"
    incoherent_path.write_text(json.dumps(incoherent_assignment))
    coherent_path.write_text(json.dumps(coherent_assignment))
    seed_path = tmp_path / "validated_frontier.json"
    seed_path.write_text(
        json.dumps(
            {
                "validated_candidates": [
                    {
                        "target_bpp": 5.3,
                        "achieved_bpp": 5.3,
                        "validation_kl": 0.01,
                        "assignment_path": str(incoherent_path),
                    },
                    {
                        "target_bpp": 5.5,
                        "achieved_bpp": 5.5,
                        "validation_kl": 0.02,
                        "assignment_path": str(coherent_path),
                    },
                ]
            }
        )
    )

    seeds, meta = ipa._load_knee_seed_results(
        SimpleNamespace(knee_seed_frontier=[str(seed_path)]),
        SimpleNamespace(
            output_root=tmp_path / "fresh",
            stats=stats,
            specs=specs,
            profile=_FakeQkvProfile(),
        ),
    )

    assert meta["loaded"] == 1
    assert meta["files"][0]["loaded"] == 1
    assert meta["files"][0]["skipped_rows"] == 1
    assert len(seeds) == 1
    assert seeds[0].assignment == coherent_assignment


def test_quality_equivalent_search_finds_lowest_matching_bpp(tmp_path, monkeypatch):
    anchor_calls = []

    def _anchor(_args, _runtime, anchor_bpp, *, measure_all_formats=False):
        anchor_calls.append((float(anchor_bpp), bool(measure_all_formats)))
        return SimpleNamespace(anchor_bpp=float(anchor_bpp))

    def _single(_args, target_bits, reusable_anchor=None):
        validation_kl = max(0.0, 8.0 - float(target_bits))
        return _dummy_budget_result(
            target_bits,
            reusable_anchor.anchor_bpp,
            tmp_path,
            validation_kl=validation_kl,
        )

    monkeypatch.setattr(ipa, "run_anchor_budget", _anchor)
    monkeypatch.setattr(ipa, "run_single_budget", _single)
    args = SimpleNamespace(
        quality_equivalent_bits=8.0,
        quality_equivalent_bpp_min=4.0,
        quality_equivalent_bpp_tolerance=0.1,
        quality_equivalent_max_evaluations=20,
        quality_equivalent_kl_slack=2.0,
        target_bits_share_tolerance=0.25,
        l3_measure_all_formats=False,
    )

    assert ipa.run_quality_equivalent_search(
        args,
        SimpleNamespace(output_root=tmp_path),
    ) == 0

    with open(tmp_path / "summary.json") as f:
        summary = json.load(f)
    qeq = summary["quality_equivalent"]
    assert qeq["reference_bpp"] == pytest.approx(8.0)
    assert qeq["reference_validation_kl"] == pytest.approx(0.0)
    assert qeq["threshold_kl"] == pytest.approx(2.0)
    assert qeq["chosen_bpp"] == pytest.approx(6.0, abs=0.1)
    assert qeq["validation_kl"] <= qeq["threshold_kl"]
    assert qeq["knee_candidate"]["mode"] == "kneedle_on_evaluated_points"
    assert anchor_calls


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
        lambda _tokenizer, n_samples, seqlen, **_kwargs: torch.zeros(
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
    assert captured["l3_mode"] == "selective"
    assert captured["targets"] == [4.5, 5.0]
    assert isinstance(captured["model"], _TinyLogitsModel)


def test_main_preserves_explicit_global_l3_for_target_bits_list(
    tmp_path,
    monkeypatch,
):
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

    captured = {}

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_Tokenizer),
    )
    monkeypatch.setattr(ipa, "validate_probe_payload", lambda *_args: None)
    monkeypatch.setattr(ipa, "validate_cost_payload", lambda *_args: None)
    monkeypatch.setattr(ipa, "load_text_model_under_work_root", lambda *_args, **_kwargs: _TinyLogitsModel())
    monkeypatch.setattr(
        ipa,
        "load_wikitext_calibration",
        lambda _tokenizer, n_samples, seqlen, **_kwargs: torch.zeros(
            (n_samples, seqlen),
            dtype=torch.long,
        ),
    )
    monkeypatch.setattr(ipa, "cache_reference_log_probs", lambda *_args: [])
    def _run_multi(args, _runtime):
        captured["l3_mode"] = args.l3_mode
        return 0

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
        "--l3-mode", "global",
    ])

    assert rc == 0
    assert captured["l3_mode"] == "global"
