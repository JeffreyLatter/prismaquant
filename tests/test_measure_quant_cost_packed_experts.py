from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.measure_quant_cost import (
    ActivationIndex,
    _batched_quantize,
    _finalize_results,
    _measure_packed_experts,
    _packed_experts_forward_with_weights,
    _packed_router_topk,
)


class _StubHDetail:
    """Minimal HDetailIndex stand-in returning fixed per-channel Fisher."""

    def __init__(self, mapping: dict[str, torch.Tensor]):
        self._m = mapping

    def __contains__(self, name: str) -> bool:
        return name in self._m

    def load(self, name: str) -> torch.Tensor:
        return self._m[name]
from prismaquant.allocator_candidates import cost_entry_uses_measured_output_mse


class TinyRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.top_k = 1
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        logits = F.linear(hidden_states, self.weight)
        scores, indices = torch.topk(torch.softmax(logits.float(), dim=-1), 1, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return logits, scores.to(hidden_states.dtype), indices


class TinyPackedExperts(nn.Module):
    def __init__(self, hidden_size: int = 16, intermediate_size: int = 16, num_experts: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(
                current_state,
                self.gate_up_proj[expert_idx],
            ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states,
                self.down_proj[expert_idx],
            )
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )
        return final_hidden_states


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = TinyRouter(hidden_size=16, num_experts=2)
        self.experts = TinyPackedExperts()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMlp()


def _write_activation_cache(cache_dir, name: str, inputs: torch.Tensor) -> None:
    fname = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
    torch.save({"inputs": inputs, "name": name}, cache_dir / fname)


def test_packed_expert_dloss_is_mean_field_not_product_of_sums():
    """Guard the packed-expert Δloss scale fix.

    h_em[e,m] = Σ_n grad² is the per-row SUM over in-features (channel
    accumulator). The correct on-scale Δloss is the mean-field estimate
    0.5·Σ h_em·mean_n(err²), matching the dense path's 0.5·Σ g²·err². The
    previous code multiplied by N (=in-features), turning it into a
    product-of-sums (Σg²)(Σerr²) that over-counts ~N× and over-promotes
    experts in the allocator. This test pins the mean-field value and
    rejects the ×N regression.
    """
    torch.manual_seed(7)
    model = TinyModel().eval()
    target_names = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}

    # Known per-channel Fisher [E, M] matching each packed weight's (E, M).
    h_map: dict[str, torch.Tensor] = {}
    for name in target_names:
        w = dict(model.named_parameters())[name]
        h_map[name] = torch.rand(w.size(0), w.size(1), dtype=torch.float32) + 0.1
    h_detail = _StubHDetail(h_map)

    spec = fr.get_format("NVFP4")
    accum: dict = {}
    _measure_packed_experts(
        model, target_names, [spec], "cpu", torch.float32, accum,
        act_cache=None, h_detail=h_detail,
    )
    results = _finalize_results(accum)

    for name in target_names:
        w = dict(model.named_parameters())[name].detach().float()
        n_in = w.size(-1)
        err = (w - _batched_quantize(spec, w)).float()
        h_em = h_map[name]
        expected_mean_field = float(
            0.5 * (h_em * err.pow(2).mean(dim=-1)).sum().item()
        )
        old_product_of_sums = expected_mean_field * n_in  # the ×N regression
        got = float(results[name]["NVFP4"]["predicted_dloss"])

        # Matches the mean-field (no ×N) value...
        assert abs(got - expected_mean_field) <= 1e-4 * max(expected_mean_field, 1e-12), (
            f"{name}: dloss {got} != mean-field {expected_mean_field}"
        )
        # ...and is NOT the ~N× product-of-sums (n_in=16 here, so a clear gap).
        assert got < 0.5 * old_product_of_sums, (
            f"{name}: dloss {got} looks like the ×N product-of-sums "
            f"{old_product_of_sums} (regression)"
        )


def test_packed_experts_measure_output_mse_from_expert_activation_cache(tmp_path):
    torch.manual_seed(1234)
    model = TinyModel().eval()
    experts_qname = "mlp.experts"
    target_names = {
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    }
    _write_activation_cache(tmp_path, experts_qname, torch.randn(8, 16))
    act_cache = ActivationIndex(
        tmp_path,
        {
            name: {"_packed_experts_module": experts_qname}
            for name in target_names
        },
    )

    accum: dict = {}
    _measure_packed_experts(
        model,
        target_names,
        [fr.get_format("NVFP4"), fr.get_format("BF16")],
        "cpu",
        torch.float32,
        accum,
        act_cache=act_cache,
    )
    results = _finalize_results(accum)

    assert set(results) == target_names
    for name in target_names:
        assert results[name]["BF16"].get("output_mse_measured", True) is True
        assert results[name]["NVFP4"].get("output_mse_measured", True) is True
        assert results[name]["BF16"]["output_mse"] == 0.0
        assert results[name]["NVFP4"]["output_mse"] > 0.0
        assert cost_entry_uses_measured_output_mse(
            {"_packed_experts_module": experts_qname},
            results[name]["NVFP4"],
        )


def test_packed_experts_use_holdout_gated_ladder_and_keep_expert_mse(
    tmp_path, monkeypatch,
):
    """Packed Qwen stacks must not silently encode every ladder rung."""
    import prismaquant.measure_quant_cost as mqc

    torch.manual_seed(4321)
    model = TinyModel().eval()
    experts_qname = "mlp.experts"
    target_names = {
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    }
    _write_activation_cache(tmp_path, experts_qname, torch.randn(16, 16))
    act_cache = ActivationIndex(
        tmp_path,
        {
            name: {"_packed_experts_module": experts_qname}
            for name in target_names
        },
    )
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "0")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "balanced")
    monkeypatch.setenv("PRISMAQUANT_CB_LADDER_INTERP", "1")
    monkeypatch.setenv(
        "PRISMAQUANT_CB_LADDER_ANCHORS", "FP8_CB_K28,FP8_CB_K48")
    monkeypatch.setenv("PRISMAQUANT_CB_LADDER_HOLDOUT", "FP8_CB_K33")
    monkeypatch.setattr(
        mqc,
        "_packed_expert_activation_quantizer",
        lambda _spec: (lambda value: value),
    )
    calls: list[str] = []

    def fake_render(spec, weight, **_kwargs):
        calls.append(spec.name)
        k = int(spec.name.rsplit("K", 1)[1])
        distortion = mqc._ladder_rate_factor(spec.name, k)
        return weight * (1.0 - distortion ** 0.5)

    monkeypatch.setattr(mqc, "_cb_cost_quantize_dequantize", fake_render)
    specs = [
        fr.get_format(name)
        for name in (
            "FP8_CB_K28", "FP8_CB_K33", "FP8_CB_K34", "FP8_CB_K48",
        )
    ]
    accum: dict = {}
    _measure_packed_experts(
        model,
        target_names,
        specs,
        "cpu",
        torch.float32,
        accum,
        act_cache=act_cache,
    )
    results = _finalize_results(accum)

    assert calls.count("FP8_CB_K34") == 0
    assert len(calls) == 3 * len(target_names)
    for name in target_names:
        assert set(results[name]) == {spec.name for spec in specs}
        predicted = results[name]["FP8_CB_K34"]
        assert predicted["cost_source"] == "band_interpolated"
        assert predicted["output_mse_measured"] is False
        assert len(predicted["weight_mse_per_expert"]) == 2
        assert predicted["cost_source_per_expert"] == [
            "band_interpolated", "band_interpolated",
        ]


def test_packed_experts_gate_per_slice_and_measure_only_rejects(
    tmp_path, monkeypatch,
):
    """One non-monotone expert must not force a whole-stack fallback."""
    import prismaquant.measure_quant_cost as mqc

    torch.manual_seed(8765)
    model = TinyModel().eval()
    experts_qname = "mlp.experts"
    target_names = {
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    }
    _write_activation_cache(tmp_path, experts_qname, torch.randn(16, 16))
    act_cache = ActivationIndex(
        tmp_path,
        {
            name: {"_packed_experts_module": experts_qname}
            for name in target_names
        },
    )
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "0")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "balanced")
    monkeypatch.setenv("PRISMAQUANT_CB_LADDER_INTERP", "1")
    monkeypatch.setenv(
        "PRISMAQUANT_CB_LADDER_ANCHORS",
        "FP8_CB_K28,FP8_CB_K38,FP8_CB_K48",
    )
    monkeypatch.setenv("PRISMAQUANT_CB_LADDER_HOLDOUT", "FP8_CB_K33")
    monkeypatch.setattr(
        mqc,
        "_packed_expert_activation_quantizer",
        lambda _spec: (lambda value: value),
    )
    calls: list[tuple[str, int]] = []

    def fake_render(spec, weight, **_kwargs):
        calls.append((spec.name, int(weight.shape[0])))
        k = int(spec.name.rsplit("K", 1)[1])
        distortion = mqc._ladder_rate_factor(spec.name, k)
        factors = torch.ones(
            weight.shape[0], 1, 1, dtype=weight.dtype, device=weight.device,
        )
        # Full-stack K33 is the holdout: expert 1 is deliberately off-law.
        # The K34 fallback arrives as a one-expert sub-batch and stays exact.
        if spec.name == "FP8_CB_K33" and weight.shape[0] == 2:
            factors[1] = 2.0
        return weight * (1.0 - (distortion * factors).sqrt())

    monkeypatch.setattr(mqc, "_cb_cost_quantize_dequantize", fake_render)
    specs = [
        fr.get_format(name)
        for name in (
            "FP8_CB_K28", "FP8_CB_K33", "FP8_CB_K34",
            "FP8_CB_K38", "FP8_CB_K48",
        )
    ]
    accum: dict = {}
    _measure_packed_experts(
        model,
        target_names,
        specs,
        "cpu",
        torch.float32,
        accum,
        act_cache=act_cache,
    )
    results = _finalize_results(accum)

    # One K34 call per packed tensor, each containing only expert slice 1.
    assert [batch for name, batch in calls if name == "FP8_CB_K34"] == [1, 1]
    for name in target_names:
        row = results[name]["FP8_CB_K34"]
        assert row["cost_source"] == "mixed"
        assert row["cost_source_per_expert"] == [
            "band_interpolated", "measured",
        ]
        assert row["weight_mse"] == pytest.approx(
            sum(row["weight_mse_per_expert"]) / 2.0,
        )
        assert row["output_mse_measured"] is False


def test_packed_experts_replay_matches_module_forward():
    torch.manual_seed(5678)
    model = TinyModel().eval()
    X = torch.randn(11, 16)

    top_k_index, top_k_weights = _packed_router_topk(model.mlp.gate, X)
    y_module = model.mlp.experts(X, top_k_index, top_k_weights)
    y_replay = _packed_experts_forward_with_weights(
        model.mlp.experts,
        X,
        top_k_index,
        top_k_weights,
        model.mlp.experts.gate_up_proj,
        model.mlp.experts.down_proj,
    )

    assert torch.allclose(y_replay, y_module)


def test_packed_experts_replay_honors_apply_gate_clamp():
    torch.manual_seed(9012)

    class ClampExperts(TinyPackedExperts):
        def __init__(self):
            super().__init__()
            self.limit = 1.0
            with torch.no_grad():
                self.gate_up_proj.mul_(4.0)

        def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            return self.act_fn(gate) * up

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                gate_up = F.linear(
                    hidden_states[token_idx],
                    self.gate_up_proj[expert_idx],
                )
                current_hidden_states = self._apply_gate(gate_up)
                current_hidden_states = F.linear(
                    current_hidden_states,
                    self.down_proj[expert_idx],
                )
                current_hidden_states = (
                    current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                )
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(final_hidden_states.dtype),
                )
            return final_hidden_states

    experts = ClampExperts().eval()
    X = torch.randn(13, 16)
    top_k_index = torch.randint(0, experts.num_experts, (13, 1))
    top_k_weights = torch.ones(13, 1)

    y_module = experts(X, top_k_index, top_k_weights)
    y_replay = _packed_experts_forward_with_weights(
        experts,
        X,
        top_k_index,
        top_k_weights,
        experts.gate_up_proj,
        experts.down_proj,
    )

    assert torch.allclose(y_replay, y_module, atol=1e-5)


def test_packed_router_topk_accepts_indices_weights_tuple_order():
    class SwappedRouter(nn.Module):
        def forward(self, hidden_states: torch.Tensor):
            indices = torch.zeros(hidden_states.size(0), 1, dtype=torch.long)
            weights = torch.ones(hidden_states.size(0), 1)
            return torch.empty(hidden_states.size(0), 1), indices, weights

    indices, weights = _packed_router_topk(SwappedRouter(), torch.randn(3, 16))

    assert indices.dtype == torch.long
    assert torch.equal(indices, torch.zeros(3, 1, dtype=torch.long))
    assert torch.equal(weights, torch.ones(3, 1))


def test_skip_packed_expert_cost_env(monkeypatch):
    """PRISMAQUANT_SKIP_PACKED_EXPERT_COST=1 (the CB M4-hybrid pipeline sets
    it when the empirical stage will REPLACE every expert row) must skip all
    packed-expert measurement — the local measurements are discarded work."""
    torch.manual_seed(7)
    model = TinyModel().eval()
    target_names = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    spec = fr.get_format("NVFP4")

    monkeypatch.setenv("PRISMAQUANT_SKIP_PACKED_EXPERT_COST", "1")
    accum: dict = {}
    _measure_packed_experts(
        model, target_names, [spec], "cpu", torch.float32, accum,
        act_cache=None, h_detail=None,
    )
    assert accum == {}

    monkeypatch.delenv("PRISMAQUANT_SKIP_PACKED_EXPERT_COST")
    _measure_packed_experts(
        model, target_names, [spec], "cpu", torch.float32, accum,
        act_cache=None, h_detail=None,
    )
    assert set(accum) == target_names


def test_dense_ladder_helpers(monkeypatch):
    """Dense-path ladder plumbing: env gate, exact-law fit, chunk-metric
    readback from the sum-based accumulator."""
    from prismaquant.measure_quant_cost import (
        _CB_LADDER_TOL,
        _accumulate_result,
        _cb_ladder_plan,
        _chunk_metric,
        _ladder_metric_fit,
        _ladder_rate_factor,
    )
    import prismaquant.format_registry as fr

    specs = [fr.get_format(f"FP8_CB_K{k}") for k in (28, 32, 36, 40, 44, 48)]
    monkeypatch.delenv("PRISMAQUANT_CB_LADDER_INTERP", raising=False)
    assert _cb_ladder_plan(specs) is None          # default OFF
    monkeypatch.setenv("PRISMAQUANT_CB_LADDER_INTERP", "1")
    plan = _cb_ladder_plan(specs)
    assert plan is not None
    ladders, predicted = plan
    assert len(ladders) == 1 and len(predicted) >= 2
    kmap, anchors, holdout, pred = ladders[0]
    # 6-rung family -> 3 anchors, 1 holdout, 2 predicted.
    assert len(anchors) == 3 and len(pred) == 2
    # Exact recovery of the law's OWN model. Commit 5184892 replaced the
    # log-linear 2^(a - b*k) fit with a split-aware FLOORED LINEAR law
    # D = F + C*R(k), where R(k) = sum_i 2^(-2*b_i/d_i) is the exact
    # ceil-first per-sub rate factor (_ladder_rate_factor) — that is what
    # kills the k % n_sub sawtooth a smooth-in-k law cannot represent. It
    # pins the decay rate to the theory-exact R and fits only (floor,
    # amplitude), so it is exact on split-model data.
    vals = {f: 0.002 + 1.5 * _ladder_rate_factor(f, kmap[f]) for f in kmap}
    for f in pred + [holdout]:
        got = _ladder_metric_fit(kmap, anchors, vals, f)
        assert got == pytest.approx(vals[f], rel=1e-9)

    # F < 0 -> the floor clamps to 0 and C is refit through the origin;
    # still exact.
    vals0 = {f: 1.5 * _ladder_rate_factor(f, kmap[f]) for f in kmap}
    for f in pred + [holdout]:
        got = _ladder_metric_fit(kmap, anchors, vals0, f)
        assert got == pytest.approx(vals0[f], rel=1e-9)

    # A FREE-rate exponential (2^(5 - 0.3k); decay != R's) is deliberately
    # NOT exact — the fit reads +14.2% high at k=32. The holdout gate is what
    # makes that safe: rel_err at the holdout is 31%, far over _CB_LADDER_TOL,
    # so measure_batched_gpu (measure_quant_cost.py:1975) rejects the fit and
    # measures the predicted rungs instead. Never let this land silently.
    expo = {f: 2.0 ** (5.0 - 0.3 * kmap[f]) for f in kmap}
    pred_h = _ladder_metric_fit(kmap, anchors, expo, holdout)
    assert abs(pred_h - expo[holdout]) / expo[holdout] > _CB_LADDER_TOL

    # Accumulator readback (sum-based, count=1).
    accum: dict = {}
    _accumulate_result(accum, "t", "FP8_CB_K36", 0.5, 0.25, 0.1,
                       predicted_dloss=0.02)
    assert _chunk_metric(accum, "t", "FP8_CB_K36", "weight_mse") == 0.5
    assert _chunk_metric(accum, "t", "FP8_CB_K36", "output_mse") == 0.25
    assert _chunk_metric(
        accum, "t", "FP8_CB_K36", "predicted_dloss") == 0.02
    assert _chunk_metric(
        accum, "t", "FP8_CB_K36", "fisher_output_mse") is None
    assert _chunk_metric(accum, "missing", "FP8_CB_K36", "output_mse") is None


def test_cost_provenance_stamps_explicit_ladder_plan(monkeypatch):
    from prismaquant.measure_quant_cost import cost_payload_provenance
    import prismaquant.format_registry as fr

    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "balanced")
    monkeypatch.setenv(
        "PRISMAQUANT_CB_LADDER_ANCHORS",
        "FP8_CB_K28,FP8_CB_K38,FP8_CB_K48",
    )
    monkeypatch.setenv("PRISMAQUANT_CB_LADDER_HOLDOUT", "FP8_CB_K33")
    provenance = cost_payload_provenance([fr.get_format("FP8_CB_K28")])
    assert provenance["cb_ladder_measurement_plan"] == {
        "anchors": ["FP8_CB_K28", "FP8_CB_K38", "FP8_CB_K48"],
        "holdout": "FP8_CB_K33",
    }
    assert provenance["cb_serialized_payload"]["ldlq"] is True


def test_cb_ladder_chain_is_shared_with_the_expert_path():
    """R20 (2026-07-30): ONE law behind both cost chains.

    The dense path fitted the split-aware floored-linear law in R(k) while
    the expert path ran a separate floor-law -> log-linear chain with NO
    R(k) term, so the ceil-first sawtooth that motivated the dense change
    (5184892) was still costing the expert ladder its holdouts. Both entry
    points now delegate to `expert_empirical_cost._cb_ladder_law`.
    """
    import math

    from prismaquant.expert_empirical_cost import (
        _cb_ladder_law,
        _cb_ladder_rate_factor,
    )
    from prismaquant.measure_quant_cost import (
        _ladder_metric_fit,
        _ladder_rate_factor,
    )

    # The dense name is a re-export of the canonical rate factor.
    assert _ladder_rate_factor("FP8_CB_K39", 39) == _cb_ladder_rate_factor(
        "FP8_CB_K39", 39)

    # THE sawtooth case from the 2026-07-21 27B cost run: even-split anchors
    # (28, 38, 48) and an odd-phase target k=39. The shared law is exact on
    # its own model at every phase; a log-linear fit through the SAME
    # anchors — what the expert chain used to run — misses k=39 by 14.9%,
    # which is where the 8-12% holdout rejections came from.
    kmap = {f"FP8_CB_K{k}": k for k in (28, 38, 39, 48)}
    anchors = ["FP8_CB_K28", "FP8_CB_K38", "FP8_CB_K48"]
    vals = {f: 0.002 + 1.5 * _cb_ladder_rate_factor(f, k)
            for f, k in kmap.items()}
    law = _cb_ladder_law(kmap, anchors, vals)
    assert law.name == "floored_linear_R"
    assert law.predict("FP8_CB_K39") == pytest.approx(
        vals["FP8_CB_K39"], rel=1e-9)
    # dense entry point == shared law
    assert _ladder_metric_fit(kmap, anchors, vals, "FP8_CB_K39") == (
        pytest.approx(law.predict("FP8_CB_K39")))
    xs = [float(kmap[f]) for f in anchors]
    ys = [math.log2(vals[f]) for f in anchors]
    mx, my = sum(xs) / 3.0, sum(ys) / 3.0
    b = -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs)
    log_linear = 2.0 ** ((my + b * mx) - b * 39)
    assert abs(log_linear - vals["FP8_CB_K39"]) / vals["FP8_CB_K39"] > 0.10


def test_cb_ladder_tolerance_is_derived_from_measured_noise():
    """R20: the holdout gate's threshold comes from the rungs' own
    between-window noise (encode_tiers.md B), not the bare 0.10.

    The 0.10 stays only where that datum is absent or degenerate — which is
    every dense-path fit, since it measures each (tensor, format) exactly
    once.
    """
    from prismaquant.expert_empirical_cost import (
        _cb_ladder_gate,
        _cb_ladder_rate_factor,
    )

    kmap = {f"FP8_CB_K{k}": k for k in (28, 32, 36, 40, 44, 48)}
    anchors = ["FP8_CB_K28", "FP8_CB_K40", "FP8_CB_K48"]
    holdout = "FP8_CB_K36"
    vals = {f: 0.002 + 1.5 * _cb_ladder_rate_factor(f, k)
            for f, k in kmap.items()}

    # No windows at all (the dense path) -> the floor stands.
    law, rel, tol = _cb_ladder_gate(kmap, anchors, vals, holdout, 0.10, None)
    assert law is not None and rel < 1e-9 and tol == 0.10
    # Zero-spread windows carry no resolution -> treated as absent.
    flat = {f: [v] * 4 for f, v in vals.items()}
    assert _cb_ladder_gate(
        kmap, anchors, vals, holdout, 0.10, flat)[2] == 0.10

    # A real per-window spread on the holdout: the paired residual's
    # standard error is 1.29%, so the gate TIGHTENS from 10% to 1.29%.
    jitter = [1.03, 0.97, 1.01, 0.99]
    win = {f: [v] * 4 for f, v in vals.items()}
    win[holdout] = [vals[holdout] * j for j in jitter]
    law, rel, tol = _cb_ladder_gate(kmap, anchors, vals, holdout, 0.10, win)
    assert tol == pytest.approx(0.0129, abs=2e-4)
    assert law is not None and rel < 1e-9          # exact fit still passes

    # ... and it BITES: a 2.9% holdout miss that the bare 0.10 would have
    # waved through is rejected, so the rungs get measured instead.
    off = dict(vals)
    off[holdout] = vals[holdout] * 1.03
    win_off = dict(win)
    win_off[holdout] = [off[holdout] * j for j in jitter]
    law_off, rel_off, tol_off = _cb_ladder_gate(
        kmap, anchors, off, holdout, 0.10, win_off)
    assert rel_off == pytest.approx(0.0291, abs=5e-4)
    assert rel_off < 0.10 and rel_off > tol_off
    assert law_off is None
