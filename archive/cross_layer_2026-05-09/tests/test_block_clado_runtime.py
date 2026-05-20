from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant import decision_units as du
from prismaquant import format_registry as fr
from prismaquant.research_components import block_clado_runtime as bcr


def _unit(name, block, options):
    return du.DecisionUnit(
        name=name,
        block_id=block,
        member_qnames=(name,),
        options=tuple(
            du.FormatCost(
                fmt=fmt,
                omega_ii=cost,
                bits_per_param=bpp,
                memory_bytes=bytes_,
            )
            for fmt, cost, bpp, bytes_ in options
        ),
    )


def test_block_clado_solver_uses_pair_interactions_and_expands_members():
    units = (
        _unit(
            "model.layers.0.attn.qkv",
            "model.layers.0",
            (("NVFP4", 1.0, 4.0, 4), ("BF16", 0.0, 16.0, 16)),
        ),
        _unit(
            "model.layers.0.mlp.up_gate",
            "model.layers.0",
            (("NVFP4", 1.0, 4.0, 4), ("BF16", 0.0, 16.0, 16)),
        ),
    )
    pair = du.BlockPair(
        unit_a=units[0].name,
        unit_b=units[1].name,
        block_id="model.layers.0",
        omega_ij={("NVFP4", "NVFP4"): -1.5},
    )

    states = bcr.enumerate_block_states(units, (pair,))
    cheapest = min(states, key=lambda state: state.bits_total)
    best_cost = min(states, key=lambda state: state.cost)

    assert cheapest.assignment == {
        "model.layers.0.attn.qkv": "NVFP4",
        "model.layers.0.mlp.up_gate": "NVFP4",
    }
    assert cheapest.cost == pytest.approx(0.5)
    assert best_cost.cost == pytest.approx(0.0)


def test_block_clado_payload_sweep_budget_and_kneedle_round_trip(tmp_path):
    q = du.DecisionUnit(
        name="model.layers.0.qk",
        block_id="model.layers.0",
        member_qnames=("model.layers.0.q", "model.layers.0.k"),
        options=(
            du.FormatCost("NVFP4", 0.4, 4.0, 8),
            du.FormatCost("BF16", 0.0, 16.0, 32),
        ),
    )
    o = _unit(
        "model.layers.0.o",
        "model.layers.0",
        (("NVFP4", 0.3, 4.0, 4), ("BF16", 0.0, 16.0, 16)),
    )
    payload = bcr.units_and_pairs_to_payload(
        blocks={"model.layers.0": (q, o)},
        singletons=(),
        pairs_by_block={
            "model.layers.0": (
                du.BlockPair(
                    unit_a=q.name,
                    unit_b=o.name,
                    block_id="model.layers.0",
                    omega_ij={("NVFP4", "NVFP4"): -0.2},
                ),
            )
        },
        meta={"test": True},
    )
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload))

    block_states = bcr.build_block_states(du.load_payload(path))
    sweep = bcr.sweep_payload(payload, n_lambdas=5)
    budget = bcr.budget_payload(payload, target_bpp=8.0, bit_precision_bits=1.0)
    summary, candidates = bcr.kneedle_payloads(payload, sweep, n_neighbors=1)

    assert "model.layers.0" in block_states
    assert sweep["schema"] == bcr.SWEEP_SCHEMA
    assert budget["schema"] == bcr.BUDGET_SCHEMA
    assert summary["schema"] == bcr.KNEEDLE_SUMMARY_SCHEMA
    assert candidates
    expanded = candidates[0]["assignment"]
    assert set(expanded) >= {"model.layers.0.q", "model.layers.0.k"}


class _QkvProfile:
    name = "qkv-test"

    def fused_sibling_group(self, qname):
        if qname in {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        }:
            return "model.layers.0.self_attn.qkv_proj"
        return None


def _qkv_legacy_units(cost_nvfp4=0.01):
    return (
        _unit(
            "model.layers.0.self_attn.q_proj",
            "model.layers.0",
            (("NVFP4", cost_nvfp4, 4.0, 4), ("BF16", 0.0, 16.0, 16)),
        ),
        _unit(
            "model.layers.0.self_attn.k_proj",
            "model.layers.0",
            (("NVFP4", cost_nvfp4, 4.0, 4), ("BF16", 0.0, 16.0, 16)),
        ),
        _unit(
            "model.layers.0.self_attn.v_proj",
            "model.layers.0",
            (("NVFP4", cost_nvfp4, 4.0, 4), ("BF16", 0.0, 16.0, 16)),
        ),
    )


def _qkv_formats(assignment):
    return {
        assignment["model.layers.0.self_attn.q_proj"],
        assignment["model.layers.0.self_attn.k_proj"],
        assignment["model.layers.0.self_attn.v_proj"],
    }


def test_runtime_legalize_promotes_mixed_fused_assignment():
    legal = bcr.legalize_assignment_for_runtime(
        {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.self_attn.k_proj": "MXFP8_E4M3",
            "model.layers.0.self_attn.v_proj": "NVFP4",
            "model.layers.0.self_attn.o_proj": "NVFP4",
        },
        profile=_QkvProfile(),
        format_rank={"NVFP4": 0, "MXFP8_E4M3": 1, "BF16": 2},
    )

    assert legal["model.layers.0.self_attn.q_proj"] == "MXFP8_E4M3"
    assert legal["model.layers.0.self_attn.k_proj"] == "MXFP8_E4M3"
    assert legal["model.layers.0.self_attn.v_proj"] == "MXFP8_E4M3"
    assert legal["model.layers.0.self_attn.o_proj"] == "NVFP4"

    legal_with_missing_rank = bcr.legalize_assignment_for_runtime(
        {
            "model.layers.0.self_attn.q_proj": "BF16",
            "model.layers.0.self_attn.k_proj": "MXFP8_E4M3",
            "model.layers.0.self_attn.v_proj": "NVFP4",
        },
        profile=_QkvProfile(),
        format_rank={"NVFP4": 0, "BF16": 2},
    )
    assert set(legal_with_missing_rank.values()) == {"BF16"}


def test_block_state_enumeration_filters_mixed_fused_sibling_states():
    units = _qkv_legacy_units(cost_nvfp4=0.01)

    unrestricted = bcr.enumerate_block_states(units, (), max_states=None)
    assert any(
        len(_qkv_formats(state.assignment)) > 1
        for state in unrestricted
    )

    restricted = bcr.enumerate_block_states(
        units,
        (),
        max_states=None,
        profile=_QkvProfile(),
        format_rank={"NVFP4": 0, "BF16": 1},
    )

    assert restricted
    assert all(len(_qkv_formats(state.assignment)) == 1 for state in restricted)


def test_kneedle_candidates_from_profiled_sweep_keep_fused_siblings_coherent():
    units = _qkv_legacy_units(cost_nvfp4=0.01)
    payload = bcr.units_and_pairs_to_payload(
        blocks={"model.layers.0": units},
        singletons=(),
        pairs_by_block={"model.layers.0": ()},
    )

    sweep = bcr.sweep_payload(
        payload,
        n_lambdas=5,
        profile=_QkvProfile(),
    )
    _summary, candidates = bcr.kneedle_payloads(
        payload,
        sweep,
        n_neighbors=10,
        profile=_QkvProfile(),
    )

    assert candidates
    for candidate in candidates:
        assert len(_qkv_formats(candidate["assignment"])) == 1


def test_runtime_structure_coarsening_merges_legacy_fused_units():
    units = _qkv_legacy_units(cost_nvfp4=1.0)
    payload = bcr.units_and_pairs_to_payload(
        blocks={"model.layers.0": units},
        singletons=(),
        pairs_by_block={
            "model.layers.0": (
                du.BlockPair(
                    unit_a="model.layers.0.self_attn.q_proj",
                    unit_b="model.layers.0.self_attn.k_proj",
                    block_id="model.layers.0",
                    omega_ij={("NVFP4", "NVFP4"): -0.2},
                ),
                du.BlockPair(
                    unit_a="model.layers.0.self_attn.q_proj",
                    unit_b="model.layers.0.self_attn.v_proj",
                    block_id="model.layers.0",
                    omega_ij={("NVFP4", "NVFP4"): -0.3},
                ),
                du.BlockPair(
                    unit_a="model.layers.0.self_attn.k_proj",
                    unit_b="model.layers.0.self_attn.v_proj",
                    block_id="model.layers.0",
                    omega_ij={("NVFP4", "NVFP4"): -0.4},
                ),
            )
        },
    )

    coarsened = bcr.coarsen_payload_to_structure(
        payload,
        scope="runtime",
        profile=_QkvProfile(),
    )
    blocks, _singletons, pairs = du.parse_payload(coarsened)

    assert pairs["model.layers.0"] == []
    assert list(blocks) == ["model.layers.0"]
    assert len(blocks["model.layers.0"]) == 1
    qkv = blocks["model.layers.0"][0]
    assert qkv.name == "model.layers.0.self_attn.qkv_proj"
    assert set(qkv.member_qnames) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    }
    by_fmt = {opt.fmt: opt for opt in qkv.options}
    assert by_fmt["NVFP4"].omega_ii == pytest.approx(2.1)
    assert by_fmt["NVFP4"].memory_bytes == 12
    assert by_fmt["BF16"].omega_ii == pytest.approx(0.0)
    assert coarsened["meta"]["structure_coarsened"] is True


def test_subblock_structure_coarsening_respects_attention_boundaries():
    q, k, v = _qkv_legacy_units(cost_nvfp4=0.5)
    o = _unit(
        "model.layers.0.self_attn.o_proj",
        "model.layers.0",
        (("NVFP4", 0.25, 4.0, 4), ("BF16", 0.0, 16.0, 16)),
    )
    down = _unit(
        "model.layers.0.mlp.down_proj",
        "model.layers.0",
        (("NVFP4", 0.75, 4.0, 4), ("BF16", 0.0, 16.0, 16)),
    )
    payload = bcr.units_and_pairs_to_payload(
        blocks={"model.layers.0": (q, k, v, o, down)},
        singletons=(),
        pairs_by_block={"model.layers.0": ()},
    )

    coarsened = bcr.coarsen_payload_to_structure(
        payload,
        scope="subblock",
        profile=_QkvProfile(),
    )
    blocks, _singletons, _pairs = du.parse_payload(coarsened)
    by_name = {
        unit.name: unit
        for unit in blocks["model.layers.0"]
    }

    assert set(by_name) == {
        "model.layers.0.self_attn",
        "model.layers.0.mlp",
    }
    assert set(by_name["model.layers.0.self_attn"].member_qnames) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
    }
    assert by_name["model.layers.0.mlp"].member_qnames == (
        "model.layers.0.mlp.down_proj",
    )


def test_structured_kneedle_rejects_flat_sweep_mismatch():
    units = _qkv_legacy_units(cost_nvfp4=0.01)
    payload = bcr.units_and_pairs_to_payload(
        blocks={"model.layers.0": units},
        singletons=(),
        pairs_by_block={"model.layers.0": ()},
    )
    flat_sweep = bcr.sweep_payload(payload, n_lambdas=5)

    with pytest.raises(ValueError, match="Rerun sweep with the same structure"):
        bcr.kneedle_payloads(
            payload,
            flat_sweep,
            n_neighbors=1,
            profile=_QkvProfile(),
            structure_scope="runtime",
        )


def test_structured_sweep_and_kneedle_expand_coarsened_units():
    units = _qkv_legacy_units(cost_nvfp4=0.01)
    payload = bcr.units_and_pairs_to_payload(
        blocks={"model.layers.0": units},
        singletons=(),
        pairs_by_block={"model.layers.0": ()},
    )

    sweep = bcr.sweep_payload(
        payload,
        n_lambdas=5,
        profile=_QkvProfile(),
        structure_scope="runtime",
    )
    assert sweep["meta"]["structure_scope"] == "runtime"
    assert {
        unit
        for row in sweep["rows"]
        for unit in row["assignment"]
    } == {"model.layers.0.self_attn.qkv_proj"}

    _summary, candidates = bcr.kneedle_payloads(
        payload,
        sweep,
        n_neighbors=1,
        profile=_QkvProfile(),
        structure_scope="runtime",
    )

    assert candidates
    for candidate in candidates:
        assert set(candidate["assignment"]) == {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        }
        assert len(_qkv_formats(candidate["assignment"])) == 1


def test_expand_unit_assignment_omits_lm_head_by_default():
    unit = _unit(
        "lm_head",
        "singletons",
        (("NVFP4", 1.0, 4.0, 4), ("BF16", 0.0, 16.0, 16)),
    )

    expanded = bcr.expand_unit_assignment({"lm_head": "NVFP4"}, [unit])

    assert expanded == {}


def test_expand_unit_assignment_can_keep_pinned_lm_head_as_bf16():
    unit = _unit(
        "lm_head",
        "singletons",
        (("NVFP4", 1.0, 4.0, 4), ("BF16", 0.0, 16.0, 16)),
    )

    expanded = bcr.expand_unit_assignment(
        {"lm_head": "NVFP4"},
        [unit],
        omit_bf16_pinned=False,
    )

    assert expanded == {"lm_head": "BF16"}


def test_collect_block_clado_uses_live_decision_units_and_kl_path(monkeypatch, tmp_path):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(32, 32, bias=False)
            self.k = nn.Linear(32, 32, bias=False)
            self.o = nn.Linear(32, 32, bias=False)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, input_ids, use_cache=False):
            batch, seqlen = input_ids.shape
            logits = torch.zeros((batch, seqlen, 8), dtype=torch.float32)
            return SimpleNamespace(logits=logits)

    class Profile:
        def fused_sibling_group(self, qname):
            if qname in {"model.layers.0.q", "model.layers.0.k"}:
                return "model.layers.0.qk"
            return None

    measured = []

    def fake_cache_reference_log_probs(model, calib_ids, device, *, kl_scope):
        assert kl_scope == "last_token"
        return ["teacher"]

    def fake_measure_assignment_kl(
        model,
        assignment,
        calib_ids,
        ref_log_probs,
        **kwargs,
    ):
        measured.append(dict(assignment))
        return 0.1 * sum(1 for fmt in assignment.values() if fmt != "BF16")

    monkeypatch.setattr(bcr, "cache_reference_log_probs", fake_cache_reference_log_probs)
    monkeypatch.setattr(bcr, "measure_assignment_kl", fake_measure_assignment_kl)

    payload = bcr.collect_block_clado(
        Tiny(),
        torch.tensor([[0, 1]], dtype=torch.long),
        [fr.get_format("NVFP4"), fr.get_format("BF16")],
        profile=Profile(),
        work_root=tmp_path,
    )

    blocks, _singletons, pairs_by_block = du.parse_payload(payload)
    assert payload["schema"] == du.SCHEMA
    assert payload["meta"]["n_unary_measurements"] == 2
    assert payload["meta"]["n_pair_measurements"] == 1
    fused_unit = next(
        unit for unit in blocks["model.layers.0"]
        if "model.layers.0.q" in unit.member_qnames
    )
    assert set(fused_unit.member_qnames) == {
        "model.layers.0.k",
        "model.layers.0.q",
    }
    assert len(pairs_by_block["model.layers.0"]) == 1
    assert measured


def test_measure_cli_exposes_dataset_and_residency_controls():
    parser = argparse.ArgumentParser()
    bcr._add_common_measure_args(parser)

    args = parser.parse_args([
        "--model",
        "m",
        "--output",
        "out.json",
        "--dataset",
        "calib.jsonl",
        "--production-cache-lru-gb",
        "64",
        "--production-cache-prefetch-workers",
        "8",
        "--source-prefetch",
        "require",
    ])

    assert args.dataset == "calib.jsonl"
    assert args.production_cache_lru_gb == pytest.approx(64.0)
    assert args.production_cache_prefetch_workers == 8
    assert args.source_prefetch == "require"


def test_production_cache_prefetch_uses_configured_workers():
    calls = []

    class Cache:
        def prefetch_assignment(self, assignment, **kwargs):
            calls.append((assignment, kwargs))
            return {"loaded": 1}

    bcr._prefetch_assignment_if_available(
        Cache(),
        {"a": "NVFP4"},
        require=True,
        max_workers=7,
    )

    assert calls == [
        (
            {"a": "NVFP4"},
            {"require": True, "max_workers": 7, "progress": False},
        )
    ]
