"""Pinning test for the pre-fix packed-Fisher probe guard."""
import os
import pickle


import pytest

from prismaquant.measure_quant_cost import prepare_cost_context


def _write_probe(tmp_path, meta, stats):
    p = tmp_path / "probe.pkl"
    with open(p, "wb") as f:
        pickle.dump({"stats": stats, "meta": meta}, f)
    return str(p)


PACKED = {"model.layers.0.mlp.experts.gate_up_proj": {
    "h_trace": 1.0, "_packed_experts_module": "model.layers.0.mlp.experts"}}
DENSE = {"model.layers.0.q_proj": {"h_trace": 1.0}}


def test_stale_packed_probe_refused(tmp_path):
    p = _write_probe(tmp_path, {}, PACKED)
    (tmp_path / "act").mkdir()
    with pytest.raises(SystemExit, match="sum-then-square"):
        prepare_cost_context(p, str(tmp_path / "act"), "NVFP4", True)


def test_stamped_packed_probe_accepted(tmp_path):
    p = _write_probe(tmp_path, {"packed_fisher_estimator": "per_token_v2"}, PACKED)
    (tmp_path / "act").mkdir()
    prepare_cost_context(p, str(tmp_path / "act"), "NVFP4", True)


def test_dense_probe_unaffected(tmp_path):
    p = _write_probe(tmp_path, {}, DENSE)
    (tmp_path / "act").mkdir()
    prepare_cost_context(p, str(tmp_path / "act"), "NVFP4", True)


def test_escape_env_accepts_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER", "1")
    p = _write_probe(tmp_path, {}, PACKED)
    (tmp_path / "act").mkdir()
    prepare_cost_context(p, str(tmp_path / "act"), "NVFP4", True)
