from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import prismaquant.aura_cost as aura
import prismaquant.production_weight_cache as pwc


class _TinyLM(nn.Module):
    def __init__(self, state=None) -> None:
        super().__init__()
        self.embed = nn.Embedding(23, 16)
        self.l1 = nn.Linear(16, 16, bias=False)
        self.l2 = nn.Linear(16, 16, bias=False)
        self.lm_head = nn.Linear(16, 23, bias=False)
        self.forward_calls = 0
        if state is not None:
            self.load_state_dict(state)

    def forward(self, input_ids):
        self.forward_calls += 1
        x = self.embed(input_ids)
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return SimpleNamespace(logits=self.lm_head(x))


class _TinyCache:
    def __init__(self, model: _TinyLM) -> None:
        self.metadata = {
            "calib_hash": "cache-calibration-content",
            "cb_cache_pair_identity": {
                "schema": "prismaquant.production_weight_cache.cb_pair_set.v1",
                "identity_sha256": "e" * 64,
                "artifact_sha256": "d" * 64,
                "entries": 2,
                "published_entries": 2,
                "calibration_hashes": ["cache-calibration-content"],
                "git_commits": ["9" * 40],
                "producer_source_sha256": ["8" * 64],
            },
        }
        self._weights = {
            (name, "FP8_CB_K28"): mod.weight.detach().clone() + 0.03125
            for name, mod in model.named_modules()
            if name in {"l1", "l2"}
        }

    def get(self, name, fmt):
        return self._weights.get((name, fmt))

    def compact_for_pickle(self):
        return 0


def _cb_provenance():
    return {
        "cb_cost_provenance_schema": "test.cb.provenance.v1",
        "cb_render_identity": {
            "schema": "test.cb.render_identity.v1",
            "col_weights_sha256": "f" * 64,
            "scale_coding": "two_tier",
            "scale_sweep_scope": "all",
            "ldlq_scope": "none",
            "layout_version": 2,
        },
    }


def _run(model, cache, checkpoint_dir, *, resume):
    return aura.compute_aura_cost(
        model,
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        ["FP8_CB_K28"],
        n_probes=2,
        n_linear_chunks=1,
        min_free_gib=0.0,
        production_cache=cache,
        require_production_cache=True,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        checkpoint_identity_extra={"gradient_checkpointing": False},
    )


def _assert_exact_cost_payload(actual, expected) -> None:
    """Exact per-unit comparison (including every probe sample and float)."""
    assert actual.keys() == expected.keys()
    assert actual["schema"] == expected["schema"]
    assert actual["n_probes"] == expected["n_probes"]
    assert actual["formats"] == expected["formats"]
    assert actual["token_scope"] == expected["token_scope"]
    assert actual["provenance"] == expected["provenance"]
    assert actual["stats"].keys() == expected["stats"].keys()
    assert actual["costs"].keys() == expected["costs"].keys()
    for qname in expected["stats"]:
        assert actual["stats"][qname] == expected["stats"][qname]
        assert actual["costs"][qname] == expected["costs"][qname]


def test_aura_interrupted_resume_is_pickle_identical(tmp_path, monkeypatch):
    torch.manual_seed(1234)
    base = _TinyLM()
    state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    monkeypatch.setattr(
        pwc,
        "production_cache_cb_render_provenance",
        lambda *_args, **_kwargs: _cb_provenance(),
    )

    uninterrupted_model = _TinyLM(state)
    uninterrupted = _run(
        uninterrupted_model,
        _TinyCache(uninterrupted_model),
        tmp_path / "uninterrupted",
        resume=False,
    )

    original_writer = aura._write_aura_unit_checkpoint
    writes = {"count": 0}

    def interrupt_after_one(*args, **kwargs):
        original_writer(*args, **kwargs)
        writes["count"] += 1
        if writes["count"] == 1:
            raise RuntimeError("synthetic explicit-PID interruption")

    monkeypatch.setattr(aura, "_write_aura_unit_checkpoint", interrupt_after_one)
    interrupted_model = _TinyLM(state)
    with pytest.raises(RuntimeError, match="synthetic explicit-PID interruption"):
        _run(
            interrupted_model,
            _TinyCache(interrupted_model),
            tmp_path / "interrupted",
            resume=False,
        )
    assert writes["count"] == 1

    monkeypatch.setattr(aura, "_write_aura_unit_checkpoint", original_writer)
    resumed_model = _TinyLM(state)
    resumed = _run(
        resumed_model,
        _TinyCache(resumed_model),
        tmp_path / "interrupted",
        resume=True,
    )

    _assert_exact_cost_payload(resumed, uninterrupted)


def test_aura_complete_resume_runs_no_forward(tmp_path, monkeypatch):
    torch.manual_seed(4321)
    first_model = _TinyLM()
    state = {
        name: value.detach().clone()
        for name, value in first_model.state_dict().items()
    }
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "2" * 40)
    monkeypatch.setattr(
        pwc,
        "production_cache_cb_render_provenance",
        lambda *_args, **_kwargs: _cb_provenance(),
    )
    expected = _run(
        first_model,
        _TinyCache(first_model),
        tmp_path / "checkpoints",
        resume=False,
    )

    resumed_model = _TinyLM(state)
    actual = _run(
        resumed_model,
        _TinyCache(resumed_model),
        tmp_path / "checkpoints",
        resume=True,
    )
    assert resumed_model.forward_calls == 0
    _assert_exact_cost_payload(actual, expected)


def test_aura_resume_names_mutated_col_weights_identity(
    tmp_path, monkeypatch
):
    torch.manual_seed(999)
    first_model = _TinyLM()
    state = {
        name: value.detach().clone()
        for name, value in first_model.state_dict().items()
    }
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "3" * 40)
    monkeypatch.setattr(
        pwc,
        "production_cache_cb_render_provenance",
        lambda *_args, **_kwargs: _cb_provenance(),
    )
    _run(
        first_model,
        _TinyCache(first_model),
        tmp_path / "checkpoints",
        resume=False,
    )

    manifest_path = tmp_path / "checkpoints" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["cb_render_identity"]["col_weights_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    resumed_model = _TinyLM(state)
    with pytest.raises(
        RuntimeError,
        match=r"cb_render_identity\.col_weights_sha256",
    ):
        _run(
            resumed_model,
            _TinyCache(resumed_model),
            tmp_path / "checkpoints",
            resume=True,
        )
    assert resumed_model.forward_calls == 0
