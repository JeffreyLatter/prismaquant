from __future__ import annotations

import json

import pytest

import prismaquant.shipcard as shipcard


def _git_unavailable(*_args, **_kwargs):
    raise FileNotFoundError("fixture has no git metadata")


def test_cb_shipcard_uses_explicit_clean_git_identity_without_worktree(
    tmp_path, monkeypatch,
):
    model = tmp_path / "artifact"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture-weight")
    layer_config = tmp_path / "layer_config.json"
    layer_config.write_text(json.dumps({
        "__prismaquant__": {"achieved_bits": 4.5},
    }))
    monkeypatch.setattr(shipcard.subprocess, "run", _git_unavailable)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_DIRTY", "0")

    path, _ = shipcard.open_cb_export_shipcard(
        model,
        {"quant_method": "gridbook", "format": "nvfp4_cb"},
        source_model=tmp_path / "source",
        layer_config_path=layer_config,
        exporter="fixture-exporter",
    )
    assert shipcard.load_shipcard(path)["build"]["git"] == {
        "commit": "a" * 40,
        "dirty": False,
    }


def test_git_override_without_clean_preflight_does_not_invent_dirty_false(
    monkeypatch,
):
    monkeypatch.setattr(shipcard.subprocess, "run", _git_unavailable)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "b" * 40)
    monkeypatch.delenv("PRISMAQUANT_IDENTITY_GIT_DIRTY", raising=False)
    assert shipcard.git_provenance() == {
        "commit": "b" * 40,
        "dirty": None,
    }


def test_git_overrides_refuse_contradictory_mounted_worktree(
    monkeypatch,
):
    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(command, **_kwargs):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return _Result("c" * 40 + "\n")
        if command[1:3] == ["status", "--short"]:
            return _Result(" M prismaquant/shipcard.py\n")
        raise AssertionError(command)

    monkeypatch.setattr(shipcard.subprocess, "run", fake_run)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "c" * 40)
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_DIRTY", "0")
    with pytest.raises(ValueError, match="contradicts.*dirty=True"):
        shipcard.git_provenance()

