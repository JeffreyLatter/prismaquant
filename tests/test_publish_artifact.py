"""Publication is the blocking point (R16, ruled 2026-07-30).

Pipeline ship gates stay advisory — the build venv cannot import `vllm`, so
`run-pipeline.sh` cannot run one. What must not be possible is putting an
artifact on the Hub whose shipcard was never closed. These tests pin the three
behaviours that make `tools/publish_artifact.py` the gate: it refuses on an
unverified card *before* printing anything an operator could copy-paste, it
passes a closed card through, and the escape hatch is loud, confirmed by
re-typing, and leaves a mark on the artifact.
"""
from __future__ import annotations

import json
import pathlib

import pytest

if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

from prismaquant.shipcard import (
    GOLD_SLOTS,
    REQUIRED_SLOTS,
    build_shipcard,
    compute_model_sha,
    fill_slot,
    load_shipcard,
    make_record,
    write_shipcard,
)
from tools.publish_artifact import main as publish_cli
from tools.publish_artifact import upload_command


def _artifact(tmp_path, name="exported"):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "qwen3"}')
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    card = build_shipcard(model_dir, build={"achieved_bpp": {"value": 4.75}})
    write_shipcard(model_dir / "shipcard.json", card)
    return model_dir


def _close_all_slots(model_dir, *, passed=True, spec=False):
    path = model_dir / "shipcard.json"
    sha = compute_model_sha(model_dir)
    for slot in REQUIRED_SLOTS:
        fill_slot(path, slot, make_record(
            slot=slot, tool="test", passed=passed, model_sha=sha,
            spec_decode_detected=(spec if slot in GOLD_SLOTS else None),
        ))
    return path


def _argv(model_dir, *extra):
    return [str(model_dir), "--repo-id", "rdtand/test-artifact",
            "--dry-run", *extra]


def test_verified_card_publishes(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir)

    assert publish_cli(_argv(model_dir)) == 0
    out = capsys.readouterr().out
    assert "shipcard OK" in out
    assert "hf upload" in out and "rdtand/test-artifact" in out
    # A clean publish leaves no override mark.
    assert "forced_unverified" not in load_shipcard(model_dir / "shipcard.json")


def test_unfilled_card_refuses_and_prints_no_upload_command(tmp_path, capsys):
    model_dir = _artifact(tmp_path)  # fresh card: every slot empty

    assert publish_cli(_argv(model_dir)) == 1
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    for slot in REQUIRED_SLOTS:
        assert f"{slot}: UNFILLED" in captured.err
    # A refusal that still hands over the command is not a refusal.
    assert "hf upload" not in captured.out + captured.err


def test_failed_slot_refuses_naming_the_slot(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir, passed=False)

    assert publish_cli(_argv(model_dir)) == 1
    err = capsys.readouterr().err
    assert "ship_gate: FAILED" in err


def test_spec_decode_tainted_gold_refuses(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir, spec=True)

    assert publish_cli(_argv(model_dir)) == 1
    err = capsys.readouterr().err
    assert "spec_decode_detected is TRUE" in err


def test_missing_shipcard_refuses(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    (model_dir / "shipcard.json").unlink()

    assert publish_cli(_argv(model_dir)) == 1
    err = capsys.readouterr().err
    assert "no shipcard" in err
    assert "hf upload" not in err


def test_force_unverified_requires_the_basename_retyped(tmp_path, capsys):
    model_dir = _artifact(tmp_path, name="ornith-35b-exported")

    # Wrong name -> usage refusal, still no upload.
    assert publish_cli(_argv(model_dir, "--force-unverified",
                             "--confirm-name", "exported")) == 2
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    assert "hf upload" not in captured.out + captured.err
    assert "forced_unverified" not in load_shipcard(model_dir / "shipcard.json")


def test_force_unverified_stamps_the_card_and_proceeds(tmp_path, capsys):
    model_dir = _artifact(tmp_path, name="ornith-35b-exported")

    assert publish_cli(_argv(model_dir, "--force-unverified",
                             "--confirm-name", "ornith-35b-exported")) == 0
    out = capsys.readouterr().out
    assert "hf upload" in out

    card = load_shipcard(model_dir / "shipcard.json")
    assert card["forced_unverified"] is True
    history = card["forced_unverified_history"]
    assert len(history) == 1
    assert history[0]["repo_id"] == "rdtand/test-artifact"
    assert history[0]["unfilled_slots"] == list(REQUIRED_SLOTS)
    assert any("UNFILLED" in p for p in history[0]["problems"])
    assert history[0]["model_sha"] == compute_model_sha(model_dir)


def test_upload_command_is_the_exact_call_it_would_make(tmp_path):
    """The degraded path must print something runnable, not a paraphrase."""
    import argparse

    args = argparse.Namespace(
        artifact_dir=str(tmp_path), repo_id="rdtand/x", repo_type="model",
        path_in_repo=".", private=True, commit_message="ship it",
        allow_patterns=None, ignore_patterns=["*.log"],
    )
    cmd = upload_command(args)
    assert cmd.startswith("hf upload rdtand/x ")
    assert "--repo-type model" in cmd
    assert "--private" in cmd
    assert "--commit-message 'ship it'" in cmd
    assert "--exclude '*.log'" in cmd


def test_verification_is_a_library_call_not_a_subprocess():
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "tools" / "publish_artifact.py").read_text()
    assert "from prismaquant.shipcard import" in src
    assert "subprocess" not in src, (
        "shipcard verification must be a library call; a subprocess shells out "
        "to a python that may not have the package")
    # And the refusal must be reachable before any upload work.
    assert src.index("check_shipcard(artifact_dir") < src.index("return _upload(args)")


def test_json_round_trip_of_a_forced_card(tmp_path):
    """The stamp must survive being written and read as plain JSON."""
    model_dir = _artifact(tmp_path, name="forced")
    publish_cli(_argv(model_dir, "--force-unverified",
                      "--confirm-name", "forced"))
    raw = json.loads((model_dir / "shipcard.json").read_text())
    assert raw["forced_unverified"] is True
