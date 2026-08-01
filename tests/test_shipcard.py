"""The ship record's refusal contract (R13).

The point of `shipcard.json` is that it says NO by default: an artifact whose
serve-lane slots were never closed must not read as shippable. These tests pin
the four ways it says no — unfilled, wrong build, failed check, spec-decode
tainted gold number — plus the fill path the serve-lane tools use.
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
    kv_shared_fisher_echo,
    load_shipcard,
    make_record,
    unfilled_slots,
    verify,
    write_shipcard,
)
from tools.shipcard import main as shipcard_cli


def _artifact(tmp_path, *, name="exported", weight_bytes=b"weights"):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "qwen3"}')
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(weight_bytes)
    return model_dir


def _open_card(tmp_path, model_dir):
    card = build_shipcard(model_dir, build={"achieved_bpp": {"value": 4.75}})
    path = model_dir / "shipcard.json"
    write_shipcard(path, card)
    return path


def _fill_all(path, model_sha, *, spec=False, passed=True):
    for slot in REQUIRED_SLOTS:
        fill_slot(path, slot, make_record(
            slot=slot, tool="test", passed=passed, model_sha=model_sha,
            spec_decode_detected=(spec if slot in GOLD_SLOTS else None),
        ))


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def test_model_sha_is_stable_and_content_sensitive(tmp_path):
    a = _artifact(tmp_path, name="a")
    b = _artifact(tmp_path, name="b")
    assert compute_model_sha(a) == compute_model_sha(a)
    assert compute_model_sha(a) == compute_model_sha(b), (
        "identical bytes and layout must hash identically — a copied artifact "
        "keeps its identity")

    (b / "config.json").write_text('{"model_type": "qwen3", "x": 1}')
    assert compute_model_sha(a) != compute_model_sha(b)

    c = _artifact(tmp_path, name="c", weight_bytes=b"weights-but-longer")
    assert compute_model_sha(a) != compute_model_sha(c)


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------
def test_fresh_card_refuses_every_slot(tmp_path):
    model_dir = _artifact(tmp_path)
    card = build_shipcard(model_dir, build={})
    assert unfilled_slots(card) == list(REQUIRED_SLOTS)
    problems = verify(card, model_dir=model_dir)
    assert len(problems) == len(REQUIRED_SLOTS)
    assert all("UNFILLED" in p for p in problems)


def test_full_card_verifies(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))
    assert verify(load_shipcard(path), model_dir=model_dir) == []


def test_record_from_another_build_is_refused(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, "deadbeef" * 8)
    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert problems and all("another build" in p for p in problems)


def test_artifact_edited_after_the_card_was_opened_is_refused(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"re-exported!")

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any("artifact changed since the shipcard was opened" in p
               for p in problems)


def test_failed_record_is_refused(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))
    fill_slot(path, "ship_gate", make_record(
        slot="ship_gate", tool="test", passed=False,
        model_sha=compute_model_sha(model_dir), detail="p99 NLL 9.4 > 6.0"))

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert problems == ["ship_gate: FAILED — p99 NLL 9.4 > 6.0"]


@pytest.mark.parametrize("spec, expected", [
    (True, "is TRUE"),
    (None, "is unknown"),
])
def test_gold_slots_refuse_spec_decode_states(tmp_path, spec, expected):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))
    fill_slot(path, "gold.kl", make_record(
        slot="gold.kl", tool="test", passed=True,
        model_sha=compute_model_sha(model_dir), spec_decode_detected=spec))

    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert len(problems) == 1
    assert problems[0].startswith("gold.kl: spec_decode_detected")
    assert expected in problems[0]


def test_unknown_slot_is_rejected(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    with pytest.raises(KeyError):
        make_record(slot="gold.mmlu", tool="test", passed=True, model_sha="x")
    with pytest.raises(KeyError):
        fill_slot(path, "gold.mmlu", {"passed": True})


# ---------------------------------------------------------------------------
# Build-lane facts
# ---------------------------------------------------------------------------
def test_kv_shared_fisher_echo_flags_an_unvalidated_allocation():
    clean = kv_shared_fisher_echo({})
    assert clean["unvalidated_kv_fisher_correction"] is False

    overridden = kv_shared_fisher_echo(
        {"PRISMAQUANT_ALLOW_KV_SHARED_FISHER": "1"})
    assert overridden["unvalidated_kv_fisher_correction"] is True

    severed = kv_shared_fisher_echo({"PRISMAQUANT_KV_COTANGENT": "0"})
    assert severed["kv_cotangent_path_enabled"] is False
    assert severed["unvalidated_kv_fisher_correction"] is True


def test_export_writes_a_card_with_build_facts_and_empty_slots(tmp_path):
    """The exporter's `_write_shipcard`, without importing torch's world."""
    from prismaquant.export_native_compressed import _write_shipcard

    model_dir = _artifact(tmp_path)
    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({"model.layers.0.mlp.up_proj": {"bits": 4}}))
    (tmp_path / "pareto.knees.json").write_text(json.dumps({
        "primary": "log_error",
        "log_error": {"achieved_bits": 4.7513, "target_bits": 4.75},
    }))

    _write_shipcard(
        model_dir,
        source_model="/models/Qwen3-4B",
        layer_config_path=str(recipe),
        assignment={"model.layers.0.mlp.up_proj": "NVFP4"},
        config_assignment={"model.layers.0.mlp.up_proj": "NVFP4"},
        hist={("NVFP4", "packed"): 1},
    )

    card = load_shipcard(model_dir / "shipcard.json")
    assert unfilled_slots(card) == list(REQUIRED_SLOTS)
    build = card["build"]
    assert build["achieved_bpp"]["value"] == pytest.approx(4.7513)
    assert build["achieved_bpp"]["source"] == "pareto.knees.json:log_error"
    assert build["layer_config_sha"] and build["assignment_hash"]
    assert build["format_histogram"] == {"NVFP4/packed": 1}
    assert "PRISMAQUANT_GPTQ_DAMP" in build["render_levers"]
    assert "unvalidated_kv_fisher_correction" in build["kv_shared_fisher"]
    assert card["artifact_bytes"] == len(b"weights")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_verify_exit_codes(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)

    assert shipcard_cli(["verify", str(path), "--model-dir", str(model_dir)]) == 1
    assert "REFUSED" in capsys.readouterr().out

    _fill_all(path, compute_model_sha(model_dir))
    assert shipcard_cli(["verify", str(path), "--model-dir", str(model_dir)]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_fill_from_a_gold_result_json(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    result = tmp_path / "kl_student.json"
    result.write_text(json.dumps({
        "model": str(model_dir),
        "kl_confident_mean": 0.0143,
        "kl_mean": 0.0201,
        "n_samples": 8,
        "seqlen": 512,
        "spec_decode_detected": False,
        "serve_fingerprint": "f" * 64,
        "git_commit": "abc123",
    }))

    assert shipcard_cli([
        "fill", str(path), "--slot", "gold.kl", "--record", str(result)]) == 0

    record = load_shipcard(path)["slots"]["gold.kl"]
    assert record["passed"] is True
    assert record["model_sha"] == compute_model_sha(model_dir)
    assert record["metrics"]["kl_confident_mean"] == pytest.approx(0.0143)
    assert record["serve_fingerprint"] == "f" * 64
    assert record["git_commit"] == "abc123"
    assert "gold.ppl" in capsys.readouterr().out  # still-unfilled list


def test_cli_fill_refuses_a_spec_decode_tainted_gold_record(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    result = tmp_path / "ppl.json"
    result.write_text(json.dumps({
        "model": str(model_dir), "ppl": 4.12, "spec_decode_detected": True}))

    assert shipcard_cli([
        "fill", str(path), "--slot", "gold.ppl", "--record", str(result)]) == 2
    assert "DRAFT model" in capsys.readouterr().err
    assert load_shipcard(path)["slots"]["gold.ppl"] is None

    # ...and an unknown detection is refused for the same reason.
    result.write_text(json.dumps({"model": str(model_dir), "ppl": 4.12}))
    assert shipcard_cli([
        "fill", str(path), "--slot", "gold.ppl", "--record", str(result)]) == 2

    # --allow-spec-decode records it, and verify still refuses.
    result.write_text(json.dumps({
        "model": str(model_dir), "ppl": 4.12, "spec_decode_detected": True}))
    assert shipcard_cli([
        "fill", str(path), "--slot", "gold.ppl", "--record", str(result),
        "--allow-spec-decode"]) == 0
    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any("gold.ppl: spec_decode_detected is TRUE" in p for p in problems)


def test_cli_show_lists_unfilled_slots(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    assert shipcard_cli(["show", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.count("UNFILLED") == len(REQUIRED_SLOTS)
