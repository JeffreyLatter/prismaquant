"""The ship record's refusal contract (R13).

The point of `shipcard.json` is that it says NO by default: an artifact whose
serve-lane slots were never closed must not read as shippable. These tests pin
the four ways it says no — unfilled, wrong build, failed check, spec-decode
tainted gold number — plus the fill path the serve-lane tools use.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

from prismaquant.shipcard import (
    CB_REQUIRED_SLOTS,
    GOLD_SLOTS,
    REQUIRED_SLOTS,
    SHIPCARD_RESERVED_BYTES,
    artifact_bytes,
    build_shipcard,
    compute_model_sha,
    fill_slot,
    kv_shared_fisher_echo,
    load_shipcard,
    make_record,
    open_cb_export_shipcard,
    required_slots,
    unfilled_slots,
    verify,
    write_shipcard,
)
from prismaquant.shipcard_cli import main as shipcard_cli


def _artifact(
    tmp_path, *, name="exported", weight_bytes=b"weights", model_type="qwen3",
):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": model_type}))
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(weight_bytes)
    return model_dir


def _open_card(tmp_path, model_dir):
    card = build_shipcard(model_dir, build={"achieved_bpp": {"value": 4.75}})
    path = model_dir / "shipcard.json"
    write_shipcard(path, card)
    return path


_FAKE_FINGERPRINT = "f" * 64
_FAKE_COMMIT = "a" * 40

#: What a real gold record carries. These helpers used to fill the gold slots
#: with an EMPTY metrics dict and the card verified anyway, because
#: `_verify_gold_record` only ran on the Gridbook CB lane — so the tests were
#: encoding the hole rather than catching it. Every generic gold requirement
#: (finite metric, serve fingerprint, producer commit, position count,
#: score_positions=all) now applies on every lane, and the fixtures have to
#: look like real measurements.
_GOLD_METRICS = {
    "gold.kl": {
        "kl_mean": 0.0151,
        "kl_confident_mean": 0.0143,
        "n_positions": 4088,
        "n_samples": 8,
        "seqlen": 512,
        "score_positions": "all",
    },
    "gold.ppl": {
        "ppl": 8.33,
        "mean_nll": 2.12,
        "n_tokens_scored": 8192,
    },
}


def _fill_all(path, model_sha, *, spec=False, passed=True):
    for slot in REQUIRED_SLOTS:
        is_gold = slot in GOLD_SLOTS
        fill_slot(path, slot, make_record(
            slot=slot, tool="test", passed=passed, model_sha=model_sha,
            spec_decode_detected=(spec if is_gold else None),
            metrics=(_GOLD_METRICS.get(slot) if is_gold else None),
            serve_fingerprint=(_FAKE_FINGERPRINT if is_gold else None),
            git_commit=(_FAKE_COMMIT if is_gold else None),
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


def test_native_card_remains_verifiable_after_legitimate_copy(tmp_path):
    import shutil

    source = _artifact(tmp_path, name="native-source")
    source_card = _open_card(tmp_path, source)
    assert "weight_stat_attestation" not in load_shipcard(source_card)

    copied = tmp_path / "native-copy"
    shutil.copytree(source, copied)
    problems = verify(load_shipcard(copied / "shipcard.json"), model_dir=copied)
    assert all("artifact changed" not in problem for problem in problems)


def test_cb_identity_binds_canonical_config_and_codebook_not_inventory(
    tmp_path,
):
    model_dir = _artifact(tmp_path)
    codebook = model_dir / "cb_codebooks.pqcb"
    codebook.write_bytes(b"codebook-A")
    quant_config = {
        "config_groups": {
            "cb": {
                "scheme": {"grid": "fp4", "k": 16},
                "targets": ["model.layers.0.mlp.up_proj"],
            }
        },
        "provenance": {
            "producer": "resident",
            "artifact_inventory": {"export_directory_bytes": 123},
        },
    }
    quant_path = model_dir / "quant_config.json"
    quant_path.write_text(json.dumps(quant_config, indent=2))
    baseline = compute_model_sha(model_dir)

    # Formatting and the self-sized final inventory are not model semantics.
    quant_config["provenance"]["artifact_inventory"] = {
        "export_directory_bytes": 987654,
        "file_bytes": {"quant_config.json": 456},
    }
    quant_path.write_text(json.dumps(
        quant_config, sort_keys=True, separators=(",", ":")
    ))
    assert compute_model_sha(model_dir) == baseline

    # Every other quant-config field remains identity-bearing.
    quant_config["config_groups"]["cb"]["scheme"]["k"] = 17
    quant_path.write_text(json.dumps(quant_config))
    assert compute_model_sha(model_dir) != baseline

    # Restore the config, then change same-length codebook bytes. Content, not
    # just the sidecar's size, must distinguish the served model.
    quant_config["config_groups"]["cb"]["scheme"]["k"] = 16
    quant_path.write_text(json.dumps(quant_config))
    codebook.write_bytes(b"codebook-B")
    assert compute_model_sha(model_dir) != baseline
    assert artifact_bytes(model_dir) == (
        (model_dir / "model-00001-of-00001.safetensors").stat().st_size
        + codebook.stat().st_size
    )


def test_cb_identity_binds_same_size_weight_content_via_export_manifest(tmp_path):
    from prismaquant.shipcard import build_weight_content_manifest

    model_dir = _artifact(tmp_path, weight_bytes=b"weights-A")
    quant_config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {
            "weight_content_manifest": build_weight_content_manifest(model_dir),
        },
    }
    (model_dir / "quant_config.json").write_text(json.dumps(quant_config))
    baseline = compute_model_sha(model_dir)

    # The immutable content claim is part of model_sha even though routine
    # verification need not reread the large shard. A changed same-size shard
    # trips the card's cheap stat attestation immediately.
    path = _open_card(tmp_path, model_dir)
    weight = model_dir / "model-00001-of-00001.safetensors"
    weight.write_bytes(b"weights-B")
    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any("artifact changed since the shipcard was opened" in p for p in problems)
    assert compute_model_sha(model_dir) == baseline


def test_cb_identity_binds_auxiliary_serving_files(tmp_path):
    from prismaquant.shipcard import build_weight_content_manifest

    model_dir = _artifact(tmp_path)
    tokenizer = model_dir / "tokenizer.json"
    tokenizer.write_bytes(b"tokenizer-A")
    quant_config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {
            "weight_content_manifest": build_weight_content_manifest(model_dir),
        },
    }
    (model_dir / "quant_config.json").write_text(json.dumps(quant_config))
    baseline = compute_model_sha(model_dir)
    tokenizer.write_bytes(b"tokenizer-B")
    assert compute_model_sha(model_dir) != baseline


def test_reattest_accepts_copy_but_refuses_changed_weight_content(tmp_path):
    import shutil

    from prismaquant.shipcard import (
        build_weight_content_manifest,
        reattest_weight_stats,
    )

    source = _artifact(tmp_path, name="source", weight_bytes=b"weights-A")
    quant_config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {
            "weight_content_manifest": build_weight_content_manifest(source),
        },
    }
    (source / "quant_config.json").write_text(json.dumps(quant_config))
    source_card = _open_card(tmp_path, source)

    copied = tmp_path / "copied"
    shutil.copytree(source, copied)
    copied_card = copied / source_card.name
    assert verify(load_shipcard(copied_card), model_dir=copied)
    reattest_weight_stats(copied_card, copied)
    assert not any(
        "artifact changed since the shipcard was opened" in problem
        for problem in verify(load_shipcard(copied_card), model_dir=copied)
    )

    (copied / "model-00001-of-00001.safetensors").write_bytes(b"weights-B")
    with pytest.raises(ValueError, match="content differs"):
        reattest_weight_stats(copied_card, copied)


def test_cb_shipcard_accepts_in_stream_digest_without_rereading_weight(
    tmp_path, monkeypatch,
):
    import prismaquant.shipcard as shipcard_module

    model_dir = _artifact(tmp_path, weight_bytes=b"streamed-weights")
    weight = model_dir / "model-00001-of-00001.safetensors"
    layer_config = tmp_path / "layer_config.json"
    layer_config.write_text("{}")

    def refuse_second_read(_model_dir):
        raise AssertionError("finished weight was read a second time")

    monkeypatch.setattr(
        shipcard_module, "build_weight_content_manifest", refuse_second_read
    )
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    path, _card = open_cb_export_shipcard(
        model_dir,
        {"quant_method": "gridbook", "format": "nvfp4_cb"},
        source_model=tmp_path / "source",
        layer_config_path=layer_config,
        exporter="test_streaming_exporter",
        weight_content_manifest={
            "schema": "prismaquant.weight_content_manifest/1",
            "algorithm": "sha256",
            "files": {
                weight.name: {
                    "bytes": weight.stat().st_size,
                    "sha256": digest,
                },
            },
        },
    )

    quant_config = json.loads((model_dir / "quant_config.json").read_text())
    assert quant_config["provenance"]["weight_content_manifest"]["files"] == {
        weight.name: {"bytes": weight.stat().st_size, "sha256": digest},
    }
    assert load_shipcard(path)["weight_stat_attestation"]["files"][weight.name]


def test_cb_shipcard_rejects_in_stream_digest_for_the_wrong_file_set(tmp_path):
    model_dir = _artifact(tmp_path)
    layer_config = tmp_path / "layer_config.json"
    layer_config.write_text("{}")

    with pytest.raises(ValueError, match="manifest file set differs"):
        open_cb_export_shipcard(
            model_dir,
            {"quant_method": "gridbook", "format": "nvfp4_cb"},
            source_model=tmp_path / "source",
            layer_config_path=layer_config,
            exporter="test_streaming_exporter",
            weight_content_manifest={
                "schema": "prismaquant.weight_content_manifest/1",
                "algorithm": "sha256",
                "files": {
                    "wrong.safetensors": {
                        "bytes": 7,
                        "sha256": "0" * 64,
                    },
                },
            },
        )


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


def test_gridbook_card_opens_plugin_performance_refusal_slot(tmp_path):
    model_dir = _artifact(tmp_path, model_type="deepseek_v4")
    (model_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
    }))
    card = build_shipcard(model_dir, build={"quant_method": "gridbook"})

    expected = REQUIRED_SLOTS + CB_REQUIRED_SLOTS
    assert tuple(card["slots"]) == expected
    assert required_slots(card, model_dir=model_dir) == expected
    assert unfilled_slots(card) == list(expected)
    problems = verify(card, model_dir=model_dir)
    assert any(
        problem == f"{CB_REQUIRED_SLOTS[0]}: UNFILLED"
        for problem in problems
    )


def test_a_cb_artifact_that_displaces_nothing_is_not_held_to_parity(tmp_path):
    """`perf.matched_budget_parity` is a DSv4 *release argument*.

    Its verifier requires five `displaced_container_*` digests naming the exact
    eligible container the release replaces at the same byte budget. A net-new
    size class -- the Qwen3.8 5080 CB artifact -- displaces nothing, so it can
    never produce them: a gate no correct artifact can pass, the same defect as
    the DSv4 gold contract. It is not opened, not required, and not silently
    marked satisfied.
    """
    model_dir = _artifact(tmp_path, model_type="qwen3_5_text")
    (model_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
    }))
    card = build_shipcard(model_dir, build={"quant_method": "gridbook"})

    assert CB_REQUIRED_SLOTS[0] not in card["slots"]
    assert required_slots(card, model_dir=model_dir) == REQUIRED_SLOTS
    assert not any(
        problem.startswith(CB_REQUIRED_SLOTS[0])
        for problem in verify(card, model_dir=model_dir)
    )


def test_a_parity_claim_that_is_made_off_lane_is_still_verified(tmp_path):
    """Scoping the DEMAND must not create a hole in the CHECK: a card that
    volunteers the slot is held to it wherever it lives."""
    model_dir = _artifact(tmp_path, model_type="qwen3_5_text")
    (model_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
    }))
    card = build_shipcard(model_dir, build={"quant_method": "gridbook"})
    card["slots"][CB_REQUIRED_SLOTS[0]] = {
        "slot": CB_REQUIRED_SLOTS[0], "passed": True, "tool": "handmade",
    }

    assert CB_REQUIRED_SLOTS[0] in required_slots(card, model_dir=model_dir)
    assert any(
        problem.startswith(CB_REQUIRED_SLOTS[0])
        for problem in verify(card, model_dir=model_dir)
    ), "an off-lane parity claim must still be replayed, not waved through"


def _cb_card(tmp_path, *, model_type, architectures, name="exported"):
    """A Gridbook CB artifact whose architecture is the one thing that varies."""
    model_dir = _artifact(tmp_path, name=name)
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": model_type, "architectures": architectures,
    }))
    (model_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook", "format": "nvfp4_cb",
    }))
    path = model_dir / "shipcard.json"
    write_shipcard(
        path, build_shipcard(model_dir, build={"quant_method": "gridbook"}))
    return model_dir, path


def _gold_problems(model_dir, path, slot):
    _fill_all(path, compute_model_sha(model_dir))
    return [p for p in verify(load_shipcard(path), model_dir=model_dir)
            if p.startswith(f"{slot}:")]


@pytest.mark.parametrize("slot", sorted(GOLD_SLOTS))
def test_the_dsv4_gold_contract_is_required_on_the_dsv4_lane(tmp_path, slot):
    """A DSv4-Flash CB release still replays its own release contract."""
    model_dir, path = _cb_card(
        tmp_path, model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"])
    assert any("DSv4 Gridbook gold contract" in p
               for p in _gold_problems(model_dir, path, slot))


@pytest.mark.parametrize("slot", sorted(GOLD_SLOTS))
def test_a_cb_artifact_off_the_dsv4_lane_is_not_held_to_dsv4s_contract(
    tmp_path, slot,
):
    """The contract pins ONE lane; `is_gridbook_cb` was only ever its proxy.

    `_verify_dsv4_gridbook_gold_contract` requires `tokenizer_mode
    ="deepseek_v4"` and `max_logprobs=248_320` — the DSv4 vocabulary — so a
    Qwen CB artifact could not fill `gold.kl`/`gold.ppl` at any effort. A gate
    no correct artifact can pass is a measurement gap, not a missing
    measurement (principle 1). Caught 2026-08-15 publishing Qwen3.8-27B CB,
    the first CB artifact off the DSv4 lane.

    What stays: every generic gold requirement, which is what the rest of this
    file pins. This test only asserts that the DSv4-specific one is gone.
    """
    model_dir, path = _cb_card(
        tmp_path, model_type="qwen3_5_text",
        architectures=["Qwen3_5ForCausalLM"])
    assert _gold_problems(model_dir, path, slot) == []


@pytest.mark.parametrize("config_text", [
    None,                                     # no config.json at all
    "{not json",                              # unreadable
    "[]",                                     # not an object
    '{"architectures": []}',                  # says nothing about the arch
])
def test_an_unreadable_architecture_keeps_the_strict_contract(
    tmp_path, config_text,
):
    """Fail-closed: a DSv4 release cannot shed its contract by hiding config.

    `config.json` is bound into `model_sha` as `config_sha`, so corrupting it
    already breaks the card — but the lane test must not be the weak link that
    turns a corrupt config into a LOOSER gate.
    """
    model_dir, path = _cb_card(
        tmp_path, model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"])
    if config_text is None:
        (model_dir / "config.json").unlink()
    else:
        (model_dir / "config.json").write_text(config_text)
    assert any("DSv4 Gridbook gold contract" in p
               for p in _gold_problems(model_dir, path, "gold.kl"))


def test_the_lane_is_read_off_disk_not_off_the_mutable_card(tmp_path):
    """Same reasoning as `_is_gridbook_card`: the receipt is mutated as gates
    close, so the obligation is resolved from the artifact."""
    from prismaquant.shipcard import _is_dsv4_gridbook_artifact

    model_dir, _ = _cb_card(
        tmp_path, model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"])
    assert _is_dsv4_gridbook_artifact(model_dir) is True
    assert _is_dsv4_gridbook_artifact(None) is True

    # The architecture list alone is enough, in both directions.
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["DeepseekV4ForCausalLM"]}))
    assert _is_dsv4_gridbook_artifact(model_dir) is True
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}))
    assert _is_dsv4_gridbook_artifact(model_dir) is False


def test_target_only_dspark_claim_absent_or_null_is_nonblocking_but_claim_is_replayed(
    tmp_path, capsys,
):
    model_dir = _artifact(tmp_path)
    (model_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
    }))
    card = build_shipcard(model_dir, build={"quant_method": "gridbook"})

    assert "mtp.dspark" not in card["slots"]
    card["slots"]["mtp.dspark"] = None
    assert "mtp.dspark" not in unfilled_slots(card)
    path = model_dir / "shipcard.json"
    write_shipcard(path, card)
    assert "mtp.dspark" not in required_slots(card, model_dir=model_dir)

    # Once a recognized optional claim is non-null it is release evidence, not
    # ignorable annotation.  Default library and CLI verification both replay
    # it without an explicit --require-slot.
    card["slots"]["mtp.dspark"] = make_record(
        slot="mtp.dspark",
        tool="test",
        passed=False,
        model_sha=card["model_sha"],
        detail="unvalidated claim",
    )
    write_shipcard(path, card)
    assert "mtp.dspark" in required_slots(card, model_dir=model_dir)
    assert any(
        problem.startswith("mtp.dspark: FAILED")
        for problem in verify(card, model_dir=model_dir)
    )
    assert shipcard_cli(["verify", str(path)]) == 1
    assert "mtp.dspark: FAILED" in capsys.readouterr().out
    assert shipcard_cli(["show", str(path)]) == 0
    assert "mtp.dspark" in capsys.readouterr().out


def test_on_disk_dspark_sidecar_requires_claim_but_target_overlay_does_not(
    tmp_path,
):
    draft_dir = _artifact(tmp_path, name="draft")
    (draft_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {"dspark_cb_sidecar": {"schema": "test"}},
    }))
    draft_card = build_shipcard(
        draft_dir, build={"quant_method": "gridbook"}
    )
    assert draft_card["slots"]["mtp.dspark"] is None
    assert required_slots(draft_card, model_dir=draft_dir)[-1] == "mtp.dspark"
    assert "mtp.dspark: UNFILLED" in verify(
        draft_card, model_dir=draft_dir
    )

    # Removing the mutable slot cannot erase an obligation proven by the
    # physical sidecar's own on-disk provenance.
    draft_card["slots"].pop("mtp.dspark")
    assert "mtp.dspark: UNFILLED" in verify(
        draft_card, model_dir=draft_dir
    )

    target_dir = _artifact(tmp_path, name="target")
    (target_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {"dspark_source_overlay": {"schema": "test"}},
    }))
    target_card = build_shipcard(
        target_dir, build={"quant_method": "gridbook"}
    )
    assert "mtp.dspark" not in target_card["slots"]
    assert "mtp.dspark" not in required_slots(
        target_card, model_dir=target_dir
    )


def test_on_disk_gridbook_identity_prevents_receipt_slot_erasure(tmp_path):
    model_dir = _artifact(tmp_path, model_type="deepseek_v4")
    (model_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
    }))
    card = build_shipcard(model_dir, build={"quant_method": "gridbook"})
    card["build"]["quant_method"] = "compressed-tensors"
    card["slots"].pop(CB_REQUIRED_SLOTS[0])

    problems = verify(card, model_dir=model_dir)
    assert f"{CB_REQUIRED_SLOTS[0]}: UNFILLED" in problems


@pytest.mark.parametrize("reserved", [None, 4096, "262144"])
def test_gridbook_card_refuses_missing_or_forged_fixed_reservation(
    tmp_path, reserved,
):
    model_dir = _artifact(tmp_path)
    (model_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
    }))
    card = build_shipcard(model_dir, build={"quant_method": "gridbook"})
    card["reserved_file_bytes"] = reserved

    assert any(
        "reserved_file_bytes" in problem
        for problem in verify(card, model_dir=model_dir)
    )


def test_gridbook_card_refuses_missing_weight_stat_attestation(tmp_path):
    model_dir = _artifact(tmp_path)
    (model_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
    }))
    card = build_shipcard(model_dir, build={"quant_method": "gridbook"})
    assert "weight_stat_attestation" not in card

    assert any(
        "lacks the required weight-stat attestation" in problem
        for problem in verify(card, model_dir=model_dir)
    )


def test_full_card_verifies(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))
    assert verify(load_shipcard(path), model_dir=model_dir) == []


def test_shipcard_fixed_reservation_survives_every_slot_fill(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    assert path.stat().st_size == SHIPCARD_RESERVED_BYTES

    model_sha = compute_model_sha(model_dir)
    for slot in REQUIRED_SLOTS:
        is_gold = slot in GOLD_SLOTS
        fill_slot(path, slot, make_record(
            slot=slot,
            tool="fixed-size-test",
            passed=True,
            model_sha=model_sha,
            metrics={"detail": "x" * 4096, **_GOLD_METRICS.get(slot, {})},
            spec_decode_detected=False if is_gold else None,
            serve_fingerprint=(_FAKE_FINGERPRINT if is_gold else None),
            git_commit=(_FAKE_COMMIT if is_gold else None),
        ))
        assert path.stat().st_size == SHIPCARD_RESERVED_BYTES

    assert verify(load_shipcard(path), model_dir=model_dir) == []


def test_shipcard_reservation_overflow_preserves_previous_record(tmp_path):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    before = path.read_bytes()
    card = load_shipcard(path)
    card["build"]["oversized"] = "x" * SHIPCARD_RESERVED_BYTES

    with pytest.raises(ValueError, match="fixed reservation"):
        write_shipcard(path, card)

    assert path.read_bytes() == before


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
    # Re-fill gold.kl as a fully valid record apart from the spec-decode
    # state, so the assertion below isolates the spec-decode refusal instead of
    # counting the generic gold-evidence problems alongside it.
    fill_slot(path, "gold.kl", make_record(
        slot="gold.kl", tool="test", passed=True,
        model_sha=compute_model_sha(model_dir), spec_decode_detected=spec,
        metrics=_GOLD_METRICS["gold.kl"],
        serve_fingerprint=_FAKE_FINGERPRINT, git_commit=_FAKE_COMMIT))

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
def test_validated_selection_beats_the_surrogate_knee_file(tmp_path):
    """A validated recipe must not be labelled with the surrogate knee's bpp.

    Regression: on Qwen3.8-27B arm B the card claimed 5.9994 bpp
    (pareto.knees.json:log_error) for bytes that were the validated frontier's
    4.7496 pick -- a 1.25 bpp false claim on the publication gate.  Both the
    knee file and the recipe's own stale `achieved_bits` are present here, as
    they are on a real validated run, so this pins the precedence rather than
    just the happy path.
    """
    from prismaquant.shipcard import allocator_achieved_bpp

    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({
        "model.layers.0.mlp.up_proj": {"bits": 4},
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "achieved_bits": 5.50016116330051,      # stale: pre-selection
            "target_bits": 5.5,
            "selected_by": "validated_frontier:kneedle",
            "selected_achieved_bits": 4.749587350041043,
            "selected_label": "allocator_target_4p7500_achieved_4p7496",
        },
    }))
    (tmp_path / "pareto.knees.json").write_text(json.dumps({
        "primary": "log_error",
        "log_error": {"achieved_bits": 5.999404111844319, "target_bits": 6.0},
    }))

    got = allocator_achieved_bpp(recipe)
    assert got["value"] == pytest.approx(4.749587350041043)
    assert got["source"] == "layer_config.json:validated_frontier:kneedle"
    assert got["selected_label"] == "allocator_target_4p7500_achieved_4p7496"


def test_allocator_written_recipe_prefers_its_own_metadata(tmp_path):
    """No validated selection: the recipe's own achieved_bits still wins.

    It is coupled to this file; the knee file is a separate artifact that can
    describe a different point.
    """
    from prismaquant.shipcard import allocator_achieved_bpp

    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({
        "model.layers.0.mlp.up_proj": {"bits": 4},
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "achieved_bits": 4.7513,
            "target_bits": 4.75,
            "achieved_bits_scope": "body_assignment_tensor_payload",
        },
    }))
    (tmp_path / "pareto.knees.json").write_text(json.dumps({
        "primary": "log_error",
        "log_error": {"achieved_bits": 5.999404111844319, "target_bits": 6.0},
    }))

    got = allocator_achieved_bpp(recipe)
    assert got["value"] == pytest.approx(4.7513)
    assert got["source"] == "layer_config.json:achieved_bits"
    assert got["scope"] == "body_assignment_tensor_payload"


def test_validated_selection_without_a_bpp_reports_nothing(tmp_path):
    """Announcing a validated selection but stamping no bpp must not fall back.

    The knee file describes the surrogate frontier, i.e. some OTHER point, so
    silence is correct and a number would be a fabrication.
    """
    from prismaquant.shipcard import allocator_achieved_bpp

    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({
        "model.layers.0.mlp.up_proj": {"bits": 4},
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "selected_by": "validated_frontier:kneedle",
        },
    }))
    (tmp_path / "pareto.knees.json").write_text(json.dumps({
        "primary": "log_error",
        "log_error": {"achieved_bits": 5.999404111844319},
    }))

    got = allocator_achieved_bpp(recipe)
    assert got["value"] is None
    assert "no bpp stamped" in got["source"]


# ---------------------------------------------------------------------------
def test_cli_verify_exit_codes(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)

    assert shipcard_cli(["verify", str(path), "--model-dir", str(model_dir)]) == 1
    assert "REFUSED" in capsys.readouterr().out

    _fill_all(path, compute_model_sha(model_dir))
    assert shipcard_cli(["verify", str(path)]) == 0
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
        "n_positions": 4096,
        "seqlen": 512,
        "spec_decode_detected": False,
        "serve_fingerprint": "f" * 64,
        "git_commit": "a" * 40,
    }))

    assert shipcard_cli([
        "fill", str(path), "--slot", "gold.kl", "--record", str(result)]) == 0

    record = load_shipcard(path)["slots"]["gold.kl"]
    assert record["passed"] is True
    assert record["model_sha"] == compute_model_sha(model_dir)
    assert record["metrics"]["kl_confident_mean"] == pytest.approx(0.0143)
    assert record["serve_fingerprint"] == "f" * 64
    assert record["git_commit"] == "a" * 40
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


@pytest.mark.parametrize("score_positions, expect_refusal", [
    ("all", False),
    ("final", True),
    (None, True),
])
def test_gold_kl_refuses_the_last_token_hook_screen(
    tmp_path, score_positions, expect_refusal
):
    """A positive sample count was never enough to make a number the gold KL.

    `measure_vllm_full_kl.py --score-positions final` — its DEFAULT — scores one
    position per sequence, the window-final context. That is the cheap
    last-token "hook KL" screen: triage only, never a promotion metric. It
    reports n_samples=8 and sailed through the count check, so the card could
    not tell an 8-position screen from a 4088-position gold measurement.

    Found 2026-08-14 on the Qwen3.8-27B lane, where the driver simply omitted
    the flag: the teacher wrote shape [8, 248077] — 8 positions, not 8x511.
    """
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))

    metrics = dict(_GOLD_METRICS["gold.kl"])
    if score_positions is None:
        metrics.pop("score_positions")
        # `final` mode does not stamp the key at all, which is why absence has
        # to be refused just as loudly as the explicit value.
        metrics["n_positions"] = 8
    else:
        metrics["score_positions"] = score_positions
    fill_slot(path, "gold.kl", make_record(
        slot="gold.kl", tool="test", passed=True,
        model_sha=compute_model_sha(model_dir), spec_decode_detected=False,
        metrics=metrics,
        serve_fingerprint=_FAKE_FINGERPRINT, git_commit=_FAKE_COMMIT))

    problems = verify(load_shipcard(path), model_dir=model_dir)
    hits = [p for p in problems if "score_positions" in p]
    assert bool(hits) is expect_refusal, problems
    if expect_refusal:
        assert "triage only" in hits[0]


def test_native_lane_gold_slots_are_verified_at_all(tmp_path):
    """The generic gold checks used to run only when the card was Gridbook CB.

    A NATIVE card — the default lane, and the one shipping artifacts — had its
    gold slots checked for nothing but spec-decode, so an empty metrics dict
    with no fingerprint and no producer commit verified clean.
    """
    model_dir = _artifact(tmp_path)
    path = _open_card(tmp_path, model_dir)
    _fill_all(path, compute_model_sha(model_dir))
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    for slot in GOLD_SLOTS:
        fill_slot(path, slot, make_record(
            slot=slot, tool="test", passed=True,
            model_sha=compute_model_sha(model_dir),
            spec_decode_detected=False))

    problems = verify(load_shipcard(path), model_dir=model_dir)
    for slot in GOLD_SLOTS:
        assert any(
            p.startswith(f"{slot}: carries no finite") for p in problems
        ), problems
        assert any(
            p == f"{slot}: missing exact serve fingerprint" for p in problems
        ), problems
        assert any(
            p == f"{slot}: missing full producer git commit" for p in problems
        ), problems


def test_native_model_sha_attests_the_chat_template(tmp_path):
    """A native card used to bind config.json plus container SIZES only.

    Auxiliary files went unhashed unless the artifact carried a
    `quant_config.json` (i.e. unless it was a CB artifact), so swapping
    `chat_template.jinja` or `tokenizer.json` on a native checkpoint left
    `model_sha` bit-identical. Demonstrated 2026-08-15 on the published
    Qwen3.8-27B native artifact. That is not cosmetic on a tool-calling model:
    the chat template decides where a tool call is emitted, so a served
    artifact with the wrong one is broken in a way no weight check sees.
    """
    model_dir = _artifact(tmp_path)
    (model_dir / "chat_template.jinja").write_text("{{ messages }}")
    (model_dir / "tokenizer.json").write_text('{"version": "1"}')

    before = compute_model_sha(model_dir)
    (model_dir / "chat_template.jinja").write_text("{{ messages }}{# swap #}")
    assert compute_model_sha(model_dir) != before

    (model_dir / "chat_template.jinja").write_text("{{ messages }}")
    assert compute_model_sha(model_dir) == before
    (model_dir / "tokenizer.json").write_text('{"version": "2"}')
    assert compute_model_sha(model_dir) != before


def test_serving_an_artifact_does_not_invalidate_its_own_card(tmp_path):
    """`serve_manifest.json` is evidence ABOUT a serve, not artifact content.

    `scripts/lib/serve_manifest.sh` writes the R15 serve fingerprint INTO the
    model dir after a server comes up. It records the serving stack -- image,
    argv, the loaded `.so` set, hostname, boot id, timestamp -- so it differs
    between two serves of byte-identical weights. Hashing it made the act of
    VALIDATING an artifact invalidate the card that validation was for:
    observed 2026-08-15 on Qwen3.8-27B CB-A, where the eager smoke moved
    `model_sha` 677f278a -> bf6abc17 and `verify` then reported "artifact
    changed since the shipcard was opened".

    Nothing goes unbound: every slot that cites a manifest binds it by its own
    `*serve_manifest_sha256`, which is where a claim about a serve belongs.
    """
    model_dir = _artifact(tmp_path)
    (model_dir / "chat_template.jinja").write_text("{{ messages }}")

    path = _open_card(tmp_path, model_dir)
    before = compute_model_sha(model_dir)
    _fill_all(path, before)
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    (model_dir / "serve_manifest.json").write_text(
        '{"schema": "prismaquant.serve_manifest/1", "boot_id": "a"}')
    assert compute_model_sha(model_dir) == before
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    # A second serve writes a DIFFERENT manifest; the identity must not move.
    (model_dir / "serve_manifest.json").write_text(
        '{"schema": "prismaquant.serve_manifest/1", "boot_id": "b"}')
    assert compute_model_sha(model_dir) == before

    # The exclusion is by exact name, not by shape: a file that merely looks
    # like it stays attested.
    (model_dir / "serve_manifest.json.bak").write_text("{}")
    assert compute_model_sha(model_dir) != before


def test_documenting_an_artifact_does_not_invalidate_its_own_card(tmp_path):
    """`README.md` in a model dir IS the HF model card, not artifact content.

    `tools/publish_artifact.py` has no --model-card argument: it uploads the
    complete local file set with no filters, so the card reaches the Hub only
    by sitting in the artifact directory under that name. Hashing it made the
    act of DOCUMENTING an artifact invalidate the card that documents it --
    the same failure as the serve fingerprint above, one step earlier in the
    release. Observed 2026-08-15 on qwen38-27b-arm-b/exported: a README
    dropped in at 18:33 moved the identity off the 17:55 card
    (e7ac09f8 -> 3c4a83a1) and locked the artifact out of publication.

    It also decides whether an artifact can quote its own measured numbers.
    Every gate record binds `model_sha`; gold KL/PPL only exists after the
    gates; writing it into the card would invalidate the records that produced
    it. Re-running the gates does not escape that -- KL drifts across docker
    sessions -- so the exclusion is what makes a self-describing card possible.
    """
    model_dir = _artifact(tmp_path)
    (model_dir / "chat_template.jinja").write_text("{{ messages }}")

    path = _open_card(tmp_path, model_dir)
    before = compute_model_sha(model_dir)
    _fill_all(path, before)
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    (model_dir / "README.md").write_text("# a model card\n")
    assert compute_model_sha(model_dir) == before
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    # And the gold numbers can be written into it afterwards.
    (model_dir / "README.md").write_text("# a model card\n\nKL 0.0142\n")
    assert compute_model_sha(model_dir) == before
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    # One exact filename, not a category: a figure the card references, or a
    # doc under any other name, stays attested.
    (model_dir / "allocation-map.png").write_bytes(b"\x89PNG\r\n")
    assert compute_model_sha(model_dir) != before
    after_png = compute_model_sha(model_dir)
    (model_dir / "NOTES.md").write_text("notes")
    assert compute_model_sha(model_dir) != after_png


def test_a_card_stamped_while_the_readme_was_hashed_still_verifies(tmp_path):
    """The fix must not unbreak future artifacts by breaking present ones.

    A card written under the old scope on a directory that already contained a
    README verifies today only because the README was hashed into it. `verify`
    accepts that identity as a fallback, and only as a fallback.
    """
    model_dir = _artifact(tmp_path)
    (model_dir / "README.md").write_text("# an already-published card\n")

    legacy_sha = compute_model_sha(model_dir, legacy_readme_hashed=True)
    assert legacy_sha != compute_model_sha(model_dir)

    path = _open_card(tmp_path, model_dir)
    card = load_shipcard(path)
    card["model_sha"] = legacy_sha
    write_shipcard(path, card)
    _fill_all(path, legacy_sha)
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    # Still a fallback: editing that README moves the legacy identity too, so
    # a card from the hashed era keeps its README attestation.
    (model_dir / "README.md").write_text("# tampered\n")
    assert any("artifact changed" in p
               for p in verify(load_shipcard(path), model_dir=model_dir))


def test_a_card_written_under_the_legacy_native_scope_still_verifies(tmp_path):
    """Published native cards must not all read as 'artifact changed'.

    The legacy identity described its artifact faithfully under the rules it
    was computed with, so `verify` accepts it as a FALLBACK. It must remain a
    fallback: the legacy scope cannot produce a current-scope sha, so a card
    written today is never checked the weak way.
    """
    model_dir = _artifact(tmp_path)
    (model_dir / "chat_template.jinja").write_text("{{ messages }}")

    legacy_sha = compute_model_sha(model_dir, legacy_native_scope=True)
    assert legacy_sha != compute_model_sha(model_dir)

    path = _open_card(tmp_path, model_dir)
    card = load_shipcard(path)
    card["model_sha"] = legacy_sha
    write_shipcard(path, card)
    _fill_all(path, legacy_sha)

    assert verify(load_shipcard(path), model_dir=model_dir) == []

    # The legacy scope hashes container SIZES, so a same-size weight swap is
    # exactly what it could never see -- and still cannot. This test pins the
    # tolerance as bounded, not as a hole: what it accepts is the old
    # guarantee, never less.
    (model_dir / "chat_template.jinja").write_text("{{ messages }}{# swap #}")
    assert verify(load_shipcard(path), model_dir=model_dir) == []

    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"longer!!")
    assert any(
        "artifact changed since the shipcard was opened" in problem
        for problem in verify(load_shipcard(path), model_dir=model_dir)
    )
