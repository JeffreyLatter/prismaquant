"""Anchors for the per-token decode read-bytes stat.

The load-bearing test is :func:`test_synthetic_ledger_matches_hand_computation`:
every byte in a tiny synthetic checkpoint is written out by hand below, and the
module must reproduce the ledger and the weighted total exactly.  A stat that
is "about right" is worthless as a bandwidth ceiling, so the anchor is exact
integers, not tolerances.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from prismaquant import footprint as fp
from prismaquant import read_traffic as rt
from prismaquant.model_profiles import detect_profile_with_warning

# ---------------------------------------------------------------------------
# The synthetic model, written out by hand.
#
# Everything is BF16 on disk (2 bytes/param), one body layer, 4 routed experts
# of which a token activates 2.
#
#   name                                              shape   params  bytes
#   model.embed_tokens.weight                        (16, 8)     128    256
#   model.layers.0.self_attn.q_proj.weight            (8, 8)      64    128
#   model.layers.0.mlp.gate.weight            (router)(4, 8)      32     64
#   model.layers.0.mlp.experts.{0..3}.gate_proj.w..   (4, 8)   4x 32  4x 64
#   model.layers.0.input_layernorm.weight               (8,)       8     16
#   lm_head.weight                                   (16, 8)     128    256
#   mtp.fc.weight                             (draft) (8, 8)      64    128
#                                                          source total 1104
# ---------------------------------------------------------------------------
HIDDEN, INTER, VOCAB, N_EXPERTS, TOPK = 8, 4, 16, 4, 2

_TENSORS: dict[str, tuple[int, ...]] = {
    "model.embed_tokens.weight": (VOCAB, HIDDEN),
    "model.layers.0.self_attn.q_proj.weight": (HIDDEN, HIDDEN),
    "model.layers.0.mlp.gate.weight": (N_EXPERTS, HIDDEN),
    **{
        f"model.layers.0.mlp.experts.{i}.gate_proj.weight": (INTER, HIDDEN)
        for i in range(N_EXPERTS)
    },
    "model.layers.0.input_layernorm.weight": (HIDDEN,),
    "lm_head.weight": (VOCAB, HIDDEN),
    "mtp.fc.weight": (HIDDEN, HIDDEN),
}

SOURCE_TOTAL_BYTES = 1104

# The two format numbers the ledger depends on, stated so the arithmetic
# below is checkable without running the registry.  FP8_E4M3 stores one byte
# per parameter plus an fp32 scale plane:
#   (8, 8)    -> 64 weight bytes + 8 row scales x 4 B  =  96 B
#   (4, 4, 8) -> 128 weight bytes + 4 expert scales x 4 B = 144 B
FP8_Q_PROJ_BYTES = 96
FP8_EXPERT_STACK_BYTES = 144

PACKED_EXPERTS = "model.layers.0.mlp.experts.gate_up_proj"
Q_PROJ = "model.layers.0.self_attn.q_proj"


def _write_safetensors(path: Path, tensors: dict[str, tuple[int, ...]]) -> None:
    """Write a valid BF16 safetensors shard; only the header is ever read."""
    header: dict[str, object] = {}
    offset = 0
    for name, shape in tensors.items():
        n = 1
        for dim in shape:
            n *= dim
        nbytes = n * 2
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic-moe"
    root.mkdir()
    _write_safetensors(root / "model.safetensors", _TENSORS)
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "num_experts": N_EXPERTS,
        "num_experts_per_tok": TOPK,
    }))
    return root


@pytest.fixture()
def profile(model_dir: Path):
    return detect_profile_with_warning(str(model_dir), entrypoint="test")


ASSIGNMENT = {Q_PROJ: "FP8_DYNAMIC", PACKED_EXPERTS: "FP8_DYNAMIC"}
STATS = {
    Q_PROJ: {"out_features": HIDDEN, "in_features": HIDDEN,
             "n_params": HIDDEN * HIDDEN},
    PACKED_EXPERTS: {"num_experts": N_EXPERTS, "out_features": INTER,
                     "in_features": HIDDEN,
                     "n_params": N_EXPERTS * INTER * HIDDEN},
}


def _report(model_dir: Path, profile, **kwargs) -> dict:
    return rt.assignment_read_traffic(
        ASSIGNMENT, STATS, model_path=str(model_dir), profile=profile,
        **kwargs)


def test_source_partition_covers_every_checkpoint_byte(model_dir: Path):
    """The floor half is only honest if it provably misses nothing."""
    spans = fp.source_tensor_span_bytes(str(model_dir))
    assert sum(spans.values()) == SOURCE_TOTAL_BYTES
    assert sum(spans.values()) == fp.source_checkpoint_bytes(str(model_dir))[0]
    assert set(spans) == set(_TENSORS)


def test_synthetic_ledger_matches_hand_computation(model_dir: Path, profile):
    report = _report(model_dir, profile)

    # --- stored bytes, class by class -------------------------------------
    # dense           = q_proj re-encoded to FP8                       96
    # routed_experts  = the 4-expert stack re-encoded to FP8          144
    # held_fixed      = router 64 + layernorm 16 + lm_head 256        336
    # excluded_embed  = embed_tokens                                  256
    # excluded_mtp    = the draft sidecar                             128
    #                                                          total  960
    classes = report["classes"]
    assert classes["dense"]["stored_bytes"] == FP8_Q_PROJ_BYTES
    assert classes["routed_experts"]["stored_bytes"] == FP8_EXPERT_STACK_BYTES
    assert classes["held_fixed"]["stored_bytes"] == 64 + 16 + 256
    assert classes["excluded_embedding"]["stored_bytes"] == 256
    assert classes["excluded_mtp"]["stored_bytes"] == 128
    assert classes["excluded_non_text_graph"]["stored_bytes"] == 0
    assert classes["resident_codebooks"]["stored_bytes"] == 0

    # --- the reconciliation the module refuses to run without --------------
    priced = fp.assignment_artifact_bytes(
        ASSIGNMENT, STATS,
        source_total_bytes=SOURCE_TOTAL_BYTES,
        source_manifest=fp.source_tensor_bytes_manifest(
            str(model_dir), profile.checkpoint_to_live_name,
            profile.packed_expert_parent_for_projection),
        cb_serialization_context=None,
    )
    assert priced["artifact_payload_bytes"] == 960
    assert report["reconciliation"]["ledger_stored_bytes"] == 960
    assert report["reconciliation"]["footprint_artifact_payload_bytes"] == 960

    # --- the weighted total ------------------------------------------------
    # 96 (dense, p=1) + 336 (held_fixed, p=1) + 144 x 2/4 (routed) = 504
    assert report["read_bytes_per_token"] == 504
    assert report["read_gb_per_token"] == 504 / fp.GB
    assert report["breakdown"] == {
        "dense": 96,
        "routed": 72,
        "held_fixed": 336,
        "resident_codebooks": 0,
    }
    assert report["excluded"]["embedding_bytes"] == 256
    assert report["excluded"]["mtp_bytes"] == 128


def test_read_probability_is_topk_over_e_for_routed_and_one_for_dense(
    model_dir: Path, profile,
):
    report = _report(model_dir, profile)
    classes = report["classes"]
    assert classes["routed_experts"]["read_probability"] == TOPK / N_EXPERTS
    assert classes["dense"]["read_probability"] == 1.0
    assert classes["held_fixed"]["read_probability"] == 1.0
    assert classes["excluded_embedding"]["read_probability"] == 0.0
    assert report["routing"]["num_experts_per_tok"] == TOPK
    assert report["routing"]["n_routed_experts"] == N_EXPERTS
    assert report["routing"]["read_probability"] == TOPK / N_EXPERTS
    # The table is the single authority, and dense/held_fixed read every token.
    assert rt.READ_CLASS_TABLE["dense"] == 1.0
    assert rt.READ_CLASS_TABLE["held_fixed"] == 1.0


@pytest.mark.parametrize("topk", [1, 2, 3, 4])
def test_routed_read_bytes_scale_linearly_with_topk(
    model_dir: Path, profile, topk: int,
):
    config = json.loads((model_dir / "config.json").read_text())
    config["num_experts_per_tok"] = topk
    report = _report(model_dir, profile, config=config)
    assert report["breakdown"]["routed"] == round(
        FP8_EXPERT_STACK_BYTES * topk / N_EXPERTS)
    assert report["breakdown"]["dense"] == FP8_Q_PROJ_BYTES  # unchanged


def test_missing_topk_declaration_refuses(model_dir: Path, profile):
    config = json.loads((model_dir / "config.json").read_text())
    config.pop("num_experts_per_tok")
    with pytest.raises(rt.ReadTrafficError, match="declares no"):
        _report(model_dir, profile, config=config)


def test_expert_count_disagreement_refuses(model_dir: Path, profile):
    config = json.loads((model_dir / "config.json").read_text())
    config["num_experts"] = N_EXPERTS + 1
    with pytest.raises(rt.ReadTrafficError, match="carries 4 experts"):
        _report(model_dir, profile, config=config)


def test_classification_table(profile):
    """The mapping from tensor class to read probability, pinned."""
    cases = {
        "model.layers.0.mlp.experts.gate_up_proj": "routed_experts",
        "model.layers.0.mlp.experts.3.gate_proj": "routed_experts",
        "model.layers.0.mlp.shared_experts.gate_proj": "held_fixed",
        "model.layers.0.mlp.gate": "held_fixed",
        "model.layers.0.input_layernorm": "held_fixed",
        "lm_head": "held_fixed",
        "model.embed_tokens": "excluded_embedding",
        "mtp.fc": "excluded_mtp",
        "mtp.layers.0.self_attn.q_proj": "excluded_mtp",
        "cb_codebook.ref0.NVFP4_CB_K12": "resident_codebooks",
    }
    for name, expected in cases.items():
        assert rt.classify_read_class(name, profile=profile) == expected, name
    # A shared expert is read on EVERY token; the segment test is what keeps
    # it out of the routed class.
    assert rt.classify_read_class(
        "model.layers.0.mlp.experts.3.gate_proj", profile=profile,
        in_assignment=True) == "routed_experts"
    assert rt.classify_read_class(
        "model.layers.0.self_attn.q_proj", profile=profile,
        in_assignment=True) == "dense"


def test_exported_checkpoint_ledger(model_dir: Path, profile):
    """The post-export form classifies every shipped byte, or refuses."""
    report = rt.exported_checkpoint_read_traffic(
        str(model_dir), profile=profile)
    assert report["reconciliation"]["ledger_stored_bytes"] == SOURCE_TOTAL_BYTES
    classes = report["classes"]
    # q_proj 128 + router 64 + layernorm 16 + lm_head 256 = 464 always-active
    assert classes["held_fixed"]["stored_bytes"] == 464
    assert classes["routed_experts"]["stored_bytes"] == 256
    assert classes["excluded_embedding"]["stored_bytes"] == 256
    assert classes["excluded_mtp"]["stored_bytes"] == 128
    assert report["read_bytes_per_token"] == 464 + 256 * TOPK // N_EXPERTS


def test_claim_shape_is_advisory(model_dir: Path, profile):
    claim = rt.read_traffic_claim(str(model_dir), profile=profile)
    assert claim["value"] == pytest.approx(592 / fp.GB)
    assert claim["scope"] == rt.READ_SCOPE
    assert set(claim["breakdown"]) == {
        "dense", "routed", "held_fixed", "resident_codebooks"}
    # A broken input reports a reason; it never raises into an export.
    assert rt.read_traffic_claim(None)["value"] is None
    assert rt.read_traffic_claim("/nonexistent-export")["value"] is None


def test_exporter_shipcard_stamps_read_gb_per_token(model_dir: Path):
    """The stat lands beside `achieved_bpp` on the card the exporter writes."""
    from prismaquant import shipcard as _shipcard
    from prismaquant.export_native_compressed import _write_shipcard

    _write_shipcard(
        model_dir,
        source_model=str(model_dir),
        layer_config_path=None,
        assignment=ASSIGNMENT,
        config_assignment=ASSIGNMENT,
        hist={},
    )
    card = json.loads(
        (model_dir / _shipcard.SHIPCARD_FILENAME).read_text())
    build = card["build"]
    assert "achieved_bpp" in build
    claim = build["read_gb_per_token"]
    assert claim["value"] == pytest.approx(592 / fp.GB)
    assert claim["breakdown"]["routed"] == 128
    assert claim["routing"]["read_probability"] == TOPK / N_EXPERTS
    assert claim["scope"] == rt.READ_SCOPE
