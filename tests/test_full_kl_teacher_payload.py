from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from tools.full_kl_teacher_payload import (
    N_SAMPLES,
    PROMPT_TOP_K,
    SEQLEN,
    TEACHER_PAYLOAD_SCHEMA,
    TOPK_MINIMUM_COVERAGE,
    TeacherPayloadError,
    atomic_json_write,
    atomic_torch_save,
    build_calibration_contract,
    canonical_sha256,
    compact_source_model_identity,
    load_teacher_evidence,
    payload_semantic_sha256,
    safe_load_torch_payload,
    teacher_meta,
    tensor_descriptor,
    topk_coverage_policy,
    topk_coverage_summary,
    tokenizer_identity,
    validate_teacher_payload,
)
from tools.build_streamed_full_kl_teacher import _tokenizer_vocab_size
from tools.measure_vllm_full_kl import _student


def _execute_marker(path: str) -> dict:
    Path(path).write_text("unsafe pickle executed")
    return {}


class _MaliciousPayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return _execute_marker, (str(self.marker),)


def _source_identity(tmp_path: Path) -> dict:
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"source weights")
    config = {"model_type": "deepseek_v4"}
    weight_map = {"model.layers.0.weight": "layers.0.weight"}
    checkpoint_map = {"layers.0.weight": shard.name}
    shards = [{
        "path": str(shard.resolve()),
        "size": shard.stat().st_size,
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
    }]
    value_bearing = {
        "config": config,
        "weight_map": weight_map,
        "shards": shards,
        "checkpoint_weight_map": checkpoint_map,
    }
    return {
        "schema": "prismaquant.streamed_model.identity.v1",
        "source": str(tmp_path),
        # A local source has no Hugging Face resolved commit.  This is the
        # actual DSv4 campaign shape and must remain legal.
        "resolved_commit": None,
        "content_sha256": canonical_json_sha256(
            value_bearing, where="test identity"
        ),
        **value_bearing,
    }


def _payload(tmp_path: Path) -> dict:
    torch.manual_seed(5)
    calib = torch.randint(0, 3000, (N_SAMPLES, SEQLEN), dtype=torch.long)
    starts = list(range(10, 10 + N_SAMPLES))
    tokenizer = {
        "schema": "prismaquant.tokenizer_identity/1",
        "content_sha256": "a" * 64,
        "files": {"tokenizer.json": {"bytes": 1, "sha256": "b" * 64}},
    }
    contract = build_calibration_contract(
        dataset_fingerprint="dataset-fingerprint",
        corpus_sha256="c" * 64,
        tokenizer=tokenizer,
        starts=starts,
        total_tokens=100_000,
        calib_ids=calib,
    )
    identity = _source_identity(tmp_path)
    ids = torch.arange(PROMPT_TOP_K, dtype=torch.int32).reshape(1, 1, -1)
    ids = ids.expand(N_SAMPLES, SEQLEN - 1, -1).contiguous()
    # Strictly decreasing top-K support carrying 99% of the probability mass.
    logits = torch.linspace(4.0, -4.0, PROMPT_TOP_K, dtype=torch.float64)
    lps = (torch.log_softmax(logits, dim=0) + math.log(0.99)).to(torch.float32)
    lps = lps.reshape(1, 1, -1).expand_as(ids).contiguous()
    payload = {
        "schema": TEACHER_PAYLOAD_SCHEMA,
        "score_positions": "all",
        "prompt_top_k": PROMPT_TOP_K,
        "topk_ids": ids,
        "topk_lps": lps,
        "calib_ids": calib,
        "starts": starts,
        "model": str(tmp_path),
        "n_samples": N_SAMPLES,
        "seqlen": SEQLEN,
        # Must exceed the top-K support, so derive it rather than hardcoding a
        # number that silently becomes invalid when PROMPT_TOP_K moves.
        "vocab_size": PROMPT_TOP_K * 4,
        "source_model_identity": identity,
        "source_model": compact_source_model_identity(identity),
        "source_model_identity_sha256": canonical_sha256(identity),
        "calibration_contract": contract,
        "calibration_contract_sha256": canonical_sha256(contract),
    }
    payload["payload_semantic_sha256"] = payload_semantic_sha256(payload)
    return payload


def test_teacher_payload_roundtrip_binds_null_commit_and_tensor_bytes(tmp_path):
    payload = _payload(tmp_path)
    assert validate_teacher_payload(payload)["source_model"][
        "resolved_commit"
    ] is None
    descriptor = tensor_descriptor(payload["topk_lps"])
    assert descriptor["dtype"] == "float32"
    assert descriptor["shape"] == [N_SAMPLES, SEQLEN - 1, PROMPT_TOP_K]
    assert len(descriptor["sha256"]) == 64
    coverage = topk_coverage_summary(
        payload["topk_ids"],
        payload["topk_lps"],
        vocab_size=payload["vocab_size"],
    )
    assert coverage["topk_coverage_mean"] == pytest.approx(0.99, abs=1e-7)
    assert coverage["topk_coverage_min"] == pytest.approx(0.99, abs=1e-7)
    assert coverage["topk_coverage_policy"] == topk_coverage_policy()
    assert (
        coverage["topk_coverage_policy"][
            "minimum_probability_mass_per_position"
        ]
        == TOPK_MINIMUM_COVERAGE
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda value: value["topk_lps"].__setitem__((0, 0, -1), -99.0),
         "semantic payload digest"),
        (lambda value: value["calib_ids"].__setitem__((0, 0), 7),
         "calib_ids bytes"),
        (lambda value: value["source_model"].__setitem__(
            "checkpoint_shards", 2), "compact source identity"),
        (lambda value: value.__setitem__("prompt_top_k", 999),
         "scoring dimensions"),
        (lambda value: value.__setitem__("extra", True), "fields are not closed"),
    ],
)
def test_teacher_payload_refuses_semantic_forgery(tmp_path, mutation, error):
    payload = copy.deepcopy(_payload(tmp_path))
    mutation(payload)
    with pytest.raises(TeacherPayloadError, match=error):
        validate_teacher_payload(payload)


def _duplicate_topk_id(payload):
    payload["topk_ids"][0, 0, 1] = payload["topk_ids"][0, 0, 0]


def _unsort_topk_logprobs(payload):
    payload["topk_lps"][0, 0, 1] = payload["topk_lps"][0, 0, 0] + 0.01


def _make_topk_logprob_nonfinite(payload):
    payload["topk_lps"][0, 0, 0] = float("nan")


def _make_topk_id_out_of_range(payload):
    payload["topk_ids"][0, 0, 0] = payload["vocab_size"]


def _over_normalize_topk_mass(payload):
    payload["topk_lps"].fill_(math.log(1.001 / PROMPT_TOP_K))


def _drop_topk_coverage(payload):
    payload["topk_lps"].fill_(math.log(0.50 / PROMPT_TOP_K))


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (_duplicate_topk_id, "duplicate token ids"),
        (_unsort_topk_logprobs, "not nonincreasing"),
        (_make_topk_logprob_nonfinite, "non-finite"),
        (_make_topk_id_out_of_range, "out of range"),
        (_over_normalize_topk_mass, "probability mass exceeds one"),
        (_drop_topk_coverage, "coverage falls below"),
    ],
)
def test_teacher_payload_refuses_resigned_topk_semantic_tamper(
    tmp_path, mutation, error
):
    payload = _payload(tmp_path)
    mutation(payload)
    # Model an attacker who can recompute the unkeyed semantic digest.  The
    # row-level probability contract must still reject the tensor values.
    payload["payload_semantic_sha256"] = payload_semantic_sha256(payload)
    with pytest.raises(TeacherPayloadError, match=error):
        validate_teacher_payload(payload)


def test_serialized_payload_and_meta_are_replayed_into_compact_evidence(tmp_path):
    payload = _payload(tmp_path)
    payload_path = tmp_path / "teacher.pt"
    meta_path = tmp_path / "teacher.json"
    atomic_torch_save(payload, payload_path)
    meta = teacher_meta(
        payload_path=payload_path,
        elapsed_s=12.5,
    )
    atomic_json_write(meta, meta_path)

    loaded, evidence = load_teacher_evidence(payload_path, meta_path)
    assert loaded["payload_semantic_sha256"] == payload["payload_semantic_sha256"]
    assert evidence["schema"] == "prismaquant.full_kl_teacher_evidence/1"
    assert evidence["meta_sha256"] == hashlib.sha256(meta_path.read_bytes()).hexdigest()
    assert evidence["source_model"] == payload["source_model"]
    assert evidence["topk_coverage_mean"] == meta["topk_coverage_mean"]
    assert evidence["topk_coverage_min"] == meta["topk_coverage_min"]
    assert evidence["topk_coverage_policy"] == topk_coverage_policy()

    with payload_path.open("ab") as handle:
        handle.write(b"forgery")
    with pytest.raises(TeacherPayloadError, match="serialized payload bytes"):
        load_teacher_evidence(payload_path, meta_path)


def test_meta_refuses_tensor_descriptor_forgery(tmp_path):
    payload = _payload(tmp_path)
    payload_path = tmp_path / "teacher.pt"
    meta_path = tmp_path / "teacher.json"
    atomic_torch_save(payload, payload_path)
    meta = teacher_meta(
        payload_path=payload_path,
        elapsed_s=1.0,
    )
    meta["tensor_descriptors"]["topk_ids"]["sha256"] = "f" * 64
    atomic_json_write(meta, meta_path)
    with pytest.raises(TeacherPayloadError, match="metadata differs"):
        load_teacher_evidence(payload_path, meta_path)


@pytest.mark.parametrize("field", ["topk_coverage_mean", "topk_coverage_min"])
def test_meta_refuses_forged_coverage_recomputed_from_tensor_bytes(
    tmp_path, field
):
    payload = _payload(tmp_path)
    payload_path = tmp_path / "teacher.pt"
    meta_path = tmp_path / "teacher.json"
    atomic_torch_save(payload, payload_path)
    meta = teacher_meta(payload_path=payload_path, elapsed_s=1.0)
    meta[field] = float(meta[field]) - 0.01
    atomic_json_write(meta, meta_path)
    with pytest.raises(TeacherPayloadError, match="metadata differs"):
        load_teacher_evidence(payload_path, meta_path)


def test_meta_refuses_forged_coverage_policy(tmp_path):
    payload_path = tmp_path / "teacher.pt"
    meta_path = tmp_path / "teacher.json"
    atomic_torch_save(_payload(tmp_path), payload_path)
    meta = teacher_meta(payload_path=payload_path, elapsed_s=1.0)
    meta["topk_coverage_policy"][
        "minimum_probability_mass_per_position"
    ] = 0.0
    atomic_json_write(meta, meta_path)
    with pytest.raises(TeacherPayloadError, match="metadata differs"):
        load_teacher_evidence(payload_path, meta_path)


def test_weights_only_teacher_load_never_executes_reduce(tmp_path):
    marker = tmp_path / "pickle-executed"
    payload_path = tmp_path / "malicious.pt"
    meta_path = tmp_path / "malicious.json"
    torch.save(_MaliciousPayload(marker), payload_path)
    meta_path.write_text("{}")

    with pytest.raises(TeacherPayloadError, match="safely load"):
        safe_load_torch_payload(payload_path)
    assert not marker.exists()

    with pytest.raises(TeacherPayloadError, match="safely load"):
        load_teacher_evidence(payload_path, meta_path)
    assert not marker.exists()

    # The legacy/no-sidecar student entry point uses the same restricted load.
    args = SimpleNamespace(
        teacher_meta=None,
        teacher_payload=str(payload_path),
    )
    with pytest.raises(TeacherPayloadError, match="safely load"):
        _student(args)
    assert not marker.exists()


def test_tokenizer_identity_hashes_only_present_contract_files(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text('{"legacy":true}')
    (tmp_path / "unrelated.bin").write_bytes(b"not tokenizer state")
    identity = tokenizer_identity(tmp_path)
    assert set(identity["files"]) == {"tokenizer.json", "tokenizer_config.json"}
    assert identity["content_sha256"] == canonical_sha256({
        "files": identity["files"]
    })


def test_atomic_publish_refuses_preexisting_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "teacher.json"
    temporary = tmp_path / f".{target.name}.tmp.123"
    temporary.write_text("do not overwrite")
    monkeypatch.setattr("tools.full_kl_teacher_payload.os.getpid", lambda: 123)
    with pytest.raises(TeacherPayloadError, match="temporary metadata"):
        atomic_json_write({"ok": True}, target)
    assert temporary.read_text() == "do not overwrite"
    assert not target.exists()


def test_teacher_uses_added_token_vocabulary_cardinality():
    class Tokenizer:
        vocab_size = 128_000

        def __len__(self):
            return 129_280

    assert _tokenizer_vocab_size(Tokenizer()) == 129_280
