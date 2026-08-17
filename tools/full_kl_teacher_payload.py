#!/usr/bin/env python3
"""Value-bearing contracts for a streamed all-position KL teacher.

The DSv4 BF16 checkpoint is larger than one Spark's unified memory, so the
release gold lane builds its teacher distribution with PrismaQuant's existing
layer streamer.  This module keeps that payload independently replayable:
tensor bytes, source checkpoint, tokenized calibration windows, and the
serialized file are all digest-bound.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import torch


TEACHER_PAYLOAD_SCHEMA = "prismaquant.full_kl_teacher_payload/1"
TEACHER_META_SCHEMA = "prismaquant.full_kl_teacher_meta/1"
TEACHER_EVIDENCE_SCHEMA = "prismaquant.full_kl_teacher_evidence/1"
CALIBRATION_SCHEMA = "prismaquant.wikitext_gold_calibration/1"
TOKENIZER_IDENTITY_SCHEMA = "prismaquant.tokenizer_identity/1"

WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "train"
# Immutable commit currently backing the repository's historical gold lane.
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
N_SAMPLES = 8
SEQLEN = 512
WINDOW_SEED = 42
PROMPT_TOP_K = 1024
EXPECTED_POSITIONS = N_SAMPLES * (SEQLEN - 1)

# FP32 log-softmax values can round the reconstructed probability sum a few
# ulps above one.  One part per million is a deliberately tight allowance for
# that representation effect; it is not permission for an over-normalized
# teacher distribution.
TOPK_PROBABILITY_MASS_ABS_TOLERANCE = 1e-6
# The tail bucket makes the statistic defined below this point, but allowing a
# mostly-tail distribution would make the top-1024 comparison uninformative.
# Requiring 90% support caps the declared aggregated teacher tail at 10% per
# scored position while remaining conservative for a language-model top-1024.
TOPK_MINIMUM_COVERAGE = 0.90
TOPK_COVERAGE_POLICY_SCHEMA = "prismaquant.topk_tail_coverage_policy/1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TENSOR_KEYS = ("calib_ids", "topk_ids", "topk_lps")
_TOKENIZER_FILENAMES = (
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


class TeacherPayloadError(ValueError):
    """The teacher payload or one of its value-bearing contracts is invalid."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode strict canonical JSON used by every digest in this contract."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeacherPayloadError("value is not strict canonical JSON") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TeacherPayloadError(f"{where} is not a lowercase SHA256")
    return value


def topk_coverage_policy() -> dict[str, object]:
    """Return the closed acceptance policy carried by metadata/evidence."""
    return {
        "schema": TOPK_COVERAGE_POLICY_SCHEMA,
        "top_k": PROMPT_TOP_K,
        "minimum_probability_mass_per_position": TOPK_MINIMUM_COVERAGE,
        "maximum_probability_mass": 1.0,
        "probability_mass_absolute_tolerance": (
            TOPK_PROBABILITY_MASS_ABS_TOLERANCE
        ),
        "maximum_declared_tail_mass_per_position": (
            1.0 - TOPK_MINIMUM_COVERAGE
        ),
        "tail_bucket": True,
    }


def topk_coverage_summary(
    topk_ids: torch.Tensor,
    topk_lps: torch.Tensor,
    *,
    vocab_size: int,
) -> dict[str, object]:
    """Validate every top-K row and derive coverage from tensor values.

    The returned values are computed in float64 from the serialized float32
    log probabilities.  No caller-provided summary participates in this
    calculation.
    """
    if topk_ids.dtype != torch.int32 or topk_lps.dtype != torch.float32:
        raise TeacherPayloadError("teacher top-k tensor dtypes are invalid")
    if topk_ids.shape != topk_lps.shape or topk_ids.ndim != 3:
        raise TeacherPayloadError("teacher top-k tensor shapes differ")
    if not torch.isfinite(topk_lps).all():
        raise TeacherPayloadError("teacher topk_lps contains non-finite values")
    if int(topk_ids.min()) < 0 or int(topk_ids.max()) >= vocab_size:
        raise TeacherPayloadError(
            "teacher top-k token ids are non-finite or out of range"
        )

    # Sorting ids, rather than relying on their probability order, makes the
    # uniqueness check exact and vectorized over all 4,088 release rows.
    ids_by_value = torch.sort(topk_ids, dim=-1).values
    if bool((ids_by_value[..., 1:] == ids_by_value[..., :-1]).any()):
        raise TeacherPayloadError("teacher top-k rows contain duplicate token ids")
    if bool((topk_lps[..., 1:] > topk_lps[..., :-1]).any()):
        raise TeacherPayloadError(
            "teacher top-k log probabilities are not nonincreasing"
        )
    if bool((topk_lps > 0.0).any()):
        raise TeacherPayloadError("teacher log probabilities exceed zero")

    coverage = topk_lps.to(dtype=torch.float64).exp().sum(dim=-1)
    if not torch.isfinite(coverage).all():
        raise TeacherPayloadError("teacher top-k probability mass is non-finite")
    coverage_mean = float(coverage.mean().item())
    coverage_min = float(coverage.min().item())
    coverage_max = float(coverage.max().item())
    if coverage_max > 1.0 + TOPK_PROBABILITY_MASS_ABS_TOLERANCE:
        raise TeacherPayloadError(
            "teacher top-k probability mass exceeds one beyond the "
            f"{TOPK_PROBABILITY_MASS_ABS_TOLERANCE:g} absolute tolerance"
        )
    if coverage_min < TOPK_MINIMUM_COVERAGE:
        # Report the shortfall, not just its existence. This refusal costs a
        # full streamed teacher pass over the source model, and the payload is
        # not written when it fires, so a bare "below 0.90" leaves the operator
        # with no way to tell a near-miss (raise K) from a genuinely flat
        # predictive distribution (the top-K formulation does not fit this
        # model) without paying for the pass again.
        below = coverage < TOPK_MINIMUM_COVERAGE
        n_below = int(below.sum().item())
        n_total = int(coverage.numel())
        quantiles = torch.tensor([0.001, 0.01, 0.05, 0.50], dtype=torch.float64)
        q = torch.quantile(coverage.flatten(), quantiles).tolist()
        raise TeacherPayloadError(
            "teacher top-k coverage falls below the declared "
            f"{TOPK_MINIMUM_COVERAGE:.2f} per-position minimum: "
            f"K={int(topk_ids.shape[-1])} over vocab={vocab_size}; "
            f"min={coverage_min:.4f} mean={coverage_mean:.4f}; "
            f"{n_below}/{n_total} positions short "
            f"({100.0 * n_below / max(n_total, 1):.2f}%); "
            f"coverage quantiles p0.1={q[0]:.4f} p1={q[1]:.4f} "
            f"p5={q[2]:.4f} p50={q[3]:.4f}"
        )
    return {
        "topk_coverage_mean": coverage_mean,
        "topk_coverage_min": coverage_min,
        "topk_coverage_policy": topk_coverage_policy(),
    }


def safe_load_torch_payload(path: str | os.PathLike) -> object:
    """Deserialize tensors/primitives without permitting pickle execution."""
    payload_path = Path(path).resolve(strict=True)
    try:
        return torch.load(
            payload_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise TeacherPayloadError(
            "could not safely load teacher tensor payload"
        ) from exc


def tensor_descriptor(value: torch.Tensor) -> dict[str, object]:
    """Hash a tensor's contiguous CPU storage, independent of torch.save."""
    if not isinstance(value, torch.Tensor):
        raise TeacherPayloadError("tensor descriptor requires a torch.Tensor")
    tensor = value.detach().to("cpu").contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(dimension) for dimension in tensor.shape],
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def compact_source_model_identity(identity: Mapping[str, Any]) -> dict[str, object]:
    """Project a full streamed identity to the artifact provenance schema."""
    shards = identity.get("shards")
    checkpoint_weight_map = identity.get("checkpoint_weight_map")
    if not isinstance(shards, list) or not shards:
        raise TeacherPayloadError("source model identity has no checkpoint shards")
    if not isinstance(checkpoint_weight_map, Mapping) or not checkpoint_weight_map:
        raise TeacherPayloadError("source model identity has no checkpoint tensor map")
    compact = {
        "schema": identity.get("schema"),
        "content_sha256": identity.get("content_sha256"),
        "resolved_commit": identity.get("resolved_commit"),
        "checkpoint_shards": len(shards),
        "checkpoint_tensors": len(checkpoint_weight_map),
    }
    _require_sha256(compact["content_sha256"], where="source content_sha256")
    return compact


def tokenizer_identity(model_dir: str | os.PathLike) -> dict[str, object]:
    """Bind the exact local files that define tokenization for this lane."""
    root = Path(model_dir).resolve(strict=True)
    files: dict[str, dict[str, object]] = {}
    for name in _TOKENIZER_FILENAMES:
        path = root / name
        if path.is_file():
            files[name] = {
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
    if not files:
        raise TeacherPayloadError(f"no tokenizer files found under {root}")
    value_bearing = {"files": files}
    return {
        "schema": TOKENIZER_IDENTITY_SCHEMA,
        "content_sha256": canonical_sha256(value_bearing),
        **value_bearing,
    }


def tensor_semantic_projection(payload: Mapping[str, Any]) -> dict[str, object]:
    """Replace tensor bodies with byte descriptors for a stable payload hash."""
    expected = set(payload) - {"payload_semantic_sha256"}
    missing = set(_TENSOR_KEYS) - expected
    if missing:
        raise TeacherPayloadError(
            f"teacher payload misses semantic tensors: {sorted(missing)}"
        )
    projection: dict[str, object] = {}
    for key in sorted(expected):
        value = payload[key]
        projection[key] = tensor_descriptor(value) if key in _TENSOR_KEYS else value
    return projection


def payload_semantic_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(tensor_semantic_projection(payload))


def build_calibration_contract(
    *,
    dataset_fingerprint: str,
    corpus_sha256: str,
    tokenizer: Mapping[str, Any],
    starts: list[int],
    total_tokens: int,
    calib_ids: torch.Tensor,
) -> dict[str, object]:
    """Create the closed WikiText window/tokenization contract."""
    if len(starts) != N_SAMPLES or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in starts
    ):
        raise TeacherPayloadError("calibration starts must be eight nonnegative ints")
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
        raise TeacherPayloadError("dataset fingerprint is missing")
    _require_sha256(corpus_sha256, where="calibration corpus_sha256")
    tokenizer_sha = tokenizer.get("content_sha256")
    _require_sha256(tokenizer_sha, where="tokenizer content_sha256")
    contract = {
        "schema": CALIBRATION_SCHEMA,
        "dataset": {
            "name": WIKITEXT_DATASET,
            "config": WIKITEXT_CONFIG,
            "split": WIKITEXT_SPLIT,
            "revision": WIKITEXT_REVISION,
            "fingerprint": dataset_fingerprint,
            "corpus_sha256": corpus_sha256,
        },
        "corpus_construction": {
            "row_filter": "include iff bool(text.strip()); preserve text verbatim",
            "join_separator": "\n\n",
            "normalization": "none",
        },
        "tokenizer": {
            "identity_sha256": tokenizer_sha,
            "trust_remote_code": True,
            "add_special_tokens": False,
        },
        "window_seed": WINDOW_SEED,
        "sampler": "python.random.Random(seed).sample(range(max_start), n_samples)/v1",
        "n_samples": N_SAMPLES,
        "seqlen": SEQLEN,
        "starts": list(starts),
        "total_tokens": int(total_tokens),
        "calib_ids_sha256": tensor_descriptor(calib_ids)["sha256"],
        "scoring": {
            "positions": "all",
            "prompt_top_k": PROMPT_TOP_K,
            "logprob_dtype": "float32",
            "tail_bucket": True,
        },
    }
    validate_calibration_contract(contract, calib_ids=calib_ids)
    return contract


def validate_calibration_contract(
    contract: object,
    *,
    calib_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise TeacherPayloadError("calibration contract is not an object")
    expected_keys = {
        "schema", "dataset", "corpus_construction", "tokenizer",
        "window_seed", "sampler", "n_samples", "seqlen", "starts",
        "total_tokens", "calib_ids_sha256", "scoring",
    }
    if set(contract) != expected_keys:
        raise TeacherPayloadError("calibration contract fields are not closed")
    if contract.get("schema") != CALIBRATION_SCHEMA:
        raise TeacherPayloadError("unsupported calibration contract schema")
    expected_dataset = {
        "name": WIKITEXT_DATASET,
        "config": WIKITEXT_CONFIG,
        "split": WIKITEXT_SPLIT,
        "revision": WIKITEXT_REVISION,
    }
    dataset = contract.get("dataset")
    if not isinstance(dataset, Mapping) or any(
        dataset.get(key) != value for key, value in expected_dataset.items()
    ) or set(dataset) != {*expected_dataset, "fingerprint", "corpus_sha256"}:
        raise TeacherPayloadError("calibration dataset identity differs")
    if not isinstance(dataset.get("fingerprint"), str) or not dataset.get(
        "fingerprint"
    ):
        raise TeacherPayloadError("calibration dataset fingerprint is missing")
    _require_sha256(dataset.get("corpus_sha256"), where="calibration corpus_sha256")
    if contract.get("corpus_construction") != {
        "row_filter": "include iff bool(text.strip()); preserve text verbatim",
        "join_separator": "\n\n",
        "normalization": "none",
    }:
        raise TeacherPayloadError("calibration corpus construction differs")
    tokenizer = contract.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or set(tokenizer) != {
        "identity_sha256", "trust_remote_code", "add_special_tokens"
    } or tokenizer.get("trust_remote_code") is not True or tokenizer.get(
        "add_special_tokens"
    ) is not False:
        raise TeacherPayloadError("calibration tokenizer contract differs")
    _require_sha256(tokenizer.get("identity_sha256"), where="tokenizer identity")
    if (
        contract.get("window_seed") != WINDOW_SEED
        or contract.get("sampler")
        != "python.random.Random(seed).sample(range(max_start), n_samples)/v1"
        or contract.get("n_samples") != N_SAMPLES
        or contract.get("seqlen") != SEQLEN
    ):
        raise TeacherPayloadError("calibration window contract differs")
    starts = contract.get("starts")
    if not isinstance(starts, list) or len(starts) != N_SAMPLES or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in starts
    ):
        raise TeacherPayloadError("calibration starts are malformed")
    if isinstance(contract.get("total_tokens"), bool) or not isinstance(
        contract.get("total_tokens"), int
    ) or int(contract["total_tokens"]) < SEQLEN + 1:
        raise TeacherPayloadError("calibration total token count is invalid")
    _require_sha256(contract.get("calib_ids_sha256"), where="calib_ids_sha256")
    if contract.get("scoring") != {
        "positions": "all",
        "prompt_top_k": PROMPT_TOP_K,
        "logprob_dtype": "float32",
        "tail_bucket": True,
    }:
        raise TeacherPayloadError("calibration scoring contract differs")
    if calib_ids is not None:
        if list(calib_ids.shape) != [N_SAMPLES, SEQLEN] or calib_ids.dtype != torch.long:
            raise TeacherPayloadError("calib_ids shape/dtype differs from contract")
        if tensor_descriptor(calib_ids)["sha256"] != contract.get(
            "calib_ids_sha256"
        ):
            raise TeacherPayloadError("calib_ids bytes differ from contract")
    return dict(contract)


def validate_teacher_payload(payload: object) -> dict[str, Any]:
    """Fail closed over every payload field and tensor byte descriptor."""
    if not isinstance(payload, Mapping):
        raise TeacherPayloadError("teacher payload is not an object")
    expected_keys = {
        "schema", "score_positions", "prompt_top_k", "topk_ids", "topk_lps",
        "calib_ids", "starts", "model", "n_samples", "seqlen", "vocab_size",
        "source_model_identity", "source_model", "source_model_identity_sha256",
        "calibration_contract", "calibration_contract_sha256",
        "payload_semantic_sha256",
    }
    if set(payload) != expected_keys:
        raise TeacherPayloadError("teacher payload fields are not closed")
    if payload.get("schema") != TEACHER_PAYLOAD_SCHEMA:
        raise TeacherPayloadError("unsupported teacher payload schema")
    if (
        payload.get("score_positions") != "all"
        or payload.get("prompt_top_k") != PROMPT_TOP_K
        or payload.get("n_samples") != N_SAMPLES
        or payload.get("seqlen") != SEQLEN
    ):
        raise TeacherPayloadError("teacher scoring dimensions differ")
    vocab_size = payload.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= PROMPT_TOP_K:
        raise TeacherPayloadError("teacher vocab_size is invalid")
    calib_ids = payload.get("calib_ids")
    topk_ids = payload.get("topk_ids")
    topk_lps = payload.get("topk_lps")
    if not isinstance(calib_ids, torch.Tensor) or calib_ids.dtype != torch.long or list(
        calib_ids.shape
    ) != [N_SAMPLES, SEQLEN]:
        raise TeacherPayloadError("teacher calib_ids shape/dtype is invalid")
    expected_topk_shape = [N_SAMPLES, SEQLEN - 1, PROMPT_TOP_K]
    if not isinstance(topk_ids, torch.Tensor) or topk_ids.dtype != torch.int32 or list(
        topk_ids.shape
    ) != expected_topk_shape:
        raise TeacherPayloadError("teacher topk_ids shape/dtype is invalid")
    if not isinstance(topk_lps, torch.Tensor) or topk_lps.dtype != torch.float32 or list(
        topk_lps.shape
    ) != expected_topk_shape:
        raise TeacherPayloadError("teacher topk_lps shape/dtype is invalid")
    topk_coverage_summary(topk_ids, topk_lps, vocab_size=vocab_size)
    identity = payload.get("source_model_identity")
    try:
        from prismaquant.cost_streaming import validate_streamed_model_identity

        full_identity = validate_streamed_model_identity(
            identity, where="full KL teacher payload"
        )
    except Exception as exc:
        raise TeacherPayloadError("teacher source-model identity is invalid") from exc
    expected_identity_sha = canonical_sha256(full_identity)
    if payload.get("source_model_identity_sha256") != expected_identity_sha:
        raise TeacherPayloadError("teacher full source identity digest differs")
    if payload.get("source_model") != compact_source_model_identity(full_identity):
        raise TeacherPayloadError("teacher compact source identity differs")
    contract = validate_calibration_contract(
        payload.get("calibration_contract"), calib_ids=calib_ids
    )
    if payload.get("calibration_contract_sha256") != canonical_sha256(contract):
        raise TeacherPayloadError("teacher calibration contract digest differs")
    if payload.get("starts") != contract.get("starts"):
        raise TeacherPayloadError("teacher starts differ from calibration contract")
    observed_semantic = payload_semantic_sha256(payload)
    if payload.get("payload_semantic_sha256") != observed_semantic:
        raise TeacherPayloadError("teacher semantic payload digest differs")
    return dict(payload)


def teacher_meta(
    *,
    payload_path: str | os.PathLike,
    elapsed_s: float,
) -> dict[str, object]:
    """Construct a sidecar strictly from the serialized tensor payload."""
    path = Path(payload_path).resolve(strict=True)
    validated = validate_teacher_payload(safe_load_torch_payload(path))
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise TeacherPayloadError("teacher elapsed time is invalid")
    coverage = topk_coverage_summary(
        validated["topk_ids"],
        validated["topk_lps"],
        vocab_size=int(validated["vocab_size"]),
    )
    return {
        "schema": TEACHER_META_SCHEMA,
        "payload": str(path),
        "payload_sha256": file_sha256(path),
        "payload_bytes": int(path.stat().st_size),
        "payload_semantic_sha256": validated["payload_semantic_sha256"],
        "source_model": validated["source_model"],
        "source_model_identity_sha256": validated["source_model_identity_sha256"],
        "calibration_contract": validated["calibration_contract"],
        "calibration_contract_sha256": validated["calibration_contract_sha256"],
        "tensor_descriptors": {
            key: tensor_descriptor(validated[key]) for key in _TENSOR_KEYS
        },
        "teacher_shape": list(validated["topk_lps"].shape),
        **coverage,
        "elapsed_s": float(elapsed_s),
    }


def atomic_torch_save(payload: object, path: str | os.PathLike) -> None:
    """Publish a torch payload by rename, never as a partial final file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise TeacherPayloadError(f"temporary payload already exists: {temporary}")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_write(payload: object, path: str | os.PathLike) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise TeacherPayloadError(f"temporary metadata already exists: {temporary}")
    data = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_teacher_evidence(
    payload_path: str | os.PathLike,
    meta_path: str | os.PathLike,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Load/replay teacher bytes and return shipcard-sized evidence."""
    payload_file = Path(payload_path).resolve(strict=True)
    meta_file = Path(meta_path).resolve(strict=True)
    try:
        payload = safe_load_torch_payload(payload_file)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except TeacherPayloadError:
        raise
    except Exception as exc:
        raise TeacherPayloadError("could not load teacher payload evidence") from exc
    validated = validate_teacher_payload(payload)
    if not isinstance(meta, Mapping) or meta.get("schema") != TEACHER_META_SCHEMA:
        raise TeacherPayloadError("unsupported teacher metadata schema")
    if meta.get("payload_sha256") != file_sha256(payload_file) or meta.get(
        "payload_bytes"
    ) != payload_file.stat().st_size:
        raise TeacherPayloadError("teacher serialized payload bytes differ")
    expected_meta_fields = {
        "schema", "payload", "payload_sha256", "payload_bytes",
        "payload_semantic_sha256", "source_model",
        "source_model_identity_sha256", "calibration_contract",
        "calibration_contract_sha256", "tensor_descriptors", "teacher_shape",
        "topk_coverage_mean", "topk_coverage_min", "topk_coverage_policy",
        "elapsed_s",
    }
    if set(meta) != expected_meta_fields:
        raise TeacherPayloadError("teacher metadata fields are not closed")
    comparisons = {
        "payload_semantic_sha256": validated["payload_semantic_sha256"],
        "source_model": validated["source_model"],
        "source_model_identity_sha256": validated["source_model_identity_sha256"],
        "calibration_contract": validated["calibration_contract"],
        "calibration_contract_sha256": validated["calibration_contract_sha256"],
        "tensor_descriptors": {
            key: tensor_descriptor(validated[key]) for key in _TENSOR_KEYS
        },
        "teacher_shape": list(validated["topk_lps"].shape),
        **topk_coverage_summary(
            validated["topk_ids"],
            validated["topk_lps"],
            vocab_size=int(validated["vocab_size"]),
        ),
    }
    if any(meta.get(key) != value for key, value in comparisons.items()):
        raise TeacherPayloadError("teacher metadata differs from payload semantics")
    elapsed_s = meta.get("elapsed_s")
    if (
        isinstance(elapsed_s, bool)
        or not isinstance(elapsed_s, (int, float))
        or not math.isfinite(float(elapsed_s))
        or float(elapsed_s) < 0.0
    ):
        raise TeacherPayloadError("teacher metadata elapsed time is invalid")
    evidence = {
        "schema": TEACHER_EVIDENCE_SCHEMA,
        "payload_sha256": meta["payload_sha256"],
        "payload_bytes": meta["payload_bytes"],
        "payload_semantic_sha256": meta["payload_semantic_sha256"],
        "meta_sha256": file_sha256(meta_file),
        "source_model": meta["source_model"],
        "source_model_identity_sha256": meta["source_model_identity_sha256"],
        "calibration_contract": meta["calibration_contract"],
        "calibration_contract_sha256": meta["calibration_contract_sha256"],
        "topk_coverage_mean": meta["topk_coverage_mean"],
        "topk_coverage_min": meta["topk_coverage_min"],
        "topk_coverage_policy": meta["topk_coverage_policy"],
    }
    return validated, evidence


__all__ = [
    "CALIBRATION_SCHEMA",
    "EXPECTED_POSITIONS",
    "N_SAMPLES",
    "PROMPT_TOP_K",
    "SEQLEN",
    "TEACHER_EVIDENCE_SCHEMA",
    "TEACHER_META_SCHEMA",
    "TEACHER_PAYLOAD_SCHEMA",
    "TeacherPayloadError",
    "TOPK_COVERAGE_POLICY_SCHEMA",
    "TOPK_MINIMUM_COVERAGE",
    "TOPK_PROBABILITY_MASS_ABS_TOLERANCE",
    "WIKITEXT_CONFIG",
    "WIKITEXT_DATASET",
    "WIKITEXT_REVISION",
    "WIKITEXT_SPLIT",
    "WINDOW_SEED",
    "atomic_json_write",
    "atomic_torch_save",
    "build_calibration_contract",
    "canonical_sha256",
    "compact_source_model_identity",
    "file_sha256",
    "load_teacher_evidence",
    "payload_semantic_sha256",
    "safe_load_torch_payload",
    "teacher_meta",
    "tensor_descriptor",
    "topk_coverage_policy",
    "topk_coverage_summary",
    "tokenizer_identity",
    "validate_calibration_contract",
    "validate_teacher_payload",
]
