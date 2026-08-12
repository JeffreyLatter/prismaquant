#!/usr/bin/env python3
"""Build the DSv4 all-position BF16 KL teacher with one streamed GPU model.

This is the one-Spark source-teacher path.  It deliberately extends the
repository's existing StreamingContext instead of inventing a second offload
or residency mechanism.  The complete BF16 source never has to be resident at
once; one decoder layer is installed at a time and logits are reduced to the
fixed top-1024 gold support on GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # package mode
    from .full_kl_teacher_payload import (
        N_SAMPLES,
        PROMPT_TOP_K,
        SEQLEN,
        TEACHER_PAYLOAD_SCHEMA,
        WIKITEXT_CONFIG,
        WIKITEXT_DATASET,
        WIKITEXT_REVISION,
        WIKITEXT_SPLIT,
        WINDOW_SEED,
        atomic_json_write,
        atomic_torch_save,
        build_calibration_contract,
        canonical_sha256,
        compact_source_model_identity,
        payload_semantic_sha256,
        teacher_meta,
        tokenizer_identity,
        validate_teacher_payload,
    )
except ImportError:  # direct script mode
    from full_kl_teacher_payload import (  # type: ignore
        N_SAMPLES,
        PROMPT_TOP_K,
        SEQLEN,
        TEACHER_PAYLOAD_SCHEMA,
        WIKITEXT_CONFIG,
        WIKITEXT_DATASET,
        WIKITEXT_REVISION,
        WIKITEXT_SPLIT,
        WINDOW_SEED,
        atomic_json_write,
        atomic_torch_save,
        build_calibration_contract,
        canonical_sha256,
        compact_source_model_identity,
        payload_semantic_sha256,
        teacher_meta,
        tokenizer_identity,
        validate_teacher_payload,
    )

try:  # package mode
    from .dsv4_wikitext_inputs import load_dsv4_wikitext_inputs
except ImportError:  # direct script mode
    from dsv4_wikitext_inputs import load_dsv4_wikitext_inputs  # type: ignore


def _tokenizer_vocab_size(tokenizer) -> int:
    """Return the output-vocabulary cardinality, including added tokens."""
    size = len(tokenizer)
    if isinstance(size, bool) or not isinstance(size, int) or size <= PROMPT_TOP_K:
        raise RuntimeError(f"invalid tokenizer vocabulary size: {size!r}")
    return size


def _topk_all_positions(
    logits: torch.Tensor,
    *,
    top_k: int = PROMPT_TOP_K,
    chunk_rows: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce causal logits to fp32 all-position top-K distributions on GPU."""
    if logits.ndim != 3:
        raise RuntimeError(f"teacher logits must be rank 3, got {logits.shape}")
    batch, sequence, vocab = (int(value) for value in logits.shape)
    if batch != N_SAMPLES or sequence != SEQLEN:
        raise RuntimeError(
            f"teacher logits must be [{N_SAMPLES},{SEQLEN},V], got "
            f"{list(logits.shape)}"
        )
    if top_k != PROMPT_TOP_K or vocab <= top_k:
        raise RuntimeError("teacher top-K/vocabulary differs from gold contract")
    if chunk_rows <= 0:
        raise RuntimeError("--logits-chunk-rows must be positive")

    # prompt_logprobs[position] predicts token[position] from the prefix ending
    # at position-1, so the matching HF causal row is logits[position-1].
    rows = logits[:, :-1, :].reshape(batch * (sequence - 1), vocab)
    ids_cpu = torch.empty((rows.size(0), top_k), dtype=torch.int32)
    lps_cpu = torch.empty((rows.size(0), top_k), dtype=torch.float32)
    for first in range(0, int(rows.size(0)), chunk_rows):
        last = min(first + chunk_rows, int(rows.size(0)))
        log_probs = torch.log_softmax(rows[first:last].float(), dim=-1)
        values, indices = torch.topk(
            log_probs, k=top_k, dim=-1, largest=True, sorted=True
        )
        ids_cpu[first:last].copy_(
            indices.to(device="cpu", dtype=torch.int32)
        )
        lps_cpu[first:last].copy_(
            values.to(device="cpu", dtype=torch.float32)
        )
        print(
            f"[streamed-teacher] reduced rows {last}/{rows.size(0)}",
            flush=True,
        )
        del log_probs, values, indices
    return (
        ids_cpu.reshape(batch, sequence - 1, top_k).contiguous(),
        lps_cpu.reshape(batch, sequence - 1, top_k).contiguous(),
    )


def _build_payload(args: argparse.Namespace) -> dict:
    from prismaquant.cost_streaming import (
        build_streamed_causal_lm,
        validate_cached_streamed_model_identity,
    )
    from prismaquant.gpu_guard import require_cuda_hot_path
    from prismaquant.model_profiles import detect_profile

    device = require_cuda_hot_path("build_streamed_full_kl_teacher", "cuda")
    model_path = Path(args.model).resolve(strict=True)
    if not model_path.is_dir():
        raise RuntimeError(f"source model is not a directory: {model_path}")
    tokenizer_attestation = tokenizer_identity(model_path)
    # Reject an absent/tampered 156-KiB token input before walking the much
    # larger checkpoint identity or constructing any model/tokenizer runtime.
    wikitext_inputs = load_dsv4_wikitext_inputs(
        args.wikitext_inputs,
        expected_tokenizer_identity=tokenizer_attestation,
    )
    full_kl_inputs = wikitext_inputs["full_kl"]
    calibration = torch.tensor(
        full_kl_inputs["token_ids"], dtype=torch.long
    ).contiguous()
    starts = list(full_kl_inputs["selection"]["starts"])
    dataset_evidence = full_kl_inputs["dataset"]
    calibration_contract = build_calibration_contract(
        dataset_fingerprint=dataset_evidence["fingerprint"],
        corpus_sha256=dataset_evidence["corpus_sha256"],
        tokenizer=tokenizer_attestation,
        starts=starts,
        total_tokens=dataset_evidence["total_tokens"],
        calib_ids=calibration,
    )
    full_identity = validate_cached_streamed_model_identity(
        model_path,
        args.identity_cache,
        require_complete_checkpoint=True,
    )
    compact_identity = compact_source_model_identity(full_identity)
    # Detection bootstraps the vendored DSv4 config with Transformers.  It
    # must precede AutoTokenizer: the source is local and intentionally has no
    # remote-code Python files of its own.
    profile = detect_profile(str(model_path))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    print(
        "[streamed-teacher] "
        f"source={compact_identity['content_sha256']} "
        f"calibration={canonical_sha256(calibration_contract)} "
        f"shape={list(calibration.shape)}",
        flush=True,
    )

    runner = build_streamed_causal_lm(
        str(model_path),
        device=device,
        dtype=torch.bfloat16,
        offload_folder=str(Path(args.offload_folder).resolve()),
        profile=profile,
        cache_headroom_gb=float(args.cache_headroom_gb),
        max_cache_slots=1,
        prefetch_workers=1,
        prefetch_lookahead=0,
    )
    try:
        if runner.context.max_cache_slots != 1 or runner.prefetch_lookahead != 0:
            raise RuntimeError(
                "streamed teacher source-cache policy is not fail-closed "
                f"(slots={runner.context.max_cache_slots}, "
                f"lookahead={runner.prefetch_lookahead})"
            )
        with torch.inference_mode():
            output = runner(calibration.to(device, non_blocking=True))
            logits = output.logits.detach()
            vocab_size = _tokenizer_vocab_size(tokenizer)
            if int(logits.shape[-1]) != vocab_size:
                raise RuntimeError(
                    "source logits vocabulary differs from tokenizer vocabulary"
                )
            topk_ids, topk_lps = _topk_all_positions(
                logits,
                chunk_rows=int(args.logits_chunk_rows),
            )
            del output, logits
    finally:
        runner.shutdown()

    payload: dict = {
        "schema": TEACHER_PAYLOAD_SCHEMA,
        "score_positions": "all",
        "prompt_top_k": PROMPT_TOP_K,
        "topk_ids": topk_ids,
        "topk_lps": topk_lps,
        "calib_ids": calibration,
        "starts": starts,
        "model": str(model_path),
        "n_samples": N_SAMPLES,
        "seqlen": SEQLEN,
        "vocab_size": _tokenizer_vocab_size(tokenizer),
        "source_model_identity": full_identity,
        "source_model": compact_identity,
        "source_model_identity_sha256": canonical_sha256(full_identity),
        "calibration_contract": calibration_contract,
        "calibration_contract_sha256": canonical_sha256(calibration_contract),
    }
    payload["payload_semantic_sha256"] = payload_semantic_sha256(payload)
    validate_teacher_payload(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--identity-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-output", required=True)
    parser.add_argument("--offload-folder", required=True)
    parser.add_argument(
        "--wikitext-inputs",
        required=True,
        help=(
            "offline payload from tools/prepare_dsv4_wikitext_inputs.py; "
            "the GPU teacher environment never imports datasets"
        ),
    )
    parser.add_argument("--cache-headroom-gb", type=float, default=100.0)
    parser.add_argument("--logits-chunk-rows", type=int, default=32)
    args = parser.parse_args()

    output = Path(args.output)
    meta_output = Path(args.meta_output)
    if output.exists() or meta_output.exists():
        parser.error(
            "refusing to overwrite an existing teacher payload or metadata file"
        )
    if output.resolve() == meta_output.resolve():
        parser.error("--output and --meta-output must be distinct")

    started = time.monotonic()
    payload = _build_payload(args)
    atomic_torch_save(payload, output)
    meta = teacher_meta(
        payload_path=output,
        elapsed_s=time.monotonic() - started,
    )
    atomic_json_write(meta, meta_output)
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
