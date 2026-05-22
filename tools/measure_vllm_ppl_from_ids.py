#!/usr/bin/env python3
"""Measure vLLM perplexity from a pre-tokenized torch payload.

This is useful when the vLLM environment should not also own dataset loading.
The payload must contain an ``ids`` tensor or list of token ids.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams


def _load_llm(args: argparse.Namespace) -> LLM:
    kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": int(args.seqlen) + 1,
        "max_num_seqs": 1,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
    }
    if args.language_model_only:
        kwargs["language_model_only"] = True
    if args.skip_mm_profiling:
        kwargs["skip_mm_profiling"] = True
    if args.quantization:
        kwargs["quantization"] = args.quantization
    return LLM(**kwargs)


def _payload_ids(payload: Any) -> list[int]:
    if isinstance(payload, dict):
        ids = payload["ids"]
    else:
        ids = payload
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    ids = [int(token_id) for token_id in ids]
    if len(ids) < 2:
        raise RuntimeError("token payload contains fewer than two ids")
    return ids


def _logprob_value(entry, token_id: int) -> float:
    if entry is None:
        raise KeyError(token_id)
    value = None
    if isinstance(entry, dict):
        value = entry.get(token_id)
        if value is None:
            value = entry.get(str(token_id))
    if value is None:
        raise KeyError(token_id)
    logprob = getattr(value, "logprob", None)
    if logprob is None and isinstance(value, dict):
        logprob = value.get("logprob")
    if logprob is None and isinstance(value, (tuple, list)):
        logprob = value[0]
    if logprob is None:
        logprob = value
    return float(logprob)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--skip-mm-profiling", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.ids, map_location="cpu")
    ids = _payload_ids(payload)
    llm = _load_llm(args)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
        detokenize=False,
    )

    nll = 0.0
    count = 0
    chunks = [
        ids[start : min(start + int(args.seqlen), len(ids))]
        for start in range(0, len(ids), int(args.seqlen))
        if len(ids[start : min(start + int(args.seqlen), len(ids))]) >= 2
    ]
    for index, chunk in enumerate(chunks, 1):
        t0 = time.monotonic()
        result = llm.generate(
            [{"prompt_token_ids": chunk}],
            sampling,
            use_tqdm=False,
        )[0]
        prompt_logprobs = result.prompt_logprobs
        if prompt_logprobs is None:
            raise RuntimeError("vLLM did not return prompt_logprobs")
        for pos in range(1, len(chunk)):
            nll -= _logprob_value(prompt_logprobs[pos], int(chunk[pos]))
            count += 1
        print(
            f"[ppl] chunk {index}/{len(chunks)} tokens={len(chunk)} "
            f"wall={time.monotonic() - t0:.2f}s",
            flush=True,
        )

    mean_nll = nll / max(count, 1)
    result = {
        "model": args.model,
        "ids": args.ids,
        "quantization": args.quantization,
        "n_tokens_requested": int(len(ids)),
        "n_tokens_scored": int(count),
        "seqlen": int(args.seqlen),
        "mean_nll": float(mean_nll),
        "ppl": float(math.exp(mean_nll)),
        "elapsed_s": float(time.monotonic() - started),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
