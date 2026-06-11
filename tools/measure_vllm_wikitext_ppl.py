#!/usr/bin/env python3
"""Measure WikiText perplexity for a vLLM-loadable model/artifact."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _load_ids(tokenizer, *, cache_dir: str, split: str, n_tokens: int) -> list[int]:
    ds = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=cache_dir,
    )
    text = "\n\n".join(row["text"] for row in ds if row.get("text", "").strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if int(ids.numel()) < 2:
        raise RuntimeError("WikiText tokenization produced fewer than two tokens")
    return ids[: int(n_tokens)].tolist()


def _load_llm(args) -> LLM:
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
    if args.quantization:
        kwargs["quantization"] = args.quantization
    return LLM(**kwargs)


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
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-cache-dir", default="/hfcache/datasets")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-tokens", type=int, default=8192)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = _load_ids(
        tokenizer,
        cache_dir=args.dataset_cache_dir,
        split=args.split,
        n_tokens=args.n_tokens,
    )
    llm = _load_llm(args)
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
        detokenize=False,
    )

    nll = 0.0
    count = 0
    chunk_nlls: list[float] = []
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
        chunk_nll = 0.0
        for pos in range(1, len(chunk)):
            chunk_nll -= _logprob_value(prompt_logprobs[pos], int(chunk[pos]))
            count += 1
        nll += chunk_nll
        chunk_nlls.append(chunk_nll / max(len(chunk) - 1, 1))
        print(
            f"[ppl] chunk {index}/{len(chunks)} tokens={len(chunk)} "
            f"wall={time.monotonic() - t0:.2f}s",
            flush=True,
        )

    mean_nll = nll / max(count, 1)
    result = {
        "model": args.model,
        "quantization": args.quantization,
        "split": args.split,
        "n_tokens_requested": int(args.n_tokens),
        "n_tokens_scored": int(count),
        "seqlen": int(args.seqlen),
        "mean_nll": float(mean_nll),
        "ppl": float(math.exp(mean_nll)),
        "per_chunk_mean_nll": [float(v) for v in chunk_nlls],
        "max_chunk_mean_nll": float(max(chunk_nlls)) if chunk_nlls else None,
        "elapsed_s": float(time.monotonic() - started),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
