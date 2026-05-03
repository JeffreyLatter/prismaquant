#!/usr/bin/env python3
"""Measure full-vocab next-token KL between two vLLM-loadable artifacts."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _load_wikitext_calibration(
    tokenizer,
    *,
    cache_dir: str,
    n_samples: int,
    seqlen: int,
) -> tuple[list[list[int]], list[int], int]:
    ds = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
        cache_dir=cache_dir,
    )
    text = "\n\n".join(row["text"] for row in ds if row.get("text", "").strip())
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    if int(ids.numel()) < seqlen + 1:
        raise RuntimeError(f"not enough calibration tokens: {int(ids.numel())}")
    max_start = int(ids.numel()) - int(seqlen)
    rng = random.Random(42)
    if max_start >= n_samples:
        starts = rng.sample(range(max_start), n_samples)
    else:
        starts = [
            min(max_start, int(i * max_start / max(n_samples, 1)))
            for i in range(n_samples)
        ]
    calib = [ids[s : s + seqlen].tolist() for s in starts]
    return calib, starts, int(ids.numel())


def _load_llm(args, *, max_model_len: int) -> LLM:
    kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": 1,
        "max_logprobs": args.max_logprobs,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
    }
    if args.quantization:
        kwargs["quantization"] = args.quantization
    return LLM(**kwargs)


def _logprob_vector(logprobs, *, vocab_size: int) -> torch.Tensor:
    vec = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
    for key, value in logprobs.items():
        token_id = int(key)
        if token_id >= vocab_size:
            continue
        logprob = getattr(value, "logprob", None)
        if logprob is None and isinstance(value, dict):
            logprob = value.get("logprob")
        if logprob is None and isinstance(value, (tuple, list)):
            logprob = value[0]
        if logprob is None:
            logprob = value
        vec[token_id] = float(logprob)
    missing = int(torch.isneginf(vec).sum().item())
    if missing:
        raise RuntimeError(
            f"vLLM returned {vocab_size - missing}/{vocab_size} logprobs; "
            "full-vocab KL requires logprobs=-1 support"
        )
    return vec


def _measure_logprobs(
    llm: LLM,
    prompts: list[list[int]],
    *,
    vocab_size: int,
) -> torch.Tensor:
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=-1,
        detokenize=False,
    )
    rows = []
    for index, prompt_ids in enumerate(prompts, 1):
        start = time.monotonic()
        output = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            params,
            use_tqdm=False,
        )[0]
        logprobs = output.outputs[0].logprobs[0]
        rows.append(_logprob_vector(logprobs, vocab_size=vocab_size))
        print(
            f"[kl] sample {index}/{len(prompts)} "
            f"logprobs={len(logprobs)} wall={time.monotonic() - start:.2f}s",
            flush=True,
        )
    return torch.stack(rows, dim=0).contiguous()


def _teacher(args) -> int:
    started = time.monotonic()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts, starts, total_tokens = _load_wikitext_calibration(
        tokenizer,
        cache_dir=args.dataset_cache_dir,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
    )
    print(
        f"[kl] teacher model={args.model} n={args.n_samples} "
        f"seqlen={args.seqlen} total_tokens={total_tokens}",
        flush=True,
    )
    llm = _load_llm(args, max_model_len=args.seqlen + 16)
    hf_config = llm.llm_engine.model_config.hf_config
    vocab_size = max(
        int(getattr(hf_config, "vocab_size", len(tokenizer))),
        int(args.max_logprobs),
    )
    logprobs = _measure_logprobs(llm, prompts, vocab_size=vocab_size)
    payload = {
        "teacher_logprobs": logprobs,
        "calib_ids": torch.tensor(prompts, dtype=torch.long),
        "starts": starts,
        "model": args.model,
        "n_samples": int(args.n_samples),
        "seqlen": int(args.seqlen),
        "vocab_size": int(vocab_size),
    }
    torch.save(payload, output)
    meta = {
        "mode": "teacher",
        "model": args.model,
        "output": str(output),
        "n_samples": int(args.n_samples),
        "seqlen": int(args.seqlen),
        "starts": starts,
        "total_tokens": total_tokens,
        "vocab_size": int(vocab_size),
        "teacher_shape": list(logprobs.shape),
        "elapsed_s": time.monotonic() - started,
    }
    Path(args.meta_output).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return 0


def _student(args) -> int:
    started = time.monotonic()
    payload = torch.load(args.teacher_payload, map_location="cpu")
    teacher = payload["teacher_logprobs"].float()
    prompts = payload["calib_ids"].tolist()
    vocab_size = int(payload["vocab_size"])
    print(
        f"[kl] student model={args.model} n={len(prompts)} "
        f"seqlen={int(payload['seqlen'])} vocab={vocab_size}",
        flush=True,
    )
    llm = _load_llm(args, max_model_len=int(payload["seqlen"]) + 16)
    student = _measure_logprobs(llm, prompts, vocab_size=vocab_size)
    teacher_probs = teacher.exp()
    per_sample = (teacher_probs * (teacher - student)).sum(dim=-1)
    if not torch.isfinite(per_sample).all():
        raise RuntimeError(f"non-finite KL values: {per_sample.tolist()}")
    result = {
        "mode": "student",
        "model": args.model,
        "teacher_model": payload.get("model"),
        "teacher_payload": str(args.teacher_payload),
        "quantization": args.quantization,
        "n_samples": len(prompts),
        "seqlen": int(payload["seqlen"]),
        "vocab_size": vocab_size,
        "kl_mean": float(per_sample.mean().item()),
        "kl_min": float(per_sample.min().item()),
        "kl_max": float(per_sample.max().item()),
        "kl_per_sample": [float(x) for x in per_sample.tolist()],
        "elapsed_s": time.monotonic() - started,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["teacher", "student"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-output", default="teacher_meta.json")
    parser.add_argument("--teacher-payload")
    parser.add_argument("--dataset-cache-dir", default="/hfcache/datasets")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--max-logprobs", type=int, default=248320)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    if args.mode == "student" and not args.teacher_payload:
        parser.error("--teacher-payload is required in student mode")
    if args.mode == "teacher":
        return _teacher(args)
    return _student(args)


if __name__ == "__main__":
    raise SystemExit(main())
