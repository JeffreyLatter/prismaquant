#!/usr/bin/env python3
"""Measure full-vocab next-token KL between two vLLM-loadable artifacts."""

from __future__ import annotations

import argparse
import json
import math
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


def _resolve_vocab_size(llm: LLM, tokenizer) -> int:
    hf_config = llm.llm_engine.model_config.hf_config
    vocab_size = int(getattr(hf_config, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        vocab_size = int(len(tokenizer))
    if vocab_size <= 0:
        raise RuntimeError("could not resolve model vocabulary size")
    return vocab_size


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


def _measure_prompt_topk(
    llm: LLM,
    prompts: list[list[int]],
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-position scoring: per prompt position, the top-K token ids and
    logprobs of the model's next-token distribution (vLLM prompt_logprobs).

    Returns (ids, lps) shaped [n_prompts, P-1, K] (position 0 has no
    prediction). Full-vocab dicts at every position are infeasible (~620M
    Python objects per pass), so K bounds the support; the tail is handled
    as a single bucket by the caller. K=1024 covers ~all teacher mass at
    nearly every position and the truncation floor is shared across arms.
    """
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=int(top_k),
        detokenize=False,
    )
    all_ids, all_lps = [], []
    for index, prompt_ids in enumerate(prompts, 1):
        start = time.monotonic()
        output = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            params,
            use_tqdm=False,
        )[0]
        plps = output.prompt_logprobs
        if plps is None:
            raise RuntimeError("vLLM did not return prompt_logprobs")
        ids_rows, lps_rows = [], []
        for pos in range(1, len(prompt_ids)):
            d = plps[pos]
            items = [(int(k), float(getattr(v, "logprob", v)))
                     for k, v in d.items()]
            items.sort(key=lambda kv: kv[1], reverse=True)
            items = items[: int(top_k)]
            if len(items) < int(top_k):
                pad = int(top_k) - len(items)
                items = items + [(-1, float("-inf"))] * pad
            ids_rows.append([kv[0] for kv in items])
            lps_rows.append([kv[1] for kv in items])
        all_ids.append(torch.tensor(ids_rows, dtype=torch.int32))
        all_lps.append(torch.tensor(lps_rows, dtype=torch.float32))
        print(
            f"[kl] sample {index}/{len(prompts)} positions={len(ids_rows)} "
            f"top_k={top_k} wall={time.monotonic() - start:.2f}s",
            flush=True,
        )
    return torch.stack(all_ids), torch.stack(all_lps)


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
    vocab_size = _resolve_vocab_size(llm, tokenizer)
    if int(args.max_logprobs) < vocab_size:
        raise RuntimeError(
            f"--max-logprobs={args.max_logprobs} is smaller than "
            f"model vocab_size={vocab_size}; full-vocab KL requires "
            "requesting at least the full vocabulary"
        )
    if args.score_positions == "all":
        topk_ids, topk_lps = _measure_prompt_topk(
            llm, prompts, top_k=args.prompt_top_k)
        payload = {
            "score_positions": "all",
            "prompt_top_k": int(args.prompt_top_k),
            "topk_ids": topk_ids,
            "topk_lps": topk_lps.to(torch.float16),
            "calib_ids": torch.tensor(prompts, dtype=torch.long),
            "starts": starts,
            "model": args.model,
            "n_samples": int(args.n_samples),
            "seqlen": int(args.seqlen),
            "vocab_size": int(vocab_size),
        }
        torch.save(payload, output)
        cov = topk_lps.double().exp().sum(dim=-1)
        meta = {
            "mode": "teacher",
            "score_positions": "all",
            "prompt_top_k": int(args.prompt_top_k),
            "model": args.model,
            "output": str(output),
            "n_samples": int(args.n_samples),
            "seqlen": int(args.seqlen),
            "starts": starts,
            "total_tokens": total_tokens,
            "vocab_size": int(vocab_size),
            "teacher_shape": list(topk_lps.shape),
            "topk_coverage_mean": float(cov.mean()),
            "topk_coverage_min": float(cov.min()),
            "elapsed_s": time.monotonic() - started,
        }
        Path(args.meta_output).write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2), flush=True)
        return 0
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


def _student_all_positions(args, payload) -> int:
    started = time.monotonic()
    prompts = payload["calib_ids"].tolist()
    vocab_size = int(payload["vocab_size"])
    top_k = int(payload["prompt_top_k"])
    t_ids = payload["topk_ids"]                       # [n, P-1, K]
    t_lps = payload["topk_lps"].float()
    print(
        f"[kl] student(all-pos) model={args.model} n={len(prompts)} "
        f"seqlen={int(payload['seqlen'])} top_k={top_k}",
        flush=True,
    )
    llm = _load_llm(args, max_model_len=int(payload["seqlen"]) + 16)
    s_ids, s_lps = _measure_prompt_topk(llm, prompts, top_k=top_k)

    n, pm1, k = t_ids.shape
    kl_pos = torch.zeros((n, pm1), dtype=torch.float64)
    t_top1 = torch.zeros((n, pm1), dtype=torch.float64)
    for i in range(n):
        for j in range(pm1):
            tid = t_ids[i, j].tolist()
            tlp = t_lps[i, j].double()
            smap = {int(a): float(b)
                    for a, b in zip(s_ids[i, j].tolist(), s_lps[i, j].tolist())
                    if a >= 0}
            floor = min(smap.values())                # kl_ab.py convention
            q = torch.tensor([smap.get(t, floor) for t in tid],
                             dtype=torch.float64)
            p = tlp.exp()
            kl = float((p * (tlp - q)).sum())
            # tail bucket: remaining teacher mass vs remaining student mass
            pt = max(1.0 - float(p.sum()), 1e-12)
            qt = max(1.0 - float(q.exp().sum()), 1e-12)
            kl += pt * (math.log(pt) - math.log(qt))
            kl_pos[i, j] = kl
            t_top1[i, j] = float(p.max())
    confident = t_top1 > 0.5
    flat = kl_pos.flatten()
    result = {
        "mode": "student",
        "score_positions": "all",
        "prompt_top_k": top_k,
        "model": args.model,
        "teacher_model": payload.get("model"),
        "teacher_payload": str(args.teacher_payload),
        "quantization": args.quantization,
        "n_samples": len(prompts),
        "seqlen": int(payload["seqlen"]),
        "vocab_size": vocab_size,
        "n_positions": int(flat.numel()),
        "kl_mean": float(flat.mean()),
        "kl_p99": float(flat.quantile(0.99)),
        "kl_max": float(flat.max()),
        "kl_confident_mean": float(kl_pos[confident].mean())
        if bool(confident.any()) else None,
        "n_confident": int(confident.sum()),
        "kl_per_sample": [float(x) for x in kl_pos.mean(dim=1).tolist()],
        "elapsed_s": time.monotonic() - started,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0


def _student(args) -> int:
    started = time.monotonic()
    payload = torch.load(args.teacher_payload, map_location="cpu")
    if payload.get("score_positions") == "all":
        return _student_all_positions(args, payload)
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
    parser.add_argument(
        "--score-positions", choices=["final", "all"], default="final",
        help="final: full-vocab KL at the window-final context only "
        "(legacy; n_positions = n_samples). all: top-K KL at every prompt "
        "position (n_positions = n_samples*(seqlen-1)).")
    parser.add_argument("--prompt-top-k", type=int, default=1024)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    if args.mode == "student" and not args.teacher_payload:
        parser.error("--teacher-payload is required in student mode")
    if args.mode == "teacher":
        return _teacher(args)
    return _student(args)


if __name__ == "__main__":
    raise SystemExit(main())
