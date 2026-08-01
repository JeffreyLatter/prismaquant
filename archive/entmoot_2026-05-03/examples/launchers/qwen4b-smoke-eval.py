#!/usr/bin/env python3
"""Wikitext-2 perplexity eval for the Qwen 4B smoke artifacts.

Runs vLLM serve on each artifact in turn (port 8000), evaluates PPL on
wikitext-2 raw test set, tears down, moves to next. Outputs a CSV
comparison table.

Usage:
    python3 qwen4b-smoke-eval.py [--configs config1,config2,...]

Defaults to all artifacts under /home/rob/dq-runs/qwen4b-smoke/*/.

Notes:
- Uses the openai-completions API (vLLM serves it). Submits the full
  test sequence in chunks, gets per-token logprobs, computes NLL/PPL.
- vLLM init takes ~30-60 sec; eval is ~5-10 min per artifact (~6.4 MB
  of test text at 4B-class throughput on Spark).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import requests


SMOKE_ROOT = Path("/home/rob/dq-runs/qwen4b-smoke")


def list_artifacts() -> list[str]:
    """Return all config names that have an exported artifact."""
    if not SMOKE_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in SMOKE_ROOT.iterdir()
        if (p / "exported" / "config.json").is_file()
    )


def serve_artifact(config_name: str) -> str:
    """Start vLLM serve for an artifact. Returns container name."""
    name = f"pq-qwen4b-eval-{config_name.replace('+', 'p').replace('_', '-')}"
    artifact = SMOKE_ROOT / config_name / "exported"
    subprocess.run(["docker", "rm", "-f", name],
                   check=False, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    cmd = [
        "docker", "run", "-d",
        "--gpus", "all", "--ipc=host", "--shm-size=8g",
        "-v", f"{artifact}:/model:ro",
        "-p", "8000:8000",
        "--name", name,
        "--entrypoint", "vllm",
        "vllm-fresh-b12x:latest",
        "serve", "/model",
        "--host", "0.0.0.0", "--port", "8000",
        "--quantization", "compressed-tensors",
        "--max-model-len", "2048",
        # Spark has unified memory — vLLM's "gpu memory" is the host's
        # memory. 0.85 over-allocates ~100 GB of KV cache for a 4B
        # model (and starves the eval driver + my session). 0.20 ≈
        # 25 GB is plenty for 4B + a small KV pool.
        "--gpu-memory-utilization", "0.20",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return name


def wait_for_ready(timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False


def load_wikitext_test() -> str:
    """Load wikitext-2 raw test set as a single string."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(t for t in ds["text"] if t.strip())


def compute_ppl(text: str, *, max_chunk_tokens: int = 2048,
                stride: int = 1024) -> float:
    """Compute wikitext PPL via /v1/completions echo+logprobs.

    Submits chunks of `max_chunk_tokens` with `stride` overlap,
    sums NLL on the non-overlapping suffix per chunk, divides by total
    non-prompt tokens.
    """
    # Tokenize via the served model — request 1-token completion with
    # echo and logprobs to get back the prompt's per-token logprobs.
    base = "http://localhost:8000/v1"
    # Get the model name from /v1/models
    models = requests.get(f"{base}/models", timeout=10).json()
    model_id = models["data"][0]["id"]

    total_nll = 0.0
    total_tokens = 0

    # Approximate tokenization by character count: ~4 chars/token for
    # Qwen-style vocab. The actual tokenization happens server-side.
    char_chunk = max_chunk_tokens * 3
    char_stride = stride * 3

    pos = 0
    while pos < len(text):
        chunk = text[pos: pos + char_chunk]
        if len(chunk) < 100:
            break
        try:
            r = requests.post(
                f"{base}/completions",
                json={
                    "model": model_id,
                    "prompt": chunk,
                    "max_tokens": 1,
                    "echo": True,
                    "logprobs": 1,
                    "temperature": 0.0,
                },
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            lp = data["choices"][0]["logprobs"]
            tok_logprobs = lp["token_logprobs"]
            # First token is prompt-start (no logprob; None).
            valid = [x for x in tok_logprobs if x is not None]
            n_new = max(0, len(valid) - (len(valid) - stride if pos > 0 else 0))
            score_slice = valid[-n_new:] if n_new > 0 else valid
            total_nll += -sum(score_slice)
            total_tokens += len(score_slice)
        except Exception as e:
            print(f"  WARN chunk error at pos {pos}: {e}")
        pos += char_stride

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None,
                    help="comma-separated config list (default: all)")
    ap.add_argument("--output", default="/home/rob/dq-runs/qwen4b-smoke/eval.csv",
                    help="CSV output path")
    args = ap.parse_args()

    configs = (args.configs.split(",")
               if args.configs else list_artifacts())
    if not configs:
        print("no artifacts found")
        return 1

    print(f"[eval] configs: {configs}")
    print(f"[eval] loading wikitext-2-raw test set ...")
    text = load_wikitext_test()
    print(f"[eval] {len(text):,} chars")

    results = []
    for cfg in configs:
        print(f"\n=== [{time.strftime('%H:%M:%S')}] {cfg} ===")
        try:
            container = serve_artifact(cfg)
            print(f"  container: {container}")
            if not wait_for_ready():
                print(f"  FAIL vLLM did not become ready")
                results.append({"config": cfg, "ppl": None,
                                "error": "vllm timeout"})
                subprocess.run(["docker", "rm", "-f", container],
                               check=False)
                continue
            print(f"  computing PPL ...")
            t0 = time.time()
            ppl = compute_ppl(text)
            print(f"  PPL = {ppl:.3f} ({time.time() - t0:.0f} sec)")
            results.append({"config": cfg, "ppl": ppl, "error": None})
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append({"config": cfg, "ppl": None, "error": str(e)})
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container],
                check=False, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    # Write CSV.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("config,ppl,delta_vs_baseline,error\n")
        baseline_ppl = next(
            (r["ppl"] for r in results if r["config"] == "baseline"
             and r["ppl"] is not None), None)
        for r in results:
            ppl = r["ppl"]
            delta = (None if ppl is None or baseline_ppl is None
                     else ppl - baseline_ppl)
            f.write(f"{r['config']},{ppl},{delta},{r['error'] or ''}\n")

    print(f"\n[eval] wrote {args.output}")
    print()
    print(f"{'config':<20} {'ppl':>10} {'Δ_vs_base':>12}")
    print("-" * 45)
    for r in results:
        ppl_s = f"{r['ppl']:.3f}" if r["ppl"] is not None else "FAIL"
        delta = (r["ppl"] - baseline_ppl
                 if r["ppl"] is not None and baseline_ppl is not None
                 else None)
        delta_s = f"{delta:+.3f}" if delta is not None else "—"
        print(f"{r['config']:<20} {ppl_s:>10} {delta_s:>12}")


if __name__ == "__main__":
    main()
