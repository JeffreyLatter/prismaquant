#!/usr/bin/env python3
"""Run prismaquant.validate_quantized_model.check_perplexity on each
qwen4b-smoke artifact in turn. Apples-to-apples vs the shipped 27B
which was scored against the same EVAL_PROMPTS suite.

Outputs CSV with config, ppl, mean_nll, p99_nll for each artifact.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, "/home/rob/prismaquant")
from prismaquant.validate_quantized_model import EVAL_PROMPTS

import os as _os
SMOKE_ROOT = Path(_os.environ.get(
    "WORK_ROOT", "/home/rob/dq-runs/qwen4b-smoke"))


def list_artifacts() -> list[str]:
    if not SMOKE_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in SMOKE_ROOT.iterdir()
        if (p / "exported" / "config.json").is_file()
    )


def serve_artifact(config_name: str) -> str:
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


def compute_validator_ppl() -> dict:
    """Mirror prismaquant.validate_quantized_model.check_perplexity but
    don't enforce thresholds — we just want raw numbers per config."""
    base = "http://localhost:8000/v1"
    models = requests.get(f"{base}/models", timeout=10).json()
    model_id = models["data"][0]["id"]
    per_prompt_avg_nll: list[float] = []
    total_tokens = 0
    total_nll = 0.0
    errors: list[str] = []
    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        try:
            r = requests.post(
                f"{base}/completions",
                json={
                    "model": model_id, "prompt": prompt,
                    "max_tokens": 1, "temperature": 0.0,
                    "logprobs": 1, "echo": True,
                },
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            tok_logprobs = data["choices"][0]["logprobs"]["token_logprobs"] or []
            valid = [x for x in tok_logprobs if x is not None]
            if not valid:
                errors.append(f"prompt {i}: empty logprobs")
                continue
            nll = -sum(valid)
            total_nll += nll
            total_tokens += len(valid)
            per_prompt_avg_nll.append(nll / len(valid))
        except Exception as e:
            errors.append(f"prompt {i}: {type(e).__name__}: {e}")
    if total_tokens == 0:
        return {"ppl": float("inf"), "mean_nll": float("inf"),
                "p99": float("inf"), "errors": errors,
                "n_prompts": len(EVAL_PROMPTS), "n_ok": 0}
    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)
    per_prompt_avg_nll.sort()
    p99 = per_prompt_avg_nll[-1] if len(per_prompt_avg_nll) <= 2 \
        else per_prompt_avg_nll[max(0, int(0.99 * len(per_prompt_avg_nll)) - 1)]
    return {"ppl": ppl, "mean_nll": mean_nll, "p99": p99,
            "errors": errors, "n_prompts": len(EVAL_PROMPTS),
            "n_ok": len(per_prompt_avg_nll)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None,
                    help="comma-separated config list")
    ap.add_argument("--output", default=str(SMOKE_ROOT / "validator_eval.csv"))
    args = ap.parse_args()

    configs = (args.configs.split(",") if args.configs
               else list_artifacts())
    if not configs:
        print("no artifacts found")
        return 1

    print(f"[validator] configs: {configs}")
    print(f"[validator] EVAL_PROMPTS: {len(EVAL_PROMPTS)} prompts")

    results = []
    for cfg in configs:
        print(f"\n=== [{time.strftime('%H:%M:%S')}] {cfg} ===")
        container = serve_artifact(cfg)
        try:
            print(f"  container: {container}")
            if not wait_for_ready():
                print(f"  FAIL vLLM did not become ready")
                results.append({
                    "config": cfg, "ppl": None, "mean_nll": None,
                    "p99": None, "n_ok": 0,
                    "error": "vllm timeout",
                })
                continue
            t0 = time.time()
            res = compute_validator_ppl()
            print(f"  PPL = {res['ppl']:.4f}, mean_nll = {res['mean_nll']:.4f}, "
                  f"p99 = {res['p99']:.4f} ({time.time() - t0:.0f} sec, "
                  f"{res['n_ok']}/{res['n_prompts']} prompts ok)")
            if res["errors"]:
                for e in res["errors"][:3]:
                    print(f"    err: {e}")
            results.append({
                "config": cfg, "ppl": res["ppl"],
                "mean_nll": res["mean_nll"], "p99": res["p99"],
                "n_ok": res["n_ok"], "error": None,
            })
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container],
                check=False, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    base_ppl = next(
        (r["ppl"] for r in results if r["config"] == "baseline"
         and r["ppl"] is not None), None)
    with open(args.output, "w") as f:
        f.write("config,ppl,mean_nll,p99,n_ok,delta_vs_baseline,error\n")
        for r in results:
            ppl = r["ppl"]
            d = (None if ppl is None or base_ppl is None
                 else ppl - base_ppl)
            f.write(f"{r['config']},{ppl},{r['mean_nll']},{r['p99']},"
                    f"{r['n_ok']},{d},{r['error'] or ''}\n")

    print(f"\n[validator] wrote {args.output}")
    print()
    print(f"{'config':<22} {'ppl':>8} {'mean_nll':>10} {'p99':>8} "
          f"{'Δ_vs_base':>11}")
    print("-" * 65)
    for r in results:
        ppl_s = f"{r['ppl']:.3f}" if r["ppl"] is not None else "FAIL"
        nll_s = f"{r['mean_nll']:.3f}" if r["mean_nll"] is not None else "—"
        p99_s = f"{r['p99']:.3f}" if r["p99"] is not None else "—"
        d = (r["ppl"] - base_ppl
             if r["ppl"] is not None and base_ppl is not None else None)
        d_s = f"{d:+.3f}" if d is not None else "—"
        print(f"{r['config']:<22} {ppl_s:>8} {nll_s:>10} {p99_s:>8} "
              f"{d_s:>11}")


if __name__ == "__main__":
    main()
