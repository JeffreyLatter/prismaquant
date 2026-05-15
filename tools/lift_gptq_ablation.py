"""Lift/MR-GPTQ ablation runner: 4-arm comparison with controlled levers.

Reproduces the existing run at
``/home/rob/dq-runs/qwen35-0p8b-lift-gptq-ablation-20260514T224232Z`` so we
can iterate on the JSO/SAO fixes and produce KL deltas under the same
contract (fixed assignment, n=32, seqlen=1024, last_token KL, prefetch
required, scale_sweep disabled).

Usage:
    python3 tools/lift_gptq_ablation.py \
        --output-root /home/rob/dq-runs/sao-fix-iter-N

Each arm builds a production cache and then validates KL against a fixed
base assignment. The arms differ only in the levers passed to
``build_production_cache``.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ARMS = {
    "vanilla_gptq_nvfp4":      "gptq",
    "joint_scale_opt_nvfp4":   "gptq,joint_scale_opt",
    # scale_sweep validation: JSO vs JSO + post-GPTQ scale_sweep polish.
    "jso_nvfp4_no_sweep":      "gptq,joint_scale_opt",
    "jso_nvfp4_sweep":         "gptq,joint_scale_opt,scale_sweep",
    # static_act_order_nvfp4 / jso_sao_nvfp4 arms removed 2026-05-15: SAO
    # showed no win on its own activation-weighted objective and is
    # archive-walled in the pipeline. See archive/sao_2026-05-15/ (TBD).
}


def run(cmd: list[str], *, env: dict[str, str], log_path: Path, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    t0 = time.time()
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-30:]
        raise SystemExit(
            f"[{label}] FAILED in {dt:.1f}s (rc={proc.returncode}); "
            f"tail of {log_path}:\n  " + "\n  ".join(tail)
        )
    print(f"[{label}] done in {dt:.1f}s", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True,
                   help="Root directory for arm artifacts.")
    p.add_argument("--model", default="/hfcache/qwen35-0p8b-bf16")
    p.add_argument(
        "--base-assignment",
        default="/dq-runs/fouroversix-smoke-20260510T225344Z/"
                "qwen35-0p8b/artifacts/layer_config.json",
    )
    p.add_argument(
        "--probe",
        default="/dq-runs/qwen35-0p8b-pipeline-recache-smoke-20260509T213900Z/"
                "artifacts/probe.pkl",
    )
    p.add_argument(
        "--costs",
        default="/dq-runs/qwen35-0p8b-pipeline-recache-smoke-20260509T213900Z/"
                "artifacts/cost.pkl",
    )
    p.add_argument(
        "--dataset",
        default="/dq-runs/calibration/diverse-v1.jsonl",
    )
    p.add_argument("--n-calib", type=int, default=32)
    p.add_argument("--calib-seqlen", type=int, default=1024)
    p.add_argument(
        "--arms",
        default=",".join(ARMS.keys()),
        help="Comma-separated arm labels to run.",
    )
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Use existing production_weight_cache.pkl in each arm dir; just run KL.",
    )
    p.add_argument(
        "--n-trials", type=int, default=1,
        help="Per-arm build trials (each gets its own cache + KL). "
        "Cross-build variance is the dominant noise source; N=3 typically "
        "stabilizes the median to within 1-2%%. Validate is deterministic "
        "after a warmup pass.",
    )
    p.add_argument(
        "--warmup", action="store_true",
        help="Run a throwaway KL validate before measuring (eliminates "
        "first-run warmup noise on subsequent validates).",
    )
    args = p.parse_args(argv)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    selected = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in selected:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; valid: {list(ARMS)}")

    # Write the contract README upfront so the artifact is self-describing.
    (out_root / "README.txt").write_text(
        "\n".join([
            "Fixed-assignment GPTQ NVFP4 Lift ablation (re-run via tools/lift_gptq_ablation.py)",
            f"model={args.model}",
            f"assignment={args.base_assignment}",
            f"probe={args.probe}",
            f"costs={args.costs}",
            f"dataset={args.dataset}",
            f"n_calib={args.n_calib}",
            f"seqlen={args.calib_seqlen}",
            f"arms={','.join(selected)}",
            "",
        ])
    )

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/home/rob/.cache/huggingface")
    env.setdefault("PYTHONPATH", "/home/rob/prismaquant")

    # Validate runs without determinism env vars (full determinism causes
    # NaN KL via deterministic-algorithm fallback paths).
    validate_env = {k: v for k, v in env.items()
                    if k not in ("CUBLAS_WORKSPACE_CONFIG", "PRISMAQUANT_DETERMINISTIC")}

    # Validate is deterministic-after-warmup on a given cache: runs 2+ give
    # bit-identical KL but run 1 has CUDA-init noise (~10% variance). Do
    # one throwaway validate at startup to prime CUDA state, using any
    # cache that exists.
    # Warmup pre-builds and validates a real-calib cache to prime CUDA
    # state; we discard the KL number. The build must use the same calib
    # size as the trials — a small (4×256) build produces undertrained
    # weights that NaN at validate time, defeating the warmup's purpose.
    if args.warmup and not args.skip_build:
        warmup_dir = out_root / "_warmup"
        warmup_dir.mkdir(parents=True, exist_ok=True)
        warmup_cache_pkl = warmup_dir / "production_weight_cache.pkl"
        warmup_cache_dir = warmup_dir / "cache_dir"
        if not warmup_cache_pkl.exists():
            build_cmd = [
                sys.executable, "-m", "prismaquant.build_production_cache",
                "--model", args.model,
                "--output", str(warmup_cache_pkl),
                "--formats", "NVFP4,MXFP8_E4M3,FP8_E4M3",
                "--render-scope", "assignment",
                "--render-layer-config", args.base_assignment,
                "--n-calib-samples", str(args.n_calib),
                "--calib-seqlen", str(args.calib_seqlen),
                "--dataset", args.dataset,
                "--enable", "gptq", "--disable", "scale_sweep",
                "--cache-dir", str(warmup_cache_dir),
            ]
            run(build_cmd, env=env, log_path=warmup_dir / "build.log", label="build/_warmup")
        warmup_cmd = [
            sys.executable, "-m", "prismaquant.validate_assignments_kl",
            "--model", args.model, "--probe", args.probe, "--costs", args.costs,
            "--base-assignment", args.base_assignment,
            "--assignment", f"warmup={args.base_assignment}",
            "--output", str(warmup_dir / "kl.json"),
            "--n-calib-samples", str(args.n_calib),
            "--calib-seqlen", str(args.calib_seqlen),
            "--dataset", args.dataset,
            "--kl-scope", "last_token",
            "--assignment-materialization", "hooks",
            "--production-weight-cache", str(warmup_cache_pkl),
            "--production-cache-dir-override", str(warmup_cache_dir),
            "--production-cache-lru-gb", "32.0",
            "--production-cache-prefetch", "require",
            "--production-cache-prefetch-workers", "4",
            "--kl-cuda-graphs", "off",
        ]
        run(warmup_cmd, env=validate_env, log_path=warmup_dir / "validate.log", label="kl/_warmup")

    # Per-arm build + validate (N trials)
    summary = []
    for arm in selected:
        arm_dir = out_root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        trial_kls: list[tuple[int, Path]] = []

        for trial in range(int(args.n_trials)):
            trial_label = arm if args.n_trials == 1 else f"{arm}_t{trial}"
            trial_dir = arm_dir if args.n_trials == 1 else arm_dir / f"trial_{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            cache_pkl = trial_dir / "production_weight_cache.pkl"
            cache_dir = trial_dir / "cache_dir"
            kl_json = trial_dir / "kl.json"

            if not args.skip_build or not cache_pkl.exists():
                enable_list = ARMS[arm]
                # Disable scale_sweep only when the arm itself doesn't ask for it.
                disable_list = (
                    "" if "scale_sweep" in enable_list.split(",")
                    else "scale_sweep"
                )
                build_cmd = [
                    sys.executable, "-m", "prismaquant.build_production_cache",
                    "--model", args.model,
                    "--output", str(cache_pkl),
                    "--formats", "NVFP4,MXFP8_E4M3,FP8_E4M3",
                    "--render-scope", "assignment",
                    "--render-layer-config", args.base_assignment,
                    "--n-calib-samples", str(args.n_calib),
                    "--calib-seqlen", str(args.calib_seqlen),
                    "--dataset", args.dataset,
                    "--enable", enable_list,
                    "--disable", disable_list,
                    "--cache-dir", str(cache_dir),
                ]
                run(build_cmd, env=env, log_path=trial_dir / "build.log",
                    label=f"build/{trial_label}")

            validate_cmd = [
                sys.executable, "-m", "prismaquant.validate_assignments_kl",
                "--model", args.model, "--probe", args.probe, "--costs", args.costs,
                "--base-assignment", args.base_assignment,
                "--assignment", f"{arm}={args.base_assignment}",
                "--output", str(kl_json),
                "--n-calib-samples", str(args.n_calib),
                "--calib-seqlen", str(args.calib_seqlen),
                "--dataset", args.dataset,
                "--kl-scope", "last_token",
                "--assignment-materialization", "hooks",
                "--production-weight-cache", str(cache_pkl),
                "--production-cache-dir-override", str(cache_dir),
                "--production-cache-lru-gb", "32.0",
                "--production-cache-prefetch", "require",
                "--production-cache-prefetch-workers", "4",
                "--kl-cuda-graphs", "off",
            ]
            run(validate_cmd, env=validate_env, log_path=trial_dir / "validate.log",
                label=f"kl/{trial_label}")
            trial_kls.append((trial, kl_json))

        summary.append((arm, trial_kls))

    # Aggregate and print summary
    import statistics
    print("\n=== KL summary ===")
    print(f"{'arm':<28}  {'median':>11}  {'min':>11}  {'max':>11}  {'Δ% vs ref':>11}")
    baseline_median = None
    for arm, trial_kls in summary:
        kls = []
        for trial, kl_json in trial_kls:
            d = json.loads(kl_json.read_text())
            for r in d["results"]:
                if r["label"] == arm:
                    kl = float(r["last_token_kl"])
                    if not (kl != kl):  # exclude NaN
                        kls.append(kl)
                    break
        if not kls:
            print(f"{arm:<28}  (all trials NaN/missing)")
            continue
        med = statistics.median(kls)
        lo, hi = min(kls), max(kls)
        if baseline_median is None:
            baseline_median = med
            delta = ""
        else:
            delta = f"{(med - baseline_median) / baseline_median * 100:+8.2f}%"
        n_label = f" n={len(kls)}" if len(trial_kls) > 1 else ""
        print(f"{arm:<28}  {med:>11.7f}  {lo:>11.7f}  {hi:>11.7f}  {delta:>11s}{n_label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
