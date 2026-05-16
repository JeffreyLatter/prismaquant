"""Polish a saved assignment with production-faithful weights.

Loads:
  - a decision-unit payload (for the fused-sibling unit/pair structure)
  - a kneedle-style assignment JSON (any cone candidate)
  - optionally a ProductionWeightCache pickle

Runs measured single-flip polish on it and writes the polished assignment +
trace to ``--output`` JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from prismaquant import format_registry as fr
from prismaquant import decision_units as du
from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    stage_multimodal,
)
from prismaquant.polish import (
    _assignment_bits,
    coord_descent_polish,
)
from prismaquant.calibration_data import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.kl_measurement import measure_assignment_kl
from prismaquant.perturbed_x_cache import calibration_data_hash
from prismaquant.model_profiles import DefaultProfile, detect_profile


def _assignment_digest(assignment: dict[str, str]) -> str:
    payload = json.dumps(dict(sorted(assignment.items())), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Polish a saved assignment")
    p.add_argument("--model", required=True)
    p.add_argument("--payload", required=True,
                   help="four-term or output-fisher payload JSON")
    p.add_argument("--assignment", required=True,
                   help="JSON file with `assignment: {qname: fmt}`")
    p.add_argument("--output", required=True)
    p.add_argument("--production-weight-cache", default=None)
    # Defaults match the shipping artifact's validation calibration
    # (8 samples × 512 tokens = 4 096 tokens).  An older default of
    # 2 × 128 was a sanity-run config that should not back any
    # publishable KL claim; it is a legacy sanity-run config.
    p.add_argument("--n-calib-samples", type=int, default=8)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--dtype", default="bf16")
    p.add_argument(
        "--attn-implementation",
        default=None,
        help=(
            "Optional Transformers attention backend for model load. "
            "Examples: sdpa, flash_attention_2, kernels-community/flash-attn2."
        ),
    )
    p.add_argument(
        "--kl-scope",
        choices=["last_token", "full_sequence"],
        default="last_token",
        help="KL reduction scope.  last_token caches only the final-token "
        "teacher distribution per calibration row.",
    )
    p.add_argument(
        "--no-activation-quant",
        action="store_true",
        help="Disable production activation quantization during polish KL.",
    )
    p.add_argument("--polish-budget-creep", type=float, default=0.05)
    p.add_argument("--polish-max-passes", type=int, default=12)
    p.add_argument("--polish-noise-floor", type=float, default=1e-5)
    p.add_argument("--polish-steepest-first", action="store_true")
    p.add_argument(
        "--pin",
        nargs="*",
        default=["lm_head"],
        help="Qnames forced to BF16 in the starting assignment AND not "
        "moved during polish.  Default: lm_head (vLLM ParallelLMHead "
        "rejects compressed-tensors layout).",
    )
    p.add_argument(
        "--direction",
        choices=["bottom_up", "top_down"],
        default="bottom_up",
        help="Polish direction.  'bottom_up' starts at low-bpp/high-KL "
        "and accepts flips that decrease KL within the bits budget. "
        "'top_down' starts at high-bpp/low-KL and accepts flips that "
        "decrease bits while keeping KL ≤ kl_budget — Buchbinder et "
        "al. (2012) double-greedy formulation.  Run both and combine "
        "trajectories for full Pareto coverage.",
    )
    p.add_argument(
        "--kl-budget",
        type=float,
        default=None,
        help="Top-down only: maximum allowed KL.  Polish accepts a flip "
        "iff trial_kl ≤ kl_budget.  Required for top_down.",
    )
    p.add_argument(
        "--delta-quantize",
        dest="delta_quantize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use delta-quantize WeightSession (single-unit in-place swap "
        "per trial) when a production_weight_cache is provided.  When "
        "None (default), falls back to the PRISMAQUANT_DELTA_QUANTIZE_POLISH "
        "env var.  --no-delta-quantize forces the legacy clone-and-restore "
        "path.  Required for big-model polish (27B+) on a 121 GB UMA host.",
    )
    p.add_argument(
        "--cache-dir-override",
        default=None,
        help="When the production_weight_cache pkl was built in disk-"
        "streaming mode in another container/environment, its stored "
        "cache_dir may be unreachable from this run.  Pass the host-"
        "resolvable path to the cache_shards directory here and the "
        "cache will be relocated before any .pt resolution.",
    )
    p.add_argument(
        "--weight-session-spill-to-disk",
        action="store_true",
        help="Spill WeightSession's BF16 source snapshots to disk (in "
        "work_root) instead of holding all of them in memory.  Bounds "
        "the polish-time snapshot footprint at the cost of one "
        "torch.save per first-touch and one torch.load per revert.  "
        "Required for very-large models (e.g. 70B+ on a 121 GB UMA host) "
        "where the cumulative BF16-source footprint of every quantizable "
        "Linear would exceed the budget.",
    )
    p.add_argument(
        "--weight-session-snapshot-dir",
        default=os.environ.get("PRISMAQUANT_WEIGHT_SESSION_SNAPSHOT_DIR"),
        help=(
            "Optional shared directory for WeightSession BF16 snapshots. "
            "Use this with --weight-session-spill-to-disk to reuse snapshots "
            "across polish/validation runs."
        ),
    )
    p.add_argument(
        "--lru-gb",
        type=float,
        default=0.0,
        help="When >0 and the production cache was built in disk-streaming "
        "mode, bound the in-memory tensor footprint to this many GiB via "
        "LRU eviction.  Required on big models that don't fit alongside "
        "the model weights (e.g. 27B+ on 121 GB UMA).  Disk churn cost "
        "is roughly (n_units_flipped × weight_size) per polish trial.",
    )
    p.add_argument(
        "--prefetch-cache",
        dest="prefetch_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="At polish startup, eagerly torch.load every disk-streamed "
        "cache entry via a thread pool.  Trades ~5 sec of startup time "
        "for elimination of per-trial torch.load latency (which would "
        "otherwise dominate wall time on disk-streamed caches with the "
        "default --lru-gb=0 — every miss reloads from disk).  Default "
        "ON.  Disable with --no-prefetch-cache only on systems where "
        "the in-memory cache footprint plus model weights would exceed "
        "the host's UMA budget.",
    )
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Fail fast if the optimized linear-attention kernels are missing
    # on a Qwen3.5/3.6 hybrid model — the Python torch fallback is
    # ~5-10x slower and silently destroys polish wall-time measurability.
    # Bypass with PRISMAQUANT_ALLOW_PYTORCH_FALLBACK=1 for debug only.
    from prismaquant._fast_kernel_guard import require_fast_kernels
    require_fast_kernels(args.model)

    payload = du.load_payload(args.payload)
    blocks_back, singletons_back, pairs_back = du.parse_payload(payload)
    units: list[du.DecisionUnit] = []
    for unit_list in blocks_back.values():
        units.extend(unit_list)
    units.extend(singletons_back)

    candidate = json.loads(Path(args.assignment).read_text())
    starting_assignment = dict(candidate["assignment"])
    starting_label = candidate.get("label") or Path(args.assignment).stem

    pinned_qnames: set[str] = set()
    pin_tokens: list[str] = list(args.pin or [])
    # MED-5: prefer the profile's canonical lm_head name (DeepSeek calls it
    # 'head' in checkpoint storage; default profiles use 'lm_head').  Add
    # both the unrenamed 'lm_head' alias as a fallback so cross-profile
    # configs still pin correctly.
    try:
        from prismaquant.model_profiles import detect_profile, DefaultProfile
        try:
            _profile = detect_profile(args.model)
        except Exception:
            _profile = DefaultProfile()
        head_name = _profile.lm_head_name()
        if head_name and head_name not in pin_tokens:
            pin_tokens.append(head_name)
        if "lm_head" not in pin_tokens:
            pin_tokens.append("lm_head")
    except Exception:
        # If profile detection fails, fall back to the user-supplied list.
        pass
    if pin_tokens:
        for token in pin_tokens:
            for qname in starting_assignment:
                if token in qname.split("."):
                    pinned_qnames.add(qname)
        if pinned_qnames:
            for q in pinned_qnames:
                if starting_assignment.get(q) != "BF16":
                    print(
                        f"[polish] pin {q}: {starting_assignment[q]} -> BF16",
                        flush=True,
                    )
                starting_assignment[q] = "BF16"
        print(f"[polish] pin tokens: {pin_tokens}; pinned {len(pinned_qnames)} qnames",
              flush=True)

    production_weight_cache = None
    prod_cache_diag: dict = {
        "path": str(args.production_weight_cache or ""),
        "entries": 0,
    }
    if args.production_weight_cache:
        with open(args.production_weight_cache, "rb") as fh:
            production_weight_cache = pickle.load(fh)
        prod_cache_diag.update({
            "entries": len(production_weight_cache),
            "cache_dir": str(
                getattr(production_weight_cache, "cache_dir", "") or ""
            ),
            "activation_max_abs_entries": len(
                getattr(production_weight_cache, "activation_max_abs", {})
                or {}
            ),
        })
        print(f"[polish] loaded prod cache with "
              f"{len(production_weight_cache)} entries", flush=True)
        if args.cache_dir_override:
            production_weight_cache.relocate(args.cache_dir_override)
            prod_cache_diag["cache_dir"] = str(args.cache_dir_override)
            print(
                f"[polish] cache_dir relocated to "
                f"{args.cache_dir_override}",
                flush=True,
            )
        # Verify backing .pt files exist BEFORE we sink a model load +
        # an activation forward pass.  A pickled disk-streaming cache
        # whose backing directory was deleted would otherwise raise
        # FileNotFoundError mid-polish — a costly failure mode.
        if getattr(production_weight_cache, "cache_dir", None) is not None:
            verify = production_weight_cache.verify_files()
            n_present = len(verify["present"])
            n_missing = len(verify["missing"])
            n_in_mem = len(verify["in_memory"])
            prod_cache_diag["verify"] = {
                "present": n_present,
                "missing": n_missing,
                "in_memory": n_in_mem,
                "missing_sample": verify["missing"][:5],
            }
            print(
                f"[polish] cache verify: {n_present} on disk, "
                f"{n_in_mem} in memory, {n_missing} missing",
                flush=True,
            )
            if n_missing > 0:
                missing_sample = verify["missing"][:5]
                raise FileNotFoundError(
                    f"production_weight_cache references {n_missing} "
                    f".pt files that are not on disk under "
                    f"{production_weight_cache.cache_dir!r}. "
                    f"Either restore the directory or pass "
                    f"--cache-dir-override <path-to-cache_shards>. "
                    f"Sample: {missing_sample}"
                )
        if args.lru_gb and args.lru_gb > 0:
            n_bytes = int(args.lru_gb * (1024 ** 3))
            production_weight_cache.enable_lru(n_bytes)
            prod_cache_diag["lru_gb"] = float(args.lru_gb)
            print(
                f"[polish] LRU eviction enabled at {args.lru_gb:.1f} GiB",
                flush=True,
            )
        if args.prefetch_cache:
            t_pre = time.monotonic()
            n_loaded = production_weight_cache.prefetch()
            elapsed = time.monotonic() - t_pre
            prod_cache_diag["prefetch"] = {
                "enabled": True,
                "loaded": int(n_loaded),
                "elapsed_seconds": float(elapsed),
            }
            print(
                f"[polish] prefetched {n_loaded} cache entries in "
                f"{elapsed:.1f}s",
                flush=True,
            )
        else:
            prod_cache_diag["prefetch"] = {"enabled": False}

    dtype = _dtype_from_name(args.dtype)
    staged, cleanup = stage_multimodal(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_root = Path(tempfile.mkdtemp(prefix="prismaquant_polish_cli_"))
    try:
        local_only = Path(staged).exists()
        tokenizer = AutoTokenizer.from_pretrained(
            staged, trust_remote_code=True, local_files_only=local_only,
        )
        calib_ids = load_wikitext_calibration_windowed(
            tokenizer,
            args.n_calib_samples, args.calib_seqlen,
            split=args.calib_split, seed=args.calib_seed,
        )
        calib_hash = calibration_data_hash(calib_ids)
        load_kwargs = {
            "torch_dtype": dtype, "trust_remote_code": True,
            "local_files_only": local_only,
        }
        if args.attn_implementation:
            load_kwargs["attn_implementation"] = args.attn_implementation
        if device.type == "cuda":
            load_kwargs["device_map"] = "cuda"
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        if device.type != "cuda":
            model.to(device)
        model.eval()
        try:
            profile = detect_profile(args.model)
        except Exception:
            profile = DefaultProfile()
        print(
            f"[polish] caching reference logprobs "
            f"(kl_scope={args.kl_scope})",
            flush=True,
        )
        ref_log_probs = cache_reference_log_probs(
            model, calib_ids, device, kl_scope=args.kl_scope,
        )

        starting_bits = _assignment_bits(units, starting_assignment)
        budget = starting_bits * (1.0 + float(args.polish_budget_creep))
        if args.direction == "top_down" and args.kl_budget is None:
            raise ValueError(
                "--direction=top_down requires --kl-budget (max real KL); "
                "see Buchbinder et al. 2012 for the constrained formulation."
            )
        print(
            f"[polish] starting from {starting_label} (direction={args.direction}): "
            f"bits={starting_bits:.0f}  bits_budget={budget:.0f} "
            f"({100*args.polish_budget_creep:.1f}% creep)  "
            f"kl_budget={args.kl_budget}",
            flush=True,
        )

        def progress(event):
            kind = event.get("event")
            if kind in {
                "accept_move",
                "budget_set",
                "starting",
                "weight_session_initialized",
            }:
                print(f"[polish] {json.dumps(event, default=str)}", flush=True)

        t0 = time.monotonic()
        polish_result = coord_descent_polish(
            model, calib_ids, ref_log_probs,
            units=units,
            starting_assignment=starting_assignment,
            profile=profile,
            work_root=work_root,
            noise_floor=float(args.polish_noise_floor),
            max_passes=int(args.polish_max_passes),
            bits_budget=budget,
            pairs_by_block=dict(pairs_back),
            steepest_first=bool(args.polish_steepest_first),
            use_frozen_weight_cache=False,
            production_weight_cache=production_weight_cache,
            pinned_units=pinned_qnames,
            direction=args.direction,
            kl_budget=args.kl_budget,
            delta_quantize=args.delta_quantize,
            kl_scope=args.kl_scope,
            include_activation_quant=not bool(args.no_activation_quant),
            weight_session_spill_to_disk=bool(
                args.weight_session_spill_to_disk
            ),
            weight_session_snapshot_dir=args.weight_session_snapshot_dir,
            progress_callback=progress,
        )
        elapsed = time.monotonic() - t0

        n_params = du.total_param_count(payload)
        final_bits = _assignment_bits(units, polish_result.final_assignment)
        final_bpp = final_bits / float(n_params) if n_params else 0.0

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "schema": "prismaquant.polish_from_assignment.v1",
            "starting_label": starting_label,
            "starting_bpp": float(starting_bits / n_params),
            "starting_kl": float(polish_result.initial_kl),
            "final_bpp": float(final_bpp),
            "final_kl": float(polish_result.final_kl),
            "improvement": float(polish_result.initial_kl - polish_result.final_kl),
            "n_kl_measurements": int(polish_result.n_kl_measurements),
            "elapsed_seconds": float(elapsed),
            "production_weight_cache": str(args.production_weight_cache or ""),
            "diagnostics": {
                "git_commit": _git_commit(),
                "model": str(args.model),
                "payload": str(args.payload),
                "assignment": str(args.assignment),
                "starting_assignment_hash": _assignment_digest(
                    starting_assignment
                ),
                "final_assignment_hash": _assignment_digest(
                    polish_result.final_assignment
                ),
                "calibration": {
                    "dataset": "wikitext/wikitext-2-raw-v1",
                    "split": str(args.calib_split),
                    "seed": int(args.calib_seed),
                    "n_samples": int(args.n_calib_samples),
                    "seqlen": int(args.calib_seqlen),
                    "hash": str(calib_hash),
                    "kl_scope": str(args.kl_scope),
                    "include_activation_quant": bool(
                        not args.no_activation_quant
                    ),
                },
                "runtime": {
                    "torch_version": str(torch.__version__),
                    "torch_cuda": str(getattr(torch.version, "cuda", "") or ""),
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else ""
                    ),
                    "torchinductor_cache_dir": os.environ.get(
                        "TORCHINDUCTOR_CACHE_DIR", ""
                    ),
                    "rtn_compile_disabled": os.environ.get(
                        "PRISMAQUANT_DISABLE_RTN_COMPILE", ""
                    ),
                    "full_sequence_kl": os.environ.get(
                        "PRISMAQUANT_FULL_SEQUENCE_KL", ""
                    ),
                    "strict_assignment_coverage": os.environ.get(
                        "PRISMAQUANT_STRICT_ASSIGNMENT_COVERAGE", "auto"
                    ),
                    "strict_production_cache": os.environ.get(
                        "PRISMAQUANT_STRICT_PRODUCTION_CACHE", "1"
                    ),
                },
                "polish_args": {
                    "direction": str(args.direction),
                    "kl_budget": args.kl_budget,
                    "polish_budget_creep": float(args.polish_budget_creep),
                    "polish_max_passes": int(args.polish_max_passes),
                    "polish_noise_floor": float(args.polish_noise_floor),
                    "kl_scope": str(args.kl_scope),
                    "include_activation_quant": bool(
                        not args.no_activation_quant
                    ),
                    "delta_quantize": args.delta_quantize,
                    "weight_session_spill_to_disk": bool(
                        args.weight_session_spill_to_disk
                    ),
                    "weight_session_snapshot_dir": str(
                        args.weight_session_snapshot_dir or ""
                    ),
                    "lru_gb": float(args.lru_gb),
                    "prefetch_cache": bool(args.prefetch_cache),
                    "pin": list(args.pin or []),
                },
                "production_weight_cache": prod_cache_diag,
                "coord_descent": polish_result.diagnostics,
            },
            "steps": [
                {
                    "pass_index": s.pass_index,
                    "unit": s.unit,
                    "from_fmt": s.from_fmt,
                    "to_fmt": s.to_fmt,
                    "kl_before": s.kl_before,
                    "kl_after": s.kl_after,
                }
                for s in polish_result.steps
            ],
            "final_assignment": polish_result.final_assignment,
        }, indent=2) + "\n")
        print(
            f"[polish] {starting_label}: "
            f"bpp {starting_bits/n_params:.4f} → {final_bpp:.4f}  "
            f"KL {polish_result.initial_kl:.4f} → {polish_result.final_kl:.4f}  "
            f"({len(polish_result.steps)} steps, {elapsed:.0f}s)",
            flush=True,
        )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
