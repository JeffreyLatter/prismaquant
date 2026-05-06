"""Polish a saved assignment with production-faithful weights.

Loads:
  - a Block-CLADO payload (for the unit/pair structure)
  - a kneedle-style assignment JSON (any cone candidate)
  - optionally a ProductionWeightCache pickle

Runs ``coord_descent_polish`` on it and writes the polished assignment +
trace to ``--output`` JSON.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from prismaquant import format_registry as fr
from prismaquant import block_clado as bc
from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    stage_multimodal,
)
from prismaquant.coord_descent_polish import (
    _assignment_bits,
    coord_descent_polish,
)
from prismaquant.iterate_perturbed_allocation import measure_assignment_kl
from prismaquant.measure_adjoint_l3 import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.model_profiles import DefaultProfile, detect_profile


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Polish a saved assignment")
    p.add_argument("--model", required=True)
    p.add_argument("--payload", required=True,
                   help="four-term or output-fisher payload JSON")
    p.add_argument("--assignment", required=True,
                   help="JSON file with `assignment: {qname: fmt}`")
    p.add_argument("--output", required=True)
    p.add_argument("--production-weight-cache", default=None)
    p.add_argument("--n-calib-samples", type=int, default=2)
    p.add_argument("--calib-seqlen", type=int, default=128)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--dtype", default="bf16")
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
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    payload = bc.load_payload(args.payload)
    blocks_back, singletons_back, pairs_back = bc.parse_payload(payload)
    units: list[bc.DecisionUnit] = []
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
    if args.production_weight_cache:
        with open(args.production_weight_cache, "rb") as fh:
            production_weight_cache = pickle.load(fh)
        print(f"[polish] loaded prod cache with "
              f"{len(production_weight_cache)} entries", flush=True)

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
        load_kwargs = {
            "torch_dtype": dtype, "trust_remote_code": True,
            "local_files_only": local_only,
        }
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
        ref_log_probs = cache_reference_log_probs(model, calib_ids, device)

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
            if kind in {"accept_move", "starting", "budget_set"}:
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
            progress_callback=progress,
        )
        elapsed = time.monotonic() - t0

        n_params = bc.total_param_count(payload)
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
