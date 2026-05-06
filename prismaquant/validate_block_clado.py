"""Measure real teacher-student KL for Block-CLADO kneedle candidates.

Reads the ``kneedle/<label>.json`` files produced by ``block_clado kneedle``,
loads the BF16 reference model, and for each candidate measures the actual
last-token KL using the same calibration recipe as the collector.

Output: a single JSON summary at ``--output`` mapping each candidate label
to its surrogate cost vs. measured KL.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import torch

from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    stage_multimodal,
)
from prismaquant.iterate_perturbed_allocation import measure_assignment_kl
from prismaquant.measure_adjoint_l3 import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.model_profiles import DefaultProfile, detect_profile


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Block-CLADO candidates")
    parser.add_argument("--model", required=True)
    parser.add_argument("--kneedle-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-calib-samples", type=int, default=2)
    parser.add_argument("--calib-seqlen", type=int, default=128)
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--production-weight-cache",
        default=None,
        help="Path to a pickled ProductionWeightCache (use the production-"
        "faithful δw for cone KL instead of bare RTN).",
    )
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    kneedle_dir = Path(args.kneedle_dir)
    candidates = sorted(
        p for p in kneedle_dir.glob("*.json") if p.name != "summary.json"
    )
    if not candidates:
        raise RuntimeError(f"no kneedle JSONs in {kneedle_dir}")
    print(f"[validate] candidates: {[p.name for p in candidates]}", flush=True)

    dtype = _dtype_from_name(args.dtype)
    staged, cleanup = stage_multimodal(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_root = Path(tempfile.mkdtemp(prefix="prismaquant_validate_bc_"))
    results = []
    try:
        local_only = Path(staged).exists()
        tokenizer = AutoTokenizer.from_pretrained(
            staged, trust_remote_code=True, local_files_only=local_only,
        )
        calib_ids = load_wikitext_calibration_windowed(
            tokenizer,
            args.n_calib_samples,
            args.calib_seqlen,
            split=args.calib_split,
            seed=args.calib_seed,
        )
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
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

        production_weight_cache = None
        if args.production_weight_cache:
            import pickle
            with open(args.production_weight_cache, "rb") as fh:
                production_weight_cache = pickle.load(fh)
            print(
                f"[validate] loaded production cache with "
                f"{len(production_weight_cache)} entries from "
                f"{args.production_weight_cache}",
                flush=True,
            )

        for path in candidates:
            payload = json.loads(path.read_text())
            assignment = payload["assignment"]
            kl = measure_assignment_kl(
                model,
                assignment,
                calib_ids,
                ref_log_probs,
                work_root=work_root,
                profile=profile,
                use_frozen_weight_cache=False,
                production_weight_cache=production_weight_cache,
                rng_seed=0,
            )
            counts = dict(Counter(assignment.values()))
            label = str(payload.get("label") or path.stem)
            row = {
                "label": label,
                "bpp": float(payload["bpp"]),
                "surrogate_cost": float(payload.get("surrogate_cost", 0.0)),
                "lambda": float(payload.get("lambda", 0.0)),
                "real_kl": float(kl),
                "format_counts": counts,
            }
            results.append(row)
            print(
                f"[validate] {label:>32s}  bpp={row['bpp']:.4f}  "
                f"surrogate_cost={row['surrogate_cost']:+.4f}  "
                f"real_kl={row['real_kl']:.6f}  counts={counts}",
                flush=True,
            )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)
        shutil.rmtree(work_root, ignore_errors=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "schema": "prismaquant.block_clado.validate.v1",
        "model": args.model,
        "calibration": {
            "n_calib_samples": int(args.n_calib_samples),
            "seqlen": int(args.calib_seqlen),
            "split": args.calib_split,
            "seed": int(args.calib_seed),
        },
        "results": results,
    }, indent=2) + "\n")
    print(f"[validate] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
