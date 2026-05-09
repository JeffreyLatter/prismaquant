"""Validate a polished Block-CLADO flip sequence on a larger calibration set."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

import torch

from prismaquant import block_clado as bc
from prismaquant import coord_descent_polish as cdp
from prismaquant.build_rtn_cache import cache_reference_log_probs, stage_multimodal
from prismaquant.iterate_block_clado import (
    _temporary_env,
    _prefetch_assignment_delta,
    assignment_for_units,
    bf16_assignment_for_units,
    load_assignment_json,
)
from prismaquant.iterate_perturbed_allocation import measure_assignment_kl
from prismaquant.measure_adjoint_l3 import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.model_profiles import DefaultProfile, detect_profile


def _load_units(block_clado_path: str | Path) -> tuple[list[bc.DecisionUnit], int]:
    payload = json.loads(Path(block_clado_path).read_text())
    blocks, singletons, _pairs = bc.parse_payload(payload)
    units: list[bc.DecisionUnit] = []
    for unit_list in blocks.values():
        units.extend(unit_list)
    units.extend(singletons)
    return units, bc.total_param_count(payload)


def _cumulative_polish_points(
    *,
    seed_assignment: Mapping[str, str],
    polish_payload: Mapping,
    units: Sequence[bc.DecisionUnit],
) -> list[dict]:
    unit_by_name = {unit.name: unit for unit in units}
    current = assignment_for_units(dict(seed_assignment), units)
    points = [
        {
            "label": "seed",
            "step_index": -1,
            "unit": None,
            "from_fmt": None,
            "to_fmt": None,
            "assignment": dict(current),
        }
    ]
    for idx, step in enumerate(polish_payload.get("steps", [])):
        unit_name = str(step["unit"])
        to_fmt = str(step["to_fmt"])
        unit = unit_by_name.get(unit_name)
        members = list(unit.member_qnames) if unit is not None else [unit_name]
        for member in members:
            current[member] = to_fmt
        points.append({
            "label": f"step_{idx:02d}_{unit_name}",
            "step_index": int(idx),
            "unit": unit_name,
            "from_fmt": str(step.get("from_fmt")),
            "to_fmt": to_fmt,
            "assignment": dict(current),
        })
    final_assignment = polish_payload.get("final_assignment")
    if isinstance(final_assignment, Mapping):
        final_norm = assignment_for_units(
            {str(k): str(v) for k, v in final_assignment.items()},
            units,
        )
        if final_norm != points[-1]["assignment"]:
            points.append({
                "label": "final_assignment",
                "step_index": len(points) - 1,
                "unit": None,
                "from_fmt": None,
                "to_fmt": None,
                "assignment": final_norm,
            })
    return points


def _load_production_cache(args):
    if not args.production_weight_cache:
        return None
    import pickle

    with open(args.production_weight_cache, "rb") as fh:
        cache = pickle.load(fh)
    if args.production_cache_dir_override:
        cache.relocate(args.production_cache_dir_override)
    if getattr(cache, "cache_dir", None) is not None:
        verify = cache.verify_files()
        if verify.get("missing"):
            raise RuntimeError(
                "production cache has missing shard files after relocation; "
                f"sample={verify['missing'][:5]}"
            )
    lru_gb = float(args.production_cache_lru_gb)
    if lru_gb > 0.0:
        cache.enable_lru(int(lru_gb * (1024 ** 3)))
    return cache


def _payload(args, rows: Sequence[Mapping]) -> dict:
    return {
        "schema": "prismaquant.validate_polish_flips.v1",
        "model": args.model,
        "calibration": {
            "n_calib_samples": int(args.n_calib_samples),
            "seqlen": int(args.calib_seqlen),
            "split": args.calib_split,
            "seed": int(args.calib_seed),
            "kl_scope": args.kl_scope,
        },
        "seed_assignment": str(args.seed_assignment),
        "polish_json": str(args.polish_json),
        "block_clado": str(args.block_clado),
        "rows": list(rows),
    }


def _write_checkpoint(output: Path, args, rows: Sequence[Mapping]) -> None:
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(_payload(args, rows), indent=2) + "\n")
    os.replace(tmp, output)


def _load_resume_rows(output: Path) -> list[dict]:
    if not output.is_file():
        return []
    payload = json.loads(output.read_text())
    if payload.get("schema") != "prismaquant.validate_polish_flips.v1":
        raise ValueError(f"unsupported resume schema in {output}")
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"resume file {output} has invalid rows")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _measure_points(
    *,
    model,
    points: Sequence[Mapping],
    units: Sequence[bc.DecisionUnit],
    total_params: int,
    calib_ids: torch.Tensor,
    ref_log_probs,
    profile,
    production_weight_cache,
    work_root: Path,
    kl_scope: str,
    include_activation_quant: bool,
    weight_session_snapshot_dir: str | None,
    resume_rows: Sequence[Mapping] | None = None,
    checkpoint_callback=None,
) -> list[dict]:
    rows: list[dict] = []
    completed = {
        str(row.get("label")): dict(row)
        for row in (resume_rows or [])
        if row.get("label") is not None and row.get("real_kl") is not None
    }
    weight_session = None
    env_cm = (
        _temporary_env("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1")
        if production_weight_cache is not None else nullcontext()
    )
    with env_cm:
        if production_weight_cache is not None:
            from prismaquant.weight_session import WeightSession

            weight_session = WeightSession(
                model,
                production_weight_cache=production_weight_cache,
                snapshot_dir=weight_session_snapshot_dir,
            )
            n_prefetch, n_loaded = _prefetch_assignment_delta(
                production_weight_cache,
                bf16_assignment_for_units(units),
                points[0]["assignment"],
            )
            weight_session.initialize(points[0]["assignment"], units)
            print(
                "[validate-flips] weight session initialized "
                f"prefetch={n_prefetch}/{n_loaded} "
                f"{weight_session.diagnostics()}",
                flush=True,
            )
        try:
            prev_kl = None
            for idx, point in enumerate(points):
                assignment = dict(point["assignment"])
                if weight_session is not None:
                    if idx == 0:
                        changed = 0
                    else:
                        current_assignment = weight_session.current_assignment()
                        n_prefetch, n_loaded = _prefetch_assignment_delta(
                            production_weight_cache,
                            current_assignment,
                            assignment,
                        )
                        changed = weight_session.apply_assignment(assignment)
                        print(
                            "[validate-flips] applied assignment delta "
                            f"{point['label']} changed={changed} "
                            f"prefetch={n_prefetch}/{n_loaded}",
                            flush=True,
                        )
                else:
                    changed = None
                existing = completed.get(str(point["label"]))
                if existing is not None:
                    rows.append(existing)
                    prev_kl = float(existing["real_kl"])
                    print(
                        "[validate-flips] resumed "
                        f"{existing['label']} bpp={float(existing['bpp']):.4f} "
                        f"kl={float(existing['real_kl']):.8f}",
                        flush=True,
                    )
                    continue
                start = time.time()
                kl = measure_assignment_kl(
                    model,
                    assignment,
                    calib_ids,
                    ref_log_probs,
                    work_root=work_root,
                    profile=profile,
                    use_frozen_weight_cache=production_weight_cache is None,
                    production_weight_cache=production_weight_cache,
                    rng_seed=0,
                    kl_scope=kl_scope,
                    include_activation_quant=include_activation_quant,
                    stream_ref_log_probs=kl_scope == "full_sequence",
                )
                bits = cdp._assignment_bits(units, assignment)
                row = {
                    "label": point["label"],
                    "step_index": point["step_index"],
                    "unit": point["unit"],
                    "from_fmt": point["from_fmt"],
                    "to_fmt": point["to_fmt"],
                    "bpp": bits / float(total_params) if total_params else 0.0,
                    "real_kl": float(kl),
                    "delta_vs_previous": (
                        None if prev_kl is None else float(kl) - float(prev_kl)
                    ),
                    "improvement_vs_seed": float(rows[0]["real_kl"] - kl) if rows else 0.0,
                    "materialized_changes": changed,
                    "elapsed_seconds": float(time.time() - start),
                    "assignment": assignment,
                }
                rows.append(row)
                prev_kl = float(kl)
                if checkpoint_callback is not None:
                    checkpoint_callback(rows)
                print(
                    "[validate-flips] "
                    f"{row['label']} bpp={row['bpp']:.4f} "
                    f"kl={row['real_kl']:.8f} "
                    f"delta_prev={row['delta_vs_previous']} "
                    f"improve_seed={row['improvement_vs_seed']:.8f}",
                    flush=True,
                )
        finally:
            if weight_session is not None:
                weight_session.apply_assignment({m: "BF16" for u in units for m in u.member_qnames})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed-assignment", required=True)
    parser.add_argument("--polish-json", required=True)
    parser.add_argument("--block-clado", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-calib-samples", type=int, default=64)
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--kl-scope", choices=["last_token", "full_sequence"], default="last_token")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="kernels-community/flash-attn2")
    parser.add_argument("--production-weight-cache", default=None)
    parser.add_argument("--production-cache-dir-override", default=None)
    parser.add_argument("--production-cache-lru-gb", type=float, default=16.0)
    parser.add_argument("--weight-session-snapshot-dir", default=os.environ.get("PRISMAQUANT_WEIGHT_SESSION_SNAPSHOT_DIR"))
    parser.add_argument("--no-activation-quant", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    units, total_params = _load_units(args.block_clado)
    seed_assignment = load_assignment_json(args.seed_assignment)
    polish_payload = json.loads(Path(args.polish_json).read_text())
    points = _cumulative_polish_points(
        seed_assignment=seed_assignment,
        polish_payload=polish_payload,
        units=units,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    resume_rows = [] if args.no_resume else _load_resume_rows(output)
    if resume_rows:
        print(
            f"[validate-flips] resuming {len(resume_rows)} rows from {output}",
            flush=True,
        )
    work_root = Path(tempfile.mkdtemp(prefix="prismaquant_validate_flips_"))
    staged, cleanup = stage_multimodal(args.model)
    try:
        local_only = bool(args.local_files_only or Path(staged).exists())
        print(
            "[validate-flips] loading tokenizer and calibration "
            f"n={args.n_calib_samples} seqlen={args.calib_seqlen} "
            f"scope={args.kl_scope}",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            staged,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        calib_ids = load_wikitext_calibration_windowed(
            tokenizer,
            args.n_calib_samples,
            args.calib_seqlen,
            split=args.calib_split,
            seed=args.calib_seed,
        )
        dtype = _dtype_from_name(args.dtype)
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": local_only,
            "attn_implementation": args.attn_implementation,
        }
        if device == "cuda":
            load_kwargs["device_map"] = "cuda"
        print(
            "[validate-flips] loading model "
            f"device={device} dtype={args.dtype} attn={args.attn_implementation}",
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        if device != "cuda":
            model.to(device)
        model.eval()
        try:
            profile = detect_profile(args.model)
        except Exception:
            profile = DefaultProfile()
        model_device = next(model.parameters()).device
        print("[validate-flips] caching reference log-probs", flush=True)
        ref_log_probs = cache_reference_log_probs(
            model,
            calib_ids,
            model_device,
            kl_scope=args.kl_scope,
        )
        print("[validate-flips] loading production cache", flush=True)
        production_weight_cache = _load_production_cache(args)
        rows = _measure_points(
            model=model,
            points=points,
            units=units,
            total_params=total_params,
            calib_ids=calib_ids,
            ref_log_probs=ref_log_probs,
            profile=profile,
            production_weight_cache=production_weight_cache,
            work_root=work_root,
            kl_scope=args.kl_scope,
            include_activation_quant=not bool(args.no_activation_quant),
            weight_session_snapshot_dir=args.weight_session_snapshot_dir,
            resume_rows=resume_rows,
            checkpoint_callback=lambda current_rows: _write_checkpoint(
                output,
                args,
                current_rows,
            ),
        )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)
        shutil.rmtree(work_root, ignore_errors=True)

    _write_checkpoint(output, args, rows)
    print(f"[validate-flips] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
