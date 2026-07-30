"""Measure real KL for one or more assignment JSON files.

Scope is ``--kl-scope``. The CLI default is ``last_token`` (a triage SCREEN,
CLAUDE.md §5) for ad-hoc/probe-gate parity, but the pipeline passes
``full_sequence`` — the gold-metric scope — for frontier selection
(M26; ``run-pipeline.sh:269,1208``, default ``VALIDATED_FRONTIER_KL_SCOPE=
full_sequence``). Note the summary key stays ``last_token_kl`` under both
scopes for backwards compatibility (:645).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch

from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    kl_divergence,
    stage_multimodal,
)
from prismaquant.calibration_data import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.layer_config import canonicalize_format
from prismaquant.model_profiles import detect_profile_with_warning
from prismaquant.kl_measurement import (
    assignment_bit_total,
    assignment_hash,
    measure_assignment_kl,
)
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    build_quantizable_map,
    calibration_data_hash,
)
from prismaquant.schemas import validate_cost_payload
from prismaquant.sensitivity_probe import load_calibration
from prismaquant.source_prefetch import prefetch_safetensors_checkpoint


def _load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def _load_probe_stats(path: str | Path) -> dict:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, Mapping) and isinstance(payload.get("stats"), Mapping):
        return dict(payload["stats"])
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError(f"probe file {path} does not contain a stats mapping")


def _load_costs(path: str | Path) -> dict:
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    validate_cost_payload(payload, str(path))
    return dict(payload["costs"])


def load_assignment_json(path: str | Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    payload = _load_json(path)
    if isinstance(payload, Mapping) and isinstance(payload.get("assignment"), Mapping):
        assignment = {str(k): canonicalize_format(v) for k, v in payload["assignment"].items()}
    elif isinstance(payload, Mapping):
        assignment = {str(k): canonicalize_format(v) for k, v in payload.items()}
    else:
        raise ValueError(f"unsupported assignment JSON shape: {path}")
    if base is not None:
        merged = {str(k): canonicalize_format(v) for k, v in base.items()}
        merged.update(assignment)
        return merged
    return assignment


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.stem, path

def _profile_excludes_bpp_name(name: str, fmt: str, profile) -> bool:
    del fmt
    if profile is None:
        return False
    is_pinned = getattr(profile, "is_pinned_name", None)
    if callable(is_pinned) and bool(is_pinned(name)):
        return True
    passthrough_prefixes = getattr(profile, "source_passthrough_prefixes", None)
    if callable(passthrough_prefixes):
        for raw_prefix in passthrough_prefixes():
            prefix = str(raw_prefix)
            if not prefix:
                continue
            if name == prefix.rstrip(".") or name.startswith(prefix):
                return True
    return False


def _assignment_bpp_details(
    stats: Mapping,
    assignment: Mapping[str, str],
    specs_by_name: Mapping[str, fr.FormatSpec],
    *,
    profile=None,
) -> dict[str, float | int]:
    names = [
        name for name, fmt in assignment.items()
        if name in stats and not _profile_excludes_bpp_name(str(name), str(fmt), profile)
    ]
    total_params = sum(int(stats[name].get("n_params", 0) or 0) for name in names)
    if total_params <= 0:
        return {
            "bpp": 0.0,
            "quantizable_entries": 0,
            "excluded_entries": sum(1 for name in assignment if name in stats),
            "quantizable_params": 0,
        }
    filtered_assignment = {name: assignment[name] for name in names}
    return {
        "bpp": assignment_bit_total(stats, filtered_assignment, specs_by_name) / float(total_params),
        "quantizable_entries": len(names),
        "excluded_entries": sum(
            1 for name, fmt in assignment.items()
            if name in stats and _profile_excludes_bpp_name(str(name), str(fmt), profile)
        ),
        "quantizable_params": total_params,
    }


def _assignment_bpp(
    stats: Mapping,
    assignment: Mapping[str, str],
    specs_by_name: Mapping[str, fr.FormatSpec],
    *,
    profile=None,
) -> float:
    return float(
        _assignment_bpp_details(
            stats,
            assignment,
            specs_by_name,
            profile=profile,
        )["bpp"]
    )


def _lookup_cost_entry(costs: Mapping, name: str, fmt: str) -> Mapping | None:
    per_name = costs.get(name)
    if not isinstance(per_name, Mapping):
        return None
    candidates = [str(fmt)]
    try:
        candidates.extend(fr.aliases_for(str(fmt)))
    except Exception:
        candidates.append(fr.canonical_format_name(str(fmt)))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        entry = per_name.get(key)
        if isinstance(entry, Mapping) and "error" not in entry:
            return entry
    return None


def _assignment_cost_summary(
    costs: Mapping,
    assignment: Mapping[str, str],
) -> dict[str, object]:
    """Summarize local cost-table MSE for an assignment.

    These are local render/probe metrics, not end-to-end KL.  BF16 entries are
    counted as zero-error because they preserve the original Linear weights.
    """
    sums = {
        "weight_mse": 0.0,
        "output_mse": 0.0,
        "fisher_output_mse": 0.0,
        "rel_output_mse": 0.0,
        "predicted_dloss": 0.0,
    }
    counts = {
        "weight_mse": 0,
        "output_mse": 0,
        "fisher_output_mse": 0,
        "rel_output_mse": 0,
        "predicted_dloss": 0,
    }
    missing: list[str] = []
    unmeasured_output = 0
    format_counts: Counter[str] = Counter()
    for name, raw_fmt in assignment.items():
        fmt = fr.canonical_format_name(str(raw_fmt).strip().upper())
        format_counts[fmt] += 1
        if fmt == "BF16":
            for key in ("weight_mse", "output_mse", "rel_output_mse"):
                counts[key] += 1
            continue
        entry = _lookup_cost_entry(costs, str(name), fmt)
        if entry is None:
            missing.append(str(name))
            continue
        if entry.get("output_mse_measured") is False:
            unmeasured_output += 1
        for key in sums:
            value = entry.get(key)
            if value is None:
                continue
            try:
                value_f = float(value)
            except Exception:
                continue
            sums[key] += value_f
            counts[key] += 1
    means = {
        key: (sums[key] / counts[key] if counts[key] else None)
        for key in sums
    }
    return {
        "objective": "local_cost_table_mse",
        "weight_mse_sum": float(sums["weight_mse"]),
        "weight_mse_mean": means["weight_mse"],
        "output_mse_sum": float(sums["output_mse"]),
        "output_mse_mean": means["output_mse"],
        "fisher_output_mse_sum": float(sums["fisher_output_mse"]),
        "fisher_output_mse_mean": means["fisher_output_mse"],
        "rel_output_mse_sum": float(sums["rel_output_mse"]),
        "rel_output_mse_mean": means["rel_output_mse"],
        "predicted_dloss_sum": float(sums["predicted_dloss"]),
        "predicted_dloss_mean": means["predicted_dloss"],
        "counts": dict(counts),
        "formats": dict(format_counts),
        "missing_count": int(len(missing)),
        "missing_sample": missing[:8],
        "output_mse_unmeasured_count": int(unmeasured_output),
    }


def _device_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


@contextmanager
def _temporary_env(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _git_provenance() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except Exception:
        commit = None
    try:
        dirty = bool(subprocess.run(
            ["git", "status", "--short"],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip())
    except Exception:
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _calibration_provenance(calib_repeats: Sequence[torch.Tensor]) -> dict[str, object]:
    repeat_hashes = [calibration_data_hash(ids) for ids in calib_repeats]
    if not repeat_hashes:
        combined = None
    elif len(repeat_hashes) == 1:
        combined = repeat_hashes[0]
    else:
        combined = hashlib.sha256(
            "\n".join(repeat_hashes).encode("utf-8")
        ).hexdigest()
    return {
        "calib_hash": combined,
        "calib_repeat_hashes": repeat_hashes,
    }


def _strict_production_cache_enabled() -> bool:
    return os.environ.get(
        "PRISMAQUANT_STRICT_PRODUCTION_CACHE", "1"
    ) not in ("", "0", "false", "False")


def _expert_lazy_fill_enabled() -> bool:
    """M4: lazily render a frontier point's packed experts (e.g. FP8) into the
    shared cache just before scoring it. The format-menu build renders NVFP4
    experts eagerly; non-eager expert formats are gap-filled per-assignment so
    the validated frontier can SELECT expert format by real KL without an eager
    all-experts FP8 render. Off => the strict gate hard-fails on expert misses
    (legacy behavior)."""
    return os.environ.get(
        "PRISMAQUANT_EXPERT_LAZY_FILL", "1"
    ) not in ("", "0", "false", "False")


def _production_cache_assignment_diagnostics(
    production_cache,
    assignment: Mapping[str, str],
) -> dict[str, object] | None:
    if production_cache is None:
        return None
    if hasattr(production_cache, "assignment_keys"):
        # M4: count packed-MoE expert misses too. The default
        # include_packed_experts=False (a residency-caller convenience) silently
        # skips uncached packed-expert qnames, which made the packed-expert
        # fail-fast below a no-op — the generic materialization raise caught it
        # later with a less actionable message. Counting them here lets the
        # informative M4 hint fire before any ref-logprob caching.
        keys, missing = production_cache.assignment_keys(
            assignment, include_packed_experts=True)
    else:
        keys = []
        missing = []
        for name, fmt in assignment.items():
            fmt_canon = fr.canonical_format_name(str(fmt))
            if fmt_canon == "BF16":
                continue
            tensor = production_cache.get(str(name), fmt_canon)
            if tensor is None:
                missing.append((str(name), fmt_canon))
            else:
                keys.append((str(name), fmt_canon))
    strict = _strict_production_cache_enabled()
    diagnostics: dict[str, object] = {
        "required_entries": int(len(keys) + len(missing)),
        "cache_hit_count": int(len(keys)),
        "cache_miss_count": int(len(missing)),
        "rtn_fallback_count": int(len(missing) if not strict else 0),
        "strict": bool(strict),
    }
    if missing:
        diagnostics["missing_sample"] = [
            [str(name), str(fmt)] for name, fmt in missing[:8]
        ]
    return diagnostics


def _materialize_assignment_inplace(
    model,
    assignment: Mapping[str, str],
    production_cache,
    *,
    progress: bool = False,
    log_prefix: str = "[validate-kl/inplace]",
) -> dict[str, object]:
    """Destructively copy one rendered assignment into the live model.

    This is intentionally one-way: reference logits must be cached before
    calling it, and callers should reload the model for another assignment.
    It avoids whole-assignment hook clone/restore overhead and keeps rendered
    weights flowing through ``ProductionWeightCache`` one tensor at a time.
    """
    quant_map = build_quantizable_map(model)
    copied = 0
    copied_bytes = 0
    format_counts: Counter[str] = Counter()
    missing_model: list[str] = []
    missing_cache: list[tuple[str, str]] = []
    shape_mismatch: list[dict[str, object]] = []
    seen_targets: set[tuple[int, str]] = set()
    start = time.time()
    total_non_bf16 = sum(
        1
        for fmt in assignment.values()
        if fr.canonical_format_name(str(fmt)) != "BF16"
    )
    if progress:
        print(
            f"{log_prefix} materializing {total_non_bf16} rendered weights "
            "into the live model",
            flush=True,
        )
    for name, fmt in assignment.items():
        fmt_canon = fr.canonical_format_name(str(fmt))
        if fmt_canon == "BF16":
            continue
        target = quant_map.get(str(name))
        if target is None:
            missing_model.append(str(name))
            continue
        module, attr = target
        target_key = (id(module), attr)
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        param = getattr(module, attr)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            missing_model.append(str(name))
            continue
        rendered = production_cache.get(str(name), fmt_canon)
        if rendered is None:
            missing_cache.append((str(name), fmt_canon))
            continue
        if tuple(rendered.shape) != tuple(param.shape):
            shape_mismatch.append(
                {
                    "name": str(name),
                    "format": fmt_canon,
                    "rendered_shape": list(rendered.shape),
                    "param_shape": list(param.shape),
                }
            )
            continue
        with torch.no_grad():
            rendered_device = rendered.to(
                device=param.device,
                dtype=param.dtype,
                non_blocking=True,
            )
            param.data.copy_(rendered_device)
        copied += 1
        copied_bytes += int(param.numel() * param.element_size())
        format_counts[fmt_canon] += 1
        if progress and (copied == total_non_bf16 or copied % 64 == 0):
            print(
                f"{log_prefix} materialized {copied}/{total_non_bf16} "
                f"weights ({copied_bytes / 1024**3:.2f} GiB copied)",
                flush=True,
            )
        del rendered
        if "rendered_device" in locals():
            del rendered_device
    if missing_model or missing_cache or shape_mismatch:
        raise RuntimeError(
            "in-place assignment materialization failed: "
            f"missing_model={len(missing_model)} sample={missing_model[:8]} "
            f"missing_cache={len(missing_cache)} sample={missing_cache[:8]} "
            f"shape_mismatch={len(shape_mismatch)} sample={shape_mismatch[:3]}"
        )
    elapsed = time.time() - start
    if progress:
        gib_s = (copied_bytes / 1024**3 / elapsed) if elapsed > 0 else 0.0
        print(
            f"{log_prefix} materialized {copied} weights, "
            f"{copied_bytes / 1024**3:.2f} GiB in {elapsed:.1f}s "
            f"({gib_s:.2f} GiB/s)",
            flush=True,
        )
    return {
        "copied": int(copied),
        "copied_bytes": int(copied_bytes),
        "elapsed_seconds": float(elapsed),
        "format_counts": dict(format_counts),
    }


def _activation_quant_assignment(
    assignment: Mapping[str, str],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, fmt in assignment.items():
        spec = fr.get_format(fmt)
        if spec.act_quant_changes_input:
            out[str(name)] = spec.name
    return out


def _load_calibration_repeats(tokenizer, args) -> list[torch.Tensor]:
    repeats = max(int(args.calib_repeats), 1)
    n_samples = int(args.n_calib_samples)
    skip = max(int(getattr(args, "calib_skip_first", 0) or 0), 0)

    def _load_jsonl(n: int) -> torch.Tensor:
        # --calib-skip-first K: drop the first K windows of the deterministic
        # loader so selection KL is measured on windows DISJOINT from the
        # probe/cost/render calibration (which consumes windows [0, K)).
        # load_calibration is prefix-stable at a fixed seed, so [K, K+n) is
        # token-disjoint from [0, K) by construction. House rule: held-out
        # split is disjoint from cost generation (review criticals C3/C5).
        all_ids = load_calibration(
            tokenizer,
            args.dataset,
            n + skip,
            args.calib_seqlen,
            calib_seed=int(getattr(args, "calib_seed", 42) or 42),
        )
        if all_ids.size(0) < n + skip:
            raise RuntimeError(
                f"calibration source yielded {all_ids.size(0)} windows; "
                f"need {n + skip} (n={n} + skip-first={skip}). Use a larger "
                "corpus or reduce --calib-skip-first."
            )
        return all_ids[skip:]

    if repeats == 1:
        if args.dataset:
            return [_load_jsonl(n_samples)]
        return [load_wikitext_calibration_windowed(
            tokenizer,
            n_samples,
            args.calib_seqlen,
            split=args.calib_split,
            seed=args.calib_seed,
        )]
    if args.dataset:
        all_ids = _load_jsonl(n_samples * repeats)
        if all_ids.size(0) < n_samples * repeats:
            raise RuntimeError(
                f"requested {repeats} calibration repeats of {n_samples} samples, "
                f"but only loaded {all_ids.size(0)} samples"
            )
        return [
            all_ids[idx * n_samples:(idx + 1) * n_samples].contiguous()
            for idx in range(repeats)
        ]
    stride = int(args.calib_repeat_seed_stride)
    return [
        load_wikitext_calibration_windowed(
            tokenizer,
            n_samples,
            args.calib_seqlen,
            split=args.calib_split,
            seed=int(args.calib_seed) + idx * stride,
        )
        for idx in range(repeats)
    ]


def _load_expert_render_calib(tokenizer, args) -> torch.Tensor | None:
    """Calib draw for the M4 lazy packed-expert (FP8) render — the BUILD/render
    split, DISJOINT from the selection split that KL is measured on.

    With ``--calib-skip-first K`` the selection calib is windows ``[K, K+n)``
    (see ``_load_calibration_repeats``); this returns ALL K withheld render
    windows ``[0, K)`` at the validator's seqlen — token-disjoint from
    selection by construction (same-seqlen windowing; do NOT draw at a
    different seqlen or the [0,K)/[K,K+n) split stops being token-disjoint).
    Known volume caveat: the eager NVFP4 rung was rendered on the BUILD's
    calib volume, which can exceed this draw under thin-frontier configs — a
    render-volume asymmetry in the NVFP4-vs-FP8 comparison that the served
    re-validation gate, not this screen, ultimately arbitrates. Without a
    disjoint render split (skip==0) or a dataset it returns ``None`` so the
    caller can fall back to the selection calib WITH a loud in-sample warning
    (the FP8 rung is then in-sample and a reject-FP8 outcome is
    conservative-only — never promote an FP8-favorable frontier point off an
    in-sample render). House rule: held-out split disjoint from cost/render
    generation."""
    skip = max(int(getattr(args, "calib_skip_first", 0) or 0), 0)
    if skip <= 0 or not args.dataset:
        return None
    ids = load_calibration(
        tokenizer, args.dataset, skip, args.calib_seqlen,
        calib_seed=int(getattr(args, "calib_seed", 42) or 42))
    return ids[:skip].contiguous() if ids.size(0) >= 1 else None


def _persist_lazy_expert_renders(
    production_cache,
    cache_path: str,
    *,
    pristine_cache_dir: object = "__unset__",
) -> None:
    """M4: make lazily-rendered packed-expert entries durable.

    The gap-fill registers new ``(qname, fmt)`` keys only on the in-memory
    cache object; downstream consumers (production_recache, export) resolve
    keys via the PICKLED weights dict, so without a re-pickle the shards on
    disk are invisible and a selected FP8-expert frontier point would be
    unshippable — export must reuse the exact bytes real KL selected
    (principle #8). Atomic same-dir replace; never a tempdir on another
    filesystem.

    Validator-session state must NOT leak into the shared build artifact:
    ``pristine_cache_dir`` (captured before ``relocate()``) is restored for
    the dump so ``--production-cache-dir-override`` is not baked in, and any
    ``enable_lru`` bookkeeping is stripped and restored after.
    """
    if hasattr(production_cache, "compact_for_pickle"):
        production_cache.compact_for_pickle()
    saved: dict[str, object] = {}
    lru_defaults = {
        "_lru_max_bytes": 0,
        "_lru_order": None,
        "_lru_paths": None,
        "_lru_bytes": 0,
    }
    for attr, default in lru_defaults.items():
        if hasattr(production_cache, attr):
            saved[attr] = getattr(production_cache, attr)
            setattr(production_cache, attr, default)
    if pristine_cache_dir != "__unset__":
        saved["cache_dir"] = production_cache.cache_dir
        production_cache.cache_dir = pristine_cache_dir
    try:
        path = Path(cache_path)
        tmp = path.with_name(f"{path.name}.m4fill.{os.getpid()}.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(production_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    finally:
        for attr, value in saved.items():
            setattr(production_cache, attr, value)


def _kl_repeat_summary(values: Sequence[float], *, ucb_z: float) -> dict[str, object]:
    vals = [float(value) for value in values]
    if not vals:
        raise ValueError("KL repeat summary received no values")
    mean = sum(vals) / len(vals)
    if len(vals) <= 1:
        std = 0.0
        stderr = 0.0
    else:
        var = sum((value - mean) ** 2 for value in vals) / (len(vals) - 1)
        std = math.sqrt(max(var, 0.0))
        stderr = std / math.sqrt(len(vals))
    return {
        "last_token_kl": float(mean),
        "kl_repeats": vals,
        "kl_repeat_count": len(vals),
        "kl_std": float(std),
        "kl_stderr": float(stderr),
        "kl_ucb": float(mean + float(ucb_z) * stderr),
        "kl_ucb_z": float(ucb_z),
    }


@torch.no_grad()
def _measure_inplace_assignment_kl(
    model,
    assignment: Mapping[str, str],
    calib_ids: torch.Tensor,
    ref_log_probs,
    *,
    work_root: Path,
    profile,
    production_cache,
    kl_scope: str,
    use_cuda_graphs: bool | None,
) -> tuple[float, dict[str, object]]:
    device = next(model.parameters()).device
    cal_hash = calibration_data_hash(calib_ids)
    calib_ids = calib_ids.to(device)
    full_sequence = kl_scope == "full_sequence"
    materialize_stats = _materialize_assignment_inplace(
        model,
        assignment,
        production_cache,
        progress=True,
    )
    hook_assignment = _activation_quant_assignment(assignment)
    hooks = PerturbedActivationCache(
        model,
        hook_assignment,
        Path(tempfile.mkdtemp(prefix="prismaquant_inplace_kl_", dir=str(work_root))),
        input_rows=0,
        cal_hash=cal_hash,
        profile=profile,
        production_weight_cache=production_cache,
        include_activation_quant=True,
        capture_inputs=False,
    )
    missing = [
        name for name in hooks.missing
        if fr.canonical_format_name(hook_assignment.get(name, "BF16")) != "BF16"
    ]
    if missing:
        raise RuntimeError(
            "assignment contains non-BF16 qnames that do not resolve on "
            f"the live model; missing={len(missing)} sample={missing[:8]}"
        )
    if hooks.skipped:
        raise RuntimeError(
            "assignment has conflicting activation-quant formats within at "
            f"least one module; sample={hooks.skipped[:3]}"
        )
    if use_cuda_graphs is None:
        # The in-place path is already one stable model graph, but CUDA graph
        # capture on 27B can exceed the GPU budget. Keep auto conservative.
        use_cuda_graphs = False
    values: list[float] = []
    graph_key = (
        id(model),
        "inplace",
        assignment_hash(assignment),
        kl_scope,
        cal_hash,
    )
    registry = None
    if use_cuda_graphs:
        from prismaquant.kl_measurement import _KL_CUDA_GRAPH_REGISTRY

        registry = _KL_CUDA_GRAPH_REGISTRY

    with _temporary_env("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", "1"):
        hooks.install()
        try:
            for i in range(calib_ids.size(0)):
                batch = calib_ids[i:i + 1]

                def _forward(batch_ids):
                    logits = model(batch_ids).logits
                    if not full_sequence:
                        logits = logits[:, -1:, :]
                    return logits.clone()

                if registry is not None:
                    logits = registry.run(
                        "assignment-kl-inplace-forward",
                        graph_key,
                        _forward,
                        batch,
                        enabled=True,
                        device=device,
                        keepalive=(hooks,),
                    )
                else:
                    logits = _forward(batch)
                teacher = ref_log_probs[i]
                if not full_sequence:
                    teacher = teacher[:, -1:, :]
                teacher = teacher.to(device, non_blocking=True)
                values.append(float(kl_divergence(logits, teacher).item()))
        finally:
            hooks.remove()
    stats = {
        "materialized": materialize_stats,
        "activation_hooks": {
            "plans": len(hooks.plans),
            "capture_inputs": False,
            "external_weight_management": True,
        },
        "cuda_graphs": bool(use_cuda_graphs),
    }
    return sum(values) / max(len(values), 1), stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate assignment JSONs with real KL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument(
        "--costs",
        default=None,
        help="Optional measure_quant_cost pickle. When supplied, each result "
        "includes assignment-level local MSE / predicted-Δloss summaries "
        "from the same cost table the allocator optimized.",
    )
    parser.add_argument("--base-assignment", required=True)
    parser.add_argument(
        "--assignment",
        action="append",
        required=True,
        help="Assignment path or label=path. Solve-result JSONs are overlaid on base.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--formats", default="NVFP4,FP8_DYNAMIC,BF16")
    parser.add_argument("--n-calib-samples", type=int, default=2)
    parser.add_argument("--calib-seqlen", type=int, default=128)
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument(
        "--calib-skip-first", type=int, default=0,
        help="Drop the first K windows of the deterministic calibration "
        "loader before drawing validation windows. Pass the render "
        "calibration's NSAMPLES here to make selection KL token-disjoint "
        "from probe/cost/render calibration (review criticals C3/C5).")
    parser.add_argument(
        "--calib-repeats",
        type=int,
        default=1,
        help=(
            "Number of independent calibration chunks to measure per assignment. "
            "For --dataset, a single larger load is split into repeat chunks; "
            "for WikiText, seeds advance by --calib-repeat-seed-stride."
        ),
    )
    parser.add_argument("--calib-repeat-seed-stride", type=int, default=997)
    parser.add_argument(
        "--kl-ucb-z",
        type=float,
        default=1.0,
        help="Upper-confidence multiplier applied to KL stderr when repeats > 1.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional calibration source accepted by sensitivity_probe "
        "(HF dataset id, .jsonl, or .txt). When omitted, preserves the "
        "historical wikitext-2 windowed loader.",
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--kl-scope",
        choices=("last_token", "full_sequence"),
        default="last_token",
        help="Token scope for KL. last_token is a triage SCREEN (CLAUDE.md "
        "§5); full_sequence is the gold-metric scope and is what run-pipeline "
        "passes for final frontier selection (M26). The full-sequence "
        "reference is streamed (hooks path), so 27B residency is not a blocker. "
        "This CLI default stays last_token for ad-hoc/probe-gate parity.",
    )
    parser.add_argument(
        "--kl-cuda-graphs",
        choices=("auto", "off", "on"),
        default="auto",
        help=(
            "CUDA graph mode for assignment KL replay. Use 'off' for large "
            "resident production-cache validations where graph capture would "
            "exceed the GPU memory budget."
        ),
    )
    parser.add_argument(
        "--assignment-materialization",
        choices=("auto", "hooks", "inplace"),
        default="auto",
        help=(
            "How to replay production-rendered assignments. 'auto' uses the "
            "in-place path for a single production-cache assignment and the "
            "legacy hook path otherwise."
        ),
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument(
        "--source-prefetch",
        choices=("off", "auto", "require"),
        default="require",
        help=(
            "Prefetch local BF16 source safetensors before loading the teacher "
            "model. Default 'require' fails instead of allowing first-forward "
            "NVMe page faults on production KL validation."
        ),
    )
    parser.add_argument(
        "--source-prefetch-max-gb",
        type=float,
        default=0.0,
        help=(
            "Resident byte budget for source safetensors prefetch. 0 derives "
            "the budget from available memory minus --source-prefetch-headroom-gb."
        ),
    )
    parser.add_argument(
        "--source-prefetch-headroom-gb",
        type=float,
        default=16.0,
    )
    parser.add_argument("--source-prefetch-workers", type=int, default=2)
    parser.add_argument("--disable-frozen-weight-cache", action="store_true")
    parser.add_argument(
        "--production-weight-cache",
        default=None,
        help="Optional pickled ProductionWeightCache. When supplied, KL is "
        "measured on the same production-rendered W_tilde path used by export.",
    )
    parser.add_argument(
        "--production-cache-dir-override",
        default=None,
        help="Relocate disk-backed production cache entries to this directory.",
    )
    parser.add_argument(
        "--production-cache-lru-gb",
        type=float,
        default=4.0,
        help="Resident tensor budget for disk-backed production cache use.",
    )
    parser.add_argument(
        "--production-cache-prefetch",
        choices=("auto", "off", "require"),
        default="require",
        help="Preload assignment-required rendered weights before KL replay. "
             "'require' fails instead of allowing an NVMe-bound validation.",
    )
    parser.add_argument(
        "--production-cache-prefetch-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--production-cache-file-prefetch-max-gb",
        type=float,
        default=0.0,
        help=(
            "Resident byte budget for production cache file-page prefetch in "
            "the in-place replay path. 0 derives the budget from available "
            "memory minus --production-cache-file-prefetch-headroom-gb."
        ),
    )
    parser.add_argument(
        "--production-cache-file-prefetch-headroom-gb",
        type=float,
        default=24.0,
    )
    args = parser.parse_args(argv)

    if args.disable_frozen_weight_cache:
        # NB: module-level `os` (do NOT `import os` here — a local import would
        # shadow it as a function-local for all of main(), leaving the
        # module-level references below unbound when this flag is off).
        os.environ["PRISMAQUANT_ASSIGNMENT_KL_FROZEN_WEIGHT_CACHE"] = "0"

    stats = _load_probe_stats(args.probe)
    costs = _load_costs(args.costs) if args.costs else None
    specs = [fr.get_format(part.strip()) for part in args.formats.split(",") if part.strip()]
    specs_by_name = {spec.name: spec for spec in specs}
    specs_by_name.update({fr.canonical_format_name(spec.name): spec for spec in specs})

    base_assignment = load_assignment_json(args.base_assignment)
    labeled_paths = [_parse_labeled_path(value) for value in args.assignment]
    assignments = [
        (label, load_assignment_json(path, base=base_assignment), str(path))
        for label, path in labeled_paths
    ]

    device_str = _device_arg(args.device)
    device = require_cuda_hot_path("validate_assignments_kl", device_str)
    device_str = str(device)
    if args.device_map not in (None, "cuda"):
        raise RuntimeError(
            "validate_assignments_kl requires a CUDA-resident model. CPU/offload "
            f"device_map={args.device_map!r} is not allowed."
        )
    dtype = _dtype_from_name(args.dtype)
    staged, cleanup = stage_multimodal(args.model)
    if args.work_dir:
        work_root = Path(args.work_dir)
    else:
        # Never default to /tmp: it is cleared under OOM on this host
        # (2026-04-23 wiped artifacts mid-run) and this stage keeps live
        # cache/manifest state for multi-hour frontier measurements.
        # mkdtemp honors TMPDIR when set; otherwise fall back to a dir
        # next to the first assignment artifact.
        fallback = os.environ.get("TMPDIR") or str(
            Path(args.assignment[0].split("=", 1)[-1]).resolve().parent
        )
        work_root = Path(tempfile.mkdtemp(
            prefix="prismaquant_validate_kl_", dir=fallback))
    if str(work_root.resolve()).startswith("/tmp"):
        raise RuntimeError(
            f"validate_assignments_kl work root {work_root} is under /tmp, "
            "which is cleared on OOM on this host. Pass --work-dir or set "
            "TMPDIR to a durable path."
        )
    work_root.mkdir(parents=True, exist_ok=True)
    remove_work_root = args.work_dir is None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_kwargs = {
            "trust_remote_code": True,
            "local_files_only": bool(args.local_files_only or Path(staged).exists()),
        }
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": bool(args.local_files_only or Path(staged).exists()),
        }
        if args.device_map:
            load_kwargs["device_map"] = args.device_map
        elif device.type == "cuda":
            load_kwargs["device_map"] = device_str
        tokenizer = AutoTokenizer.from_pretrained(staged, **tokenizer_kwargs)
        calib_repeats = _load_calibration_repeats(tokenizer, args)
        calib_provenance = _calibration_provenance(calib_repeats)
        source_prefetch_stats = prefetch_safetensors_checkpoint(
            staged,
            mode=args.source_prefetch,
            max_resident_bytes=(
                int(float(args.source_prefetch_max_gb) * 1024**3)
                if float(args.source_prefetch_max_gb) > 0
                else None
            ),
            headroom_gb=float(args.source_prefetch_headroom_gb),
            workers=int(args.source_prefetch_workers),
            progress=True,
            log_prefix="[validate-kl/source]",
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        except ValueError as exc:
            if (
                "requires `accelerate`" not in str(exc)
                and "requires accelerate" not in str(exc)
            ):
                raise
            load_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
            if device.type == "cuda":
                model.to(device)
        if not args.device_map and device.type != "cuda":
            model.to(device)
        model.eval()
        model_device = next(model.parameters()).device
        production_cache = None
        pristine_cache_dir: object = "__unset__"
        if args.production_weight_cache:
            import pickle

            with Path(args.production_weight_cache).open("rb") as fh:
                production_cache = pickle.load(fh)
            if args.production_cache_dir_override:
                # Remember the as-pickled cache_dir: if the M4 lazy gap-fill
                # re-pickles this cache, the session's dir override must not
                # be baked into the shared build artifact.
                pristine_cache_dir = getattr(production_cache, "cache_dir", None)
                production_cache.relocate(args.production_cache_dir_override)
            if (
                args.production_cache_lru_gb
                and float(args.production_cache_lru_gb) > 0
                and hasattr(production_cache, "enable_lru")
            ):
                production_cache.enable_lru(
                    int(float(args.production_cache_lru_gb) * 1024**3)
                )
        materialization_mode = args.assignment_materialization
        if materialization_mode == "auto":
            if production_cache is not None and len(assignments) == 1:
                materialization_mode = "inplace"
            else:
                materialization_mode = "hooks"
        if materialization_mode == "inplace":
            if production_cache is None:
                raise RuntimeError(
                    "--assignment-materialization=inplace requires "
                    "--production-weight-cache"
                )
            if len(assignments) != 1:
                raise RuntimeError(
                    "--assignment-materialization=inplace is destructive and "
                    "supports exactly one assignment per model load; run "
                    "multiple assignments as separate validator invocations."
                )
            if float(args.production_cache_lru_gb) <= 0:
                raise RuntimeError(
                    "in-place production-cache validation requires a bounded "
                    "--production-cache-lru-gb budget"
                )

        profile = detect_profile_with_warning(
            args.model,
            entrypoint="validate-kl",
        )
        ref_log_prob_repeats = [
            cache_reference_log_probs(
                model,
                calib_ids,
                model_device,
                kl_scope=args.kl_scope,
            )
            for calib_ids in calib_repeats
        ]

        # M4: lazily render each frontier point's packed experts into the
        # shared cache BEFORE the strict gate. The format-menu build renders
        # NVFP4 experts eagerly; this gap-fills the rare Pareto point proposing
        # FP8 (or other non-eager) expert formats, rendering ONLY that point's
        # packed experts (resume-skip no-ops already-cached NVFP4). Keeps FP8
        # measurable so real-KL rejects it (route-flip floor) without the eager
        # ~64 GB / ~1 hr all-experts FP8 render. Disable: PRISMAQUANT_EXPERT_LAZY_FILL=0.
        if (
            production_cache is not None
            and _expert_lazy_fill_enabled()
            and hasattr(production_cache, "assignment_keys")
        ):
            from prismaquant.production_weight_cache import (
                fill_packed_expert_cache_entries,
                is_uncached_packed_expert_qname,
            )
            # Render the lazy experts on the BUILD/render split (DISJOINT from
            # the selection calib KL is measured on), matching the eager NVFP4
            # render — else the FP8 rung is fit in-sample to the split it is
            # selected on. NOTE: like the eager NVFP4 render this omits the
            # cross-domain do-no-harm gate the assignment-mode ship recipe uses;
            # the frontier compares NVFP4 vs FP8 under the same no-gate recipe,
            # but ship export must REUSE this cache (not re-render assignment-
            # mode) or the selection rides on different bytes (principle #8).
            _expert_render_ids = _load_expert_render_calib(tokenizer, args)
            if _expert_render_ids is None:
                _expert_render_ids = calib_repeats[0]
                print(
                    "[validate-kl/M4] WARNING: lazy expert render uses the "
                    "SELECTION calib (no --calib-skip-first render split); the "
                    "FP8 expert rung is IN-SAMPLE — a reject-FP8 outcome is "
                    "conservative-only; do not promote an FP8-favorable "
                    "frontier point off this measurement.",
                    flush=True,
                )
            _lazy_filled_total = 0
            for _label, _assignment, _path in assignments:
                _keys, _missing = production_cache.assignment_keys(
                    _assignment, include_packed_experts=True)
                _expert_miss = {
                    str(n) for n, _f in _missing
                    if is_uncached_packed_expert_qname(str(n))
                }
                _expert_ra = {
                    n: f for n, f in _assignment.items()
                    if str(n) in _expert_miss
                    and fr.canonical_format_name(str(f)) != "BF16"
                }
                if not _expert_ra:
                    continue
                print(
                    f"[validate-kl/M4] '{_label}': lazy-rendering "
                    f"{len(_expert_ra)} packed-expert tensor(s) at "
                    f"{sorted(set(_expert_ra.values()))} (gap-fill)",
                    flush=True,
                )
                # Each call re-captures activations for this point's modules
                # (not shared across points); fine since FP8-expert points are
                # rare near the knee.
                _cov = fill_packed_expert_cache_entries(
                    production_cache, model, _expert_render_ids,
                    render_assignment=_expert_ra,
                    levers=production_cache.levers, profile=profile,
                    cache_dir=getattr(production_cache, "cache_dir", None),
                    render_mode="batched",
                )
                _lazy_filled_total += len(_cov)
                if len(_cov) < len(_expert_ra):
                    print(
                        f"[validate-kl/M4] WARNING: '{_label}': gap-fill "
                        f"rendered only {len(_cov)}/{len(_expert_ra)} "
                        "requested packed-expert tensors (unpacked-expert "
                        "model or live/recipe name mismatch) — the strict "
                        "gate below will fail on the remainder.",
                        flush=True,
                    )
            if _lazy_filled_total:
                # Persist BEFORE measurement: shards without pickled keys are
                # invisible to recache/export (they resolve via the pickled
                # weights dict), which would make a selected FP8-expert point
                # unshippable. Failing here aborts pre-KL, not mid-KL.
                _persist_lazy_expert_renders(
                    production_cache, args.production_weight_cache,
                    pristine_cache_dir=pristine_cache_dir)
                print(
                    f"[validate-kl/M4] persisted {_lazy_filled_total} "
                    f"lazy-rendered packed-expert entr"
                    f"{'y' if _lazy_filled_total == 1 else 'ies'} to "
                    f"{args.production_weight_cache}",
                    flush=True,
                )

        if production_cache is not None and _strict_production_cache_enabled():
            # Fail-fast BEFORE any measurement: with the strict default
            # (M6), a cache missing required renders would otherwise abort
            # mid-KL after minutes-to-hours. After the M4 lazy expert gap-fill
            # above, any REMAINING miss is a non-expert (or lazy-fill-disabled)
            # packed-expert gap — the hint distinguishes them.
            for _label, _assignment, _path in assignments:
                _diag = _production_cache_assignment_diagnostics(
                    production_cache, _assignment)
                if _diag and _diag.get("cache_miss_count"):
                    _sample = _diag.get("missing_sample") or []
                    from prismaquant.production_weight_cache import (
                        is_uncached_packed_expert_qname,
                    )
                    _expert = [m for m in _sample
                               if is_uncached_packed_expert_qname(str(m[0]))]
                    _hint = (
                        (
                            " Missing entries include packed-MoE experts: "
                            "the lazy gap-fill RAN but could not render them "
                            "(unpacked-expert model, or the profile's expert "
                            "live/recipe naming does not round-trip through "
                            "cache resolve_key). Use SELECTION_MODE=surrogate "
                            "(--render-scope assignment) for this model, or "
                            "fix the profile naming. "
                            if _expert_lazy_fill_enabled() else
                            " Missing entries include packed-MoE experts: "
                            "lazy expert gap-fill is OFF "
                            "(PRISMAQUANT_EXPERT_LAZY_FILL=0), so non-eager "
                            "expert formats (e.g. FP8) were not rendered. "
                            "Re-enable it (default 1) to gap-fill each "
                            "Pareto point's experts just-in-time, or use "
                            "SELECTION_MODE=surrogate "
                            "(--render-scope assignment). "
                        )
                        + "PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 falls "
                        "back to RTN for research runs only."
                        if _expert else
                        " Rebuild the cache to cover the assignment, or set "
                        "PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 (research "
                        "only — RTN fallback)."
                    )
                    raise RuntimeError(
                        f"[validate-kl] assignment '{_label}' requires "
                        f"{_diag['cache_miss_count']} production-cache "
                        f"renders the cache lacks (sample={_sample[:4]})."
                        + _hint
                    )

        results = []
        for label, assignment, path in assignments:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            prefetch_stats = None
            cache_diagnostics = _production_cache_assignment_diagnostics(
                production_cache,
                assignment,
            )
            if (
                production_cache is not None
                and args.production_cache_prefetch != "off"
            ):
                if materialization_mode == "inplace":
                    prefetch_stats = production_cache.prefetch_assignment_file_pages(
                        assignment,
                        mode=args.production_cache_prefetch,
                        max_resident_bytes=(
                            int(float(args.production_cache_file_prefetch_max_gb) * 1024**3)
                            if float(args.production_cache_file_prefetch_max_gb) > 0
                            else None
                        ),
                        headroom_gb=float(
                            args.production_cache_file_prefetch_headroom_gb
                        ),
                        max_workers=args.production_cache_prefetch_workers,
                        progress=True,
                        log_prefix="[validate-kl/prod-cache-files]",
                    )
                else:
                    preload_budget = (
                        getattr(production_cache, "_lru_max_bytes", 0) or None
                    )
                    prefetch_stats = production_cache.prefetch_assignment(
                        assignment,
                        max_resident_bytes=preload_budget,
                        max_workers=args.production_cache_prefetch_workers,
                        require=args.production_cache_prefetch == "require",
                        progress=True,
                        log_prefix="[validate-kl]",
                    )
            kl_values: list[float] = []
            replay_runs: list[dict[str, object]] = []
            for repeat_idx, (calib_ids, ref_log_probs) in enumerate(
                zip(calib_repeats, ref_log_prob_repeats, strict=True)
            ):
                if materialization_mode == "inplace":
                    kl_value, replay_stats = _measure_inplace_assignment_kl(
                        model,
                        assignment,
                        calib_ids,
                        ref_log_probs,
                        work_root=work_root,
                        profile=profile,
                        production_cache=production_cache,
                        kl_scope=args.kl_scope,
                        use_cuda_graphs=(
                            None if args.kl_cuda_graphs == "auto"
                            else args.kl_cuda_graphs == "on"
                        ),
                    )
                else:
                    kl_value = measure_assignment_kl(
                        model,
                        assignment,
                        calib_ids,
                        ref_log_probs,
                        work_root=work_root,
                        profile=profile,
                        use_frozen_weight_cache=not args.disable_frozen_weight_cache,
                        production_weight_cache=production_cache,
                        use_cuda_graphs=(
                            None if args.kl_cuda_graphs == "auto"
                            else args.kl_cuda_graphs == "on"
                        ),
                        kl_scope=args.kl_scope,
                        stream_ref_log_probs=args.kl_scope == "full_sequence",
                    )
                    replay_stats = {"mode": "hooks"}
                kl_values.append(float(kl_value))
                replay_runs.append({
                    "repeat": int(repeat_idx),
                    **dict(replay_stats),
                })
            kl_summary = _kl_repeat_summary(kl_values, ucb_z=float(args.kl_ucb_z))
            kl = float(kl_summary["last_token_kl"])
            replay_stats = {
                "mode": materialization_mode,
                "repeats": replay_runs,
            }
            counts = dict(Counter(assignment.values()))
            changed = sum(
                1
                for name, fmt in assignment.items()
                if base_assignment.get(name) != fmt
            )
            bpp_details = _assignment_bpp_details(
                stats,
                assignment,
                specs_by_name,
                profile=profile,
            )
            result = {
                "label": label,
                "path": path,
                **kl_summary,
                "bpp": float(bpp_details["bpp"]),
                "bpp_quantizable_entries": int(bpp_details["quantizable_entries"]),
                "bpp_excluded_entries": int(bpp_details["excluded_entries"]),
                "bpp_quantizable_params": int(bpp_details["quantizable_params"]),
                "format_counts": counts,
                "changed_vs_base": int(changed),
                "assignment_entries": len(assignment),
                "assignment_hash": assignment_hash(assignment),
                "kl_scope": args.kl_scope,
                "assignment_materialization": materialization_mode,
                "replay": replay_stats,
            }
            if costs is not None:
                result["mse"] = _assignment_cost_summary(costs, assignment)
            if cache_diagnostics is not None:
                result["production_cache_diagnostics"] = cache_diagnostics
            if prefetch_stats is not None:
                result["production_cache_prefetch"] = prefetch_stats
            results.append(result)
            mse_msg = ""
            if costs is not None:
                mse = result["mse"]
                mse_msg = (
                    f" output_mse={mse['output_mse_sum']:.6g}"
                    f" pred_dloss={mse['predicted_dloss_sum']:.6g}"
                    f" mse_missing={mse['missing_count']}"
                )
            print(
                f"[validate-kl] {label}: KL={kl:.8g} "
                f"bpp={result['bpp']:.6f}{mse_msg} "
                f"changed={changed} counts={counts}",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        git = _git_provenance()
        out = {
            "model": args.model,
            "probe": args.probe,
            "costs": args.costs,
            "base_assignment": args.base_assignment,
            "base_assignment_hash": assignment_hash(base_assignment),
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "formats": [spec.name for spec in specs],
            "calibration": {
                "n_calib_samples": int(args.n_calib_samples),
                "calib_seqlen": int(args.calib_seqlen),
                "calib_split": args.calib_split,
                "calib_seed": int(args.calib_seed),
                "calib_repeats": int(args.calib_repeats),
                "calib_repeat_seed_stride": int(args.calib_repeat_seed_stride),
                "dataset": args.dataset,
                "kl_scope": args.kl_scope,
                **calib_provenance,
            },
            "kl_cuda_graphs": args.kl_cuda_graphs,
            "assignment_materialization": materialization_mode,
            "production_cache": {
                "path": args.production_weight_cache,
                "cache_dir_override": args.production_cache_dir_override,
                "lru_gb": float(args.production_cache_lru_gb),
                "prefetch": args.production_cache_prefetch,
                "prefetch_workers": int(args.production_cache_prefetch_workers),
                "file_prefetch_max_gb": float(
                    args.production_cache_file_prefetch_max_gb
                ),
                "file_prefetch_headroom_gb": float(
                    args.production_cache_file_prefetch_headroom_gb
                ),
            } if args.production_weight_cache else None,
            "source_prefetch": source_prefetch_stats,
            "results": results,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[validate-kl] wrote {output}", flush=True)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        if remove_work_root:
            shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
