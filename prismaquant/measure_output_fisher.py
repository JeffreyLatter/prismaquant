"""Output-Fisher form of Block-CLADO measurement.

The four-term identity in :mod:`measure_block_clado` requires
``O(|𝔹|² · I²)`` forward passes — one per ``(layer_a, layer_b, fmt_a,
fmt_b)`` tuple — to populate the cross-layer interaction matrix.  At LLM
scale this is tractable only after restricting to within-block edges
(Block-CLADO).  This module replaces the four-term collector with the
**Output-Fisher** form, which factorises the same quadratic surrogate
through the closed-form per-token Fisher of the teacher distribution.

Math
----

For a teacher distribution ``p_t = softmax(z_teacher(t))`` and a student
that differs from the teacher only by a logit perturbation ``δz``,

.. math::

    KL(p_t \\| p_t \\oplus δz) = \\frac{1}{2}\\, δz^\\top F_z(t) δz
                                + O(δz^3)

where ``F_z(t) = diag(p_t) - p_t p_t^\\top`` is the categorical Fisher
matrix.  Because ``F_z`` factorises elementwise,

.. math::

    δz^\\top F_z δz = \\mathrm{Var}_{p_t}(δz),
    \\qquad δz_a^\\top F_z δz_b = \\mathrm{Cov}_{p_t}(δz_a, δz_b)

where ``Var_p(x) = E_p[x^2] - E_p[x]^2``.

Concretely we cache, for every fused-sibling decision unit ``U`` and
every non-BF16 format ``f``, the **output-logit perturbation**

.. math::

    δz_{U,f}(t) = z\\big(\\text{student with } U → f\\big)(t) - z_{teacher}(t)

That requires ``|𝔹| · I`` total forward passes — ``O(|𝔹| · I)`` instead
of ``O(|𝔹|^2 · I^2)``.  Pair interactions are then computed analytically:

.. math::

    Ω_{ii} & = \\tfrac{1}{2}\\, \\mathbb{E}_t[\\mathrm{Var}_{p_t}(δz_i)] \\\\
    Ω_{ij} & = \\mathbb{E}_t[\\mathrm{Cov}_{p_t}(δz_i, δz_j)]

These match exactly the ``Ω_{ii}, Ω_{ij}`` computed by the four-term
identity at the second-order Taylor approximation; they differ only in
which higher-order terms each method retains (the four-term identity
captures ``O(δw^3)`` and beyond via finite differences; Output-Fisher
keeps only the analytic second-order term).  For typical 4–8 bpp
quantization the second-order term dominates.

Limitations of this MVP
-----------------------

1. **BF16-centered only.**  Sandwich recalibration centers the Taylor
   expansion at a non-trivial assignment ``w_c`` where the gradient is
   non-zero; the second-order Fisher form alone is missing the linear
   ``g_c · Δ`` term in that regime.  Falls back to four-term for
   sandwich measurements (``--center-assignment`` not supported here).
2. **Weight-only perturbation.**  The collector swaps quantized weights
   in/out manually rather than going through ``PerturbedActivationCache``.
   This omits activation quantization from the surrogate.  For a
   deployment-faithful surrogate, use the four-term collector.

Output payload schema
---------------------

The output JSON uses the same ``prismaquant.block_clado.v1`` schema as
the four-term collector, so the rest of the pipeline (sweep, kneedle,
validate, polish, iterate) works without changes.  ``meta.method`` is
set to ``"output_fisher"`` to distinguish artifacts.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch

from prismaquant import block_clado as bc
from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import iter_quantizable_tensors, stage_multimodal
from prismaquant.iterate_perturbed_allocation import calibration_data_hash
from prismaquant.measure_adjoint_l3 import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.measure_block_clado import (
    discover_blocks,
    enumerate_block_pairs,
    _bf16_assignment,
)
from prismaquant.model_profiles import DefaultProfile, detect_profile
from prismaquant.perturbed_x_cache import PerturbedActivationCache


# ---------------------------------------------------------------------------
# Weight perturbation helpers
# ---------------------------------------------------------------------------


def _slug(value: str) -> str:
    out = []
    for c in value:
        if c.isalnum() or c in "-_":
            out.append(c)
        else:
            out.append("_")
    return "".join(out)[:120]


def _quantize_unit_in_place(
    model,
    unit: bc.DecisionUnit,
    fmt: str,
) -> dict[str, torch.Tensor]:
    """Replace each member weight of ``unit`` with its ``fmt`` quantization.

    Returns a dict ``{recipe_name -> original_data}`` so the caller can
    restore the unperturbed weights via :func:`_restore_unit`.
    """
    if fmt == "BF16":
        return {}
    spec = fr.get_format(fmt)
    member_set = set(unit.member_qnames)
    saved: dict[str, torch.Tensor] = {}
    for full_name, mod, attr in iter_quantizable_tensors(model):
        recipe = full_name[:-7] if full_name.endswith(".weight") else full_name
        if recipe not in member_set:
            continue
        param = getattr(mod, attr, None)
        if not isinstance(param, torch.nn.Parameter):
            continue
        original = param.data
        try:
            quantized = spec.quantize_dequantize(original.detach())
        except Exception:
            continue
        saved[recipe] = original
        param.data = quantized.to(device=original.device, dtype=original.dtype)
    return saved


def _restore_unit(model, saved: Mapping[str, torch.Tensor]) -> None:
    if not saved:
        return
    for full_name, mod, attr in iter_quantizable_tensors(model):
        recipe = full_name[:-7] if full_name.endswith(".weight") else full_name
        if recipe in saved:
            getattr(mod, attr).data = saved[recipe]


@torch.no_grad()
def _forward_logits(
    model,
    calib_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Forward each calibration sample; return a list of fp32 ``[T, V]`` tensors on CPU."""
    device = next(model.parameters()).device
    out: list[torch.Tensor] = []
    for i in range(calib_ids.size(0)):
        batch = calib_ids[i:i + 1].to(device)
        logits = model(batch).logits[0].detach().float()
        out.append(logits.cpu())
    return out


@torch.no_grad()
def _forward_logits_with_assignment(
    model,
    assignment: Mapping[str, str],
    calib_ids: torch.Tensor,
    *,
    work_root: Path,
    profile=None,
    use_frozen_weight_cache: bool = False,
) -> list[torch.Tensor]:
    """Run forward with the given quantization assignment applied via
    PerturbedActivationCache, returning per-sample fp32 ``[T, V]`` logits.

    Unlike ``_quantize_unit_in_place`` + ``_forward_logits``, this path
    matches ``measure_assignment_kl``'s deployment-faithful measurement
    that includes activation quantization at any unit assigned to a
    non-BF16 format.
    """
    device = next(model.parameters()).device
    cache_dir = Path(tempfile.mkdtemp(prefix="prismaquant_of_pcache_", dir=str(work_root)))
    cal_hash = calibration_data_hash(calib_ids)
    hooks = PerturbedActivationCache(
        model, assignment, cache_dir,
        input_rows=0, cal_hash=cal_hash, profile=profile,
    )
    out: list[torch.Tensor] = []
    try:
        from contextlib import nullcontext
        cache_cm = nullcontext()
        if use_frozen_weight_cache and hooks._frozen_weight_cache is None:
            cache_cm = hooks.frozen_weight_cache()
        with cache_cm:
            hooks.install()
            try:
                for i in range(calib_ids.size(0)):
                    batch = calib_ids[i:i + 1].to(device)
                    logits = model(batch).logits[0].detach().float()
                    out.append(logits.cpu())
            finally:
                hooks.remove()
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return out


# ---------------------------------------------------------------------------
# Fisher quadratic forms (variance / covariance under teacher)
# ---------------------------------------------------------------------------


@torch.no_grad()
def fisher_omega_ii(
    teacher_probs: Sequence[torch.Tensor],
    delta_z: Sequence[torch.Tensor],
    *,
    linear_offset: Sequence[torch.Tensor] | None = None,
) -> float:
    """Return ``Ω_ii`` = (1/2) · mean_t Var_p(δz) [+ linear correction].

    For BF16-centered measurement (``linear_offset=None``), this returns
    the standard ``(1/2) Var_{p_t}(δz)`` second-order Fisher term.

    For sandwich-centered measurement, the Taylor expansion of
    ``KL(p_t ‖ student)`` at the centered student state ``z_c`` has a
    non-zero linear term ``⟨p_c − p_t, δz⟩``.  Pass ``linear_offset =
    (p_c − p_t)`` per sample to include it; the function then returns

        E_t[⟨linear_offset, δz⟩] + (1/2) E_t[Var_{p_c}(δz)]

    which equals ``KL(p_t ‖ student_perturbed) − KL(p_t ‖ student_c)``
    at second order — exactly what the four-term identity measures.
    """
    total_quad = 0.0
    total_lin = 0.0
    total_tokens = 0
    for idx, (p, dz) in enumerate(zip(teacher_probs, delta_z)):
        p32 = p.float()
        dz32 = dz.float()
        e_dz = (p32 * dz32).sum(dim=-1)
        e_dz2 = (p32 * dz32 * dz32).sum(dim=-1)
        var = e_dz2 - e_dz * e_dz
        total_quad += float(var.sum())
        total_tokens += var.numel()
        if linear_offset is not None:
            offset = linear_offset[idx].float()
            lin = (offset * dz32).sum(dim=-1)
            total_lin += float(lin.sum())
    if total_tokens == 0:
        return 0.0
    return (total_lin + 0.5 * total_quad) / total_tokens


@torch.no_grad()
def fisher_omega_ij(
    teacher_probs: Sequence[torch.Tensor],
    delta_z_a: Sequence[torch.Tensor],
    delta_z_b: Sequence[torch.Tensor],
) -> float:
    """Return ``mean_t Cov_{p_t}(δz_a, δz_b)`` averaged over tokens."""
    total_cov = 0.0
    total_tokens = 0
    for p, dz_a, dz_b in zip(teacher_probs, delta_z_a, delta_z_b):
        p32 = p.float()
        a32 = dz_a.float()
        b32 = dz_b.float()
        e_ab = (p32 * a32 * b32).sum(dim=-1)
        e_a = (p32 * a32).sum(dim=-1)
        e_b = (p32 * b32).sum(dim=-1)
        cov = e_ab - e_a * e_b
        total_cov += float(cov.sum())
        total_tokens += cov.numel()
    if total_tokens == 0:
        return 0.0
    return total_cov / total_tokens


# ---------------------------------------------------------------------------
# Top-level collector
# ---------------------------------------------------------------------------


def collect_output_fisher(
    model,
    calib_ids: torch.Tensor,
    formats: Sequence[fr.FormatSpec],
    *,
    profile=None,
    cache_dir: str | Path | None = None,
    keep_disk_cache: bool = False,
    delta_z_dtype: torch.dtype = torch.float16,
    progress_callback: Callable[[dict], None] | None = None,
    skip_pairs: bool = False,
    include_activation_quant: bool = True,
    use_frozen_weight_cache: bool = False,
    center_assignment: Mapping[str, str] | None = None,
) -> dict:
    """Build the Output-Fisher Block-CLADO payload.

    Returns a dict in the ``prismaquant.block_clado.v1`` schema (so the
    rest of the pipeline can consume it unchanged).  ``meta.method`` is
    set to ``"output_fisher"`` for traceability.
    """
    if not isinstance(calib_ids, torch.Tensor) or calib_ids.dim() != 2:
        raise ValueError("calib_ids must be a 2D tensor [samples, seqlen]")

    spec_by_name = {fr.canonical_format_name(s.name): s for s in formats}
    if "BF16" not in spec_by_name:
        spec_by_name["BF16"] = fr.get_format("BF16")
    specs_sorted = [spec_by_name[name] for name in sorted(spec_by_name)]

    blocks, singletons, _n_params = discover_blocks(model, profile, specs_sorted)
    units_by_name: dict[str, bc.DecisionUnit] = {
        u.name: u for ulist in blocks.values() for u in ulist
    }
    singletons_by_name: dict[str, bc.DecisionUnit] = {u.name: u for u in singletons}
    if not units_by_name and not singletons_by_name:
        raise RuntimeError("no quantizable units discovered in model")

    all_units = list(units_by_name.values()) + list(singletons_by_name.values())

    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="prismaquant_of_cache_"))
        cleanup_cache_dir = not keep_disk_cache
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cleanup_cache_dir = False

    start = time.time()
    is_sandwich = center_assignment is not None and any(
        fr.canonical_format_name(str(fmt)) != "BF16"
        for fmt in center_assignment.values()
    )
    try:
        # ---------------------------------------------------------------
        # Phase 1: teacher logits + teacher probs (always needed for the
        # linear correction term in sandwich mode).
        # ---------------------------------------------------------------
        if progress_callback is not None:
            progress_callback({
                "event": "teacher_forward_start",
                "include_activation_quant": include_activation_quant,
                "centered": bool(is_sandwich),
            })
        z_teacher = _forward_logits(model, calib_ids)  # list of [T, V] fp32 (CPU)
        teacher_probs: list[torch.Tensor] = [
            torch.softmax(z, dim=-1) for z in z_teacher
        ]
        if progress_callback is not None:
            progress_callback({
                "event": "teacher_forward_done",
                "n_samples": len(z_teacher),
            })

        # Build BF16-everywhere base for the perturbed-cache path (used
        # when centering at BF16 or when we need a clean baseline).
        bf16_base: dict[str, str] = {}
        for unit in all_units:
            for member in unit.member_qnames:
                bf16_base[member] = "BF16"

        # ---------------------------------------------------------------
        # Phase 1b: sandwich state (z_c, p_c, KL(p_t ‖ p_c), linear offset)
        # ---------------------------------------------------------------
        if is_sandwich:
            # Use the user-supplied centered assignment as the base.
            sandwich_base: dict[str, str] = {}
            for member, fmt in center_assignment.items():
                sandwich_base[str(member)] = fr.canonical_format_name(str(fmt))
            # Fill any missing members with BF16
            for unit in all_units:
                for member in unit.member_qnames:
                    sandwich_base.setdefault(member, "BF16")
            if progress_callback is not None:
                progress_callback({"event": "centered_forward_start"})
            z_centered = _forward_logits_with_assignment(
                model, sandwich_base, calib_ids,
                work_root=cache_dir,
                profile=profile,
                use_frozen_weight_cache=use_frozen_weight_cache,
            )
            centered_probs: list[torch.Tensor] = [
                torch.softmax(z, dim=-1) for z in z_centered
            ]
            # Compute KL(p_t || p_centered) for telemetry
            log_p_c = [torch.log(p.clamp(min=1e-30)) for p in centered_probs]
            log_p_t = [torch.log(p.clamp(min=1e-30)) for p in teacher_probs]
            total_kl = 0.0
            total_tokens = 0
            for p_t_i, lp_t_i, lp_c_i in zip(teacher_probs, log_p_t, log_p_c):
                kl_per_token = (p_t_i * (lp_t_i - lp_c_i)).sum(dim=-1)
                total_kl += float(kl_per_token.sum())
                total_tokens += kl_per_token.numel()
            center_kl = total_kl / max(total_tokens, 1)
            # Linear offset for Ω_ii correction: (p_c - p_t) per sample
            linear_offset = [pc - pt for pc, pt in zip(centered_probs, teacher_probs)]
            # Pair Ω_ij math uses centered_probs as the quadratic measure
            quad_probs = centered_probs
            if progress_callback is not None:
                progress_callback({
                    "event": "centered_forward_done",
                    "center_kl": float(center_kl),
                })
            # The base from which we apply per-unit overrides
            base_assignment = sandwich_base
            # Keep z_centered for δz computation
            z_baseline = z_centered
        else:
            sandwich_base = None
            centered_probs = None
            linear_offset = None
            quad_probs = teacher_probs
            center_kl = 0.0
            base_assignment = bf16_base
            z_baseline = z_teacher

        # ---------------------------------------------------------------
        # Phase 2: compute and cache δz_{U,f} for each (unit, format != BF16);
        # compute Ω_ii immediately while δz is in memory.
        # ---------------------------------------------------------------
        omega_ii: dict[tuple[str, str], float] = {}
        delta_z_paths: dict[tuple[str, str], Path] = {}
        # In sandwich mode we override starting from the centered base; the
        # "no-op" format for each unit is whatever it has in the center.
        per_unit_center: dict[str, str] = {}
        for unit in all_units:
            cur_fmt = "BF16"
            for member in unit.member_qnames:
                if base_assignment.get(member, "BF16") != "BF16":
                    cur_fmt = base_assignment[member]
                    break
            per_unit_center[unit.name] = cur_fmt
        n_pert_total = sum(
            sum(1 for opt in u.options if opt.fmt != per_unit_center[u.name])
            for u in all_units
        )
        n_pert_done = 0
        for unit in all_units:
            center_fmt = per_unit_center[unit.name]
            for opt in unit.options:
                if opt.fmt == center_fmt:
                    omega_ii[(unit.name, opt.fmt)] = 0.0
                    continue

                if include_activation_quant or is_sandwich:
                    # Use PerturbedActivationCache so activation quantization is
                    # applied at this unit, matching deployment / four-term.
                    # Required for sandwich (need consistent quant on all
                    # already-non-BF16 units).
                    assignment = dict(base_assignment)
                    canonical_fmt = fr.canonical_format_name(opt.fmt)
                    for member in unit.member_qnames:
                        assignment[member] = canonical_fmt
                    try:
                        z_pert = _forward_logits_with_assignment(
                            model, assignment, calib_ids,
                            work_root=cache_dir,
                            profile=profile,
                            use_frozen_weight_cache=use_frozen_weight_cache,
                        )
                    except Exception as exc:
                        if progress_callback is not None:
                            progress_callback({
                                "event": "perturbation_error",
                                "unit": unit.name,
                                "format": opt.fmt,
                                "error": str(exc),
                            })
                        omega_ii[(unit.name, opt.fmt)] = 0.0
                        continue
                else:
                    # Weight-only fast path (legacy, BF16 base only)
                    saved = _quantize_unit_in_place(model, unit, opt.fmt)
                    if not saved:
                        omega_ii[(unit.name, opt.fmt)] = 0.0
                        continue
                    try:
                        z_pert = _forward_logits(model, calib_ids)
                    finally:
                        _restore_unit(model, saved)

                # δz is measured FROM the baseline (BF16 in normal mode,
                # centered state in sandwich mode).
                delta_z = [(zp - zb).contiguous() for zp, zb in zip(z_pert, z_baseline)]
                omega_ii[(unit.name, opt.fmt)] = fisher_omega_ii(
                    quad_probs, delta_z,
                    linear_offset=linear_offset,
                )

                # Save for pair compute later
                delta_z_path = cache_dir / f"{_slug(unit.name)}__{_slug(opt.fmt)}.pt"
                torch.save(
                    [t.to(delta_z_dtype) for t in delta_z],
                    delta_z_path,
                )
                delta_z_paths[(unit.name, opt.fmt)] = delta_z_path

                n_pert_done += 1
                if progress_callback is not None:
                    progress_callback({
                        "event": "perturbation_done",
                        "unit": unit.name,
                        "format": opt.fmt,
                        "omega_ii": float(omega_ii[(unit.name, opt.fmt)]),
                        "completed": n_pert_done,
                        "total": n_pert_total,
                    })

                del z_pert, delta_z
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ---------------------------------------------------------------
        # Phase 3: per-block intra-block Ω_ij (analytic via covariance)
        # ---------------------------------------------------------------
        pairs_by_block: dict[str, list[bc.BlockPair]] = {}
        if not skip_pairs:
            for block_id, units_in_block in blocks.items():
                # Load this block's δz tensors (stay on CPU, fp32 for compute)
                block_cache: dict[tuple[str, str], list[torch.Tensor]] = {}
                for unit in units_in_block:
                    center_fmt = per_unit_center[unit.name]
                    for opt in unit.options:
                        if opt.fmt == center_fmt:
                            continue
                        path = delta_z_paths.get((unit.name, opt.fmt))
                        if path is None:
                            continue
                        loaded = torch.load(path, weights_only=False)
                        block_cache[(unit.name, opt.fmt)] = [t.float() for t in loaded]

                pair_list: list[bc.BlockPair] = []
                for unit_a_name, unit_b_name in enumerate_block_pairs(units_in_block):
                    unit_a = units_by_name[unit_a_name]
                    unit_b = units_by_name[unit_b_name]
                    center_a = per_unit_center[unit_a.name]
                    center_b = per_unit_center[unit_b.name]
                    omega_ij_dict: dict[tuple[str, str], float] = {}
                    for opt_a in unit_a.options:
                        for opt_b in unit_b.options:
                            if opt_a.fmt == center_a or opt_b.fmt == center_b:
                                # Either format == centered format → δz = 0,
                                # so Ω_ij = 0 by the four-term identity.
                                omega_ij_dict[(opt_a.fmt, opt_b.fmt)] = 0.0
                                continue
                            dz_a = block_cache.get((unit_a.name, opt_a.fmt))
                            dz_b = block_cache.get((unit_b.name, opt_b.fmt))
                            if dz_a is None or dz_b is None:
                                omega_ij_dict[(opt_a.fmt, opt_b.fmt)] = 0.0
                                continue
                            omega_ij_dict[(opt_a.fmt, opt_b.fmt)] = fisher_omega_ij(
                                quad_probs, dz_a, dz_b,
                            )
                    pair_list.append(bc.BlockPair(
                        unit_a=unit_a.name,
                        unit_b=unit_b.name,
                        block_id=block_id,
                        omega_ij=omega_ij_dict,
                    ))
                pairs_by_block[block_id] = pair_list

                if progress_callback is not None:
                    progress_callback({
                        "event": "block_pairs_done",
                        "block_id": block_id,
                        "n_pairs": len(pair_list),
                    })

                del block_cache
                gc.collect()
        else:
            for block_id in blocks:
                pairs_by_block[block_id] = []

        # ---------------------------------------------------------------
        # Phase 4: assemble payload
        # ---------------------------------------------------------------
        new_blocks: dict[str, list[bc.DecisionUnit]] = {}
        for block_id, units_in_block in blocks.items():
            new_units = []
            for unit in units_in_block:
                new_options = tuple(
                    bc.FormatCost(
                        fmt=opt.fmt,
                        omega_ii=float(omega_ii.get((unit.name, opt.fmt), 0.0)),
                        bits_per_param=opt.bits_per_param,
                        memory_bytes=opt.memory_bytes,
                    )
                    for opt in unit.options
                )
                new_units.append(bc.DecisionUnit(
                    name=unit.name,
                    block_id=unit.block_id,
                    member_qnames=unit.member_qnames,
                    options=new_options,
                ))
            new_blocks[block_id] = new_units
        new_singletons: list[bc.DecisionUnit] = []
        for unit in singletons:
            new_options = tuple(
                bc.FormatCost(
                    fmt=opt.fmt,
                    omega_ii=float(omega_ii.get((unit.name, opt.fmt), 0.0)),
                    bits_per_param=opt.bits_per_param,
                    memory_bytes=opt.memory_bytes,
                )
                for opt in unit.options
            )
            new_singletons.append(bc.DecisionUnit(
                name=unit.name,
                block_id=unit.block_id,
                member_qnames=unit.member_qnames,
                options=new_options,
            ))

        elapsed = time.time() - start
        meta = {
            "elapsed_seconds": float(elapsed),
            "n_calib_samples": int(calib_ids.size(0)),
            "calib_seqlen": int(calib_ids.size(1)),
            "formats": [s.name for s in specs_sorted],
            "objective_metric": "teacher_forward_kl_output_fisher",
            "loss": "teacher_student_kl",
            "method": "output_fisher",
            "method_notes": (
                ("Sandwich-centered " if is_sandwich else "BF16-centered ") +
                "Output-Fisher; " +
                ("activation quantization included via PerturbedActivationCache. "
                 if (include_activation_quant or is_sandwich) else
                 "weight-only perturbation (no activation quantization). ")
            ),
            "include_activation_quant": bool(include_activation_quant or is_sandwich),
            "centered": bool(is_sandwich),
            "center_kl": float(center_kl),
            "block_count": len(blocks),
            "singleton_count": len(singletons),
            "n_perturbation_forwards": int(n_pert_done) + (2 if is_sandwich else 1),
            "skip_pairs": bool(skip_pairs),
            "delta_z_dtype": str(delta_z_dtype),
        }
        return bc.units_and_pairs_to_payload(
            blocks=new_blocks,
            singletons=new_singletons,
            pairs_by_block=pairs_by_block,
            meta=meta,
        )
    finally:
        if cleanup_cache_dir:
            shutil.rmtree(cache_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _device_from_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _progress_printer(event: dict) -> None:
    kind = event.get("event")
    if kind == "teacher_forward_start":
        print("[output-fisher] computing teacher logits", flush=True)
    elif kind == "teacher_forward_done":
        print(
            f"[output-fisher] teacher forward done "
            f"({int(event['n_samples'])} samples)",
            flush=True,
        )
    elif kind == "perturbation_done":
        c = int(event["completed"])
        t = int(event["total"])
        if c % 25 == 0 or c == t:
            print(
                f"[output-fisher] perturbation {c}/{t} "
                f"{event['unit']}@{event['format']} "
                f"Ω_ii={float(event['omega_ii']):.6g}",
                flush=True,
            )
    elif kind == "block_pairs_done":
        # Reduce noise: log every 5th block
        block_id = event["block_id"]
        if "layers." in block_id:
            try:
                idx = int(block_id.rsplit(".", 1)[-1])
                if idx % 5 != 0:
                    return
            except Exception:
                pass
        print(
            f"[output-fisher] block {block_id} pairs={int(event['n_pairs'])}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Output-Fisher Block-CLADO measurement")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--formats", default="NVFP4,MXFP8_E4M3,BF16")
    parser.add_argument("--n-calib-samples", type=int, default=2)
    parser.add_argument("--calib-seqlen", type=int, default=128)
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", default=None,
                        help="Optional persistent dir for cached δz tensors")
    parser.add_argument("--keep-disk-cache", action="store_true",
                        help="Keep δz tensors on disk after run (default: delete)")
    parser.add_argument("--skip-pairs", action="store_true",
                        help="Compute only Ω_ii, skip pair Ω_ij — useful as an "
                             "additive baseline")
    parser.add_argument(
        "--delta-z-dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Disk storage dtype for δz tensors",
    )
    parser.add_argument(
        "--no-activation-quant",
        action="store_true",
        help=(
            "Use weight-only perturbation (manual swap, no PerturbedActivationCache). "
            "Faster but doesn't match deployment-faithful surrogates."
        ),
    )
    parser.add_argument(
        "--use-frozen-weight-cache",
        action="store_true",
        help=(
            "Reuse pre-quantized weights across measurements (only relevant when "
            "activation-quant is enabled)."
        ),
    )
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_str = _device_from_arg(args.device)
    dtype = _dtype_from_name(args.dtype)
    delta_z_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.delta_z_dtype]

    staged, cleanup = stage_multimodal(args.model)
    try:
        local_only = bool(args.local_files_only or Path(staged).exists())
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_only,
        }
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": local_only,
        }
        if args.device_map:
            load_kwargs["device_map"] = args.device_map
        elif device_str == "cuda":
            load_kwargs["device_map"] = device_str

        tokenizer = AutoTokenizer.from_pretrained(staged, **tokenizer_kwargs)
        calib_ids = load_wikitext_calibration_windowed(
            tokenizer,
            args.n_calib_samples,
            args.calib_seqlen,
            split=args.calib_split,
            seed=args.calib_seed,
        )
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        if not load_kwargs.get("device_map") and device_str != "cuda":
            model.to(device_str)
        model.eval()
        try:
            profile = detect_profile(args.model)
        except Exception:
            profile = DefaultProfile()

        specs = [fr.get_format(p.strip()) for p in args.formats.split(",") if p.strip()]
        payload = collect_output_fisher(
            model,
            calib_ids,
            specs,
            profile=profile,
            cache_dir=args.cache_dir,
            keep_disk_cache=bool(args.keep_disk_cache),
            delta_z_dtype=delta_z_dtype,
            skip_pairs=bool(args.skip_pairs),
            include_activation_quant=not bool(args.no_activation_quant),
            use_frozen_weight_cache=bool(args.use_frozen_weight_cache),
            progress_callback=_progress_printer,
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        m = payload["meta"]
        print(
            f"[output-fisher] wrote {out_path} "
            f"blocks={m['block_count']} singletons={m['singleton_count']} "
            f"forwards={m['n_perturbation_forwards']} "
            f"elapsed={m['elapsed_seconds']:.1f}s",
            flush=True,
        )
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
