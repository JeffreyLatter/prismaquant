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
import os
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


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
    *,
    microbatch: int = 1,
    output_device: torch.device | str | None = "cpu",
    logit_scope: str = "full_sequence",
) -> list[torch.Tensor]:
    """Forward each calibration sample; return fp32 ``[T, V]`` tensors."""
    device = next(model.parameters()).device
    out_device = torch.device(output_device) if output_device is not None else device
    out: list[torch.Tensor] = []
    step = max(int(microbatch), 1)
    scope = str(logit_scope)
    for i in range(0, calib_ids.size(0), step):
        batch = calib_ids[i:i + step].to(device)
        logits = model(batch).logits.detach()
        if scope == "last_token":
            logits = logits[:, -1:, :]
        logits = logits.float().to(out_device)
        out.extend(logits[j].contiguous() for j in range(logits.size(0)))
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
    production_weight_cache=None,
    microbatch: int = 1,
    include_activation_quant: bool = True,
    output_device: torch.device | str | None = "cpu",
    logit_scope: str = "full_sequence",
) -> list[torch.Tensor]:
    """Run forward with the given quantization assignment applied via
    PerturbedActivationCache, returning per-sample fp32 ``[T, V]`` logits.

    Unlike ``_quantize_unit_in_place`` + ``_forward_logits``, this path
    matches ``measure_assignment_kl``'s deployment-faithful measurement
    that includes activation quantization at any unit assigned to a
    non-BF16 format.
    """
    device = next(model.parameters()).device
    out_device = torch.device(output_device) if output_device is not None else device
    scope = str(logit_scope)
    cache_dir = Path(tempfile.mkdtemp(prefix="prismaquant_of_pcache_", dir=str(work_root)))
    cal_hash = calibration_data_hash(calib_ids)
    hooks = PerturbedActivationCache(
        model, assignment, cache_dir,
        input_rows=0, cal_hash=cal_hash, profile=profile,
        production_weight_cache=production_weight_cache,
        include_activation_quant=include_activation_quant,
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
                step = max(int(microbatch), 1)
                for i in range(0, calib_ids.size(0), step):
                    batch = calib_ids[i:i + step].to(device)
                    logits = model(batch).logits.detach()
                    if scope == "last_token":
                        logits = logits[:, -1:, :]
                    logits = logits.float().to(out_device)
                    out.extend(logits[j].contiguous() for j in range(logits.size(0)))
            finally:
                hooks.remove()
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return out


def _resolve_reduction_device(
    model,
    requested: str | torch.device | None,
    *,
    calib_ids: torch.Tensor | None = None,
    is_sandwich: bool = False,
    logit_scope: str = "full_sequence",
) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    value = str(requested or "cpu").strip().lower()
    model_device = next(model.parameters()).device
    if value in {"cuda", "gpu"}:
        return model_device if model_device.type == "cuda" else torch.device("cpu")
    if value == "auto":
        if model_device.type != "cuda" or calib_ids is None:
            return torch.device("cpu")
        vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
        if vocab_size <= 0:
            return torch.device("cpu")
        effective_seqlen = 1 if str(logit_scope) == "last_token" else int(calib_ids.size(1))
        # Resident full-vocab tensors: teacher logits + teacher probs, plus
        # centered logits/probs and linear offset in sandwich mode. Block-local
        # deltas add temporary pressure, so require a conservative headroom.
        fp32_bytes = int(calib_ids.size(0)) * effective_seqlen * vocab_size * 4
        resident_bytes = fp32_bytes * (5 if is_sandwich else 2)
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(model_device)
        except Exception:
            return torch.device("cpu")
        cuda_cap = 48 * (1024 ** 3)
        budget = min(int(free_bytes * 0.45), cuda_cap)
        return model_device if resident_bytes <= budget else torch.device("cpu")
    return torch.device("cpu")


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
    total_quad = None
    total_lin = None
    total_tokens = 0
    for idx, (p, dz) in enumerate(zip(teacher_probs, delta_z)):
        p32 = p.float()
        dz32 = dz.float()
        e_dz = (p32 * dz32).sum(dim=-1)
        e_dz2 = (p32 * dz32 * dz32).sum(dim=-1)
        var = e_dz2 - e_dz * e_dz
        var_sum = var.sum()
        total_quad = var_sum if total_quad is None else total_quad + var_sum
        total_tokens += var.numel()
        if linear_offset is not None:
            offset = linear_offset[idx].float()
            lin = (offset * dz32).sum(dim=-1)
            lin_sum = lin.sum()
            total_lin = lin_sum if total_lin is None else total_lin + lin_sum
    if total_tokens == 0:
        return 0.0
    if total_quad is None:
        return 0.0
    if total_lin is None:
        total_lin = torch.zeros((), device=total_quad.device, dtype=total_quad.dtype)
    return float((total_lin + 0.5 * total_quad) / total_tokens)


@torch.no_grad()
def fisher_omega_ij(
    teacher_probs: Sequence[torch.Tensor],
    delta_z_a: Sequence[torch.Tensor],
    delta_z_b: Sequence[torch.Tensor],
) -> float:
    """Return ``mean_t Cov_{p_t}(δz_a, δz_b)`` averaged over tokens."""
    total_cov = None
    total_tokens = 0
    for p, dz_a, dz_b in zip(teacher_probs, delta_z_a, delta_z_b):
        p32 = p.float()
        a32 = dz_a.float()
        b32 = dz_b.float()
        e_ab = (p32 * a32 * b32).sum(dim=-1)
        e_a = (p32 * a32).sum(dim=-1)
        e_b = (p32 * b32).sum(dim=-1)
        cov = e_ab - e_a * e_b
        cov_sum = cov.sum()
        total_cov = cov_sum if total_cov is None else total_cov + cov_sum
        total_tokens += cov.numel()
    if total_tokens == 0 or total_cov is None:
        return 0.0
    return float(total_cov / total_tokens)


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
    production_weight_cache=None,
    calib_microbatch: int = 1,
    reduction_device: str | torch.device | None = "auto",
    pin_to_bf16: Sequence[str] = ("lm_head",),
    logit_scope: str = "full_sequence",
) -> dict:
    """Build the Output-Fisher Block-CLADO payload.

    Returns a dict in the ``prismaquant.block_clado.v1`` schema (so the
    rest of the pipeline can consume it unchanged).  ``meta.method`` is
    set to ``"output_fisher"`` for traceability.
    """
    if not isinstance(calib_ids, torch.Tensor) or calib_ids.dim() != 2:
        raise ValueError("calib_ids must be a 2D tensor [samples, seqlen]")
    if logit_scope not in {"full_sequence", "last_token"}:
        raise ValueError("logit_scope must be 'full_sequence' or 'last_token'")

    spec_by_name = {fr.canonical_format_name(s.name): s for s in formats}
    if "BF16" not in spec_by_name:
        spec_by_name["BF16"] = fr.get_format("BF16")
    specs_sorted = [spec_by_name[name] for name in sorted(spec_by_name)]

    blocks, singletons, _n_params = discover_blocks(model, profile, specs_sorted)
    blocks, singletons = bc.apply_bf16_pins_to_units(
        blocks, singletons, pin_to_bf16=pin_to_bf16,
    )
    units_by_name: dict[str, bc.DecisionUnit] = {
        u.name: u for ulist in blocks.values() for u in ulist
    }
    singletons_by_name: dict[str, bc.DecisionUnit] = {u.name: u for u in singletons}
    if not units_by_name and not singletons_by_name:
        raise RuntimeError("no quantizable units discovered in model")

    all_units = list(units_by_name.values()) + list(singletons_by_name.values())
    def _force_pinned_bf16(assignment: dict[str, str]) -> None:
        for unit in all_units:
            if not bc.unit_is_bf16_pinned(unit, pin_to_bf16):
                continue
            for member in unit.member_qnames:
                assignment[member] = "BF16"

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
    reduction_dev = _resolve_reduction_device(
        model,
        reduction_device,
        calib_ids=calib_ids,
        is_sandwich=is_sandwich,
        logit_scope=logit_scope,
    )
    weight_session = None
    restore_assignment: dict[str, str] | None = None
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
        z_teacher = _forward_logits(
            model,
            calib_ids,
            microbatch=calib_microbatch,
            output_device=reduction_dev,
            logit_scope=logit_scope,
        )  # list of [T, V] fp32 tensors on the selected reduction device
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
            _force_pinned_bf16(sandwich_base)
            if progress_callback is not None:
                progress_callback({"event": "centered_forward_start"})
            if include_activation_quant:
                z_centered = _forward_logits_with_assignment(
                    model, sandwich_base, calib_ids,
                    work_root=cache_dir,
                    profile=profile,
                    use_frozen_weight_cache=use_frozen_weight_cache,
                    production_weight_cache=production_weight_cache,
                    microbatch=calib_microbatch,
                    include_activation_quant=True,
                    output_device=reduction_dev,
                    logit_scope=logit_scope,
                )
            else:
                from prismaquant.weight_session import WeightSession

                weight_session = WeightSession(
                    model,
                    production_weight_cache=production_weight_cache,
                    snapshot_dir=(
                        str(cache_dir / "weight_session_snapshots")
                        if _env_truthy("PRISMAQUANT_OF_SPILL_WEIGHT_SESSION")
                        else None
                    ),
                )
                weight_session.initialize(sandwich_base, all_units)
                z_centered = _forward_logits(
                    model,
                    calib_ids,
                    microbatch=calib_microbatch,
                    output_device=reduction_dev,
                    logit_scope=logit_scope,
                )
            centered_probs: list[torch.Tensor] = [
                torch.softmax(z, dim=-1) for z in z_centered
            ]
            # Compute KL(p_t || p_centered) for telemetry; use log_softmax
            # directly to match measure_assignment_kl's numerics rather
            # than log-of-clamp(p) which over-estimates for tiny probs.
            total_kl = None
            total_tokens = 0
            for z_t, z_c, p_t_i in zip(z_teacher, z_centered, teacher_probs):
                lp_t = torch.log_softmax(z_t.float(), dim=-1)
                lp_c = torch.log_softmax(z_c.float(), dim=-1)
                kl_per_token = (p_t_i.float() * (lp_t - lp_c)).sum(dim=-1)
                kl_sum = kl_per_token.sum()
                total_kl = kl_sum if total_kl is None else total_kl + kl_sum
                total_tokens += kl_per_token.numel()
            center_kl = (
                float(total_kl / max(total_tokens, 1))
                if total_kl is not None else 0.0
            )
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
        restore_assignment = bf16_base
        if not include_activation_quant and weight_session is None:
            from prismaquant.weight_session import WeightSession

            weight_session = WeightSession(
                model,
                production_weight_cache=production_weight_cache,
                snapshot_dir=(
                    str(cache_dir / "weight_session_snapshots")
                    if _env_truthy("PRISMAQUANT_OF_SPILL_WEIGHT_SESSION")
                    else None
                ),
            )
            weight_session.initialize(base_assignment, all_units)
        dirty_weight_members: set[str] = set()

        # ---------------------------------------------------------------
        # Phase 2/3: compute δz_{U,f}, Ω_ii, and per-block Ω_ij.
        #
        # Keep only one transformer's block worth of δz tensors live at a
        # time. The previous disk-backed implementation wrote every δz to
        # .pt first, which can consume hundreds of GB on 4B+ full-vocab
        # output-Fisher runs.
        # ---------------------------------------------------------------
        omega_ii: dict[tuple[str, str], float] = {}
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

        def _measure_delta_for_option(
            unit: bc.DecisionUnit,
            opt: bc.FormatCost,
        ) -> list[torch.Tensor] | None:
            nonlocal n_pert_done
            center_fmt = per_unit_center[unit.name]
            if opt.fmt == center_fmt:
                omega_ii[(unit.name, opt.fmt)] = 0.0
                return None

            if include_activation_quant:
                # Use PerturbedActivationCache so activation quantization is
                # applied at this unit, matching deployment / four-term.
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
                        production_weight_cache=production_weight_cache,
                        microbatch=calib_microbatch,
                        include_activation_quant=True,
                        output_device=reduction_dev,
                        logit_scope=logit_scope,
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
                    return None
            elif weight_session is not None:
                canonical_fmt = fr.canonical_format_name(opt.fmt)
                changes: dict[str, str] = {}
                for member in dirty_weight_members:
                    changes[member] = base_assignment.get(member, "BF16")
                dirty_weight_members.clear()
                for member in unit.member_qnames:
                    changes[member] = canonical_fmt
                    if canonical_fmt != base_assignment.get(member, "BF16"):
                        dirty_weight_members.add(member)
                weight_session.apply_assignment(changes)
                z_pert = _forward_logits(
                    model,
                    calib_ids,
                    microbatch=calib_microbatch,
                    output_device=reduction_dev,
                    logit_scope=logit_scope,
                )
            else:
                # Weight-only fast path (legacy, BF16 base only)
                saved = _quantize_unit_in_place(model, unit, opt.fmt)
                if not saved:
                    omega_ii[(unit.name, opt.fmt)] = 0.0
                    return None
                try:
                    z_pert = _forward_logits(
                        model,
                        calib_ids,
                        microbatch=calib_microbatch,
                        output_device=reduction_dev,
                        logit_scope=logit_scope,
                    )
                finally:
                    _restore_unit(model, saved)

            # δz is measured FROM the baseline (BF16 in normal mode,
            # centered state in sandwich mode).
            delta_z = [(zp - zb).contiguous() for zp, zb in zip(z_pert, z_baseline)]
            omega_ii[(unit.name, opt.fmt)] = fisher_omega_ii(
                quad_probs, delta_z,
                linear_offset=linear_offset,
            )

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

            del z_pert
            return [t.to(delta_z_dtype).contiguous() for t in delta_z]

        pairs_by_block: dict[str, list[bc.BlockPair]] = {}
        for block_id, units_in_block in blocks.items():
            block_cache: dict[tuple[str, str], list[torch.Tensor]] = {}
            for unit in units_in_block:
                center_fmt = per_unit_center[unit.name]
                for opt in unit.options:
                    if opt.fmt == center_fmt:
                        omega_ii[(unit.name, opt.fmt)] = 0.0
                        continue
                    delta_z = _measure_delta_for_option(unit, opt)
                    if delta_z is not None and not skip_pairs:
                        block_cache[(unit.name, opt.fmt)] = delta_z
                    del delta_z

            if skip_pairs:
                pairs_by_block[block_id] = []
            else:
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
                                quad_probs,
                                [t.float() for t in dz_a],
                                [t.float() for t in dz_b],
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

        for unit in singletons:
            center_fmt = per_unit_center[unit.name]
            for opt in unit.options:
                if opt.fmt == center_fmt:
                    omega_ii[(unit.name, opt.fmt)] = 0.0
                    continue
                delta_z = _measure_delta_for_option(unit, opt)
                del delta_z

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
            "logit_scope": str(logit_scope),
            "calib_microbatch": int(max(int(calib_microbatch), 1)),
            "reduction_device": str(reduction_dev),
            "formats": [s.name for s in specs_sorted],
            "objective_metric": "teacher_forward_kl_output_fisher",
            "loss": "teacher_student_kl",
            "method": "output_fisher",
            "method_notes": (
                ("Sandwich-centered " if is_sandwich else "BF16-centered ") +
                "Output-Fisher; " +
                ("activation quantization included via PerturbedActivationCache. "
                 if include_activation_quant else
                 "weight-only perturbation (no activation quantization). ")
            ),
            "include_activation_quant": bool(include_activation_quant),
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
        if weight_session is not None and restore_assignment is not None:
            try:
                weight_session.apply_assignment(restore_assignment)
            except Exception:
                pass
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
    elif kind == "centered_forward_start":
        print("[output-fisher] computing centered logits", flush=True)
    elif kind == "centered_forward_done":
        print(
            f"[output-fisher] centered forward done "
            f"center_kl={float(event['center_kl']):.6g}",
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
    parser.add_argument("--calib-microbatch", type=int, default=1)
    parser.add_argument(
        "--logit-scope",
        choices=["full_sequence", "last_token"],
        default="full_sequence",
        help=(
            "Which logits participate in the output-Fisher KL surrogate. "
            "last_token keeps the teacher/center tensor stack bounded for "
            "long calibration windows."
        ),
    )
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help=(
            "Transformers attention backend for model load. Accepts built-ins "
            "such as sdpa/flash_attention_2 and Kernel Hub backends such as "
            "kernels-community/flash-attn2."
        ),
    )
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
        "--reduction-device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help=(
            "Device for output-Fisher probability/delta reductions. 'cuda' "
            "keeps full-vocab logits and block-local deltas on GPU and only "
            "returns scalar Ω values; 'auto' uses CUDA only when the full-vocab "
            "resident tensor stack fits a conservative memory budget."
        ),
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
        load_kwargs["attn_implementation"] = getattr(
            args,
            "attn_implementation",
            "sdpa",
        )
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
            calib_microbatch=int(args.calib_microbatch),
            reduction_device=args.reduction_device,
            logit_scope=args.logit_scope,
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
