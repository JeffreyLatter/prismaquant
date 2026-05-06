"""Collector for Block-CLADO costs.

Builds the ``prismaquant.block_clado.v1`` artifact consumed by ``block_clado``
solver routines.  Strategy:

1. Identify all quantizable Linears in the model and group them by
   profile-aware fused-sibling key (e.g. q/k/v share one decision unit).
2. Group decision units by architectural block (transformer layer).  Layers
   not inside a transformer block — lm_head, embeddings, MTP heads — are
   treated as singleton blocks with no pair edges.
3. Cache reference (BF16) log-probabilities per calibration sample.
4. For every (unit, format) combination: measure
       Ω_ii(f) = KL(teacher ‖ student[unit ← f, others = BF16])
5. For every intra-block pair (unit_a, unit_b) and (fmt_a, fmt_b): measure
       KL_ab = KL(teacher ‖ student[a ← fmt_a, b ← fmt_b, others = BF16])
   and recover
       Ω_ij(fmt_a, fmt_b) = KL_ab − Ω_ii(unit_a, fmt_a) − Ω_ii(unit_b, fmt_b)
   (KL(teacher‖teacher) = 0 makes the four-term identity collapse to three.)

Cross-block pairs are not measured; the empirical motivation is that
LayerNorms reset error magnitude between blocks, so cross-block Ω_ij is
much smaller than within-block.

Loss = teacher-student KL means the linear term in the Taylor expansion is
exactly zero (KL is minimised at the teacher state) and the quadratic term
is the categorical Fisher of the teacher distribution.  No PSD projection
is needed because the Fisher is PSD by construction; sample noise can make
empirical entries low-rank but not indefinite.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import torch

from prismaquant import block_clado as bc
from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import (
    cache_reference_log_probs,
    iter_quantizable_tensors,
    stage_multimodal,
)
from prismaquant.iterate_perturbed_allocation import measure_assignment_kl
from prismaquant.measure_adjoint_l3 import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.model_profiles import DefaultProfile, detect_profile


# ---------------------------------------------------------------------------
# Discovery — units, blocks, and pair list
# ---------------------------------------------------------------------------


def _recipe_name(full_name: str) -> str:
    return full_name[:-7] if full_name.endswith(".weight") else full_name


def _enumerate_quantizable_linears(model) -> list[str]:
    names: list[str] = []
    for full_name, _module, _attr in iter_quantizable_tensors(model):
        names.append(_recipe_name(full_name))
    return sorted(set(names))


def _shape_of_param(model, qname: str) -> tuple[int, ...] | None:
    target = qname
    for full_name, module, attr in iter_quantizable_tensors(model):
        if _recipe_name(full_name) == target:
            param = getattr(module, attr, None)
            if param is None:
                return None
            return tuple(int(v) for v in param.shape)
    return None


def discover_blocks(
    model,
    profile,
    formats: Sequence[fr.FormatSpec],
) -> tuple[
    dict[str, list[bc.DecisionUnit]],
    list[bc.DecisionUnit],
    dict[str, int],
]:
    """Build the {block_id → [DecisionUnit]} dict, plus singletons.

    Returns:
      blocks:     transformer-block decision units (have intra-block pairs)
      singletons: lm_head/embed/MTP/etc. (one option set, no pair edges)
      n_params_by_unit: param counts for telemetry/total-bpp computation
    """
    qnames = _enumerate_quantizable_linears(model)

    # Map each linear → fused group key, preserving member sets.
    groups: dict[str, list[str]] = defaultdict(list)
    for qname in qnames:
        key = bc.fused_group_key(profile, qname)
        groups[key].append(qname)

    blocks: dict[str, list[bc.DecisionUnit]] = defaultdict(list)
    singletons: list[bc.DecisionUnit] = []
    n_params_by_unit: dict[str, int] = {}

    for group_name, members in sorted(groups.items()):
        members = sorted(set(members))
        # All members of a fused group share an architectural block.
        # Pick the most common block id; fall back to the group's name.
        block_ids = [bc.block_id_from_qname(m) for m in members]
        block_id = max(set(block_ids), key=block_ids.count) if block_ids else group_name
        # Build option list — one FormatCost per supported format.  ω_ii is
        # filled later by the collector.  bits/memory derived from the union
        # of member shapes.
        member_shapes: list[tuple[int, ...]] = []
        for member in members:
            shape = _shape_of_param(model, member)
            if shape is None:
                continue
            member_shapes.append(shape)
        if not member_shapes:
            continue
        n_params_unit = sum(int(_prod(s)) for s in member_shapes)
        n_params_by_unit[group_name] = n_params_unit

        options = []
        for spec in formats:
            spec_canon = fr.canonical_format_name(spec.name)
            mem_bytes = sum(spec.memory_bytes_for_shape(s) for s in member_shapes)
            # Effective bits: total memory (bytes × 8) ÷ params.
            bits_per_param = 8.0 * mem_bytes / max(n_params_unit, 1)
            options.append(bc.FormatCost(
                fmt=spec_canon,
                omega_ii=0.0,
                bits_per_param=float(bits_per_param),
                memory_bytes=int(mem_bytes),
            ))
        options.sort(key=lambda opt: (opt.bits_per_param, opt.fmt))
        unit = bc.DecisionUnit(
            name=group_name,
            block_id=block_id,
            member_qnames=tuple(members),
            options=tuple(options),
        )
        # Heuristic: if block_id contains 'layers.<i>', it's a transformer
        # block; otherwise it's a singleton (lm_head, embed_tokens, etc.).
        if block_id == group_name and ".layers." not in block_id:
            singletons.append(unit)
        else:
            blocks[block_id].append(unit)

    # Singletons can also be detected via "this block has only one unit, AND
    # the unit's member_qnames don't sit in a transformer body."  Inspect
    # blocks that have only one member and emit them as singletons if they're
    # outside the transformer body.
    pruned_blocks: dict[str, list[bc.DecisionUnit]] = {}
    for block_id, units in blocks.items():
        if len(units) == 1 and ".layers." not in block_id:
            singletons.append(units[0])
        else:
            pruned_blocks[block_id] = units
    return pruned_blocks, singletons, n_params_by_unit


def _prod(seq: Iterable[int]) -> int:
    out = 1
    for v in seq:
        out *= int(v)
    return out


def enumerate_block_pairs(
    block_units: Sequence[bc.DecisionUnit],
) -> list[tuple[str, str]]:
    """Return all unordered pairs (unit_a, unit_b) within one block."""
    names = [unit.name for unit in block_units]
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j]))
    return pairs


# ---------------------------------------------------------------------------
# Measurement loop
# ---------------------------------------------------------------------------


def _bf16_assignment(
    units_by_name: Mapping[str, bc.DecisionUnit],
    singletons_by_name: Mapping[str, bc.DecisionUnit],
) -> dict[str, str]:
    """Assignment with every member set to BF16."""
    out: dict[str, str] = {}
    for unit in list(units_by_name.values()) + list(singletons_by_name.values()):
        for member in unit.member_qnames:
            out[member] = "BF16"
    return out


def _center_histogram(per_unit_center: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fmt in per_unit_center.values():
        counts[str(fmt)] = counts.get(str(fmt), 0) + 1
    return counts


def _center_assignment_for_units(
    units_by_name: Mapping[str, bc.DecisionUnit],
    singletons_by_name: Mapping[str, bc.DecisionUnit],
    center_assignment: Mapping[str, str] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build the per-Linear base assignment for sandwich recalibration.

    Returns ``(per_linear_base, per_unit_center)``.  Each unit's per-Linear
    members are coerced to a single canonical format taken from
    ``center_assignment``; if no entry is present, BF16 is used.
    """
    base: dict[str, str] = {}
    per_unit: dict[str, str] = {}
    all_units = list(units_by_name.values()) + list(singletons_by_name.values())
    for unit in all_units:
        chosen = None
        if center_assignment:
            for member in unit.member_qnames:
                if member in center_assignment:
                    chosen = fr.canonical_format_name(str(center_assignment[member]))
                    break
        if chosen is None:
            chosen = "BF16"
        per_unit[unit.name] = chosen
        for member in unit.member_qnames:
            base[member] = chosen
    return base, per_unit


def _override_assignment(
    base: Mapping[str, str],
    unit: bc.DecisionUnit,
    fmt: str,
) -> dict[str, str]:
    out = dict(base)
    canonical = fr.canonical_format_name(str(fmt))
    for member in unit.member_qnames:
        out[member] = canonical
    return out


def _override_pair(
    base: Mapping[str, str],
    unit_a: bc.DecisionUnit,
    fmt_a: str,
    unit_b: bc.DecisionUnit,
    fmt_b: str,
) -> dict[str, str]:
    out = dict(base)
    canonical_a = fr.canonical_format_name(str(fmt_a))
    canonical_b = fr.canonical_format_name(str(fmt_b))
    for member in unit_a.member_qnames:
        out[member] = canonical_a
    for member in unit_b.member_qnames:
        out[member] = canonical_b
    return out


def collect_block_clado(
    model,
    calib_ids: torch.Tensor,
    formats: Sequence[fr.FormatSpec],
    *,
    profile=None,
    work_root: str | Path | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    skip_pairs: bool = False,
    center_assignment: Mapping[str, str] | None = None,
    use_frozen_weight_cache: bool = False,
) -> dict:
    """Run the Block-CLADO measurement.

    Returns the portable JSON payload (dict).  Does not write to disk; the
    caller is responsible for serialising.

    ``center_assignment`` selects the centering point for the Taylor
    expansion.  When ``None`` (default), all Linears are pinned to BF16 —
    i.e. the standard CLADO four-term identity at the teacher state.  When
    provided as a per-Linear ``{qname: format}`` mapping, every Ω_ii is
    measured as the *delta* KL from that centered state, and the four-term
    identity becomes::

        Ω_ij(f_a, f_b) = KL(x_c ⊕ a→f_a, b→f_b)
                         − KL(x_c ⊕ a→f_a) − KL(x_c ⊕ b→f_b) + KL(x_c)

    where ``KL(x_c)`` is no longer 0 in general.  This is the sandwich
    recalibration sketched in deliberation round 02.
    """
    if not isinstance(calib_ids, torch.Tensor) or calib_ids.dim() != 2:
        raise ValueError("calib_ids must be a 2D tensor [samples, seqlen]")

    # Always include BF16 in the format menu.
    spec_by_name = {fr.canonical_format_name(spec.name): spec for spec in formats}
    if "BF16" not in spec_by_name:
        spec_by_name["BF16"] = fr.get_format("BF16")
    specs_sorted = [spec_by_name[name] for name in sorted(spec_by_name)]

    blocks, singletons, n_params_by_unit = discover_blocks(model, profile, specs_sorted)
    units_by_name = {
        unit.name: unit
        for units in blocks.values()
        for unit in units
    }
    singletons_by_name = {unit.name: unit for unit in singletons}
    if not units_by_name and not singletons_by_name:
        raise RuntimeError("no quantizable units discovered in model")

    device = next(model.parameters()).device
    if work_root is None:
        work_root = Path(tempfile.mkdtemp(prefix="prismaquant_block_clado_"))
        cleanup_work_root = True
    else:
        work_root = Path(work_root)
        work_root.mkdir(parents=True, exist_ok=True)
        cleanup_work_root = False

    start = time.time()
    try:
        ref_log_probs = cache_reference_log_probs(model, calib_ids, device)

        base, per_unit_center = _center_assignment_for_units(
            units_by_name, singletons_by_name, center_assignment,
        )
        center_kl = 0.0
        if center_assignment is not None and any(
            fmt != "BF16" for fmt in per_unit_center.values()
        ):
            center_kl = float(measure_assignment_kl(
                model, base, calib_ids, ref_log_probs,
                work_root=work_root, profile=profile,
                use_frozen_weight_cache=use_frozen_weight_cache,
                rng_seed=0,
            ))
            if progress_callback is not None:
                progress_callback({
                    "event": "center_kl",
                    "kl": float(center_kl),
                })

        # ------------------------------------------------------------------
        # Pass 1 — unary measurements.
        # ω_ii(unit, f) = KL(teacher ‖ x_c ⊕ unit→f) − KL(x_c)
        # When x_c is BF16 everywhere, KL(x_c) = 0 and this collapses to the
        # standard form.
        # ------------------------------------------------------------------
        omega_ii: dict[tuple[str, str], float] = {}
        all_units = list(units_by_name.values()) + list(singletons_by_name.values())
        # Skip measurements where the override matches the centered format —
        # ω_ii of the centering format is 0 by definition.
        n_unary = sum(
            sum(1 for opt in unit.options if opt.fmt != per_unit_center.get(unit.name, "BF16"))
            for unit in all_units
        )
        unary_done = 0
        for unit in all_units:
            center_fmt = per_unit_center.get(unit.name, "BF16")
            for opt in unit.options:
                if opt.fmt == center_fmt:
                    omega_ii[(unit.name, opt.fmt)] = 0.0
                    continue
                assignment = _override_assignment(base, unit, opt.fmt)
                kl = measure_assignment_kl(
                    model,
                    assignment,
                    calib_ids,
                    ref_log_probs,
                    work_root=work_root,
                    profile=profile,
                    use_frozen_weight_cache=use_frozen_weight_cache,
                    rng_seed=0,
                )
                omega_ii[(unit.name, opt.fmt)] = float(kl) - center_kl
                unary_done += 1
                if progress_callback is not None:
                    progress_callback({
                        "event": "unary_done",
                        "unit": unit.name,
                        "format": opt.fmt,
                        "kl": float(kl) - center_kl,
                        "completed": unary_done,
                        "total": n_unary,
                    })
        # Update unit options with measured ω_ii.
        all_units = []
        for block_id, unit_list in blocks.items():
            new_units = []
            for unit in unit_list:
                new_options = tuple(
                    bc.FormatCost(
                        fmt=opt.fmt,
                        omega_ii=float(omega_ii[(unit.name, opt.fmt)]),
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
            blocks[block_id] = new_units
        new_singletons: list[bc.DecisionUnit] = []
        for unit in singletons:
            new_options = tuple(
                bc.FormatCost(
                    fmt=opt.fmt,
                    omega_ii=float(omega_ii[(unit.name, opt.fmt)]),
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
        singletons = new_singletons
        units_by_name = {unit.name: unit for unit in (
            u for units in blocks.values() for u in units
        )}

        # ------------------------------------------------------------------
        # Pass 2 — intra-block pair measurements (four-term identity).
        #   Ω_ij(f_a, f_b) = KL(x_c ⊕ a→f_a, b→f_b)
        #                    − Ω_ii(a, f_a) − Ω_ii(b, f_b) − KL(x_c)
        # When either format == centered format for that unit, the pair
        # interaction reduces to Ω_ii of the other unit alone, so the pair
        # contribution is identically 0 — no measurement needed.
        # ------------------------------------------------------------------
        pairs_by_block: dict[str, list[bc.BlockPair]] = {}
        if not skip_pairs:
            n_pairs_total = 0
            for block_id, units in blocks.items():
                for unit_a_name, unit_b_name in enumerate_block_pairs(units):
                    unit_a = units_by_name[unit_a_name]
                    unit_b = units_by_name[unit_b_name]
                    center_a = per_unit_center.get(unit_a.name, "BF16")
                    center_b = per_unit_center.get(unit_b.name, "BF16")
                    for opt_a in unit_a.options:
                        if opt_a.fmt == center_a:
                            continue
                        for opt_b in unit_b.options:
                            if opt_b.fmt == center_b:
                                continue
                            n_pairs_total += 1
            pair_done = 0
            for block_id, units in blocks.items():
                pair_list = []
                for unit_a_name, unit_b_name in enumerate_block_pairs(units):
                    unit_a = units_by_name[unit_a_name]
                    unit_b = units_by_name[unit_b_name]
                    center_a = per_unit_center.get(unit_a.name, "BF16")
                    center_b = per_unit_center.get(unit_b.name, "BF16")
                    omega_ij: dict[tuple[str, str], float] = {}
                    for opt_a in unit_a.options:
                        for opt_b in unit_b.options:
                            if opt_a.fmt == center_a or opt_b.fmt == center_b:
                                # No measurable interaction when either unit
                                # is at the centered format — Ω_ij ≡ 0.
                                omega_ij[(opt_a.fmt, opt_b.fmt)] = 0.0
                                continue
                            assignment = _override_pair(
                                base, unit_a, opt_a.fmt, unit_b, opt_b.fmt,
                            )
                            kl_ab = measure_assignment_kl(
                                model,
                                assignment,
                                calib_ids,
                                ref_log_probs,
                                work_root=work_root,
                                profile=profile,
                                use_frozen_weight_cache=use_frozen_weight_cache,
                                rng_seed=0,
                            )
                            omega_a = float(omega_ii[(unit_a.name, opt_a.fmt)])
                            omega_b = float(omega_ii[(unit_b.name, opt_b.fmt)])
                            omega_ij[(opt_a.fmt, opt_b.fmt)] = (
                                float(kl_ab) - omega_a - omega_b - center_kl
                            )
                            pair_done += 1
                            if progress_callback is not None:
                                progress_callback({
                                    "event": "pair_done",
                                    "block_id": block_id,
                                    "unit_a": unit_a.name,
                                    "unit_b": unit_b.name,
                                    "fmt_a": opt_a.fmt,
                                    "fmt_b": opt_b.fmt,
                                    "kl_ab": float(kl_ab),
                                    "omega_ij": omega_ij[(opt_a.fmt, opt_b.fmt)],
                                    "completed": pair_done,
                                    "total": n_pairs_total,
                                })
                    pair_list.append(bc.BlockPair(
                        unit_a=unit_a.name,
                        unit_b=unit_b.name,
                        block_id=block_id,
                        omega_ij=omega_ij,
                    ))
                pairs_by_block[block_id] = pair_list
        else:
            for block_id in blocks:
                pairs_by_block[block_id] = []

        elapsed = time.time() - start
        is_centered = center_assignment is not None and any(
            fmt != "BF16" for fmt in per_unit_center.values()
        )
        meta = {
            "elapsed_seconds": float(elapsed),
            "n_calib_samples": int(calib_ids.size(0)),
            "calib_seqlen": int(calib_ids.size(1)),
            "formats": [spec.name for spec in specs_sorted],
            "objective_metric": "teacher_forward_kl_four_term",
            "loss": "teacher_student_kl",
            "block_count": len(blocks),
            "singleton_count": len(singletons),
            "n_unary_measurements": int(n_unary),
            "n_pair_measurements": (
                sum(len(p.omega_ij) for plist in pairs_by_block.values() for p in plist)
                if not skip_pairs else 0
            ),
            "skip_pairs": bool(skip_pairs),
            "centered": bool(is_centered),
            "center_kl": float(center_kl),
            "center_format_histogram": _center_histogram(per_unit_center),
        }
        return bc.units_and_pairs_to_payload(
            blocks=blocks,
            singletons=singletons,
            pairs_by_block=pairs_by_block,
            meta=meta,
        )
    finally:
        if cleanup_work_root:
            shutil.rmtree(work_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _device_from_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _progress_printer(event: dict) -> None:
    kind = event.get("event")
    if kind == "unary_done":
        print(
            "[block-clado] unary "
            f"{int(event['completed'])}/{int(event['total'])} "
            f"{event['unit']}@{event['format']} "
            f"KL={float(event['kl']):.6g}",
            flush=True,
        )
    elif kind == "pair_done":
        print(
            "[block-clado] pair "
            f"{int(event['completed'])}/{int(event['total'])} "
            f"{event['unit_a']}@{event['fmt_a']} × "
            f"{event['unit_b']}@{event['fmt_b']} "
            f"KL_ab={float(event['kl_ab']):.6g} "
            f"Ω_ij={float(event['omega_ij']):.6g}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Block-CLADO costs")
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
    parser.add_argument("--work-dir", default=None)
    parser.add_argument(
        "--skip-pairs",
        action="store_true",
        help=(
            "Measure only diagonal Ω_ii (unary).  Useful as a HAWQ-V3-style "
            "baseline against full Block-CLADO."
        ),
    )
    parser.add_argument(
        "--center-assignment",
        default=None,
        help=(
            "Per-Linear assignment JSON used as the centering point for the "
            "Taylor expansion (sandwich recalibration).  Default centers at "
            "BF16 (standard CLADO)."
        ),
    )
    parser.add_argument(
        "--use-frozen-weight-cache",
        action="store_true",
        help=(
            "Pre-quantize the centered base assignment once and reuse cached "
            "weights across measurements.  Massively faster on sandwich runs "
            "(non-BF16 centers) but uses memory; safe at small/medium scale."
        ),
    )
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_str = _device_from_arg(args.device)
    dtype = _dtype_from_name(args.dtype)
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

        specs = [fr.get_format(part.strip()) for part in args.formats.split(",") if part.strip()]
        center_assignment: Mapping[str, str] | None = None
        if args.center_assignment:
            raw = json.loads(Path(args.center_assignment).read_text())
            if isinstance(raw, Mapping) and "assignment" in raw and isinstance(raw["assignment"], Mapping):
                center_assignment = {
                    str(k): str(v) for k, v in raw["assignment"].items()
                }
            elif isinstance(raw, Mapping):
                center_assignment = {str(k): str(v) for k, v in raw.items()}
            else:
                raise ValueError(
                    f"unsupported --center-assignment shape: {args.center_assignment}"
                )
            print(
                f"[block-clado] sandwich recalibration: centering at "
                f"{args.center_assignment} ({len(center_assignment)} entries)",
                flush=True,
            )

        payload = collect_block_clado(
            model,
            calib_ids,
            specs,
            profile=profile,
            work_root=args.work_dir,
            progress_callback=_progress_printer,
            skip_pairs=args.skip_pairs,
            center_assignment=center_assignment,
            use_frozen_weight_cache=bool(args.use_frozen_weight_cache),
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        block_count = payload["meta"]["block_count"]
        singletons = payload["meta"]["singleton_count"]
        n_unary = payload["meta"]["n_unary_measurements"]
        n_pair = payload["meta"]["n_pair_measurements"]
        elapsed = payload["meta"]["elapsed_seconds"]
        print(
            f"[block-clado] wrote {out_path} "
            f"blocks={block_count} singletons={singletons} "
            f"unary_meas={n_unary} pair_meas={n_pair} "
            f"elapsed={elapsed:.1f}s",
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
