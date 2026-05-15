#!/usr/bin/env python3
"""allocator.py — multi-choice knapsack mixed-precision assignment.

Given:
  - per-Linear empirical Fisher diagonal trace (from sensitivity_probe.py)
  - per-(Linear, format) measured quantization cost (from measure_quant_cost.py)
  - a bit budget (target average bits per parameter)
  - a format registry (any subset of registered formats)

Solve for a per-Linear format assignment that minimizes total predicted
loss increase subject to the bit budget.

Derivation of the per-(layer, format) predicted loss term
---------------------------------------------------------
Let L be the per-token loss (negative log-likelihood). Quantizing layer
ℓ's weight tensor W by ΔW = W_q - W produces a perturbed loss whose
expectation under the calibration distribution admits the standard
second-order expansion:

    E[ΔL] ≈ 0.5 · ΔW · F · ΔWᵀ                         (1)

where F is the Fisher information matrix of L w.r.t. W. Replacing F by
its diagonal (the standard HAWQ-V1 simplification) and approximating
F_ww by the empirical Fisher diagonal F̂_ww = E_token[(∂L/∂W_w)²]:

    E[ΔL] ≈ 0.5 · Σ_w F̂_ww · (ΔW_w)²                   (2)

Under the further assumption that the per-weight quantization error
(ΔW_w)² and the per-weight Fisher diagonal F̂_ww are uncorrelated across
w (which is the same assumption HAWQ already makes when it summarizes a
layer by a single scalar), this collapses to the product of two
per-layer scalars:

    E[ΔL] ≈ 0.5 · H_trace · MSE_W                       (3)

where
    H_trace = Σ_w F̂_ww            (per-token Fisher diagonal trace)
    MSE_W   = (1/n_w) · Σ_w (ΔW_w)²

Both quantities are produced by upstream stages:
    H_trace ← sensitivity_probe.py / FisherAccumulator (`h_trace`)
    MSE_W   ← measure_quant_cost.py (per-(layer, format) `weight_mse`)

So we use eq. (3) directly. There is no `* d_out` factor; the previous
implementation carried one but it does not appear in the derivation —
it was a holdover from an earlier output-side formulation that mixed
units and was off by a per-layer multiplicative constant that varies
with d_out.

For MoE experts an additional route-probability normalization is folded
into H_trace inside the probe so that sparsely-routed experts' Fisher
contributions are on the same per-token footing as dense layers'.

Solver:
  Multi-choice knapsack via DP with bit-budget discretization (we round
  bit costs to 0.001-bit bins). For 35B with ~300 Linears × 8 formats ×
  ~5000 budget bins, runtime is under 1s.

Fused-projection siblings (q/k/v/o, gate/up, ...) are post-processed:
  all siblings promoted to the highest format chosen for any of them,
  to match vLLM's fused-tensor loader constraints. Since promotion can
  push achieved bits past the requested budget, the DP is re-run with a
  tightened target until achieved is within tolerance.

Optional empirical calibration:
  If `--calibration` points at a JSON containing
  `calibrated_gains[fmt] = α_fmt`, the predicted Δloss for format f is
  multiplied by α_f before the DP runs. The historical tiny-bakeoff
  producer for this payload is archived; production recipes normally run
  uncalibrated and validate assignments with direct KL measurement.

Auto-Pareto knee via Kneedle (Satopää et al.). Reports the knee target
plus a few flanking points so you can eyeball.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

from . import format_registry as fr
from .allocator_solver import (
    Candidate,
    _shape_from_stats,
    compute_achieved,
    compute_assignment_predicted_dloss,
    promote_fused,
    promote_moe_pair,
    solve_with_promotion,
)
from .allocator_candidates import (
    PASSTHROUGH_SOURCE_REQUIREMENTS,
    _FUSED_SIBLING_MARKER,
    _flashinfer_kernel_accepts,
    _format_kernel_supports_shape,
    _is_passthrough_format,
    _passthrough_source_ok,
    _scan_source_dtype_manifest,
    aggregate_fused_siblings,
    build_candidates,
    expand_fused_sibling_assignment,
    summarize_applicability_masks,
)
from .serving_profiles import check_serving_format, serving_profile_names
from .schemas import validate_cost_payload, validate_probe_payload



# ---------------------------------------------------------------------------
# Kneedle knee detection
# ---------------------------------------------------------------------------
def kneedle(x: list[float], y: list[float]) -> int:
    """Return index of the knee in a convex-decreasing curve."""
    if len(x) < 3:
        return 0
    xs = [xi for xi in x]
    ys = [yi for yi in y]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin or ymax == ymin:
        return 0
    x_norm = [(xi - xmin) / (xmax - xmin) for xi in xs]
    y_norm = [(yi - ymin) / (ymax - ymin) for yi in ys]
    # For a convex-decreasing curve, the knee is the point with max
    # distance below the chord from (0,1) to (1,0).
    diffs = [yn - (1.0 - xn) for xn, yn in zip(x_norm, y_norm)]
    # Convex-decreasing, so we want the most-negative diff (max dip).
    return min(range(len(diffs)), key=lambda i: diffs[i])




def _allowed_format(target_profile: str, name: str, fmt: str) -> bool:
    decision = check_serving_format(target_profile, name, fmt)
    if not decision.legal and decision.detail.startswith("unknown target profile"):
        raise ValueError(decision.detail)
    return decision.legal


def filter_candidates_for_profile(
    candidates: dict[str, list[Candidate]],
    target_profile: str,
) -> dict[str, list[Candidate]]:
    out = {}
    for name, cands in candidates.items():
        kept = [c for c in cands if _allowed_format(target_profile, name, c.fmt)]
        if kept:
            out[name] = kept
    return out


# ---------------------------------------------------------------------------
# Visual encoder override
# ---------------------------------------------------------------------------
# Phase 1 visual-encoder support: the probe's text-only calibration does not
# exercise the visual tower, so per-Linear Fisher gradients for
# `model.visual.blocks.*` Linears are zero — the knapsack DP has no
# sensitivity signal to allocate on. Rather than let every visual Linear
# default to the cheapest format or go through stale passthrough, we accept
# a single uniform target format (`BF16`, `NVFP4`, or `MXFP8`) and assign
# every visual Linear to it. BF16 (the default) reproduces the previous
# passthrough behavior; NVFP4/MXFP8 shrink the tower to quantized storage
# using the same RTN math the body gets.
#
# Phase 2 (tracked separately) will replace this override with a real
# multimodal Fisher: load images + text, run full forward through the
# visual encoder → projector → body LM, capture per-Linear empirical Fisher
# gradients, and feed those into the allocator's closed-form Δloss. That
# requires a multimodal dataset loader, multimodal tokenizer wiring, and a
# probe path that doesn't strip the visual tower — none of which ship in
# Phase 1.
_VISUAL_PREFIX_RE = re.compile(r"^(?:model\.)?visual\.")


def _is_visual_linear(name: str) -> bool:
    """True when `name` refers to a Linear inside the visual encoder.

    Matches both the raw HF checkpoint form (`model.visual.blocks.*`) and
    the post-remap form (`visual.blocks.*`) so the override behaves the
    same regardless of which side of `profile.live_to_recipe_name` the
    allocator's stats dictionary landed on.
    """
    return bool(_VISUAL_PREFIX_RE.match(name))


def apply_visual_format_override(
    assignment: dict[str, str],
    visual_format: str,
) -> dict[str, str]:
    """Force every visual-encoder Linear in `assignment` to `visual_format`.

    Called after the knapsack DP + fused-sibling promotion so the override
    wins even if the solver would have picked a different format per
    per-Linear sensitivity noise (which is meaningless for visual Linears
    under text-only calibration — see module comment above).

    `visual_format="BF16"` is a no-op when a visual Linear already has no
    allocator entry (the export's existing passthrough keeps it at BF16);
    we still write `BF16` into the returned assignment so the layer_config
    round-trip is explicit and downstream tooling (export, validate) has a
    uniform record of the decision.
    """
    out = dict(assignment)
    for name in list(out.keys()):
        if _is_visual_linear(name):
            out[name] = visual_format
    return out


def apply_mtp_format_override(
    assignment: dict[str, str],
    mtp_format: str,
) -> dict[str, str]:
    """Force MTP Linears to a recipe-level format.

    The production vLLM path currently validates main-target logits and keeps
    MTP in BF16 until speculative-decode acceptance is measured.  This override
    is applied after DP/fused-sibling promotion so a sensitive MTP projection
    cannot be accidentally quantized by the allocator.
    """
    out = dict(assignment)
    for name in list(out.keys()):
        if name.startswith("mtp."):
            out[name] = mtp_format
    return out


def discover_visual_linears_from_source(model_path: str) -> list[str]:
    """Scan the source safetensors index for `model.visual.blocks.*.weight`
    entries with rank-2 shapes — these are the Linear modules the visual
    encoder exposes.

    Returned names are the basename (`.weight` stripped) so they slot
    directly into the allocator's assignment dictionary and the exporter's
    quantize-by-recipe dispatch.

    The probe's text-only staging strips the visual tower, so visual
    Linears never appear in the probe or cost pickles. This helper lets
    the allocator emit a layer_config entry for them anyway when
    `--visual-format` is non-BF16 — the exporter can then quantize each
    of them uniformly under the requested format. Without this scan, the
    allocator has no way to enumerate visual Linear names (there is no
    in-memory visual module at allocation time).
    """
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    candidates: list[tuple[str, tuple[int, ...]]] = []
    if idx_path.exists():
        with open(idx_path) as f:
            wm = json.load(f).get("weight_map", {})
        # Index file carries only names, not shapes. We need to open each
        # referenced shard once to read rank.
        from collections import defaultdict as _dd
        by_shard: dict[str, list[str]] = _dd(list)
        for key, shard in wm.items():
            if not key.endswith(".weight"):
                continue
            if not _VISUAL_PREFIX_RE.match(key):
                continue
            by_shard[shard].append(key)
        try:
            from safetensors import safe_open
        except ImportError:
            return []
        for shard, keys in by_shard.items():
            shard_path = src / shard
            if not shard_path.exists():
                continue
            with safe_open(str(shard_path), framework="pt") as sf:
                for k in keys:
                    try:
                        shape = tuple(sf.get_slice(k).get_shape())
                    except Exception:
                        continue
                    candidates.append((k, shape))
    else:
        # No index file — scan every safetensors shard directly. Used for
        # small, single-file checkpoints.
        try:
            from safetensors import safe_open
        except ImportError:
            return []
        import os as _os
        if not src.exists():
            return []
        for f in sorted(_os.listdir(src)):
            if not f.endswith(".safetensors"):
                continue
            with safe_open(str(src / f), framework="pt") as sf:
                for k in sf.keys():
                    if not k.endswith(".weight"):
                        continue
                    if not _VISUAL_PREFIX_RE.match(k):
                        continue
                    try:
                        shape = tuple(sf.get_slice(k).get_shape())
                    except Exception:
                        continue
                    candidates.append((k, shape))

    # Only rank-2 weights are Linear-like; conv1d / norms / biases are
    # kept at BF16 passthrough regardless of --visual-format.
    # Additionally, blacklist known rank-2 tensors that live in
    # `nn.Parameter` / `nn.Embedding` modules (NOT `nn.Linear`), which
    # the compressed-tensors loader in vLLM cannot consume. Example:
    # `model.visual.pos_embed.weight` is an Embedding-like learned
    # parameter with shape (num_pos, hidden) — rank-2 but NOT a Linear.
    # Quantizing it produces `pos_embed.input_global_scale` etc. which
    # vLLM's VL runtime rejects with `KeyError: pos_embed.input_global_scale`
    # because its `model.visual.pos_embed` is a bare Parameter, not a
    # quantizable Linear module.
    _NON_LINEAR_RE = re.compile(
        r"(?:^|\.)("
        r"pos_embed"            # positional embedding (nn.Parameter/Embedding)
        r"|rotary_emb"          # rotary pos embed cache
        r")(?:\.|$)"
    )
    out: list[str] = []
    for name, shape in candidates:
        if len(shape) != 2:
            continue
        if _NON_LINEAR_RE.search(name):
            continue
        out.append(name[:-len(".weight")] if name.endswith(".weight") else name)
    return sorted(set(out))



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="sensitivity_probe pickle")
    ap.add_argument("--costs", required=True, help="measure_quant_cost pickle")
    ap.add_argument("--model-override", default=None,
                    help="Override the model path stored in probe.pkl's meta. "
                         "Useful when re-running allocator against a probe "
                         "whose container-side paths no longer exist (e.g., "
                         "the original source was at /src/qwen36 in a prior "
                         "container run but is now only accessible via a "
                         "different mount). Overrides both profile detection "
                         "and visual-Linear source discovery.")
    ap.add_argument("--target-bits", type=float, default=4.75)
    ap.add_argument("--formats", default="",
                    help="Comma-separated format names to consider; empty=all")
    ap.add_argument("--pareto-targets",
                    default="4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25",
                    help="Comma-separated budgets to sweep for Pareto curve")
    ap.add_argument("--layer-config", required=True,
                    help="Output AutoRound layer_config JSON")
    ap.add_argument("--pareto-csv", required=True, help="Output Pareto CSV")
    ap.add_argument(
        "--applicability-report",
        default=None,
        help=(
            "Optional JSON sidecar for candidates masked before DP by source, "
            "profile, divisibility, or runtime kernel-shape constraints. "
            "Defaults to format_applicability.json beside --pareto-csv."
        ),
    )
    ap.add_argument(
        "--pareto-output-dir",
        default=None,
        help=(
            "Optional directory where each feasible Pareto point is written "
            "as a per-Linear assignment JSON suitable for "
            "kl_sensitivity_probe --seed-assignment."
        ),
    )
    ap.add_argument("--no-fused-promote", action="store_true",
                    help="Skip fused-projection sibling promotion")
    ap.add_argument("--no-fused-aggregation", action="store_true",
                    help="Disable pre-DP aggregation of fused siblings "
                         "(qkv_proj / gate_up_proj). Falls back to the "
                         "legacy promote_fused post-pass with tightening "
                         "retries. Pre-aggregation is strictly better "
                         "for hitting the target bit budget exactly on "
                         "dense models; use this flag only for "
                         "back-compat experiments.")
    ap.add_argument("--enforce-family-coherence", action="store_true",
                    help="Error (instead of warn) if the format set contains "
                         "multiple candidates for the same bit tier (e.g. "
                         "NVFP4 and MXFP4 both at 4 bits)")
    ap.add_argument("--bit-precision", type=float, default=0.0001,
                    help="Knapsack bit-bin granularity in avg-bits/param "
                         "(smaller = slower; default 0.0001 → ~50000 bins). "
                         "Measured on MiniMax-M2.7: going from 0.001 to 0.0001 "
                         "cuts predicted Δloss ~10%% at the same bit budget. "
                         "Coarser values (0.01) leave 40%% on the table.")
    ap.add_argument("--threads", type=int, default=0,
                    help="OMP/numpy threads for DP (0 = default)")
    ap.add_argument("--target-profile",
                    choices=serving_profile_names(),
                    default="research",
                    help="Serving/backend constraint profile loaded from "
                         "prismaquant/serving_profile_specs.")
    ap.add_argument("--calibration", default=None,
                    help="Optional path to a JSON containing "
                         "'calibrated_gains[fmt] = α_fmt'. When present, "
                         "the per-(layer, format) predicted Δloss "
                         "is multiplied by α_fmt before the DP runs.")
    ap.add_argument("--overshoot-tolerance", type=float, default=0.01,
                    help="Maximum allowed overshoot (bits/param) of the "
                         "achieved budget over the requested target after "
                         "fused-sibling promotion. The DP is re-run with a "
                         "tightened target until overshoot is within tol.")
    ap.add_argument("--visual-format",
                    choices=["BF16", "NVFP4", "MXFP8"],
                    default="BF16",
                    help="Uniform format for all visual-encoder Linears "
                         "(`model.visual.blocks.*`). Phase 1 fallback: "
                         "assigned to every visual Linear when "
                         "--visual-sensitivity=uniform OR when --visual-"
                         "sensitivity=fisher but the probe / cost pickles "
                         "don't carry real visual Fisher data. BF16 (default) "
                         "reproduces passthrough behavior; NVFP4 / MXFP8 "
                         "shrink the tower to quantized storage via the "
                         "existing RTN math at export time.")
    ap.add_argument("--visual-sensitivity",
                    choices=["fisher", "uniform"],
                    default="fisher",
                    help="How visual-encoder Linears enter the allocator. "
                         "'fisher' (default) treats them as regular DP "
                         "candidates when the probe pickle carries real "
                         "multimodal Fisher stats (produced by "
                         "`incremental_probe --calibration-modality="
                         "multimodal`). If those stats are missing, falls "
                         "back to uniform --visual-format. 'uniform' forces "
                         "the Phase 1 path: every visual Linear gets "
                         "--visual-format regardless of what's in the probe.")
    ap.add_argument("--mtp-format",
                    choices=["BF16", "NVFP4", "MXFP8"],
                    default="BF16",
                    help="Uniform format for MTP Linears. BF16 is the "
                         "production default until MTP speculative-decode "
                         "acceptance is validated for quantized MTP weights.")
    args = ap.parse_args()

    if args.threads > 0:
        import os
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    # Detect the model profile from the probe's metadata. The probe
    # writes `meta.model` when it runs, so we can look up the HF
    # config at that path and map it to a registered ModelProfile.
    # Profile governs fused-sibling promotion (allocator's
    # `promote_fused`) and the vLLM-internal name remap
    # (`build_quantization_config` via export_native_compressed).
    from .model_profiles import detect_profile, DefaultProfile
    model_profile = DefaultProfile()
    with open(args.probe, "rb") as f:
        _probe_peek = pickle.load(f)
    validate_probe_payload(_probe_peek, args.probe)
    probe_model_path = _probe_peek.get("meta", {}).get("model")
    del _probe_peek
    if args.model_override:
        probe_model_path = args.model_override
        print(f"[alloc] model-override: {probe_model_path}", flush=True)
    if probe_model_path:
        model_profile = detect_profile(probe_model_path)
        print(f"[alloc] model profile: {model_profile.name} "
              f"(derived from {probe_model_path})", flush=True)

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)
    with open(args.costs, "rb") as f:
        cost_data = pickle.load(f)
    validate_probe_payload(probe, args.probe)
    validate_cost_payload(cost_data, args.costs)
    stats = probe["stats"]
    costs = cost_data["costs"]
    print(f"[alloc] stats: {len(stats)} Linears, costs: {len(costs)} Linears")

    if args.formats:
        fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    else:
        fmt_names = cost_data["formats"]
    specs = [fr.get_format(n) for n in fmt_names]
    specs_sorted = sorted(specs, key=lambda s: s.effective_bits)

    # --- Format-family coherence check -----------------------------------
    # A sensible format ladder has at most ONE format per bit tier. Having
    # both NVFP4 and MXFP4 (or MXFP6_E3M2 and MXFP6_E2M3) means the allocator
    # picks between them based on tiny measurement noise per-layer, which
    # produces a serving mess: two separate kernel paths for the same tier.
    #
    # We bucket formats by effective_bits rounded to 0.25 and warn when a
    # bucket has more than one member. If --enforce-family-coherence is
    # set we error instead.
    from collections import Counter as _Counter
    buckets: dict[float, list[str]] = {}
    for s in specs_sorted:
        key = round(s.effective_bits * 4) / 4
        buckets.setdefault(key, []).append(s.name)
    collisions = {k: v for k, v in buckets.items() if len(v) > 1}
    if collisions:
        msg = ("format set has multiple candidates at the same bit tier; "
               "the allocator will pick among them based on per-layer RTN "
               "noise, which is usually not what you want:\n"
               + "\n".join(f"  {k} bits: {v}" for k, v in collisions.items())
               + "\nRecommended bundles (vLLM serving, today):\n"
               "  Ship-ready     : NVFP4,MXFP8       (validated)\n"
               "  MX-pure        : MXFP4,MXFP8\n"
               "  Experimental   : NVFP4,MXFP6_E3M2,MXFP8   "
               "(MXFP6 hardware-supported on Blackwell, vLLM kernels not yet landed)")
        if args.enforce_family_coherence:
            raise SystemExit(f"[alloc] ERROR: {msg}")
        else:
            print(f"[alloc] WARNING: {msg}", flush=True)
    format_rank = {s.name: i for i, s in enumerate(specs_sorted)}
    format_specs = {s.name: s for s in specs}
    print(f"[alloc] formats (low→high bits): "
          f"{[f'{s.name}({s.effective_bits:.2f}b)' for s in specs_sorted]}")

    # Optional empirical calibration: per-format scalar gain α_f. When
    # absent, all gains default to 1.0.
    calibrated_gains: dict[str, float] = {}
    if args.calibration:
        with open(args.calibration) as f:
            cal_payload = json.load(f)
        cal_raw = cal_payload.get("calibrated_gains") or {}
        for fmt_name, gain_val in cal_raw.items():
            try:
                calibrated_gains[fmt_name] = float(gain_val)
            except (TypeError, ValueError):
                continue
        if calibrated_gains:
            print(f"[alloc] calibration loaded from {args.calibration}: "
                  f"{ {k: round(v, 4) for k, v in calibrated_gains.items()} }",
                  flush=True)
        else:
            print(f"[alloc] WARNING: {args.calibration} has no usable "
                  f"calibrated_gains; running uncalibrated", flush=True)

    # Source-dtype manifest drives passthrough-integrity filtering in
    # build_candidates. None when model path is unknown — candidates
    # fall back to cost-pickle-only gating (pre-passthrough behavior).
    source_manifest: dict[str, str] | None = None
    if probe_model_path:
        source_manifest = _scan_source_dtype_manifest(
            probe_model_path, model_profile)
        if source_manifest:
            n_fp8 = sum(1 for v in source_manifest.values() if v == "fp8")
            n_bf16 = sum(1 for v in source_manifest.values() if v == "bf16")
            print(f"[alloc] source-dtype manifest: {n_fp8} fp8, "
                  f"{n_bf16} bf16 (gates FP8_SOURCE/BF16 per source)",
                  flush=True)

    candidate_mask_records: list[dict] = []
    candidates = build_candidates(
        stats, costs, specs_sorted, calibrated_gains,
        source_manifest=source_manifest,
        target_profile=args.target_profile,
        mask_records=candidate_mask_records,
    )
    print(f"[alloc] candidates built for {len(candidates)} Linears")

    applicability_report_path = (
        Path(args.applicability_report)
        if args.applicability_report
        else Path(args.pareto_csv).with_name("format_applicability.json")
    )
    applicability_report_path.parent.mkdir(parents=True, exist_ok=True)
    pre_aggregation_availability = {
        spec.name: sum(
            1 for per_name in candidates.values()
            if any(c.fmt == spec.name for c in per_name)
        )
        for spec in specs_sorted
    }

    # Pre-aggregate fused siblings (qkv_proj, gate_up_proj, ...) into
    # single DP items. The DP can't pick mixed-sibling solutions because
    # there's only one item per group — so promote_fused becomes a no-op
    # on aggregated items and the overshoot-tightening loop collapses to
    # a single pass on well-behaved models. Must run AFTER the MoE
    # aggregation (it skips `.__fused__.` entries explicitly).
    if not args.no_fused_aggregation:
        stats, costs, candidates = aggregate_fused_siblings(
            stats, costs, specs_sorted, candidates, profile=model_profile,
            calibrated_gains=calibrated_gains)
        sib_groups = sum(1 for n in candidates if _FUSED_SIBLING_MARKER in n)
        print(f"[alloc] fused-sibling aggregation: {sib_groups} groups "
              f"(qkv_proj / gate_up_proj / ...)")

    candidates = filter_candidates_for_profile(candidates, args.target_profile)

    post_aggregation_availability = {
        spec.name: sum(
            1 for per_name in candidates.values()
            if any(c.fmt == spec.name for c in per_name)
        )
        for spec in specs_sorted
    }
    applicability_payload = {
        "schema": "prismaquant.format_applicability.v1",
        "target_profile": args.target_profile,
        "model_profile": getattr(model_profile, "name", ""),
        "formats": [spec.name for spec in specs_sorted],
        "probe": str(args.probe),
        "costs": str(args.costs),
        "pre_aggregation_candidate_availability": pre_aggregation_availability,
        "post_aggregation_candidate_availability": post_aggregation_availability,
        **summarize_applicability_masks(candidate_mask_records),
    }
    applicability_report_path.write_text(
        json.dumps(applicability_payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"[alloc] format applicability → {applicability_report_path}")

    def _solve_for_target(target_bits: float):
        """Solve the DP at one target bit budget."""
        assign, achieved_r = solve_with_promotion(
            stats, candidates, target_bits, format_specs, format_rank,
            args.bit_precision,
            no_fused_promote=args.no_fused_promote,
            overshoot_tolerance=args.overshoot_tolerance,
            profile=model_profile,
        )
        if assign is None:
            return None, float("nan"), float("inf")
        total = compute_assignment_predicted_dloss(assign, candidates)
        return assign, achieved_r, total

    def _expand_assignment_for_seed_json(
        assignment: dict[str, str],
    ) -> dict[str, str]:
        """Expand DP super-items into the per-Linear seed-assignment shape.

        The allocator can solve over fused-sibling super-items.
        The KL probe's seed path wants ordinary module qnames; it already
        handles legality, pinning, and fused coherence, but giving it expanded
        names preserves the intended frontier point instead of making the
        super-item markers look like unknown entries.
        """
        expanded = dict(assignment)
        if not args.no_fused_aggregation:
            expanded = expand_fused_sibling_assignment(expanded, stats)
        return promote_moe_pair(expanded, format_rank, profile=model_profile)

    def _assignment_bits_total(assignment: dict[str, str]) -> float:
        total = 0.0
        for name, fmt in assignment.items():
            entry = stats.get(name)
            if not isinstance(entry, dict):
                continue
            shape = _shape_from_stats(entry)
            total += 8.0 * fr.get_format(fmt).memory_bytes_for_shape(shape)
        return float(total)

    pareto_seed_records: list[dict] = []

    # Pareto sweep.
    targets = [float(x) for x in args.pareto_targets.split(",")]
    curve = []
    for t in targets:
        assign, achieved, total = _solve_for_target(t)
        if assign is None:
            curve.append({"target_bits": t, "feasible": False})
            continue
        format_counts = defaultdict(int)
        format_params = defaultdict(int)
        for name, fmt in assign.items():
            format_counts[fmt] += 1
            format_params[fmt] += stats[name]["n_params"]
        curve.append({
            "target_bits": t,
            "feasible": True,
            "achieved_bits": achieved,
            "predicted_dloss": total,
            **{f"layers_{k}": v for k, v in format_counts.items()},
            **{f"params_{k}": v for k, v in format_params.items()},
        })
        if args.pareto_output_dir:
            expanded = _expand_assignment_for_seed_json(assign)
            expanded_counts = defaultdict(int)
            for fmt in expanded.values():
                expanded_counts[fmt] += 1
            pareto_seed_records.append({
                "target_bits": float(t),
                "achieved_bits": float(achieved),
                "predicted_dloss": float(total),
                "assignment": expanded,
                "format_counts": dict(sorted(expanded_counts.items())),
                "bits_total": _assignment_bits_total(expanded),
            })

    # Output Pareto CSV
    keys = sorted({k for row in curve for k in row.keys()})
    with open(args.pareto_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in curve:
            w.writerow(row)
    print(f"[alloc] Pareto curve → {args.pareto_csv}")

    if args.pareto_output_dir:
        out_dir = Path(args.pareto_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        seen_payloads: set[str] = set()
        for idx, record in enumerate(pareto_seed_records):
            assignment = record["assignment"]
            digest_src = json.dumps(assignment, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(digest_src.encode()).hexdigest()[:12]
            if digest in seen_payloads:
                continue
            seen_payloads.add(digest)
            label = (
                f"allocator_target_{record['target_bits']:.4f}"
                f"_achieved_{record['achieved_bits']:.4f}_{digest}"
            ).replace(".", "p")
            path = out_dir / f"{label}.json"
            payload = {
                "schema": "prismaquant.allocator.pareto_assignment.v1",
                "label": label,
                "source": "allocator_pareto",
                "target_bits": float(record["target_bits"]),
                "achieved_bits": float(record["achieved_bits"]),
                "bits_total": float(record["bits_total"]),
                "predicted_dloss": float(record["predicted_dloss"]),
                "format_counts": record["format_counts"],
                "assignment": dict(sorted(assignment.items())),
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            manifest_rows.append({
                "label": label,
                "path": str(path),
                "target_bits": float(record["target_bits"]),
                "achieved_bits": float(record["achieved_bits"]),
                "bits_total": float(record["bits_total"]),
                "predicted_dloss": float(record["predicted_dloss"]),
                "format_counts": record["format_counts"],
            })
        (out_dir / "manifest.json").write_text(json.dumps({
            "schema": "prismaquant.allocator.pareto_manifest.v1",
            "probe": str(args.probe),
            "costs": str(args.costs),
            "formats": [s.name for s in specs_sorted],
            "target_bits": [float(x) for x in targets],
            "candidates": manifest_rows,
        }, indent=2, sort_keys=True) + "\n")
        print(
            f"[alloc] Pareto seed assignments → {out_dir} "
            f"({len(manifest_rows)} unique)",
            flush=True,
        )

    # Kneedle
    feasible = [row for row in curve if row.get("feasible")]
    if len(feasible) >= 3:
        kidx = kneedle([r["achieved_bits"] for r in feasible],
                       [r["predicted_dloss"] for r in feasible])
        knee = feasible[kidx]
        print(f"[alloc] suggested knee: target={knee['target_bits']}, "
              f"achieved={knee['achieved_bits']:.3f}, "
              f"Δloss={knee['predicted_dloss']:.3e}")

    # Print table
    print("\n  target  achieved     Δloss (pred)   " + "   ".join(
        f"{s.name[:11]:>11}" for s in specs_sorted))
    for row in curve:
        if not row.get("feasible"):
            print(f"  {row['target_bits']:>6.3f}  INFEASIBLE")
            continue
        fmt_str = "   ".join(
            f"{row.get(f'layers_{s.name}', 0):>11,}" for s in specs_sorted)
        print(f"  {row['target_bits']:>6.3f}  {row['achieved_bits']:>7.3f}  "
              f"{row['predicted_dloss']:>14.4e}   {fmt_str}")

    # Emit chosen layer_config for target_bits.
    assignment, achieved, total = _solve_for_target(args.target_bits)
    if assignment is None:
        raise SystemExit(
            f"Infeasible at target_bits={args.target_bits}. "
            "Consider raising the target or widening the format set.")
    print(
        f"[alloc] target_bits={args.target_bits}: "
        f"achieved_bits={achieved:.3f}, Δloss={total:.3e}",
        flush=True,
    )

    assignment_expanded = dict(assignment)

    # Expand fused-sibling super-Linears (qkv_proj / gate_up_proj).
    if not args.no_fused_aggregation:
        assignment_expanded = expand_fused_sibling_assignment(
            assignment_expanded, stats)

    # vLLM's FusedMoE requires all projections of the same expert to share
    # one scheme. This keeps per-Linear assignments serveable without
    # collapsing experts into allocator super-items.
    assignment_expanded = promote_moe_pair(
        assignment_expanded,
        format_rank,
        profile=model_profile,
    )

    # Visual-encoder Linear handling. Two paths:
    #
    # 1. --visual-sensitivity=fisher (default) + probe/cost have real
    #    visual entries → visual Linears already participated in the
    #    knapsack DP above with their own per-Linear Fisher + per-format
    #    RTN cost. No override needed; just make sure every discoverable
    #    visual Linear has an assignment entry (fall back to --visual-
    #    format for any that the probe missed, e.g. patch_embed Linears
    #    that the probe's regex didn't hit).
    #
    # 2. --visual-sensitivity=uniform OR Fisher missing → Phase 1 path:
    #    scan source checkpoint for visual Linears and stamp them all
    #    with --visual-format.
    visual_format = args.visual_format
    visual_sensitivity = args.visual_sensitivity

    def _visual_fisher_available(stats_d: dict, costs_d: dict) -> bool:
        """True when both the probe and cost pickles carry real visual
        entries — the signal a multimodal calibration pass ran."""
        any_visual_stats = any(_is_visual_linear(n) for n in stats_d)
        any_visual_costs = any(_is_visual_linear(n) for n in costs_d)
        return any_visual_stats and any_visual_costs

    fisher_visual_ok = (visual_sensitivity == "fisher"
                        and _visual_fisher_available(stats, costs))
    if visual_sensitivity == "fisher" and not fisher_visual_ok:
        print("[alloc] --visual-sensitivity=fisher requested but probe / "
              "cost pickles have no visual Linear entries; falling back "
              f"to --visual-format={visual_format} (Phase 1 uniform).",
              flush=True)

    if probe_model_path:
        visual_names_src = discover_visual_linears_from_source(probe_model_path)
    else:
        visual_names_src = []

    if fisher_visual_ok:
        # Fisher path: DP already placed visual Linears. Fill in any
        # discoverable visual Linear that the DP missed (e.g. the probe
        # regex matched only `visual.blocks.*` but the source has
        # `visual.merger.*` or `visual.patch_embed.*` too) with the
        # uniform --visual-format as a safety net.
        dp_visual_count = sum(1 for n in assignment_expanded
                              if _is_visual_linear(n))
        filled = 0
        for vname in visual_names_src:
            if vname not in assignment_expanded:
                assignment_expanded[vname] = visual_format
                filled += 1
        print(f"[alloc] --visual-sensitivity=fisher: DP placed "
              f"{dp_visual_count} visual Linears via per-Linear Fisher; "
              f"{filled} additional visual Linears (un-probed) stamped "
              f"with --visual-format={visual_format}.", flush=True)
    else:
        # Uniform path (Phase 1): stamp every discoverable visual Linear.
        if visual_names_src:
            for vname in visual_names_src:
                assignment_expanded[vname] = visual_format
            print(f"[alloc] --visual-format={visual_format}: assigned "
                  f"{len(visual_names_src)} visual Linears uniformly "
                  f"(source={probe_model_path})", flush=True)
        elif visual_format != "BF16":
            print(f"[alloc] --visual-format={visual_format}: no visual "
                  f"Linears found in source checkpoint — override is a "
                  f"no-op", flush=True)

    mtp_count = sum(1 for n in assignment_expanded if n.startswith("mtp."))
    if mtp_count:
        assignment_expanded = apply_mtp_format_override(
            assignment_expanded,
            args.mtp_format,
        )
        print(
            f"[alloc] --mtp-format={args.mtp_format}: assigned "
            f"{mtp_count} MTP Linears uniformly",
            flush=True,
        )

    # Passthrough-integrity belt-and-suspenders. The filter in
    # build_candidates drops mismatched FP8_SOURCE / BF16 per-Linear
    # candidate, but downstream aggregation + promotion (fused
    # siblings, MoE expert-unity) can in principle push a format onto
    # a group whose members have heterogeneous source dtypes. On
    # modern checkpoints this doesn't happen (siblings share source
    # dtype), but if it ever does we want a loud early failure rather
    # than a broken export artifact.
    if source_manifest:
        violations: list[tuple[str, str, str]] = []
        for name, fmt in assignment_expanded.items():
            if not _is_passthrough_format(fmt):
                continue
            kind = source_manifest.get(name)
            if kind is None:
                # Not in manifest — likely a visual Linear stamped via
                # --visual-format (bypasses the manifest by design) or
                # a name the profile rewrite didn't map. Skip.
                continue
            if not _passthrough_source_ok(fmt, kind):
                violations.append((name, fmt, kind))
        if violations:
            head = "\n  ".join(
                f"{n}: picked {f} but source is {k} "
                f"(requires {PASSTHROUGH_SOURCE_REQUIREMENTS[f]})"
                for n, f, k in violations[:10]
            )
            raise SystemExit(
                f"[alloc] passthrough-integrity violation: "
                f"{len(violations)} Linears have a passthrough format "
                f"picked over a mismatched source dtype. Sample:\n"
                f"  {head}\n"
                "The per-Linear filter should have excluded these — "
                "investigate fused-sibling / MoE-unity promotion."
            )

    layer_cfg = {}
    for name, fmt in assignment_expanded.items():
        if fmt in format_specs:
            layer_cfg[name] = format_specs[fmt].autoround_config()
        else:
            # Visual format outside the body's format set (e.g., user
            # passed --formats NVFP4,BF16 plus --visual-format MXFP8).
            # Resolve from the global registry.
            layer_cfg[name] = fr.get_format(fmt).autoround_config()

    out = Path(args.layer_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(layer_cfg, f, indent=2)

    counts = defaultdict(int)
    for fmt in assignment.values():
        counts[fmt] += 1
    print(f"\n[alloc] target={args.target_bits} achieved={achieved:.3f}")
    for fmt, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fmt:>14}: {n:>5} layers")
    print(f"\nLayer config → {out}")
    print(f"Feed to AutoRound via --layer_config {out}")


if __name__ == "__main__":
    main()
