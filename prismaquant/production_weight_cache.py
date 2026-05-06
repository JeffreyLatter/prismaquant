"""Production-faithful δw cache for Block-CLADO measurement.

Block-CLADO's surrogate (four-term identity) and real-KL gates measure
how perturbations propagate downstream.  Without this cache the
perturbation rendered into the model is bare RTN.  The export pipeline
renders weights with several activation-aware passes; the shipped δw is
much smaller than the RTN δw at the same format.

This module pre-renders `W_tilde[name, fmt]` once, using the production
quantization path:

  IMPLEMENTED (v1):
    * scalar GPTQ (with damp sweep when env-enabled)
    * scalar scale-sweep
    * joint NVFP4 fused-sibling globals (q/k/v share a per-tensor scale,
      gate/up share theirs)
    * calibrated `input_global_scale` per fused-sibling group
      (max_abs(activations) / 6.0; the same value the export persists
      to the artifact)

  KNOWN GAPS (v2 work, NOT implemented):
    * batched NVFP4 GPTQ + scale-sweep across same-shape Linears
      (defaults-on in the export when activations are cached;
      mathematically equivalent to scalar but ~3-8× faster on MoE)
    * block-output match (post-GPTQ refinement against BF16 block output)
    * HALO Hadamard rotation (only relevant when tied embeddings are
      untied; not used on Qwen3-4B / 27B)
    * AWQ predecessor folding (`_awq_fold_layer_predecessors`)
    * any export-only refinements added after this docstring is written

  MXFP8/BF16:
    * MXFP8 is RTN-only by construction in the export.
    * BF16 is passthrough.
    Both produce the same weight under this cache as under
    `spec.quantize_dequantize`, so they get fast-path passthrough here.

PerturbedActivationCache installs `W_tilde` (and applies the calibrated
`input_global_scale` on activations) instead of RTN-quantizing on the
fly, so every measurement — four-term, cone validate, polish gate — uses
the same δw the export will deliver, modulo the v2 gaps above.

Usage:

    cache = fill_production_weight_cache(
        model, calib_ids, qnames=qnames, formats=["NVFP4"],
    )
    cache.validate_coverage(qnames, ["NVFP4"])  # raise on misses
    perturbed_cache = PerturbedActivationCache(
        ..., production_weight_cache=cache,
    )
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from prismaquant.build_rtn_cache import iter_quantizable_tensors


@dataclass
class ProductionWeightCache:
    """Dict-like cache of production-faithful dequantized weights.

    Keys: ``(qname, fmt_canonical)``.  Values: ``[out, in]`` float32 or
    bf16 tensors that match the live module weight shape after applying
    the production pipeline (GPTQ + scale_sweep + joint sibling NVFP4
    globals at minimum; HALO, etc. when those layers are extended).

    ``activation_max_abs[qname]`` is the calibrated max(|activations|)
    used by the act-clip step in the export pipeline.  PerturbedActivation
    Cache reads this and clamps activations to ``[-max_abs, +max_abs]``
    before per-group RTN, matching the export's act-clip behavior.

    Note: the *exported metadata* convention for this field is
    ``input_global_scale = 6.0 / max_abs`` (reciprocal — vLLM multiplies
    activations by it).  We store ``max_abs`` directly here because
    that's the value the act-clip path needs; consumers can convert if
    they need the metadata convention.
    """
    weights: dict[tuple[str, str], torch.Tensor]
    levers: dict[str, bool]
    activation_max_abs: dict[str, float] | None = None
    failed: dict[tuple[str, str], str] | None = None
    # Backward-compat alias for code that still reads ``activation_scales``.
    activation_scales: dict[str, float] | None = None

    def __post_init__(self) -> None:
        # Normalize to ``activation_max_abs`` if a caller used the legacy
        # name.  After this, both attributes hold the same dict (max_abs).
        if self.activation_max_abs is None and self.activation_scales is not None:
            self.activation_max_abs = self.activation_scales
        elif self.activation_scales is None and self.activation_max_abs is not None:
            self.activation_scales = self.activation_max_abs

    def get(self, name: str, fmt: str) -> torch.Tensor | None:
        # Try canonical alias variants the perturbed map exposes.
        candidates = [name]
        if name.endswith(".weight"):
            candidates.append(name[:-len(".weight")])
        if name.startswith("model.language_model."):
            candidates.append("model." + name[len("model.language_model."):])
        for cand in candidates:
            if (cand, fmt) in self.weights:
                return self.weights[(cand, fmt)]
        return None

    def __contains__(self, key: tuple[str, str]) -> bool:
        # Mirror the alias-resolution that ``get`` performs.
        name, fmt = key
        candidates = [name]
        if name.endswith(".weight"):
            candidates.append(name[:-len(".weight")])
        if name.startswith("model.language_model."):
            candidates.append("model." + name[len("model.language_model."):])
        return any((c, fmt) in self.weights for c in candidates)

    def __len__(self) -> int:
        return len(self.weights)

    def coverage_report(
        self,
        expected_qnames: Sequence[str],
        formats: Sequence[str],
    ) -> dict:
        """Return a dict with ``hits``, ``misses``, ``failed`` lists keyed
        by (qname, fmt).  Use ``validate_coverage`` to raise on any miss."""
        hits: list[tuple[str, str]] = []
        misses: list[tuple[str, str]] = []
        for q in expected_qnames:
            for f in formats:
                if f.upper() in {"BF16", "MXFP8", "MXFP8_E4M3"}:
                    continue
                if self.get(q, f.upper()) is not None:
                    hits.append((q, f.upper()))
                else:
                    misses.append((q, f.upper()))
        return {
            "hits": hits,
            "misses": misses,
            "failed": list((self.failed or {}).keys()),
        }

    def validate_coverage(
        self,
        expected_qnames: Sequence[str],
        formats: Sequence[str],
    ) -> None:
        """Raise ``RuntimeError`` if any (qname, fmt) is missing from the
        cache.  Call this immediately after fill to catch silent gaps
        from naming aliases or render failures."""
        report = self.coverage_report(expected_qnames, formats)
        if report["misses"] or report["failed"]:
            samples = (report["misses"][:5] + report["failed"][:5])
            raise RuntimeError(
                f"ProductionWeightCache coverage failure: "
                f"{len(report['misses'])} misses, "
                f"{len(report['failed'])} failed renders; "
                f"sample={samples}"
            )


class _LinearActivationCollector:
    """Hook every quantizable nn.Linear's input on a forward pass.

    Stores up to ``max_rows`` rows of activations per Linear (concatenated
    across calibration samples) as float32 CPU tensors.  Only handles
    ``nn.Linear`` for now — packed MoE experts route through different
    APIs in the export pipeline and would need a separate collector.
    """

    def __init__(self, model: nn.Module, qnames: set[str], max_rows: int):
        self.model = model
        self.qnames = qnames
        self.max_rows = int(max_rows)
        self.activations: dict[str, list[torch.Tensor]] = {}
        self._handles: list = []
        self._name_by_id: dict[int, str] = {}
        for full_name, mod, attr in iter_quantizable_tensors(model):
            if attr != "weight" or not isinstance(mod, nn.Linear):
                continue
            qname = full_name[:-7] if full_name.endswith(".weight") else full_name
            if qname not in qnames and full_name not in qnames:
                continue
            key = qname
            self._name_by_id[id(mod)] = key
            self.activations[key] = []

    def install(self) -> None:
        for mod_id, key in self._name_by_id.items():
            for full_name, mod, attr in iter_quantizable_tensors(self.model):
                if id(mod) != mod_id or attr != "weight":
                    continue
                self._handles.append(
                    mod.register_forward_pre_hook(self._make_hook(key))
                )
                break

    def _make_hook(self, key: str):
        def hook(module, args):
            if not args:
                return
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            flat = x.detach().reshape(-1, x.shape[-1]).to(torch.float32).cpu()
            existing = sum(t.shape[0] for t in self.activations[key])
            if existing >= self.max_rows:
                return
            take = min(flat.shape[0], self.max_rows - existing)
            self.activations[key].append(flat[:take].clone())
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def collected(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, parts in self.activations.items():
            if not parts:
                continue
            out[key] = torch.cat(parts, dim=0)
        return out


@contextmanager
def _temporarily_install_act_aware(
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, bool],
):
    """Install module-level state expected by ``_quantize_2d``.

    The export module reads ``_CACHED_ACTIVATIONS`` and ``_ACT_AWARE_FLAGS``
    from its own globals to decide what passes to run.  We mutate these
    inside a try/finally so concurrent export work isn't disturbed.
    """
    from prismaquant import export_native_compressed as enc

    prev_cache = enc._CACHED_ACTIVATIONS
    prev_flags = dict(enc._ACT_AWARE_FLAGS)
    enc._CACHED_ACTIVATIONS = _DictActivations(activations)
    enc._ACT_AWARE_FLAGS = {
        "awq": bool(levers.get("awq", False)),
        "gptq": bool(levers.get("gptq", True)),
        "awq_round": bool(levers.get("awq_round", False)),
        "scale_sweep": bool(levers.get("scale_sweep", True)),
    }
    try:
        yield
    finally:
        enc._CACHED_ACTIVATIONS = prev_cache
        enc._ACT_AWARE_FLAGS.clear()
        enc._ACT_AWARE_FLAGS.update(prev_flags)


class _DictActivations:
    """`.get(name)` shim matching `_LazyActivationCache`'s interface."""

    def __init__(self, mapping: Mapping[str, torch.Tensor]):
        self._mapping = mapping

    def get(self, name: str) -> torch.Tensor | None:
        a = self._mapping.get(name)
        if a is None and name.endswith(".weight"):
            a = self._mapping.get(name[:-7])
        return a


def render_production_weight(
    weight: torch.Tensor,
    fmt: str,
    *,
    qname: str,
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, bool],
    joint_global_real: torch.Tensor | None = None,
    input_global_scale: float | None = None,
) -> torch.Tensor:
    """Compute the production-faithful dequantized weight for ``(qname, fmt)``.

    Returns a tensor matching ``weight.shape`` and dtype.  For NVFP4 this
    runs GPTQ + scale_sweep (the activation-aware passes) with the joint
    fused-sibling NVFP4 global if supplied; for MXFP8 / BF16 it falls
    back to the format's quantize_dequantize because those formats don't
    benefit from activation-aware refinement in the production pipeline.

    ``joint_global_real`` is the max-across-fused-siblings NVFP4 global
    used to keep q/k/v (or gate/up) per-tensor scales unified — same as
    the export's ``_compute_nvfp4_joint_global``.  When ``None`` the
    per-Linear computed value is used (legacy behavior, only correct for
    isolated Linears with no fused siblings).
    """
    fmt = fmt.upper()
    if fmt in ("MXFP8", "MXFP8_E4M3", "BF16"):
        from prismaquant import format_registry as fr
        spec = fr.get_format(fmt)
        return spec.quantize_dequantize(weight.detach().clone()).to(
            device=weight.device, dtype=weight.dtype,
        )
    if fmt != "NVFP4":
        raise ValueError(f"render_production_weight: unsupported fmt={fmt!r}")

    from prismaquant.export_native_compressed import _quantize_2d

    with _temporarily_install_act_aware(activations, levers):
        result = _quantize_2d(
            weight.detach().clone(),
            fmt="NVFP4",
            linear_name=qname,
            nvfp4_global_real_override=joint_global_real,
            input_global_scale_override=input_global_scale,
            compute_only=True,
        )
    w_dq = result["_w_dq"]
    return w_dq.to(device=weight.device, dtype=weight.dtype).contiguous()


def fill_production_weight_cache(
    model: nn.Module,
    calib_ids: torch.Tensor,
    qnames: Sequence[str],
    *,
    formats: Sequence[str] = ("NVFP4",),
    levers: Mapping[str, bool] | None = None,
    max_act_rows: int = 256,
    progress: bool = True,
) -> ProductionWeightCache:
    """End-to-end fill: collect activations, render production δw per
    (qname, fmt), return a `ProductionWeightCache`.

    Args:
      model: live HF model on the export device.
      calib_ids: ``[N, T]`` token id tensor for activation collection.
      qnames: which Linears to render (skips MoE packed experts; handle
        those separately via `_quantize_3d_packed` extensions).
      formats: which formats to pre-render.  MXFP8/BF16 are RTN-equivalent
        so we still cache them so the lookup is uniform.
      levers: which production levers to enable (default: gptq+scale_sweep).
    """
    levers = dict(levers) if levers is not None else {}
    levers.setdefault("gptq", True)
    levers.setdefault("scale_sweep", True)
    levers.setdefault("awq", False)
    levers.setdefault("awq_round", False)

    qname_set = set(qnames)
    if not qname_set:
        return ProductionWeightCache(weights={}, levers=dict(levers))

    device = next(model.parameters()).device
    collector = _LinearActivationCollector(
        model, qnames=qname_set, max_rows=max_act_rows,
    )
    collector.install()
    try:
        with torch.no_grad():
            for i in range(calib_ids.size(0)):
                batch = calib_ids[i:i + 1].to(device)
                model(batch)
    finally:
        collector.remove()
    activations = collector.collected()

    if progress:
        print(
            f"[prod-cache] collected activations for "
            f"{len(activations)}/{len(qname_set)} Linears",
            flush=True,
        )

    weights: dict[tuple[str, str], torch.Tensor] = {}
    failed: dict[tuple[str, str], str] = {}
    qname_to_module: dict[str, nn.Module] = {}
    for full_name, mod, attr in iter_quantizable_tensors(model):
        if attr != "weight" or not isinstance(mod, nn.Linear):
            continue
        qname = full_name[:-7] if full_name.endswith(".weight") else full_name
        if qname in qname_set:
            qname_to_module[qname] = mod

    # HIGH-1: compute joint NVFP4 fused-sibling globals so q/k/v share a
    # per-tensor scale (and gate/up likewise), matching the export's
    # `_compute_nvfp4_joint_global` behavior.  Without this each sibling
    # gets its own scale and vLLM's loader either rejects the artifact or
    # silently runs with degraded accuracy.
    joint_globals: dict[str, torch.Tensor] = {}
    if "NVFP4" in {f.upper() for f in formats}:
        from prismaquant.export_native_compressed import (
            _compute_nvfp4_joint_global,
        )
        synthetic_assignment = {q: "NVFP4" for q in qname_to_module}
        joint_globals = _compute_nvfp4_joint_global(model, synthetic_assignment)
        if progress:
            print(
                f"[prod-cache] computed joint NVFP4 globals for "
                f"{len(joint_globals)} fused-sibling members",
                flush=True,
            )

    # MED-3: per-Linear calibrated max_abs used by the export's act-clip
    # step.  For fused-sibling groups the value is unified (max across
    # siblings), matching the export's joint input_global_scale derivation.
    # We store max_abs directly (not 6/max_abs) — see ProductionWeightCache
    # docstring on the convention difference.
    activation_max_abs: dict[str, float] = {}
    fmt_set = {f.upper() for f in formats}
    if "NVFP4" in fmt_set:
        # Group by fused sibling key for max-across-siblings unification.
        from prismaquant.block_clado import fused_group_key
        try:
            from prismaquant.model_profiles import detect_profile
            profile = detect_profile(getattr(model, "name_or_path", "")) \
                if hasattr(model, "name_or_path") else None
        except Exception:
            profile = None

        per_qname_max_abs: dict[str, float] = {}
        for qname, _ in qname_to_module.items():
            a = activations.get(qname)
            if a is None:
                continue
            mx = float(a.abs().max().item())
            if mx <= 0:
                continue
            per_qname_max_abs[qname] = mx

        # Unify across fused sibling groups by taking the max.
        groups: dict[str, list[str]] = {}
        for qname in per_qname_max_abs:
            try:
                gk = fused_group_key(profile, qname) if profile else qname
            except Exception:
                gk = qname
            groups.setdefault(gk, []).append(qname)
        for gk, members in groups.items():
            shared = max(per_qname_max_abs[m] for m in members)
            for m in members:
                activation_max_abs[m] = shared
        if progress and activation_max_abs:
            print(
                f"[prod-cache] computed activation max_abs for "
                f"{len(activation_max_abs)} Linears "
                f"({len(groups)} fused groups)",
                flush=True,
            )

    n = len(qname_to_module) * len(formats)
    done = 0
    # MEM: free per-Linear activation tensors after each render so peak
    # memory stays bounded.  On 27B, 497 Linears × ~10K in_features × 512
    # rows × fp32 = ~10 GB just for activations.  Freeing in-loop drops
    # this to ~20 MB resident.
    import gc as _gc
    activations_local = dict(activations)  # shallow copy; we'll pop entries
    for qname, mod in qname_to_module.items():
        weight = mod.weight.data
        joint = joint_globals.get(qname)
        max_abs = activation_max_abs.get(qname)
        # _quantize_2d's input_global_scale_override expects the export
        # convention (6.0 / max_abs).  It only affects emitted metadata
        # in compute_only mode (not the dequantized weight values), but
        # we pass the correct convention so the metadata is honest in
        # case future code consumes it.
        export_scale = (6.0 / max_abs) if (max_abs is not None and max_abs > 0) else None
        for fmt in formats:
            try:
                w_dq = render_production_weight(
                    weight, fmt,
                    qname=qname,
                    activations=activations_local,
                    levers=levers,
                    joint_global_real=joint,
                    input_global_scale=export_scale,
                )
            except Exception as e:
                failed[(qname, fmt.upper())] = str(e)
                if progress:
                    print(
                        f"[prod-cache] FAILED {qname} @ {fmt}: {e}",
                        flush=True,
                    )
                continue
            # MEM: store as the model's native dtype (bf16 by default)
            # rather than fp32 — _quantize_2d's compute_only path returns
            # fp32 but we always re-cast at install time, so storing fp32
            # is wasteful (2× memory).  On 27B this drops the cache from
            # ~25 GB to ~12 GB.
            target_dtype = weight.dtype if weight.dtype != torch.float32 else torch.bfloat16
            weights[(qname, fmt.upper())] = w_dq.to(target_dtype).cpu()
            done += 1
            del w_dq
            if progress and done % 25 == 0:
                print(f"[prod-cache] {done}/{n}", flush=True)
        # Free this Linear's activation tensor — won't render this qname
        # again, and the activation can be tens of MB on big models.
        activations_local.pop(qname, None)
        if done % 50 == 0:
            _gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
    if progress:
        print(f"[prod-cache] rendered {len(weights)} (qname, fmt) entries; "
              f"{len(failed)} failures", flush=True)
    return ProductionWeightCache(
        weights=weights,
        levers=dict(levers),
        activation_max_abs=activation_max_abs or None,
        failed=failed,
    )
