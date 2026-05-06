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
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant.build_rtn_cache import iter_quantizable_tensors


@dataclass
class ProductionWeightCache:
    """Dict-like cache of production-faithful dequantized weights.

    Keys: ``(qname, fmt_canonical)``.  Values are EITHER:
      * ``torch.Tensor`` ([out, in] float32 or bf16) — in-memory cache
      * ``str`` (a path) — points to a per-Linear .pt file on disk;
        ``get()`` lazy-loads on first access and memoizes the tensor

    Disk-streaming mode (when ``cache_dir`` is set during fill) keeps
    fill-time peak memory bounded — only one weight is in RAM at a time
    instead of the full ~25 GB stack of all rendered Linears.  At
    polish time the lazy-load caches each weight in memory after first
    access, so steady-state behavior matches the in-memory mode.

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
    weights: dict[tuple[str, str], object]  # tensor OR str(path)
    levers: dict[str, bool]
    activation_max_abs: dict[str, float] | None = None
    failed: dict[tuple[str, str], str] | None = None
    cache_dir: str | None = None  # set when disk-streaming was used at fill time
    # Backward-compat alias for code that still reads ``activation_scales``.
    activation_scales: dict[str, float] | None = None
    # LRU eviction state for memoized tensor loads.  When non-None, the
    # in-memory cache holds at most ``mem_lru_max_bytes`` of tensor data;
    # least-recently-used entries are evicted back to their on-disk
    # filename when the budget is exceeded.  Default OFF for backward
    # compat; opt-in via ``enable_lru(...)``.
    _lru_order: list[tuple[str, str]] | None = None
    _lru_paths: dict[tuple[str, str], str] | None = None
    _lru_bytes: int = 0
    _lru_max_bytes: int = 0

    def __post_init__(self) -> None:
        # Normalize to ``activation_max_abs`` if a caller used the legacy
        # name.  After this, both attributes hold the same dict (max_abs).
        if self.activation_max_abs is None and self.activation_scales is not None:
            self.activation_max_abs = self.activation_scales
        elif self.activation_scales is None and self.activation_max_abs is not None:
            self.activation_scales = self.activation_max_abs

    def enable_lru(self, max_bytes: int) -> None:
        """Bound the in-memory tensor footprint to ``max_bytes`` via LRU
        eviction.  Required for very large disk-streamed caches (e.g.
        Qwen3.6-27B's ~46 GB of bf16 weights wouldn't fit in a 121 GB
        UMA box alongside the model + working set)."""
        self._lru_max_bytes = int(max_bytes)
        self._lru_order = []
        self._lru_paths = {}
        self._lru_bytes = 0

    def _evict_to_budget(self) -> None:
        if self._lru_order is None or self._lru_max_bytes <= 0:
            return
        while self._lru_bytes > self._lru_max_bytes and self._lru_order:
            evict_key = self._lru_order.pop(0)
            t = self.weights.get(evict_key)
            if isinstance(t, torch.Tensor):
                self._lru_bytes -= t.element_size() * t.numel()
                # Restore the filename so subsequent lookups still resolve.
                if self._lru_paths is not None and evict_key in self._lru_paths:
                    self.weights[evict_key] = self._lru_paths[evict_key]

    def prefetch(self, keys: Sequence[tuple[str, str]] | None = None,
                 max_workers: int = 4) -> int:
        """Eagerly load (a subset of) cache entries via a thread pool.

        ``keys=None`` prefetches every entry that's still on disk (the
        common case at polish startup).  Returns the number of newly-
        materialized tensors.

        Disk-streamed caches typically have torch.load latency ~50 ms
        per file (deserialization-bound, not I/O-bound).  Loading
        serially through 496 entries = ~25 sec; with 4 threads this
        drops to ~6 sec.  Subsequent ``.get()`` calls hit the in-memory
        copy (no torch.load), so per-trial materialization in polish
        becomes essentially free.
        """
        from concurrent.futures import ThreadPoolExecutor

        if keys is None:
            keys = [k for k, v in self.weights.items()
                    if not isinstance(v, torch.Tensor)]
        else:
            keys = [k for k in keys
                    if not isinstance(self.weights.get(k), torch.Tensor)]
        if not keys:
            return 0

        def _load_one(key):
            self._resolve_to_tensor(key)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_load_one, keys))
        return len(keys)

    def _resolve_to_tensor(self, key: tuple[str, str]) -> torch.Tensor | None:
        """Return the tensor at ``key`` (lazy-load from disk if needed).
        With LRU enabled, the freshly-loaded tensor is bookkept and the
        oldest entries get evicted back to filenames when the byte budget
        is exceeded.  Returns None if the key isn't present."""
        v = self.weights.get(key)
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            # Refresh LRU position.
            if self._lru_order is not None:
                if key in self._lru_order:
                    self._lru_order.remove(key)
                self._lru_order.append(key)
            return v
        # Treat anything non-tensor as a filename / path.
        path = str(v)
        if self.cache_dir and not Path(path).is_absolute():
            path = str(Path(self.cache_dir) / path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        self.weights[key] = loaded
        if self._lru_order is not None:
            # Remember the filename so we can evict back to it later.
            if self._lru_paths is None:
                self._lru_paths = {}
            self._lru_paths[key] = str(v)
            self._lru_bytes += loaded.element_size() * loaded.numel()
            self._lru_order.append(key)
            self._evict_to_budget()
        return loaded

    def get(self, name: str, fmt: str) -> torch.Tensor | None:
        # Try canonical alias variants the perturbed map exposes.
        candidates = [name]
        if name.endswith(".weight"):
            candidates.append(name[:-len(".weight")])
        if name.startswith("model.language_model."):
            candidates.append("model." + name[len("model.language_model."):])
        for cand in candidates:
            if (cand, fmt) in self.weights:
                return self._resolve_to_tensor((cand, fmt))
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
        by (qname, fmt).  Use ``validate_coverage`` to raise on any miss.

        Crucially, this checks key membership only — does NOT lazy-load
        tensors — so it stays cheap on disk-streaming caches with
        thousands of entries totalling tens of GB.
        """
        def _has_key(name: str, fmt: str) -> bool:
            cands = [name]
            if name.endswith(".weight"):
                cands.append(name[:-len(".weight")])
            if name.startswith("model.language_model."):
                cands.append("model." + name[len("model.language_model."):])
            return any((c, fmt) in self.weights for c in cands)

        hits: list[tuple[str, str]] = []
        misses: list[tuple[str, str]] = []
        for q in expected_qnames:
            for f in formats:
                if f.upper() in {"BF16", "MXFP8", "MXFP8_E4M3"}:
                    continue
                if _has_key(q, f.upper()):
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

    ``store_qnames`` controls which Linears get full activation tensors
    stored (memory-bounded by ``max_rows``).  All Linears in
    ``qnames`` get a per-Linear scalar ``max_abs`` recorded — that's
    cheap (one float per Linear) and needed by the cache's act-clip
    metadata even for Linears whose render is skipped via resume.
    """

    def __init__(self, model: nn.Module, qnames: set[str], max_rows: int,
                 store_qnames: set[str] | None = None):
        self.model = model
        self.qnames = qnames
        self.store_qnames = set(store_qnames) if store_qnames is not None else set(qnames)
        self.max_rows = int(max_rows)
        self.activations: dict[str, list[torch.Tensor]] = {}
        self.max_abs: dict[str, float] = {}
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
            if key in self.store_qnames:
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
            # Always update the cheap per-Linear max_abs scalar — needed
            # even for Linears we won't store activations for (so cache
            # has act-clip values for every assigned Linear).
            x_abs_max = float(x.detach().abs().max().item())
            prev = self.max_abs.get(key, 0.0)
            if x_abs_max > prev:
                self.max_abs[key] = x_abs_max
            # Only store the full activation tensor if this Linear is in
            # the store set.  Memory bound: store_qnames × max_rows × in.
            if key not in self.store_qnames:
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
    cache_dir: str | Path | None = None,
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

    # RESUME: when disk-streaming is on and prior shards exist, only
    # collect activations for Linears whose shards we still need to
    # render.  On a job that's 99%+ complete this drops activation
    # collection memory + compute by 99% — and lets a borderline-OOM
    # job finish on the same hardware.
    cache_dir_path: Path | None = None
    if cache_dir is not None:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)

    def _safe_path_early(qname: str, fmt: str) -> str:
        safe = qname.replace("/", "__").replace(".", "_")
        return f"{safe}__{fmt}.pt"

    qnames_to_render: set[str] = set(qname_set)
    if cache_dir_path is not None:
        # A qname is FULLY done if every requested format has a shard.
        fmts_upper = [f.upper() for f in formats]
        prerendered = 0
        for q in list(qname_set):
            if all((cache_dir_path / _safe_path_early(q, f)).is_file()
                   for f in fmts_upper):
                qnames_to_render.discard(q)
                prerendered += 1
        if progress and prerendered:
            print(
                f"[prod-cache] resume: {prerendered} qnames already on disk "
                f"({len(qnames_to_render)} still need rendering)",
                flush=True,
            )

    device = next(model.parameters()).device
    # RESUME: if all qnames are already rendered AND we have either a
    # sidecar OR no need for max_abs (no NVFP4 in formats), skip the
    # forward pass entirely.  Avoids OOM from the model's forward pass
    # itself on big models (e.g. linear-attention torch fallback can
    # spike memory mid-pass on Qwen3.5/3.6 27B+).
    sidecar_path: Path | None = (
        cache_dir_path / "activation_max_abs.json"
        if cache_dir_path is not None else None
    )
    skip_forward = (
        cache_dir_path is not None
        and not qnames_to_render
        and (
            (sidecar_path is not None and sidecar_path.is_file())
            or "NVFP4" not in {f.upper() for f in formats}
        )
    )
    collector = None  # may stay None on the skip_forward path
    if skip_forward:
        if progress:
            print(
                "[prod-cache] resume: all qnames pre-rendered + max_abs "
                "available, skipping activation forward pass",
                flush=True,
            )
        activations: dict[str, torch.Tensor] = {}
    else:
        # Hook every relevant Linear so we always get max_abs (cheap), but
        # only STORE full activations for Linears we still need to render.
        collector = _LinearActivationCollector(
            model,
            qnames=qname_set,
            max_rows=max_act_rows,
            store_qnames=qnames_to_render,
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

    weights: dict[tuple[str, str], object] = {}
    failed: dict[tuple[str, str], str] = {}
    qname_to_module: dict[str, nn.Module] = {}

    if cache_dir_path is not None and progress:
        print(f"[prod-cache] streaming cache to {cache_dir_path}/", flush=True)

    def _safe_path(qname: str, fmt: str) -> str:
        # Replace path-unsafe chars in qnames with __ for filename use.
        safe = qname.replace("/", "__").replace(".", "_")
        return f"{safe}__{fmt}.pt"
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

    # RESUME: load previously-computed max_abs values from the sidecar
    # JSON if disk-streaming + sidecar exists.  Lets a resume run skip
    # both activation collection and max_abs recomputation for already-
    # rendered qnames.  ``sidecar_path`` was defined earlier (before the
    # forward-skip decision); re-using it here.
    if sidecar_path is not None and sidecar_path.is_file():
        import json as _json
        try:
            activation_max_abs.update(_json.loads(sidecar_path.read_text()))
            if progress:
                print(
                    f"[prod-cache] resume: loaded {len(activation_max_abs)} "
                    f"max_abs entries from sidecar",
                    flush=True,
                )
        except Exception as e:
            if progress:
                print(
                    f"[prod-cache] sidecar load failed ({e}); recomputing",
                    flush=True,
                )

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
            # 1. Sidecar (resume) wins — these are the pre-computed values
            #    from a prior run.
            if qname in activation_max_abs:
                per_qname_max_abs[qname] = activation_max_abs[qname]
                continue
            # 2. Collector's per-Linear scalar (always populated for
            #    Linears that were hooked, even if no full activation
            #    tensor was stored).  ``collector`` is None on the
            #    skip_forward path, in which case we can only fall
            #    through to the activations-tensor path (which is
            #    empty on skip_forward, so we just continue).
            mx = (
                collector.max_abs.get(qname, 0.0)
                if collector is not None else 0.0
            )
            if mx <= 0:
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
        # Persist max_abs to sidecar so future resume runs can skip
        # activation collection entirely for completed qnames.
        if sidecar_path is not None and activation_max_abs:
            import json as _json
            sidecar_path.write_text(_json.dumps(activation_max_abs, indent=2))

    n = len(qname_to_module) * len(formats)
    done = 0
    skipped_resumed = 0
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
            # RESUME: in disk-streaming mode, if a shard already exists
            # for (qname, fmt) on disk, treat it as previously rendered
            # and skip re-rendering.  This lets a job that OOM'd at 95%
            # resume without re-doing the work — just rebuild the manifest
            # from the surviving .pt files.
            if cache_dir_path is not None:
                fname = _safe_path(qname, fmt.upper())
                if (cache_dir_path / fname).is_file():
                    weights[(qname, fmt.upper())] = fname
                    skipped_resumed += 1
                    # Do NOT pop activations_local[qname] here: this
                    # loop iterates through every format for this
                    # Linear, and a later format in the same outer
                    # iteration may still need the activation tensor
                    # to render.  The outer pop after the format loop
                    # drops it once all formats are done.
                    continue
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
            tensor = w_dq.to(target_dtype).cpu()
            if cache_dir_path is not None:
                # Disk-streaming mode: save to per-Linear .pt and store
                # only the relative filename in the cache.  Peak memory
                # during fill is bounded by the largest single render.
                # Atomic via tmp + rename: if the process is killed
                # mid-write, resume sees no .pt at all (and re-renders)
                # rather than a corrupt one (which would deserialize-
                # crash later).
                fname = _safe_path(qname, fmt.upper())
                final_path = cache_dir_path / fname
                tmp_path = cache_dir_path / (fname + ".tmp")
                torch.save(tensor, tmp_path)
                import os as _os
                _os.replace(tmp_path, final_path)
                weights[(qname, fmt.upper())] = fname
                del tensor
            else:
                weights[(qname, fmt.upper())] = tensor
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
        print(
            f"[prod-cache] rendered {len(weights)} (qname, fmt) entries "
            f"({skipped_resumed} resumed from disk); {len(failed)} failures",
            flush=True,
        )
    return ProductionWeightCache(
        weights=weights,
        levers=dict(levers),
        activation_max_abs=activation_max_abs or None,
        failed=failed,
        cache_dir=str(cache_dir_path) if cache_dir_path is not None else None,
    )
