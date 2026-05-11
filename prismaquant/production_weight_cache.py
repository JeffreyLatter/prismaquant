"""Production-faithful rendered-weight cache for measured candidates.

Per-Linear candidate probes and real-KL gates need to measure the same
rendered weights that export will ship. Without this cache the perturbation
installed into the model is bare RTN. The export pipeline renders weights with
several activation-aware passes; the shipped δw is much smaller than the RTN
δw at the same format.

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
    * optional PrismaClip explicit activation-clipping solver for NVFP4 render-time
      GPTQ + scale_sweep inputs
    * optional PrismaFisherClip candidate scoring, which reuses h-detail
      per-token Fisher weights to accept/reject PrismaClip thresholds
    * optional Fisher-weighted local objectives from h-detail
    * opt-in activation-weighted MXFP8 E8M0 scale search when nonzero
      PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS are configured
    * opt-in AWQ-v2 normalization-predecessor fold scales, searched by
      rendered-weight output MSE and stored as effective cache weights

  KNOWN GAPS (v2 work, NOT implemented):
    * batched NVFP4 GPTQ + scale-sweep across same-shape Linears
      (defaults-on in the export when activations are cached;
      mathematically equivalent to scalar but ~3-8× faster on MoE)
    * block-output match (post-GPTQ refinement against BF16 block output)
    * HALO Hadamard rotation (only relevant when tied embeddings are
      untied; not used on Qwen3-4B / 27B)
    * any export-only refinements added after this docstring is written

  MXFP8/BF16:
    * MXFP8 uses the same scale path as export. The default is RTN-equivalent;
      nonzero exponent-shift search remains opt-in pending a positive KL gate.
    * BF16 is passthrough.

PerturbedActivationCache installs `W_tilde` (and applies the calibrated
`input_global_scale` on activations) instead of RTN-quantizing on the
fly, so per-Linear probes, frontier validation, and polish gates use the
same δw the export will deliver, modulo the v2 gaps above.

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
import math
import os
from pathlib import Path
import re

import torch
import torch.nn as nn

from prismaquant.awq import (
    DEFAULT_AWQ_MIN_GAIN,
    AwqSearchTarget,
    awq_output_mse,
    normalize_awq_scale,
    search_awq_scale,
)
from prismaquant.build_rtn_cache import iter_quantizable_tensors


DEFAULT_ACT_CLIP_SOLVER_MIN_GAIN = 0.0
DEFAULT_ACT_CLIP_SOLVER_HOLDOUT_FRACTION = 0.25
PRISMACLIP_FORMAT = "NVFP4_CLIPPED"


def _is_prismaclip_format(fmt: str) -> bool:
    return str(fmt).strip().upper() == PRISMACLIP_FORMAT


def _render_base_format(fmt: str) -> str:
    fmt_u = str(fmt).strip().upper()
    return "NVFP4" if fmt_u == PRISMACLIP_FORMAT else fmt_u


def _cache_weight_filename(qname: str, fmt: str) -> str:
    safe = qname.replace("/", "__").replace(".", "_")
    return f"{safe}__{fmt}.pt"


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
    metadata: dict[str, object] | None = None
    # Backward-compat alias for code that still reads ``activation_scales``.
    activation_scales: dict[str, float] | None = None
    # Per-Linear AWQ fold scales.  Cache weights are stored in original
    # coordinates as Q(W*s)/s for faithful measurement in the unfolded model;
    # export multiplies by this scale and folds predecessor norms so the
    # artifact stores Q(W*s) and serves the same function.
    awq_scales: dict[str, torch.Tensor] | None = None
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

    def compact_for_pickle(self) -> int:
        """Restore disk-backed resident tensors to path references.

        A recache/polish pass may lazily load many disk-streamed entries into
        ``weights``.  Pickling that state would serialize the tensors and turn a
        small manifest into a multi-GB file.  This method keeps the cache
        portable by replacing resident LRU-loaded tensors with their original
        paths before serialization.  Returns the number of entries compacted.
        """
        compacted = 0
        for key, path in (self._lru_paths or {}).items():
            if isinstance(self.weights.get(key), torch.Tensor):
                self.weights[key] = path
                compacted += 1
        if self.cache_dir:
            cache_dir = Path(self.cache_dir)
            for key, value in list(self.weights.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                fname = _cache_weight_filename(key[0], key[1])
                if (cache_dir / fname).is_file():
                    self.weights[key] = fname
                    compacted += 1
        self._lru_order = [] if self._lru_order is not None else None
        self._lru_bytes = 0
        return compacted

    def _path_for_value(self, value: object) -> str:
        path = str(value)
        if self.cache_dir and not Path(path).is_absolute():
            path = str(Path(self.cache_dir) / path)
        return path

    def _name_candidates(self, name: str) -> list[str]:
        candidates = [name]
        if name.endswith(".weight"):
            candidates.append(name[:-len(".weight")])
        if name.startswith("model.language_model."):
            candidates.append("model." + name[len("model.language_model."):])
        return list(dict.fromkeys(candidates))

    def _format_candidates(self, fmt: str) -> list[str]:
        raw = str(fmt)
        candidates = [raw, raw.upper()]
        try:
            from prismaquant import format_registry as fr
            candidates.append(fr.canonical_format_name(raw))
        except Exception:
            pass
        if "MXFP8_E4M3" in candidates:
            candidates.append("MXFP8")
        if "MXFP8" in candidates:
            candidates.append("MXFP8_E4M3")
        if "FP8_E4M3" in candidates:
            candidates.append("FP8")
        if "FP8" in candidates:
            candidates.append("FP8_E4M3")
        return list(dict.fromkeys(candidates))

    def resolve_key(self, name: str, fmt: str) -> tuple[str, str] | None:
        """Resolve recipe aliases to the concrete stored cache key."""
        for cand in self._name_candidates(name):
            for fmt_cand in self._format_candidates(fmt):
                key = (cand, fmt_cand)
                if key in self.weights:
                    return key
        return None

    def estimate_nbytes(
        self,
        keys: Sequence[tuple[str, str]] | None = None,
    ) -> int:
        """Estimate resident bytes for cache entries without loading them."""
        total = 0
        for key in (list(self.weights) if keys is None else list(keys)):
            value = self.weights.get(key)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                total += value.element_size() * value.numel()
            else:
                total += Path(self._path_for_value(value)).stat().st_size
        return total

    def assignment_keys(
        self,
        assignment: Mapping[str, str],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Return concrete non-BF16 cache keys needed by an assignment.

        This centralizes recipe alias handling for recache, polish, KL
        probes, and export: callers should ask the cache which stored key a
        recipe entry maps to, then feed those keys into ``prefetch``.
        """
        from prismaquant import format_registry as fr

        keys: list[tuple[str, str]] = []
        missing: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for qname, fmt in assignment.items():
            fmt_canon = fr.canonical_format_name(str(fmt))
            if fmt_canon == "BF16":
                continue
            key = self.resolve_key(str(qname), fmt_canon)
            if key is None:
                missing.append((str(qname), fmt_canon))
                continue
            if key not in seen:
                keys.append(key)
                seen.add(key)
        return keys, missing

    def prefetch_assignment(
        self,
        assignment: Mapping[str, str],
        *,
        max_resident_bytes: int | None = None,
        max_workers: int = 4,
        require: bool = False,
        progress: bool = False,
        log_prefix: str = "[prod-cache]",
    ) -> dict[str, object]:
        """Prefetch rendered weights required by a concrete assignment.

        ``require`` converts missing entries or resident-budget overflow into
        a hard failure.  That is the production-safe mode for GPU-bound
        recache/export runs because it prevents accidental NVMe streaming.
        """
        keys, missing = self.assignment_keys(assignment)
        nbytes = self.estimate_nbytes(keys)
        budget = (
            int(max_resident_bytes)
            if max_resident_bytes is not None and int(max_resident_bytes) > 0
            else None
        )
        stats: dict[str, object] = {
            "keys": len(keys),
            "missing": len(missing),
            "bytes": int(nbytes),
            "budget_bytes": int(budget or 0),
            "loaded": 0,
            "skipped": False,
        }
        if missing:
            stats["missing_sample"] = missing[:8]
            msg = (
                f"production cache missing {len(missing)} assignment entries; "
                f"sample={missing[:8]}"
            )
            if require:
                raise RuntimeError(msg)
            if progress:
                print(f"{log_prefix} WARNING: {msg}", flush=True)
        if budget is not None and nbytes > budget:
            stats["skipped"] = True
            msg = (
                "production cache preload would exceed resident budget: "
                f"{nbytes / 1024**3:.2f} GiB needed, "
                f"{budget / 1024**3:.2f} GiB budget"
            )
            if require:
                raise RuntimeError(msg)
            if progress:
                print(f"{log_prefix} WARNING: {msg}; skipping preload", flush=True)
            return stats

        if progress:
            print(
                f"{log_prefix} preloading production cache: "
                f"{len(keys)} entries, {nbytes / 1024**3:.2f} GiB",
                flush=True,
            )
        loaded = self.prefetch(keys, max_workers=max_workers)
        stats["loaded"] = int(loaded)
        if progress:
            print(
                f"{log_prefix} preloaded {loaded}/{len(keys)} production "
                "cache entries",
                flush=True,
            )
        return stats

    def _record_lru_load(
        self,
        key: tuple[str, str],
        original_value: object,
        tensor: torch.Tensor,
    ) -> None:
        if self._lru_paths is None:
            self._lru_paths = {}
        if key not in self._lru_paths:
            self._lru_paths[key] = str(original_value)
        if self._lru_order is None:
            return
        if key in self._lru_order:
            self._lru_order.remove(key)
        self._lru_bytes += tensor.element_size() * tensor.numel()
        self._lru_order.append(key)
        self._evict_to_budget()

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
            value = self.weights.get(key)
            if value is None or isinstance(value, torch.Tensor):
                return None
            return (
                key,
                value,
                torch.load(
                    self._path_for_value(value),
                    map_location="cpu",
                    weights_only=True,
                ),
            )

        loaded_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for item in pool.map(_load_one, keys):
                if item is None:
                    continue
                key, original_value, tensor = item
                if isinstance(self.weights.get(key), torch.Tensor):
                    continue
                self.weights[key] = tensor
                self._record_lru_load(key, original_value, tensor)
                loaded_count += 1
        return loaded_count

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
        path = self._path_for_value(v)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        self.weights[key] = loaded
        self._record_lru_load(key, v, loaded)
        return loaded

    def get(self, name: str, fmt: str) -> torch.Tensor | None:
        key = self.resolve_key(name, fmt)
        if key is not None:
            return self._resolve_to_tensor(key)
        return None

    def relocate(self, new_cache_dir: str | Path) -> None:
        """Point the cache at a new on-disk directory of .pt shards.

        Used when a pickled cache is moved to a new host or when a
        cache_dir set inside one container is re-mounted at a different
        path on a second container.  No tensor reload happens here; the
        next ``get()`` will resolve against the new path.
        """
        self.cache_dir = str(new_cache_dir) if new_cache_dir is not None else None

    def verify_files(
        self,
        expected: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, list[tuple[str, str]]]:
        """Verify every disk-resident cache entry's .pt file exists.

        Returns ``{"present": [...], "missing": [...], "in_memory": [...]}``
        keyed by (qname, fmt).  In-memory entries (already-loaded tensors)
        are reported separately and never count as missing.

        On a disk-streaming cache that has been moved or whose backing
        directory was deleted, this is the canonical way to detect the
        problem at startup rather than at first ``get()`` (which raises
        FileNotFoundError mid-polish).  Callers should treat any
        ``missing`` entry as fatal: the cache must be rebuilt or its
        directory restored before use.

        ``expected``, when given, restricts the check to that subset of
        keys.  Default checks every entry in ``self.weights``.
        """
        present: list[tuple[str, str]] = []
        missing: list[tuple[str, str]] = []
        in_memory: list[tuple[str, str]] = []
        keys = list(self.weights) if expected is None else list(expected)
        for key in keys:
            v = self.weights.get(key)
            if v is None:
                missing.append(key)
                continue
            if isinstance(v, torch.Tensor):
                in_memory.append(key)
                continue
            path = str(v)
            if self.cache_dir and not Path(path).is_absolute():
                path = str(Path(self.cache_dir) / path)
            if Path(path).is_file():
                present.append(key)
            else:
                missing.append(key)
        return {"present": present, "missing": missing, "in_memory": in_memory}

    def __contains__(self, key: tuple[str, str]) -> bool:
        # Mirror the alias-resolution that ``get`` performs.
        name, fmt = key
        return self.resolve_key(name, fmt) is not None

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
        hits: list[tuple[str, str]] = []
        misses: list[tuple[str, str]] = []
        for q in expected_qnames:
            for f in formats:
                if f.upper() == "BF16":
                    continue
                if self.resolve_key(q, f.upper()) is not None:
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


class ProductionWeightCacheVariantView:
    """Read-only cache view that maps runtime formats to rendered variants.

    PrismaClip is not a runtime format. The base assignment still says NVFP4,
    while selected qnames may need the cache entry rendered under the internal
    ``NVFP4_CLIPPED`` key. This view keeps that distinction out of allocator
    and layer-config format space.
    """

    def __init__(
        self,
        base: ProductionWeightCache,
        format_overrides: Mapping[str, str],
    ):
        self.base = base
        self.format_overrides = {
            str(name): str(fmt).strip().upper()
            for name, fmt in dict(format_overrides or {}).items()
            if str(name).strip() and str(fmt).strip()
        }

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def _variant_for(self, name: str, fmt: str) -> str:
        fmt_u = str(fmt).strip().upper()
        if _render_base_format(fmt_u) != "NVFP4":
            return fmt_u
        for cand in self.base._name_candidates(str(name)):
            override = self.format_overrides.get(cand)
            if override:
                return override
        return fmt_u

    def resolve_key(self, name: str, fmt: str) -> tuple[str, str] | None:
        return self.base.resolve_key(name, self._variant_for(name, fmt))

    def get(self, name: str, fmt: str) -> torch.Tensor | None:
        return self.base.get(name, self._variant_for(name, fmt))

    def prefetch(
        self,
        keys: Sequence[tuple[str, str]] | None = None,
        max_workers: int = 4,
    ) -> int:
        if keys is not None:
            keys = [
                (str(name), self._variant_for(str(name), str(fmt)))
                for name, fmt in keys
            ]
        return self.base.prefetch(keys, max_workers=max_workers)

    def verify_files(
        self,
        expected: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, list[tuple[str, str]]]:
        if expected is not None:
            expected = [
                (str(name), self._variant_for(str(name), str(fmt)))
                for name, fmt in expected
            ]
        return self.base.verify_files(expected)

    def assignment_keys(
        self,
        assignment: Mapping[str, str],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        keys, missing = self.base.assignment_keys(assignment)
        if not self.format_overrides:
            return keys, missing
        keys = []
        missing = []
        seen: set[tuple[str, str]] = set()
        for qname, fmt in assignment.items():
            fmt_u = str(fmt).strip().upper()
            if _render_base_format(fmt_u) == "BF16":
                continue
            key = self.resolve_key(str(qname), fmt_u)
            if key is None:
                missing.append((str(qname), self._variant_for(str(qname), fmt_u)))
                continue
            if key not in seen:
                keys.append(key)
                seen.add(key)
        return keys, missing

    def prefetch_assignment(self, *args, **kwargs):
        # Reuse the base implementation against this view's assignment_keys.
        return ProductionWeightCache.prefetch_assignment(self, *args, **kwargs)

    def estimate_nbytes(self, keys: Sequence[tuple[str, str]] | None = None) -> int:
        if keys is not None:
            keys = [
                (str(name), self._variant_for(str(name), str(fmt)))
                for name, fmt in keys
            ]
        return self.base.estimate_nbytes(keys)

    def __contains__(self, key: tuple[str, str]) -> bool:
        name, fmt = key
        return self.resolve_key(name, fmt) is not None

    def __len__(self) -> int:
        return len(self.base)


class _LinearActivationCollector:
    """Hook every quantizable nn.Linear's input on a forward pass.

    Stores up to ``max_rows`` rows of activations per Linear (concatenated
    across calibration samples) on the configured resident device.  Only handles
    ``nn.Linear`` for now — packed MoE experts route through different
    APIs in the export pipeline and would need a separate collector.

    ``store_qnames`` controls which Linears get full activation tensors
    stored (memory-bounded by ``max_rows``).  All Linears in
    ``qnames`` get a per-Linear scalar ``max_abs`` recorded — that's
    cheap (one float per Linear) and needed by the cache's act-clip
    metadata even for Linears whose render is skipped via resume.
    """

    def __init__(
        self,
        model: nn.Module,
        qnames: set[str],
        max_rows: int,
        store_qnames: set[str] | None = None,
        *,
        store_device: torch.device | str | None = None,
        store_dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.qnames = qnames
        self.store_qnames = set(store_qnames) if store_qnames is not None else set(qnames)
        self.max_rows = int(max_rows)
        self.store_device = torch.device(store_device or "cpu")
        self.store_dtype = store_dtype
        self.activations: dict[str, list[torch.Tensor]] = {}
        self.max_abs: dict[str, float] = {}
        self._max_abs_tensors: dict[str, torch.Tensor] = {}
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
            x_abs_max = x.detach().abs().amax()
            prev = self._max_abs_tensors.get(key)
            self._max_abs_tensors[key] = (
                x_abs_max.detach()
                if prev is None
                else torch.maximum(prev, x_abs_max.detach())
            )
            # Only store the full activation tensor if this Linear is in
            # the store set.  Memory bound: store_qnames × max_rows × in.
            if key not in self.store_qnames:
                return
            flat = x.detach().reshape(-1, x.shape[-1]).to(
                device=self.store_device,
                dtype=self.store_dtype,
                non_blocking=True,
            )
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
        self.max_abs = {
            key: float(value.detach().to("cpu").item())
            for key, value in self._max_abs_tensors.items()
        }
        return out


@contextmanager
def _temporarily_install_act_aware(
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, object],
):
    """Install module-level state expected by ``_quantize_2d``.

    The export module reads ``_CACHED_ACTIVATIONS`` and ``_ACT_AWARE_FLAGS``
    from its own globals to decide what passes to run.  We mutate these
    inside a try/finally so concurrent export work isn't disturbed.
    """
    from prismaquant import export_native_compressed as enc

    prev_cache = enc._CACHED_ACTIVATIONS
    prev_flags = dict(enc._ACT_AWARE_FLAGS)
    prev_scale_rule = enc._NVFP4_SCALE_RULE
    enc._CACHED_ACTIVATIONS = _DictActivations(activations)
    enc._ACT_AWARE_FLAGS = {
        "awq": bool(levers.get("awq", False)),
        "gptq": bool(levers.get("gptq", True)),
        "awq_round": bool(levers.get("awq_round", False)),
        "scale_sweep": bool(levers.get("scale_sweep", True)),
    }
    enc._NVFP4_SCALE_RULE = enc.resolve_nvfp4_scale_rule(
        str(levers.get("nvfp4_scale_rule", "static_6"))
    )
    try:
        yield
    finally:
        enc._CACHED_ACTIVATIONS = prev_cache
        enc._ACT_AWARE_FLAGS.clear()
        enc._ACT_AWARE_FLAGS.update(prev_flags)
        enc._NVFP4_SCALE_RULE = prev_scale_rule


class _DictActivations:
    """`.get(name)` shim matching `_LazyActivationCache`'s interface."""

    def __init__(self, mapping: Mapping[str, torch.Tensor]):
        self._mapping = mapping

    def get(self, name: str) -> torch.Tensor | None:
        a = self._mapping.get(name)
        if a is None and name.endswith(".weight"):
            a = self._mapping.get(name[:-7])
        return a


class _FisherRowWeightCache:
    """Lazy loader for h-detail `g2_per_token` vectors."""

    _FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")

    def __init__(self, h_detail_dir: str | Path | None):
        self.detail_dir = Path(h_detail_dir) if h_detail_dir else None
        self._cache: dict[str, torch.Tensor | None] = {}
        self.loads = 0
        self.misses = 0

    def _path_for_name(self, name: str) -> Path | None:
        if self.detail_dir is None:
            return None
        return self.detail_dir / (self._FNAME_SUB.sub("__", name) + ".pt")

    def _load_exact(self, name: str) -> torch.Tensor | None:
        if self.detail_dir is None:
            return None
        path = self._path_for_name(name)
        if path is None:
            return None
        if not path.is_file():
            return None
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            weights = blob.get("g2_per_token") if isinstance(blob, dict) else None
            if not isinstance(weights, torch.Tensor) or weights.numel() == 0:
                weights = None
            else:
                weights = weights.detach().to(torch.float32).cpu()
        except Exception:
            weights = None
        return weights

    @staticmethod
    def _split_fused_names(qname: str) -> tuple[str, ...]:
        if qname.endswith(".qkv_proj"):
            prefix = qname[:-len(".qkv_proj")]
            return (
                f"{prefix}.q_proj",
                f"{prefix}.k_proj",
                f"{prefix}.v_proj",
            )
        if qname.endswith(".gate_up_proj"):
            prefix = qname[:-len(".gate_up_proj")]
            return (f"{prefix}.gate_proj", f"{prefix}.up_proj")
        if qname.endswith(".in_proj_qkvz"):
            prefix = qname[:-len(".in_proj_qkvz")]
            return (f"{prefix}.in_proj_qkv", f"{prefix}.in_proj_z")
        return ()

    @staticmethod
    def _combine_split_weights(parts: Sequence[torch.Tensor]) -> torch.Tensor | None:
        tensors = [
            p.detach().reshape(-1).to(torch.float32).cpu()
            for p in parts
            if isinstance(p, torch.Tensor) and p.numel() > 0
        ]
        if not tensors:
            return None
        n = min(int(t.numel()) for t in tensors)
        if n <= 0:
            return None
        stacked = torch.stack([t[:n] for t in tensors], dim=0)
        return stacked.mean(dim=0)

    def get(self, qname: str) -> torch.Tensor | None:
        if self.detail_dir is None:
            return None
        if qname in self._cache:
            return self._cache[qname]

        weights = self._load_exact(qname)
        if weights is None:
            split = self._split_fused_names(qname)
            if split:
                parts = [
                    part for name in split
                    if (part := self._load_exact(name)) is not None
                ]
                weights = self._combine_split_weights(parts)

        if weights is None:
            self.misses += 1
        else:
            self.loads += 1
        self._cache[qname] = weights
        return weights


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = int(default)
    return max(lo, min(hi, value))


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = float(default)
    return max(lo, min(hi, value))


def _activation_output_error(
    weight: torch.Tensor,
    rendered: torch.Tensor,
    activations: torch.Tensor,
    *,
    row_weights: torch.Tensor | None = None,
    row_chunk: int = 128,
) -> float:
    """Score ``rendered`` by output MSE on original, unclipped activations."""
    rows, cols = weight.shape
    if rendered.shape != weight.shape or activations.shape[-1] != cols:
        return float("inf")
    device = weight.device
    diff_t = (
        weight.detach().to(device=device, dtype=torch.float32)
        - rendered.detach().to(device=device, dtype=torch.float32)
    ).t().contiguous()
    X = activations.detach().to(device=device, dtype=torch.float32).reshape(-1, cols)
    rw = None
    if row_weights is not None and X.numel() > 0:
        rw = _normalize_prismafisherclip_row_weights(
            row_weights,
            X.shape[0],
            device,
        )
    total = torch.zeros((), dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, X.shape[0], row_chunk):
            y = X[start:start + row_chunk] @ diff_t
            err = y.pow(2)
            if rw is not None:
                err = err * rw[start:start + err.shape[0]].unsqueeze(1)
            total = total + err.sum()
    return float(total.item()) / max(1, int(X.shape[0]) * int(rows))


def _normalize_prismafisherclip_row_weights(
    row_weights: torch.Tensor | None,
    n_rows: int,
    device: torch.device,
) -> torch.Tensor | None:
    if row_weights is None or n_rows <= 0:
        return None
    try:
        rw = row_weights.detach().reshape(-1).to(device=device, dtype=torch.float32)
    except Exception:
        return None
    if rw.numel() < n_rows:
        return None
    rw = torch.nan_to_num(rw[:n_rows], nan=0.0, posinf=0.0, neginf=0.0)
    rw = rw.clamp_min(0.0)
    rw = rw / rw.mean().clamp_min(1e-12)
    try:
        clip = float(os.environ.get("PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP", "64"))
    except Exception:
        clip = 64.0
    if clip > 0.0:
        rw = rw.clamp_max(float(clip))
        rw = rw / rw.mean().clamp_min(1e-12)
    return rw


def _prismaclip_activation_split(
    activations: torch.Tensor,
    *,
    split: str,
    holdout_stride: int,
    holdout_offset: int,
) -> torch.Tensor:
    """Return deterministic train/holdout activation rows for PrismaClip scoring."""
    if split == "all" or holdout_stride <= 1:
        return activations
    cols = int(activations.shape[-1])
    X = activations.detach().reshape(-1, cols)
    if X.shape[0] < 2:
        return activations
    row_ids = torch.arange(X.shape[0], device=X.device)
    holdout = (row_ids % holdout_stride) == (holdout_offset % holdout_stride)
    if split == "train":
        mask = ~holdout
    elif split == "holdout":
        mask = holdout
    else:
        raise ValueError(f"unknown PrismaClip activation split: {split}")
    if not bool(mask.any()):
        return activations
    return X[mask]


def _prismaclip_row_weight_split(
    row_weights: torch.Tensor | None,
    activations: torch.Tensor,
    *,
    split: str,
    holdout_stride: int,
    holdout_offset: int,
) -> torch.Tensor | None:
    if row_weights is None or split == "all" or holdout_stride <= 1:
        return row_weights
    cols = int(activations.shape[-1])
    n_rows = int(activations.detach().reshape(-1, cols).shape[0])
    rw = row_weights.detach().reshape(-1)
    if rw.numel() < n_rows or n_rows < 2:
        return row_weights
    row_ids = torch.arange(n_rows, device=rw.device)
    holdout = (row_ids % holdout_stride) == (holdout_offset % holdout_stride)
    if split == "train":
        mask = ~holdout
    elif split == "holdout":
        mask = holdout
    else:
        raise ValueError(f"unknown PrismaClip activation split: {split}")
    if not bool(mask.any()):
        return row_weights
    return rw[:n_rows][mask]


_AWQ_CACHE_FORMATS = frozenset({"NVFP4", "MXFP8", "MXFP8_E4M3", "FP8_E4M3"})
_AWQ_INPUT_LN_LEAVES = frozenset({
    "q_proj", "k_proj", "v_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
})
_AWQ_POST_LN_LEAVES = frozenset({
    "gate_proj", "up_proj", "gate_up_proj", "w1", "w3", "gate", "router",
})


def _awq_group_key_from_qname(qname: str) -> str | None:
    """Return the normalization predecessor group an AWQ scale can fold into."""
    from prismaquant.decision_units import block_id_from_qname

    leaf = qname.rsplit(".", 1)[-1]
    block = block_id_from_qname(qname)
    if leaf in _AWQ_INPUT_LN_LEAVES:
        return f"{block}.input_layernorm"
    if leaf in _AWQ_POST_LN_LEAVES:
        return f"{block}.post_attention_layernorm"
    return None


def _awq_format_group_size(fmt: str) -> int:
    fmt = _render_base_format(fmt)
    if fmt == "NVFP4":
        return 16
    if fmt in {"MXFP8", "MXFP8_E4M3"}:
        return 32
    return 0


def _render_awq_scaled_for_cache(
    *,
    qname: str,
    fmt: str,
    weight_scaled: torch.Tensor,
    activations_scaled: torch.Tensor,
    levers: Mapping[str, object],
    joint_global_real: torch.Tensor | None,
    act_clip_threshold: float | None = None,
    fisher_row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    from prismaquant import export_native_compressed as enc

    fmt = _render_base_format(fmt)
    with _temporarily_install_act_aware(
        {qname: activations_scaled},
        {**dict(levers), "awq": False},
    ):
        result = enc._quantize_2d(
            weight_scaled,
            fmt,
            linear_name=qname,
            nvfp4_global_real_override=joint_global_real,
            cached_activations=activations_scaled,
            awq_enabled=False,
            gptq_enabled=bool(levers.get("gptq", True)),
            awq_round_enabled=bool(levers.get("awq_round", False)),
            scale_sweep_enabled=bool(levers.get("scale_sweep", True)),
            act_clip_threshold=act_clip_threshold,
            fisher_row_weights=fisher_row_weights,
            compute_only=(fmt == "NVFP4"),
        )
    if fmt == "NVFP4":
        return result["_w_dq"].to(torch.float32)
    if fmt in {"MXFP8", "MXFP8_E4M3"}:
        weight = result["weight"]
        scale = result["weight_scale"]
        grouped = weight.reshape(weight.shape[0], weight.shape[1] // 32, 32)
        return enc._mxfp8_dequantize_grouped(grouped, scale).reshape(weight.shape)
    if fmt == "FP8_E4M3":
        return result["weight"].to(torch.float32) * result["weight_scale"].to(torch.float32)
    raise ValueError(f"AWQ cache renderer unsupported for {fmt}")


def _compute_awq_scaled_joint_globals(
    *,
    qname_to_module: Mapping[str, nn.Module],
    render_formats_by_qname: Mapping[str, Sequence[str]],
    awq_scales: Mapping[str, torch.Tensor],
    profile,
) -> dict[str, torch.Tensor]:
    """Compute NVFP4 joint globals after AWQ column scaling."""
    from prismaquant.decision_units import fused_group_key
    from prismaquant.export_native_compressed import compute_nvfp4_global_real

    groups: dict[str, list[str]] = {}
    for qname, fmts in render_formats_by_qname.items():
        if "NVFP4" not in {str(f).upper() for f in fmts}:
            continue
        if qname not in qname_to_module:
            continue
        try:
            gk = fused_group_key(profile, qname) if profile else qname
        except Exception:
            gk = qname
        groups.setdefault(gk, []).append(qname)

    out: dict[str, torch.Tensor] = {}
    for _gk, members in groups.items():
        candidates = []
        for qname in members:
            weight = qname_to_module[qname].weight.detach().to(torch.float32)
            scale = awq_scales.get(qname)
            if scale is not None and scale.numel() == weight.shape[1]:
                weight = weight * scale.to(weight.device, dtype=torch.float32).unsqueeze(0)
            candidates.append(compute_nvfp4_global_real(weight))
        if not candidates:
            continue
        joint = torch.stack(candidates).max()
        for qname in members:
            out[qname] = joint
    return out


def _solve_awq_scales(
    *,
    qname_to_module: Mapping[str, nn.Module],
    activations: Mapping[str, torch.Tensor],
    render_formats_by_qname: Mapping[str, Sequence[str]],
    levers: Mapping[str, object],
    progress: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Solve normalization-fold AWQ scales for a concrete assignment."""
    groups: dict[str, list[str]] = {}
    for qname, fmts in render_formats_by_qname.items():
        if qname not in qname_to_module or qname not in activations:
            continue
        fmt = next((str(f).upper() for f in fmts if str(f).upper() != "BF16"), "")
        if fmt not in _AWQ_CACHE_FORMATS:
            continue
        group_key = _awq_group_key_from_qname(qname)
        if group_key is None:
            continue
        groups.setdefault(group_key, []).append(qname)

    metadata: dict[str, object] = {
        "enabled": True,
        "mode": "duo_output_mse",
        "groups": {},
        "scales": {},
        "n_groups": 0,
        "n_scaled_linears": 0,
    }
    scales: dict[str, torch.Tensor] = {}
    if not groups:
        metadata["status"] = "no_eligible_groups"
        return scales, metadata

    n_grid = _env_int("PRISMAQUANT_AWQ_GRID", 20, lo=2, hi=80)
    clamp_ratio = _env_float("PRISMAQUANT_AWQ_CLAMP", 10.0, lo=1.0, hi=100.0)
    min_gain = _env_float(
        "PRISMAQUANT_AWQ_MIN_GAIN",
        DEFAULT_AWQ_MIN_GAIN,
        lo=-1.0,
        hi=1.0,
    )
    duo = _env_flag("PRISMAQUANT_AWQ_DUO", True)
    search_levers = dict(levers)
    search_levers["gptq"] = _env_flag("PRISMAQUANT_AWQ_SEARCH_GPTQ", False)
    search_levers["scale_sweep"] = _env_flag(
        "PRISMAQUANT_AWQ_SEARCH_SCALE_SWEEP",
        bool(levers.get("scale_sweep", True)),
    )
    search_levers["fisher_gptq"] = _env_flag(
        "PRISMAQUANT_AWQ_SEARCH_FISHER_GPTQ",
        False,
    )
    search_levers["awq_round"] = _env_flag(
        "PRISMAQUANT_AWQ_SEARCH_AWQ_ROUND",
        False,
    )
    metadata["min_gain"] = float(min_gain)
    metadata["n_grid"] = int(n_grid)
    metadata["clamp_ratio"] = float(clamp_ratio)
    metadata["duo_scaling"] = bool(duo)
    metadata["search_levers"] = {
        "gptq": bool(search_levers.get("gptq", False)),
        "scale_sweep": bool(search_levers.get("scale_sweep", True)),
        "fisher_gptq": bool(search_levers.get("fisher_gptq", False)),
        "awq_round": bool(search_levers.get("awq_round", False)),
    }

    for idx, (group_key, raw_members) in enumerate(sorted(groups.items()), start=1):
        members = sorted(set(raw_members))
        targets: list[AwqSearchTarget] = []
        target_qnames: list[str] = []
        for qname in members:
            fmt = next(
                (str(f).upper() for f in render_formats_by_qname[qname]
                 if str(f).upper() != "BF16"),
                "",
            )
            weight = qname_to_module[qname].weight.detach().to(torch.float32)
            acts = activations[qname].detach().to(weight.device, dtype=torch.float32)
            targets.append(AwqSearchTarget(
                name=qname,
                fmt=fmt,
                weight=weight,
                activations=acts,
                group_size=_awq_format_group_size(fmt),
            ))
            target_qnames.append(qname)
        if not targets:
            continue

        def render_target(tidx, w_scaled, a_scaled, scale):
            target = targets[tidx]
            return _render_awq_scaled_for_cache(
                qname=target.name,
                fmt=target.fmt,
                weight_scaled=w_scaled,
                activations_scaled=a_scaled,
                levers=search_levers,
                joint_global_real=None,
            )

        try:
            result = search_awq_scale(
                targets,
                render_target,
                n_grid=n_grid,
                duo_scaling=duo,
                clamp_ratio=clamp_ratio,
                min_gain=min_gain,
            )
        except Exception as exc:
            metadata["groups"][group_key] = {
                "members": target_qnames,
                "status": "failed",
                "error": str(exc),
            }
            if _env_flag("PRISMAQUANT_AWQ_STRICT", False):
                raise
            continue

        selected = result.selected_label != "identity"
        for qname in target_qnames:
            if selected:
                scales[qname] = result.scale.detach().cpu()
                metadata["scales"][qname] = {
                    "group": group_key,
                    "selected_label": result.selected_label,
                    "relative_gain": result.relative_gain,
                }
        metadata["groups"][group_key] = {
            "members": target_qnames,
            "status": "solved",
            "selected": result.selected_label,
            "selected_ratio": result.selected_ratio,
            "baseline_score": result.baseline_score,
            "best_score": result.best_score,
            "relative_gain": result.relative_gain,
            "n_candidates": result.n_candidates,
            "trace": result.trace,
        }
        if progress and (idx % 10 == 0 or idx == len(groups)):
            print(
                f"[prod-cache] AWQ searched {idx}/{len(groups)} groups",
                flush=True,
            )

    metadata["status"] = "applied"
    metadata["n_groups"] = len(groups)
    metadata["n_scaled_linears"] = len(scales)
    return scales, metadata


def _env_float_list(name: str, default: Sequence[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return [float(v) for v in default]
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = float(part)
        except ValueError:
            continue
        if 0.0 <= val <= 1.0:
            out.append(val)
    return out or [float(v) for v in default]


def _smoothquant_scale_from_targets(
    targets: Sequence[AwqSearchTarget],
    *,
    alpha: float,
    clamp_ratio: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    if not targets:
        raise ValueError("SmoothQuant requires at least one target")
    cols = int(targets[0].weight.shape[1])
    device = targets[0].weight.device
    x_max = torch.zeros(cols, device=device, dtype=torch.float32)
    w_max = torch.zeros(cols, device=device, dtype=torch.float32)
    for target in targets:
        if int(target.weight.shape[1]) != cols:
            raise ValueError("all SmoothQuant targets must share input width")
        x = target.activations.detach().to(device=device, dtype=torch.float32)
        x = x.reshape(-1, cols)
        w = target.weight.detach().to(device=device, dtype=torch.float32)
        x_max = torch.maximum(x_max, x.abs().amax(dim=0))
        w_max = torch.maximum(w_max, w.abs().amax(dim=0))
    a = float(alpha)
    raw = x_max.clamp_min(eps).pow(a) / w_max.clamp_min(eps).pow(1.0 - a)
    return normalize_awq_scale(raw, eps=eps, clamp_ratio=clamp_ratio)


def _solve_smoothquant_scales(
    *,
    qname_to_module: Mapping[str, nn.Module],
    activations: Mapping[str, torch.Tensor],
    render_formats_by_qname: Mapping[str, Sequence[str]],
    levers: Mapping[str, object],
    progress: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Solve SmoothQuant-style normalization-fold scales for an assignment."""
    groups: dict[str, list[str]] = {}
    for qname, fmts in render_formats_by_qname.items():
        if qname not in qname_to_module or qname not in activations:
            continue
        fmt = next((str(f).upper() for f in fmts if str(f).upper() != "BF16"), "")
        if fmt not in _AWQ_CACHE_FORMATS:
            continue
        group_key = _awq_group_key_from_qname(qname)
        if group_key is None:
            continue
        groups.setdefault(group_key, []).append(qname)

    metadata: dict[str, object] = {
        "enabled": True,
        "mode": "max_stat_alpha_search",
        "groups": {},
        "scales": {},
        "n_groups": 0,
        "n_scaled_linears": 0,
    }
    scales: dict[str, torch.Tensor] = {}
    if not groups:
        metadata["status"] = "no_eligible_groups"
        return scales, metadata

    alphas = _env_float_list("PRISMAQUANT_SMOOTHQUANT_ALPHAS", [0.5])
    clamp_ratio = _env_float("PRISMAQUANT_SMOOTHQUANT_CLAMP", 10.0, lo=1.0, hi=100.0)
    min_gain = _env_float("PRISMAQUANT_SMOOTHQUANT_MIN_GAIN", 0.0, lo=-1.0, hi=1.0)
    search_levers = dict(levers)
    search_levers["gptq"] = _env_flag("PRISMAQUANT_SMOOTHQUANT_SEARCH_GPTQ", False)
    search_levers["scale_sweep"] = _env_flag(
        "PRISMAQUANT_SMOOTHQUANT_SEARCH_SCALE_SWEEP",
        False,
    )
    search_levers["fisher_gptq"] = _env_flag(
        "PRISMAQUANT_SMOOTHQUANT_SEARCH_FISHER_GPTQ",
        False,
    )
    search_levers["awq_round"] = False
    metadata["alphas"] = [float(a) for a in alphas]
    metadata["clamp_ratio"] = float(clamp_ratio)
    metadata["min_gain"] = float(min_gain)
    metadata["search_levers"] = {
        "gptq": bool(search_levers.get("gptq", False)),
        "scale_sweep": bool(search_levers.get("scale_sweep", False)),
        "fisher_gptq": bool(search_levers.get("fisher_gptq", False)),
        "awq_round": False,
    }

    for idx, (group_key, raw_members) in enumerate(sorted(groups.items()), start=1):
        members = sorted(set(raw_members))
        targets: list[AwqSearchTarget] = []
        target_qnames: list[str] = []
        for qname in members:
            fmt = next(
                (str(f).upper() for f in render_formats_by_qname[qname]
                 if str(f).upper() != "BF16"),
                "",
            )
            weight = qname_to_module[qname].weight.detach().to(torch.float32)
            acts = activations[qname].detach().to(weight.device, dtype=torch.float32)
            targets.append(AwqSearchTarget(
                name=qname,
                fmt=fmt,
                weight=weight,
                activations=acts,
                group_size=_awq_format_group_size(fmt),
            ))
            target_qnames.append(qname)
        if not targets:
            continue

        def score_scale(scale: torch.Tensor) -> float:
            total = 0.0
            scale = scale.to(device=targets[0].weight.device, dtype=torch.float32)
            for tidx, target in enumerate(targets):
                w = target.weight.detach().to(dtype=torch.float32)
                x = target.activations.detach().to(w.device, dtype=torch.float32)
                w_scaled = w * scale.unsqueeze(0)
                a_scaled = x.reshape(-1, x.shape[-1]) / scale.clamp_min(1e-12).unsqueeze(0)
                rendered = _render_awq_scaled_for_cache(
                    qname=target.name,
                    fmt=target.fmt,
                    weight_scaled=w_scaled,
                    activations_scaled=a_scaled,
                    levers=search_levers,
                    joint_global_real=None,
                )
                effective = (
                    rendered.to(w.device, dtype=torch.float32)
                    / scale.clamp_min(1e-12).unsqueeze(0)
                )
                total += awq_output_mse(
                    w,
                    effective,
                    target.activations,
                )
            return float(total)

        try:
            identity = torch.ones(
                int(targets[0].weight.shape[1]),
                device=targets[0].weight.device,
                dtype=torch.float32,
            )
            baseline_score = score_scale(identity)
            best_scale = identity
            best_alpha: float | None = None
            best_score = float(baseline_score)
            trace: list[dict[str, float | str | None]] = [{
                "label": "identity",
                "alpha": None,
                "score": float(baseline_score),
                "best_score": float(best_score),
            }]
            for alpha in alphas:
                cand = _smoothquant_scale_from_targets(
                    targets,
                    alpha=float(alpha),
                    clamp_ratio=clamp_ratio,
                )
                score = score_scale(cand)
                if score < best_score:
                    best_score = float(score)
                    best_scale = cand.detach().clone()
                    best_alpha = float(alpha)
                trace.append({
                    "label": f"alpha_{float(alpha):.4f}",
                    "alpha": float(alpha),
                    "score": float(score),
                    "best_score": float(best_score),
                })
        except Exception as exc:
            metadata["groups"][group_key] = {
                "members": target_qnames,
                "status": "failed",
                "error": str(exc),
            }
            if _env_flag("PRISMAQUANT_SMOOTHQUANT_STRICT", False):
                raise
            continue

        denom = max(abs(float(baseline_score)), 1e-30)
        relative_gain = (float(baseline_score) - float(best_score)) / denom
        selected = best_alpha is not None and relative_gain >= float(min_gain)
        if selected:
            for qname in target_qnames:
                scales[qname] = best_scale.detach().cpu()
                metadata["scales"][qname] = {
                    "group": group_key,
                    "selected_alpha": best_alpha,
                    "relative_gain": relative_gain,
                }
        metadata["groups"][group_key] = {
            "members": target_qnames,
            "status": "solved",
            "selected": "identity" if not selected else f"alpha_{best_alpha:.4f}",
            "selected_alpha": None if not selected else best_alpha,
            "baseline_score": float(baseline_score),
            "best_score": float(best_score if selected else baseline_score),
            "relative_gain": float(relative_gain if selected else 0.0),
            "n_candidates": len(alphas) + 1,
            "trace": trace,
        }
        if progress and (idx % 10 == 0 or idx == len(groups)):
            print(
                f"[prod-cache] SmoothQuant searched {idx}/{len(groups)} groups",
                flush=True,
            )

    metadata["status"] = "applied"
    metadata["n_groups"] = len(groups)
    metadata["n_scaled_linears"] = len(scales)
    return scales, metadata


def _group_activation_bounds(
    members: Sequence[str],
    activations: Mapping[str, torch.Tensor],
) -> tuple[float, float] | None:
    max_abs = 0.0
    sq_sum = 0.0
    count = 0
    for qname in members:
        a = activations.get(qname)
        if a is None or a.numel() == 0:
            return None
        af = a.detach().to(torch.float32)
        max_abs = max(max_abs, float(af.abs().max().item()))
        sq_sum += float(af.pow(2).sum().item())
        count += int(af.numel())
    if count <= 0 or max_abs <= 0.0:
        return None
    rms = math.sqrt(max(sq_sum / count, 1e-30))
    lo = max(max_abs * 1e-4, rms)
    if not math.isfinite(lo) or not math.isfinite(max_abs) or lo >= max_abs:
        return None
    return lo, max_abs


def _store_rendered_weight_entry(
    *,
    weights: dict[tuple[str, str], object],
    cache_dir_path: Path | None,
    qname: str,
    fmt: str,
    tensor: torch.Tensor,
    weight_dtype: torch.dtype,
) -> None:
    fmt = fmt.upper()
    target_dtype = weight_dtype if weight_dtype != torch.float32 else torch.bfloat16
    stored = tensor.to(target_dtype).cpu()
    if cache_dir_path is not None:
        fname = _cache_weight_filename(qname, fmt)
        final_path = cache_dir_path / fname
        tmp_path = cache_dir_path / (fname + ".tmp")
        torch.save(stored, tmp_path)
        os.replace(tmp_path, final_path)
        weights[(qname, fmt)] = fname
        del stored
    else:
        weights[(qname, fmt)] = stored


def _solve_nvfp4_activation_clip_groups(
    *,
    groups: Mapping[str, Sequence[str]],
    qname_to_module: Mapping[str, nn.Module],
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, bool],
    joint_globals: Mapping[str, torch.Tensor],
    activation_max_abs: Mapping[str, float],
    fisher_rows: _FisherRowWeightCache | None,
    weights_out: dict[tuple[str, str], object],
    cache_dir_path: Path | None,
    progress: bool,
    awq_scales: Mapping[str, torch.Tensor] | None = None,
    cache_formats: Sequence[str] = ("NVFP4",),
) -> tuple[dict[str, float | None], dict[str, object]]:
    """Solve one scalar render-time activation clamp per fused NVFP4 group.

    The baseline is the existing render path.  Candidate thresholds are
    optimized in log space, and every candidate is scored against original
    unclipped activations so the solver cannot win by merely hiding outliers
    from its evaluator.
    """
    max_evals = _env_int(
        "PRISMAQUANT_ACT_CLIP_SOLVER_MAX_EVALS",
        6,
        lo=4,
        hi=16,
    )
    min_gain = DEFAULT_ACT_CLIP_SOLVER_MIN_GAIN
    try:
        min_gain = float(
            os.environ.get(
                "PRISMAQUANT_ACT_CLIP_SOLVER_MIN_GAIN",
                str(DEFAULT_ACT_CLIP_SOLVER_MIN_GAIN),
            )
        )
    except Exception:
        min_gain = DEFAULT_ACT_CLIP_SOLVER_MIN_GAIN
    top_fraction = _env_float(
        "PRISMAQUANT_ACT_CLIP_SOLVER_TOP_FRACTION",
        1.0,
        lo=0.0,
        hi=1.0,
    )
    top_k = _env_int(
        "PRISMAQUANT_ACT_CLIP_SOLVER_TOP_K",
        0,
        lo=0,
        hi=1_000_000,
    )
    holdout_enabled = _env_flag(
        "PRISMAQUANT_ACT_CLIP_SOLVER_HOLDOUT",
        False,
    )
    holdout_fraction = _env_float(
        "PRISMAQUANT_ACT_CLIP_SOLVER_HOLDOUT_FRACTION",
        DEFAULT_ACT_CLIP_SOLVER_HOLDOUT_FRACTION,
        lo=0.0,
        hi=0.5,
    )
    holdout_min_gain = _env_float(
        "PRISMAQUANT_ACT_CLIP_SOLVER_HOLDOUT_MIN_GAIN",
        0.0,
        lo=0.0,
        hi=1.0,
    )
    holdout_stride = (
        max(2, int(round(1.0 / max(holdout_fraction, 1e-9))))
        if holdout_enabled and holdout_fraction > 0.0
        else 0
    )
    if holdout_stride <= 1:
        holdout_enabled = False
    fisher_clip_enabled = bool(levers.get("fisher_clip", False))
    fisher_render_enabled = bool(levers.get("fisher_gptq", False))
    fisher_clip_available = bool(fisher_clip_enabled and fisher_rows is not None)
    fisher_clip_mode = str(
        os.environ.get("PRISMAQUANT_PRISMAFISHERCLIP_MODE", "audit")
    ).strip().lower()
    if fisher_clip_mode not in {"audit", "veto", "score"}:
        fisher_clip_mode = "audit"
    if not fisher_clip_available:
        fisher_clip_mode = "off"
    fisher_primary = fisher_clip_available and fisher_clip_mode == "score"
    fisher_veto = fisher_clip_available and fisher_clip_mode == "veto"
    fisher_min_gain = _env_float(
        "PRISMAQUANT_PRISMAFISHERCLIP_MIN_GAIN",
        0.0,
        lo=0.0,
        hi=1.0,
    )
    verbose = _env_flag("PRISMAQUANT_ACT_CLIP_SOLVER_VERBOSE", False)
    output_cache_formats = tuple(
        dict.fromkeys(str(fmt).strip().upper() for fmt in cache_formats if str(fmt).strip())
    ) or ("NVFP4",)
    threshold_by_qname: dict[str, float | None] = {}
    method_name = "PrismaFisherClip" if fisher_clip_available else "PrismaClip"
    objective = (
        "fisher_weighted_output_mse_original_activations"
        if fisher_primary else
        "output_mse_original_activations_fisher_veto"
        if fisher_veto else
        "output_mse_original_activations_fisher_audit"
        if fisher_clip_available else
        "output_mse_original_activations"
    )
    if holdout_enabled:
        objective += "_holdout_veto"
    metadata: dict[str, object] = {
        "enabled": True,
        "method": method_name,
        "format": "NVFP4",
        "cache_formats": list(output_cache_formats),
        "objective": objective,
        "solver": "log_golden_section",
        "max_evals": int(max_evals),
        "min_gain": float(min_gain),
        "top_fraction": float(top_fraction),
        "top_k": int(top_k),
        "fisher_clip_enabled": bool(fisher_clip_enabled),
        "fisher_clip_available": bool(fisher_clip_available),
        "fisher_clip_mode": str(fisher_clip_mode),
        "fisher_clip_min_gain": float(fisher_min_gain),
        "fisher_clip_render_weighted": bool(fisher_render_enabled),
        "holdout_enabled": bool(holdout_enabled),
        "holdout_fraction": float(holdout_fraction if holdout_enabled else 0.0),
        "holdout_stride": int(holdout_stride),
        "holdout_offset": 0,
        "holdout_min_gain": float(holdout_min_gain),
        "groups": {},
        "selected_by_qname": {},
        "prewritten_qnames": [],
        "prewrite_baseline": _env_flag(
            "PRISMAQUANT_ACT_CLIP_SOLVER_PREWRITE_BASELINE",
            False,
        ),
        "prewritten_baseline_qnames": [],
        "batched_same_shape": _env_flag(
            "PRISMAQUANT_ACT_CLIP_SOLVER_BATCHED",
            False,
        ),
        "batched_evaluations": 0,
        "scalar_evaluations": 0,
    }

    def render_member(qname: str, threshold: float | None) -> torch.Tensor:
        mod = qname_to_module[qname]
        max_abs = activation_max_abs.get(qname)
        export_scale = (
            6.0 / max_abs
            if max_abs is not None and max_abs > 0.0
            else None
        )
        return render_production_weight(
            mod.weight.data,
            "NVFP4",
            qname=qname,
            activations=activations,
            levers=levers,
            joint_global_real=joint_globals.get(qname),
            input_global_scale=export_scale,
            act_clip_threshold=threshold,
            fisher_row_weights=(
                fisher_rows.get(qname)
                if fisher_render_enabled and fisher_rows is not None
                else None
            ),
            awq_scale=(awq_scales or {}).get(qname),
        )

    group_entries: list[dict[str, object]] = []

    def render_members_batched(
        members: Sequence[str],
        threshold: float | None,
    ) -> dict[str, torch.Tensor] | None:
        if not metadata["batched_same_shape"] or len(members) < 2:
            return None
        if bool(levers.get("awq_round", False)):
            return None
        if any((awq_scales or {}).get(qname) is not None for qname in members):
            return None
        modules = [qname_to_module[qname] for qname in members]
        shapes = {tuple(mod.weight.shape) for mod in modules}
        if len(shapes) != 1:
            return None
        device = modules[0].weight.device
        dtype = modules[0].weight.dtype
        if any(mod.weight.device != device for mod in modules):
            return None
        cols = int(modules[0].weight.shape[1])
        if any(
            activations.get(qname) is None
            or activations[qname].shape[-1] != cols
            for qname in members
        ):
            return None
        overrides = [joint_globals.get(qname) for qname in members]
        if all(value is not None for value in overrides):
            global_real_overrides = torch.stack([
                value.to(device=device, dtype=torch.float32).reshape(())
                for value in overrides
            ])
        elif all(value is None for value in overrides):
            global_real_overrides = None
        else:
            return None

        try:
            from prismaquant.export_batched_gptq import (
                gptq_obs_rounding_nvfp4_batched,
                scale_sweep_nvfp4_batched,
            )
            from prismaquant.export_native_compressed import (
                _activation_col_importance_for_gptq,
                _activation_matrix_for_gptq,
                _rtn_dequant_nvfp4,
            )
        except Exception:
            return None

        weights = torch.stack([
            mod.weight.detach().to(device=device, dtype=torch.float32)
            for mod in modules
        ], dim=0)
        reference_weights = weights.clone()
        acts_list = [
            activations[qname].detach().to(device=device, dtype=torch.float32)
            for qname in members
        ]
        row_weights_list = [
            (
                fisher_rows.get(qname)
                if fisher_render_enabled and fisher_rows is not None
                else None
            )
            for qname in members
        ]
        row_weights_list = [
            value.to(device=device, dtype=torch.float32)
            if isinstance(value, torch.Tensor) else None
            for value in row_weights_list
        ]

        if bool(levers.get("gptq", True)):
            damp_sweep_on = (
                os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1") != "0"
            )
            if damp_sweep_on:
                damp_candidates = (0.001, 0.005, 0.01, 0.05, 0.1)
                best_w: torch.Tensor | None = None
                best_err: torch.Tensor | None = None
                h_eval = torch.empty(
                    (len(members), cols, cols),
                    device=device,
                    dtype=torch.float32,
                )
                for idx, acts in enumerate(acts_list):
                    x_eval = _activation_matrix_for_gptq(
                        acts,
                        cols,
                        device=device,
                        clip_threshold=threshold,
                        row_weights=row_weights_list[idx],
                    )
                    h_eval[idx] = x_eval.t() @ x_eval
                for damp in damp_candidates:
                    cand_w = gptq_obs_rounding_nvfp4_batched(
                        weights,
                        acts_list,
                        damp=damp,
                        global_real_overrides=global_real_overrides,
                        clip_threshold=threshold,
                        row_weights_list=row_weights_list,
                    )
                    diff = reference_weights - cand_w
                    err = torch.einsum("eoi,eij,eoj->e", diff, h_eval, diff)
                    if best_w is None or best_err is None:
                        best_w = cand_w
                        best_err = err
                    else:
                        take = err < best_err
                        if take.any():
                            idx = take.nonzero(as_tuple=True)[0]
                            best_w[idx] = cand_w[idx]
                            best_err[idx] = err[idx]
                if best_w is not None:
                    weights = best_w
            else:
                weights = gptq_obs_rounding_nvfp4_batched(
                    weights,
                    acts_list,
                    global_real_overrides=global_real_overrides,
                    clip_threshold=threshold,
                    row_weights_list=row_weights_list,
                )

        if bool(levers.get("scale_sweep", True)):
            weights = scale_sweep_nvfp4_batched(
                weights,
                acts_list,
                reference_weights=reference_weights,
                global_real_overrides=global_real_overrides,
                clip_threshold=threshold,
                row_weights_list=row_weights_list,
            )

        if (
            bool(levers.get("gptq", True))
            and os.environ.get("PRISMAQUANT_DO_NO_HARM", "1") != "0"
        ):
            try:
                for idx, qname in enumerate(members):
                    override = overrides[idx]
                    w_rtn = _rtn_dequant_nvfp4(
                        reference_weights[idx],
                        group_size=16,
                        global_real_override=override,
                    )
                    imp = _activation_col_importance_for_gptq(
                        acts_list[idx],
                        cols,
                        device=device,
                        clip_threshold=threshold,
                        row_weights=row_weights_list[idx],
                    )
                    ref = reference_weights[idx]
                    mse_pass = float(
                        (imp * (ref - weights[idx]).pow(2).sum(dim=0)).sum()
                    )
                    mse_rtn = float(
                        (imp * (ref - w_rtn).pow(2).sum(dim=0)).sum()
                    )
                    if mse_rtn < mse_pass:
                        weights[idx] = w_rtn
            except Exception:
                pass

        return {
            qname: weights[idx].to(device=device, dtype=dtype).contiguous()
            for idx, qname in enumerate(members)
        }

    def evaluate_members(
        members: Sequence[str],
        threshold: float | None,
        *,
        keep_weights: bool = False,
        score_splits: Sequence[str] = ("all",),
    ) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
        scores = {split: 0.0 for split in score_splits}
        if fisher_clip_available:
            for split in score_splits:
                scores[f"fisher_{split}"] = 0.0
        kept: dict[str, torch.Tensor] | None = {} if keep_weights else None
        rendered = render_members_batched(members, threshold)
        if rendered is not None:
            metadata["batched_evaluations"] = (
                int(metadata.get("batched_evaluations", 0)) + 1
            )
        else:
            metadata["scalar_evaluations"] = (
                int(metadata.get("scalar_evaluations", 0)) + len(members)
            )
            rendered = {qname: render_member(qname, threshold) for qname in members}
        for qname, w_dq in rendered.items():
            score_row_weights = (
                fisher_rows.get(qname)
                if fisher_clip_available and fisher_rows is not None
                else None
            )
            for split in score_splits:
                split_activations = _prismaclip_activation_split(
                    activations[qname],
                    split=split,
                    holdout_stride=holdout_stride,
                    holdout_offset=0,
                )
                scores[split] += _activation_output_error(
                    qname_to_module[qname].weight.data,
                    w_dq,
                    split_activations,
                )
                if not fisher_clip_available:
                    continue
                split_row_weights = _prismaclip_row_weight_split(
                    score_row_weights,
                    activations[qname],
                    split=split,
                    holdout_stride=holdout_stride,
                    holdout_offset=0,
                )
                scores[f"fisher_{split}"] += _activation_output_error(
                    qname_to_module[qname].weight.data,
                    w_dq,
                    split_activations,
                    row_weights=split_row_weights,
                )
            if kept is not None:
                kept[qname] = w_dq
            else:
                del w_dq
        return scores, kept

    primary_score_key = "fisher_all" if fisher_primary else "all"
    primary_holdout_key = "fisher_holdout" if fisher_primary else "holdout"
    baseline_items = list(sorted(groups.items()))
    for idx, (group_key, raw_members) in enumerate(baseline_items, start=1):
        members = [
            q for q in sorted(set(raw_members))
            if q in qname_to_module and activations.get(q) is not None
        ]
        if not members:
            continue
        bounds = _group_activation_bounds(members, activations)
        if bounds is None:
            continue
        lo, hi = bounds

        baseline_weights: dict[str, torch.Tensor] | None = None
        try:
            score_splits = ("all", "holdout") if holdout_enabled else ("all",)
            baseline_scores, baseline_weights = evaluate_members(
                members,
                None,
                keep_weights=bool(metadata["prewrite_baseline"]),
                score_splits=score_splits,
            )
        except Exception as e:
            metadata["groups"][group_key] = {
                "members": members,
                "status": "baseline_failed",
                "error": str(e),
            }
            continue
        if baseline_weights is not None:
            for qname, tensor in baseline_weights.items():
                for cache_fmt in output_cache_formats:
                    _store_rendered_weight_entry(
                        weights=weights_out,
                        cache_dir_path=cache_dir_path,
                        qname=qname,
                        fmt=cache_fmt,
                        tensor=tensor,
                        weight_dtype=qname_to_module[qname].weight.dtype,
                    )
                metadata["prewritten_baseline_qnames"].append(qname)
                del tensor
            baseline_weights.clear()
        baseline_score = float(baseline_scores[primary_score_key])
        baseline_fisher_score = (
            float(baseline_scores["fisher_all"])
            if fisher_clip_available else None
        )
        baseline_unweighted_score = float(baseline_scores["all"])
        baseline_holdout_score = (
            float(baseline_scores[primary_holdout_key]) if holdout_enabled else None
        )
        baseline_fisher_holdout_score = (
            float(baseline_scores["fisher_holdout"])
            if fisher_clip_available and holdout_enabled else None
        )
        group_entries.append({
            "group_key": group_key,
            "members": members,
            "lo": float(lo),
            "hi": float(hi),
            "baseline_score": float(baseline_score),
            "baseline_unweighted_score": float(baseline_unweighted_score),
            "baseline_fisher_score": baseline_fisher_score,
            "baseline_holdout_score": baseline_holdout_score,
            "baseline_fisher_holdout_score": baseline_fisher_holdout_score,
        })
        if progress and (idx % 10 == 0 or idx == len(baseline_items)):
            print(
                f"[prod-cache] PrismaClip baseline "
                f"{idx}/{len(baseline_items)} groups",
                flush=True,
            )

    ranked = sorted(
        group_entries,
        key=lambda item: float(item["baseline_score"]),
        reverse=True,
    )
    keep = len(ranked)
    if top_fraction < 1.0:
        keep = min(keep, max(1, math.ceil(len(ranked) * top_fraction)))
    if top_k > 0:
        keep = min(keep, top_k)
    eligible_groups = {
        str(item["group_key"]) for item in ranked[:keep]
    }
    metadata["eligible_groups"] = int(len(eligible_groups))
    metadata["total_groups"] = int(len(ranked))
    if progress:
        print(
            f"[prod-cache] PrismaClip threshold search: "
            f"{len(eligible_groups)}/{len(ranked)} groups "
            f"(top_fraction={top_fraction:.3g}, top_k={top_k})",
            flush=True,
        )

    eligible_done = 0
    for item in group_entries:
        group_key = str(item["group_key"])
        members = list(item["members"])
        lo = float(item["lo"])
        hi = float(item["hi"])
        baseline_score = float(item["baseline_score"])
        baseline_unweighted_score = float(
            item.get("baseline_unweighted_score", baseline_score)
        )
        baseline_fisher_score = (
            float(item["baseline_fisher_score"])
            if item.get("baseline_fisher_score") is not None else None
        )
        baseline_holdout_score = (
            float(item["baseline_holdout_score"])
            if item.get("baseline_holdout_score") is not None
            else None
        )
        baseline_fisher_holdout_score = (
            float(item["baseline_fisher_holdout_score"])
            if item.get("baseline_fisher_holdout_score") is not None else None
        )
        if group_key not in eligible_groups:
            for qname in members:
                threshold_by_qname[qname] = None
                metadata["selected_by_qname"][qname] = None
            metadata["groups"][group_key] = {
                "members": members,
                "status": "skipped_low_error",
                "selected": "baseline",
                "selected_threshold": None,
                "baseline_score": float(baseline_score),
                "baseline_unweighted_score": float(baseline_unweighted_score),
                "baseline_fisher_score": baseline_fisher_score,
                "baseline_holdout_score": baseline_holdout_score,
                "baseline_fisher_holdout_score": baseline_fisher_holdout_score,
                "best_score": float(baseline_score),
                "best_fisher_score": baseline_fisher_score,
                "holdout_score": baseline_holdout_score,
                "holdout_relative_gain": 0.0,
                "holdout_accepted": None,
                "fisher_relative_gain": 0.0,
                "fisher_accepted": None,
                "relative_gain": 0.0,
                "lo": float(lo),
                "hi": float(hi),
                "n_evals": 0,
            }
            continue

        log_lo = math.log(lo)
        log_hi = math.log(hi)
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        cache: dict[float, float] = {}
        eval_trace: list[dict[str, float | int]] = []
        best_candidate_threshold: float | None = None
        best_candidate_score = float("inf")
        best_candidate_unweighted_score: float | None = None
        best_candidate_fisher_score: float | None = None
        best_candidate_holdout_score: float | None = None
        best_candidate_weights: dict[str, torch.Tensor] | None = None
        best_candidate_eval_index: int | None = None
        denom = max(abs(float(baseline_score)), 1e-30)
        fisher_denom = (
            max(abs(float(baseline_fisher_score)), 1e-30)
            if baseline_fisher_score is not None else 1e-30
        )
        holdout_denom = max(
            abs(float(baseline_holdout_score)),
            1e-30,
        ) if baseline_holdout_score is not None else 1e-30

        def eval_log(log_threshold: float) -> float:
            nonlocal best_candidate_threshold, best_candidate_score
            nonlocal best_candidate_unweighted_score
            nonlocal best_candidate_fisher_score
            nonlocal best_candidate_holdout_score
            nonlocal best_candidate_weights
            nonlocal best_candidate_eval_index
            threshold = float(math.exp(log_threshold))
            key = round(threshold, 12)
            if key not in cache:
                score_splits = ("all", "holdout") if holdout_enabled else ("all",)
                scores, cand_weights = evaluate_members(
                    members,
                    threshold,
                    keep_weights=True,
                    score_splits=score_splits,
                )
                score = float(scores[primary_score_key])
                unweighted_score = float(scores["all"])
                fisher_score = (
                    float(scores["fisher_all"]) if fisher_clip_available else None
                )
                holdout_score = (
                    float(scores[primary_holdout_key]) if holdout_enabled else None
                )
                cache[key] = score
                if score < best_candidate_score:
                    if best_candidate_weights is not None:
                        for old in best_candidate_weights.values():
                            del old
                    best_candidate_eval_index = len(eval_trace) + 1
                    best_candidate_threshold = float(key)
                    best_candidate_score = float(score)
                    best_candidate_unweighted_score = float(unweighted_score)
                    best_candidate_fisher_score = fisher_score
                    best_candidate_holdout_score = holdout_score
                    best_candidate_weights = cand_weights
                elif cand_weights is not None:
                    for old in cand_weights.values():
                        del old
                best_so_far = min(float(baseline_score), float(best_candidate_score))
                trace_entry: dict[str, float | int] = {
                    "eval": int(len(eval_trace) + 1),
                    "threshold": float(key),
                    "score": float(score),
                    "unweighted_score": float(unweighted_score),
                    "best_score": float(best_so_far),
                    "relative_gain": float(
                        (float(baseline_score) - best_so_far) / denom
                    ),
                }
                if fisher_score is not None and baseline_fisher_score is not None:
                    trace_entry["fisher_score"] = float(fisher_score)
                    trace_entry["fisher_relative_gain"] = float(
                        (float(baseline_fisher_score) - float(fisher_score))
                        / fisher_denom
                    )
                if holdout_enabled and holdout_score is not None:
                    trace_entry["holdout_score"] = float(holdout_score)
                    trace_entry["holdout_relative_gain"] = float(
                        (float(baseline_holdout_score) - holdout_score)
                        / holdout_denom
                    )
                eval_trace.append(trace_entry)
            return cache[key]

        try:
            eval_log(log_lo)
            eval_log(log_hi)
            a = log_lo
            b = log_hi
            c = b - inv_phi * (b - a)
            d = a + inv_phi * (b - a)
            fc = eval_log(c)
            fd = eval_log(d)
            iterations = 0
            while len(cache) < max_evals and iterations < max_evals * 4:
                iterations += 1
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - inv_phi * (b - a)
                    fc = eval_log(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + inv_phi * (b - a)
                    fd = eval_log(d)
        except Exception as e:
            if best_candidate_weights is not None:
                for old in best_candidate_weights.values():
                    del old
                best_candidate_weights = None
            metadata["groups"][group_key] = {
                "members": members,
                "status": "solver_failed",
                "baseline_score": float(baseline_score),
                "error": str(e),
            }
            for qname in members:
                threshold_by_qname[qname] = None
                metadata["selected_by_qname"][qname] = None
            continue

        best_threshold = best_candidate_threshold
        best_score = min(float(baseline_score), float(best_candidate_score))
        rel_gain = (float(baseline_score) - float(best_score)) / denom
        fisher_rel_gain = (
            (float(baseline_fisher_score) - float(best_candidate_fisher_score))
            / fisher_denom
            if baseline_fisher_score is not None
            and best_candidate_fisher_score is not None
            else None
        )
        fisher_accepted = (
            None if not fisher_veto else bool(
                baseline_fisher_score is not None
                and best_candidate_fisher_score is not None
                and float(best_candidate_fisher_score) < float(baseline_fisher_score)
                and float(fisher_rel_gain) >= fisher_min_gain
            )
        )
        holdout_score = best_candidate_holdout_score
        holdout_rel_gain = (
            (float(baseline_holdout_score) - float(holdout_score))
            / holdout_denom
            if holdout_enabled
            and baseline_holdout_score is not None
            and holdout_score is not None
            else None
        )
        holdout_accepted = (
            None if not holdout_enabled else bool(
                holdout_score is not None
                and baseline_holdout_score is not None
                and float(holdout_score) < float(baseline_holdout_score)
                and float(holdout_rel_gain) >= holdout_min_gain
            )
        )
        selected = (
            best_threshold is not None
            and float(best_candidate_score) < float(baseline_score)
            and rel_gain >= min_gain
            and (not fisher_veto or bool(fisher_accepted))
            and (not holdout_enabled or bool(holdout_accepted))
        )
        if not selected:
            best_threshold = None
            best_score = float(baseline_score)
            rel_gain = 0.0
            fisher_rel_gain = 0.0
            if best_candidate_weights is not None:
                for old in best_candidate_weights.values():
                    del old
                best_candidate_weights = None
        for qname in members:
            threshold_by_qname[qname] = best_threshold
            metadata["selected_by_qname"][qname] = best_threshold
        if selected and best_candidate_weights is not None:
            for qname, tensor in best_candidate_weights.items():
                for cache_fmt in output_cache_formats:
                    _store_rendered_weight_entry(
                        weights=weights_out,
                        cache_dir_path=cache_dir_path,
                        qname=qname,
                        fmt=cache_fmt,
                        tensor=tensor,
                        weight_dtype=qname_to_module[qname].weight.dtype,
                    )
                metadata["prewritten_qnames"].append(qname)
                del tensor
            best_candidate_weights.clear()
            best_candidate_weights = None
        metadata["groups"][group_key] = {
            "members": members,
            "status": "solved",
            "selected": "solved" if selected else "baseline",
            "selected_threshold": best_threshold,
            "baseline_score": float(baseline_score),
            "baseline_unweighted_score": float(baseline_unweighted_score),
            "baseline_fisher_score": baseline_fisher_score,
            "baseline_holdout_score": baseline_holdout_score,
            "baseline_fisher_holdout_score": baseline_fisher_holdout_score,
            "best_score": float(best_score),
            "best_unweighted_score": (
                float(best_candidate_unweighted_score)
                if best_candidate_unweighted_score is not None and selected
                else float(baseline_unweighted_score)
            ),
            "best_fisher_score": (
                float(best_candidate_fisher_score)
                if best_candidate_fisher_score is not None and selected
                else baseline_fisher_score
            ),
            "relative_gain": float(rel_gain),
            "fisher_relative_gain": (
                float(fisher_rel_gain) if fisher_rel_gain is not None else 0.0
            ),
            "fisher_accepted": fisher_accepted,
            "holdout_score": (
                float(holdout_score)
                if holdout_score is not None and selected else
                baseline_holdout_score
            ),
            "holdout_relative_gain": (
                float(holdout_rel_gain)
                if holdout_rel_gain is not None and selected else
                0.0
            ),
            "holdout_accepted": holdout_accepted,
            "lo": float(lo),
            "hi": float(hi),
            "n_evals": int(len(cache)),
            "selected_eval_index": (
                int(best_candidate_eval_index)
                if selected and best_candidate_eval_index is not None
                else None
            ),
            "eval_trace": eval_trace,
        }
        if progress and verbose:
            print(
                f"[prod-cache] PrismaClip {group_key}: "
                f"{'solved' if selected else 'baseline'} "
                f"gain={rel_gain:.4%} threshold={best_threshold}",
                flush=True,
            )
        eligible_done += 1
        if progress and (
            eligible_done % 5 == 0 or eligible_done == len(eligible_groups)
        ):
            print(
                f"[prod-cache] PrismaClip searched "
                f"{eligible_done}/{len(eligible_groups)} eligible groups",
                flush=True,
            )

    return threshold_by_qname, metadata


def render_production_weight(
    weight: torch.Tensor,
    fmt: str,
    *,
    qname: str,
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, object],
    joint_global_real: torch.Tensor | None = None,
    input_global_scale: float | None = None,
    act_clip_threshold: float | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    awq_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the production-faithful dequantized weight for ``(qname, fmt)``.

    Returns a tensor matching ``weight.shape`` and dtype.  For NVFP4 this
    runs GPTQ + scale_sweep (the activation-aware passes) with the joint
    fused-sibling NVFP4 global if supplied; for BF16 and RTN-only formats
    it falls back to the registry quantize_dequantize because those formats
    don't benefit from activation-aware refinement in the production pipeline.

    ``joint_global_real`` is the max-across-fused-siblings NVFP4 global
    used to keep q/k/v (or gate/up) per-tensor scales unified — same as
    the export's ``_compute_nvfp4_joint_global``.  When ``None`` the
    per-Linear computed value is used (legacy behavior, only correct for
    isolated Linears with no fused siblings).

    ``act_clip_threshold`` is an optional scalar clamp for activation-aware
    render passes. ``fisher_row_weights`` optionally weights local objectives
    by per-token gradient² from h-detail. ``awq_scale`` stores weights in the
    cache's measurement coordinates: Q(W*s)/s, while export later multiplies by
    ``s`` and folds the predecessor norm to emit Q(W*s).
    """
    fmt = fmt.upper()
    if awq_scale is not None:
        scale = awq_scale.to(device=weight.device, dtype=torch.float32)
        if scale.numel() != weight.shape[1]:
            raise ValueError(
                f"AWQ scale width {scale.numel()} != {qname} weight.in {weight.shape[1]}"
            )
        acts = activations.get(qname)
        if acts is None:
            raise ValueError(f"AWQ render for {qname} requires activations")
        scaled = _render_awq_scaled_for_cache(
            qname=qname,
            fmt=fmt,
            weight_scaled=weight.detach().to(torch.float32) * scale.unsqueeze(0),
            activations_scaled=(
                acts.detach().to(device=weight.device, dtype=torch.float32)
                .reshape(-1, acts.shape[-1])
                / scale.clamp_min(1e-12).unsqueeze(0)
            ),
            levers=levers,
            joint_global_real=joint_global_real,
            act_clip_threshold=act_clip_threshold,
            fisher_row_weights=fisher_row_weights,
        )
        effective = scaled / scale.clamp_min(1e-12).unsqueeze(0)
        return effective.to(device=weight.device, dtype=weight.dtype).contiguous()

    if fmt != "NVFP4":
        if (
            fmt in {"MXFP8", "MXFP8_E4M3"}
            and bool(levers.get("scale_sweep", True))
            and qname in activations
        ):
            from prismaquant.export_native_compressed import (
                _mxfp8_scale_sweep_quantize,
            )

            _, _, w_dq = _mxfp8_scale_sweep_quantize(
                weight.detach().to(torch.float32),
                activations[qname],
                group_size=32,
                clip_threshold=act_clip_threshold,
                fisher_row_weights=fisher_row_weights,
            )
            return w_dq.to(device=weight.device, dtype=weight.dtype).contiguous()
        from prismaquant import format_registry as fr
        spec = fr.get_format(fmt)
        return spec.quantize_dequantize(weight.detach().clone()).to(
            device=weight.device, dtype=weight.dtype,
        )

    from prismaquant.export_native_compressed import _quantize_2d

    with _temporarily_install_act_aware(activations, levers):
        result = _quantize_2d(
            weight.detach().clone(),
            fmt="NVFP4",
            linear_name=qname,
            nvfp4_global_real_override=joint_global_real,
            input_global_scale_override=input_global_scale,
            act_clip_threshold=act_clip_threshold,
            fisher_row_weights=fisher_row_weights,
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
    render_assignment: Mapping[str, str] | None = None,
    levers: Mapping[str, bool] | None = None,
    max_act_rows: int = 256,
    progress: bool = True,
    cache_dir: str | Path | None = None,
    recache_pass: bool = False,
    recache_assignment: Mapping[str, str] | None = None,
    recache_profile=None,
    recache_include_activation_quant: bool = True,
    recache_microbatch_size: int = 1,
    h_detail_dir: str | Path | None = None,
) -> ProductionWeightCache:
    """End-to-end fill: collect activations, render production δw per
    (qname, fmt), return a `ProductionWeightCache`.

    Args:
      model: live HF model on the export device.
      calib_ids: ``[N, T]`` token id tensor for activation collection.
      qnames: which Linears are eligible to render (skips MoE packed
        experts; handle those separately via `_quantize_3d_packed`
        extensions).
      formats: which formats to pre-render when `render_assignment` is not
        supplied.
      render_assignment: optional concrete export assignment. When supplied,
        render exactly the non-BF16 `(qname, fmt)` entries used by that
        assignment instead of the full `qnames x formats` menu.
      levers: which production levers to enable (default: gptq+scale_sweep).
      recache_pass: when True, run a second calibration forward with the
        concrete production assignment installed from this cache and refit
        ``activation_max_abs`` under quantized upstream weights.
      recache_assignment: required when ``recache_pass`` is True.  Candidate
        caches with multiple possible formats per Linear are ambiguous; recache
        needs the actual export assignment.
      h_detail_dir: optional probe h-detail directory. When `fisher_gptq` is
        enabled, per-token `g2_per_token` vectors from this directory weight
        NVFP4 GPTQ/scale-sweep and MXFP8 scale-sweep objectives. When
        `fisher_clip` is enabled, the same vectors weight PrismaClip candidate
        scoring without changing the render objective unless `fisher_gptq` is
        also enabled.
    """
    if recache_pass and not recache_assignment:
        raise ValueError(
            "recache_pass=True requires recache_assignment with the concrete "
            "production assignment"
        )
    levers = dict(levers) if levers is not None else {}
    levers.setdefault("gptq", True)
    levers.setdefault(
        "gptq_damp_sweep",
        bool(levers.get("gptq", True))
        and os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1") != "0",
    )
    levers.setdefault("scale_sweep", True)
    levers.setdefault("awq", False)
    levers.setdefault("awq_round", False)
    levers.setdefault("smoothquant", False)
    levers.setdefault(
        "act_clip_solver",
        _env_flag("PRISMAQUANT_ACT_CLIP_SOLVER", False),
    )
    levers.setdefault(
        "fisher_gptq",
        _env_flag("PRISMAQUANT_FISHER_WEIGHTED_GPTQ", False),
    )
    levers.setdefault(
        "fisher_clip",
        _env_flag("PRISMAQUANT_PRISMAFISHERCLIP", False)
        or _env_flag("PRISMAQUANT_ACT_CLIP_SOLVER_FISHER", False),
    )
    if bool(levers.get("fisher_clip", False)):
        levers["act_clip_solver"] = True
        if not h_detail_dir:
            raise ValueError(
                "fisher_clip/PrismaFisherClip requires h_detail_dir with "
                "per-token g2_per_token weights"
            )
    from prismaquant.export_native_compressed import resolve_nvfp4_scale_rule
    levers.setdefault("nvfp4_scale_rule", resolve_nvfp4_scale_rule())
    if bool(levers.get("awq", False)) and bool(levers.get("smoothquant", False)):
        raise ValueError("AWQ and SmoothQuant are mutually exclusive fold-scale levers")

    from prismaquant import format_registry as fr

    def _canon(fmt: str) -> str:
        fmt_u = str(fmt).strip().upper()
        if fmt_u == PRISMACLIP_FORMAT:
            return PRISMACLIP_FORMAT
        return fr.canonical_format_name(fmt_u)

    requested_formats = tuple(
        dict.fromkeys(_canon(f) for f in formats if str(f).strip())
    )
    if PRISMACLIP_FORMAT in requested_formats:
        levers["act_clip_candidates"] = True
    eligible_qnames = set(qnames)
    if render_assignment is not None:
        render_formats_by_qname: dict[str, tuple[str, ...]] = {}
        for qname, fmt in render_assignment.items():
            q = str(qname)
            if q not in eligible_qnames:
                continue
            fmt_canon = _canon(fmt)
            if fmt_canon == "BF16":
                continue
            render_formats_by_qname[q] = (fmt_canon,)
        qname_set = set(render_formats_by_qname)
        render_scope = "assignment"
    else:
        non_bf16_formats = tuple(
            f for f in requested_formats if f != "BF16"
        )
        render_formats_by_qname = {
            q: non_bf16_formats for q in eligible_qnames
        }
        qname_set = {
            q for q, fmts in render_formats_by_qname.items() if fmts
        }
        render_scope = "format-menu"

    if not qname_set:
        return ProductionWeightCache(
            weights={},
            levers=dict(levers),
            metadata={
                "render_scope": render_scope,
                "requested_formats": list(requested_formats),
                "requested_entries": 0,
            },
        )

    if progress:
        requested_entries = sum(
            len(fmts) for fmts in render_formats_by_qname.values()
        )
        print(f"[prod-cache] levers={dict(sorted(levers.items()))}", flush=True)
        print(
            f"[prod-cache] render_scope={render_scope} "
            f"qnames={len(qname_set)} entries={requested_entries}",
            flush=True,
        )

    # RESUME: when disk-streaming is on and prior shards exist, only
    # collect activations for Linears whose shards we still need to
    # render.  On a job that's 99%+ complete this drops activation
    # collection memory + compute by 99% — and lets a borderline-OOM
    # job finish on the same hardware.
    cache_dir_path: Path | None = None
    if cache_dir is not None:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)

    fmt_set = {
        fmt
        for fmts in render_formats_by_qname.values()
        for fmt in fmts
    }
    render_base_fmt_set = {_render_base_format(fmt) for fmt in fmt_set}
    activation_aware_formats = {"NVFP4", PRISMACLIP_FORMAT}
    if bool(levers.get("scale_sweep", True)):
        activation_aware_formats.update({"MXFP8", "MXFP8_E4M3"})
    if bool(levers.get("awq", False)):
        activation_aware_formats.update(_AWQ_CACHE_FORMATS)
    if bool(levers.get("smoothquant", False)):
        activation_aware_formats.update(_AWQ_CACHE_FORMATS)
    qnames_to_render: set[str] = set(qname_set)
    missing_formats_by_qname: dict[str, set[str]] = {
        q: set(render_formats_by_qname.get(q, ())) for q in qname_set
    }
    if cache_dir_path is not None:
        # A qname is FULLY done if every requested format has a shard.
        prerendered = 0
        for q in list(qname_set):
            missing = {
                f for f in render_formats_by_qname.get(q, ())
                if not (cache_dir_path / _cache_weight_filename(q, f)).is_file()
            }
            missing_formats_by_qname[q] = missing
            if not missing:
                qnames_to_render.discard(q)
                prerendered += 1
        if progress and prerendered:
            print(
                f"[prod-cache] resume: {prerendered} qnames already on disk "
                f"({len(qnames_to_render)} still need rendering)",
                flush=True,
            )
    qnames_needing_activation = {
        q for q, missing in missing_formats_by_qname.items()
        if any(f in activation_aware_formats for f in missing)
    }
    if bool(levers.get("awq", False)):
        qnames_needing_activation.update(
            q for q, fmts in render_formats_by_qname.items()
            if any(str(f).upper() in _AWQ_CACHE_FORMATS for f in fmts)
        )
    if bool(levers.get("smoothquant", False)):
        qnames_needing_activation.update(
            q for q, fmts in render_formats_by_qname.items()
            if any(str(f).upper() in _AWQ_CACHE_FORMATS for f in fmts)
        )

    device = next(model.parameters()).device
    awq_enabled = bool(levers.get("awq", False))
    activation_store_device = (
        device if device.type == "cuda" else torch.device("cpu")
    )
    activation_store_dtype = torch.float32
    if progress and qnames_needing_activation:
        print(
            f"[prod-cache] activation_capture "
            f"store_device={activation_store_device} "
            f"store_dtype={activation_store_dtype} "
            f"qnames={len(qnames_needing_activation)}",
            flush=True,
        )
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
        and not qnames_needing_activation
        and not bool(levers.get("awq", False))
        and (
            (sidecar_path is not None and sidecar_path.is_file())
            or "NVFP4" not in render_base_fmt_set
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
            store_qnames=qnames_needing_activation,
            store_device=activation_store_device,
            store_dtype=activation_store_dtype,
        )
        collector.install()
        try:
            with torch.no_grad():
                for i in range(calib_ids.size(0)):
                    batch = calib_ids[i:i + 1].to(device)
                    try:
                        model(batch, use_cache=False)
                    except TypeError:
                        # Some non-HF or older model wrappers do not expose
                        # use_cache. The cache is only an inference speed
                        # feature; activation collection is still correct
                        # without the explicit flag on those models.
                        model(batch)
        finally:
            collector.remove()
        activations = collector.collected()

    if device.type == "cuda" and qnames_needing_activation:
        cpu_activations = [
            name for name, acts in activations.items()
            if acts.device.type != "cuda"
        ]
        if cpu_activations:
            raise RuntimeError(
                "production cache captured non-CUDA activations for "
                f"{len(cpu_activations)} Linears; sample={cpu_activations[:3]}"
            )

    if progress:
        activation_bytes = sum(
            int(t.numel()) * int(t.element_size())
            for t in activations.values()
        )
        activation_devices = sorted({str(t.device) for t in activations.values()})
        print(
            f"[prod-cache] collected activations for "
            f"{len(activations)}/{len(qname_set)} Linears "
            f"resident_bytes={activation_bytes:,} "
            f"devices={activation_devices}",
            flush=True,
        )

    weights: dict[tuple[str, str], object] = {}
    failed: dict[tuple[str, str], str] = {}
    qname_to_module: dict[str, nn.Module] = {}

    if cache_dir_path is not None and progress:
        print(f"[prod-cache] streaming cache to {cache_dir_path}/", flush=True)

    for full_name, mod, attr in iter_quantizable_tensors(model):
        if attr != "weight" or not isinstance(mod, nn.Linear):
            continue
        qname = full_name[:-7] if full_name.endswith(".weight") else full_name
        if qname in qname_set:
            qname_to_module[qname] = mod

    fisher_rows = (
        _FisherRowWeightCache(h_detail_dir)
        if (
            (bool(levers.get("fisher_gptq", False))
             or bool(levers.get("fisher_clip", False)))
            and h_detail_dir
        )
        else None
    )
    if progress and (
        bool(levers.get("fisher_gptq", False))
        or bool(levers.get("fisher_clip", False))
    ):
        if fisher_rows is None:
            print(
                "[prod-cache] Fisher weighting requested but no h_detail_dir "
                "was provided; falling back to unweighted objectives",
                flush=True,
            )
        else:
            print(
                f"[prod-cache] Fisher weighting using h-detail dir "
                f"{fisher_rows.detail_dir}",
                flush=True,
            )

    # HIGH-1: compute joint NVFP4 fused-sibling globals so q/k/v share a
    # per-tensor scale (and gate/up likewise), matching the export's
    # `_compute_nvfp4_joint_global` behavior.  Without this each sibling
    # gets its own scale and vLLM's loader either rejects the artifact or
    # silently runs with degraded accuracy.
    joint_globals: dict[str, torch.Tensor] = {}
    profile = None
    needs_nvfp4_render = any(
        any(_render_base_format(fmt) == "NVFP4" for fmt in missing)
        for missing in missing_formats_by_qname.values()
    )
    if needs_nvfp4_render:
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

    if "NVFP4" in render_base_fmt_set:
        # Group by fused sibling key for max-across-siblings unification.
        from prismaquant.decision_units import fused_group_key
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

    awq_scales: dict[str, torch.Tensor] = {}
    awq_metadata: dict[str, object] = {
        "enabled": bool(levers.get("awq", False)),
        "status": "disabled",
    }
    smoothquant_metadata: dict[str, object] = {
        "enabled": bool(levers.get("smoothquant", False)),
        "status": "disabled",
    }
    if bool(levers.get("awq", False)):
        if render_assignment is None:
            raise ValueError(
                "production AWQ requires render_assignment/assignment scope; "
                "format-menu AWQ would need multiple incompatible fold scales "
                "for the same predecessor"
            )
        if progress:
            print("[prod-cache] solving AWQ scales", flush=True)
        awq_scales, awq_metadata = _solve_awq_scales(
            qname_to_module=qname_to_module,
            activations=activations,
            render_formats_by_qname=render_formats_by_qname,
            levers=levers,
            progress=progress,
        )
        if needs_nvfp4_render and awq_scales:
            joint_globals = _compute_awq_scaled_joint_globals(
                qname_to_module=qname_to_module,
                render_formats_by_qname=render_formats_by_qname,
                awq_scales=awq_scales,
                profile=profile,
            )
            if progress:
                print(
                    f"[prod-cache] recomputed joint NVFP4 globals after "
                    f"AWQ for {len(joint_globals)} fused-sibling members",
                    flush=True,
                )

    if bool(levers.get("smoothquant", False)):
        if render_assignment is None:
            raise ValueError(
                "production SmoothQuant requires render_assignment/assignment "
                "scope because fold scales are tied to the concrete format "
                "assignment"
            )
        if progress:
            print("[prod-cache] solving SmoothQuant scales", flush=True)
        awq_scales, smoothquant_metadata = _solve_smoothquant_scales(
            qname_to_module=qname_to_module,
            activations=activations,
            render_formats_by_qname=render_formats_by_qname,
            levers=levers,
            progress=progress,
        )
        if needs_nvfp4_render and awq_scales:
            joint_globals = _compute_awq_scaled_joint_globals(
                qname_to_module=qname_to_module,
                render_formats_by_qname=render_formats_by_qname,
                awq_scales=awq_scales,
                profile=profile,
            )
            if progress:
                print(
                    f"[prod-cache] recomputed joint NVFP4 globals after "
                    f"SmoothQuant for {len(joint_globals)} fused-sibling "
                    "members",
                    flush=True,
                )

    solved_nvfp4_thresholds: dict[str, float | None] = {}
    solve_global_nvfp4_clip = bool(levers.get("act_clip_solver", False))
    solve_variant_nvfp4_clip = any(
        PRISMACLIP_FORMAT in missing
        for missing in missing_formats_by_qname.values()
    )
    clip_solver_metadata: dict[str, object] = {
        "enabled": bool(solve_global_nvfp4_clip or solve_variant_nvfp4_clip),
        "method": "PrismaClip",
        "format": "NVFP4",
        "candidate_format": PRISMACLIP_FORMAT if solve_variant_nvfp4_clip else None,
        "status": "disabled",
    }
    if (solve_global_nvfp4_clip or solve_variant_nvfp4_clip) and needs_nvfp4_render:
        from prismaquant.decision_units import fused_group_key

        solver_groups: dict[str, list[str]] = {}
        for qname, missing in missing_formats_by_qname.items():
            if not (
                (solve_global_nvfp4_clip and "NVFP4" in missing)
                or PRISMACLIP_FORMAT in missing
            ):
                continue
            if qname not in qname_to_module or qname not in activations:
                continue
            try:
                group_key = fused_group_key(profile, qname) if profile else qname
            except Exception:
                group_key = qname
            solver_groups.setdefault(group_key, []).append(qname)
        if solver_groups:
            if progress:
                print(
                    f"[prod-cache] solving PrismaClip NVFP4 thresholds for "
                    f"{len(solver_groups)} fused groups",
                    flush=True,
                )
            clip_cache_formats: list[str] = []
            if solve_global_nvfp4_clip:
                clip_cache_formats.append("NVFP4")
            if solve_variant_nvfp4_clip:
                clip_cache_formats.append(PRISMACLIP_FORMAT)
            solved_nvfp4_thresholds, clip_solver_metadata = (
                _solve_nvfp4_activation_clip_groups(
                    groups=solver_groups,
                    qname_to_module=qname_to_module,
                    activations=activations,
                    levers=levers,
                    joint_globals=joint_globals,
                    activation_max_abs=activation_max_abs,
                    fisher_rows=fisher_rows,
                    weights_out=weights,
                    cache_dir_path=cache_dir_path,
                    progress=progress,
                    awq_scales=awq_scales,
                    cache_formats=clip_cache_formats,
                )
            )
            clip_solver_metadata["candidate_format"] = (
                PRISMACLIP_FORMAT if solve_variant_nvfp4_clip else None
            )
            clip_solver_metadata["status"] = "applied"
        else:
            clip_solver_metadata["status"] = "no_eligible_groups"

    n = sum(len(render_formats_by_qname.get(q, ())) for q in qname_to_module)
    done = 0
    skipped_resumed = 0
    skipped_prewritten = 0
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
        for fmt in render_formats_by_qname.get(qname, ()):
            fmt_key = str(fmt).upper()
            render_fmt = _render_base_format(fmt_key)
            key = (qname, fmt_key)
            if key in weights:
                skipped_prewritten += 1
                done += 1
                if progress and done % 25 == 0:
                    print(f"[prod-cache] {done}/{n}", flush=True)
                continue
            # RESUME: in disk-streaming mode, if a shard already exists
            # for (qname, fmt) on disk, treat it as previously rendered
            # and skip re-rendering.  This lets a job that OOM'd at 95%
            # resume without re-doing the work — just rebuild the manifest
            # from the surviving .pt files.
            if cache_dir_path is not None:
                fname = _cache_weight_filename(qname, fmt_key)
                if (cache_dir_path / fname).is_file():
                    weights[(qname, fmt_key)] = fname
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
                    weight, render_fmt,
                    qname=qname,
                    activations=activations_local,
                    levers=levers,
                    joint_global_real=joint,
                    input_global_scale=export_scale,
                    act_clip_threshold=(
                        solved_nvfp4_thresholds.get(qname)
                        if (
                            _is_prismaclip_format(fmt_key)
                            or (solve_global_nvfp4_clip and render_fmt == "NVFP4")
                        )
                        else None
                    ),
                    fisher_row_weights=(
                        fisher_rows.get(qname)
                        if bool(levers.get("fisher_gptq", False))
                        and fisher_rows is not None
                        else None
                    ),
                    awq_scale=(
                        awq_scales.get(qname)
                        if render_fmt in _AWQ_CACHE_FORMATS else None
                    ),
                )
            except Exception as e:
                failed[(qname, fmt_key)] = str(e)
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
            _store_rendered_weight_entry(
                weights=weights,
                cache_dir_path=cache_dir_path,
                qname=qname,
                fmt=fmt_key,
                tensor=w_dq,
                weight_dtype=weight.dtype,
            )
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
            f"({skipped_resumed} resumed from disk, "
            f"{skipped_prewritten} prewritten by solver); "
            f"{len(failed)} failures",
            flush=True,
        )
    cache = ProductionWeightCache(
        weights=weights,
        levers=dict(levers),
        activation_max_abs=activation_max_abs or None,
        failed=failed,
        cache_dir=str(cache_dir_path) if cache_dir_path is not None else None,
        metadata={
            "activation_clip_solver": clip_solver_metadata,
            "awq": awq_metadata,
            "smoothquant": smoothquant_metadata,
            "fisher_weighted_gptq": {
                "enabled": bool(levers.get("fisher_gptq", False)),
                "h_detail_dir": str(h_detail_dir) if h_detail_dir else None,
                "loaded": (
                    int(fisher_rows.loads)
                    if fisher_rows is not None
                    and bool(levers.get("fisher_gptq", False))
                    else 0
                ),
                "misses": (
                    int(fisher_rows.misses)
                    if fisher_rows is not None
                    and bool(levers.get("fisher_gptq", False))
                    else 0
                ),
            },
            "prismafisherclip": {
                "enabled": bool(levers.get("fisher_clip", False)),
                "h_detail_dir": str(h_detail_dir) if h_detail_dir else None,
                "loaded": (
                    int(fisher_rows.loads)
                    if fisher_rows is not None
                    and bool(levers.get("fisher_clip", False))
                    else 0
                ),
                "misses": (
                    int(fisher_rows.misses)
                    if fisher_rows is not None
                    and bool(levers.get("fisher_clip", False))
                    else 0
                ),
            },
            "render_scope": render_scope,
            "requested_formats": list(requested_formats),
            "requested_entries": int(n),
        },
        awq_scales=awq_scales or None,
    )
    if recache_pass:
        from prismaquant.production_recache import recache_production_weight_cache

        if progress:
            print("[prod-cache] running production activation re-cache", flush=True)
        cache.prefetch_assignment(
            recache_assignment or {},
            max_resident_bytes=(
                cache._lru_max_bytes if cache._lru_max_bytes > 0 else None
            ),
            max_workers=4,
            require=False,
            progress=progress,
        )
        recache_production_weight_cache(
            model,
            calib_ids,
            recache_assignment or {},
            cache,
            profile=recache_profile,
            include_activation_quant=recache_include_activation_quant,
            microbatch_size=recache_microbatch_size,
            progress=progress,
        )
        compacted = cache.compact_for_pickle()
        if progress and compacted:
            print(
                f"[prod-cache] compacted {compacted} resident cache tensors "
                "back to path references after re-cache",
                flush=True,
            )
    return cache
