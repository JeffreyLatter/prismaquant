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
    * optional scalar scale-sweep
    * joint NVFP4 fused-sibling globals (q/k/v share a per-tensor scale,
      gate/up share theirs)
    * calibrated `input_global_scale` per fused-sibling group
      (max_abs(activations) / 6.0; the same value the export persists
      to the artifact)
    * progressive local render gates using the shared mechanism order
      baseline -> format scale rule -> GPTQ -> optional scale_sweep;
      individual formats explicitly opt out of unsupported mechanisms,
      and regressive candidates fall back to the previous accepted render
      while recording metadata
    * activation-weighted MXFP8_E4M3 E8M0 scale search and FP8 dynamic per-row
      scale search when scale_sweep is enabled; these refine the current
      accepted render rather than starting a separate format-specific path
    * activation-weighted GPTQ for FP8_E4M3/FP8_E5M2 and
      MXFP8_E4M3/MXFP8_E5M2. NVFP4 is the only production format that uses
      joint_scale_opt; MXFP8 uses the canonical E8M0 scale rule.
    * retired Fisher-weighted local objectives are archived under
      ``archive/fisher_2026-05-15/`` and are not part of the production
      pipeline
    * retired input-axis rotation experiments are not part of the production
      cache path.

  KNOWN GAPS (v2 work, NOT implemented):
    * batched NVFP4 GPTQ + scale-sweep across same-shape Linears
      (defaults-on in the export when activations are cached;
      mathematically equivalent to scalar but ~3-8× faster on MoE)
    * block-output match (post-GPTQ refinement against BF16 block output)
    * any export-only refinements added after this docstring is written

  FP8 / MXFP8_E4M3 / MXFP8_E5M2 / BF16:
    * FP8_E4M3/FP8_E5M2 and MXFP8_E4M3/MXFP8_E5M2 use the same GPTQ and
      scale paths as export when the corresponding levers are enabled.
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

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re

import torch
import torch.nn as nn

from prismaquant.activation_sampling import update_priority_reservoir
from prismaquant.build_rtn_cache import iter_quantizable_tensors
from prismaquant.render_score import (
    gate_render_candidate,
    normalize_row_weights,
    resolve_render_mechanism_order,
    score_render_error,
)
from prismaquant.source_prefetch import prefetch_files_to_page_cache


def _render_base_format(fmt: str) -> str:
    return str(fmt).strip().upper()


def _cache_weight_filename(qname: str, fmt: str) -> str:
    safe = qname.replace("/", "__").replace(".", "_")
    return f"{safe}__{fmt}.pt"


_UNCACHED_PACKED_EXPERT_RE = re.compile(
    r"\.experts(?:\.\d+)?\."
    r"(?:gate_up_proj|down_proj|gate_proj|up_proj|w1|w2|w3)$"
)


def is_uncached_packed_expert_qname(qname: str) -> bool:
    """Return True for packed-MoE expert tensors not rendered by this cache.

    ``ProductionWeightCache`` currently renders production-faithful 2D
    ``nn.Linear`` weights. Packed 3D MoE expert tensors are quantized by the
    packed exporter/validation fallback path, so missing cache entries for
    those names must not fail production-cache residency checks.
    """
    return bool(_UNCACHED_PACKED_EXPERT_RE.search(str(qname)))


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
    validation or recache time the lazy-load caches each weight in memory
    after first access, so steady-state behavior matches the in-memory mode.

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
                if is_uncached_packed_expert_qname(str(qname)):
                    continue
                missing.append((str(qname), fmt_canon))
                continue
            if key not in seen:
                keys.append(key)
                seen.add(key)
        return keys, missing

    def assignment_file_paths(
        self,
        assignment: Mapping[str, str],
    ) -> tuple[list[Path], list[tuple[str, str]], list[tuple[str, str]]]:
        """Return disk files backing an assignment without loading tensors.

        This is a page-cache residency helper for validation paths that
        destructively materialize one assignment into the model. It reuses the
        same cache key resolution as ``prefetch_assignment`` but intentionally
        does not call ``torch.load`` or create another rendered-weight cache.
        """
        keys, missing = self.assignment_keys(assignment)
        paths: list[Path] = []
        in_memory: list[tuple[str, str]] = []
        seen_paths: set[Path] = set()
        for key in keys:
            value = self.weights.get(key)
            if value is None:
                missing.append(key)
                continue
            if isinstance(value, torch.Tensor):
                path_value = (
                    self._lru_paths.get(key)
                    if self._lru_paths is not None else None
                )
                if path_value is None:
                    in_memory.append(key)
                    continue
                value = path_value
            path = Path(self._path_for_value(value)).resolve()
            if path not in seen_paths:
                paths.append(path)
                seen_paths.add(path)
        return paths, missing, in_memory

    def prefetch_assignment_file_pages(
        self,
        assignment: Mapping[str, str],
        *,
        mode: str = "require",
        max_resident_bytes: int | None = None,
        headroom_gb: float = 24.0,
        max_workers: int = 4,
        progress: bool = False,
        log_prefix: str = "[prod-cache-files]",
    ) -> dict[str, object]:
        """Prefetch assignment cache files into the OS page cache.

        Unlike ``prefetch_assignment``, this keeps rendered weights out of the
        Python heap. The following ``get`` calls still go through
        ``ProductionWeightCache`` and its LRU, but deserialization reads from
        resident file pages instead of faulting against NVMe.
        """
        paths, missing, in_memory = self.assignment_file_paths(assignment)
        mode = str(mode or "off").lower()
        if paths:
            stats = prefetch_files_to_page_cache(
                paths,
                mode=mode,
                max_resident_bytes=max_resident_bytes,
                headroom_gb=headroom_gb,
                workers=max_workers,
                progress=progress,
                log_prefix=log_prefix,
                label="production cache files",
            )
        else:
            stats = {
                "mode": mode,
                "label": "production cache files",
                "files": 0,
                "bytes": 0,
                "max_resident_bytes": int(max_resident_bytes or 0),
                "available_bytes": None,
                "prefetched_bytes": 0,
                "elapsed_seconds": 0.0,
                "skipped": True,
                "reason": "no disk-backed production cache files",
            }
        stats["keys"] = len(paths) + len(in_memory)
        stats["in_memory"] = len(in_memory)
        stats["missing"] = len(missing)
        if missing:
            stats["missing_sample"] = missing[:8]
            msg = (
                f"production cache missing {len(missing)} assignment entries; "
                f"sample={missing[:8]}"
            )
            if mode == "require":
                raise RuntimeError(msg)
            if progress:
                print(f"{log_prefix} WARNING: {msg}", flush=True)
        return stats

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
        profile=None,
    ):
        self.model = model
        self.profile = profile
        self.qnames = qnames
        self.store_qnames = set(store_qnames) if store_qnames is not None else set(qnames)
        self.max_rows = int(max_rows)
        self.store_device = torch.device(store_device or "cpu")
        self.store_dtype = store_dtype
        self.activations: dict[str, list[torch.Tensor]] = {}
        self._activation_priorities: dict[str, torch.Tensor] = {}
        self._activation_generator = torch.Generator(device="cpu")
        self._activation_generator.manual_seed(42)
        self.max_abs: dict[str, float] = {}
        self._max_abs_tensors: dict[str, torch.Tensor] = {}
        self._handles: list = []
        self._name_by_id: dict[int, str] = {}
        for full_name, mod, attr in iter_quantizable_tensors(model, self.profile):
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
            for full_name, mod, attr in iter_quantizable_tensors(
                self.model,
                self.profile,
            ):
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
            current = (
                torch.cat(self.activations[key], dim=0)
                if self.activations[key]
                else None
            )
            sampled, priorities = update_priority_reservoir(
                current,
                self._activation_priorities.get(key),
                flat,
                max_rows=self.max_rows,
                generator=self._activation_generator,
            )
            self.activations[key] = [] if sampled is None else [sampled]
            if priorities is None:
                self._activation_priorities.pop(key, None)
            else:
                self._activation_priorities[key] = priorities
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
        "gptq": bool(levers.get("gptq", True)),
        "scale_sweep": bool(levers.get("scale_sweep", False)),
        "static_act_order": bool(levers.get("static_act_order", False)),
        "joint_scale_opt": bool(levers.get("joint_scale_opt", False)),
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

    def __init__(
        self,
        h_detail_dir: str | Path | None,
        fused_sibling_mapping: Mapping[str, Sequence[str]] | None = None,
    ):
        self.detail_dir = Path(h_detail_dir) if h_detail_dir else None
        self.fused_sibling_mapping = {
            str(fused): tuple(str(member) for member in members)
            for fused, members in (fused_sibling_mapping or {}).items()
        }
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

    def _split_fused_names(self, qname: str) -> tuple[str, ...]:
        if "." not in qname:
            return ()
        prefix, leaf = qname.rsplit(".", 1)
        members = self.fused_sibling_mapping.get(leaf)
        if not members:
            return ()
        return tuple(f"{prefix}.{member}" for member in members)

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


def _fused_sibling_leaf_mapping_from_profile(profile) -> dict[str, tuple[str, ...]]:
    if profile is None:
        return {}
    getter = getattr(profile, "fused_sibling_leaf_mapping", None)
    if not callable(getter):
        return {}
    try:
        mapping = getter()
    except Exception:
        return {}
    return {
        str(fused): tuple(str(member) for member in members)
        for fused, members in (mapping or {}).items()
    }


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


@contextmanager
def _temporary_nvfp4_scale_rule(rule: str):
    from prismaquant import export_native_compressed as enc

    previous = enc._NVFP4_SCALE_RULE
    enc._NVFP4_SCALE_RULE = enc.resolve_nvfp4_scale_rule(rule)
    try:
        yield
    finally:
        enc._NVFP4_SCALE_RULE = previous

def _store_rendered_weight_entry(
    *,
    weights: dict[tuple[str, str], object],
    cache_dir_path: Path | None,
    qname: str,
    fmt: str,
    tensor: torch.Tensor,
    weight_dtype: torch.dtype,
) -> None:
    from prismaquant import format_registry as fr

    fmt = fr.canonical_format_name(str(fmt).strip().upper())
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

@dataclass
class _RenderedCandidate:
    label: str
    weight: torch.Tensor
    score: float
    metric: str
    scale_rule: str
    package: tuple[str, ...]
    has_gptq: bool


def _render_score_for_gate(
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor | None,
) -> tuple[float, str]:
    """Score a local render candidate with the shared scorer.

    Activations should normally be present in production cache renders.  The
    weight-MSE fallback keeps pure RTN/FourOverSix unit tests and non-act-aware
    formats measurable without adding a second scoring abstraction.
    """
    if (
        activations is not None
        and activations.numel() > 0
        and int(activations.shape[-1]) == int(reference_weight.shape[1])
    ):
        return (
            score_render_error(
                reference_weight,
                rendered_weight,
                activations,
                row_weights=None,
            ),
            "output_mse",
        )
    diff = (
        reference_weight.detach().to(torch.float32)
        - rendered_weight.detach().to(
            device=reference_weight.device,
            dtype=torch.float32,
        )
    )
    return float(diff.pow(2).mean().item()), "weight_mse"


def _render_score_record_key(qname: str, fmt: str) -> str:
    return f"{qname}|{fmt.upper()}"


def _render_score_normalizer(
    reference_weight: torch.Tensor,
    activations: torch.Tensor | None,
    metric: str,
) -> tuple[float, int]:
    rows, cols = reference_weight.shape
    if (
        metric in {"output_mse", "fisher_output_mse"}
        and activations is not None
        and activations.numel() > 0
        and int(activations.shape[-1]) == int(cols)
    ):
        n_act_rows = int(activations.reshape(-1, cols).shape[0])
        return float(max(1, n_act_rows) * int(rows)), n_act_rows
    return float(max(1, int(rows) * int(cols))), 0


def _render_score_record(
    *,
    qname: str,
    fmt: str,
    render_format: str,
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor | None,
    activation_max_abs: float | None,
) -> dict[str, object]:
    raw_score, raw_metric = _render_score_for_gate(
        reference_weight.detach().to(torch.float32),
        rendered_weight,
        activations,
    )
    score = raw_score
    metric = raw_metric
    activation_quantized = False
    activation_clipped = False
    activation_clip_max = (
        activation_max_abs
        if _format_uses_static_activation_clip(fmt) else
        None
    )
    if (
        activations is not None
        and activations.numel() > 0
        and int(activations.shape[-1]) == int(reference_weight.shape[1])
    ):
        try:
            from prismaquant import format_registry as fr

            spec = fr.get_format(fr.canonical_format_name(fmt))
            score, metric, activation_quantized, activation_clipped = (
                _local_forward_render_score(
                    reference_weight=reference_weight,
                    rendered_weight=rendered_weight,
                    activations=activations,
                    activation_quantize=spec.activation_quantize_dequantize,
                    activation_max_abs=activation_clip_max,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"activation-aware render scoring failed for {qname} @ {fmt}: "
                f"{exc}"
            ) from exc
    normalizer, activation_rows = _render_score_normalizer(
        reference_weight,
        activations,
        metric,
    )
    # weight_mse is the original prismaquant cost surrogate: pure
    # (W_orig - W_rendered)^2 averaged over weights. Activation-independent,
    # low variance; the allocator multiplies by h_trace for predicted_dloss.
    ref_f = reference_weight.detach().to(
        device=rendered_weight.device, dtype=torch.float32,
    )
    rendered_f = rendered_weight.detach().to(torch.float32)
    diff = ref_f - rendered_f
    n_weights = int(diff.numel())
    weight_mse = float(diff.pow(2).mean().item()) if n_weights > 0 else 0.0
    rows, cols = reference_weight.shape
    return {
        "qname": str(qname),
        "format": str(fmt).upper(),
        "render_format": str(render_format).upper(),
        "metric": str(metric),
        "score": float(score),
        "score_sum": float(score) * float(normalizer),
        "raw_render_metric": str(raw_metric),
        "raw_render_score": float(raw_score),
        "raw_render_score_sum": float(raw_score) * float(normalizer),
        "weight_mse": float(weight_mse),
        "weight_mse_sum": float(weight_mse) * float(n_weights),
        "n_weights": int(n_weights),
        "normalizer": float(normalizer),
        "activation_rows": int(activation_rows),
        "activation_quantized": bool(activation_quantized),
        "activation_clipped": bool(activation_clipped),
        "activation_max_abs": (
            float(activation_clip_max)
            if activation_clip_max is not None and activation_clip_max > 0
            else None
        ),
        "out_features": int(rows),
        "in_features": int(cols),
    }


def _format_uses_static_activation_clip(fmt: str) -> bool:
    """Return whether local scoring should apply a calibrated activation max.

    NVFP4 serving uses a calibrated tensor-level activation scale plus local
    tensor-group quantization. MXFP8/FP8 dynamic serving computes activation
    scales at runtime, so applying the NVFP4 activation max to those formats
    prices the wrong kernel contract.
    """
    from prismaquant import format_registry as fr

    return fr.canonical_format_name(str(fmt).strip().upper()) == "NVFP4"


def _local_forward_render_score(
    *,
    reference_weight: torch.Tensor,
    rendered_weight: torch.Tensor,
    activations: torch.Tensor,
    activation_quantize,
    activation_max_abs: float | None,
    row_chunk: int = 128,
) -> tuple[float, str, bool, bool]:
    rows, cols = reference_weight.shape
    if rendered_weight.shape != reference_weight.shape:
        return float("inf"), "output_mse", False, False
    if activations.shape[-1] != cols:
        return float("inf"), "output_mse", False, False
    device = reference_weight.device
    ref_t = reference_weight.detach().to(device=device, dtype=torch.float32).t()
    rendered_t = rendered_weight.detach().to(device=device, dtype=torch.float32).t()
    x = activations.detach().to(device=device, dtype=torch.float32).reshape(-1, cols)
    clipped = False
    if (
        activation_max_abs is not None
        and float(activation_max_abs) > 0
        and _env_flag("PRISMAQUANT_PROD_ACT_SCALES", True)
    ):
        x_quant_input = x.clamp(-float(activation_max_abs), float(activation_max_abs))
        clipped = True
    else:
        x_quant_input = x

    total = torch.zeros((), dtype=torch.float32, device=device)
    quantized_any = False
    with torch.no_grad():
        for start in range(0, x.shape[0], int(row_chunk)):
            x_ref = x[start:start + int(row_chunk)]
            x_q = activation_quantize(x_quant_input[start:start + int(row_chunk)])
            if x_q is not x_ref:
                quantized_any = quantized_any or not torch.equal(
                    x_q.detach().to(device=device, dtype=torch.float32),
                    x_ref,
                )
            x_q = x_q.to(device=device, dtype=torch.float32)
            y_ref = x_ref @ ref_t
            y_q = x_q @ rendered_t
            err = (y_ref - y_q).pow(2)
            total = total + err.sum()
    score = float(total.item()) / max(1, int(x.shape[0]) * int(rows))
    return score, "output_mse", bool(quantized_any), bool(clipped)


def _load_render_score_sidecar(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.is_file():
        return {}
    import json as _json

    try:
        raw = _json.loads(path.read_text())
    except Exception:
        return {}
    records = raw.get("records") if isinstance(raw, Mapping) else None
    if not isinstance(records, Mapping):
        return {}
    out: dict[str, dict[str, object]] = {}
    for key, value in records.items():
        if isinstance(value, Mapping):
            out[str(key)] = dict(value)
    return out


def _write_render_score_sidecar(
    path: Path | None,
    records: Mapping[str, Mapping[str, object]],
) -> None:
    if path is None:
        return
    import json as _json

    payload = {
        "schema": "prismaquant.production_render_scores.v1",
        "records": dict(sorted((str(k), dict(v)) for k, v in records.items())),
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _format_supports_render_mechanism(fmt: str, mechanism: str) -> bool:
    """Return whether a shared render mechanism is meaningful for ``fmt``.

    The production render pipeline is format-agnostic in order. Individual
    formats opt out of mechanisms whose math or exported schema does not
    apply.
    """

    fmt_u = str(fmt).strip().upper()
    mech = str(mechanism).strip()
    if fmt_u == "NVFP4":
        return mech in {
            "four_over_six",
            "gptq",
            "static_act_order",
            "joint_scale_opt",
            "fisher_gptq",
            "scale_sweep",
        }
    if fmt_u in {"FP8_E4M3", "FP8_E5M2"}:
        return mech == "gptq" or (mech == "scale_sweep" and fmt_u == "FP8_E4M3")
    if fmt_u == "MXFP4":
        return mech in {"gptq", "static_act_order"}
    if fmt_u in {"MXFP8_E4M3", "MXFP8_E5M2"}:
        return mech in {"gptq", "static_act_order"} or (
            mech == "scale_sweep" and fmt_u == "MXFP8_E4M3"
        )
    return False


def _render_nvfp4_progressive_candidate(
    *,
    qname: str,
    weight_scaled: torch.Tensor,
    activations_scaled: torch.Tensor | None,
    levers: Mapping[str, object],
    scale_rule: str,
    joint_global_real: torch.Tensor | None,
    act_clip_threshold: float | None,
    act_clip_rescale: str | None,
    fisher_row_weights: torch.Tensor | None,
    include_gptq: bool,
    include_scale_sweep: bool,
) -> torch.Tensor:
    from prismaquant import export_native_compressed as enc

    with _temporary_nvfp4_scale_rule(scale_rule):
        current = enc._rtn_dequant_nvfp4(
            weight_scaled,
            group_size=16,
            global_real_override=joint_global_real,
        )
        if activations_scaled is None or activations_scaled.numel() == 0:
            return current
        if include_gptq:
            if os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1") != "0":
                current = enc._gptq_obs_rounding_nvfp4_swept(
                    weight_scaled,
                    activations_scaled,
                    group_size=16,
                    global_real_override=joint_global_real,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    static_act_order=bool(
                        levers.get("static_act_order", False)
                    ),
                    joint_scale_opt=bool(
                        levers.get("joint_scale_opt", False)
                    ),
                )
            else:
                current = enc._gptq_obs_rounding_nvfp4(
                    weight_scaled,
                    activations_scaled,
                    group_size=16,
                    global_real_override=joint_global_real,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=act_clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                    static_act_order=bool(
                        levers.get("static_act_order", False)
                    ),
                    joint_scale_opt=bool(
                        levers.get("joint_scale_opt", False)
                    ),
                )
        if include_scale_sweep:
            current = enc._scale_sweep_nvfp4(
                current,
                activations_scaled,
                group_size=16,
                global_real_override=joint_global_real,
                reference_weight=weight_scaled,
                clip_threshold=act_clip_threshold,
                clip_rescale=act_clip_rescale,
                fisher_row_weights=fisher_row_weights,
            )
        return current


def _render_nvfp4_progressively(
    weight: torch.Tensor,
    *,
    qname: str,
    activations: Mapping[str, torch.Tensor],
    levers: Mapping[str, object],
    joint_global_real: torch.Tensor | None,
    act_clip_threshold: float | None,
    act_clip_rescale: str | None,
    fisher_row_weights: torch.Tensor | None,
    gate_trace: list[dict[str, object]] | None,
) -> torch.Tensor:
    from prismaquant import export_native_compressed as enc

    requested_rule = enc.resolve_nvfp4_scale_rule(
        str(levers.get("nvfp4_scale_rule", "static_6"))
    )
    f6_enabled = requested_rule == enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE
    gptq_enabled = bool(levers.get("gptq", True))
    scale_sweep_enabled = bool(levers.get("scale_sweep", False))
    static_act_order_enabled = bool(
        gptq_enabled and levers.get("static_act_order", False)
    )
    joint_scale_opt_enabled = bool(
        gptq_enabled and levers.get("joint_scale_opt", False)
    )
    gptq_modifiers = tuple(
        name for name, enabled in (
            ("static_act_order", static_act_order_enabled),
            ("joint_scale_opt", joint_scale_opt_enabled),
        )
        if enabled
    )
    gptq_scale_rule = (
        enc.NVFP4_SCALE_RULE_JOINT_MSE
        if joint_scale_opt_enabled
        else None
    )
    min_gain = _env_float(
        "PRISMAQUANT_RENDER_GATE_MIN_GAIN",
        0.0,
        lo=-1.0,
        hi=1.0,
    )

    reference = weight.detach().to(device=weight.device, dtype=torch.float32)
    acts = activations.get(qname)
    acts_for_render = (
        acts.detach().to(device=weight.device, dtype=torch.float32)
        if acts is not None and int(acts.shape[-1]) == int(weight.shape[1])
        else None
    )
    reference_for_render = reference

    def candidate(
        *,
        label: str,
        scale_rule: str,
        package: tuple[str, ...],
        include_gptq: bool,
        include_scale_sweep: bool,
    ) -> _RenderedCandidate:
        rendered_scaled = _render_nvfp4_progressive_candidate(
            qname=qname,
            weight_scaled=reference_for_render,
            activations_scaled=acts_for_render,
            levers=levers,
            scale_rule=scale_rule,
            joint_global_real=joint_global_real,
            act_clip_threshold=act_clip_threshold,
            act_clip_rescale=act_clip_rescale,
            fisher_row_weights=fisher_row_weights,
            include_gptq=include_gptq,
            include_scale_sweep=include_scale_sweep,
        )
        rendered = rendered_scaled
        score, metric = _render_score_for_gate(
            reference,
            rendered,
            acts,
        )
        return _RenderedCandidate(
            label=label,
            weight=rendered,
            score=float(score),
            metric=metric,
            scale_rule=scale_rule,
            package=package,
            has_gptq=bool(include_gptq),
        )

    static_rule = enc.NVFP4_SCALE_RULE_STATIC_6
    current = candidate(
        label="rtn_static_6",
        scale_rule=static_rule,
        package=(),
        include_gptq=False,
        include_scale_sweep=False,
    )
    if gate_trace is not None:
        gate_trace.append({
            "mechanism": "baseline",
            "selected": current.label,
            "score": float(current.score),
            "metric": current.metric,
            "scale_rule": current.scale_rule,
            "package": list(current.package),
        })

    def apply_gate(
        *,
        mechanism: str,
        candidates: Sequence[_RenderedCandidate],
    ) -> None:
        nonlocal current
        if not candidates:
            return
        best = min(candidates, key=lambda item: item.score)
        decision = gate_render_candidate(
            baseline_score=current.score,
            candidate_score=best.score,
            metric=best.metric,
            min_relative_gain=min_gain,
        )
        accepted = bool(decision.accepted)
        if gate_trace is not None:
            gate_trace.append({
                "mechanism": mechanism,
                "accepted": accepted,
                "selected": best.label if accepted else current.label,
                "candidate": best.label,
                "baseline_score": float(current.score),
                "candidate_score": float(best.score),
                "relative_gain": float(decision.relative_gain),
                "metric": best.metric,
                "reason": str(decision.reason),
                "scale_rule": best.scale_rule,
                "package": list(best.package),
                "candidates": [
                    {
                        "label": cand.label,
                        "score": float(cand.score),
                        "metric": cand.metric,
                        "scale_rule": cand.scale_rule,
                        "package": list(cand.package),
                    }
                    for cand in candidates
                ],
            })
        if accepted:
            old = current.weight
            current = best
            if old is not best.weight:
                del old
        for cand in candidates:
            if cand is not current:
                del cand.weight

    if f6_enabled:
        apply_gate(
            mechanism="four_over_six",
            candidates=[
                candidate(
                    label="four_over_six",
                    scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
                    package=("four_over_six",),
                    include_gptq=False,
                    include_scale_sweep=False,
                )
            ],
        )

    if gptq_enabled and acts_for_render is not None:
        gptq_name = "fisher_gptq" if fisher_row_weights is not None else "gptq"
        primary_scale_rule = gptq_scale_rule or current.scale_rule
        primary_package = (
            (gptq_name, "gptq") if gptq_name != "gptq" else ("gptq",)
        )
        primary_package = tuple(dict.fromkeys((*gptq_modifiers, *primary_package)))
        packages: list[_RenderedCandidate] = [
            candidate(
                label="+".join((primary_scale_rule, *gptq_modifiers, gptq_name)),
                scale_rule=primary_scale_rule,
                package=primary_package,
                include_gptq=True,
                include_scale_sweep=False,
            )
        ]
        if (
            f6_enabled
            and not joint_scale_opt_enabled
            and current.scale_rule != enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE
        ):
            packages.append(candidate(
                label=f"four_over_six+{gptq_name}",
                scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
                package=(
                    (*gptq_modifiers, "four_over_six", gptq_name, "gptq")
                    if gptq_name != "gptq" else
                    (*gptq_modifiers, "four_over_six", "gptq")
                ),
                include_gptq=True,
                include_scale_sweep=False,
            ))
        apply_gate(mechanism=gptq_name, candidates=packages)

    if scale_sweep_enabled and acts_for_render is not None:
        scale_candidates: list[_RenderedCandidate] = [
            candidate(
                label=f"{current.label}+scale_sweep",
                scale_rule=current.scale_rule,
                package=tuple(dict.fromkeys((*current.package, "scale_sweep"))),
                include_gptq=current.has_gptq,
                include_scale_sweep=True,
            )
        ]
        if gptq_enabled and not current.has_gptq:
            gptq_name = "fisher_gptq" if fisher_row_weights is not None else "gptq"
            scale_rule = gptq_scale_rule or current.scale_rule
            pkg = (
                (gptq_name, "gptq", "scale_sweep")
                if gptq_name != "gptq" else
                ("gptq", "scale_sweep")
            )
            pkg = tuple(dict.fromkeys((*gptq_modifiers, *pkg)))
            scale_candidates.append(candidate(
                label="+".join((scale_rule, *gptq_modifiers, gptq_name, "scale_sweep")),
                scale_rule=scale_rule,
                package=pkg,
                include_gptq=True,
                include_scale_sweep=True,
            ))
        if (
            f6_enabled
            and not joint_scale_opt_enabled
            and current.scale_rule != enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE
        ):
            gptq_name = "fisher_gptq" if fisher_row_weights is not None else "gptq"
            include_gptq = bool(gptq_enabled)
            pkg = [*gptq_modifiers, "four_over_six"]
            if include_gptq:
                if gptq_name != "gptq":
                    pkg.extend([gptq_name, "gptq"])
                else:
                    pkg.append("gptq")
            pkg.append("scale_sweep")
            scale_candidates.append(candidate(
                label="+".join(pkg),
                scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
                package=tuple(pkg),
                include_gptq=include_gptq,
                include_scale_sweep=True,
            ))
        apply_gate(mechanism="scale_sweep", candidates=scale_candidates)

    return current.weight.to(device=weight.device, dtype=weight.dtype).contiguous()


def _summarize_render_gate_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "enabled": True,
        "entries": int(len(records)),
        "mechanisms": {},
    }
    mechanisms: dict[str, dict[str, object]] = {}
    for record in records:
        for step in record.get("trace", []):  # type: ignore[union-attr]
            if not isinstance(step, Mapping):
                continue
            mech = str(step.get("mechanism", "unknown"))
            if mech == "baseline":
                continue
            bucket = mechanisms.setdefault(mech, {
                "accepted": 0,
                "rejected": 0,
                "reasons": {},
                "package_accepted": 0,
            })
            accepted = bool(step.get("accepted", False))
            if accepted:
                bucket["accepted"] = int(bucket["accepted"]) + 1
                package = step.get("package")
                if (
                    isinstance(package, Sequence)
                    and not isinstance(package, str)
                    and mech in package
                ):
                    bucket["package_accepted"] = int(bucket["package_accepted"]) + 1
            else:
                bucket["rejected"] = int(bucket["rejected"]) + 1
            reason = str(step.get("reason", "unknown"))
            reasons = bucket["reasons"]
            if isinstance(reasons, dict):
                reasons[reason] = int(reasons.get(reason, 0)) + 1

            package = step.get("package")
            if isinstance(package, Sequence) and not isinstance(package, str):
                for member in package:
                    member_name = str(member)
                    if member_name == mech:
                        continue
                    member_bucket = mechanisms.setdefault(member_name, {
                        "accepted": 0,
                        "rejected": 0,
                        "reasons": {},
                        "package_accepted": 0,
                    })
                    if accepted:
                        member_bucket["package_accepted"] = (
                            int(member_bucket["package_accepted"]) + 1
                        )
    summary["mechanisms"] = mechanisms
    return summary


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
    act_clip_rescale: str | None = None,
    fisher_row_weights: torch.Tensor | None = None,
    gate_trace: list[dict[str, object]] | None = None,
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
    by per-token gradient² from h-detail.

    """
    from prismaquant import format_registry as fr

    fmt = fr.canonical_format_name(str(fmt).strip().upper())
    clip_rescale = "none"
    if str(act_clip_rescale or "none").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
        "none",
    }:
        raise ValueError("activation clip rescaling is not supported")
    progressive_gates = _env_flag("PRISMAQUANT_RENDER_PROGRESSIVE_GATES", True)
    if fmt == "NVFP4" and progressive_gates:
        return _render_nvfp4_progressively(
            weight,
            qname=qname,
            activations=activations,
            levers=levers,
            joint_global_real=joint_global_real,
            act_clip_threshold=act_clip_threshold,
            act_clip_rescale=clip_rescale,
            fisher_row_weights=fisher_row_weights,
            gate_trace=gate_trace,
        )

    if fmt != "NVFP4":
        spec = fr.get_format(fmt)
        baseline = spec.quantize_dequantize(weight.detach().clone()).to(
            device=weight.device, dtype=weight.dtype,
        )
        reference = weight.detach().to(torch.float32)
        acts = activations.get(qname)
        acts_for_render = (
            acts.detach().to(device=weight.device, dtype=torch.float32)
            if acts is not None and int(acts.shape[-1]) == int(weight.shape[1])
            else None
        )
        baseline_score, baseline_metric = _render_score_for_gate(
            reference,
            baseline,
            acts,
        )
        current = _RenderedCandidate(
            label=f"{fmt.lower()}+rtn",
            weight=baseline.contiguous(),
            score=float(baseline_score),
            metric=baseline_metric,
            scale_rule="",
            package=(),
            has_gptq=False,
        )
        if gate_trace is not None:
            gate_trace.append({
                "mechanism": "baseline",
                "selected": current.label,
                "score": float(current.score),
                "metric": current.metric,
                "package": [],
            })

        def _apply_non_nv_gate(
            *,
            mechanism: str,
            candidates: Sequence[_RenderedCandidate],
        ) -> None:
            nonlocal current
            if not candidates:
                return
            best = min(candidates, key=lambda item: item.score)
            decision = gate_render_candidate(
                baseline_score=current.score,
                candidate_score=best.score,
                metric=best.metric,
                min_relative_gain=_env_float(
                    "PRISMAQUANT_RENDER_GATE_MIN_GAIN",
                    0.0,
                    lo=-1.0,
                    hi=1.0,
                ),
            )
            if gate_trace is not None:
                gate_trace.append({
                    "mechanism": mechanism,
                    "accepted": bool(decision.accepted),
                    "selected": (
                        best.label if decision.accepted else current.label
                    ),
                    "candidate": best.label,
                    "baseline_score": float(current.score),
                    "candidate_score": float(best.score),
                    "relative_gain": float(decision.relative_gain),
                    "metric": best.metric,
                    "reason": str(decision.reason),
                    "package": list(best.package),
                    "candidates": [
                        {
                            "label": cand.label,
                            "score": float(cand.score),
                            "metric": cand.metric,
                            "package": list(cand.package),
                        }
                        for cand in candidates
                    ],
                })
            if decision.accepted:
                old = current.weight
                current = best
                if old is not best.weight:
                    del old
            for cand in candidates:
                if cand is not current and cand.weight is not current.weight:
                    del cand.weight

        def _non_nv_candidate(
            *,
            label: str,
            weight_dq: torch.Tensor,
            package: tuple[str, ...],
            has_gptq: bool,
        ) -> _RenderedCandidate:
            rendered = weight_dq.to(device=weight.device, dtype=weight.dtype).contiguous()
            score, metric = _render_score_for_gate(reference, rendered, acts)
            return _RenderedCandidate(
                label=label,
                weight=rendered,
                score=float(score),
                metric=metric,
                scale_rule="",
                package=package,
                has_gptq=bool(has_gptq),
            )

        if (
            _format_supports_render_mechanism(fmt, "gptq")
            and bool(levers.get("gptq", True))
            and acts_for_render is not None
        ):
            from prismaquant import export_native_compressed as enc

            joint_scale_opt = bool(
                levers.get("joint_scale_opt", False)
                and _format_supports_render_mechanism(fmt, "joint_scale_opt")
            )
            static_act_order = bool(
                levers.get("static_act_order", False)
                and _format_supports_render_mechanism(fmt, "static_act_order")
            )
            base_package = (
                ("joint_scale_opt", "gptq")
                if joint_scale_opt else
                ("gptq",)
            )
            use_damp_sweep = (
                os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1") != "0"
            )

            def _gptq_candidate(use_static_act_order: bool) -> _RenderedCandidate:
                package = tuple(dict.fromkeys((
                    *(
                        ("static_act_order",)
                        if use_static_act_order else
                        ()
                    ),
                    *base_package,
                )))
                if fmt == "MXFP4":
                    if use_damp_sweep:
                        _q, _s, candidate = enc._gptq_obs_rounding_mxfp4_swept(
                            reference,
                            acts_for_render,
                            group_size=32,
                            clip_threshold=act_clip_threshold,
                            clip_rescale=clip_rescale,
                            fisher_row_weights=fisher_row_weights,
                            static_act_order=use_static_act_order,
                        )
                    else:
                        _q, _s, candidate = enc._gptq_obs_rounding_mxfp4(
                            reference,
                            acts_for_render,
                            group_size=32,
                            clip_threshold=act_clip_threshold,
                            clip_rescale=clip_rescale,
                            fisher_row_weights=fisher_row_weights,
                            static_act_order=use_static_act_order,
                        )
                elif use_damp_sweep:
                    _q, _s, candidate = enc._gptq_obs_rounding_fp8_like_swept(
                        reference,
                        acts_for_render,
                        fmt=fmt,
                        group_size=32,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        joint_scale_opt=joint_scale_opt,
                        static_act_order=use_static_act_order,
                    )
                else:
                    _q, _s, candidate = enc._gptq_obs_rounding_fp8_like(
                        reference,
                        acts_for_render,
                        fmt=fmt,
                        group_size=32,
                        clip_threshold=act_clip_threshold,
                        clip_rescale=clip_rescale,
                        fisher_row_weights=fisher_row_weights,
                        joint_scale_opt=joint_scale_opt,
                        static_act_order=use_static_act_order,
                    )
                return _non_nv_candidate(
                    label=f"{fmt.lower()}+{'+'.join(package)}",
                    weight_dq=candidate,
                    package=package,
                    has_gptq=True,
                )

            gptq_candidates = [_gptq_candidate(False)]
            if static_act_order:
                gptq_candidates.append(_gptq_candidate(True))
            _apply_non_nv_gate(
                mechanism="gptq",
                candidates=gptq_candidates,
            )

        if (
            _format_supports_render_mechanism(fmt, "scale_sweep")
            and bool(levers.get("scale_sweep", False))
            and acts_for_render is not None
        ):
            if fmt == "MXFP8_E4M3":
                from prismaquant.export_native_compressed import (
                    _mxfp8_scale_sweep_quantize,
                )

                _, _, w_dq = _mxfp8_scale_sweep_quantize(
                    current.weight.detach().to(torch.float32),
                    acts_for_render,
                    group_size=32,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                )
            else:
                from prismaquant.export_native_compressed import (
                    _fp8_dynamic_scale_sweep_quantize,
                )

                _, _, w_dq = _fp8_dynamic_scale_sweep_quantize(
                    current.weight.detach().to(torch.float32),
                    acts_for_render,
                    clip_threshold=act_clip_threshold,
                    clip_rescale=clip_rescale,
                    fisher_row_weights=fisher_row_weights,
                )
            candidate = _non_nv_candidate(
                label=f"{current.label}+scale_sweep",
                weight_dq=w_dq,
                package=tuple(dict.fromkeys((*current.package, "scale_sweep"))),
                has_gptq=current.has_gptq,
            )
            if progressive_gates:
                _apply_non_nv_gate(
                    mechanism="scale_sweep",
                    candidates=[candidate],
                )
                return current.weight.contiguous()
            return candidate.weight.contiguous()
        return current.weight.contiguous()

    from prismaquant.export_native_compressed import _quantize_2d

    with _temporarily_install_act_aware(activations, levers):
        result = _quantize_2d(
            weight.detach().clone(),
            fmt="NVFP4",
            linear_name=qname,
            nvfp4_global_real_override=joint_global_real,
            input_global_scale_override=input_global_scale,
            act_clip_threshold=act_clip_threshold,
            act_clip_rescale=clip_rescale,
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
      levers: which production levers to enable (default: GPTQ with optional
        joint NVFP4 scale optimization when requested by the caller).
      recache_pass: when True, run a second calibration forward with the
        concrete production assignment installed from this cache and refit
        ``activation_max_abs`` under quantized upstream weights.
      recache_assignment: required when ``recache_pass`` is True.  Candidate
        caches with multiple possible formats per Linear are ambiguous; recache
        needs the actual export assignment.
      h_detail_dir: optional probe h-detail directory retained for archived
        Fisher ablations. V1 production defaults do not require it.
    """
    if recache_pass and not recache_assignment:
        raise ValueError(
            "recache_pass=True requires recache_assignment with the concrete "
            "production assignment"
        )
    levers = dict(levers) if levers is not None else {}
    default_optional_levers = not bool(levers.pop("none", False))
    if not default_optional_levers:
        for name in (
            "gptq",
            "scale_sweep",
            "fisher_gptq",
            "static_act_order",
            "joint_scale_opt",
        ):
            levers.setdefault(name, False)
    levers.setdefault("gptq", True)
    levers.setdefault(
        "gptq_damp_sweep",
        bool(levers.get("gptq", True))
        and os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1") != "0",
    )
    levers.setdefault("scale_sweep", False)
    levers.setdefault(
        "static_act_order",
        _env_flag("PRISMAQUANT_GPTQ_STATIC_ACT_ORDER", False),
    )
    levers.setdefault(
        "joint_scale_opt",
        _env_flag("PRISMAQUANT_NVFP4_JOINT_SCALE_OPT", False),
    )
    if not bool(levers.get("gptq", True)):
        levers["static_act_order"] = False
        levers["joint_scale_opt"] = False
    levers.setdefault(
        "fisher_gptq",
        _env_flag("PRISMAQUANT_FISHER_WEIGHTED_GPTQ", False),
    )
    from prismaquant.export_native_compressed import (
        NVFP4_SCALE_RULE_ENV,
        NVFP4_SCALE_RULE_JOINT_MSE,
        resolve_nvfp4_scale_rule,
    )
    if (
        bool(levers.get("joint_scale_opt", False))
        and "nvfp4_scale_rule" not in levers
        and NVFP4_SCALE_RULE_ENV not in os.environ
    ):
        levers["nvfp4_scale_rule"] = NVFP4_SCALE_RULE_JOINT_MSE
    levers.setdefault("nvfp4_scale_rule", resolve_nvfp4_scale_rule())
    enabled_mechanisms: list[str] = []
    if str(levers.get("nvfp4_scale_rule", "")).strip() == "four_over_six_mse":
        enabled_mechanisms.append("four_over_six")
    if bool(levers.get("gptq", True)):
        enabled_mechanisms.append("gptq")
    if bool(levers.get("static_act_order", False)):
        enabled_mechanisms.append("static_act_order")
    if bool(levers.get("joint_scale_opt", False)):
        enabled_mechanisms.append("joint_scale_opt")
    if bool(levers.get("fisher_gptq", False)):
        enabled_mechanisms.append("fisher_gptq")
    if bool(levers.get("scale_sweep", False)):
        enabled_mechanisms.append("scale_sweep")
    mechanism_plan = resolve_render_mechanism_order(enabled_mechanisms)
    if mechanism_plan.errors:
        raise ValueError(
            "invalid render mechanism plan: " + "; ".join(mechanism_plan.errors)
        )
    if progress and mechanism_plan.ordered:
        print(
            "[prod-cache] render mechanism order: "
            + " -> ".join(spec.name for spec in mechanism_plan.ordered),
            flush=True,
        )

    from prismaquant import format_registry as fr

    def _canon(fmt: str) -> str:
        return fr.canonical_format_name(str(fmt).strip().upper())

    requested_formats = tuple(
        dict.fromkeys(_canon(f) for f in formats if str(f).strip())
    )
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
    model_profile = recache_profile
    if model_profile is None:
        try:
            from .model_profiles import profile_from_model
            model_profile = profile_from_model(model)
        except Exception:
            model_profile = None

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
    render_score_sidecar_path: Path | None = (
        cache_dir_path / "render_scores.json"
        if cache_dir_path is not None else None
    )
    render_score_records: dict[str, dict[str, object]] = (
        _load_render_score_sidecar(render_score_sidecar_path)
    )
    if progress and render_score_records:
        print(
            f"[prod-cache] resume: loaded {len(render_score_records)} "
            "render-score entries from sidecar",
            flush=True,
        )

    fmt_set = {
        fmt
        for fmts in render_formats_by_qname.values()
        for fmt in fmts
    }
    render_base_fmt_set = {_render_base_format(fmt) for fmt in fmt_set}
    # Store activations for every missing rendered format.  NVFP4 needs them
    # for GPTQ/JSO; FP8_E4M3/FP8_E5M2 and MXFP8_E4M3/MXFP8_E5M2 need them
    # for their activation-aware renders, and the production-render allocator
    # cost always needs them to score the final local forward error after the
    # format's activation quantizer.
    activation_aware_formats = set(fmt_set)
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
    qnames_needing_activation = set()
    for q, missing in missing_formats_by_qname.items():
        requested = tuple(render_formats_by_qname.get(q, ()))
        missing_activation_render = any(
            f in activation_aware_formats for f in missing
        )
        missing_activation_score = any(
            f in activation_aware_formats
            and _render_score_record_key(q, f) not in render_score_records
            for f in requested
        )
        if missing_activation_render or missing_activation_score:
            qnames_needing_activation.add(q)
    device = next(model.parameters()).device
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
            profile=model_profile,
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

    for full_name, mod, attr in iter_quantizable_tensors(model, model_profile):
        if attr != "weight" or not isinstance(mod, nn.Linear):
            continue
        qname = full_name[:-7] if full_name.endswith(".weight") else full_name
        if qname in qname_set:
            qname_to_module[qname] = mod

    fused_sibling_mapping = (
        _fused_sibling_leaf_mapping_from_profile(model_profile)
        if model_profile is not None
        else {}
    )
    fisher_rows = (
        _FisherRowWeightCache(
            h_detail_dir,
            fused_sibling_mapping or None,
        )
        if (bool(levers.get("fisher_gptq", False)) and h_detail_dir)
        else None
    )
    if progress and bool(levers.get("fisher_gptq", False)):
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
    needs_nvfp4_render = any(
        any(_render_base_format(fmt) == "NVFP4" for fmt in missing)
        for missing in missing_formats_by_qname.values()
    )
    if needs_nvfp4_render:
        from prismaquant.export_native_compressed import (
            _compute_nvfp4_joint_global,
        )
        synthetic_assignment = {q: "NVFP4" for q in qname_to_module}
        joint_globals = _compute_nvfp4_joint_global(
            model,
            synthetic_assignment,
            profile=model_profile,
        )
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
            gk = (
                fused_group_key(model_profile, qname)
                if model_profile is not None else qname
            )
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

    # MEM: free per-Linear activation tensors after each render so peak
    # memory stays bounded.  On 27B, 497 Linears × ~10K in_features × 512
    # rows × fp32 = ~10 GB just for activations.  Freeing in-loop drops
    # this to ~20 MB resident.
    import gc as _gc
    activations_local = dict(activations)  # shallow copy; we'll pop entries
    n = sum(len(render_formats_by_qname.get(q, ())) for q in qname_to_module)
    done = 0
    skipped_resumed = 0
    skipped_prewritten = 0
    render_gate_records: list[dict[str, object]] = []
    for qname, mod in qname_to_module.items():
        weight = mod.weight.data
        joint = joint_globals.get(qname)
        max_abs = activation_max_abs.get(qname)
        row_weights = (
            fisher_rows.get(qname)
            if bool(levers.get("fisher_gptq", False))
            and fisher_rows is not None
            else None
        )
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
                disk_path = cache_dir_path / fname
                if disk_path.is_file():
                    weights[(qname, fmt_key)] = fname
                    skipped_resumed += 1
                    score_key = _render_score_record_key(qname, fmt_key)
                    if score_key not in render_score_records:
                        try:
                            cached = torch.load(
                                disk_path,
                                map_location=weight.device,
                                weights_only=True,
                            ).to(device=weight.device, dtype=weight.dtype)
                            render_score_records[score_key] = _render_score_record(
                                qname=qname,
                                fmt=fmt_key,
                                render_format=render_fmt,
                                reference_weight=weight,
                                rendered_weight=cached,
                                activations=activations_local.get(qname),
                                activation_max_abs=max_abs,
                            )
                            del cached
                        except Exception:
                            pass
                    # Do NOT pop activations_local[qname] here: this
                    # loop iterates through every format for this
                    # Linear, and a later format in the same outer
                    # iteration may still need the activation tensor
                    # to render.  The outer pop after the format loop
                    # drops it once all formats are done.
                    continue
            try:
                gate_trace: list[dict[str, object]] = []
                w_dq = render_production_weight(
                    weight, render_fmt,
                    qname=qname,
                    activations=activations_local,
                    levers=levers,
                    joint_global_real=joint,
                    input_global_scale=export_scale,
                    fisher_row_weights=row_weights,
                    gate_trace=gate_trace,
                )
                render_score_records[_render_score_record_key(qname, fmt_key)] = (
                    _render_score_record(
                        qname=qname,
                        fmt=fmt_key,
                        render_format=render_fmt,
                        reference_weight=weight,
                        rendered_weight=w_dq,
                        activations=activations_local.get(qname),
                        activation_max_abs=max_abs,
                    )
                )
                if gate_trace:
                    render_gate_records.append({
                        "qname": qname,
                        "format": fmt_key,
                        "render_format": render_fmt,
                        "trace": gate_trace,
                    })
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
                _write_render_score_sidecar(
                    render_score_sidecar_path,
                    render_score_records,
                )
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
            f"{skipped_prewritten} pre-existing entries); "
            f"{len(failed)} failures",
            flush=True,
        )
    _write_render_score_sidecar(render_score_sidecar_path, render_score_records)
    render_gate_summary = _summarize_render_gate_records(render_gate_records)
    cache = ProductionWeightCache(
        weights=weights,
        levers=dict(levers),
        activation_max_abs=activation_max_abs or None,
        failed=failed,
        cache_dir=str(cache_dir_path) if cache_dir_path is not None else None,
        metadata={
            "render_mechanism_order": [
                {
                    "name": spec.name,
                    "operation": spec.operation,
                    "scope": spec.scope,
                    "gate_metric": spec.gate_metric,
                }
                for spec in mechanism_plan.ordered
            ],
            "render_failures": {
                f"{qname}|{fmt}": str(error)
                for (qname, fmt), error in sorted(failed.items())
            },
            "render_gates": {
                **render_gate_summary,
                "records": render_gate_records,
            },
            "render_scores": {
                "schema": "prismaquant.production_render_scores.v1",
                "entries": int(len(render_score_records)),
                "records": dict(sorted(render_score_records.items())),
                "cost_semantics": (
                    "score is the rendered candidate's local forward-error "
                    "mean after the format activation quantizer; score_sum "
                    "is score multiplied by activation_rows * out_features "
                    "for output metrics, or by parameter count for weight_mse "
                    "fallback. raw_render_score records the post-render "
                    "weight-only reconstruction objective used by local "
                    "render gates."
                ),
            },
            "four_over_six": (
                render_gate_summary.get("mechanisms", {}).get("four_over_six", {
                    "accepted": 0,
                    "rejected": 0,
                    "package_accepted": 0,
                    "reasons": {},
                })
                if isinstance(render_gate_summary.get("mechanisms"), dict)
                else {
                    "accepted": 0,
                    "rejected": 0,
                    "package_accepted": 0,
                    "reasons": {},
                }
            ),
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
            "render_scope": render_scope,
            "requested_formats": list(requested_formats),
            "requested_entries": int(n),
        },
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
            profile=model_profile,
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
