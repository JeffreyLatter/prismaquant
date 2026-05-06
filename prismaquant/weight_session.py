"""Incremental weight materialization session for polish.

Polish does many KL measurements; each measurement differs from the
previous by a single unit's format.  The naive path re-materializes
every Linear from BF16 source per measurement, which on a 27B model
with 305 units doubles GPU memory (model + clones of every materialized
slot) and OOMs on a 121 GB UMA.

WeightSession instead:

- Materializes the baseline assignment ONCE on the live model.params.
- Tracks the BF16 source for each unit (lazy-populated on first
  quantization) so any unit can be reverted later without keeping a
  full model backup.
- Exposes ``stage_format(qname, new_fmt)`` to swap a single unit and
  ``revert_last()`` / ``commit_last()`` to undo or accept the swap.
- Per trial, the only new allocation is one unit's worth of weight
  data (~50–500 MB).  No per-trial 54 GB clone.

Memory: 1× model (live) + bf16_originals (~half a model after every
unit has been quantized once) + working set.  On 27B that's ~70 GB
total + working ~10 GB, comfortably under a 121 GB UMA.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from prismaquant import block_clado as bc
from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import iter_quantizable_tensors


@dataclass
class _UndoEntry:
    qname: str
    prev_fmt: str
    prev_weight: torch.Tensor  # CPU/UMA snapshot of the weight before the swap


class WeightSession:
    """Owns the model's quantizable Linear weights for polish.

    Lifecycle:
      1. Construct.  Builds qname→Linear map.
      2. ``initialize(assignment, units)``.  Materializes the assignment
         on the live model.params.  Saves BF16 source for each
         unit-member touched (so future reverts don't need a full
         model clone).
      3. ``stage_format(qname, fmt)``.  Saves current weight onto an
         undo stack and applies the new format's weight in place.
      4. ``revert_last()`` / ``commit_last()``.  Pops the undo entry;
         revert restores the saved weight, commit drops the snapshot.
    """

    def __init__(
        self,
        model: nn.Module,
        production_weight_cache=None,
        snapshot_dir: str | None = None,
    ):
        """If ``snapshot_dir`` is provided, BF16 source snapshots are
        spilled to disk after capture instead of held in memory.  This
        bounds the in-memory snapshot footprint to a single tensor at a
        time, at the cost of one ``torch.save`` per first-touch and one
        ``torch.load`` per revert.  Required for very-large models
        (e.g. 70B+ on a 121 GB UMA host) where the cumulative BF16
        snapshot footprint of every quantizable Linear would exceed
        the budget.
        """
        self._model = model
        self._cache = production_weight_cache
        self._linear_by_qname: dict[str, tuple[nn.Module, str]] = {}
        for full_name, mod, attr in iter_quantizable_tensors(model):
            if attr != "weight" or not isinstance(mod, nn.Linear):
                continue
            qname = full_name[:-7] if full_name.endswith(".weight") else full_name
            self._linear_by_qname[qname] = (mod, attr)
        self._bf16_originals: dict[str, torch.Tensor] = {}
        self._snapshot_dir = None
        if snapshot_dir is not None:
            from pathlib import Path as _P
            self._snapshot_dir = _P(snapshot_dir)
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        # Maps qname -> spilled-to-disk filename (relative to snapshot_dir).
        # Mutually exclusive with _bf16_originals[qname].
        self._spilled: dict[str, str] = {}
        self._current: dict[str, str] = {}
        self._undo_stack: list[_UndoEntry] = []

    # ------------------------------------------------------------------
    # qname → live weight resolution
    # ------------------------------------------------------------------
    def _live_weight(self, qname: str) -> torch.Tensor | None:
        target = self._linear_by_qname.get(qname)
        if target is None:
            return None
        mod, attr = target
        param = getattr(mod, attr, None)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            return None
        return param.data

    def _ensure_bf16_snapshot(self, qname: str) -> torch.Tensor | None:
        """Return the BF16 source weight for ``qname``, snapshotting on
        first call so subsequent reverts can copy from it.

        Snapshot is taken AT THE TIME OF FIRST CALL, so this MUST be
        called BEFORE the live weight is overwritten by a quantized
        version.  Callers handle that ordering in ``initialize`` and
        ``stage_format``.

        Spill behavior: when ``snapshot_dir`` was passed to
        ``__init__``, the snapshot is written to disk after capture
        and dropped from memory; subsequent calls re-load from disk
        (one-shot, then dropped again).  This bounds the in-memory
        snapshot footprint at the cost of ``torch.save`` per
        first-touch and ``torch.load`` per revert.
        """
        if qname in self._bf16_originals:
            return self._bf16_originals[qname]
        if qname in self._spilled and self._snapshot_dir is not None:
            return torch.load(
                self._snapshot_dir / self._spilled[qname],
                map_location="cpu",
            )
        live = self._live_weight(qname)
        if live is None:
            return None
        # Detach + clone to UMA (same physical memory; 'cpu' just means
        # not part of the model's param graph).  This is a one-time cost
        # per qname.
        snap = live.detach().clone()
        if self._snapshot_dir is not None:
            # Spill to disk; do not hold in memory.  Atomic via tmp +
            # rename so a kill mid-write leaves no half-written file.
            safe = qname.replace("/", "__").replace(".", "_")
            fname = f"{safe}__bf16src.pt"
            tmp = self._snapshot_dir / (fname + ".tmp")
            torch.save(snap, tmp)
            import os as _os
            _os.replace(tmp, self._snapshot_dir / fname)
            self._spilled[qname] = fname
            return snap  # caller still uses the in-flight tensor; we'll
                         # re-load from disk on subsequent calls.
        self._bf16_originals[qname] = snap
        return snap

    def _format_weight(self, qname: str, fmt: str) -> torch.Tensor | None:
        """Return the weight tensor that should be installed when
        ``qname`` is at ``fmt``.

        BF16 → ``bf16_originals`` (lazy-snapshot from live).
        Other → production cache lookup; on miss, RTN-quantize the BF16
        source via the format's ``quantize_dequantize`` (matches the
        non-delta path's fallback in ``_quantized_weight_for``).
        """
        fmt_canon = fr.canonical_format_name(fmt)
        if fmt_canon == "BF16":
            return self._ensure_bf16_snapshot(qname)
        # Try cache first.
        if self._cache is not None:
            cached = self._cache.get(qname, fmt_canon)
            if cached is not None:
                return cached
        # Fall back to RTN-quantize from BF16 source (matches what the
        # OLD per-module hook path does when production cache misses).
        # MXFP8 commonly takes this path because the production cache
        # only fills NVFP4 by default.
        bf16 = self._ensure_bf16_snapshot(qname)
        if bf16 is None:
            return None
        try:
            spec = fr.get_format(fmt_canon)
        except Exception:
            return None
        return spec.quantize_dequantize(bf16.detach().clone())

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------
    def initialize(
        self,
        assignment: Mapping[str, str],
        units: Sequence[bc.DecisionUnit],
    ) -> None:
        """Apply ``assignment`` to live model.params.

        For each unit's qname, snapshot the current (BF16 source) weight
        and overwrite with the assigned format's weight from the cache.
        Units assigned BF16 are left as-is on the live model but still
        snapshotted so future reverts work.
        """
        member_to_unit: dict[str, bc.DecisionUnit] = {}
        for unit in units:
            for member in unit.member_qnames:
                member_to_unit[member] = unit

        for qname, fmt in assignment.items():
            if qname not in self._linear_by_qname:
                continue
            # Snapshot BEFORE any overwrite.
            self._ensure_bf16_snapshot(qname)
            self._current[qname] = fr.canonical_format_name(fmt)
            if self._current[qname] == "BF16":
                continue  # live weight already holds BF16 source
            replacement = self._format_weight(qname, self._current[qname])
            if replacement is None:
                continue  # cache miss; leave at BF16
            live = self._live_weight(qname)
            if live is None:
                continue
            live.copy_(replacement.to(device=live.device, dtype=live.dtype))

    # ------------------------------------------------------------------
    # Staged format swaps
    # ------------------------------------------------------------------
    def stage_format(
        self,
        qname: str,
        new_fmt: str,
    ) -> _UndoEntry | None:
        """Swap ``qname`` to ``new_fmt`` and push an undo entry.

        Returns the undo entry (also stored on self._undo_stack).
        ``revert_last`` restores from this entry; ``commit_last`` drops it.
        """
        if qname not in self._linear_by_qname:
            return None
        new_canon = fr.canonical_format_name(new_fmt)
        prev_fmt = self._current.get(qname, "BF16")
        if new_canon == prev_fmt:
            return None
        live = self._live_weight(qname)
        if live is None:
            return None
        prev_weight = live.detach().clone()
        replacement = self._format_weight(qname, new_canon)
        if replacement is None:
            return None
        live.copy_(replacement.to(device=live.device, dtype=live.dtype))
        entry = _UndoEntry(
            qname=qname, prev_fmt=prev_fmt, prev_weight=prev_weight,
        )
        self._undo_stack.append(entry)
        # Speculatively update current; commit_last leaves it; revert_last
        # rolls it back.
        self._current[qname] = new_canon
        return entry

    def revert_last(self) -> None:
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        live = self._live_weight(entry.qname)
        if live is not None:
            live.copy_(entry.prev_weight.to(
                device=live.device, dtype=live.dtype,
            ))
        self._current[entry.qname] = entry.prev_fmt
        # Drop the snapshot reference so the GC can reclaim.
        del entry

    def commit_last(self) -> None:
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        # current already updated speculatively; just drop the snapshot.
        del entry

    # ------------------------------------------------------------------
    # Sibling-aware multi-flip helpers (a fused unit's members all flip
    # together — committed/reverted as one atomic group).
    # ------------------------------------------------------------------
    def stage_unit(self, unit: bc.DecisionUnit, new_fmt: str) -> int:
        """Stage all members of ``unit`` to ``new_fmt``.  Returns the
        number of stage_format calls accepted (== number of undo
        entries pushed).  Use ``revert_unit_last(n)`` /
        ``commit_unit_last(n)`` to undo/accept the group atomically."""
        n = 0
        for member in unit.member_qnames:
            entry = self.stage_format(member, new_fmt)
            if entry is not None:
                n += 1
        return n

    def revert_unit_last(self, n: int) -> None:
        for _ in range(n):
            self.revert_last()

    def commit_unit_last(self, n: int) -> None:
        for _ in range(n):
            self.commit_last()

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    @property
    def n_bf16_snapshots(self) -> int:
        return len(self._bf16_originals) + len(self._spilled)

    @property
    def n_pending_undo(self) -> int:
        return len(self._undo_stack)

    def current_assignment(self) -> dict[str, str]:
        return dict(self._current)
