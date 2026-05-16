"""Allocator-side consumer for the Hadamard-DuQuant joint-search sidecar.

Lets the existing per-Linear DP/IP solver pick the joint (rotation, format)
optimum without any solver changes. The trick:

  1. **Cost override**: for each (qname, fmt) pair where the cluster offers
     a rotation candidate, replace the cost-table entry with that qname's
     share of ``min(no_rot+fmt, rot+fmt)``. The sidecar score is a joint
     cluster score, so it is divided across present consumer Linears before
     entering the per-Linear table; fused-sibling aggregation then sums back
     to the intended cluster cost instead of multiplying it by q/k/v or
     gate/up fanout.

  2. **Sibling coherence is automatic**: per-cluster sidecar costs are
     identical across all sibling consumers. The DP sees the same
     overridden cost at every consumer ⇒ fused-sibling promotion
     coheres them to one format ⇒ one rotation pick per cluster.

  3. **Pick derivation**: after the DP commits to per-Linear formats,
     :func:`derive_picks` reads each cluster's chosen format from the
     assignment and recovers the rotation pick (the one that produced
     the minimum the DP saw). Emitted as
     ``{cluster_key: "rot+FMT" | "no_rot+FMT"}`` for
     :mod:`prismaquant.hadamard_duquant_cache` to consume.

The sidecar is the same file
:mod:`prismaquant.joint_hadamard_format_search` writes, extended to
include ``consumer_qnames`` / ``producer_qnames`` per cluster so the
allocator can route per-Linear overrides without a separate specs file.

Default cost key is ``"output_mse"`` to match the allocator's existing
expectation (the joint search writes Fisher-weighted MSE under this key
for consistency with the production cache pipeline).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


__all__ = [
    "JointSearchSidecar",
    "ClusterCandidate",
    "ClusterOverride",
    "load_sidecar",
    "build_cluster_overrides",
    "apply_cost_overrides",
    "derive_picks",
    "emit_picks",
    "load_picks",
    "specs_from_sidecar",
    "load_state_from_artifacts",
]


# ---------------------------------------------------------------------------
# Sidecar parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterCandidate:
    """One (rotation, format) candidate cost entry from the sidecar."""
    label: str                       # e.g., "rot+NVFP4" or "no_rot+BF16"
    rotation: str                    # "rot" or "no_rot"
    format_label: str                # e.g., "NVFP4"
    fisher_mse: float
    bpp: float
    rotation_key: str | None = None  # safetensors key for rot+ entries
    runtime_transform_type: str | None = None
    runtime_head_dim: int | None = None


@dataclass(frozen=True)
class JointSearchSidecar:
    """Parsed sidecar from :mod:`prismaquant.joint_hadamard_format_search`."""

    version: str
    # cluster_key → {insertion_kind, group_size, online, consumer_qnames,
    #                producer_qnames, candidates: {label: ClusterCandidate}}
    clusters: dict[str, dict[str, Any]]

    def consumer_qnames(self, cluster_key: str) -> tuple[str, ...]:
        return tuple(self.clusters.get(cluster_key, {}).get("consumer_qnames", ()))

    def producer_qnames(self, cluster_key: str) -> tuple[str, ...]:
        return tuple(self.clusters.get(cluster_key, {}).get("producer_qnames", ()))

    def candidates(self, cluster_key: str) -> dict[str, ClusterCandidate]:
        return self.clusters.get(cluster_key, {}).get("candidates", {})


def _parse_label(label: str) -> tuple[str, str] | None:
    """Split a candidate label like ``rot+NVFP4`` into ``("rot", "NVFP4")``."""
    if "+" not in label:
        return None
    rot_part, fmt_part = label.split("+", 1)
    if rot_part not in {"rot", "no_rot"}:
        return None
    return (rot_part, fmt_part)


def load_sidecar(sidecar_path: Path | str) -> JointSearchSidecar:
    """Load and parse the joint-search sidecar JSON."""
    raw = json.loads(Path(sidecar_path).read_text())
    version = str(raw.get("version", "1"))
    clusters: dict[str, dict[str, Any]] = {}
    for ckey, cdata in (raw.get("clusters") or {}).items():
        raw_candidates = cdata.get("candidates") or {}
        parsed_candidates: dict[str, ClusterCandidate] = {}
        for label, cost_dict in raw_candidates.items():
            parsed = _parse_label(label)
            if parsed is None:
                continue
            rot, fmt = parsed
            parsed_candidates[label] = ClusterCandidate(
                label=label,
                rotation=rot,
                format_label=fmt,
                fisher_mse=float(cost_dict.get("fisher_mse", float("inf"))),
                bpp=float(cost_dict.get("bpp", 0.0)),
                rotation_key=cost_dict.get("rotation_key"),
                runtime_transform_type=cost_dict.get("runtime_transform_type"),
                runtime_head_dim=(
                    int(cost_dict["runtime_head_dim"])
                    if cost_dict.get("runtime_head_dim") is not None else None
                ),
            )
        clusters[ckey] = {
            "insertion_kind": str(cdata.get("insertion_kind", "")),
            "group_size": int(cdata.get("group_size", 0)),
            "input_dim": int(cdata.get("input_dim", 0)),
            "online": bool(cdata.get("online", False)),
            "consumer_qnames": list(cdata.get("consumer_qnames", [])),
            "producer_qnames": list(cdata.get("producer_qnames", [])),
            "candidates": parsed_candidates,
        }
    return JointSearchSidecar(version=version, clusters=clusters)


# ---------------------------------------------------------------------------
# Cost overrides — feeds the existing DP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterOverride:
    """One cluster's per-format min-cost decision.

    Per format ``f``: ``best_cost = min(cluster.no_rot+f, cluster.rot+f)``.
    ``virtual_picks[f]`` records which rotation produced the min — used
    after the DP commits to a format to recover the rotation pick.
    """

    cluster_key: str
    consumer_qnames: tuple[str, ...]
    # format_label → best_cost (Fisher-weighted MSE)
    best_cost_by_format: dict[str, float]
    # format_label → "rot" or "no_rot" (the winner of the per-format min)
    virtual_picks: dict[str, str]


def build_cluster_overrides(
    sidecar: JointSearchSidecar,
) -> dict[str, ClusterOverride]:
    """For each cluster, compute per-format min cost and which rotation won.

    Skips clusters whose candidate set has no rotation alternatives (e.g.,
    only ``no_rot+BF16``) — those need no override.

    Returns ``{cluster_key: ClusterOverride}``.
    """
    overrides: dict[str, ClusterOverride] = {}
    for ckey, cdata in sidecar.clusters.items():
        candidates: dict[str, ClusterCandidate] = cdata.get("candidates", {})
        if not candidates:
            continue
        consumer_qnames = tuple(cdata.get("consumer_qnames", ()))

        # Group candidates by format. Each group has up to 2 entries
        # (no_rot and rot).
        by_format: dict[str, dict[str, ClusterCandidate]] = {}
        for label, cand in candidates.items():
            by_format.setdefault(cand.format_label, {})[cand.rotation] = cand

        best_cost: dict[str, float] = {}
        virtual: dict[str, str] = {}
        for fmt, by_rot in by_format.items():
            finite_choices = [
                (rot, cand)
                for rot, cand in by_rot.items()
                if math.isfinite(float(cand.fisher_mse))
            ]
            if not finite_choices:
                # A noisy search/probe may occasionally produce NaN/Inf for
                # one cluster-format pair. Do not let that poison the DP; leave
                # the original measured allocator cost in place for this format.
                continue
            choices = sorted(
                finite_choices,
                key=lambda kv: kv[1].fisher_mse,
            )
            winner_rot, winner_cand = choices[0]
            best_cost[fmt] = winner_cand.fisher_mse
            virtual[fmt] = winner_rot

        if not best_cost:
            continue
        overrides[ckey] = ClusterOverride(
            cluster_key=ckey,
            consumer_qnames=consumer_qnames,
            best_cost_by_format=best_cost,
            virtual_picks=virtual,
        )
    return overrides


def apply_cost_overrides(
    cost_table: dict[str, dict[str, dict[str, Any]]],
    overrides: Mapping[str, ClusterOverride],
    *,
    cost_key: str = "output_mse",
) -> int:
    """Mutate ``cost_table`` in place with per-cluster min-rotation costs.

    For each (cluster, format), splits the cluster-level
    ``best_cost_by_format[format]`` evenly across consumer qnames that
    actually have that format in the cost table. After fused-sibling
    aggregation, the total objective contribution is the cluster score
    reported by joint search.

    Args:
        cost_table: the allocator's ``costs`` dict
            (``costs[qname][format] = cost_dict``).
        overrides: per-cluster overrides from :func:`build_cluster_overrides`.
        cost_key: name of the field inside the inner dict that the DP reads
            as the optimization objective. Defaults to ``"output_mse"``.

    Returns:
        The number of (qname, format) cells modified.
    """
    n_modified = 0
    for override in overrides.values():
        for fmt, best_cost in override.best_cost_by_format.items():
            eligible: list[str] = []
            for qname in override.consumer_qnames:
                qname_costs = cost_table.get(qname)
                if qname_costs is None or fmt not in qname_costs:
                    continue
                cell = qname_costs[fmt]
                if not isinstance(cell, dict):
                    continue
                eligible.append(qname)
            if not eligible:
                continue
            per_qname_cost = float(best_cost) / float(len(eligible))
            for qname in eligible:
                cell = cost_table[qname][fmt]
                if not isinstance(cell, dict):
                    continue
                cell[cost_key] = per_qname_cost
                n_modified += 1
    return n_modified


# ---------------------------------------------------------------------------
# Pick derivation — post-DP rotation recovery
# ---------------------------------------------------------------------------


def derive_picks(
    sidecar: JointSearchSidecar,
    overrides: Mapping[str, ClusterOverride],
    assignment: Mapping[str, str],
) -> dict[str, str]:
    """Recover the per-cluster rotation pick from the DP's final assignment.

    For each cluster with an override:
      - Identify the cluster's "canonical format" from the first consumer
        whose assignment is known. Fused-sibling promotion guarantees all
        consumers share one format in the production allocator path.
      - Look up the rotation winner for that format from
        ``override.virtual_picks``.
      - Emit ``{cluster_key: "<rotation>+<format>"}``.

    Clusters where no consumer appears in the assignment are skipped.
    Clusters whose chosen format has no rotation alternative (e.g.,
    ``no_rot+BF16`` only) yield ``"no_rot+<format>"``.
    """
    picks: dict[str, str] = {}
    for ckey, override in overrides.items():
        cluster_fmt: str | None = None
        for qname in override.consumer_qnames:
            cluster_fmt = assignment.get(qname)
            if cluster_fmt:
                break
        if cluster_fmt is None:
            continue
        rotation = override.virtual_picks.get(cluster_fmt, "no_rot")
        picks[ckey] = f"{rotation}+{cluster_fmt}"
    return picks


def emit_picks(
    picks: Mapping[str, str],
    output_path: Path | str,
) -> None:
    """Write the per-cluster picks dict to a JSON file.

    The output is consumed by :func:`prismaquant.hadamard_duquant_cache.load_cache_state`
    as the ``allocator_picks`` argument when the production cache fill
    and exporter are invoked downstream.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1",
        "picks": {str(k): str(v) for k, v in sorted(picks.items())},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Phase 6 downstream helpers — bridge sidecar/picks artifacts into the cache
# adapter and exporter
# ---------------------------------------------------------------------------


def load_picks(picks_path: Path | str) -> dict[str, str]:
    """Inverse of :func:`emit_picks`. Returns ``{cluster_key: pick_label}``."""
    raw = json.loads(Path(picks_path).read_text())
    picks_raw = raw.get("picks", {})
    return {str(k): str(v) for k, v in picks_raw.items()}


def specs_from_sidecar(sidecar: JointSearchSidecar) -> list:
    """Reconstruct :class:`HadamardDuQuantSpec` objects from the sidecar.

    The sidecar carries the cluster metadata that the cache adapter
    (:func:`prismaquant.hadamard_duquant_cache.load_cache_state`) needs.
    This helper lets downstream consumers skip the model-traversal step
    that the joint search originally did to discover insertion points.

    Returns a list ordered by ``cluster_key`` for deterministic output.
    """
    from prismaquant.hadamard_duquant import (
        HadamardDuQuantSpec,
        InsertionPointKind,
    )

    specs: list = []
    for cluster_key in sorted(sidecar.clusters.keys()):
        cdata = sidecar.clusters[cluster_key]
        kind_raw = cdata.get("insertion_kind", "")
        try:
            kind = InsertionPointKind(kind_raw)
        except (ValueError, KeyError):
            # Skip clusters with unrecognized insertion kinds — defensive
            # against forward-compatible sidecars that may add new kinds.
            continue
        consumer_qnames = tuple(cdata.get("consumer_qnames", ()))
        producer_qnames = tuple(cdata.get("producer_qnames", ()))
        if not consumer_qnames:
            continue
        try:
            spec = HadamardDuQuantSpec(
                cluster_key=cluster_key,
                kind=kind,
                input_dim=int(cdata.get("input_dim", 0)),
                group_size=int(cdata.get("group_size", 0)),
                consumer_qnames=consumer_qnames,
                producer_qnames=producer_qnames,
                online=bool(cdata.get("online", False)),
            )
        except ValueError:
            # Defensive: skip specs that fail validation rather than abort
            # the whole load (e.g., a downstream sidecar produced for a
            # different model architecture).
            continue
        specs.append(spec)
    return specs


def load_state_from_artifacts(
    sidecar_path: Path | str,
    rotation_safetensors_path: Path | str | None,
    picks_path: Path | str,
    *,
    device: str = "cpu",
):
    """One-shot loader: sidecar + rotations + picks ⇒ HadamardDuQuantCacheState.

    Builds insertion specs from the sidecar metadata (no model needed),
    reads allocator picks from disk, and constructs the cache state via
    :func:`prismaquant.hadamard_duquant_cache.load_cache_state`. Returns
    the state ready to pass to ``fill_production_weight_cache`` (Phase 3)
    and to the exporter's rotation helpers (Phase 4).

    Args:
        sidecar_path: joint-search sidecar JSON.
        rotation_safetensors_path: rotation matrices safetensors file. May
            be ``None`` or non-existent when no cluster was picked with
            rotation (the cache adapter handles that case).
        picks_path: per-cluster rotation picks emitted by the allocator.
        device: where to place the rotation tensors when loaded.

    Returns:
        :class:`HadamardDuQuantCacheState`. Empty state when the picks
        file is empty or no rotations were selected.
    """
    from prismaquant.hadamard_duquant_cache import load_cache_state

    sidecar = load_sidecar(sidecar_path)
    specs = specs_from_sidecar(sidecar)
    picks = load_picks(picks_path)
    return load_cache_state(
        sidecar_path=sidecar_path,
        rotation_safetensors_path=rotation_safetensors_path,
        allocator_picks=picks,
        insertion_specs=specs,
        device=device,
    )
