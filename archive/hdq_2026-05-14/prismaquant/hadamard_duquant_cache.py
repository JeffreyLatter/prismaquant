"""Adapter between joint-search outputs and the production weight cache.

Loads the joint search sidecar + rotation safetensors + the allocator's
per-cluster picks into a unified :class:`HadamardDuQuantCacheState`. The cache
fill loop and exporter both consume this state — neither re-solves the
rotation; both read it verbatim. This closes the cascade-bug loophole
(Locked Decision 9) by keeping a single source of truth for what was
chosen and what matrix to apply.

Three rotation operations:

  - **Consumer weight** (``WEIGHT_INPUT``): ``W → W @ M^T`` per G-block on
    the input axis. Applied at render time so downstream local methods
    (four_over_six, GPTQ, scale_sweep) operate in rotated coordinates.
  - **Consumer activations** (paired with weight rotation during render):
    ``x → x @ M^T`` per G-block on the last axis. Keeps the matmul algebra
    consistent so the cached weight ends up in original input coordinates
    after the round-trip ``Q(W M^T) @ M``.
  - **Producer weight** (``WEIGHT_OUTPUT``): ``W → M @ W`` per G-block on
    the output axis (used by offline-fold sites: residual stream, V→O).
    Applied by the exporter on the producer's stored weight; bias gets the
    same rotation since ``y = (M @ W) @ x + (M @ b)``.

The on-disk artifacts this module reads were produced by
:mod:`prismaquant.joint_hadamard_format_search`. The allocator picks are
expected to be supplied by the caller — Phase 5 wiring will plumb them
from ``prismaquant/allocator.py``'s output.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors

from prismaquant.hadamard_duquant import (
    HadamardDuQuantSpec,
    apply_block_rotation_input,
)


__all__ = [
    "CachedRotation",
    "HadamardDuQuantCacheState",
    "load_cache_state",
    "parse_pick_label",
    "rotate_consumer_weight",
    "rotate_consumer_activations",
    "rotate_producer_weight",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedRotation:
    """One cluster's rotation as installed in the production cache.

    Attributes:
        cluster_key: the fused-sibling cluster identifier (matches the
            ``cluster_key`` on a :class:`HadamardDuQuantSpec`).
        format_label: the allocator-chosen format that this rotation was
            tuned for (e.g., ``"NVFP4"``).
        composed_matrix: ``(G, G)`` matrix to apply — the
            ``P^T R P``-style composition produced by the joint search.
        group_size: ``G``, must equal ``composed_matrix.shape[0]``.
        insertion_kind: string value of the originating
            :class:`InsertionPointKind`.
        online: ``True`` for runtime-applied insertion points (i.e.,
            ``down_proj``). Offline-fold sites set this to ``False``.
        runtime_transform_type: compressed-tensors runtime transform type
            for ONLINE clusters. ``"hadamard"`` is vLLM-native and does
            not require a runtime artifact safetensors entry, though the
            search/cache artifacts still store the matrix used to render
            rotated weights. ``"random-matrix"`` is the dense learned form
            and is currently research-only for NVFP4 online transforms.
    """

    cluster_key: str
    format_label: str
    composed_matrix: torch.Tensor
    group_size: int
    insertion_kind: str
    online: bool
    runtime_transform_type: str = "random-matrix"

    def __post_init__(self) -> None:
        m = self.composed_matrix
        if m.dim() != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(
                f"composed_matrix must be a square 2D tensor, got {tuple(m.shape)}"
            )
        if int(m.shape[0]) != int(self.group_size):
            raise ValueError(
                f"composed_matrix shape {tuple(m.shape)} does not match "
                f"group_size {self.group_size}"
            )
        if self.runtime_transform_type not in {"random-matrix", "hadamard"}:
            raise ValueError(
                f"unsupported runtime_transform_type "
                f"{self.runtime_transform_type!r}"
            )


@dataclass
class HadamardDuQuantCacheState:
    """Per-cluster rotation state for the production cache.

    Built by :func:`load_cache_state` from the joint search outputs and the
    allocator's per-cluster picks. Lookup methods provide O(1) qname →
    rotation routing for the cache fill loop and the exporter.

    Attributes:
        rotations_by_cluster: ``{cluster_key: CachedRotation}`` — only
            populated for clusters the allocator picked a ``rot+FMT``
            candidate for.
        consumer_to_cluster: ``{qname: cluster_key}`` index for Linears
            whose input axis is rotated.
        producer_to_cluster: ``{qname: cluster_key}`` index for Linears
            whose output axis is rotated (offline-fold producer side).
    """

    rotations_by_cluster: dict[str, CachedRotation] = field(default_factory=dict)
    consumer_to_cluster: dict[str, str] = field(default_factory=dict)
    producer_to_cluster: dict[str, str] = field(default_factory=dict)

    def rotation_for_consumer(self, qname: str) -> CachedRotation | None:
        cluster = self.consumer_to_cluster.get(qname)
        return self.rotations_by_cluster.get(cluster) if cluster else None

    def rotation_for_producer(self, qname: str) -> CachedRotation | None:
        cluster = self.producer_to_cluster.get(qname)
        return self.rotations_by_cluster.get(cluster) if cluster else None

    def is_empty(self) -> bool:
        return not self.rotations_by_cluster

    def as_block_rotations_field(self) -> dict[str, torch.Tensor]:
        """Materialize the ``block_rotations`` dict for
        :class:`ProductionWeightCache`.

        Stores one CPU tensor per cluster keyed by ``cluster_key``. The
        exporter and any future install-time consumer can read this back
        without needing the sidecar or safetensors files.
        """
        return {
            ckey: rot.composed_matrix.detach().cpu().contiguous()
            for ckey, rot in self.rotations_by_cluster.items()
        }

    def as_metadata_field(self) -> dict[str, Any]:
        """Serialize the non-tensor routing state for production replay.

        ``ProductionWeightCache.block_rotations`` carries the matrices; this
        metadata carries the qname→cluster routing and scalar attributes needed
        by recache / perturbation replay to apply the same runtime basis that
        the exported artifact will use.
        """
        return {
            "enabled": bool(self.rotations_by_cluster),
            "consumer_to_cluster": dict(sorted(self.consumer_to_cluster.items())),
            "producer_to_cluster": dict(sorted(self.producer_to_cluster.items())),
            "clusters": {
                ckey: {
                    "format_label": rot.format_label,
                    "group_size": int(rot.group_size),
                    "insertion_kind": rot.insertion_kind,
                    "online": bool(rot.online),
                    "runtime_transform_type": rot.runtime_transform_type,
                }
                for ckey, rot in sorted(self.rotations_by_cluster.items())
            },
        }


# ---------------------------------------------------------------------------
# Sidecar / safetensors loading
# ---------------------------------------------------------------------------


def parse_pick_label(pick: str) -> tuple[bool, str] | None:
    """Parse an allocator pick like ``"rot+NVFP4"`` or ``"no_rot+BF16"``.

    Returns ``(with_rotation, format_label)`` or ``None`` if the label
    doesn't match the expected schema.
    """
    if not pick or "+" not in pick:
        return None
    rot_part, fmt_part = pick.split("+", 1)
    if rot_part == "rot":
        return (True, fmt_part)
    if rot_part == "no_rot":
        return (False, fmt_part)
    return None


def _safetensors_key(cluster_key: str, format_label: str) -> str:
    """Canonical safetensors key — must match
    :mod:`prismaquant.joint_hadamard_format_search`."""
    return f"{cluster_key}/{format_label}/composed_matrix"


def load_cache_state(
    sidecar_path: Path | str,
    rotation_safetensors_path: Path | str | None,
    allocator_picks: dict[str, str],
    insertion_specs: Sequence[HadamardDuQuantSpec],
    *,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> HadamardDuQuantCacheState:
    """Load joint-search outputs + allocator decisions into cache state.

    For each cluster in ``allocator_picks``:
      - Parse the pick label. If the allocator did not pick a ``rot+FMT``
        candidate for this cluster, skip — no rotation will be installed.
      - Look up the corresponding composed matrix from the safetensors
        file using the canonical key. A missing matrix is a hard error by
        default because an allocator ``rot+FMT`` pick without the matching
        rotation would export a numerically different artifact.
      - Build the qname → cluster_key indices from the matching
        :class:`HadamardDuQuantSpec` so the cache fill loop can route
        per-Linear lookups in O(1).

    Args:
        sidecar_path: joint-search sidecar JSON (presently parsed only to
            validate the file exists; the allocator's picks supersede it
            at install time). Tolerated as ``None`` only if the picks
            don't reference any cluster.
        rotation_safetensors_path: rotation matrices store. May be
            ``None`` or non-existent if no cluster was picked with
            rotation — in that case the returned state is empty.
        allocator_picks: ``{cluster_key: pick_label}`` from the allocator.
        insertion_specs: the specs used during joint search; used to
            recover the consumer/producer qname lists.
        device: device on which the rotation tensors should be placed.
        strict: when ``True`` (default), raise on missing specs or missing
            rotation tensors for any ``rot+FMT`` pick. ``False`` preserves
            the old best-effort behavior for forensic reads of partial
            artifacts.
    """
    # Validate sidecar exists (the picks reference its cluster_keys).
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"sidecar not found: {sidecar_path}")
    # Load the sidecar; we keep the parsed dict for future per-candidate
    # cost introspection though we don't currently key off it here.
    sidecar_raw = json.loads(sidecar_path.read_text())
    runtime_transform_types: dict[tuple[str, str], str] = {}
    runtime_head_dims: dict[tuple[str, str], int] = {}
    for cluster_key, cdata in (sidecar_raw.get("clusters") or {}).items():
        for label, candidate in (cdata.get("candidates") or {}).items():
            parsed = parse_pick_label(label)
            if parsed is None or not parsed[0]:
                continue
            lookup_key = (str(cluster_key), parsed[1])
            runtime_transform_types[lookup_key] = str(
                candidate.get("runtime_transform_type")
                or candidate.get("transform_type")
                or "random-matrix"
            )
            if candidate.get("runtime_head_dim") is not None:
                runtime_head_dims[lookup_key] = int(candidate["runtime_head_dim"])

    parsed_rotation_picks: list[tuple[str, str]] = []
    for cluster_key, pick in allocator_picks.items():
        parsed = parse_pick_label(pick)
        if parsed is not None and parsed[0]:
            parsed_rotation_picks.append((cluster_key, parsed[1]))

    # Load rotation tensors (if any).
    rotation_tensors: dict[str, torch.Tensor] = {}
    if rotation_safetensors_path is not None:
        safetensors_path = Path(rotation_safetensors_path)
        if safetensors_path.exists():
            rotation_tensors = load_safetensors(
                str(safetensors_path), device=str(device)
            )
        elif parsed_rotation_picks and strict:
            raise FileNotFoundError(
                "allocator picked rotated Hadamard-DuQuant clusters but "
                f"rotation safetensors file is missing: {safetensors_path}"
            )
    elif parsed_rotation_picks and strict:
        raise FileNotFoundError(
            "allocator picked rotated Hadamard-DuQuant clusters but no "
            "rotation_safetensors_path was provided"
        )

    spec_by_cluster: dict[str, HadamardDuQuantSpec] = {
        s.cluster_key: s for s in insertion_specs
    }

    rotations: dict[str, CachedRotation] = {}
    consumer_to_cluster: dict[str, str] = {}
    producer_to_cluster: dict[str, str] = {}

    for cluster_key, pick in allocator_picks.items():
        parsed = parse_pick_label(pick)
        if parsed is None or not parsed[0]:
            continue  # allocator did not pick rotation for this cluster
        _, fmt_label = parsed

        spec = spec_by_cluster.get(cluster_key)
        if spec is None:
            if strict:
                raise KeyError(
                    f"allocator picked rotation for cluster {cluster_key!r} "
                    "but no matching insertion spec was provided"
                )
            continue  # no matching spec — skip silently

        key = _safetensors_key(cluster_key, fmt_label)
        if key not in rotation_tensors:
            if strict:
                raise KeyError(
                    f"allocator picked {pick!r} for cluster {cluster_key!r} "
                    f"but rotation tensor {key!r} is missing"
                )
            continue  # joint search did not store this rotation

        matrix = rotation_tensors[key].to(device=device)
        runtime_head_dim = runtime_head_dims.get(
            (cluster_key, fmt_label),
            int(matrix.shape[0]),
        )
        rotation = CachedRotation(
            cluster_key=cluster_key,
            format_label=fmt_label,
            composed_matrix=matrix,
            group_size=int(runtime_head_dim),
            insertion_kind=str(spec.kind.value),
            online=bool(spec.online),
            runtime_transform_type=runtime_transform_types.get(
                (cluster_key, fmt_label),
                "random-matrix",
            ),
        )
        rotations[cluster_key] = rotation

        for qname in spec.consumer_qnames:
            consumer_to_cluster[qname] = cluster_key
        for qname in spec.producer_qnames:
            producer_to_cluster[qname] = cluster_key

    return HadamardDuQuantCacheState(
        rotations_by_cluster=rotations,
        consumer_to_cluster=consumer_to_cluster,
        producer_to_cluster=producer_to_cluster,
    )


# ---------------------------------------------------------------------------
# Rotation application — consumer (input axis) side
# ---------------------------------------------------------------------------


def rotate_consumer_weight(
    weight: torch.Tensor,
    rotation: CachedRotation,
) -> torch.Tensor:
    """Rotate the consumer's weight input axis: ``W → W @ M^T`` per G-block.

    Output has the same shape and device/dtype as ``weight``. Used by the
    cache fill loop to prepare the weight before format-specific rendering.
    """
    M_t = rotation.composed_matrix.to(device=weight.device, dtype=weight.dtype).t()
    return apply_block_rotation_input(weight, M_t)


def rotate_consumer_activations(
    activations: torch.Tensor,
    rotation: CachedRotation,
) -> torch.Tensor:
    """Rotate consumer's input activations: ``x → x @ M^T`` per G-block.

    Acts on the last axis. Output has the same shape and dtype as
    ``activations``. The cache fill loop uses this so downstream local
    methods (four_over_six, GPTQ, scale_sweep) see rotated activations
    aligned with the rotated weight.
    """
    M_t = rotation.composed_matrix.to(
        device=activations.device, dtype=activations.dtype
    ).t()
    return apply_block_rotation_input(activations, M_t)


# ---------------------------------------------------------------------------
# Rotation application — producer (output axis) side
# ---------------------------------------------------------------------------


def rotate_producer_weight(
    weight: torch.Tensor,
    rotation: CachedRotation,
    *,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Rotate the producer's weight output axis: ``W → M @ W`` per G-block.

    For Linear's matmul ``y = x @ W^T``, rotating the output axis means
    transforming W so the output ``y`` lives in the rotated basis. This is
    the offline-fold counterpart applied at the producer side (residual
    stream rotation, V→O rotation).

    If a bias is provided, the same rotation is applied (``b → M @ b`` per
    G-block) so the full affine ``y + b`` stays consistent.

    Args:
        weight: ``(out_features, in_features)`` tensor.
        rotation: ``CachedRotation`` whose composed matrix is applied
            block-diagonally along the *output* axis. ``out_features`` must
            be divisible by ``rotation.group_size``.
        bias: optional ``(out_features,)`` tensor.

    Returns:
        ``(rotated_weight, rotated_bias_or_None)``.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(weight.shape)}")
    g = int(rotation.group_size)
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    if out_features % g != 0:
        raise ValueError(
            f"producer out_features {out_features} not divisible by group_size {g}"
        )

    M = rotation.composed_matrix.to(device=weight.device, dtype=weight.dtype)
    # W has shape (out, in). View as (out/g, g, in) and left-multiply each
    # g-block by M: (g, g) @ (out/g, g, in) → (out/g, g, in).
    grouped = weight.reshape(out_features // g, g, in_features)
    rotated = M @ grouped
    new_weight = rotated.reshape(out_features, in_features)

    if bias is None:
        return new_weight, None

    if bias.dim() != 1 or int(bias.shape[0]) != out_features:
        raise ValueError(
            f"bias shape {tuple(bias.shape)} does not match out_features={out_features}"
        )
    bias_grouped = bias.reshape(out_features // g, g)
    bias_rotated = (M @ bias_grouped.unsqueeze(-1)).squeeze(-1)
    new_bias = bias_rotated.reshape(out_features)
    return new_weight, new_bias
