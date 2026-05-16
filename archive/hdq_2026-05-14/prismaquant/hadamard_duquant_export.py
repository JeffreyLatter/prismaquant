"""Export-side application of Hadamard-DuQuant rotations.

Translates cached weights + cache state (loaded by
:mod:`prismaquant.hadamard_duquant_cache` from the joint-search outputs and the
allocator's per-cluster picks) into the artifact's runtime form:

  - **Per-Linear rotated weights** ready to write to safetensors:
      * Consumer side: ``W_artifact = W_cache @ M^T`` per ``G``-block on the
        input axis. This recovers ``Q(W @ M^T)`` from the cache convention
        ``W_eff = Q(W @ M^T) @ M``.
      * Producer side (offline-fold clusters only): ``W_artifact = M @ W_cache``
        per ``G``-block on the output axis. Bias gets the same rotation.

  - **transforms_config** JSON entries for runtime ONLINE clusters
    (``down_proj`` — SwiGLU breaks the offline-fold algebra). Emits one
    :class:`TransformScheme`-style dict per cluster with location ``input``
    targeting the consumer Linear. Dense research transforms store
    ``sqrt(G) * M^T`` because vLLM's generic online transform wrapper applies
    an internal ``1 / sqrt(G)`` normalization. Native Hadamard transforms use
    the same runtime normalization and do not emit a matrix tensor.

  - **Rotation safetensors entries**: one ``<target>.<scheme>_input.weight``
    entry per ONLINE dense target module, matching compressed-tensors'
    module-local transform parameter naming. Native Hadamard runtime
    transforms do not need these artifact entries.

For OFFLINE clusters (RESIDUAL, V_O, ATTN_OUT), both producer and consumer
weights are rotated at write time and **no transforms_config entry** is
emitted — the artifact is vanilla-vLLM-loadable with the rotation fully
folded into weights.

Algebra check (orthogonal M; subscripts denote per-G-block ops):

  Cache W_eff = Q(W @ M^T) @ M.
  Consumer artifact: W_artifact = W_eff @ M^T = Q(W @ M^T) @ M @ M^T = Q(W @ M^T).
  ONLINE runtime: y = (x @ M^T) @ Q(W @ M^T)^T ≈ x @ W^T (≈ from quant).
  OFFLINE producer output: y_p = x @ (M @ W_p)^T = x @ W_p^T @ M^T = (y_p_orig @ M^T).
  OFFLINE consumer: y_p_rotated @ Q(W_c @ M^T)^T = y_p_orig @ W_c^T (≈).

The exporter consumes :class:`HadamardDuQuantCacheState` directly so it
inherits the same source-of-truth that cache-fill used; there's no risk of
divergence between what the cache rendered and what the artifact ships.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

import torch.nn as nn

from prismaquant.hadamard_duquant import apply_block_rotation_input
from prismaquant.hadamard_duquant_cache import (
    CachedRotation,
    HadamardDuQuantCacheState,
    rotate_consumer_weight,
    rotate_producer_weight,
)


__all__ = [
    "TransformsConfigEntry",
    "apply_rotations_to_layer",
    "apply_rotations_to_model_in_place",
    "consumer_artifact_weight",
    "producer_artifact_weight",
    "build_transforms_config",
    "build_rotation_safetensors_entries",
    "vllm_incompatible_online_clusters",
    "assert_vllm_online_transforms_supported",
    "iter_consumer_rewrites",
    "iter_producer_rewrites",
    "state_fingerprint",
]


# ---------------------------------------------------------------------------
# Per-Linear artifact weight derivations
# ---------------------------------------------------------------------------


def consumer_artifact_weight(
    cache_weight: torch.Tensor,
    rotation: CachedRotation,
) -> torch.Tensor:
    """Recover the artifact weight for a consumer Linear.

    Inverts the cache's ``W_eff = Q(W @ M^T) @ M`` convention by applying
    ``M^T`` on the input axis per ``G``-block: ``W_artifact = Q(W @ M^T) =
    W_eff @ M^T``.

    Used at export time for every Linear listed in
    ``rotation.cluster_key``'s ``consumer_qnames``.
    """
    M_t = rotation.composed_matrix.to(
        device=cache_weight.device, dtype=cache_weight.dtype
    ).t()
    return apply_block_rotation_input(cache_weight, M_t).contiguous()


def producer_artifact_weight(
    cache_weight: torch.Tensor,
    rotation: CachedRotation,
    *,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the offline-fold output-axis rotation to a producer Linear.

    Producer Linears in OFFLINE clusters have their output axis transformed
    so the residual / V→O / attn-out_proj stream feeds the next consumer in
    the rotated basis. Returns ``(M @ W_cache per G-block, M @ bias per G-block)``.

    Only meaningful for OFFLINE clusters (``rotation.online is False``);
    ONLINE clusters have no producer Linear (SwiGLU is the producer).
    """
    new_w, new_b = rotate_producer_weight(cache_weight, rotation, bias=bias)
    return new_w.contiguous(), (new_b.contiguous() if new_b is not None else None)


# ---------------------------------------------------------------------------
# Per-Linear rewrite iterators — what the exporter loops over
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumerRewrite:
    """One consumer Linear's rotation rewrite at export time."""
    qname: str
    cluster_key: str
    format_label: str
    rotation: CachedRotation


@dataclass(frozen=True)
class ProducerRewrite:
    """One producer Linear's rotation rewrite at export time."""
    qname: str
    cluster_key: str
    format_label: str
    rotation: CachedRotation


def iter_consumer_rewrites(
    state: HadamardDuQuantCacheState,
) -> Iterable[ConsumerRewrite]:
    """Yield one :class:`ConsumerRewrite` per consumer Linear in any
    rotated cluster (both ONLINE and OFFLINE).

    The exporter applies :func:`consumer_artifact_weight` to each yielded
    Linear's cached weight before writing the safetensors.
    """
    for qname, cluster_key in sorted(state.consumer_to_cluster.items()):
        rotation = state.rotations_by_cluster.get(cluster_key)
        if rotation is None:
            continue
        yield ConsumerRewrite(
            qname=qname,
            cluster_key=cluster_key,
            format_label=rotation.format_label,
            rotation=rotation,
        )


def iter_producer_rewrites(
    state: HadamardDuQuantCacheState,
) -> Iterable[ProducerRewrite]:
    """Yield one :class:`ProducerRewrite` per producer Linear in any
    OFFLINE rotated cluster.

    ONLINE clusters have no producers (their producer is SwiGLU, not a
    Linear) — they're skipped here.
    """
    for qname, cluster_key in sorted(state.producer_to_cluster.items()):
        rotation = state.rotations_by_cluster.get(cluster_key)
        if rotation is None or rotation.online:
            continue
        yield ProducerRewrite(
            qname=qname,
            cluster_key=cluster_key,
            format_label=rotation.format_label,
            rotation=rotation,
        )


# ---------------------------------------------------------------------------
# transforms_config emission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformsConfigEntry:
    """One :class:`TransformScheme`-shaped entry the exporter writes.

    Renders to JSON-serializable form via :meth:`to_dict`. The exporter
    aggregates these into ``config.json``'s ``transforms_config`` field.
    """
    config_group_name: str
    type: str
    head_dim: int
    apply: list[dict[str, Any]]
    requires_grad: bool = False
    randomize: bool = False
    precision: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "head_dim": int(self.head_dim),
            "apply": [dict(a) for a in self.apply],
            "requires_grad": bool(self.requires_grad),
            "randomize": bool(self.randomize),
            "precision": str(self.precision),
        }


def _transform_scheme_name(cluster_key: str) -> str:
    """Canonical config_groups key for one rotated cluster.

    Cluster keys may contain dots and other characters that are awkward in
    JSON config_groups identifiers; replace dots with underscores. The
    safetensors key keeps the original dotted form via the ``apply.targets``
    list and the runtime ``input_transform`` lookup uses the targets, not
    this name — it's just a JSON map key.
    """
    return "hadamard_duquant__" + cluster_key.replace(".", "__").replace("/", "__")


def build_transforms_config(
    state: HadamardDuQuantCacheState,
    *,
    qname_mapper: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Build the ``transforms_config`` dict for ``config.json``.

    Emits one entry per ONLINE cluster. OFFLINE clusters fold into weights
    at export time and don't need a runtime transform.

    Returns a dict shaped like compressed-tensors'
    :class:`TransformConfig`:

    .. code-block:: python

        {
            "config_groups": {
                "hadamard_duquant__<cluster>": {
                    "type": "hadamard",
                    "head_dim": <runtime_head_dim>,
                    "apply": [
                        {
                            "targets": ["<consumer_qname>"],
                            "location": "input",
                            "inverse": False,
                            "ignore": [],
                        }
                    ],
                    ...
                },
                ...
            }
        }

    The artifact's ``config.json`` should set ``transforms_config`` to the
    returned dict (or omit the field entirely when the state is empty —
    callers handle the empty case so vanilla artifacts stay vanilla).
    """
    map_qname = qname_mapper or (lambda qname: qname)
    config_groups: dict[str, dict[str, Any]] = {}
    for cluster_key, rotation in sorted(state.rotations_by_cluster.items()):
        if not rotation.online:
            continue  # offline clusters are folded into weights
        # Find consumer qnames for this cluster.
        consumers = sorted(
            map_qname(qname)
            for qname, ckey in state.consumer_to_cluster.items()
            if ckey == cluster_key
        )
        if not consumers:
            continue
        entry = TransformsConfigEntry(
            config_group_name=_transform_scheme_name(cluster_key),
            type=str(rotation.runtime_transform_type),
            head_dim=int(rotation.group_size),
            apply=[
                {
                    "targets": consumers,
                    "location": "input",
                    "inverse": False,
                    "ignore": [],
                }
            ],
        )
        config_groups[entry.config_group_name] = entry.to_dict()
    return {"config_groups": config_groups}


# ---------------------------------------------------------------------------
# Rotation safetensors entries — runtime-loaded matrices
# ---------------------------------------------------------------------------


def state_fingerprint(state: HadamardDuQuantCacheState) -> str:
    """Short stable hash of the rotation state, for export-cache invalidation.

    Includes cluster keys, format labels, and the bytes of each composed
    matrix. Changes to the state — different picks, different solver
    output, different cluster set — invalidate cached per-layer exports.
    Empty state returns ``"none"`` so cached layers from non-rotation runs
    remain valid.
    """
    import hashlib

    if state.is_empty():
        return "none"
    h = hashlib.sha256()
    for cluster_key in sorted(state.rotations_by_cluster.keys()):
        rot = state.rotations_by_cluster[cluster_key]
        h.update(cluster_key.encode("utf-8"))
        h.update(b"|")
        h.update(rot.format_label.encode("utf-8"))
        h.update(b"|")
        h.update(str(int(rot.group_size)).encode("utf-8"))
        h.update(b"|")
        h.update(b"online" if rot.online else b"offline")
        h.update(b"|")
        h.update(str(rot.runtime_transform_type).encode("utf-8"))
        h.update(b"|")
        h.update(
            rot.composed_matrix.detach().cpu().contiguous()
            .to(torch.float32).numpy().tobytes()
        )
        h.update(b"\n")
    return h.hexdigest()[:16]


def apply_rotations_to_layer(
    model: nn.Module,
    layer_qname: str,
    state: HadamardDuQuantCacheState,
) -> dict[str, int]:
    """Per-layer scope of :func:`apply_rotations_to_model_in_place`.

    Filters the state's consumer/producer indices to qnames inside
    ``layer_qname`` (plus the dot separator) and applies rotations only
    to those Linears. The streaming exporter calls this once per
    decoder layer in the materialization loop — earlier/later layers'
    weights are still on meta and would error if we tried to mutate
    them.

    Returns ``{"consumer": int, "producer": int}`` counts.
    """
    counts = {"consumer": 0, "producer": 0}
    if state.is_empty():
        return counts
    prefix = layer_qname + "." if not layer_qname.endswith(".") else layer_qname

    # Consumer pass
    for qname, cluster_key in state.consumer_to_cluster.items():
        if not qname.startswith(prefix):
            continue
        rotation = state.rotations_by_cluster.get(cluster_key)
        if rotation is None:
            continue
        module = _resolve_module(model, qname)
        if module is None or not hasattr(module, "weight"):
            continue
        with torch.no_grad():
            new_w = rotate_consumer_weight(
                module.weight.detach(), rotation
            )
            module.weight.data.copy_(new_w.to(module.weight.dtype))
        counts["consumer"] += 1

    # Producer pass (offline clusters only — iter filtering implicit via
    # rotation.online check)
    for qname, cluster_key in state.producer_to_cluster.items():
        if not qname.startswith(prefix):
            continue
        rotation = state.rotations_by_cluster.get(cluster_key)
        if rotation is None or rotation.online:
            continue
        module = _resolve_module(model, qname)
        if module is None or not hasattr(module, "weight"):
            continue
        with torch.no_grad():
            bias = (
                module.bias.detach()
                if getattr(module, "bias", None) is not None
                else None
            )
            new_w, new_b = producer_artifact_weight(
                module.weight.detach(), rotation, bias=bias,
            )
            module.weight.data.copy_(new_w.to(module.weight.dtype))
            if new_b is not None and module.bias is not None:
                module.bias.data.copy_(new_b.to(module.bias.dtype))
        counts["producer"] += 1

    return counts


def _resolve_module(model: nn.Module, qname: str) -> nn.Module | None:
    """Resolve a dotted qname to a submodule of ``model``, or None if missing."""
    parts = [p for p in qname.split(".") if p]
    cur: object = model
    for p in parts:
        try:
            if p.isdigit():
                cur = cur[int(p)]  # type: ignore[index]
            else:
                cur = getattr(cur, p)
        except (AttributeError, IndexError, KeyError, TypeError):
            return None
    return cur if isinstance(cur, nn.Module) else None


def apply_rotations_to_model_in_place(
    model: nn.Module,
    state: HadamardDuQuantCacheState,
) -> dict[str, int]:
    """Pre-rotate model Linear weights so the exporter quantizes the rotated form.

    Called by the pipeline immediately before the export step. After this
    pass:
      - Each consumer Linear's weight is replaced with ``W @ M^T`` per
        ``G``-block on the input axis. The exporter then quantizes
        ``Q(W @ M^T)`` directly — no separate "consumer recovery" step is
        needed on the safetensors output.
      - Each producer Linear in an OFFLINE cluster has its weight replaced
        with ``M @ W_p`` per ``G``-block on the output axis. Bias gets the
        same rotation. The exporter quantizes the rotated form.
      - Linears in ONLINE clusters need both the rotated weight (done here)
        AND a runtime ``input_transform`` entry (emitted via
        :func:`build_transforms_config`).

    The transformation is destructive on the model — pass a model that's
    only used for export, or snapshot weights beforehand if you'll need the
    unrotated form again.

    Args:
        model: live HF model; Linear weights will be rewritten in place.
        state: rotation state loaded via :func:`load_cache_state` from the
            joint-search outputs + allocator picks.

    Returns:
        Per-role count summary ``{"consumer": int, "producer": int}`` —
        useful for the pipeline to log how many Linears were rotated.
    """
    counts = {"consumer": 0, "producer": 0}
    if state.is_empty():
        return counts

    # Consumer pass — both ONLINE and OFFLINE clusters need the input-axis
    # rotation applied to the weight.
    for rewrite in iter_consumer_rewrites(state):
        module = _resolve_module(model, rewrite.qname)
        if module is None or not hasattr(module, "weight"):
            continue
        with torch.no_grad():
            new_w = rotate_consumer_weight(
                module.weight.detach(), rewrite.rotation
            )
            module.weight.data.copy_(new_w.to(module.weight.dtype))
        counts["consumer"] += 1

    # Producer pass — OFFLINE clusters only (iter_producer_rewrites already
    # filters online clusters out). Bias is rotated too if present.
    for rewrite in iter_producer_rewrites(state):
        module = _resolve_module(model, rewrite.qname)
        if module is None or not hasattr(module, "weight"):
            continue
        with torch.no_grad():
            bias = (
                module.bias.detach()
                if getattr(module, "bias", None) is not None
                else None
            )
            new_w, new_b = producer_artifact_weight(
                module.weight.detach(), rewrite.rotation, bias=bias,
            )
            module.weight.data.copy_(new_w.to(module.weight.dtype))
            if new_b is not None and module.bias is not None:
                module.bias.data.copy_(new_b.to(module.bias.dtype))
        counts["producer"] += 1

    return counts


def _safetensors_key_for_runtime(cluster_key: str, target_qname: str) -> str:
    """Canonical safetensors key for the runtime ONLINE transform matrix.

    compressed-tensors installs the transform parameter on each target
    module using ``<scheme_name>_<location>.weight``. Therefore the state
    dict key is ``<target_qname>.<scheme_name>_input.weight``.
    """
    return f"{target_qname}.{_transform_scheme_name(cluster_key)}_input.weight"


def build_rotation_safetensors_entries(
    state: HadamardDuQuantCacheState,
    *,
    qname_mapper: Callable[[str], str] | None = None,
) -> dict[str, torch.Tensor]:
    """Build the safetensors entries the artifact ships for ONLINE clusters.

    For each ONLINE dense/random-matrix cluster target, writes
    ``sqrt(G) * M^T`` as the stored transform weight. vLLM's generic online
    transform wrapper multiplies dense transforms by ``1 / sqrt(G)`` to
    mirror native Hadamard normalization, so the net activation transform is
    ``x @ M^T`` — the rotation required to make the artifact's quantized
    weight ``Q(W @ M^T)`` reproduce the original computation.

    Returns ``{safetensors_key: tensor}`` with CPU contiguous tensors.
    OFFLINE clusters are not included because their rotations are already
    folded into the per-Linear weights elsewhere. Native Hadamard runtime
    clusters are also skipped because vLLM generates that transform from
    ``type="hadamard"`` + ``head_dim``.
    """
    map_qname = qname_mapper or (lambda qname: qname)
    entries: dict[str, torch.Tensor] = {}
    for cluster_key, rotation in sorted(state.rotations_by_cluster.items()):
        if not rotation.online:
            continue
        if rotation.runtime_transform_type == "hadamard":
            continue
        consumers = sorted(
            map_qname(qname)
            for qname, ckey in state.consumer_to_cluster.items()
            if ckey == cluster_key
        )
        if not consumers:
            continue
        runtime_weight = (
            rotation.composed_matrix.detach().cpu().contiguous().t().contiguous()
            * (float(rotation.group_size) ** 0.5)
        )
        for target_qname in consumers:
            key = _safetensors_key_for_runtime(cluster_key, target_qname)
            entries[key] = runtime_weight.clone()
    return entries


def vllm_incompatible_online_clusters(
    state: HadamardDuQuantCacheState,
) -> list[str]:
    """Return ONLINE clusters that current vLLM NVFP4 cannot serve.

    The deployment-safe online path is the native Hadamard transform: vLLM
    routes it through the dedicated hadacore transform wrapper. Dense learned
    ``random-matrix`` online transforms can be made to load with a local
    Qutlass-selector patch, but they run through a generic dense GEMM wrapper
    and are not production-performant on NVFP4 H16. Keep them behind
    ``allow_unsupported`` for research artifacts only.
    """
    bad: list[str] = []
    for cluster_key, rotation in sorted(state.rotations_by_cluster.items()):
        if not rotation.online or rotation.format_label.upper() != "NVFP4":
            continue
        if rotation.runtime_transform_type != "hadamard":
            bad.append(cluster_key)
            continue
    return bad


def assert_vllm_online_transforms_supported(
    state: HadamardDuQuantCacheState,
    *,
    allow_unsupported: bool = False,
) -> None:
    bad = vllm_incompatible_online_clusters(state)
    if not bad or allow_unsupported:
        return
    sample = ", ".join(bad[:6])
    more = "" if len(bad) <= 6 else f", ... (+{len(bad) - 6})"
    raise RuntimeError(
        "Hadamard-DuQuant ONLINE NVFP4 rotations are not production-safe "
        "for this vLLM compressed-tensors runtime. Use native Hadamard "
        "online rotations for deployment, use "
        "HADAMARD_DUQUANT_ROTATION_SCOPE=folded_only to avoid online "
        "transforms, or set PRISMAQUANT_ALLOW_UNSUPPORTED_HDQ_ONLINE=1 "
        "only when deliberately emitting a research artifact. "
        f"Incompatible clusters: {sample}{more}"
    )
