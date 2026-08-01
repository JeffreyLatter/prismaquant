"""Canonical producer-side Gridbook CB artifact configuration emitter.

The resident and streaming exporters differ in how they discover and write
weights, but they serialize the same CB scheme and ``quant_config.json``
contract.  Keep that contract here so a new layout field cannot land in one
exporter while silently being omitted from the other.

Exporter-specific namespace and provenance choices are explicit parameters:
callers provide the CB/delegated target-name mappers, identify weight-only
stock targets, and select the provenance fields that intentionally differ.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from typing import Any

import torch

from prismaquant.cb_layout import (
    FP4_GROUP,
    SCALE_CODING_TWO_TIER,
    SCALE_CODING_V1,
    SUPERBLOCK,
    VEC_DIM,
    codebook_subtable_shapes,
    family_for,
    parse_format_name,
    type_size,
)
from prismaquant.export_native_compressed import (
    FP8_E4M3_SCHEME,
    FP8_SOURCE_SCHEME,
    NVFP4_SCHEME,
    _explicit_regex,
)


CBTarget = tuple[str, str, int]
TargetName = Callable[[str], str]
_STOCK_CT_SCHEMES = {
    "NVFP4": NVFP4_SCHEME,
    "FP8_E4M3": FP8_E4M3_SCHEME,
}


def _validated_codebook_sequence(
    fmt: str,
    codebook: object,
) -> tuple[torch.Tensor, ...]:
    """Return tensors only after exact canonical sidecar-shape validation."""

    parsed = parse_format_name(fmt)
    if parsed is None:
        raise ValueError(f"not a producer CB format: {fmt!r}")
    family, k = parsed
    expected = codebook_subtable_shapes(k, family.mode, family.n_sub)
    if isinstance(codebook, torch.Tensor):
        tensors: tuple[object, ...] = (codebook,)
    elif isinstance(codebook, (tuple, list)):
        tensors = tuple(codebook)
    else:
        raise TypeError(
            f"{fmt} codebook must be a tensor or tensor sequence, got "
            f"{type(codebook).__name__}"
        )
    if len(tensors) != len(expected):
        raise ValueError(
            f"{fmt} requires {len(expected)} codebook subtables, got "
            f"{len(tensors)}"
        )
    validated: list[torch.Tensor] = []
    for index, (tensor, expected_shape) in enumerate(
        zip(tensors, expected, strict=True)
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{fmt} codebook subtable {index} must be a torch.Tensor, "
                f"got {type(tensor).__name__}"
            )
        if tensor.ndim != 2:
            raise ValueError(
                f"{fmt} codebook subtable {index} must have rank 2, got "
                f"rank {tensor.ndim}"
            )
        actual_shape = tuple(int(dim) for dim in tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{fmt} codebook subtable {index} shape {actual_shape} does "
                f"not match canonical shape {expected_shape}"
            )
        validated.append(tensor)
    return tuple(validated)


def _codebook_names_for_count(
    ref: str,
    fmt: str,
    count: int,
) -> tuple[str, ...]:
    base = f"cb_codebook.{ref}.{fmt}"
    if count > 1:
        return tuple(f"{base}.sub{i}" for i in range(count))
    return (base,)


def codebook_tensor_names(
    ref: str,
    fmt: str,
    codebook: object,
) -> tuple[str, ...]:
    """Physical sidecar tensor names for one resolved codebook."""

    tensors = _validated_codebook_sequence(fmt, codebook)
    return _codebook_names_for_count(ref, fmt, len(tensors))


def codebook_tensors(
    ref: str,
    fmt: str,
    codebook: object,
) -> dict[str, torch.Tensor]:
    """Serialize one codebook under its canonical sidecar tensor names."""

    tensors = _validated_codebook_sequence(fmt, codebook)
    names = _codebook_names_for_count(ref, fmt, len(tensors))
    return {
        name: tensor.to(torch.float16).cpu().contiguous()
        for name, tensor in zip(names, tensors, strict=True)
    }


def _two_tier_scale_coding() -> dict[str, Any]:
    # The table generator is shared with the encoder so the self-describing
    # config cannot drift from the exact E4M3 values used for packing.
    from prismaquant.nvfp4_cb_formats import (
        TWO_TIER_SUPER_BIAS,
        _two_tier_tables,
    )

    table, _, _ = _two_tier_tables("cpu")
    return {
        "kind": "two_tier",
        "sub_bits": 4,
        "super_bias": TWO_TIER_SUPER_BIAS,
        "table": [float(value) for value in table.tolist()],
    }


def build_cb_scheme(
    *,
    ref: str,
    fmt: str,
    grid: str,
    mode: str,
    k: int,
    codebook: object,
    scale_coding: str,
) -> dict[str, Any]:
    """Build the canonical scheme for one CB target/group.

    Layout identity comes from :mod:`prismaquant.cb_layout`; the actual
    sidecar object is checked against the family's required subtable count.
    FP8 has no serialized scale plane and therefore always carries the v1
    layout identity regardless of the exporter's FP4 scale-coding selection.
    """

    grid = str(grid).lower()
    mode = str(mode).lower()
    k = int(k)
    family = family_for(grid, mode)
    parsed = parse_format_name(fmt)
    if parsed is None or parsed[0] != family or parsed[1] != k:
        raise ValueError(
            f"CB format/fields disagree: {fmt!r} vs "
            f"grid={grid!r}, mode={mode!r}, k={k}"
        )
    tensors = _validated_codebook_sequence(fmt, codebook)
    names = _codebook_names_for_count(ref, fmt, len(tensors))
    coding = scale_coding if grid == "fp4" else SCALE_CODING_V1
    scheme: dict[str, Any] = {
        "grid": grid,
        "mode": mode,
        "k": k,
        "superblock": SUPERBLOCK,
        "group_size": FP4_GROUP if grid == "fp4" else 0,
        "vec_dim": VEC_DIM,
        "n_sub": family.n_sub,
        "type_size": type_size(k, grid, coding),
        "act_bits": 4 if grid == "fp4" else 8,
        "codebook_source": "lattice" if ref == "lattice" else "learned",
        "codebook_ref": list(names) if len(names) > 1 else names[0],
        "codebook_group": None if ref == "lattice" else ref,
    }
    if coding == SCALE_CODING_TWO_TIER:
        scheme["scale_coding"] = _two_tier_scale_coding()
    return scheme


def cb_scheme_reuse_signature(scheme: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the scheme fields that make a packed CB tensor reusable."""

    scale_coding = scheme.get("scale_coding")
    coding = (
        SCALE_CODING_TWO_TIER
        if isinstance(scale_coding, Mapping)
        and scale_coding.get("kind") == "two_tier"
        else SCALE_CODING_V1
    )
    return {
        "grid": scheme.get("grid"),
        "mode": scheme.get("mode"),
        "k": scheme.get("k"),
        "n_sub": scheme.get("n_sub"),
        "type_size": scheme.get("type_size"),
        "codebook_ref": scheme.get("codebook_ref"),
        "scale_coding": coding,
    }


def _identity_target(qname: str) -> str:
    return qname


def build_quant_config(
    *,
    assignment: Mapping[str, str],
    cb_targets: Mapping[str, CBTarget],
    source_targets: Iterable[str],
    stock_targets: Mapping[str, str],
    by_group: Mapping[tuple[str, str], Sequence[str]],
    codebooks: Mapping[tuple[str, str], object],
    col_weights: Mapping[str, torch.Tensor],
    codebook_tensors_by_name: Mapping[str, torch.Tensor],
    ignore: Iterable[str],
    codebook_file: str | None,
    scale_coding: str,
    codebook_source: str,
    serialized_payload_summary: Mapping[str, Any],
    serialization_context: object,
    cb_render_identity: Mapping[str, Any] | None,
    git_commit: str,
    cb_target_name: TargetName = _identity_target,
    delegated_target_name: TargetName = _identity_target,
    source_target_name: TargetName = _identity_target,
    weight_only_stock_targets: Iterable[str] = (),
    streaming_provenance: bool | None = None,
    include_tensor_formats: bool = False,
) -> dict[str, Any]:
    """Build the complete producer-owned ``quant_config.json`` payload."""

    source_targets = list(source_targets)
    weight_only_stock_targets = set(weight_only_stock_targets)
    assignment_sha = hashlib.sha256(
        json.dumps(
            dict(sorted(assignment.items())),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    imatrix_hasher = hashlib.sha256()
    for qname in sorted(col_weights):
        imatrix_hasher.update(qname.encode())
        imatrix_hasher.update(
            col_weights[qname].to(torch.float32).cpu().numpy().tobytes()
        )
    codebook_sha = {
        name: hashlib.sha256(
            tensor.to(torch.float16).cpu().numpy().tobytes()
        ).hexdigest()
        for name, tensor in codebook_tensors_by_name.items()
    }

    config_groups: dict[str, dict[str, Any]] = {}
    for group_index, ((ref, fmt), qnames) in enumerate(
        sorted(by_group.items())
    ):
        grid, mode, k = cb_targets[qnames[0]]
        scheme = build_cb_scheme(
            ref=ref,
            fmt=fmt,
            grid=grid,
            mode=mode,
            k=k,
            codebook=codebooks[(ref, fmt)],
            scale_coding=scale_coding,
        )
        config_groups[f"group_{group_index}"] = {
            "targets": sorted(cb_target_name(qname) for qname in qnames),
            "format": fmt,
            "scheme": scheme,
        }

    stock_by_group: dict[tuple[str, bool], list[str]] = {}
    for qname, fmt in stock_targets.items():
        stock_by_group.setdefault(
            (fmt, qname in weight_only_stock_targets), []
        ).append(qname)
    for (fmt, weight_only), qnames in sorted(stock_by_group.items()):
        group = deepcopy(_STOCK_CT_SCHEMES[fmt])
        if weight_only:
            group["input_activations"] = None
        group["targets"] = sorted(
            _explicit_regex(delegated_target_name(qname))
            for qname in qnames
        )
        config_groups[f"group_{len(config_groups)}"] = group
    if source_targets:
        source_group = deepcopy(FP8_SOURCE_SCHEME)
        source_group["targets"] = sorted(
            _explicit_regex(source_target_name(qname))
            for qname in source_targets
        )
        config_groups[f"group_{len(config_groups)}"] = source_group

    provenance: dict[str, Any] = {
        "git_commit": git_commit,
        "assignment_sha256": assignment_sha,
        "imatrix_sha256": imatrix_hasher.hexdigest(),
        "codebook_sha256": codebook_sha,
        "codebook_source": codebook_source,
        "scale_sweep": bool(getattr(serialization_context, "scale_sweep")),
        "encode_tier": getattr(serialization_context, "encode_tier"),
        "renderer_abi": getattr(serialization_context, "renderer_abi"),
        "scale_coding": scale_coding,
        "cb_targets": len(cb_targets),
        "stock_ct_targets": len(stock_targets),
        "fp8_source_targets": len(source_targets),
        "serialized_payload": dict(serialized_payload_summary),
        "render_identity_verified": cb_render_identity is not None,
    }
    if streaming_provenance is not None:
        provenance["streaming"] = bool(streaming_provenance)
    if cb_render_identity is not None:
        provenance["cb_render_identity"] = cb_render_identity
    if include_tensor_formats:
        provenance["tensor_formats"] = {
            qname: assignment[qname]
            for qname in sorted(
                set(cb_targets) | set(stock_targets) | set(source_targets)
            )
        }

    quant_config: dict[str, Any] = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "config_groups": config_groups,
        "ignore": sorted(set(ignore)),
        **({"codebook_file": codebook_file} if codebook_file else {}),
        "provenance": provenance,
    }
    if scale_coding == SCALE_CODING_TWO_TIER:
        # Missing layout_version remains the permanent v1 compatibility rule.
        quant_config["layout_version"] = 2
    return quant_config


__all__ = [
    "build_cb_scheme",
    "build_quant_config",
    "cb_scheme_reuse_signature",
    "codebook_tensor_names",
    "codebook_tensors",
]
