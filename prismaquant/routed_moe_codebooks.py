"""Producer-side logical-role plan for learned routed-MoE codebooks.

Gridbook keeps the physical expert checkpoint tensors fused as ``w13``
(``gate_up_proj``), but the ABI attested by Gridbook 0.8.4 resolves ordinary logical
``gate_proj`` and ``up_proj`` config targets independently.  This module owns
that small producer naming bridge; it contains no Gridbook runtime code.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


# Immutable coverage of the completed DSv4 pooled-Lloyd burn.  Candidate
# legality remains the allocator/source-bpp policy's responsibility.
ROUTED_MOE_CBL_BANK_RUNGS = frozenset(range(28, 34))
_FORMAT_GROUP_PREFIX = "format_group_"


@dataclass(frozen=True)
class RoutedMoECodebookRole:
    """One logical book applied to a contiguous physical expert-row segment."""

    projection: str
    qname: str
    ref: str
    format_name: str
    codebook: Any
    col_weights: torch.Tensor
    output_rows: int
    member_qnames: tuple[str, ...]


def _packed_parent_and_suffix(packed_qname: str) -> tuple[str, str | None]:
    """Split an optional per-expert format-group discriminator.

    The split-stack producer names physical tensors as
    ``...gate_up_proj.format_group_fp8_cb_k28``.  Gridbook's logical role
    target preserves that final discriminator while replacing the packed
    parent immediately before it.
    """

    packed = str(packed_qname)
    if "." not in packed:
        return packed, None
    parent, leaf = packed.rsplit(".", 1)
    if leaf.startswith(_FORMAT_GROUP_PREFIX):
        return parent, leaf
    return packed, None


def logical_role_qname(packed_qname: str, projection: str) -> str:
    """Replace a packed expert leaf with one ordinary logical projection.

    An optional ``format_group_*`` suffix is retained.  This is the exact
    target namespace consumed by Gridbook's split routed-MoE declaration.
    """

    packed = str(packed_qname)
    projection = str(projection)
    packed_parent, discriminator = _packed_parent_and_suffix(packed)
    if (
        ".experts." not in packed_parent
        or not packed_parent.rsplit(".", 1)[-1]
    ):
        raise ValueError(f"{packed_qname!r} is not a routed expert target")
    if projection not in {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError(f"unsupported routed expert projection {projection!r}")
    logical = packed_parent.rsplit(".", 1)[0] + "." + projection
    return (
        logical
        if discriminator is None
        else logical + "." + discriminator
    )


def bundle_role_qname(packed_qname: str, projection: str) -> str:
    """Logical role cell name in the immutable bundle, without a subgroup.

    One learned book is trained per ``(layer, projection, rung)`` over the
    complete expert population.  A split format subgroup therefore reuses the
    same bundle cell and validates the participating experts through its
    per-member aliases; the format-group discriminator belongs only to the
    artifact/runtime target namespace.
    """

    packed_parent, _discriminator = _packed_parent_and_suffix(packed_qname)
    return logical_role_qname(packed_parent, projection)


def learned_role_qnames_for_packed(packed_qname: str) -> tuple[str, ...]:
    """Canonical Gridbook logical roles for a physical expert stack."""

    packed = str(packed_qname)
    packed_parent, _discriminator = _packed_parent_and_suffix(packed)
    if ".experts." not in packed_parent:
        return ()
    leaf = packed_parent.rsplit(".", 1)[-1]
    if leaf == "gate_up_proj":
        return (
            logical_role_qname(packed, "gate_proj"),
            logical_role_qname(packed, "up_proj"),
        )
    if leaf == "down_proj":
        return (logical_role_qname(packed, "down_proj"),)
    return ()


def stacked_role_col_weights(
    *,
    packed_qname: str,
    projection: str,
    member_qnames: Mapping[tuple[str, int], str],
    col_weights: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Return exact per-expert imatrix rows for one logical pooled book."""

    selected = sorted(
        (
            int(expert_id),
            str(member),
        )
        for (member_projection, expert_id), member in member_qnames.items()
        if str(member_projection) == str(projection)
    )
    if not selected:
        raise ValueError(
            f"{packed_qname}: no {projection} members for learned role book"
        )
    # A full bundle build is separately required to cover 0..E-1.  Export may
    # pass a format subgroup such as expert ids [1, 7, 19]; sorting here defines
    # the local packed order that the per-expert declaration records.
    rows: list[torch.Tensor] = []
    names: list[str] = []
    for _expert_id, member in selected:
        if member not in col_weights:
            raise ValueError(
                f"{packed_qname}: learned role member {member!r} has no "
                "col_weights entry"
            )
        rows.append(torch.as_tensor(col_weights[member]).reshape(1, -1))
        names.append(member)
    widths = {int(row.shape[-1]) for row in rows}
    if len(widths) != 1:
        raise ValueError(
            f"{packed_qname}: {projection} col-weight widths disagree: {widths}"
        )
    return torch.stack(rows).contiguous(), tuple(names)


def split_role_rows(
    weight: torch.Tensor,
    roles: tuple[RoutedMoECodebookRole, ...],
) -> tuple[tuple[RoutedMoECodebookRole, torch.Tensor], ...]:
    """Split a physical rank-3 expert stack by declared contiguous role rows."""

    value = torch.as_tensor(weight)
    if value.ndim != 3:
        raise ValueError(
            f"routed learned role packing needs rank-3 weight, got {value.shape}"
        )
    if not roles:
        raise ValueError("routed learned role packing has no logical roles")
    offset = 0
    result = []
    for role in roles:
        rows = int(role.output_rows)
        if rows <= 0:
            raise ValueError(f"{role.qname}: output_rows must be positive")
        result.append((role, value[:, offset:offset + rows, :]))
        offset += rows
    if offset != int(value.shape[1]):
        raise ValueError(
            "routed learned role rows do not cover the fused stack: "
            f"covered={offset}, physical={int(value.shape[1])}"
        )
    return tuple(result)


__all__ = [
    "ROUTED_MOE_CBL_BANK_RUNGS",
    "RoutedMoECodebookRole",
    "bundle_role_qname",
    "learned_role_qnames_for_packed",
    "logical_role_qname",
    "split_role_rows",
    "stacked_role_col_weights",
]
