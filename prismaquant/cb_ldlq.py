"""Activation-cache adapter for production CB LDLQ assignment.

This is a reader over the established probe cache, not a second cache. Dense
and per-expert Linears reuse their captured input rows directly; packed-MoE
stages use the existing checkpoint replay in :mod:`prismaquant.moe_imatrix`.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import torch


_ACT_FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")


class CBLDLQActivationLoader:
    """Load one target's Hessian rows lazily from the production act cache."""

    def __init__(
        self,
        activation_cache_dir: str | Path,
        *,
        model_dir: str | Path,
        profile,
        expert_stack_members: Mapping[
            str, Mapping[tuple[str, int], str]
        ] | None = None,
        replay_device: str | None = None,
    ) -> None:
        self.activation_cache_dir = Path(activation_cache_dir)
        self.model_dir = Path(model_dir)
        self.profile = profile
        self.expert_stack_members = dict(expert_stack_members or {})
        self.replay_device = replay_device

    def _direct(self, qname: str) -> torch.Tensor | None:
        path = self.activation_cache_dir / (
            _ACT_FNAME_SUB.sub("__", str(qname)) + ".pt"
        )
        if not path.is_file():
            return None
        blob = torch.load(path, map_location="cpu", weights_only=False)
        value = blob.get("inputs") if isinstance(blob, dict) else None
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(
                f"{qname}: LDLQ activation cache entry has no rank-2 inputs"
            )
        return value.detach().to(torch.float32).contiguous()

    def _per_expert(self, qname: str) -> tuple[torch.Tensor, ...] | None:
        members = self.expert_stack_members.get(qname)
        if not members:
            return None
        by_expert: dict[int, list[str]] = {}
        for (_projection, expert), member in sorted(members.items()):
            by_expert.setdefault(int(expert), []).append(str(member))
        rows: list[torch.Tensor] = []
        for expert in sorted(by_expert):
            candidates = [self._direct(name) for name in by_expert[expert]]
            candidates = [value for value in candidates if value is not None]
            if not candidates:
                raise ValueError(
                    f"{qname}: expert {expert} has no LDLQ activation rows"
                )
            reference = candidates[0]
            if any(
                tuple(value.shape) != tuple(reference.shape)
                or not torch.equal(value, reference)
                for value in candidates[1:]
            ):
                raise ValueError(
                    f"{qname}: fused expert {expert} members disagree on "
                    "their captured LDLQ input rows"
                )
            rows.append(reference)
        return tuple(rows)

    def load(
        self,
        qname: str,
        *,
        stack_size: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        direct = self._direct(qname)
        if direct is not None:
            return direct
        per_expert = self._per_expert(qname)
        if per_expert is not None:
            if stack_size is not None and len(per_expert) != int(stack_size):
                raise ValueError(
                    f"{qname}: LDLQ activation stacks={len(per_expert)} != "
                    f"weight stacks={stack_size}"
                )
            return per_expert

        from .moe_imatrix import (
            RoutedActivationSamples,
            synthesize_packed_expert_activation_samples,
        )

        replayed = synthesize_packed_expert_activation_samples(
            self.model_dir,
            self.activation_cache_dir,
            {str(qname)},
            self.profile,
            device=self.replay_device,
        ).get(str(qname))
        if isinstance(replayed, RoutedActivationSamples):
            if stack_size is None:
                return replayed.values
            rows = tuple(
                replayed.values[replayed.expert_indices == expert].contiguous()
                for expert in range(int(stack_size))
            )
            if any(int(value.shape[0]) == 0 for value in rows):
                missing = [i for i, value in enumerate(rows) if not value.shape[0]]
                raise ValueError(
                    f"{qname}: routed LDLQ replay has no rows for experts "
                    f"{missing[:8]}"
                )
            return rows
        if isinstance(replayed, torch.Tensor) and replayed.ndim == 2:
            return replayed.detach().to(torch.float32).contiguous()
        raise ValueError(
            f"{qname}: no value-bearing activation rows for LDLQ export; "
            "rebuild the production activation cache"
        )


__all__ = ["CBLDLQActivationLoader"]
