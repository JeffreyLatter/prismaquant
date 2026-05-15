"""Shared fail-closed guard for archived expert pruning paths."""
from __future__ import annotations


EXPERT_PRUNE_DISABLED_MESSAGE = (
    "Expert pruning is disabled: the expert-drop path is archived "
    "and must not be used for shipping artifacts."
)


class ExpertPruneDisabledError(RuntimeError):
    """Raised when code attempts to generate or consume expert-prune data."""


def raise_expert_prune_disabled(context: str | None = None) -> None:
    prefix = f"{context}: " if context else ""
    raise ExpertPruneDisabledError(prefix + EXPERT_PRUNE_DISABLED_MESSAGE)
