"""`SpecMatchProfile` — a `ModelProfile` whose *detection* is also declarative.

Until now `matches()` was the one abstract method no JSON could express
(`base.py:57-61`), which is why "tier A — pure JSON" never happened: every
architecture needed at least a Python class to claim its own config. The
`match` block already existed in all nine spec files and `structure.py` already
parsed it; nothing read it, and one transcription (`qwen3_5_dense.json`) had
silently drifted out of agreement with its Python because dead config decays.

This module is the missing reader. A `SpecMatchProfile` binds one
`ModelStructureSpec` and answers `claims()` from `spec.match`; everything else
comes from the ordinary tier-2 spec path in `ModelProfile` (naming rules, fused
groups, packed experts, pinned names, lanes, …). It deliberately declares **no**
vLLM architecture class: tier-1 auto-derivation is a Python-profile facility,
so a spec-only architecture must declare its `fused_groups` and naming rules
outright. Behaviour that materialises modules or touches the forward pass (MTP
construction, streaming adapters, cross-layer state) stays Python — JSON is the
wrong medium for those, and `SpecMatchProfile` does not pretend otherwise.

`registry._resolve` builds one per spec file that no registered Python profile
claims, ordered by the spec's `priority`. While a Python profile of the same
name exists it wins outright, so this class is currently exercised by the
equivalence gate in `tests/test_spec_match_profile.py` rather than in
production — which is the point of the mitigation: the spec verdict must be
proven identical to the Python verdict for a full release before any
`matches()` body is deleted, one architecture at a time.
"""
from __future__ import annotations

from collections.abc import Iterable

from .base import ModelProfile
from .structure import ModelStructureSpec


class SpecMatchProfile(ModelProfile):
    """A concrete profile whose predicate and structure both come from JSON."""

    def __init__(self, spec: ModelStructureSpec) -> None:
        super().__init__()
        self._spec = spec
        # The spec is already in hand: short-circuit the name-keyed lookup in
        # `ModelProfile.structure_spec()` so this profile cannot bind a
        # different file than the one it was constructed from.
        self._structure_spec = spec
        self._structure_spec_loaded = True
        self.priority = int(spec.priority)

    # ------------------------------------------------------------
    # Identity + match
    # ------------------------------------------------------------
    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        """Never claims at class level.

        The predicate belongs to an *instance's* spec, so there is no
        meaningful class-level answer. Registry resolution asks instances via
        :meth:`claims`; returning False here keeps the abstract contract
        honest instead of guessing.
        """
        return False

    def claims(
        self,
        model_type: str | None,
        architectures: Iterable[str] | None,
    ) -> bool:
        """Evaluate this spec's `match` block against an HF config."""
        return self._spec.match.claims(model_type, architectures)

    @property
    def spec(self) -> ModelStructureSpec:
        """The bound spec (registry re-instantiates from it per resolution)."""
        return self._spec

    @property
    def name(self) -> str:
        return self._spec.id

    def vllm_architecture_class(self) -> str | None:
        # Tier 1 (vLLM `packed_modules_mapping` / `hf_to_vllm_mapper`
        # auto-derivation) is not spec-expressible; a spec-only architecture
        # declares `fused_groups` and `naming` instead.
        return None

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SpecMatchProfile({self._spec.id!r}, priority={self.priority})"
