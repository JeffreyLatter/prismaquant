"""A routed expert stack may only be claimed by its halves, and the gate must know.

THE BUG THIS PINS. On a routed-MoE stack the per-role spelling is not an
alternative, it is the only one the ABI permits. A per-role LEARNED codebook
fits one book per ``(layer, projection)``, and a packed ``gate_up_proj`` target
can bind exactly one ``codebook_ref``, so such a layer *must* name
``…experts.gate_proj`` and ``…experts.up_proj`` separately. Lattice layers share
one book and legally name the packed stack, so both spellings coexist across
layers in one correct artifact.

`_fused_member_units` already bridged fused-unit to half-claims, but its only
fusion source was `profile.fused_sibling_leaf_mapping()` — vLLM's
`packed_modules_mapping`, which describes *dense* fusions. DeepseekV4 exposes no
vLLM architecture class at all, so that mapping is `{}` and its structure spec
declares no `fused_groups`; the bridge could never fire there. A correct
DSv4-Flash 92 GB artifact was rejected after its full ~82 GB export with
"11 scale-bearing weight(s) are claimed by no mechanism at all", one per learned
layer. (Recorded as task #14 in commit `1ccdf58`, which added the bridge and saw
8 stacks fail on the 112.69 GB artifact for the same reason.)

The fix falls back, for routed units only, to the profile's declarative
packed-expert decomposition — the same `packed_experts.projection_splits` the
exporter used to emit the halves, and the same table Gridbook keeps as
`_FUSED_FALLBACK` because DeepseekV4 gives it no `packed_modules_mapping`
either.

These tests exercise the seam with stub profiles: the property under test is
about which mapping is consulted for which kind of unit, not about any one
architecture. The EVERY-member rule is pinned here too, because relaxing it is
the way this fix would turn into a gate that passes an unservable artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from prismaquant.artifact_completeness import (
    ArtifactIncomplete,
    assert_artifact_complete,
    check_artifact_completeness,
)
import prismaquant.artifact_completeness as completeness


_PACKED = "model.layers.0.mlp.experts.gate_up_proj"
_GATE = "model.layers.0.mlp.experts.gate_proj"
_UP = "model.layers.0.mlp.experts.up_proj"


class _RoutedProfile:
    """A profile shaped like DeepseekV4Profile on the two axes that matter.

    No vLLM class, therefore no fused-sibling mapping; a declared packed-expert
    decomposition, therefore a routed w13 split. Identity name maps, so the
    namespace bridges stay out of the way of what is under test.
    """

    def __init__(self, *, fused_leaves=None):
        self._fused_leaves = fused_leaves or {}

    def fused_sibling_leaf_mapping(self):
        return dict(self._fused_leaves)

    def packed_expert_projection_names(self, param_name: str):
        if param_name == "gate_up_proj":
            return ("gate_proj", "up_proj")
        return (str(param_name),)

    def to_vllm_internal_name(self, name: str) -> str:
        return name

    def source_tensor_name(self, name: str) -> str:
        return name


def _write_artifact(root: Path, *, targets, tensors) -> None:
    """A minimal artifact: one safetensors shard header plus a quant_config."""

    root.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {}
    offset = 0
    for name, dtype, shape, span in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + span],
        }
        offset += span
    blob = json.dumps(header).encode("utf-8")
    with (root / "model.safetensors").open("wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\0" * offset)
    (root / "quant_config.json").write_text(json.dumps({
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": list(targets),
                "weights": {"num_bits": 8, "type": "float",
                            "strategy": "channel"},
                "input_activations": None,
            },
        },
        "ignore": [],
    }), encoding="utf-8")


def _packed_stack(prefix: str = _PACKED):
    """The physical tensors a routed CB expert stack ships."""

    return (
        (f"{prefix}.weight", "F8_E4M3", (256, 64, 32), 256 * 64 * 32),
        (f"{prefix}.weight_scale", "F32", (256, 64, 1), 256 * 64 * 4),
    )


@pytest.fixture()
def routed_profile(monkeypatch):
    def _install(**kwargs):
        profile = _RoutedProfile(**kwargs)
        monkeypatch.setattr(
            completeness, "_detect_profile_quietly", lambda _root: profile)
        return profile
    return _install


def test_both_halves_claim_the_routed_packed_stack(tmp_path, routed_profile):
    """The failure that rejected a correct 82 GB artifact."""

    routed_profile()
    _write_artifact(tmp_path, targets=[_GATE, _UP], tensors=_packed_stack())
    report = check_artifact_completeness(tmp_path, verbatim_prefixes=())
    assert report.ok, report.failure_text()
    assert _PACKED in report.cb_units
    assert not report.undeclared


def test_one_half_is_not_enough(tmp_path, routed_profile):
    """A half-claimed stack is a mixed-format fused group, i.e. unservable.

    This is the property that keeps the fix from being a gate that waves an
    unservable artifact through; Gridbook refuses the same partial declaration.
    """

    routed_profile()
    _write_artifact(tmp_path, targets=[_GATE], tensors=_packed_stack())
    with pytest.raises(ArtifactIncomplete, match="claimed by no mechanism"):
        assert_artifact_complete(tmp_path, verbatim_prefixes=())


def test_the_packed_spelling_still_claims_its_own_stack(tmp_path, routed_profile):
    """Lattice layers name the packed stack, and must keep working."""

    routed_profile()
    _write_artifact(tmp_path, targets=[_PACKED], tensors=_packed_stack())
    report = check_artifact_completeness(tmp_path, verbatim_prefixes=())
    assert report.ok, report.failure_text()


def test_a_dense_fusion_does_not_get_the_routed_fallback(tmp_path, routed_profile):
    """The fallback is routed-only, on purpose.

    `packed_expert_projection_names` answers about ROUTED experts. Letting it
    speak for a dense `mlp.gate_up_proj` would widen the gate on every
    architecture using a mapping that was never about dense modules — and dense
    fusions are exactly what `packed_modules_mapping` already covers.
    """

    routed_profile()
    dense = "model.layers.0.mlp.gate_up_proj"
    _write_artifact(
        tmp_path,
        targets=["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"],
        tensors=_packed_stack(dense),
    )
    with pytest.raises(ArtifactIncomplete, match="claimed by no mechanism"):
        assert_artifact_complete(tmp_path, verbatim_prefixes=())


def test_a_dense_fusion_is_claimed_when_vllm_declares_it(tmp_path, routed_profile):
    """...and the vLLM mapping still covers dense fusions, unchanged."""

    routed_profile(fused_leaves={"gate_up_proj": ("gate_proj", "up_proj")})
    dense = "model.layers.0.mlp.gate_up_proj"
    _write_artifact(
        tmp_path,
        targets=["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"],
        tensors=_packed_stack(dense),
    )
    report = check_artifact_completeness(tmp_path, verbatim_prefixes=())
    assert report.ok, report.failure_text()


def test_a_neighbouring_container_never_borrows_the_claim(tmp_path, routed_profile):
    """`experts2` is not `experts`; the routed test is on dotted boundaries."""

    routed_profile()
    other = "model.layers.0.mlp.experts2.gate_up_proj"
    _write_artifact(
        tmp_path,
        targets=["model.layers.0.mlp.experts2.gate_proj",
                 "model.layers.0.mlp.experts2.up_proj"],
        tensors=_packed_stack(other),
    )
    with pytest.raises(ArtifactIncomplete, match="claimed by no mechanism"):
        assert_artifact_complete(tmp_path, verbatim_prefixes=())
