"""The completeness gate must read a config-group target in the namespace the
exporter actually wrote it in.

THE BUG THIS PINS. A DELEGATED config group is spelled in vLLM's module
namespace, because compressed-tensors matches its targets against vLLM's module
tree at load: `export_nvfp4_cb_streaming._delegated_target_name` is
`profile.to_vllm_internal_name`. On every architecture this gate had seen, that
map was the identity, so the difference never showed. It is not the identity on
a multimodal wrapper — Qwen3.8-27B stores `lm_head.weight` while vLLM builds the
head at `language_model.lm_head` — and the gate rejected a *correct* 12.98 GB
artifact at the end of a 50-minute export with "1 scale-bearing weight(s) are
claimed by no mechanism at all: ['lm_head']".

The fix maps the UNIT forward through the profile rather than inverting the
claim, because `to_vllm_internal_name` is the producer's own map and has no
inverse to call. These tests exercise that seam directly with a stub profile:
the real profile needs a full checkpoint to detect, and the property under test
is about the namespaces, not about any one architecture.
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


class _WrapperProfile:
    """The two maps a multimodal wrapper profile provides, and nothing else.

    `lm_head` keeps its checkpoint spelling but moves under `language_model` in
    vLLM's tree; the body keeps `model.` on disk and gains the wrapper prefix in
    vLLM. Both are taken from the real Qwen3_5DenseProfile's answers.
    """

    def to_vllm_internal_name(self, name: str) -> str:
        if name == "lm_head":
            return "language_model.lm_head"
        if name.startswith("model.language_model."):
            return "language_model.model." + name[len("model.language_model."):]
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


_LM_HEAD = (
    ("lm_head.weight", "F8_E4M3", (32, 8), 32 * 8),
    ("lm_head.weight_scale", "F32", (32, 1), 32 * 4),
)


@pytest.fixture()
def wrapper_profile(monkeypatch):
    monkeypatch.setattr(
        completeness, "_detect_profile_quietly",
        lambda _root: _WrapperProfile())


def test_vllm_spelled_group_target_claims_its_checkpoint_unit(
        tmp_path, wrapper_profile):
    """The exact shape of the Qwen3.8-27B artifact-A failure."""

    root = tmp_path / "artifact"
    _write_artifact(
        root,
        targets=["re:^language_model[.]lm_head$"],
        tensors=_LM_HEAD,
    )
    report = check_artifact_completeness(root)
    assert report.undeclared == []
    assert report.orphan_scale == []
    assert report.cb_units == ["lm_head"]
    assert report.ok
    assert_artifact_complete(root)


def test_checkpoint_spelled_group_target_still_claims_its_unit(
        tmp_path, wrapper_profile):
    """The forward map is additive: an identity-namespace artifact, which is
    every pre-wrapper artifact ever shipped, keeps passing unchanged."""

    root = tmp_path / "artifact"
    _write_artifact(root, targets=["re:^lm_head$"], tensors=_LM_HEAD)
    report = check_artifact_completeness(root)
    assert report.cb_units == ["lm_head"]
    assert report.ok


def test_a_group_for_a_different_module_still_fails(tmp_path, wrapper_profile):
    """The negative control. Mapping the unit forward must not turn the gate
    into one that accepts any target at all: a group naming a NEIGHBOURING
    module leaves the head unclaimed, which is the bug the gate exists for."""

    root = tmp_path / "artifact"
    _write_artifact(
        root,
        targets=["re:^language_model[.]model[.]layers[.]0[.]mlp[.]down_proj$"],
        tensors=_LM_HEAD,
    )
    with pytest.raises(ArtifactIncomplete, match="claimed by no mechanism"):
        assert_artifact_complete(root)


def test_without_a_profile_the_gate_falls_back_to_the_literal_name(tmp_path,
                                                                  monkeypatch):
    """`_detect_profile_quietly` returns None on an architecture this build
    does not know. The gate must still run — and on a vLLM-spelled target it
    then has no way to resolve the unit, so it reports rather than guessing."""

    monkeypatch.setattr(completeness, "_detect_profile_quietly",
                        lambda _root: None)
    root = tmp_path / "artifact"
    _write_artifact(
        root,
        targets=["re:^language_model[.]lm_head$"],
        tensors=_LM_HEAD,
    )
    with pytest.raises(ArtifactIncomplete, match="claimed by no mechanism"):
        assert_artifact_complete(root)


def test_split_format_group_fused_units_resolve_through_the_projection():
    """A per-expert split-format unit spells its group token AFTER the
    projection, so the fusion map has to be applied to the projection and the
    token re-attached.

    Reading `format_group_fp8_cb_k28` as the leaf finds nothing in
    `packed_modules_mapping`, and a split export — whose config groups MUST
    name the unfused halves, since vLLM canonical scheme names are a hard
    serving invariant — then reports every fused stack it ships as unclaimed.
    """

    fused = {"gate_up_proj": ("gate_proj", "up_proj")}
    stack = "model.layers.0.mlp.experts.gate_up_proj"

    assert completeness._fused_member_units(stack, fused) == (
        "model.layers.0.mlp.experts.gate_proj",
        "model.layers.0.mlp.experts.up_proj",
    )
    assert completeness._fused_member_units(
        f"{stack}.format_group_fp8_cb_k28", fused
    ) == (
        "model.layers.0.mlp.experts.gate_proj.format_group_fp8_cb_k28",
        "model.layers.0.mlp.experts.up_proj.format_group_fp8_cb_k28",
    )
    # An UNFUSED projection carrying the same token stays unmapped: the bridge
    # must not invent members for a unit the fusion map says nothing about.
    assert completeness._fused_member_units(
        "model.layers.0.mlp.experts.down_proj.format_group_fp8_cb_k28", fused
    ) == ()


def test_a_sidecar_alias_map_is_empty_without_a_published_sidecar(tmp_path):
    """The fifth namespace is opt-in: no `dspark_cb_sidecar` record, no alias,
    and therefore no way for the bridge to launder an unclaimed plane."""

    assert completeness._dspark_sidecar_aliases(tmp_path, {}) == {}
    assert completeness._dspark_sidecar_aliases(
        tmp_path, {"provenance": {"dspark_cb_sidecar": {}}}
    ) == {}
    # Published explicit pairs survive even when config.json cannot be read,
    # while the CB planes it could not resolve simply stay unaliased.
    assert completeness._dspark_sidecar_aliases(
        tmp_path,
        {"provenance": {"dspark_cb_sidecar": {
            "source_passthrough_physical_to_construction": {
                "mtp.0.attn.wo_a": "model.layers.3.attn.wo_a"},
            "physical_cb_targets": ["mtp.0.attn.wkv"],
        }}},
    ) == {"mtp.0.attn.wo_a": "model.layers.3.attn.wo_a"}
