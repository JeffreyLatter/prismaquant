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


# --- THE FIFTH NAMESPACE: DSpark physical vs construction ------------------
#
# A DSpark draft ships its tensors as `mtp.{stage}.<tail>` but vLLM builds those
# blocks as body layers past the end of the body, so the exporter writes their
# config-group targets as `model.layers.{num_hidden_layers+stage}.<tail>`
# (`_cb_target_name` -> `dspark_cb_construction_target_for_physical_output`).
# A TARGET artifact never shows this because `mtp.` is a verbatim prefix there.
# A SIDECAR passes `verbatim_prefixes=()` on purpose — proving those units is
# the entire point of the artifact — so all 27 CB units reported as claimed by
# no mechanism at all.

_SIDECAR_BODY_LAYERS = 3
_SIDECAR_STAGES = 3

#: One CB unit, in the physical namespace its tensors actually ship under.
_DSPARK_UNIT = (
    ("mtp.0.attn.wkv.cb_qweight", "U8", (32, 8), 32 * 8),
)


def _write_dspark_sidecar_artifact(
    root: Path, *, targets, declare_sidecar: bool = True,
) -> None:
    """A minimal sidecar: the physical tensor, plus the config/provenance the
    bridge keys off. `declare_sidecar=False` writes the same bytes with no
    sidecar declaration, which is how the inertness control is expressed."""

    _write_artifact(root, targets=targets, tensors=_DSPARK_UNIT)
    (root / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "num_hidden_layers": _SIDECAR_BODY_LAYERS,
        "n_mtp_layers": _SIDECAR_STAGES,
    }), encoding="utf-8")
    quant = json.loads((root / "quant_config.json").read_text())
    if declare_sidecar:
        quant["provenance"] = {"dspark_cb_sidecar": {
            "schema": "prismaquant.dspark_cb_sidecar.v1",
            "num_hidden_layers": _SIDECAR_BODY_LAYERS,
            "n_mtp_layers": _SIDECAR_STAGES,
        }}
    (root / "quant_config.json").write_text(
        json.dumps(quant), encoding="utf-8")


@pytest.fixture()
def no_profile(monkeypatch):
    """The bridge is architecture arithmetic, not a profile map: it must work
    with no profile at all, which is also what a synthetic artifact detects."""

    monkeypatch.setattr(
        completeness, "_detect_profile_quietly", lambda _root: None)


def test_construction_spelled_target_claims_its_physical_dspark_unit(
        tmp_path, no_profile):
    """The exact shape of the sidecar failure: stage 0 of a 3-layer body is
    built at `model.layers.3`, and that is what the correct artifact claims."""

    root = tmp_path / "artifact"
    _write_dspark_sidecar_artifact(
        root, targets=["re:^model[.]layers[.]3[.]attn[.]wkv$"])
    report = check_artifact_completeness(root, verbatim_prefixes=())
    assert report.undeclared == []
    assert report.cb_units == ["mtp.0.attn.wkv"]
    assert report.ok
    assert_artifact_complete(root, verbatim_prefixes=())


def test_a_construction_target_for_the_wrong_stage_still_fails(
        tmp_path, no_profile):
    """The negative control, and the reason the layer arithmetic is recomputed
    from the model config rather than read back from the sidecar's own recorded
    physical->construction pairing. `model.layers.4` is stage 1; claiming it
    does not claim stage 0's tensor, and an off-by-one the gate forgave would
    ship a draft whose blocks load into the wrong slots."""

    root = tmp_path / "artifact"
    _write_dspark_sidecar_artifact(
        root, targets=["re:^model[.]layers[.]4[.]attn[.]wkv$"])
    with pytest.raises(ArtifactIncomplete, match="claimed by no mechanism"):
        assert_artifact_complete(root, verbatim_prefixes=())


def test_without_a_sidecar_declaration_the_construction_bridge_is_inert(
        tmp_path, no_profile):
    """Additive, like every other bridge here. The construction spelling is
    only a legitimate claim on an artifact that declares itself a DSpark
    sidecar; on anything else — every target artifact ever shipped — the same
    target must still leave the unit unclaimed."""

    root = tmp_path / "artifact"
    _write_dspark_sidecar_artifact(
        root,
        targets=["re:^model[.]layers[.]3[.]attn[.]wkv$"],
        declare_sidecar=False,
    )
    with pytest.raises(ArtifactIncomplete, match="claimed by no mechanism"):
        assert_artifact_complete(root, verbatim_prefixes=())
