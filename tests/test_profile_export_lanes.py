"""R6 (reader half) — export-lane eligibility as model configuration.

`EXPORT_CONTAINER` is an operator env var with no relationship to whether the
architecture is wired for that lane. Nothing stops `EXPORT_CONTAINER=nvfp4_cb`
on an arch whose gridbook CB expert loader is a TODO: the run completes, the
artifact serves, and the FusedMoE reads uninitialised memory — coherent-looking
garbage, not a crash (commit `9a79963`, Laguna, 93% of parameters). The honest
CB-eligible set is four architectures and until now nothing in the tree said
so.

This file pins the reader half: the spec fields, the profile accessors, and the
preflight helper. The `run-pipeline.sh` wiring is a later wave — the helper is
written and tested first so the wiring is a one-line call, not a redesign.
"""
from __future__ import annotations

import pytest

from prismaquant.model_profiles import detect_profile  # noqa: F401  (API shape)
from prismaquant.model_profiles import registry as _registry
from prismaquant.model_profiles.structure import (
    DEFAULT_EXPORT_LANE,
    EXPORT_LANES,
    ModelStructureSpec,
    SCHEMA,
    canonical_export_lane,
    load_structure_spec,
)
from prismaquant.serving_profiles import require_lane_supported

# The four architectures with a gridbook CB loader
# (`plugins/gridbook/gridbook/plugin.py`), and nothing else.
CB_WIRED = {"qwen3_5", "qwen3_5_dense", "hy_v3", "laguna"}
GGUF_WIRED = {"hy_v3"}

PROFILE_CLASSES = list(_registry._REGISTERED)
PROFILE_IDS = [c.__name__ for c in PROFILE_CLASSES]


def _profile(cls):
    return cls()


# ------------------------------------------------------------------ vocabulary


def test_lane_vocabulary_is_the_export_container_vocabulary():
    assert EXPORT_LANES == ("compressed-tensors", "nvfp4_cb", "gguf")
    assert DEFAULT_EXPORT_LANE == "compressed-tensors"
    # One declared alias: the serving-profile side spells the native lane with
    # an underscore (`export_lane.id == "compressed_tensors"`).
    assert canonical_export_lane("compressed_tensors") == "compressed-tensors"
    with pytest.raises(ValueError, match="unknown export lane"):
        canonical_export_lane("nvfp4-cb")


def test_preferred_lane_must_be_supported():
    with pytest.raises(ValueError, match="preferred_lane"):
        ModelStructureSpec.from_dict({
            "schema": SCHEMA,
            "id": "lane_test",
            "supported_lanes": ["compressed-tensors"],
            "preferred_lane": "gguf",
        })


# --------------------------------------------------------------- declarations


@pytest.mark.parametrize("cls", PROFILE_CLASSES, ids=PROFILE_IDS)
def test_declared_lanes_match_the_wired_reality(cls):
    """The declared set must equal the wiring that exists, not the wiring we
    would like to exist. Over-declaring is the failure this exists to stop."""
    profile = _profile(cls)
    lanes = set(profile.supported_export_lanes())
    assert DEFAULT_EXPORT_LANE in lanes, (
        f"{profile.name}: every architecture ships through the native lane")
    assert ("nvfp4_cb" in lanes) == (profile.name in CB_WIRED), (
        f"{profile.name}: nvfp4_cb declaration disagrees with the gridbook "
        f"loader list {sorted(CB_WIRED)}")
    assert ("gguf" in lanes) == (profile.name in GGUF_WIRED), (
        f"{profile.name}: gguf declaration disagrees with {sorted(GGUF_WIRED)}")


@pytest.mark.parametrize("cls", PROFILE_CLASSES, ids=PROFILE_IDS)
def test_preferred_lane_is_supported_and_defaults_to_native(cls):
    profile = _profile(cls)
    preferred = profile.preferred_export_lane()
    assert preferred in profile.supported_export_lanes()
    spec = load_structure_spec(profile.name)
    if spec is None or not spec.preferred_lane:
        assert preferred == DEFAULT_EXPORT_LANE


def test_the_two_script_driven_lanes_are_declared():
    """`scripts/run_*_prod_nvfp4cb.sh` and the GGUF lane are tribal knowledge
    today; these two are the archs whose shipped artifacts came off them."""
    assert load_structure_spec("hy_v3").preferred_lane == "gguf"
    assert load_structure_spec("laguna").preferred_lane == "nvfp4_cb"


# ------------------------------------------------------------------- preflight


def test_require_lane_supported_accepts_declared_lanes():
    from prismaquant.model_profiles.laguna import LagunaProfile
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    assert require_lane_supported(Qwen3_5Profile(), "nvfp4_cb") == "nvfp4_cb"
    assert require_lane_supported(LagunaProfile(), None) == "compressed-tensors"
    assert require_lane_supported(
        Qwen3_5Profile(), "compressed_tensors") == "compressed-tensors"


def test_require_lane_supported_refuses_an_undeclared_lane():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    with pytest.raises(SystemExit) as excinfo:
        require_lane_supported(DeepseekV4Profile(), "nvfp4_cb")
    message = str(excinfo.value)
    assert "deepseek_v4" in message
    assert "compressed-tensors" in message      # names the declared set
    assert "garbage" in message                 # names the failure mode


def test_require_lane_supported_refuses_an_unknown_lane():
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    with pytest.raises(SystemExit, match="unknown export lane"):
        require_lane_supported(Qwen3_5Profile(), "nvfp4-cb")


def test_require_lane_supported_is_inert_without_the_accessor():
    """Duck-typed like `resolve_target_profile`: a profile object that predates
    the accessor must not break a run."""
    class _Legacy:
        name = "legacy"

    assert require_lane_supported(_Legacy(), "gguf") == "gguf"


def test_no_shipped_lane_run_becomes_illegal():
    """Non-regression: every in-tree launch script's (arch, EXPORT_CONTAINER)
    pair must still pass the preflight."""
    cases = [
        ("qwen3_5", "nvfp4_cb"),        # run_27b_prod_nvfp4cb, run_35b_prod_nvfp4cb
        ("qwen3_5_dense", "nvfp4_cb"),
        ("hy_v3", "nvfp4_cb"),          # run_hy3_prod_nvfp4cb / _joint
        ("hy_v3", "gguf"),
        ("laguna", "nvfp4_cb"),         # run_laguna_s21_prod
        ("qwen3_5", "compressed-tensors"),
        ("gemma4", "compressed-tensors"),
        ("lfm2_moe", "compressed-tensors"),
        ("minimax_m2", "compressed-tensors"),
        ("deepseek_v4", "compressed-tensors"),
    ]
    by_name = {c().name: c() for c in PROFILE_CLASSES}
    for name, lane in cases:
        assert require_lane_supported(by_name[name], lane) == lane
