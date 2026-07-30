"""Conformance run of `prismaquant/model_profiles/validate.py` over every
registered ModelProfile.

`validate.py` is a manual CLI with (until this file) zero callers, which is
how the defects it exists to catch survived inside it. This test pins the
CPU-safe part of it plus the invariants that de-vacuum the checks that
short-circuit to green without vLLM.

Lanes:
  - default: pure python/CPU. No model weights, no GPU, no network. Runs
    checks 1, 6 (against synthetic index fixtures) and 8, plus four
    structural invariants (spec presence, fused-sibling source, registry
    order, name uniqueness).
  - `integration`: the vLLM-registry checks (2/3/4), skipped when vLLM is
    not importable. Their answer is vLLM-version-dependent, so they are not
    part of the default lane.
  - `slow`: the safetensors-index checks (6/7) against real checkpoints
    named by $PQ_CONFORMANCE_MODELS.

Check 5 (MTP) is deliberately absent: `build_mtp_module()` materialises a
full decoder layer (multi-GB CPU allocation). Use the manual CLI for it:

    python -m prismaquant.model_profiles.validate --model /path/to/Model

Known gaps are encoded as *ratchets*, not bare xfails: each one first
asserts the gap is still real and only then xfails. Closing the gap turns
the test red with an instruction to shrink the list, so the exemption
cannot go stale silently.
"""
from __future__ import annotations

import json
import os
import struct

import pytest

from prismaquant.model_profiles import registry as _registry
from prismaquant.model_profiles import validate as V
from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.model_profiles.laguna import LagunaProfile
from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

PROFILE_CLASSES = list(_registry._REGISTERED) + [DefaultProfile]
PROFILE_IDS = [c.__name__ for c in PROFILE_CLASSES]

# Representative (model_type, architectures) per profile, so check 1 and the
# resolution check run with no checkpoint on disk. Every entry below was
# taken from a real config.json where one exists locally.
REPRESENTATIVE_CONFIGS: dict[str, tuple[tuple[str, list[str]], ...]] = {
    "Qwen3Profile": (("qwen3", ["Qwen3ForCausalLM"]),),
    "Qwen3MoeProfile": (("qwen3_moe", ["Qwen3MoeForCausalLM"]),),
    "Qwen3_5DenseProfile": (("qwen3_5", ["Qwen3_5ForConditionalGeneration"]),),
    "Qwen3_5Profile": (
        ("qwen3_5_moe", ["Qwen3_5MoeForConditionalGeneration"]),
    ),
    # Gemma 4 ships two config flavours; the profile claims both.
    "Gemma4Profile": (
        ("gemma4", ["Gemma4ForConditionalGeneration"]),
        ("gemma4_unified", ["Gemma4UnifiedForConditionalGeneration"]),
    ),
    "Lfm2MoeProfile": (("lfm2_moe", ["Lfm2MoeForCausalLM"]),),
    "MiniMaxM2Profile": (("minimax_m2", ["MiniMaxM2ForCausalLM"]),),
    "DeepseekV4Profile": (("deepseek_v4", ["DeepseekV4ForCausalLM"]),),
    "HyV3Profile": (("hy_v3", ["HYV3ForCausalLM"]),),
    "LagunaProfile": (("laguna", ["LagunaForCausalLM"]),),
}

CONFIG_CASES = [
    (cls_name, cfg)
    for cls_name in sorted(REPRESENTATIVE_CONFIGS)
    for cfg in REPRESENTATIVE_CONFIGS[cls_name]
]
CONFIG_IDS = [f"{n}-{cfg[0]}" for n, cfg in CONFIG_CASES]

# DefaultProfile is the terminal fallback for architectures nobody claims;
# specs are keyed by profile name, so a `specs/default.json` would describe
# no architecture. Its absence is by design, not a gap.
SPEC_EXEMPT_BY_DESIGN = {"DefaultProfile"}

# Known gaps (ratcheted — see module docstring).
NO_SPEC_XFAIL = {
    # minimax_m2 has no specs/*.json; it overrides fused_sibling_group()
    # directly instead, so behaviour is right but it is off the declarative
    # path, and validate.py's "vLLM not importable" amnesty (validate.py:172)
    # requires a spec, so check 2 fails spuriously on any non-vLLM box.
    "MiniMaxM2Profile",
}
NO_FUSED_SOURCE_XFAIL = {
    # deepseek_v4 returns None from vllm_architecture_class() (deliberate,
    # probe-only) AND declares `fused_groups: []`, so fused_sibling_group()
    # is a constant None: its dense-MLP gate/up siblings would never be
    # promoted to one format. Check 3 cannot see this — it returns a green
    # "no vLLM class to cross-check against" (validate.py:191-193).
    "DeepseekV4Profile",
}

FUSED_PROBE_NAMES = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
)


def _instantiate(cls):
    if cls is DefaultProfile:
        return cls(architectures=["LlamaForCausalLM"])
    return cls()


@pytest.fixture(scope="module", params=PROFILE_CLASSES, ids=PROFILE_IDS)
def profile(request):
    return _instantiate(request.param)


# ---------------------------------------------------------------- CPU lane


def test_every_registered_profile_instantiates(profile):
    assert profile.name


def test_profile_names_are_unique():
    """Profile names key `specs/<name>.json` (base.py:909), so a collision
    silently hands one profile another's structure spec."""
    names = [_instantiate(c).name for c in PROFILE_CLASSES]
    assert len(names) == len(set(names)), names


def test_registry_order_is_stable():
    """registry.py:47-50 documents these precedences in comments; assert
    them. Getting profile resolution wrong is how the fused-coherence bug
    shipped unservable artifacts (DefaultProfile -> mixed-scheme fused
    groups)."""
    order = [c.__name__ for c in _registry._REGISTERED]
    assert order.index("Qwen3_5DenseProfile") < order.index("Qwen3_5Profile")
    assert order.index("Qwen3MoeProfile") < order.index("Qwen3Profile")


@pytest.mark.parametrize("cls_name,cfg", CONFIG_CASES, ids=CONFIG_IDS)
def test_check1_matches_representative_config(cls_name, cfg):
    """validate.py check 1 — the profile claims its own representative
    config."""
    cls = next(c for c in PROFILE_CLASSES if c.__name__ == cls_name)
    model_type, archs = cfg
    result = V._check_matches(
        _instantiate(cls),
        {"model_type": model_type, "architectures": archs},
    )
    assert result.ok, result.detail


@pytest.mark.parametrize("cls_name,cfg", CONFIG_CASES, ids=CONFIG_IDS)
def test_representative_config_resolves_to_its_own_profile(cls_name, cfg):
    """No shadowing: detect-by-config lands on the intended profile."""
    model_type, archs = cfg
    resolved = _registry.profile_from_config(
        {"model_type": model_type, "architectures": archs})
    assert type(resolved).__name__ == cls_name


def test_unknown_arch_falls_back_to_default_profile():
    resolved = _registry.profile_from_config(
        {"model_type": "definitely_not_a_real_arch",
         "architectures": ["NotARealForCausalLM"]})
    assert isinstance(resolved, DefaultProfile)


def test_check8_serving_profile_loads(profile):
    """validate.py check 8 — pure python + JSON, always safe to run."""
    result = V._check_serving_profile(profile)
    assert result.ok, result.detail


def test_profile_has_structure_spec(profile):
    """De-vacuums check 2 on CPU: the "vLLM not importable" amnesty at
    validate.py:172 only applies when a declarative spec exists."""
    name = type(profile).__name__
    if name in SPEC_EXEMPT_BY_DESIGN:
        pytest.skip(f"{name} is the terminal fallback; a spec would name "
                    "no architecture")
    has_spec = profile.structure_spec() is not None
    if name in NO_SPEC_XFAIL:
        assert not has_spec, (
            f"{name} now HAS a structure spec — remove it from "
            "NO_SPEC_XFAIL")
        pytest.xfail(f"{name} has no model_profiles/specs/*.json yet")
    assert has_spec


def test_profile_has_a_fused_sibling_source(profile):
    """De-vacuums check 3 on CPU. Check 3 green-lights any profile with no
    vLLM class, so it cannot catch one whose fused_sibling_group() is a
    constant None — which breaks the hard serving invariant that a fused
    group carries exactly one format."""
    name = type(profile).__name__
    has_vllm_cls = profile.vllm_architecture_class() is not None
    spec = profile.structure_spec()
    has_spec_groups = bool(spec is not None and spec.fused_groups)
    overrides = "fused_sibling_group" in vars(type(profile))
    has_source = has_vllm_cls or has_spec_groups or overrides
    if name in NO_FUSED_SOURCE_XFAIL:
        assert not has_source, (
            f"{name} now has a fused-sibling source — remove it from "
            "NO_FUSED_SOURCE_XFAIL")
        pytest.xfail(f"{name}: no vLLM class and empty spec.fused_groups")
    assert has_source


def test_fused_group_is_self_consistent_on_cpu(profile):
    """Spec-driven variant of check 3 that works without vLLM: all q/k/v
    siblings must map to ONE canonical key (or all to None)."""
    keys = {profile.fused_sibling_group(n) for n in FUSED_PROBE_NAMES}
    assert len(keys) == 1, f"{type(profile).__name__}: {keys}"


# -------------------------------------------- check 6, synthetic fixtures
#
# Both on-disk expert layouts must validate. The pre-2026-07-30 check only
# accepted the packed one (`k.endswith(f"experts.{n}")`), so every stock HF
# MoE source — Laguna, ornith-35B, DSv4 — failed a check it should pass.


def _write_index(tmp_path, keys):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001.safetensors" for k in keys}})
    )
    return str(tmp_path)


def _write_single_file(tmp_path, shapes: dict[str, list[int]]):
    """Write a safetensors file that is header-only (no tensor payload); the
    validator reads the header and nothing else."""
    header = {
        k: {"dtype": "F32", "shape": s, "data_offsets": [0, 0]}
        for k, s in shapes.items()
    }
    blob = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob)
    return str(tmp_path)


def test_check6_accepts_packed_expert_layout(tmp_path):
    path = _write_index(tmp_path, [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    ])
    result = V._check_packed_experts(Qwen3_5Profile(), path)
    assert result.ok, result.detail
    assert "packed" in result.detail


def test_check6_accepts_per_expert_layout(tmp_path):
    """The regression this file exists for: a stock HF MoE source ships
    per-expert 2D tensors, and packing happens at load/export time."""
    keys = []
    for e in range(4):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys.append(f"model.layers.3.mlp.experts.{e}.{proj}.weight")
    keys.append("model.layers.3.mlp.experts.e_score_correction_bias")
    path = _write_index(tmp_path, keys)
    result = V._check_packed_experts(LagunaProfile(), path)
    assert result.ok, result.detail
    assert "per-expert" in result.detail


def test_check6_accepts_mixed_layout(tmp_path):
    """Qwen3.5-35B-A3B really does ship both: packed body experts and
    per-expert MTP experts."""
    keys = [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.mlp.experts.7.gate_proj.weight",
        "mtp.layers.0.mlp.experts.7.down_proj.weight",
    ]
    result = V._check_packed_experts(Qwen3_5Profile(), _write_index(tmp_path, keys))
    assert result.ok, result.detail
    assert "packed" in result.detail and "per-expert" in result.detail


def test_check6_flags_undeclared_3d_expert_param(tmp_path):
    """The docstring's second clause, now real: a 3D expert tensor the
    profile does not declare would be silently skipped by the pipeline."""
    path = _write_single_file(
        tmp_path, {"model.layers.0.mlp.experts.mystery_proj": [8, 512, 256]})
    result = V._check_packed_experts(Qwen3_5Profile(), path)
    assert not result.ok
    assert "mystery_proj" in result.detail


def test_check6_is_lenient_on_a_dense_family_member(tmp_path):
    """One profile covers a family; a dense member (Gemma 4 31B-IT vs
    26B-A4B) legitimately has no expert tensors at all."""
    path = _write_index(tmp_path, [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ])
    result = V._check_packed_experts(Qwen3_5Profile(), path)
    assert result.ok, result.detail


def test_check6_fails_when_experts_match_no_declared_name(tmp_path):
    path = _write_index(tmp_path, [
        "model.layers.0.mlp.experts.0.wA.weight",
        "model.layers.0.mlp.experts.0.wB.weight",
    ])
    result = V._check_packed_experts(Qwen3_5Profile(), path)
    assert not result.ok
    assert "expert tensors on disk" in result.detail


def test_check6_reads_a_single_file_checkpoint(tmp_path):
    """A single-shard checkpoint has no index; passing it green
    ("cannot verify") verified nothing."""
    path = _write_single_file(tmp_path, {
        "model.layers.0.mlp.experts.gate_up_proj": [8, 512, 256],
        "model.layers.0.mlp.experts.down_proj": [8, 256, 256],
    })
    result = V._check_packed_experts(Qwen3_5Profile(), path)
    assert result.ok, result.detail
    assert "single-file header" in result.detail


def test_check6_reports_cannot_verify_without_weights(tmp_path):
    result = V._check_packed_experts(Qwen3_5Profile(), str(tmp_path))
    assert result.ok
    assert "cannot verify" in result.detail


# -------------------------------------------------- integration lane (vLLM)
#
# These answers are vLLM-version-dependent, which is why they are not in the
# default lane: a check-2 red here can mean "this vLLM predates the model"
# rather than "the profile is wrong" (an April-2026 vLLM fails
# LagunaProfile, which the production image serves). Cross-check the image
# before calling a red here a profile defect.


@pytest.mark.integration
def test_check2_vllm_class_resolves(profile):
    pytest.importorskip("vllm", reason="vLLM registry checks need vLLM")
    result = V._check_vllm_class(profile)
    assert result.ok, result.detail


@pytest.mark.integration
def test_check3_fused_siblings_against_vllm(profile):
    pytest.importorskip("vllm", reason="vLLM registry checks need vLLM")
    result = V._check_fused_siblings(profile)
    assert result.ok, result.detail


@pytest.mark.integration
def test_check4_name_remap_against_vllm(profile):
    pytest.importorskip("vllm", reason="vLLM registry checks need vLLM")
    result = V._check_name_remap(profile)
    assert result.ok, result.detail


# ------------------------------------------------- slow lane (checkpoints)


def _configured_checkpoints() -> dict[str, str]:
    """PQ_CONFORMANCE_MODELS='Qwen3Profile=/path,LagunaProfile=/path'"""
    raw = os.environ.get("PQ_CONFORMANCE_MODELS", "").strip()
    out: dict[str, str] = {}
    for item in filter(None, (s.strip() for s in raw.split(","))):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


@pytest.mark.slow
def test_check6_packed_expert_names(profile):
    path = _configured_checkpoints().get(type(profile).__name__)
    if not path:
        pytest.skip("no checkpoint configured in $PQ_CONFORMANCE_MODELS")
    result = V._check_packed_experts(profile, path)
    assert result.ok, result.detail


@pytest.mark.slow
def test_check7_source_passthrough_prefixes(profile):
    path = _configured_checkpoints().get(type(profile).__name__)
    if not path:
        pytest.skip("no checkpoint configured in $PQ_CONFORMANCE_MODELS")
    result = V._check_source_passthrough(profile, path)
    assert result.ok, result.detail
