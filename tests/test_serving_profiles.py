from __future__ import annotations

from prismaquant.serving_profiles import (
    ServingProfile,
    check_serving_format,
    check_serving_shape,
    load_serving_profile,
    serving_profile_names,
)


VLLM_PROFILE = "vllm_packed_moe"


def test_serving_profile_names_are_config_discovered():
    assert "research" in serving_profile_names()
    assert VLLM_PROFILE in serving_profile_names()


def test_vllm_profile_extends_runtime_shape_rules():
    profile = load_serving_profile(VLLM_PROFILE)

    assert profile.extends == ("research",)
    assert any(rule.id == "mxfp8_cutlass_shape" for rule in profile.shape_rules)
    assert any(
        rule.id == "flashinfer_mxfp8_problem_size"
        and rule.callable_path
        == "prismaquant.runtime_shape_validators:flashinfer_mxfp8_problem_size_accepts"
        for rule in profile.runtime_shape_validators
    )
    flashinfer = profile.runtime_package("flashinfer")
    assert flashinfer is not None
    assert flashinfer.version == "0.6.8.post1"
    assert flashinfer.pip_packages == ("flashinfer-python", "flashinfer-cubin")
    assert flashinfer.env_dict()["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
    assert any(rule.id == "packed_moe_expert_formats" for rule in profile.format_rules)


def test_qwen_serving_profile_id_remains_compatibility_alias():
    profile = load_serving_profile("vllm_qwen3_5_packed_moe")

    assert profile.extends == ("vllm_packed_moe",)
    assert any(rule.id == "packed_moe_expert_formats" for rule in profile.format_rules)


def test_serving_profile_format_rules_are_config_backed():
    expert = "model.layers.0.mlp.experts.gate_up_proj"
    root_expert = "model.layers.0.experts.gate_up_proj"
    dense = "model.layers.0.self_attn.q_proj"

    assert check_serving_format(VLLM_PROFILE, expert, "MXFP8_E4M3").legal
    assert check_serving_format(VLLM_PROFILE, root_expert, "MXFP4").legal
    expert_fp8 = check_serving_format(VLLM_PROFILE, expert, "FP8_E4M3")
    assert expert_fp8.legal
    root_fp8 = check_serving_format(VLLM_PROFILE, root_expert, "FP8_E4M3")
    assert root_fp8.legal

    dense_mxfp4 = check_serving_format(VLLM_PROFILE, dense, "MXFP4")
    assert not dense_mxfp4.legal
    assert dense_mxfp4.rule == "dense_formats_without_vllm_fast_path"


def test_serving_profile_shape_rules_are_config_backed():
    small_n = check_serving_shape(
        "research",
        "MXFP8_E4M3",
        in_features=5120,
        out_features=48,
    )
    standard = check_serving_shape(
        VLLM_PROFILE,
        "MXFP8_E4M3",
        in_features=5120,
        out_features=10240,
    )
    nvfp4_bad_k = check_serving_shape(
        "research",
        "NVFP4",
        in_features=17,
        out_features=128,
    )

    assert not small_n.legal
    assert small_n.reason == "kernel_shape"
    assert "out_features=48" in small_n.detail
    assert standard.legal
    assert not nvfp4_bad_k.legal


def test_shape_rules_can_be_name_scoped():
    profile = ServingProfile.from_dict({
        "schema": "prismaquant.serving_profile.v1",
        "id": "unit_scoped",
        "shape_rules": [
            {
                "id": "expert_only_alignment",
                "when": {"contains": ".experts."},
                "formats": ["MXFP8_E4M3"],
                "out_features_multiple_of": 128,
            }
        ],
    })

    expert = profile.check_shape(
        "MXFP8_E4M3",
        qname="model.layers.0.mlp.experts.0.gate_proj",
        in_features=256,
        out_features=96,
    )
    dense = profile.check_shape(
        "MXFP8_E4M3",
        qname="model.layers.0.mlp.gate_proj",
        in_features=256,
        out_features=96,
    )

    assert not expert.legal
    assert expert.rule == "expert_only_alignment"
    assert dense.legal


def test_runtime_shape_validator_rules_are_config_backed(monkeypatch):
    import prismaquant.serving_profiles as serving_profiles

    def fake_loader(callable_path):
        assert callable_path == (
            "prismaquant.runtime_shape_validators:"
            "flashinfer_mxfp8_problem_size_accepts"
        )

        def fake_validator(fmt, *, in_features, out_features):
            assert fmt == "MXFP8_E4M3"
            assert (in_features, out_features) == (5120, 10240)
            return False

        return fake_validator

    monkeypatch.setattr(
        serving_profiles,
        "_load_runtime_validator",
        fake_loader,
    )

    decision = serving_profiles.check_serving_shape(
        "research",
        "MXFP8_E4M3",
        in_features=5120,
        out_features=10240,
    )

    assert not decision.legal
    assert decision.rule == "flashinfer_mxfp8_problem_size"
    assert decision.reason == "kernel_shape"


def test_runtime_shape_validator_treats_fp8_setup_failure_as_unavailable(
    monkeypatch,
):
    import sys
    import types

    from prismaquant.runtime_shape_validators import (
        flashinfer_mxfp8_problem_size_accepts,
    )

    fake_torch = types.ModuleType("torch")
    fake_torch.uint8 = object()

    def fake_empty(*_args, **_kwargs):
        raise RuntimeError("fp8 setup unavailable")

    fake_torch.empty = fake_empty

    fake_flashinfer = types.ModuleType("flashinfer")
    fake_gemm = types.ModuleType("flashinfer.gemm")
    fake_gemm_base = types.ModuleType("flashinfer.gemm.gemm_base")
    fake_gemm_base._check_mm_mxfp8_problem_size = lambda *_args: True
    fake_gemm_base._mxfp8_swizzled_scale_len = lambda *_args: 1
    fake_gemm_base.SfLayout = types.SimpleNamespace(layout_8x4=object())

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", fake_gemm)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm.gemm_base", fake_gemm_base)

    assert (
        flashinfer_mxfp8_problem_size_accepts(
            "MXFP8_E4M3",
            in_features=5120,
            out_features=10240,
        )
        is None
    )


def test_runtime_shape_validator_legacy_id_fallback(monkeypatch):
    import prismaquant.serving_profiles as serving_profiles

    def fake_loader(callable_path):
        assert callable_path == (
            "prismaquant.runtime_shape_validators:"
            "flashinfer_mxfp8_problem_size_accepts"
        )

        def fake_validator(fmt, *, in_features, out_features):
            assert fmt == "MXFP8_E4M3"
            assert (in_features, out_features) == (5120, 10240)
            return False

        return fake_validator

    monkeypatch.setattr(
        serving_profiles,
        "_load_runtime_validator",
        fake_loader,
    )

    decision = serving_profiles._runtime_shape_validator_accepts(
        "flashinfer_mxfp8_problem_size",
        "MXFP8_E4M3",
        in_features=5120,
        out_features=10240,
    )

    assert decision is False


def test_runtime_shape_validators_can_be_name_scoped(monkeypatch):
    import prismaquant.serving_profiles as serving_profiles

    calls = []

    def fake_loader(_callable_path):
        def fake_validator(fmt, *, in_features, out_features):
            calls.append((fmt, in_features, out_features))
            return False

        return fake_validator

    monkeypatch.setattr(
        serving_profiles,
        "_load_runtime_validator",
        fake_loader,
    )

    profile = ServingProfile.from_dict({
        "schema": "prismaquant.serving_profile.v1",
        "id": "unit_runtime_scoped",
        "runtime_shape_validators": [
            {
                "id": "expert_runtime",
                "when": {"contains": ".experts."},
                "formats": ["MXFP8_E4M3"],
                "callable": "tests.fake:validator",
            }
        ],
    })

    dense = profile.check_shape(
        "MXFP8_E4M3",
        qname="model.layers.0.mlp.gate_proj",
        in_features=256,
        out_features=256,
    )
    expert = profile.check_shape(
        "MXFP8_E4M3",
        qname="model.layers.0.mlp.experts.0.gate_proj",
        in_features=256,
        out_features=256,
    )

    assert dense.legal
    assert not expert.legal
    assert expert.rule == "expert_runtime"
    assert calls == [("MXFP8_E4M3", 256, 256)]
