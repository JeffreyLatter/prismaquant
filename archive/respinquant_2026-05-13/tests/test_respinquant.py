import json

import pytest

from prismaquant.respinquant import (
    ReSpinLayerBasis,
    ReSpinRuntimeAdapterRequired,
    analyze_residual_basis_plan,
    assert_kernel_free,
    hidden_layers_from_config,
    make_global_rotation_plan,
    make_layerwise_respin_plan,
)


def test_identity_plan_is_kernel_free():
    plan = analyze_residual_basis_plan([
        ReSpinLayerBasis(0),
        ReSpinLayerBasis(1),
    ])

    assert plan.kernel_free
    assert not plan.requires_runtime_adapter
    assert plan.equivalent == "identity"
    assert plan.transitions == ()


def test_global_basis_plan_is_kernel_free_but_not_layerwise():
    plan = make_global_rotation_plan(3)

    assert plan.kernel_free
    assert plan.equivalent == "global_rotation"
    assert not plan.requires_runtime_adapter


def test_layerwise_respin_requires_runtime_residual_adapter():
    plan = make_layerwise_respin_plan(3, "all")

    assert not plan.kernel_free
    assert plan.requires_runtime_adapter
    assert plan.equivalent == "runtime_residual_adapter"
    assert any(t.kind == "within_residual_layer" for t in plan.transitions)
    with pytest.raises(ReSpinRuntimeAdapterRequired):
        assert_kernel_free(plan)


def test_selective_layerwise_respin_is_not_foldable_either():
    plan = make_layerwise_respin_plan(4, "1")

    assert not plan.kernel_free
    assert {t.kind for t in plan.transitions} == {"within_residual_layer"}


def test_runtime_adapter_override_keeps_research_plan_available():
    plan = make_layerwise_respin_plan(2, "all")

    assert assert_kernel_free(plan, allow_runtime_adapter=True) is plan


def test_hidden_layers_from_config_accepts_nested_text_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_config": {"num_hidden_layers": 7}}))

    assert hidden_layers_from_config(tmp_path) == 7
