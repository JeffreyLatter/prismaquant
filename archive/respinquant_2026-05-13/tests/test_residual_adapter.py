import json

import torch
from safetensors.torch import load_file

from prismaquant.residual_adapter import (
    CONFIG_KEY,
    MANIFEST_FILENAME,
    PLUGIN_ARCHITECTURE,
    LowRankResidualAdapter,
    ResidualAdapterSpec,
    apply_adapter_to_output,
    make_identity_manifest,
    patch_config_for_residual_adapter,
)
from tools.create_residual_adapter_variant import create_variant
from tools.create_respin_equivalent_variant import (
    _identity_norm_weight,
    _fold_norm_and_rotate_input,
    _rotate_output_projection,
    build_alternating_transitions,
    build_transitions_from_basis_checkpoint,
    paper_subspace_residual_transition,
)


def test_patch_config_preserves_base_architecture():
    config = {"architectures": ["ToyForCausalLM"], "hidden_size": 8}
    manifest = make_identity_manifest(config, ["model.layers.0"], rank=0)

    patched = patch_config_for_residual_adapter(config, manifest)

    assert patched["architectures"] == [PLUGIN_ARCHITECTURE]
    assert patched[CONFIG_KEY]["base_architectures"] == ["ToyForCausalLM"]
    assert patched[CONFIG_KEY]["manifest_file"] == MANIFEST_FILENAME


def test_rank_zero_adapter_is_identity():
    adapter = LowRankResidualAdapter(hidden_size=4, rank=0)
    x = torch.randn(2, 3, 4)

    assert adapter(x) is x


def test_low_rank_adapter_math():
    adapter = LowRankResidualAdapter(hidden_size=2, rank=1)
    with torch.no_grad():
        adapter.u.copy_(torch.tensor([[1.0], [2.0]]))
        adapter.v.copy_(torch.tensor([[3.0, 4.0]]))
    x = torch.tensor([[[5.0, 7.0]]])

    expected = x + (x @ adapter.u) @ adapter.v
    assert torch.allclose(adapter(x), expected)


def test_residual_pair_output_transforms_hidden_and_residual():
    adapter = LowRankResidualAdapter(hidden_size=2, rank=1)
    with torch.no_grad():
        adapter.u.copy_(torch.tensor([[1.0], [0.0]]))
        adapter.v.copy_(torch.tensor([[0.5, 0.0]]))
    hidden = torch.tensor([[2.0, 3.0]])
    residual = torch.tensor([[4.0, 5.0]])

    out_hidden, out_residual = apply_adapter_to_output(
        adapter,
        (hidden, residual),
        mode="residual_pair",
    )

    assert torch.allclose(out_hidden, adapter(hidden))
    assert torch.allclose(out_residual, adapter(residual))


def test_residual_adapter_spec_from_dict_fills_weight_names():
    spec = ResidualAdapterSpec.from_dict({
        "module_path": "model.layers.0",
        "rank": 2,
    })

    assert spec.u_name == "prisma_residual_adapters.model_layers_0.u"
    assert spec.v_name == "prisma_residual_adapters.model_layers_0.v"


def test_create_variant_patches_config_and_rank_tensors(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({
        "architectures": ["ToyForCausalLM"],
        "hidden_size": 8,
    }))
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": {},
    }))
    (src / "tokenizer.json").write_text("{}")

    dst = tmp_path / "dst"
    summary = create_variant(
        src,
        dst,
        module_paths=["model.layers.0", "model.layers.1"],
        rank=2,
    )

    source_config = json.loads((src / "config.json").read_text())
    source_index = json.loads((src / "model.safetensors.index.json").read_text())
    patched = json.loads((dst / "config.json").read_text())
    manifest = json.loads((dst / MANIFEST_FILENAME).read_text())
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    tensors = load_file(dst / "prisma-residual-adapters.safetensors")

    assert source_config["architectures"] == ["ToyForCausalLM"]
    assert CONFIG_KEY not in source_config
    assert source_index["weight_map"] == {}
    assert source_index["metadata"]["total_size"] == 0
    assert summary["architecture"] == PLUGIN_ARCHITECTURE
    assert patched["architectures"] == [PLUGIN_ARCHITECTURE]
    assert manifest["base_architectures"] == ["ToyForCausalLM"]
    assert len(manifest["adapters"]) == 2
    assert tensors["prisma_residual_adapters.model_layers_0.u"].shape == (8, 2)
    assert tensors["prisma_residual_adapters.model_layers_0.v"].shape == (2, 8)
    assert index["weight_map"]["prisma_residual_adapters.model_layers_1.v"] == (
        "prisma-residual-adapters.safetensors"
    )


def test_create_variant_respin_givens_initializer_is_deterministic(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({
        "architectures": ["ToyForCausalLM"],
        "hidden_size": 8,
    }))
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": {},
    }))

    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {
        "module_paths": ["model.layers.0"],
        "rank": 4,
        "initializer": "respin-givens",
        "angle": 0.05,
        "seed": 123,
    }
    create_variant(src, first, **kwargs)
    create_variant(src, second, **kwargs)

    a = load_file(first / "prisma-residual-adapters.safetensors")
    b = load_file(second / "prisma-residual-adapters.safetensors")
    u_name = "prisma_residual_adapters.model_layers_0.u"
    v_name = "prisma_residual_adapters.model_layers_0.v"

    assert torch.equal(a[u_name], b[u_name])
    assert torch.equal(a[v_name], b[v_name])
    assert torch.count_nonzero(a[u_name]) == 4
    assert torch.count_nonzero(a[v_name]) == 8
    assert float(a[v_name].abs().max()) > 0.04


def test_respin_equivalent_alternating_transitions_close_to_identity():
    transitions, bases = build_alternating_transitions(
        hidden_size=8,
        n_layers=4,
        rank=4,
        angle=0.1,
        seed=7,
        device=torch.device("cpu"),
    )
    current = torch.eye(8)
    for transition in transitions:
        current = current @ transition.matrix

    assert len(transitions) == 4
    assert len(bases) == 4
    assert torch.allclose(current, torch.eye(8), atol=1e-6)
    assert torch.allclose(bases[0], torch.eye(8), atol=1e-6)
    assert torch.allclose(bases[2], torch.eye(8), atol=1e-6)
    assert not torch.allclose(bases[1], torch.eye(8), atol=1e-6)


def test_paper_subspace_transition_recovers_full_rank_givens_subspace():
    transitions, _bases = build_alternating_transitions(
        hidden_size=8,
        n_layers=2,
        rank=4,
        angle=0.1,
        seed=13,
        device=torch.device("cpu"),
    )
    exact = transitions[0].matrix

    approx = paper_subspace_residual_transition(
        exact,
        4,
        device=torch.device("cpu"),
    )

    assert approx.approximation is not None
    assert approx.approximation["mode"] == "paper_svd_polar"
    assert approx.approximation["rank"] == 4
    assert approx.approximation["sv_energy_retained"] > 0.999
    assert torch.allclose(approx.matrix, exact, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        torch.eye(8) + approx.u @ approx.v,
        approx.matrix,
        atol=1e-5,
        rtol=1e-5,
    )


def test_paper_subspace_transition_rank_zero_is_identity():
    transitions, _bases = build_alternating_transitions(
        hidden_size=8,
        n_layers=2,
        rank=4,
        angle=0.1,
        seed=19,
        device=torch.device("cpu"),
    )

    approx = paper_subspace_residual_transition(
        transitions[0].matrix,
        0,
        device=torch.device("cpu"),
    )

    assert approx.u.shape == (8, 0)
    assert approx.v.shape == (0, 8)
    assert torch.allclose(approx.matrix, torch.eye(8), atol=1e-6)
    assert approx.approximation is not None
    assert approx.approximation["rank"] == 0


def test_alternating_transitions_can_use_paper_svd_approximation():
    transitions, bases = build_alternating_transitions(
        hidden_size=8,
        n_layers=4,
        rank=4,
        angle=0.1,
        seed=7,
        device=torch.device("cpu"),
        transition_mode="paper-svd",
    )
    current = torch.eye(8)
    for transition in transitions:
        current = current @ transition.matrix
        assert transition.approximation is not None
        assert transition.approximation["mode"] == "paper_svd_polar"

    assert len(transitions) == 4
    assert len(bases) == 4
    assert torch.allclose(current, torch.eye(8), atol=1e-5)


def test_transitions_from_trained_basis_checkpoint_close_identity(tmp_path):
    theta = 0.1
    basis = torch.eye(8)
    basis[:2, :2] = torch.tensor([
        [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
        [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))],
    ])
    checkpoint = tmp_path / "rot.pt"
    torch.save({"layer.1": basis}, checkpoint)

    transitions, bases, meta = build_transitions_from_basis_checkpoint(
        checkpoint,
        hidden_size=8,
        n_layers=3,
        rank=2,
        device=torch.device("cpu"),
        transition_mode="paper-svd",
    )

    current = torch.eye(8)
    for transition in transitions:
        current = current @ transition.matrix
        assert transition.approximation is not None
        assert transition.approximation["mode"] == "paper_svd_polar"

    assert len(bases) == 3
    assert torch.allclose(current, torch.eye(8), atol=1e-5, rtol=1e-5)
    assert meta["basis_source"] == "trained_checkpoint"
    assert meta["used_rotation_keys"] == ["layer.1"]
    assert meta["missing_identity_layers"] == [0, 2]


def test_respin_equivalent_weight_transforms_preserve_layer_math():
    transitions, bases = build_alternating_transitions(
        hidden_size=4,
        n_layers=2,
        rank=2,
        angle=0.2,
        seed=11,
        device=torch.device("cpu"),
    )
    del transitions
    basis = bases[1]
    x = torch.randn(6, 4)
    gamma = torch.rand(4) + 0.5
    w_in = torch.randn(3, 4)
    eps = 1e-6
    x_norm = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_rot = x @ basis
    x_rot_norm = x_rot / torch.sqrt(x_rot.pow(2).mean(dim=-1, keepdim=True) + eps)

    w_in_rot = _fold_norm_and_rotate_input(
        w_in,
        gamma,
        basis,
        device=torch.device("cpu"),
    )

    expected_in = (x_norm * gamma) @ w_in.t()
    actual_in = x_rot_norm @ w_in_rot.t()
    assert torch.allclose(actual_in, expected_in, atol=1e-5, rtol=1e-5)

    internal = torch.randn(6, 3)
    w_out = torch.randn(4, 3)
    w_out_rot = _rotate_output_projection(
        w_out,
        basis,
        device=torch.device("cpu"),
    )

    expected_out = (internal @ w_out.t()) @ basis
    actual_out = internal @ w_out_rot.t()
    assert torch.allclose(actual_out, expected_out, atol=1e-5, rtol=1e-5)


def test_respin_equivalent_gemma_norm_fold_uses_offset_gamma():
    transitions, bases = build_alternating_transitions(
        hidden_size=4,
        n_layers=2,
        rank=2,
        angle=0.2,
        seed=17,
        device=torch.device("cpu"),
    )
    del transitions
    basis = bases[1]
    x = torch.randn(5, 4)
    raw_weight = torch.randn(4) * 0.05
    effective_gamma = raw_weight + 1.0
    w_in = torch.randn(3, 4)
    eps = 1e-6
    x_norm = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_rot = x @ basis
    x_rot_norm = x_rot / torch.sqrt(x_rot.pow(2).mean(dim=-1, keepdim=True) + eps)

    w_in_rot = _fold_norm_and_rotate_input(
        w_in,
        raw_weight,
        basis,
        device=torch.device("cpu"),
        norm_style="gemma",
    )

    expected = (x_norm * effective_gamma) @ w_in.t()
    actual = x_rot_norm @ w_in_rot.t()
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(_identity_norm_weight(raw_weight, "gemma"),
                       torch.zeros_like(raw_weight))
