"""Unit tests for prismaquant.hadamard_duquant_export."""
from __future__ import annotations

from dataclasses import replace

import torch

from prismaquant.hadamard_duquant import (
    NVFP4_GROUP_SIZE,
    MXFP8_GROUP_SIZE,
    apply_block_rotation_input,
    sylvester_hadamard,
)
from prismaquant.hadamard_duquant_cache import (
    CachedRotation,
    HadamardDuQuantCacheState,
)
from prismaquant.hadamard_duquant_export import (
    ConsumerRewrite,
    ProducerRewrite,
    TransformsConfigEntry,
    assert_vllm_online_transforms_supported,
    apply_rotations_to_model_in_place,
    build_rotation_safetensors_entries,
    build_transforms_config,
    consumer_artifact_weight,
    iter_consumer_rewrites,
    iter_producer_rewrites,
    producer_artifact_weight,
    vllm_incompatible_online_clusters,
)
from torch import nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rotation(
    cluster_key: str = "model.layers.0.mlp.down",
    *,
    online: bool = True,
    g: int = NVFP4_GROUP_SIZE,
    format_label: str = "NVFP4",
    insertion_kind: str = "down_proj",
    runtime_transform_type: str = "random-matrix",
) -> CachedRotation:
    return CachedRotation(
        cluster_key=cluster_key,
        format_label=format_label,
        composed_matrix=sylvester_hadamard(g, dtype=torch.float64),
        group_size=g,
        insertion_kind=insertion_kind,
        online=online,
        runtime_transform_type=runtime_transform_type,
    )


def _make_dense_nonsymmetric_rotation(
    cluster_key: str = "model.layers.0.mlp.down",
    *,
    g: int = NVFP4_GROUP_SIZE,
) -> CachedRotation:
    M = torch.eye(g, dtype=torch.float64)
    M[:2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    return CachedRotation(
        cluster_key=cluster_key,
        format_label="NVFP4",
        composed_matrix=M,
        group_size=g,
        insertion_kind="down_proj",
        online=True,
        runtime_transform_type="random-matrix",
    )


def _make_state_online_only() -> HadamardDuQuantCacheState:
    """One online cluster (down_proj) with two consumer qnames."""
    rot = _make_rotation(online=True)
    return HadamardDuQuantCacheState(
        rotations_by_cluster={"model.layers.0.mlp.down": rot},
        consumer_to_cluster={
            "model.layers.0.mlp.down_proj": "model.layers.0.mlp.down",
        },
        producer_to_cluster={},
    )


def _make_state_offline_only() -> HadamardDuQuantCacheState:
    """One offline cluster (V→O) with one consumer + one producer."""
    rot = _make_rotation(
        cluster_key="model.layers.0.attn.v_o",
        online=False,
        insertion_kind="v_o",
    )
    return HadamardDuQuantCacheState(
        rotations_by_cluster={"model.layers.0.attn.v_o": rot},
        consumer_to_cluster={
            "model.layers.0.self_attn.o_proj": "model.layers.0.attn.v_o",
        },
        producer_to_cluster={
            "model.layers.0.self_attn.v_proj": "model.layers.0.attn.v_o",
        },
    )


# ---------------------------------------------------------------------------
# consumer_artifact_weight
# ---------------------------------------------------------------------------


def test_consumer_artifact_recovers_quantized_rotated_form():
    """Cache W_eff = Q(W @ M^T) @ M ⇒ artifact = W_eff @ M^T = Q(W @ M^T)
    (exact recovery for orthogonal M, no quantization in this test)."""
    g = NVFP4_GROUP_SIZE
    M = sylvester_hadamard(g, dtype=torch.float64)
    rot = CachedRotation(
        cluster_key="x", format_label="NVFP4", composed_matrix=M,
        group_size=g, insertion_kind="residual", online=False,
    )
    W = torch.randn(8, g * 2, dtype=torch.float64)
    # Simulate cache: W_eff = W @ M^T @ M = W (since no quant + orthogonal M)
    W_eff = apply_block_rotation_input(
        apply_block_rotation_input(W, M.t()), M
    )
    # Sanity: round-trip is identity for orthogonal M
    torch.testing.assert_close(W_eff, W, atol=1e-10, rtol=0)
    # Artifact recovery
    W_artifact = consumer_artifact_weight(W_eff, rot)
    expected = apply_block_rotation_input(W, M.t())  # W @ M^T
    torch.testing.assert_close(W_artifact, expected, atol=1e-10, rtol=0)


def test_consumer_artifact_preserves_shape():
    rot = _make_rotation()
    W = torch.randn(8, NVFP4_GROUP_SIZE * 2, dtype=torch.float32)
    W_artifact = consumer_artifact_weight(W, rot)
    assert W_artifact.shape == W.shape
    assert W_artifact.is_contiguous()


# ---------------------------------------------------------------------------
# producer_artifact_weight
# ---------------------------------------------------------------------------


def test_producer_artifact_applies_output_axis_rotation():
    """M @ W per G-block of output rows."""
    g = NVFP4_GROUP_SIZE
    M = sylvester_hadamard(g, dtype=torch.float64)
    rot = CachedRotation(
        cluster_key="x", format_label="NVFP4", composed_matrix=M,
        group_size=g, insertion_kind="v_o", online=False,
    )
    W = torch.randn(g * 2, 4, dtype=torch.float64)
    new_W, new_b = producer_artifact_weight(W, rot)
    assert new_b is None
    # Each G-row-block should be left-multiplied by M
    for blk in range(W.shape[0] // g):
        expected = M @ W[blk * g:(blk + 1) * g]
        torch.testing.assert_close(
            new_W[blk * g:(blk + 1) * g], expected, atol=1e-10, rtol=0
        )


def test_producer_artifact_rotates_bias_too():
    g = NVFP4_GROUP_SIZE
    M = sylvester_hadamard(g, dtype=torch.float64)
    rot = CachedRotation(
        cluster_key="x", format_label="NVFP4", composed_matrix=M,
        group_size=g, insertion_kind="v_o", online=False,
    )
    W = torch.randn(g * 2, 4, dtype=torch.float64)
    b = torch.randn(g * 2, dtype=torch.float64)
    new_W, new_b = producer_artifact_weight(W, rot, bias=b)
    assert new_b is not None
    for blk in range(b.shape[0] // g):
        expected = M @ b[blk * g:(blk + 1) * g]
        torch.testing.assert_close(
            new_b[blk * g:(blk + 1) * g], expected, atol=1e-10, rtol=0
        )


def test_producer_artifact_returns_contiguous():
    rot = _make_rotation(online=False)
    W = torch.randn(NVFP4_GROUP_SIZE * 2, 4)
    new_W, _ = producer_artifact_weight(W, rot)
    assert new_W.is_contiguous()


# ---------------------------------------------------------------------------
# iter_consumer_rewrites / iter_producer_rewrites
# ---------------------------------------------------------------------------


def test_iter_consumer_rewrites_yields_for_all_clusters():
    """Both ONLINE and OFFLINE clusters' consumers should be yielded."""
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={
            "a": _make_rotation(cluster_key="a", online=True),
            "b": _make_rotation(cluster_key="b", online=False),
        },
        consumer_to_cluster={"q_a": "a", "q_b": "b"},
        producer_to_cluster={},
    )
    rewrites = list(iter_consumer_rewrites(state))
    qnames = {r.qname for r in rewrites}
    assert qnames == {"q_a", "q_b"}


def test_iter_consumer_rewrites_sorted_deterministic():
    state = _make_state_offline_only()
    state2 = _make_state_offline_only()
    list1 = [r.qname for r in iter_consumer_rewrites(state)]
    list2 = [r.qname for r in iter_consumer_rewrites(state2)]
    assert list1 == list2
    assert list1 == sorted(list1)


def test_iter_producer_rewrites_skips_online_clusters():
    """ONLINE clusters have no producers — skip even if listed."""
    online_rot = _make_rotation(cluster_key="online", online=True)
    offline_rot = _make_rotation(cluster_key="offline", online=False)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"online": online_rot, "offline": offline_rot},
        consumer_to_cluster={},
        producer_to_cluster={
            "should_skip": "online",     # online ⇒ skip
            "should_yield": "offline",   # offline ⇒ yield
        },
    )
    rewrites = list(iter_producer_rewrites(state))
    qnames = {r.qname for r in rewrites}
    assert qnames == {"should_yield"}


def test_iter_consumer_rewrites_skips_unrotated_clusters():
    """Consumer qnames whose cluster has no rotation are silently skipped."""
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={},  # empty — no clusters were picked with rotation
        consumer_to_cluster={"q": "unrotated_cluster"},
        producer_to_cluster={},
    )
    assert list(iter_consumer_rewrites(state)) == []


# ---------------------------------------------------------------------------
# build_transforms_config
# ---------------------------------------------------------------------------


def test_build_transforms_config_emits_only_online_clusters():
    """Mix of online + offline ⇒ only online clusters appear in config."""
    online = _make_rotation(cluster_key="online", online=True)
    offline = _make_rotation(cluster_key="offline", online=False)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"online": online, "offline": offline},
        consumer_to_cluster={"q_on": "online", "q_off": "offline"},
        producer_to_cluster={},
    )
    config = build_transforms_config(state)
    groups = config["config_groups"]
    # Only the online cluster gets an entry
    assert len(groups) == 1
    only_key = next(iter(groups))
    assert "online" in only_key


def test_build_transforms_config_empty_state_yields_empty():
    state = HadamardDuQuantCacheState()
    config = build_transforms_config(state)
    assert config == {"config_groups": {}}


def test_build_transforms_config_entry_structure():
    state = _make_state_online_only()
    config = build_transforms_config(state)
    only_entry = next(iter(config["config_groups"].values()))
    assert only_entry["type"] == "random-matrix"
    assert only_entry["head_dim"] == NVFP4_GROUP_SIZE
    assert only_entry["randomize"] is False
    assert only_entry["requires_grad"] is False
    assert only_entry["precision"] == "float32"
    assert len(only_entry["apply"]) == 1
    apply_arg = only_entry["apply"][0]
    assert apply_arg["location"] == "input"
    assert apply_arg["inverse"] is False
    assert apply_arg["targets"] == ["model.layers.0.mlp.down_proj"]


def test_build_transforms_config_emits_hadamard_runtime_type():
    rot = _make_rotation(
        cluster_key="cluster_x",
        online=True,
        runtime_transform_type="hadamard",
    )
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"cluster_x": rot},
        consumer_to_cluster={"q": "cluster_x"},
        producer_to_cluster={},
    )
    config = build_transforms_config(state)
    only_entry = next(iter(config["config_groups"].values()))
    assert only_entry["type"] == "hadamard"
    assert only_entry["head_dim"] == NVFP4_GROUP_SIZE


def test_build_transforms_config_includes_all_consumers_of_online_cluster():
    """A cluster with multiple consumers lists them all in targets."""
    rot = _make_rotation(cluster_key="cluster_x", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"cluster_x": rot},
        consumer_to_cluster={"q_a": "cluster_x", "q_b": "cluster_x"},
        producer_to_cluster={},
    )
    config = build_transforms_config(state)
    only_entry = next(iter(config["config_groups"].values()))
    targets = only_entry["apply"][0]["targets"]
    assert set(targets) == {"q_a", "q_b"}
    assert targets == sorted(targets)  # deterministic order


def test_build_transforms_config_maps_runtime_targets():
    rot = _make_rotation(cluster_key="cluster_x", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"cluster_x": rot},
        consumer_to_cluster={
            "model.layers.0.mlp.down_proj": "cluster_x",
        },
        producer_to_cluster={},
    )
    config = build_transforms_config(
        state,
        qname_mapper=lambda qname: qname.replace(
            "model.layers.", "language_model.model.layers.", 1
        ),
    )
    targets = next(iter(config["config_groups"].values()))["apply"][0]["targets"]
    assert targets == ["language_model.model.layers.0.mlp.down_proj"]


def test_build_transforms_config_naming_replaces_dots():
    """Cluster keys with dots become config_groups keys with underscores."""
    rot = _make_rotation(cluster_key="model.layers.5.mlp.down", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"model.layers.5.mlp.down": rot},
        consumer_to_cluster={"q": "model.layers.5.mlp.down"},
        producer_to_cluster={},
    )
    config = build_transforms_config(state)
    only_key = next(iter(config["config_groups"]))
    assert "." not in only_key  # dots replaced
    assert "model__layers__5__mlp__down" in only_key


# ---------------------------------------------------------------------------
# build_rotation_safetensors_entries
# ---------------------------------------------------------------------------


def test_build_rotation_safetensors_only_online():
    """Only ONLINE clusters emit safetensors entries; OFFLINE don't (they
    fold into weights)."""
    online = _make_rotation(cluster_key="online", online=True)
    offline = _make_rotation(cluster_key="offline", online=False)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"online": online, "offline": offline},
        consumer_to_cluster={"q_on": "online", "q_off": "offline"},
        producer_to_cluster={"p_off": "offline"},
    )
    entries = build_rotation_safetensors_entries(state)
    keys = list(entries.keys())
    assert len(keys) == 1
    assert keys == [
        "q_on.hadamard_duquant__online_input.weight"
    ]


def test_build_rotation_safetensors_skips_hadamard_runtime_transform():
    rot = _make_rotation(
        cluster_key="cluster_x",
        online=True,
        runtime_transform_type="hadamard",
    )
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"cluster_x": rot},
        consumer_to_cluster={"q": "cluster_x"},
        producer_to_cluster={},
    )
    assert build_rotation_safetensors_entries(state) == {}


def test_build_rotation_safetensors_stores_vllm_scaled_dense_runtime_matrix():
    """vLLM scales dense online transforms by 1/sqrt(G) at runtime."""
    rot = _make_dense_nonsymmetric_rotation()
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={rot.cluster_key: rot},
        consumer_to_cluster={"q": rot.cluster_key},
        producer_to_cluster={},
    )
    entries = build_rotation_safetensors_entries(state)
    assert len(entries) == 1
    assert list(entries) == [
        f"q.hadamard_duquant__{rot.cluster_key.replace('.', '__')}_input.weight"
    ]
    only_tensor = next(iter(entries.values()))
    expected_runtime_weight = (
        rot.composed_matrix.detach().cpu().t().contiguous()
        * (float(rot.group_size) ** 0.5)
    )
    torch.testing.assert_close(only_tensor, expected_runtime_weight)
    # vLLM's generic transform wrapper multiplies by 1/sqrt(G), so storing
    # sqrt(G) * M^T yields the intended activation transform x @ M^T.


def test_build_rotation_safetensors_empty_state():
    state = HadamardDuQuantCacheState()
    assert build_rotation_safetensors_entries(state) == {}


def test_build_rotation_safetensors_tensors_are_cpu_contiguous():
    state = _make_state_online_only()
    entries = build_rotation_safetensors_entries(state)
    for t in entries.values():
        assert t.device.type == "cpu"
        assert t.is_contiguous()


def test_build_rotation_safetensors_emits_one_key_per_online_target():
    rot = _make_rotation(cluster_key="cluster_x", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"cluster_x": rot},
        consumer_to_cluster={"q_a": "cluster_x", "q_b": "cluster_x"},
        producer_to_cluster={},
    )
    entries = build_rotation_safetensors_entries(state)
    assert set(entries) == {
        "q_a.hadamard_duquant__cluster_x_input.weight",
        "q_b.hadamard_duquant__cluster_x_input.weight",
    }


def test_build_rotation_safetensors_maps_runtime_keys():
    rot = _make_rotation(cluster_key="cluster_x", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"cluster_x": rot},
        consumer_to_cluster={
            "model.layers.0.mlp.down_proj": "cluster_x",
        },
        producer_to_cluster={},
    )
    entries = build_rotation_safetensors_entries(
        state,
        qname_mapper=lambda qname: qname.replace(
            "model.layers.", "language_model.model.layers.", 1
        ),
    )
    assert set(entries) == {
        "language_model.model.layers.0.mlp.down_proj."
        "hadamard_duquant__cluster_x_input.weight"
    }


def test_vllm_online_gate_rejects_nvfp4_dense_online_clusters_by_default():
    rot = _make_rotation(cluster_key="cluster_x", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"cluster_x": rot},
        consumer_to_cluster={"q": "cluster_x"},
        producer_to_cluster={},
    )
    assert vllm_incompatible_online_clusters(state) == ["cluster_x"]
    try:
        assert_vllm_online_transforms_supported(state)
    except RuntimeError as exc:
        assert "research artifact" in str(exc)
    else:
        raise AssertionError("dense online random-matrix transform was not gated")
    assert_vllm_online_transforms_supported(state, allow_unsupported=True)


def test_vllm_online_gate_allows_nvfp4_hadamard_online_clusters():
    rot = _make_rotation(
        cluster_key="cluster_x",
        online=True,
        runtime_transform_type="hadamard",
    )
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"cluster_x": rot},
        consumer_to_cluster={"q": "cluster_x"},
        producer_to_cluster={},
    )
    assert vllm_incompatible_online_clusters(state) == []
    assert_vllm_online_transforms_supported(state)


def test_vllm_online_gate_allows_folded_and_research_override():
    offline = _make_rotation(cluster_key="offline", online=False)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"offline": offline},
        consumer_to_cluster={"q": "offline"},
        producer_to_cluster={"p": "offline"},
    )
    assert vllm_incompatible_online_clusters(state) == []
    assert_vllm_online_transforms_supported(state)

    online = _make_rotation(cluster_key="online", online=True)
    state.rotations_by_cluster["online"] = online
    assert_vllm_online_transforms_supported(state, allow_unsupported=True)


# ---------------------------------------------------------------------------
# Round-trip algebra: rotated artifact reproduces original computation
# ---------------------------------------------------------------------------


def test_consumer_artifact_runtime_algebra_reproduces_original():
    """Algebra check (no quantization, orthogonal M):

    Cache W_eff = W @ M^T @ M = W.
    Artifact W_a = W_eff @ M^T = W @ M^T.
    Runtime applies M^T to x.
    Output: (x @ M^T) @ W_a^T = (x @ M^T) @ (W @ M^T)^T
                              = x @ M^T @ M @ W^T = x @ W^T. ✓
    """
    g = NVFP4_GROUP_SIZE
    M = sylvester_hadamard(g, dtype=torch.float64)
    rot = CachedRotation(
        cluster_key="x", format_label="NVFP4", composed_matrix=M,
        group_size=g, insertion_kind="down_proj", online=True,
    )
    W = torch.randn(8, g * 2, dtype=torch.float64)
    x = torch.randn(32, g * 2, dtype=torch.float64)

    # No-quant cache simulation
    W_eff = apply_block_rotation_input(
        apply_block_rotation_input(W, M.t()), M
    )

    # Recover artifact
    W_artifact = consumer_artifact_weight(W_eff, rot)

    # Runtime applies M^T to x (because safetensors stores M^T)
    x_runtime = apply_block_rotation_input(x, M.t())

    y_artifact = x_runtime @ W_artifact.t()
    y_orig = x @ W.t()
    torch.testing.assert_close(y_artifact, y_orig, atol=1e-9, rtol=0)


# ---------------------------------------------------------------------------
# apply_rotations_to_model_in_place
# ---------------------------------------------------------------------------


class _SmallLinear(nn.Linear):
    """nn.Linear with explicit fp32 weights for deterministic tests."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        with torch.no_grad():
            self.weight.fill_(0.0)


class _MockAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class _MockLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = _MockAttn(dim)


class _MockModel(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([_MockLayer(dim)])


def test_apply_rotations_empty_state_is_noop():
    """Empty state returns zero counts and doesn't touch any weights."""
    model = _MockModel(dim=64)
    snapshot = {
        n: p.detach().clone() for n, p in model.named_parameters()
    }
    counts = apply_rotations_to_model_in_place(model, HadamardDuQuantCacheState())
    assert counts == {"consumer": 0, "producer": 0}
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, snapshot[n])


def test_apply_rotations_consumer_writes_rotated_weight():
    """Consumer's weight in the rotated cluster is replaced with W @ M^T per block."""
    model = _MockModel(dim=64)
    rot = _make_rotation(
        cluster_key="layers.0.attn", online=True, g=NVFP4_GROUP_SIZE,
    )
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"layers.0.attn": rot},
        consumer_to_cluster={"layers.0.self_attn.q_proj": "layers.0.attn"},
        producer_to_cluster={},
    )
    original_w = model.layers[0].self_attn.q_proj.weight.detach().clone()
    expected = apply_block_rotation_input(
        original_w.to(torch.float64), rot.composed_matrix.t()
    )
    counts = apply_rotations_to_model_in_place(model, state)
    assert counts["consumer"] == 1
    new_w = model.layers[0].self_attn.q_proj.weight
    # Original dtype, value matches expected (within fp32→f64 tolerance)
    torch.testing.assert_close(
        new_w.to(torch.float64), expected, atol=1e-6, rtol=1e-6
    )


def test_apply_rotations_only_rotates_consumer_qnames():
    """Linears not in the cluster's consumer list are untouched."""
    model = _MockModel(dim=64)
    rot = _make_rotation(cluster_key="layers.0.attn", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"layers.0.attn": rot},
        consumer_to_cluster={"layers.0.self_attn.q_proj": "layers.0.attn"},
        producer_to_cluster={},
    )
    k_before = model.layers[0].self_attn.k_proj.weight.detach().clone()
    v_before = model.layers[0].self_attn.v_proj.weight.detach().clone()
    apply_rotations_to_model_in_place(model, state)
    torch.testing.assert_close(model.layers[0].self_attn.k_proj.weight, k_before)
    torch.testing.assert_close(model.layers[0].self_attn.v_proj.weight, v_before)


def test_apply_rotations_producer_writes_output_axis_rotated_weight():
    """Producer's weight in OFFLINE cluster: M @ W per output G-block."""
    model = _MockModel(dim=64)
    rot = _make_rotation(
        cluster_key="layers.0.v_o", online=False, insertion_kind="v_o",
    )
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"layers.0.v_o": rot},
        consumer_to_cluster={"layers.0.self_attn.o_proj": "layers.0.v_o"},
        producer_to_cluster={"layers.0.self_attn.v_proj": "layers.0.v_o"},
    )
    v_before = model.layers[0].self_attn.v_proj.weight.detach().clone()
    counts = apply_rotations_to_model_in_place(model, state)
    assert counts["producer"] == 1
    v_after = model.layers[0].self_attn.v_proj.weight
    # Each G-row-block of v_after should be M @ v_before
    M = rot.composed_matrix.to(torch.float64)
    g = NVFP4_GROUP_SIZE
    for blk in range(v_before.shape[0] // g):
        expected = M @ v_before[blk * g:(blk + 1) * g].to(torch.float64)
        torch.testing.assert_close(
            v_after[blk * g:(blk + 1) * g].to(torch.float64),
            expected,
            atol=1e-6, rtol=1e-6,
        )


def test_apply_rotations_producer_skipped_for_online_cluster():
    """ONLINE clusters have no producers — even if listed, they're not rotated."""
    model = _MockModel(dim=64)
    rot = _make_rotation(cluster_key="layers.0.online", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"layers.0.online": rot},
        consumer_to_cluster={},
        producer_to_cluster={
            "layers.0.self_attn.v_proj": "layers.0.online",
        },
    )
    v_before = model.layers[0].self_attn.v_proj.weight.detach().clone()
    counts = apply_rotations_to_model_in_place(model, state)
    assert counts["producer"] == 0
    torch.testing.assert_close(model.layers[0].self_attn.v_proj.weight, v_before)


def test_apply_rotations_handles_missing_module_silently():
    """A qname that doesn't resolve to a module is skipped without error."""
    model = _MockModel(dim=64)
    rot = _make_rotation(cluster_key="ghost", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"ghost": rot},
        consumer_to_cluster={"layers.0.no_such_module": "ghost"},
        producer_to_cluster={},
    )
    counts = apply_rotations_to_model_in_place(model, state)
    assert counts["consumer"] == 0


def test_apply_rotations_to_layer_scopes_to_prefix():
    """Per-layer helper only touches Linears under the given layer prefix."""
    from prismaquant.hadamard_duquant_export import apply_rotations_to_layer

    model = nn.Module()
    model.layers = nn.ModuleList([
        _MockLayer(64),
        _MockLayer(64),
    ])

    rot0 = _make_rotation(cluster_key="layers.0.attn", online=True)
    rot1 = _make_rotation(cluster_key="layers.1.attn", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={
            "layers.0.attn": rot0,
            "layers.1.attn": rot1,
        },
        consumer_to_cluster={
            "layers.0.self_attn.q_proj": "layers.0.attn",
            "layers.1.self_attn.q_proj": "layers.1.attn",
        },
        producer_to_cluster={},
    )
    q0_before = model.layers[0].self_attn.q_proj.weight.detach().clone()
    q1_before = model.layers[1].self_attn.q_proj.weight.detach().clone()

    counts = apply_rotations_to_layer(model, "layers.0", state)
    assert counts["consumer"] == 1
    # Layer 1 untouched
    torch.testing.assert_close(model.layers[1].self_attn.q_proj.weight, q1_before)
    # Layer 0 rotated
    assert not torch.allclose(model.layers[0].self_attn.q_proj.weight, q0_before)


def test_apply_rotations_to_layer_empty_state_noop():
    from prismaquant.hadamard_duquant_export import apply_rotations_to_layer

    model = nn.Module()
    model.layers = nn.ModuleList([_MockLayer(64)])
    counts = apply_rotations_to_layer(
        model, "layers.0", HadamardDuQuantCacheState()
    )
    assert counts == {"consumer": 0, "producer": 0}


def test_apply_rotations_to_layer_skips_online_producers():
    """Online clusters have no producer Linears; producer pass should skip."""
    from prismaquant.hadamard_duquant_export import apply_rotations_to_layer

    model = nn.Module()
    model.layers = nn.ModuleList([_MockLayer(64)])

    rot = _make_rotation(cluster_key="layers.0.online", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"layers.0.online": rot},
        consumer_to_cluster={},
        producer_to_cluster={
            # ONLINE cluster with a producer listed — should NOT be rotated.
            "layers.0.self_attn.v_proj": "layers.0.online",
        },
    )
    v_before = model.layers[0].self_attn.v_proj.weight.detach().clone()
    counts = apply_rotations_to_layer(model, "layers.0", state)
    assert counts["producer"] == 0
    torch.testing.assert_close(model.layers[0].self_attn.v_proj.weight, v_before)


def test_state_fingerprint_stable_and_changes_on_state_change():
    from prismaquant.hadamard_duquant_export import state_fingerprint

    fp_empty = state_fingerprint(HadamardDuQuantCacheState())
    assert fp_empty == "none"

    rot = _make_rotation()
    state_a = HadamardDuQuantCacheState(
        rotations_by_cluster={rot.cluster_key: rot},
    )
    state_b = HadamardDuQuantCacheState(
        rotations_by_cluster={rot.cluster_key: rot},
    )
    fp_a = state_fingerprint(state_a)
    fp_b = state_fingerprint(state_b)
    assert fp_a == fp_b
    assert fp_a != fp_empty

    # Different matrix ⇒ different fingerprint
    rot_diff = _make_rotation()
    # Tweak the matrix to a different value
    M = sylvester_hadamard(NVFP4_GROUP_SIZE, dtype=torch.float64) * 2
    rot_diff_obj = CachedRotation(
        cluster_key=rot.cluster_key,
        format_label=rot.format_label,
        composed_matrix=M,
        group_size=rot.group_size,
        insertion_kind=rot.insertion_kind,
        online=rot.online,
    )
    state_c = HadamardDuQuantCacheState(
        rotations_by_cluster={rot.cluster_key: rot_diff_obj},
    )
    assert state_fingerprint(state_c) != fp_a


def test_apply_rotations_preserves_weight_dtype():
    """Weight dtype is preserved after rotation."""
    model = _MockModel(dim=64)
    model = model.to(torch.bfloat16)
    rot = _make_rotation(cluster_key="layers.0.attn", online=True)
    state = HadamardDuQuantCacheState(
        rotations_by_cluster={"layers.0.attn": rot},
        consumer_to_cluster={"layers.0.self_attn.q_proj": "layers.0.attn"},
        producer_to_cluster={},
    )
    apply_rotations_to_model_in_place(model, state)
    assert model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Round-trip algebra: rotated artifact reproduces original computation
# ---------------------------------------------------------------------------


def test_offline_producer_consumer_algebra_reproduces_original():
    """OFFLINE algebra: producer rotated on output, consumer rotated on input,
    no runtime transform. Composition should reproduce the original.
    """
    g = NVFP4_GROUP_SIZE
    M = sylvester_hadamard(g, dtype=torch.float64)
    rot = CachedRotation(
        cluster_key="x", format_label="NVFP4", composed_matrix=M,
        group_size=g, insertion_kind="v_o", online=False,
    )
    # Producer: emits an output of size G*2 from an input of size 4
    W_p = torch.randn(g * 2, 4, dtype=torch.float64)
    # Consumer: consumes the producer's output (size G*2) → output size 5
    W_c = torch.randn(5, g * 2, dtype=torch.float64)
    x = torch.randn(32, 4, dtype=torch.float64)

    # Apply offline rotation to producer (output axis)
    W_p_artifact, _ = producer_artifact_weight(W_p, rot)

    # For consumer: simulate cache W_eff = Q(W_c @ M^T) @ M (no quant)
    W_c_eff = apply_block_rotation_input(
        apply_block_rotation_input(W_c, M.t()), M
    )
    W_c_artifact = consumer_artifact_weight(W_c_eff, rot)

    # End-to-end forward (NO runtime transform — offline fold)
    y_artifact = (x @ W_p_artifact.t()) @ W_c_artifact.t()
    y_orig = (x @ W_p.t()) @ W_c.t()
    torch.testing.assert_close(y_artifact, y_orig, atol=1e-9, rtol=0)
