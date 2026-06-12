"""Generic vLLM fused-load invariant: a fused-sibling group missing a member
must ship its present members as BF16 (vLLM can't partially-quantize a fused
projection whose absent sibling carries no scale — e.g. Gemma4 k_eq_v layers
synthesize v=k and have no v_proj/v_scale, which KeyErrors on a quantized k).
"""
from prismaquant.decision_units import incomplete_fused_group_members
from prismaquant.allocator import incomplete_fused_group_dp_exclusions


class _FakeProfile:
    LEAF = {"qkv_proj": ("q_proj", "k_proj", "v_proj"),
            "gate_up_proj": ("gate_proj", "up_proj")}

    def fused_sibling_leaf_mapping(self):
        return self.LEAF

    def fused_sibling_group(self, q):
        for fused, members in self.LEAF.items():
            for m in members:
                if q.endswith("." + m):
                    return q[: -len(m)] + fused
        return q


P = _FakeProfile()
A = "model.layers.{}.self_attn.{}"
M = "model.layers.{}.mlp.{}"


def test_incomplete_qkv_pins_present_members():
    names = {A.format(5, "q_proj"), A.format(5, "k_proj")}  # no v_proj (k_eq_v)
    assert incomplete_fused_group_members(names, P) == names


def test_complete_qkv_pins_nothing():
    names = {A.format(0, "q_proj"), A.format(0, "k_proj"), A.format(0, "v_proj")}
    assert incomplete_fused_group_members(names, P) == set()


def test_complete_gate_up_pins_nothing():
    names = {M.format(0, "gate_proj"), M.format(0, "up_proj")}
    assert incomplete_fused_group_members(names, P) == set()


def test_incomplete_gate_up_pins_present():
    names = {M.format(0, "gate_proj")}  # up_proj missing
    assert incomplete_fused_group_members(names, P) == names


def test_non_fused_linears_ignored():
    names = {A.format(0, "o_proj"), M.format(0, "down_proj")}
    assert incomplete_fused_group_members(names, P) == set()


def test_mixed_realistic_set():
    names = {
        A.format(0, "q_proj"), A.format(0, "k_proj"), A.format(0, "v_proj"),  # sliding: complete
        A.format(5, "q_proj"), A.format(5, "k_proj"),                          # full k_eq_v: incomplete
        A.format(0, "o_proj"), M.format(0, "gate_proj"), M.format(0, "up_proj"),
    }
    assert incomplete_fused_group_members(names, P) == {
        A.format(5, "q_proj"), A.format(5, "k_proj")}


def test_no_profile_returns_empty():
    assert incomplete_fused_group_members({A.format(5, "q_proj")}, None) == set()


def test_allocator_excludes_incomplete_fused_members_from_dp():
    q = A.format(5, "q_proj")
    k = A.format(5, "k_proj")
    o = A.format(5, "o_proj")
    gate = M.format(0, "gate_proj")
    up = M.format(0, "up_proj")
    stats = {q: {}, k: {}, o: {}, gate: {}, up: {}}
    costs = {q: {}, k: {}, o: {}, gate: {}, up: {}}

    excluded = set(incomplete_fused_group_dp_exclusions(stats, costs, P))
    mutable_stats = {name: value for name, value in stats.items()
                     if name not in excluded}
    mutable_costs = {name: value for name, value in costs.items()
                     if name not in excluded}

    assert excluded == {q, k}
    assert q not in mutable_stats
    assert k not in mutable_costs
    assert o in mutable_stats
    assert gate in mutable_stats
    assert up in mutable_costs
