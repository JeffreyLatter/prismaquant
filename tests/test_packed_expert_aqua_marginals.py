"""The AQUA A-side must cover routed experts, not just the dense trunk.

`activation_dloss` needs three things: the dense weight, the per-output-row
`g_sq_sum`, and the format's activation grid. Dense Linears get `g_sq_sum` from
a `register_full_backward_hook`. A packed [E, M, N] expert parameter is an
`nn.Parameter` on an experts module, not an `nn.Linear`, so it has no such hook
and carried no A-side at all -- on a 35B-A3B that is 94% of the parameters
priced weight-only while the dense 5.5% got the activation term.

The `F.linear` interception in `install_packed_expert_hooks` is the equivalent
site: it already holds `(x, gy)` for each expert slice, INCLUDING down_proj's,
whose input is the post-SwiGLU intermediate.

What these tests pin:

1. the marginals equal a per-expert hand computation over routed tokens;
2. `expert_tokens` counts ROUTED tokens, not the global batch -- it is the
   denominator for the variance fit and nothing else;
3. the per-expert A-side is not recoverable from expert-aggregated statistics,
   which is why the card carries [E, *] arrays rather than 1-D vectors;
4. an expert that saw no calibration tokens prices at 0.0 rather than
   fabricating a distribution.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.format_cost_protocol import (
    _activation_dloss_packed,
    expert_act_sigma,
)
from prismaquant.sensitivity_card import SensitivityUnit, UnitTopology
from prismaquant.sensitivity_probe import install_packed_expert_hooks

from test_packed_expert_per_token_fisher import (
    _PackedWrap,
    _routed_batch,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _unit(name, out_f, in_f, n_e, n_tokens, **arrays):
    return SensitivityUnit(
        topology=UnitTopology(name=name, packed_group="g"),
        out_features=out_f, in_features=in_f,
        n_params=n_e * out_f * in_f, n_tokens=n_tokens,
        h_trace_raw=0.0, h_w2_sum_raw=0.0, w_norm_sq=0.0, w_max_abs=0.0,
        **arrays)


def test_marginals_match_a_per_expert_hand_computation():
    """gy² and x² summed over exactly the tokens the router sent to e."""
    torch.manual_seed(0)
    E, hidden, inter, T, K = 4, 5, 6, 16, 2
    model = _PackedWrap(E, hidden, inter).to(DEV)
    x, idx, w, v = _routed_batch(T, E, K, hidden, DEV)

    marg = {}
    install_packed_expert_hooks(
        model, accumulator={}, channel_accumulator={},
        marginal_accumulator=marg)
    xg = x.detach().requires_grad_(True)
    (model.experts(xg, idx, w) * v).sum().backward()

    assert set(marg) == {"experts.gate_up_proj", "experts.down_proj"}

    # gate_up_proj's input is the routed hidden state, so its per-expert
    # act_sq_sum is reconstructible from the batch without re-running the MLP.
    slot = marg["experts.gate_up_proj"]
    tokens = slot["expert_tokens"].cpu().numpy()
    act_sq = slot["expert_act_sq_sum"].cpu().numpy()
    for e in range(E):
        rows = (idx == e).any(dim=-1).nonzero().flatten()
        assert int(tokens[e]) == int(rows.numel())
        want = x[rows].float().pow(2).sum(dim=0).cpu().numpy()
        np.testing.assert_allclose(act_sq[e], want, rtol=2e-3, atol=1e-4)

    # Shapes: [E, M] on the output side, [E, N] on the input side. down_proj's
    # input is the post-SwiGLU intermediate -- the projection that had NO
    # activation statistics of any kind before this.
    down = marg["experts.down_proj"]
    assert down["expert_g_sq_sum"].shape == (E, hidden)
    assert down["expert_act_sq_sum"].shape == (E, inter)
    assert float(down["expert_act_sq_sum"].sum()) > 0.0


def test_expert_tokens_is_routed_not_global():
    """The variance denominator. Global would discount rare experts twice."""
    torch.manual_seed(1)
    E, hidden, inter, T, K = 8, 4, 4, 32, 1
    model = _PackedWrap(E, hidden, inter).to(DEV)
    x, idx, w, v = _routed_batch(T, E, K, hidden, DEV, seed=3)

    marg = {}
    install_packed_expert_hooks(
        model, accumulator={}, channel_accumulator={},
        marginal_accumulator=marg)
    (model.experts(x.detach().requires_grad_(True), idx, w) * v).sum().backward()

    tokens = marg["experts.gate_up_proj"]["expert_tokens"].cpu().numpy()
    assert tokens.sum() == T * K          # top-1 over T tokens
    assert (tokens < T).any()             # and NOT the global count per row


def test_absmax_is_a_bound_not_a_sum():
    torch.manual_seed(2)
    model = _PackedWrap(4, 5, 6).to(DEV)
    x, idx, w, v = _routed_batch(16, 4, 2, 5, DEV, seed=7)
    marg = {}
    install_packed_expert_hooks(
        model, accumulator={}, channel_accumulator={},
        marginal_accumulator=marg)
    (model.experts(x.detach().requires_grad_(True), idx, w) * v).sum().backward()
    slot = marg["experts.gate_up_proj"]
    amax = slot["expert_act_absmax"].cpu().numpy()
    for e in range(4):
        rows = (idx == e).any(dim=-1).nonzero().flatten()
        if rows.numel() == 0:
            continue
        want = x[rows].abs().amax(dim=0).float().cpu().numpy()
        np.testing.assert_allclose(amax[e], want, rtol=2e-3, atol=1e-5)


def test_sigma_divides_by_routed_tokens():
    """expert_act_sigma[e] uses expert_tokens[e], not the global n_tokens."""
    E, N = 3, 4
    unit = _unit("u", 2, N, E, n_tokens=1000,
                 expert_g_sq_sum=np.ones((E, 2), dtype=np.float32),
                 expert_act_sq_sum=np.full((E, N), 8.0, dtype=np.float32),
                 expert_tokens=np.array([2.0, 8.0, 32.0], dtype=np.float32))
    sig = expert_act_sigma(unit)
    np.testing.assert_allclose(sig[0], np.sqrt(8.0 / 2.0))
    np.testing.assert_allclose(sig[1], np.sqrt(8.0 / 8.0))
    np.testing.assert_allclose(sig[2], np.sqrt(8.0 / 32.0))
    # A global denominator would have made all three rows identical.
    assert sig[0][0] > sig[2][0] * 3


def test_packed_a_side_is_not_the_aggregated_a_side():
    """The reason the card carries [E, *] arrays instead of 1-D vectors.

    Two experts: one with a large weight and near-zero gradient, one with a
    small weight and a large gradient. Their aggregates are identical to a
    swapped pairing, but the true per-expert sum is not.
    """
    E, M, N, T = 2, 1, 1, 100
    w = np.array([[[10.0]], [[1.0]]], dtype=np.float32)      # [E, M, N]
    g = np.array([[1.0], [100.0]], dtype=np.float32)         # [E, M]
    unit = _unit("u", M, N, E, n_tokens=T,
                 expert_g_sq_sum=g,
                 expert_act_sq_sum=np.ones((E, N), dtype=np.float32),
                 expert_tokens=np.full(E, T, dtype=np.float32))
    var = np.ones((E, N), dtype=np.float64)
    got = _activation_dloss_packed(unit, w, var)
    # True: 0.5/T * (100*1 + 1*100) = 0.5/T * 200
    np.testing.assert_allclose(got, 0.5 * 200.0 / T, rtol=1e-9)

    # The collapsed form -- one aggregate g, one aggregate W -- gives
    # 0.5/T * (sum g)(sum W^2) = 0.5/T * 101 * 101, off by >50x.
    collapsed = 0.5 * float(g.sum()) * float((w ** 2).sum()) / T
    assert collapsed > got * 50


def test_zero_routed_expert_prices_zero_not_nan():
    E, M, N, T = 2, 2, 3, 64
    unit = _unit("u", M, N, E, n_tokens=T,
                 expert_g_sq_sum=np.ones((E, M), dtype=np.float32),
                 expert_act_sq_sum=np.zeros((E, N), dtype=np.float32),
                 expert_tokens=np.array([T, 0.0], dtype=np.float32))
    sig = expert_act_sigma(unit)
    assert np.all(np.isfinite(sig))
    assert float(sig[1].sum()) == 0.0
    got = _activation_dloss_packed(
        unit, np.ones((E, M, N), dtype=np.float32),
        np.zeros((E, N), dtype=np.float64))
    assert got == 0.0


def test_gpu_and_host_reductions_agree():
    """The A-side reduction runs on the GPU (principle 7); it must return the
    same number the host float64 path returns.

    Every term is a square times a variance, so the sum has no cancellation and
    float32 products accumulated in float64 stay within ~1e-7 relative. This
    pins that, rather than assuming it.
    """
    import prismaquant.format_cost_protocol as fcp

    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 512)).astype(np.float32)
    var = rng.random(512).astype(np.float64) * 1e-3
    g = rng.random(64).astype(np.float64) * 1e2

    got = fcp._weighted_row_sum(w, var, g)
    ref = float(g @ ((w.astype(np.float64) ** 2) @ var))
    np.testing.assert_allclose(got, ref, rtol=2e-6)


def test_packed_dloss_is_chunk_invariant():
    """Row blocking is an implementation detail, not a different quantity."""
    import prismaquant.format_cost_protocol as fcp

    rng = np.random.default_rng(1)
    E, M, N, T = 3, 40, 32, 512
    w = rng.standard_normal((E, M, N)).astype(np.float32)
    unit = _unit("u", M, N, E, n_tokens=T,
                 expert_g_sq_sum=rng.random((E, M)).astype(np.float32),
                 expert_act_sq_sum=np.ones((E, N), dtype=np.float32),
                 expert_tokens=np.full(E, T, dtype=np.float32))
    var = (rng.random((E, N)) * 1e-3)

    real = fcp._row_chunk
    try:
        fcp._row_chunk = lambda *_a, **_k: 1
        one = _activation_dloss_packed(unit, w, var)
        fcp._row_chunk = lambda *_a, **_k: 10_000
        whole = _activation_dloss_packed(unit, w, var)
    finally:
        fcp._row_chunk = real
    np.testing.assert_allclose(one, whole, rtol=2e-6)
