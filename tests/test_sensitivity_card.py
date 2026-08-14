"""Acceptance tests for the Sensitivity Card / format-cost seam.

The load-bearing test is `test_scalar_model_reproduces_allocator_solver`: the
card must reproduce today's allocator cost *exactly*, because it is the same
formula fed the same numbers. Any drift there is a bug, not a modelling choice.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismaquant.allocator_solver import predicted_dloss
from prismaquant.format_cost_protocol import (
    CostModel,
    FormatDescriptor,
    activation_dloss,
    price,
    uniform_act_quant_variance,
    weight_dloss_marginal,
    weight_dloss_scalar,
)
from prismaquant.sensitivity_card import (
    CardProvenance,
    Currency,
    RenderBasis,
    SensitivityCard,
    SensitivityUnit,
    UnitTopology,
)

OUT, IN, TOKENS = 8, 6, 100


def _synthetic_unit(seed: int = 0, *, with_vectors: bool = True,
                    name: str = "model.layers.0.self_attn.q_proj",
                    source_dtype: str = "bfloat16") -> SensitivityUnit:
    """Build a unit whose vectors are the TRUE marginals of a real H.

    H is constructed the way the probe constructs it -- a sum over tokens of
    outer(g^2, x^2) -- so the marginal identities are exact by construction and
    the test checks our arithmetic, not our assumptions.
    """
    rng = np.random.default_rng(seed)
    g_sq = rng.random((TOKENS, OUT)) + 0.1
    x_sq = rng.random((TOKENS, IN)) + 0.1
    H = g_sq.T @ x_sq                      # [OUT, IN], exactly the probe's chunk_h
    w = rng.standard_normal((OUT, IN))

    kwargs = {}
    if with_vectors:
        kwargs = dict(
            fisher_row=H.sum(axis=1),
            fisher_col=H.sum(axis=0),
            act_sq_sum=x_sq.sum(axis=0),
            g_sq_sum=g_sq.sum(axis=0),
            act_absmax=np.sqrt(x_sq).max(axis=0),
        )

    return SensitivityUnit(
        topology=UnitTopology(name=name, layer_index=0, role="q",
                              fused_group="L0.attn", source_dtype=source_dtype),
        out_features=OUT, in_features=IN, n_params=OUT * IN, n_tokens=TOKENS,
        h_trace_raw=float(H.sum()),
        h_w2_sum_raw=float((H * w ** 2).sum()),
        w_norm_sq=float((w ** 2).sum()), w_max_abs=float(np.abs(w).max()),
        **kwargs,
    )


def _provenance() -> CardProvenance:
    return CardProvenance(
        model_id="test/model", calib_hash="deadbeef" * 8, n_calib_samples=8,
        seq_len=512, probe_commit="abc1234", render_basis=RenderBasis.RTN)


# ----------------------------------------------------------------- invariants


def test_marginal_identity_holds():
    """sum(row) == sum(col) == h_trace_raw. Free consistency check."""
    u = _synthetic_unit()
    assert np.isclose(u.fisher_row.sum(), u.h_trace_raw)
    assert np.isclose(u.fisher_col.sum(), u.h_trace_raw)
    u.validate()


def test_validate_rejects_inconsistent_marginals():
    u = _synthetic_unit()
    bad = SensitivityUnit(**{**u.__dict__, "fisher_row": u.fisher_row * 2.0})
    with pytest.raises(ValueError, match="does not match h_trace_raw"):
        bad.validate()


def test_validate_rejects_wrong_shape():
    u = _synthetic_unit()
    bad = SensitivityUnit(**{**u.__dict__, "fisher_col": np.ones(IN + 1)})
    with pytest.raises(ValueError, match="expected"):
        bad.validate()


# ------------------------------------------------- backward-compatibility gate


def test_scalar_model_reproduces_allocator_solver():
    """The card's scalar path IS allocator_solver.predicted_dloss, exactly."""
    u = _synthetic_unit()
    for weight_mse in (1e-6, 3.25e-4, 0.5):
        for gain in (1.0, 0.75):
            assert weight_dloss_scalar(u, weight_mse, gain) == predicted_dloss(
                u.h_trace, weight_mse, gain)


def test_scalar_path_survives_a_card_without_vectors():
    """A vector-less card degrades to today's behaviour, it does not break."""
    u = _synthetic_unit(with_vectors=False)
    assert not u.has_vectors
    dw_sq = np.full((OUT, IN), 4e-4)
    got = weight_dloss_marginal(u, dw_sq)
    assert got == predicted_dloss(u.h_trace, float(dw_sq.mean()))


# --------------------------------------------------------- the marginal model


def test_marginal_model_matches_exact_fisher_on_rank1():
    """When H is genuinely rank-1, the marginal model is EXACT.

    This is the sharpest available check that the quadratic form and its
    normalization are right: build a single-token H (hence exactly rank-1) and
    compare against the full 0.5 * sum H*dW^2 computed elementwise.
    """
    rng = np.random.default_rng(7)
    g_sq = rng.random((1, OUT)) + 0.1
    x_sq = rng.random((1, IN)) + 0.1
    H = g_sq.T @ x_sq
    u = SensitivityUnit(
        topology=UnitTopology(name="u", source_dtype="bfloat16"),
        out_features=OUT, in_features=IN, n_params=OUT * IN, n_tokens=1,
        h_trace_raw=float(H.sum()), h_w2_sum_raw=0.0,
        w_norm_sq=1.0, w_max_abs=1.0,
        fisher_row=H.sum(axis=1), fisher_col=H.sum(axis=0),
    )
    dw_sq = rng.random((OUT, IN)) * 1e-3
    exact = 0.5 * float((H * dw_sq).sum())
    assert np.isclose(weight_dloss_marginal(u, dw_sq), exact, rtol=1e-10)


def test_marginal_model_differs_from_scalar_when_sensitivity_is_uneven():
    """The point of the marginals: they see structure the scalar cannot.

    Concentrate the weight error on the LOW-sensitivity output channels and the
    marginal model must price it below the scalar model, which averages.
    """
    u = _synthetic_unit(seed=3)
    order = np.argsort(u.fisher_row)
    dw_sq = np.zeros((OUT, IN))
    dw_sq[order[:2], :] = 1e-3            # error only on the 2 least sensitive
    cheap = weight_dloss_marginal(u, dw_sq)

    dw_sq2 = np.zeros((OUT, IN))
    dw_sq2[order[-2:], :] = 1e-3          # same magnitude, most sensitive
    dear = weight_dloss_marginal(u, dw_sq2)

    assert cheap < dear
    # The scalar model is blind to the difference: identical mean error.
    assert np.isclose(dw_sq.mean(), dw_sq2.mean())
    assert weight_dloss_scalar(u, float(dw_sq.mean())) == \
        weight_dloss_scalar(u, float(dw_sq2.mean()))


# ------------------------------------------------------------------ AQUA-AURA


def test_activation_dloss_uses_output_space_fisher_not_h_trace():
    """A card lacking g_sq_sum returns None -- never 0.0.

    An unmeasured activation cost must not read as a free one.
    """
    u = _synthetic_unit()
    stripped = SensitivityUnit(**{**u.__dict__, "g_sq_sum": None})
    w = np.ones((OUT, IN))
    var = np.full(IN, 1e-4)
    assert activation_dloss(stripped, w, var) is None
    assert activation_dloss(u, w, var) is not None


def test_w4a4_and_w4a8_are_distinct_candidates():
    """The whole point of AQUA-AURA: A4 must cost more than A8.

    Under CostModel.AQUA the two formats differ ONLY in act_bits, so any
    difference in predicted loss is the activation term doing its job. Under the
    old scalar model they are indistinguishable.
    """
    u = _synthetic_unit()
    w = np.random.default_rng(11).standard_normal((OUT, IN))

    class _Plug:
        def __init__(self, desc):
            self.descriptor = desc

        def weight_error(self, unit, weight):
            return np.full((unit.out_features, unit.in_features), 4e-4)

    a4 = _Plug(FormatDescriptor(name="W4A4", weight_bits=4.0, act_bits=4))
    a8 = _Plug(FormatDescriptor(name="W4A8", weight_bits=4.0, act_bits=8))

    c4 = price(u, w, a4, render_basis=RenderBasis.RTN, model=CostModel.AQUA)
    c8 = price(u, w, a8, render_basis=RenderBasis.RTN, model=CostModel.AQUA)

    assert c4.act_dloss is not None and c8.act_dloss is not None
    assert c4.act_dloss > c8.act_dloss, "4-bit activations must cost more than 8-bit"
    assert c4.to_predicted_dloss() > c8.to_predicted_dloss()
    # Same weight side -- the difference is entirely the A-side.
    assert c4.weight_dloss == c8.weight_dloss
    # And the two are identical under the weight-only model, which is the bug
    # AQUA-AURA exists to fix.
    s4 = price(u, w, a4, render_basis=RenderBasis.RTN, model=CostModel.MARGINAL)
    s8 = price(u, w, a8, render_basis=RenderBasis.RTN, model=CostModel.MARGINAL)
    assert s4.to_predicted_dloss() == s8.to_predicted_dloss()


def test_act_quant_variance_shrinks_with_bits():
    u = _synthetic_unit()
    v4 = uniform_act_quant_variance(u, 4)
    v8 = uniform_act_quant_variance(u, 8)
    # Each extra bit halves the step, so variance drops 4x per bit.
    assert np.allclose(v4 / v8, 4.0 ** 4)


# ------------------------------------------------------- legality and currency


def test_passthrough_is_refused_when_source_dtype_mismatches():
    """BF16/FP8_SOURCE are passthrough-only; never synthesize them."""
    u = _synthetic_unit(source_dtype="float8_e4m3fn")
    w = np.zeros((OUT, IN))

    class _Plug:
        descriptor = FormatDescriptor(
            name="BF16", weight_bits=16.0, passthrough=True,
            requires_source_dtype="bfloat16")

        def weight_error(self, unit, weight):
            return np.zeros((unit.out_features, unit.in_features))

    assert price(u, w, _Plug(), render_basis=RenderBasis.RTN) is None

    u_bf16 = _synthetic_unit(source_dtype="bfloat16")
    assert price(u_bf16, w, _Plug(), render_basis=RenderBasis.RTN) is not None


def test_costs_leave_only_in_delta_loss_currency():
    u = _synthetic_unit()
    w = np.zeros((OUT, IN))

    class _Plug:
        descriptor = FormatDescriptor(name="NVFP4", weight_bits=4.25)

        def weight_error(self, unit, weight):
            return np.full((unit.out_features, unit.in_features), 1e-4)

    c = price(u, w, _Plug(), render_basis=RenderBasis.RTN)
    c.assert_currency(Currency.DELTA_LOSS)
    with pytest.raises(ValueError, match="Mixing bases"):
        c.assert_currency(Currency.WEIGHT_MSE)


# ------------------------------------------------------------ card-level rules


def test_calibration_is_identity():
    a = SensitivityCard(_provenance(), [_synthetic_unit()])
    other = CardProvenance(**{**_provenance().__dict__, "calib_hash": "f" * 64})
    b = SensitivityCard(other, [_synthetic_unit()])
    with pytest.raises(ValueError, match="different\n?\\s*calibrations|Calibration is identity"):
        a.assert_compatible(b)


def test_render_basis_mismatch_is_refused():
    a = SensitivityCard(_provenance(), [_synthetic_unit()])
    other = CardProvenance(
        **{**_provenance().__dict__, "render_basis": RenderBasis.COMPENSATED})
    b = SensitivityCard(other, [_synthetic_unit()])
    with pytest.raises(ValueError, match="render basis mismatch"):
        a.assert_compatible(b)


def test_roundtrip_npz_preserves_everything(tmp_path):
    card = SensitivityCard(_provenance(), [
        _synthetic_unit(0, name="a"), _synthetic_unit(1, name="b")])
    card.validate()
    path = str(tmp_path / "card.npz")
    card.to_npz(path)
    back = SensitivityCard.from_npz(path)
    back.validate()

    assert len(back) == 2
    assert back.provenance == card.provenance
    for name in card.names():
        u0, u1 = card[name], back[name]
        assert u0.topology == u1.topology
        assert u0.h_trace_raw == pytest.approx(u1.h_trace_raw)
        for field in ("fisher_row", "fisher_col", "act_sq_sum", "g_sq_sum"):
            assert np.allclose(getattr(u0, field), getattr(u1, field))


def test_npz_is_loadable_without_pickle(tmp_path):
    """A shareable artifact must not execute arbitrary objects on load."""
    card = SensitivityCard(_provenance(), [_synthetic_unit()])
    path = str(tmp_path / "card.npz")
    card.to_npz(path)
    with np.load(path, allow_pickle=False) as data:   # must not raise
        assert "__header__" in data


def test_structure_is_carried_but_not_policy():
    """Sibling identity travels; must-share-format does not."""
    card = SensitivityCard(_provenance(), [
        _synthetic_unit(0, name="model.layers.0.self_attn.q_proj"),
        _synthetic_unit(1, name="model.layers.0.self_attn.k_proj"),
    ])
    groups = card.fused_groups()
    assert set(groups["L0.attn"]) == {
        "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"}
    # The card exposes no serving policy at all -- that is the consumer's job.
    assert not hasattr(card, "must_share_format")
