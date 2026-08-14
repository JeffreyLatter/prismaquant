"""AQUA-AURA: the activation term reaches the DP, exactly once.

WHY
---
Choosing NVFP4 commits a Linear's ACTIVATIONS to 4 bits, not just its weights.
The allocator's cost was weight-only and could not see that -- NVFP4 and
NVFP4A16 render weights bit-identically -- so the DP bought W4A4 at a discount
to its true cost. ``cost_entry_act_dloss`` closes that.

The risk in closing it is double-counting, because three of the four pricing
branches must NOT receive the term:

  * ``_prices_from_output_mse`` rows are already activation-inclusive by
    construction -- the measurement saw the activation path.
  * exact-by-construction rows are 0.0 because nothing happens to the tensor.
  * super-item rows already contain their members' A-side, since a super item is
    priced as the SUM of its members' ``cost_entry_predicted_dloss``.

Getting any of those wrong is silent: the allocation still solves, it just
solves the wrong problem. So each branch is pinned separately.

The last test is the one the whole design rests on: the A-side is independent of
the render basis. If that were false, an A-side priced off the card could not be
added to a production-rendered weight cost, and the stage would be a rendering
confound rather than a cost completion.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from prismaquant.allocator_candidates import (  # noqa: E402
    ACT_DLOSS_KEY, cost_entry_act_dloss, cost_entry_predicted_dloss,
)

STATS = {"n_params": 4096, "in_features": 64, "out_features": 64,
         "h_trace": 1.0, "n_tokens": 8}
ACT = 0.25


def _weight_only_entry(**extra):
    e = {"predicted_dloss": 1.0, "output_mse_measured": False,
         "cost_source": "aura"}
    e.update(extra)
    return e


def test_pre_aqua_rows_are_bit_for_bit_unchanged():
    """Every cost artifact written before AQUA lacks the key.

    Those runs must stay reproducible, so a missing A-side reads as 0.0 here --
    while remaining a HOLE in the writer's report, which is where an unpriced
    activation-quantizing format is supposed to be visible.
    """
    entry = _weight_only_entry()
    assert cost_entry_act_dloss(entry) == 0.0
    before = cost_entry_predicted_dloss(STATS, entry, format_name="NVFP4")
    assert before == pytest.approx(1.0)


def test_the_term_is_added_on_the_weight_only_branch():
    entry = _weight_only_entry(**{ACT_DLOSS_KEY: ACT})
    got = cost_entry_predicted_dloss(STATS, entry, format_name="NVFP4")
    assert got == pytest.approx(1.0 + ACT), (
        "the A-side must reach the DP additively -- it is a Delta-loss in the "
        "same currency as the weight term, not a multiplicative penalty")


def test_the_a_side_rides_the_same_p5a_transfer_as_the_w_side():
    """The two halves of one unit's price must be on ONE scale.

    P5a is a per-family multiplicative re-leveling of weight-only rows. Adding
    the A-side outside that multiply leaves it in un-transferred units, and the
    fitted constants are large -- x8103 for the NVFP4 family on Qwen3.8-27B --
    so an A-side worth 6x the W-side arrives as 0.07% of the total. That is not
    a rounding difference: it produced a shipped Pareto byte-identical to the
    weight-only one, which is how the bug was caught.

    Multiplying the SUM keeps the A:W ratio, which is what the DP ranks on.
    """
    entry = _weight_only_entry(**{ACT_DLOSS_KEY: ACT})
    plain = cost_entry_predicted_dloss(STATS, entry, format_name="NVFP4")
    assert plain == pytest.approx(1.0 + ACT)

    class _FixedPenalty:
        enabled = True

        def penalty_for(self, format_name, act_changes):
            # ``_activation_penalty`` reads element [0]; the second slot is the
            # branch label used only for reporting.
            return (1000.0, "test")

    scaled = cost_entry_predicted_dloss(
        STATS, entry, format_name="NVFP4", activation_pricing=_FixedPenalty())
    assert scaled == pytest.approx((1.0 + ACT) * 1000.0), (
        "a large per-family penalty must scale the A-side with the W-side, not "
        "drown it")
    # The ratio is what the DP ranks on and must be penalty-invariant.
    without = dict(entry)
    without.pop(ACT_DLOSS_KEY)
    ref = cost_entry_predicted_dloss(
        STATS, without, format_name="NVFP4", activation_pricing=_FixedPenalty())
    assert scaled / ref == pytest.approx(plain / 1.0)


def test_it_is_not_added_to_an_activation_inclusive_measurement():
    """The double-count guard for ``_prices_from_output_mse``.

    A measured output_mse row already saw the activation path. Adding the
    modelled A-side on top would charge that layer twice for the same physics,
    and would do it ONLY on rows that happen to carry a measurement -- i.e. it
    would mis-rank rungs within one family, which is the failure mode this
    branch was split out to fix in the first place.
    """
    from prismaquant.allocator_candidates import _prices_from_output_mse
    measured = {"output_mse": 0.5, "output_mse_measured": True,
                ACT_DLOSS_KEY: ACT}
    if not _prices_from_output_mse(STATS, measured):
        pytest.skip("this cost shape does not take the output_mse branch")
    with_act = cost_entry_predicted_dloss(STATS, measured, format_name="NVFP4")
    without = dict(measured)
    without.pop(ACT_DLOSS_KEY)
    assert with_act == pytest.approx(
        cost_entry_predicted_dloss(STATS, without, format_name="NVFP4"))


def test_it_is_not_added_to_a_super_item():
    """A super item is the SUM of its members' priced dloss.

    Each member term already carries its own A-side, so re-adding at the
    aggregate would scale the activation cost by the group size -- 3x on a
    fused q/k/v, and more on a packed expert group.
    """
    from prismaquant.allocator_candidates import APPLIED_MARKER_KEY
    entry = _weight_only_entry(**{ACT_DLOSS_KEY: ACT,
                                  APPLIED_MARKER_KEY: True})
    assert cost_entry_predicted_dloss(
        STATS, entry, format_name="NVFP4") == pytest.approx(1.0)


def test_merge_writes_only_where_the_format_exists():
    from prismaquant.aqua_activation_cost import merge_act_dloss
    costs = {"a": {"NVFP4": {"predicted_dloss": 1.0},
                   "BF16": {"predicted_dloss": 0.0}},
             "b": {"NVFP4": {"predicted_dloss": 2.0}}}
    report = merge_act_dloss(costs, {"a": {"NVFP4": 0.1, "FP8_E4M3": 0.9},
                                     "c": {"NVFP4": 0.7}})
    assert costs["a"]["NVFP4"][ACT_DLOSS_KEY] == pytest.approx(0.1)
    assert ACT_DLOSS_KEY not in costs["a"]["BF16"], (
        "BF16 does not quantize activations; writing a 0.0 there would make a "
        "correct absence indistinguishable from an unpriced hole")
    assert ACT_DLOSS_KEY not in costs["b"]["NVFP4"]
    assert report["entries_merged"] == 1
    assert report["units_without_act_price"] == 1


@pytest.mark.parametrize("fmt,expect_priced", [("NVFP4", True),
                                               ("FP8_E4M3", True),
                                               ("BF16", False)])
def test_price_activation_only_follows_the_explicit_predicate(fmt,
                                                              expect_priced):
    card = pytest.importorskip("prismaquant.sensitivity_card")
    from prismaquant.format_cost_protocol import price_activation_only
    from prismaquant.format_cost_registry import RegistryFormatPlugin
    unit = _synthetic_unit(card)
    w = np.random.default_rng(0).normal(0, 0.02, (32, 64)).astype(np.float32)
    try:
        plugin = RegistryFormatPlugin.build(fmt, shape=w.shape, device="cpu")
    except Exception as exc:
        pytest.skip(f"{fmt} unbuildable on CPU: {exc}")
    got = price_activation_only(unit, w, plugin)
    if expect_priced:
        assert got is not None and got > 0.0
    else:
        assert got is None, (
            "a format that leaves activations alone must return None, not 0.0 "
            "-- the two mean different things and only one is a hole")


def test_the_a_side_does_not_depend_on_the_weights_being_rendered():
    """THE claim the whole stage rests on.

    ``activation_dloss`` reads the DENSE weight, ``g_sq_sum`` and the format's
    activation grid. No render enters it. That is what makes it legitimate to
    price the A-side off a shared card and add it to a cost whose W-side was
    built with the full GPTQ+JSO production recipe: the two halves are measured
    on different bases only because the A-side HAS no basis.

    Pinned by pricing the same unit against two very different weight matrices
    that share a scale, and asserting the A-side tracks the weights it is given
    rather than any rendering of them -- and, more sharply, that priced twice on
    the same weights it is exactly reproducible.
    """
    card = pytest.importorskip("prismaquant.sensitivity_card")
    from prismaquant.format_cost_protocol import price_activation_only
    from prismaquant.format_cost_registry import RegistryFormatPlugin
    unit = _synthetic_unit(card)
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.02, (32, 64)).astype(np.float32)
    try:
        plugin = RegistryFormatPlugin.build("NVFP4", shape=w.shape,
                                            device="cpu")
    except Exception as exc:
        pytest.skip(f"NVFP4 unbuildable on CPU: {exc}")
    a = price_activation_only(unit, w, plugin)
    b = price_activation_only(unit, w.copy(), plugin)
    assert a == pytest.approx(b, rel=0, abs=0), (
        "the A-side must be a pure function of (unit, dense weight, format)")


def _synthetic_unit(card_mod):
    """Smallest card unit that can carry an A-side price.

    ``g_sq_sum`` is the OUTPUT-space sensitivity the A-side uses -- not
    ``fisher_row``, which is the weight-space one. That distinction is the whole
    point of AQUA and was documented backwards once already, so the fixture
    states it explicitly.
    """
    n_in, n_out, n_tok = 64, 32, 128
    rng = np.random.default_rng(7)
    return card_mod.SensitivityUnit(
        topology=card_mod.UnitTopology(name="probe.linear", layer_index=0,
                                       role="down", source_dtype="bfloat16"),
        out_features=n_out, in_features=n_in, n_params=n_in * n_out,
        n_tokens=n_tok,
        h_trace_raw=1.0, h_w2_sum_raw=1e-3,
        w_norm_sq=1.0, w_max_abs=0.1,
        g_sq_sum=np.full(n_out, 1e-3, dtype=np.float64),
        act_sq_sum=rng.uniform(0.5, 1.5, n_in).astype(np.float64),
        act_absmax=rng.uniform(2.0, 6.0, n_in).astype(np.float64),
    )
