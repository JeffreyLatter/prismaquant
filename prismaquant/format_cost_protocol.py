"""The costing seam: turn a Sensitivity Card + a format into an allocator candidate.

WHERE THIS SITS
---------------
``allocator_solver.py`` already defines the entire contract the optimizer needs::

    @dataclass
    class Candidate:
        fmt: str
        bits_per_param: float
        memory_bytes: int
        predicted_dloss: float

and the scalar model that fills the last field::

    def predicted_dloss(h_trace, weight_mse, gain=1.0) -> float:
        return 0.5 * float(h_trace) * float(weight_mse) * float(gain)

This module does not replace that seam -- it *feeds* it. A format plugin turns
``(SensitivityUnit, W, FormatDescriptor)`` into a :class:`CostComponents`, and
:meth:`CostComponents.to_predicted_dloss` collapses that to the one float the
knapsack DP consumes. The optimizer is untouched, so an arbitrary format menu
becomes an arbitrary list of plugins rather than a change to the solver.

THE THREE COST MODELS, IN INCREASING FIDELITY
---------------------------------------------
1. ``SCALAR`` -- ``0.5 * h_trace * weight_mse``. Exactly today's behaviour, and
   the fallback when a card carries no vectors. Reproducing this byte-identically
   from a card is the primary acceptance test for this module: same formula,
   same inputs, so any drift is a bug and not a modelling choice.

2. ``MARGINAL`` -- weight error weighted by the per-channel Fisher marginals
   instead of by a single scalar. The scalar model is the rank-0 collapse of the
   same object; this is the rank-1 reconstruction. It is strictly more
   descriptive per unit and costs ``out + in`` floats.

3. ``AQUA`` -- adds an activation-quantization term so that W4A4 and W4A8 stop
   being the same candidate. See below.

AQUA-AURA: WHY THE A-SIDE NEEDS ITS OWN SENSITIVITY
---------------------------------------------------
``h_trace`` is a *weight-space* curvature. An activation-quantization error is
an *input-side* perturbation ``x -> x + dx``, which reaches the loss as
``dy = W dx``. Multiplying an input-side error by a weight-space sensitivity is
a currency error of exactly the kind ``activation_fair_pricing.py`` is the
autopsy of, so this module refuses to do it.

Under a diagonal model the output perturbation is::

    E||W dx||^2 = sum_j  var(dx_j) * ||W[:, j]||^2

and it becomes a loss delta through the OUTPUT-space Fisher ``g_sq_sum[o]``::

    dLoss_a ~= 0.5 * sum_o  g_sq[o] * (W[o, :]^2 . var_dx)

which is what :func:`activation_dloss` computes. Note this uses ``g_sq_sum``,
never ``h_trace``.

STATUS: the A-side term is a **screening surrogate**. It is research-tier until
a served W4A4-vs-W4A8 A/B exists. ``activation_fair_pricing.py`` is deliberately
left untouched: superseding it is a promotion decision on evidence, not a
drive-by refactor.

WHAT THIS MODULE REFUSES TO DO
------------------------------
- Sum costs measured in different currencies (raises).
- Apply a passthrough format to a unit whose source dtype does not already match
  (BF16/FP8_SOURCE are passthrough-only; synthesising them wastes 8 bpp).
- Invent a speed/quality scalarization constant. Speed and quality are returned
  as separate axes; the choice between them is a frontier selection, not a
  weighted sum.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Protocol

import numpy as np

from .sensitivity_card import Currency, RenderBasis, SensitivityUnit


class CostModel(enum.Enum):
    """Fidelity tier used to price a unit. Recorded on every cost."""

    SCALAR = "scalar"
    MARGINAL = "marginal"
    AQUA = "aqua"


@dataclasses.dataclass(frozen=True)
class FormatDescriptor:
    """Everything the costing seam needs to know about a format.

    This is intentionally a *description*, not an implementation: a downstream
    author naming a platform supplies these fields for their own formats without
    touching PrismaQuant. ``weight_bits`` and friends describe storage;
    ``act_bits`` describes what the kernel does to activations at serve time.
    """

    name: str
    #: Effective stored bits per weight parameter, INCLUDING scale/codebook
    #: overhead amortized over the group. This is what the byte budget spends.
    weight_bits: float
    #: WIDTH of the serve-time activation grid, when there is one. This is the
    #: quantity the error model needs; it is NOT the predicate for "does this
    #: format quantize activations" -- see :attr:`quantizes_activations`.
    #: Re-deriving that predicate from a width is the bug
    #: `test_activation_quant_predicate_has_one_definition` exists to prevent,
    #: because consumers that did so disagreed with the allocator's gate.
    act_bits: int | None = None
    #: THE predicate. Explicit data rather than an inference, sourced from
    #: ``FormatSpec.act_quant_changes_input`` by `format_cost_registry`. This
    #: field is the entire difference between W4A4 and W4A16.
    quantizes_activations: bool = False
    #: Quantization group size along the input dimension, if any.
    group_size: int | None = None
    #: True when the format is a verbatim copy of an already-matching source
    #: tensor (BF16, FP8_SOURCE). Legal only when the source dtype matches.
    passthrough: bool = False
    #: Source dtype this passthrough format requires, e.g. "bfloat16".
    requires_source_dtype: str | None = None
    #: Optional relative serve-time throughput hint, higher is faster. Used only
    #: to report the speed axis of a frontier; never folded into the loss.
    speed_index: float | None = None

    def is_legal_for(self, unit: SensitivityUnit) -> bool:
        """Passthrough integrity: never synthesize a passthrough format."""
        if not self.passthrough:
            return True
        if self.requires_source_dtype is None:
            return False
        return unit.topology.source_dtype == self.requires_source_dtype


@dataclasses.dataclass(frozen=True)
class CostComponents:
    """A priced (unit, format) pair, with the currency of every part declared."""

    unit_name: str
    format_name: str
    model: CostModel
    render_basis: RenderBasis

    #: Weight-side error, in WEIGHT_MSE currency.
    weight_mse: float
    #: Weight-side predicted loss delta, in DELTA_LOSS currency.
    weight_dloss: float
    #: Activation-side predicted loss delta, in DELTA_LOSS currency.
    #: ``None`` when the format does not quantize activations, or when the card
    #: lacks the vectors needed to price it -- which is NOT the same as zero and
    #: is kept distinct so a missing measurement never reads as "free".
    act_dloss: float | None = None

    bits_per_param: float = 0.0
    memory_bytes: int = 0
    speed_index: float | None = None

    def to_predicted_dloss(self) -> float:
        """Collapse to the single float the knapsack DP sums.

        Only DELTA_LOSS is additive across units, which is why this is the only
        currency that may leave this module toward the solver.
        """
        total = self.weight_dloss
        if self.act_dloss is not None:
            total += self.act_dloss
        return float(total)

    def assert_currency(self, expected: Currency) -> None:
        if expected is not Currency.DELTA_LOSS:
            raise ValueError(
                f"{self.unit_name}/{self.format_name}: costs leave this module "
                f"in {Currency.DELTA_LOSS.value}; refusing to serve "
                f"{expected.value}. Mixing bases is the failure mode "
                "activation_fair_pricing.py documents.")


# --------------------------------------------------------------------- pricing


def weight_dloss_scalar(unit: SensitivityUnit, weight_mse: float,
                        gain: float = 1.0) -> float:
    """Today's model, unchanged: ``0.5 * h_trace * weight_mse * gain``.

    Kept as a named function so the byte-identical reproduction test has
    something to call, and so any divergence from `allocator_solver` is a
    one-line diff rather than a hunt.
    """
    return 0.5 * float(unit.h_trace) * float(weight_mse) * float(gain)


def weight_dloss_marginal(unit: SensitivityUnit, dw_sq: np.ndarray,
                          gain: float = 1.0) -> float:
    """Fisher-weighted weight error using the per-channel marginals.

    ``dw_sq`` is the elementwise squared weight error [out, in] the format would
    incur. Under the rank-1 reconstruction ``H ~= outer(row, col) / h_trace_raw``
    the loss delta is::

        0.5 * sum_{o,i} H[o,i] * dw_sq[o,i]
          ~= 0.5 * (row @ dw_sq @ col) / h_trace_raw

    which never forms ``H``. Falls back to the scalar model when the card has no
    vectors, so a card without them is degraded, not broken.
    """
    if not unit.has_vectors or unit.h_trace_raw <= 0.0:
        return weight_dloss_scalar(unit, float(np.mean(dw_sq)), gain)

    row = np.asarray(unit.fisher_row, dtype=np.float64)
    col = np.asarray(unit.fisher_col, dtype=np.float64)
    dw = np.asarray(dw_sq, dtype=np.float64)
    if dw.shape != (unit.out_features, unit.in_features):
        raise ValueError(
            f"{unit.topology.name}: dw_sq has shape {dw.shape}, expected "
            f"({unit.out_features}, {unit.in_features})")

    quad = float(row @ dw @ col) / unit.h_trace_raw
    # h_trace_raw is a token SUM; h_trace is the token MEAN. row/col are sums
    # too, so the quadratic form above is in "sum" units and needs the same
    # normalization the scalar path applies.
    return 0.5 * (quad / max(1, unit.n_tokens)) * float(gain)


def activation_dloss(unit: SensitivityUnit, weight: np.ndarray,
                     act_var: np.ndarray, gain: float = 1.0) -> float | None:
    """AQUA-AURA: loss delta from quantizing the layer's INPUT activations.

    ``act_var[j]`` is the per-input-channel variance of the activation
    quantization error. The perturbation reaches the loss as ``dy = W dx``, so
    under a diagonal model::

        dLoss ~= 0.5 * sum_o g_sq[o] * sum_j W[o,j]^2 * act_var[j]

    Returns ``None`` -- never 0.0 -- when the card lacks ``g_sq_sum``, so an
    unmeasured A-side is distinguishable from a free one.

    NOTE the sensitivity used here is ``g_sq_sum`` (output space), NOT
    ``h_trace`` (weight space). That distinction is the whole point.
    """
    if unit.g_sq_sum is None:
        return None
    g_sq = np.asarray(unit.g_sq_sum, dtype=np.float64)
    w_sq = np.asarray(weight, dtype=np.float64) ** 2
    var = np.asarray(act_var, dtype=np.float64)
    if w_sq.shape != (unit.out_features, unit.in_features):
        raise ValueError(f"{unit.topology.name}: weight shape mismatch")
    if var.shape != (unit.in_features,):
        raise ValueError(f"{unit.topology.name}: act_var shape mismatch")
    per_out = w_sq @ var                      # [out]
    total = float(g_sq @ per_out)
    return 0.5 * (total / max(1, unit.n_tokens)) * float(gain)


def uniform_act_quant_variance(unit: SensitivityUnit, act_bits: int,
                               ) -> np.ndarray | None:
    """Per-channel activation-quantization error variance for a uniform grid.

    Uses ``act_absmax`` when the card carries it, because activation
    quantization error is driven by the dynamic range a channel actually spans;
    otherwise falls back to a Gaussian-equivalent range from ``act_sq_sum``.

    The 1/12 factor is the variance of a uniform distribution over one step --
    a property of the quantizer, not a tuned constant.
    """
    n_levels = float(2 ** act_bits)
    if unit.act_absmax is not None:
        rng = 2.0 * np.asarray(unit.act_absmax, dtype=np.float64)
    elif unit.act_sq_sum is not None:
        sigma = np.sqrt(np.asarray(unit.act_sq_sum, dtype=np.float64)
                        / max(1, unit.n_tokens))
        # A symmetric quantizer must span roughly +/-4 sigma to avoid clipping
        # dominating; this is the standard Gaussian-range surrogate used when a
        # true absmax was not captured, and it is why act_absmax is preferred.
        rng = 8.0 * sigma
    else:
        return None
    step = rng / n_levels
    return (step ** 2) / 12.0


class FormatCostPlugin(Protocol):
    """What a format must implement to be priced by an arbitrary consumer.

    A downstream author adds a format by supplying one of these -- no change to
    the probe, the card, or the solver.
    """

    descriptor: FormatDescriptor

    def weight_error(self, unit: SensitivityUnit,
                     weight: np.ndarray) -> np.ndarray:
        """Elementwise squared weight error [out, in] this format would incur.

        Computed from the weight alone under the card's declared render basis
        (RTN for a shareable card). No Hessian, no calibration replay.
        """
        ...


def price(unit: SensitivityUnit, weight: np.ndarray,
          plugin: FormatCostPlugin, *, render_basis: RenderBasis,
          model: CostModel = CostModel.MARGINAL,
          gain: float = 1.0) -> CostComponents | None:
    """Price one (unit, format) pair. Returns None when the format is illegal.

    Legality is passthrough integrity only -- a format is never rejected here
    for looking risky. Banning formats in the coster is the "post-allocator
    rewrite" antipattern; the platform bounds error, it does not restrict what
    the allocator may consider.
    """
    desc = plugin.descriptor
    if not desc.is_legal_for(unit):
        return None

    dw_sq = plugin.weight_error(unit, weight)
    weight_mse = float(np.mean(dw_sq))

    if model is CostModel.SCALAR:
        w_dloss = weight_dloss_scalar(unit, weight_mse, gain)
    else:
        w_dloss = weight_dloss_marginal(unit, dw_sq, gain)

    a_dloss: float | None = None
    if model is CostModel.AQUA and desc.quantizes_activations:
        # Prefer a variance MEASURED through the format's own activation
        # quantizer; fall back to the analytic uniform-grid model only when the
        # plugin cannot supply one. Measuring beats assuming a grid shape.
        var = None
        measure = getattr(plugin, "activation_error_variance", None)
        if callable(measure):
            var = measure(unit)
        if var is None:
            var = uniform_act_quant_variance(unit, desc.act_bits)
        if var is not None:
            a_dloss = activation_dloss(unit, weight, var, gain)

    return CostComponents(
        unit_name=unit.topology.name,
        format_name=desc.name,
        model=model,
        render_basis=render_basis,
        weight_mse=weight_mse,
        weight_dloss=w_dloss,
        act_dloss=a_dloss,
        bits_per_param=desc.weight_bits,
        memory_bytes=int(round(unit.n_params * desc.weight_bits / 8.0)),
        speed_index=desc.speed_index,
    )
