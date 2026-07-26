"""Correctness gate for the GROUPED-FUSED MoE PREFILL path
(``moe.PrismaQuantCBMoEMethod._apply_prefill_grouped_fused``, round 1 of the
MoE fused campaign; ``PRISMAQUANT_CB_PREFILL=grouped_fused``).

The grouped-fused path removes the stock path's HBM e4m3 expand round-trip by
decoding each expert's packed CB rows inside the CUTLASS prologue
(``cb_fused_prefill_mm_scaled``). Weights decode bit-exactly and the activation
QDQ is the stock path's own per-token fp8 dynamic, so the two differ only by
GEMM accumulation + cross-expert combine reassociation — the suite's
REASSOCIATION-CLASS 2e-2 contract (same bound as loop-vs-batched).

Run scopes:

* ``-k routing`` (build venv, NO vLLM/CUDA needed): the sort/boundary property
  the one-sync routing design rests on.
* everything else (serving container: vLLM + CUDA + the fused extension):
    docker run --rm --gpus all -v /home/rob/prismaquant:/repo \\
      --entrypoint bash vllm-node-tf5-cu132-lfm:latest -c 'pip install -q pytest; \\
      PYTHONPATH=/repo:/repo/plugins/gridbook python3 -m pytest \\
      /repo/plugins/gridbook/tests/test_moe_grouped_fused.py -v'
"""
import pytest
import torch

pytest.importorskip("gridbook.codec")

from test_moe_batched_prefill import (  # noqa: E402
    DEV,
    _REL,
    _build,
    _report,
    _require_stack,
    _routing,
    _silu_act,
)


# --------------------------------------------------------------------------- #
# Routing property (CPU ok): stable expert-sort + cumsum boundaries reproduce   #
# exactly the loop path's per-expert row selection, in the loop's order.        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_routing_boundaries_match_loop_selection(topk):
    torch.manual_seed(0)
    T, E = 37, 11
    topk_ids = torch.stack([torch.randperm(E)[:topk] for _ in range(T)])
    pair_expert = topk_ids.reshape(-1).to(torch.long)
    pair_token = torch.arange(T).repeat_interleave(topk)
    order = torch.argsort(pair_expert, stable=True)
    ptok_sorted = pair_token[order]
    counts = torch.bincount(pair_expert, minlength=E)
    bounds = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)]).tolist()

    assert bounds[-1] == T * topk
    for e in range(E):
        p0, p1 = bounds[e], bounds[e + 1]
        tok_idx, _slot = torch.where(topk_ids == e)     # the loop's selection
        assert torch.equal(ptok_sorted[p0:p1], tok_idx), (
            f"expert {e}: segment != loop selection (order or bounds wrong)")


def test_zero_row_experts_are_skippable_without_extra_syncs():
    """Empty experts show up as p1 == p0 on the ALREADY-fetched boundaries, so
    skipping them costs no additional device read."""
    E, topk = 8, 1
    topk_ids = torch.full((16, topk), 3, dtype=torch.long)
    counts = torch.bincount(topk_ids.reshape(-1), minlength=E)
    bounds = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)]).tolist()
    hit = [e for e in range(E) if bounds[e + 1] > bounds[e]]
    assert hit == [3]


# --------------------------------------------------------------------------- #
# GPU parity: grouped_fused vs stock (the path it replaces).                    #
# --------------------------------------------------------------------------- #
def _require_fused(m, layer):
    _require_stack()
    if not m._gf_ok(layer):
        pytest.skip("fused CB extension / rung constraints unmet")


@pytest.mark.parametrize("dist", ["uniform", "subset"])
@pytest.mark.parametrize("topk", [2, 4])
def test_grouped_fused_vs_stock_parity(dist, topk):
    _require_stack()
    m, layer, d = _build("fp8", seed=1)
    _require_fused(m, layer)
    act = _silu_act()
    T = 48
    ti, tw = _routing(T, d["E"], topk, dist, seed=7)
    torch.manual_seed(2)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5

    o_stock = m._apply_prefill_stock(layer, x, tw, ti, act)
    o_gf = m._apply_prefill_grouped_fused(layer, x, tw, ti, act)
    assert o_gf is not None, "grouped_fused returned None despite _gf_ok"
    assert o_gf.shape == o_stock.shape == (T, d["hidden"])
    rel = _report(f"gf-vs-stock[{dist},topk={topk}]", o_stock, o_gf)
    assert rel <= _REL, f"{dist}/topk={topk}: rel {rel:.3e} > {_REL}"


@pytest.mark.parametrize("T", [1, 3, 17, 129])
def test_grouped_fused_small_and_partial_tile_m(T):
    """No minimum M: an expert with a handful of rows must run through one
    partial CUTLASS tile, not crash and not need padding."""
    _require_stack()
    m, layer, d = _build("fp8", seed=3)
    _require_fused(m, layer)
    act = _silu_act()
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=5)
    torch.manual_seed(4)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_stock = m._apply_prefill_stock(layer, x, tw, ti, act)
    o_gf = m._apply_prefill_grouped_fused(layer, x, tw, ti, act)
    rel = _report(f"gf-vs-stock[M={T}]", o_stock, o_gf)
    assert rel <= _REL


def test_grouped_fused_all_tokens_one_expert():
    """Ragged extreme: E-1 zero-row experts + one full segment."""
    _require_stack()
    m, layer, d = _build("fp8", seed=6)
    _require_fused(m, layer)
    act = _silu_act()
    T = 40
    ti, tw = _routing(T, d["E"], 1, "one_expert", seed=1)
    torch.manual_seed(8)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o_stock = m._apply_prefill_stock(layer, x, tw, ti, act)
    o_gf = m._apply_prefill_grouped_fused(layer, x, tw, ti, act)
    rel = _report("gf-vs-stock[one_expert]", o_stock, o_gf)
    assert rel <= _REL


def test_fp4_falls_through():
    """fp4-CB is not eligible (the prologue can't compose a two-tier scale):
    _gf_ok must be False and the path must return None, not raise."""
    _require_stack()
    m, layer, d = _build("fp4v2")
    assert m._gf_ok(layer) is False
    act = _silu_act()
    ti, tw = _routing(8, d["E"], 2, "uniform", seed=0)
    x = torch.randn(8, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    assert m._apply_prefill_grouped_fused(layer, x, tw, ti, act) is None


def test_mode_dispatch_selects_grouped_fused(monkeypatch):
    """PRISMAQUANT_CB_PREFILL=grouped_fused routes _apply_inline into the new
    path (and falls through to stock when ineligible)."""
    _require_stack()
    m, layer, d = _build("fp8", seed=9)
    _require_fused(m, layer)
    monkeypatch.setenv("PRISMAQUANT_CB_PREFILL", "grouped_fused")
    seen = {}
    orig = m._apply_prefill_grouped_fused

    def _spy(*a, **kw):
        seen["hit"] = True
        return orig(*a, **kw)

    m._apply_prefill_grouped_fused = _spy
    T = 32
    ti, tw = _routing(T, d["E"], 2, "uniform", seed=2)
    x = torch.randn(T, d["hidden"], dtype=torch.bfloat16, device=DEV) * 0.5
    o = m._apply_inline(layer, x, tw, ti)
    assert seen.get("hit"), "mode dispatch did not reach grouped_fused"
    assert o.shape == (T, d["hidden"])
