"""Unit coverage for per-column KL-Fisher weights (nvfp4-cb exp 4).

Pins the three claims:
  (a) the harvested per-column energy equals the autograd sum over output rows
      of the squared probe-gradient, per column (exact within fp tolerance);
  (b) enabling the collection is strictly additive — aura_cost's h_trace and
      predicted_dloss are bit-identical with the feature OFF, and with it ON the
      per-column vector sums back to h_trace;
  (c) both compositions produce a positive, finite (in,) / (N,1,in) vector that
      the GGUF encoder's col_weights / qw slot accepts.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from prismaquant.aura_cost import compute_aura_cost
from prismaquant.fisher_col_weights import (
    fisher_only,
    fisher_x_act,
)
from prismaquant.kl_fisher import fisher_probe_scalar


class TinyMLP(nn.Module):
    """embed -> fc1 -> relu -> fc2 -> relu -> lm_head -> logits (a 2-layer MLP
    body). Deterministic in eval(); the probe backward is reproducible."""

    def __init__(self, vocab: int = 48, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        h = torch.relu(self.fc1(self.embed(input_ids)))
        h = torch.relu(self.fc2(h))
        return SimpleNamespace(logits=self.lm_head(h))


def _ids(batch=2, seqlen=6, vocab=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seqlen), generator=g)


def test_col_energy_matches_autograd_per_column():
    """Harvested fisher_col == Σ_out (∂probe/∂W)[:,j]², column by column."""
    torch.manual_seed(11)
    model = TinyMLP().eval()
    ids = _ids(seed=2)
    seed_base = 123

    payload = compute_aura_cost(
        model, ids, ["NVFP4"], n_probes=1, seed_base=seed_base,
        min_free_gib=0.0, n_linear_chunks=1, collect_col_energy=True,
    )

    # Reproduce the single probe's weight-gradient by hand: identical forward,
    # identical Rademacher seed (seed_base + probe 0), single micro-batch (no
    # token_count_override), token_scope="all", temperature 1.0.
    for p in model.parameters():
        p.requires_grad_(False)
    for n in ("fc1", "fc2"):
        model.get_submodule(n).weight.requires_grad_(True)
        model.get_submodule(n).weight.grad = None
    logits = model(ids).logits
    probe = fisher_probe_scalar(
        logits, seed=seed_base, token_scope="all", temperature=1.0,
        distribution="rademacher",
    )
    probe.backward()

    for n in ("fc1", "fc2"):
        w = model.get_submodule(n).weight
        want = (w.grad.float() ** 2).sum(dim=0)  # over output rows -> (in,)
        got = payload["stats"][n]["fisher_col"]
        assert got.shape == want.shape == (w.shape[1],)
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-7), (n, got, want)


def test_collection_is_additive_and_sums_to_h_trace():
    """OFF: byte-identical h_trace/predicted_dloss + no fisher_col key.
    ON: the per-column vector sums back to the scalar h_trace."""
    torch.manual_seed(7)
    model = TinyMLP().eval()
    ids = _ids(seed=3)
    kw = dict(n_probes=6, seed_base=55, min_free_gib=0.0, n_linear_chunks=1)

    off = compute_aura_cost(model, ids, ["NVFP4"], **kw)
    on = compute_aura_cost(
        model, ids, ["NVFP4"], collect_col_energy=True, **kw)

    # (b) regression: everything the allocator consumes is bit-identical.
    assert off["stats"].keys() == on["stats"].keys()
    for n in off["stats"]:
        assert "fisher_col" not in off["stats"][n]
        assert off["stats"][n]["h_trace"] == on["stats"][n]["h_trace"]
    for n in off["costs"]:
        for f in off["costs"][n]:
            assert (off["costs"][n][f]["predicted_dloss"]
                    == on["costs"][n][f]["predicted_dloss"]), (n, f)

    # (b) consistency: col_energy is a partition of the scalar KL-Fisher energy.
    for n, st in on["stats"].items():
        vec = st.get("fisher_col")
        if vec is None:
            continue
        assert torch.isfinite(vec).all() and (vec >= 0).all()
        assert abs(float(vec.sum()) - st["h_trace"]) <= 1e-4 * (
            abs(st["h_trace"]) + 1e-9)


def test_compositions_land_in_qw_slot():
    """fisher_only / fisher_x_act -> positive, finite col_weights the GGUF
    encoder accepts, for (in,) and stacked (N,1,in) shapes."""
    from prismaquant.gguf_formats import (
        QK_K,
        _qw_blocks,
        gguf_quantize_dequantize,
    )

    in_f = 64
    v = torch.rand(in_f) + 0.05
    v[3] = 0.0  # a dead column must survive as strictly positive
    xbar2 = torch.rand(in_f) + 0.01
    sigma2 = 2.0 * float(xbar2.mean())

    for cw in (fisher_only(v), fisher_x_act(v, sigma2, xbar2)):
        assert cw.shape == (in_f,)
        assert torch.isfinite(cw).all() and (cw > 0).all()
        w = torch.randn(32, in_f)
        out = gguf_quantize_dequantize(w, "Q4_K", col_weights=cw)
        assert out.shape == w.shape and torch.isfinite(out).all()

    # Stacked / expert layout: (N, 1, in) drops into _qw_blocks unchanged.
    v3 = torch.rand(4, 1, in_f) + 0.05
    xbar2_3 = torch.rand(4, 1, in_f) + 0.01
    for cw3 in (fisher_only(v3), fisher_x_act(v3, 0.01, xbar2_3)):
        assert cw3.shape == (4, 1, in_f)
        assert torch.isfinite(cw3).all() and (cw3 > 0).all()
        blocks = _qw_blocks(
            cw3, (4, 8, in_f), pad=(-in_f) % QK_K, block=QK_K)
        assert torch.isfinite(blocks).all() and (blocks >= 0).all()
