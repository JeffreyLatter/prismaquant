"""First unit coverage for aura_cost (the paper-spine allocation cost).

Pins the two load-bearing claims the module makes about itself:
  (a) 0.5·mean_k⟨g_k, dW⟩² estimates the exact Fisher quadratic
      (fisher_quadratic_form of the true logit displacement);
  (b) chunked execution (G>1) is bit-identical to single-pass (G=1).
Plus the guards: passthrough zero rows, strict cache mode, the
tied-embeddings include_lm_head guard, and the stderr/provenance fields.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant.aura_cost import compute_aura_cost
from prismaquant.kl_fisher import fisher_quadratic_form


class TinyLM(nn.Module):
    """embed -> body Linear -> relu -> head Linear -> logits.

    Logits are affine in the head weight, so a head-weight perturbation has an
    exactly computable logit displacement — the ground truth for test (a).
    """

    def __init__(self, vocab: int = 64, hidden: int = 32, tie: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.body = nn.Linear(hidden, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids):
        h = torch.relu(self.body(self.embed(input_ids)))
        return SimpleNamespace(logits=self.lm_head(h))


class _FakeCache:
    """Production-cache stand-in: returns rendered = W + dW for chosen keys."""

    def __init__(self, rendered: dict[tuple[str, str], torch.Tensor]):
        self._rendered = rendered

    def get(self, name: str, fmt: str):
        return self._rendered.get((name, fmt))


def _ids(batch=2, seqlen=8, vocab=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seqlen), generator=g)


def test_estimator_matches_exact_fisher_quadratic():
    torch.manual_seed(3)
    model = TinyLM().eval()
    ids = _ids()

    # A known head-weight perturbation, exactly bf16-representable so the
    # bf16 dW storage path is lossless for it.
    dw = (torch.randn_like(model.lm_head.weight) * 0.25).to(torch.bfloat16)
    dw = dw.float() * 0.03125  # power-of-two scale keeps bf16 exactness
    cache = _FakeCache({
        ("lm_head", "NVFP4"): model.lm_head.weight.detach() + dw,
    })

    # Exact: logits are affine in the head weight, so the displacement of a
    # +dw perturbation is computable in closed form via a second forward.
    with torch.no_grad():
        teacher = model(ids).logits
        model.lm_head.weight += dw
        student = model(ids).logits
        model.lm_head.weight -= dw
    exact = float(fisher_quadratic_form(
        teacher, student - teacher, token_scope="all"))

    payload = compute_aura_cost(
        model, ids, ["NVFP4"],
        n_probes=4096, production_cache=cache,
        min_free_gib=0.0, n_linear_chunks=1, include_lm_head=True,
    )
    row = payload["costs"]["lm_head"]["NVFP4"]
    est = row["predicted_dloss"]
    assert row["dw_source"] == "rendered"
    # K=4096 Rademacher probes -> ~2-3% sampling error on the mean; 15% is a
    # flake-proof bound that still rejects any normalization mistake (which
    # would be off by a factor of T, V, or 2).
    assert abs(est - exact) <= 0.15 * exact, (est, exact)
    # stderr should be a plausible scale for the sampling error
    assert 0 < row["predicted_dloss_stderr"] < 0.25 * exact


def test_chunked_is_bit_identical_to_single_pass():
    torch.manual_seed(5)
    model = TinyLM().eval()
    ids = _ids(seed=1)
    kw = dict(n_probes=8, min_free_gib=0.0)

    one = compute_aura_cost(model, ids, ["NVFP4"], n_linear_chunks=1, **kw)
    three = compute_aura_cost(model, ids, ["NVFP4"], n_linear_chunks=3, **kw)

    assert one["costs"].keys() == three["costs"].keys()
    for n in one["costs"]:
        for f in one["costs"][n]:
            a, b = one["costs"][n][f], three["costs"][n][f]
            assert a["predicted_dloss"] == b["predicted_dloss"], (n, f)
            assert a["predicted_dloss_stderr"] == b["predicted_dloss_stderr"]
    for n in one["stats"]:
        assert one["stats"][n]["h_trace"] == three["stats"][n]["h_trace"]


def test_passthrough_formats_emit_zero_cost_rows():
    model = TinyLM().eval()
    payload = compute_aura_cost(
        model, _ids(), ["NVFP4", "BF16"],
        n_probes=2, min_free_gib=0.0, n_linear_chunks=1,
    )
    for n, rows in payload["costs"].items():
        assert rows["BF16"]["predicted_dloss"] == 0.0
        assert rows["BF16"]["cost_source"] == "aura_passthrough_zero"
        assert rows["NVFP4"]["predicted_dloss"] >= 0.0
        assert rows["NVFP4"]["dw_source"] == "rtn"


def test_require_production_cache_refuses_silent_rtn():
    model = TinyLM().eval()
    with pytest.raises(RuntimeError, match="require_production_cache"):
        compute_aura_cost(
            model, _ids(), ["NVFP4"],
            n_probes=2, min_free_gib=0.0, n_linear_chunks=1,
            production_cache=_FakeCache({}),
            require_production_cache=True,
        )


def test_tied_lm_head_guard_fires():
    model = TinyLM(tie=True).eval()
    with pytest.raises(RuntimeError, match="tie_word_embeddings"):
        compute_aura_cost(
            model, _ids(), ["NVFP4"],
            n_probes=2, min_free_gib=0.0, n_linear_chunks=1,
            include_lm_head=True,
        )
    # Without include_lm_head the tied model is fine (lm_head excluded).
    payload = compute_aura_cost(
        model, _ids(), ["NVFP4"],
        n_probes=2, min_free_gib=0.0, n_linear_chunks=1,
    )
    assert "lm_head" not in payload["costs"]


def test_provenance_records_seed_and_dw_split():
    model = TinyLM().eval()
    ids = _ids()
    cache = _FakeCache({
        ("body", "NVFP4"): model.body.weight.detach() * 1.001,
    })
    payload = compute_aura_cost(
        model, ids, ["NVFP4"],
        n_probes=2, min_free_gib=0.0, n_linear_chunks=1,
        production_cache=cache, seed_base=12345,
    )
    prov = payload["provenance"]
    assert prov["seed_base"] == 12345
    assert prov["dw_rendered_rows"] == 1   # body via the cache
    assert prov["dw_rtn_fallback_rows"] == 0  # lm_head excluded by default
    assert prov["calib_shape"] == list(ids.shape)
    assert len(prov["calib_sha256"]) == 64
    assert payload["costs"]["body"]["NVFP4"]["dw_source"] == "rendered"
