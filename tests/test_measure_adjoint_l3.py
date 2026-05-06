from types import SimpleNamespace

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.measure_adjoint_l3 import (
    _sample_token_windows_from_texts,
    collect_adjoint_l3,
)


class _ToyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.embed = nn.Embedding(64, 64)
        self.block = nn.Linear(64, 64)
        self.lm_head = nn.Linear(64, 64)

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        hidden = torch.tanh(self.block(hidden))
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)


def test_collect_adjoint_l3_toy_causal_lm_includes_mxfp8():
    torch.manual_seed(1234)
    model = _ToyCausalLM()
    with torch.no_grad():
        model.block.weight.mul_(3.0)
    calib_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )

    payload = collect_adjoint_l3(
        model,
        calib_ids,
        [fr.get_format("NVFP4"), fr.get_format("MXFP8"), fr.get_format("BF16")],
        target_names={"block"},
    )

    assert payload["schema"] == "prismaquant.adjoint_l3.v1"
    assert payload["rank"] == 2
    assert payload["meta"]["target_count"] == 1
    formats = payload["units"]["block"]["formats"]
    assert set(formats) == {"BF16", "MXFP8_E4M3", "NVFP4"}
    assert formats["BF16"]["sketch"] == [0.0, 0.0]
    assert len(formats["MXFP8_E4M3"]["sketch"]) == 2
    assert len(formats["NVFP4"]["sketch"]) == 2
    assert any(abs(v) > 0.0 for v in formats["NVFP4"]["sketch"])
    assert formats["NVFP4"]["diagonal_cost"] > 0.0


def test_collect_adjoint_l3_fisher_last_token_uses_probe_rank():
    torch.manual_seed(1234)
    model = _ToyCausalLM()
    with torch.no_grad():
        model.block.weight.mul_(3.0)
    calib_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )

    payload = collect_adjoint_l3(
        model,
        calib_ids,
        [fr.get_format("NVFP4"), fr.get_format("BF16")],
        target_names={"block"},
        direction_mode="fisher-last-token",
        fisher_probes_per_sample=2,
        fisher_seed=7,
        mse_diagonal_floor_frac=0.5,
    )

    formats = payload["units"]["block"]["formats"]
    assert payload["rank"] == 4
    assert payload["meta"]["calib_samples"] == 2
    assert payload["meta"]["fisher_probes_per_sample"] == 2
    assert len(formats["NVFP4"]["sketch"]) == 4
    assert any(abs(v) > 0.0 for v in formats["NVFP4"]["sketch"])
    assert formats["NVFP4"]["mse_floor_cost"] >= 0.0
    assert formats["NVFP4"]["diagonal_cost"] >= formats["NVFP4"]["adjoint_self_cost"]
    assert payload["meta"]["objective_metric"] == "teacher_forward_kl_single_point_fisher"
    assert payload["meta"]["curvature"] == "categorical_fisher_psd"
    assert payload["meta"]["fisher_token_scope"] == "last"


def test_collect_adjoint_l3_kl_fisher_accepts_temperature_and_rademacher():
    torch.manual_seed(4321)
    model = _ToyCausalLM()
    with torch.no_grad():
        model.block.weight.mul_(2.0)
    calib_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    payload = collect_adjoint_l3(
        model,
        calib_ids,
        [fr.get_format("NVFP4"), fr.get_format("BF16")],
        target_names={"block"},
        direction_mode="kl-fisher",
        fisher_temperature=1.5,
        fisher_token_scope="causal",
        fisher_probe_distribution="rademacher",
    )

    formats = payload["units"]["block"]["formats"]
    assert payload["rank"] == 1
    assert payload["meta"]["direction_mode"] == "kl-fisher"
    assert payload["meta"]["fisher_temperature"] == 1.5
    assert payload["meta"]["fisher_token_scope"] == "causal"
    assert payload["meta"]["fisher_probe_distribution"] == "rademacher"
    assert len(formats["NVFP4"]["sketch"]) == 1
    assert any(abs(v) > 0.0 for v in formats["NVFP4"]["sketch"])


def test_sample_token_windows_does_not_require_full_corpus_tokenization():
    class _Tokenizer:
        eos_token_id = 0

        def __call__(self, text, **_kwargs):
            ids = [ord(ch) % 31 + 1 for ch in text]
            return SimpleNamespace(input_ids=ids)

    windows = _sample_token_windows_from_texts(
        ["abc", "defgh", "ijklmnop"],
        _Tokenizer(),
        2,
        4,
        seed=1,
    )

    assert tuple(windows.shape) == (2, 4)
    assert windows.dtype == torch.long
