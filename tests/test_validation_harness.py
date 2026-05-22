from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import validation_harness as vh
from prismaquant.validation_harness import validate_artifact


def test_validate_artifact_returns_finite_metrics(tmp_path):
    layer_config = {
        "model.layers.0.mlp.down_proj": {
            "bits": 16,
            "group_size": 0,
            "data_type": "float",
            "act_bits": 16,
            "act_data_type": "float",
        }
    }

    def fake_backend(**kwargs):
        assert kwargs["model_path"] == "fake-qwen"
        assert kwargs["layer_config"] == layer_config
        assert kwargs["n_wikitext_tokens"] == 2048
        assert kwargs["n_mmlu_questions"] == 10
        return {
            "ppl_wikitext": 8.25,
            "ppl_mmlu_acc": 0.40,
            "end_kl": 0.0125,
        }

    result = validate_artifact(
        "fake-qwen",
        layer_config,
        cache_dir=tmp_path,
        device="cpu",
        dtype="fp32",
        n_wikitext_tokens=2048,
        n_mmlu_questions=10,
        calib_seqlen=64,
        calib_n_samples=2,
        progress=False,
        _metric_backend=fake_backend,
    )

    for key in ("ppl_wikitext", "ppl_mmlu_acc", "end_kl", "eval_seconds"):
        assert isinstance(result[key], float)
        assert math.isfinite(result[key])
    assert len(result["model_sha"]) == 64
    assert len(result["layer_config_sha"]) == 64


def test_validate_artifact_accepts_layer_config_path_with_stub_backend(tmp_path):
    layer_config = {"model.layers.0.self_attn.q_proj": "BF16"}
    config_path = tmp_path / "layer_config.json"
    config_path.write_text(json.dumps(layer_config))

    result = validate_artifact(
        "fake-qwen",
        str(config_path),
        cache_dir=tmp_path,
        device="cpu",
        dtype="fp32",
        n_wikitext_tokens=2048,
        n_mmlu_questions=10,
        calib_seqlen=64,
        calib_n_samples=2,
        progress=False,
        _metric_backend=lambda **_kwargs: {
            "ppl_wikitext": 9.0,
            "ppl_mmlu_acc": 0.30,
            "end_kl": 0.0,
        },
    )

    assert result["ppl_wikitext"] == 9.0
    assert result["ppl_mmlu_acc"] == 0.30
    assert result["end_kl"] == 0.0


class _ValidationToyModel(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 16):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=128)
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, labels=None):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        loss = logits.float().sum() * 0.0
        if labels is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


class _ValidationTokenizer:
    def __call__(self, text, *, add_special_tokens=False, return_tensors="pt"):
        del add_special_tokens
        ids = [(ord(ch) % 63) + 1 for ch in text]
        if not ids:
            ids = [1]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def test_load_wikitext_ids_falls_back_to_legacy_dataset_name(tmp_path):
    calls = []

    def fake_load_dataset(name, config, *, split, cache_dir):
        calls.append((name, config, split, cache_dir))
        if name == "Salesforce/wikitext":
            raise RuntimeError("namespaced dataset unavailable")
        return [{"text": "hello"}, {"text": ""}, {"text": "world"}]

    ids = vh._load_wikitext_ids(
        _ValidationTokenizer(),
        fake_load_dataset,
        cache_dir=tmp_path,
        split="test",
        n_tokens=4,
    )

    assert ids.shape == (1, 4)
    assert [call[0] for call in calls] == ["Salesforce/wikitext", "wikitext"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_validation_with_cuda_graphs_matches_eager(monkeypatch):
    torch.manual_seed(0)
    model = _ValidationToyModel().eval().cuda()
    tokenizer = _ValidationTokenizer()
    prompt = "Question: 1 + 1?\nA. 1\nB. 2\nAnswer:"

    def _measure(graphs_enabled: bool) -> list[float]:
        monkeypatch.setenv(
            "PRISMAQUANT_VALIDATION_CUDA_GRAPHS",
            "1" if graphs_enabled else "0",
        )
        return [
            vh._choice_letter_nll(model, tokenizer, prompt, idx, torch.device("cuda"))
            for idx in range(2)
        ]

    eager = _measure(False)
    graphed = _measure(True)
    assert graphed == pytest.approx(eager, abs=1e-8, rel=0.0)


@pytest.mark.slow
def test_validate_artifact_real_tiny_model(tmp_path):
    if not os.environ.get("PRISMAQUANT_RUN_HF_VALIDATION"):
        pytest.skip("set PRISMAQUANT_RUN_HF_VALIDATION=1 to run real HF validation")
    model = os.environ.get("PRISMAQUANT_TINY_MODEL", "Qwen/Qwen3-0.6B")
    result = validate_artifact(
        model,
        {},
        cache_dir=tmp_path,
        device=os.environ.get("PRISMAQUANT_VALIDATION_DEVICE", "cpu"),
        dtype="fp32",
        n_wikitext_tokens=2048,
        n_mmlu_questions=10,
        calib_seqlen=64,
        calib_n_samples=2,
        progress=False,
    )
    assert math.isfinite(result["ppl_wikitext"])
    assert math.isfinite(result["ppl_mmlu_acc"])
    assert math.isfinite(result["end_kl"])
