import os
import random

import torch

from tools import smoke_graph_memory


def test_pin_rng_sets_expected_streams(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    smoke_graph_memory._pin_rng(12345)

    assert random.random() == 0.41661987254534116
    assert torch.rand(1).item() == 0.9817181825637817
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_smoke_seed_env_override(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_SMOKE_SEED", "42")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    smoke_graph_memory._set_smoke_env()
    smoke_graph_memory._pin_rng(int(os.environ["PRISMAQUANT_SMOKE_SEED"]))

    assert random.random() == 0.6394267984578837
    assert torch.rand(1).item() == 0.8822692632675171
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
