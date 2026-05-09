from __future__ import annotations

import builtins
import json

import torch

from prismaquant.sensitivity_probe import load_calibration


class _ToyTokenizer:
    eos_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=False):
        del return_tensors, truncation
        ids = [max(1, (ord(ch) % 31)) for ch in text]
        return type("Tokenized", (), {"input_ids": torch.tensor([ids])})


def test_local_jsonl_calibration_does_not_require_datasets(monkeypatch, tmp_path):
    path = tmp_path / "calib.jsonl"
    path.write_text(json.dumps({"text": "abcdefghijklmnopqrstuvwxyz"}) + "\n")

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "datasets":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    calib = load_calibration(_ToyTokenizer(), str(path), n_samples=1, seqlen=8)

    assert calib.shape == (1, 8)
