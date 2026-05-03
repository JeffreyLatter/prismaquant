"""Tests for cheap per-sample router survey."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.expert_calibration_survey import (
    RouterSurveyTracker,
    discover_router_qnames,
    survey_jsonl,
    text_from_row,
)
from prismaquant.expert_calibration_select import (
    load_survey_jsonl,
    select_expert_balanced_samples,
)


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


class TinyMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 3, bias=False)
        self.experts = nn.ModuleList([TinyExpert() for _ in range(3)])
        with torch.no_grad():
            self.gate.weight.copy_(torch.tensor([
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0],
            ]))

    def forward(self, x):
        return self.gate(x)


class TinySurveyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_experts_per_tok=1)
        self.block = TinyMoeBlock()

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        x = F.one_hot(input_ids % 4, num_classes=4).to(torch.float32)
        return self.block(x)


class DummyTokenizer:
    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        del return_tensors
        ids = [int(tok) for tok in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def apply_chat_template(self, messages, tokenize=False):
        del tokenize
        return " ".join(str(m["content"]) for m in messages)


def test_router_survey_tracker_records_normalized_top1_hits():
    model = TinySurveyModel()
    tracker = RouterSurveyTracker(model, ["block.gate"], top_k=1)
    try:
        input_ids = torch.tensor([[0, 1, 1, 2]], dtype=torch.long)
        model(input_ids=input_ids)
        hits = tracker.hits(normalize_by_tokens=True)
    finally:
        tracker.remove_hooks()

    assert hits["block.gate"][0] == pytest.approx(0.25)
    assert hits["block.gate"][1] == pytest.approx(0.50)
    assert hits["block.gate"][2] == pytest.approx(0.25)


def test_discover_router_qnames_uses_existing_moe_discovery():
    model = TinySurveyModel()

    assert discover_router_qnames(model) == ["block.gate"]


def test_survey_jsonl_emits_hits_that_selector_can_read(tmp_path):
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "survey.jsonl"
    input_path.write_text(
        "\n".join([
            json.dumps({"id": "a", "domain": "code", "text": "0 0 1 2"}),
            json.dumps({"id": "b", "domain": "math", "text": "2 2 2 1"}),
        ])
        + "\n",
        encoding="utf-8",
    )

    summary = survey_jsonl(
        TinySurveyModel(),
        DummyTokenizer(),
        input_path,
        output_path,
        device=torch.device("cpu"),
        top_k=1,
        max_length=8,
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["rows_written"] == 2
    assert rows[0]["hits"]["block.gate"]["0"] == pytest.approx(0.5)
    assert rows[0]["hits"]["block.gate"]["1"] == pytest.approx(0.25)
    assert rows[0]["hits"]["block.gate"]["2"] == pytest.approx(0.25)
    assert rows[1]["domain"] == "math"
    assert rows[1]["router_tokens"]["block.gate"] == 4

    samples = load_survey_jsonl(output_path)
    selected = select_expert_balanced_samples(samples, budget=1)
    assert selected.selected_ids == ["a"]


def test_text_from_row_handles_messages_without_template():
    class NoTemplateTokenizer:
        def apply_chat_template(self, messages, tokenize=False):
            del messages, tokenize
            raise RuntimeError("no template")

    text = text_from_row(
        {"messages": [{"role": "user", "content": "1 2"}, {"content": "3"}]},
        NoTemplateTokenizer(),
    )

    assert text == "1 2\n\n3"
