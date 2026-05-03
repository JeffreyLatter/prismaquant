from __future__ import annotations

import json

import torch

from prismaquant.entmoot_cli import (
    _run_collection_forward,
    _truncate_decoder_layers,
    choose_router_strategies,
    main,
    plan_from_collector,
    summarize_collector,
    summarize_manifest,
)
from prismaquant.entmoot_collector import LayerSketchBuffer
from prismaquant.schemas import validate_merge_manifest_payload


def _collector_artifact(path):
    layer = LayerSketchBuffer(
        "model.layers.0.mlp.gate",
        num_experts=3,
        max_samples_per_expert=4,
    )
    layer.add_router_batch(torch.ones(4, 2))
    layer.add_expert_batch(
        0,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([1.0, 1.0]),
    )
    layer.add_expert_batch(
        1,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[0.99, 0.01], [1.0, 0.0]]),
        torch.tensor([1.0, 1.0]),
    )
    layer.add_expert_batch(
        2,
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        torch.tensor([1.0, 1.0]),
    )
    torch.save({
        "format": "entmoot_activation_collector_v1",
        "layers": {layer.router_qname: layer.state_dict()},
    }, path)


def test_plan_from_collector_writes_valid_manifest(tmp_path):
    collector = tmp_path / "collector.pt"
    manifest_path = tmp_path / "manifest.json"
    _collector_artifact(collector)

    summary = plan_from_collector(
        collector,
        manifest_path,
        target_experts=2,
        min_samples=0,
        activation_accept_threshold=0.01,
    )
    manifest = json.loads(manifest_path.read_text())

    assert summary["n_layers"] == 1
    validate_merge_manifest_payload(manifest, str(manifest_path))
    entry = manifest["model.layers.0.mlp.gate"]
    assert entry["method"] == "entmoot_router_id_v1"
    assert entry["router_strategy"] == "anchor"
    assert entry["num_experts_orig"] == 3
    assert entry["num_experts_kept"] == 2


def test_summaries_return_expected_shapes(tmp_path):
    collector = tmp_path / "collector.pt"
    manifest_path = tmp_path / "manifest.json"
    _collector_artifact(collector)
    plan_from_collector(
        collector,
        manifest_path,
        target_experts=2,
        min_samples=0,
    )

    csum = summarize_collector(collector)
    msum = summarize_manifest(manifest_path)

    assert csum["n_layers"] == 1
    assert csum["layers"]["model.layers.0.mlp.gate"]["experts_with_samples"] == 3
    assert msum["n_layers"] == 1
    assert msum["layers"]["model.layers.0.mlp.gate"]["method"] == "entmoot_router_id_v1"


def test_cli_plan_from_collector(tmp_path):
    collector = tmp_path / "collector.pt"
    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "summary.json"
    _collector_artifact(collector)

    rc = main([
        "plan-from-collector",
        "--collector", str(collector),
        "--output", str(manifest_path),
        "--target-experts", "2",
        "--min-samples", "0",
        "--summary", str(summary_path),
    ])

    assert rc == 0
    assert manifest_path.exists()
    assert json.loads(summary_path.read_text())["n_layers"] == 1


def test_collection_forward_uses_backbone_and_can_truncate_layers():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(1, 1),
                torch.nn.Linear(1, 1),
                torch.nn.Linear(1, 1),
            ])
            self.last_layer_count = 0

        def forward(self, input_ids, **_kwargs):
            self.last_layer_count = len(self.layers)
            return input_ids

    class CausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.called = False

        def forward(self, input_ids, **_kwargs):
            self.called = True
            return self.model(input_ids)

    model = CausalLM()
    restore = _truncate_decoder_layers(model, 0)
    try:
        _run_collection_forward(
            model,
            {"input_ids": torch.ones(1, 1, dtype=torch.long)},
            forward_mode="backbone",
        )
        assert not model.called
        assert model.model.last_layer_count == 1
    finally:
        restore()

    assert len(model.model.layers) == 3


def test_choose_router_strategies_reads_safetensor_router_weight(tmp_path):
    from safetensors.torch import save_file

    collector = tmp_path / "collector.pt"
    manifest_path = tmp_path / "manifest.json"
    out_manifest = tmp_path / "manifest.routed.json"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _collector_artifact(collector)
    plan_from_collector(
        collector,
        manifest_path,
        target_experts=2,
        min_samples=0,
    )
    (model_dir / "config.json").write_text(json.dumps({
        "text_config": {"num_experts_per_tok": 2}
    }))
    save_file({
        "model.language_model.layers.0.mlp.gate.weight": torch.tensor([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ])
    }, model_dir / "model.safetensors")

    summary = choose_router_strategies(
        collector,
        manifest_path,
        model_dir,
        out_manifest,
        top1_floor=0.0,
        topk_floor=0.0,
        kl_cap=10.0,
    )
    routed = json.loads(out_manifest.read_text())
    entry = routed["model.layers.0.mlp.gate"]

    assert summary["routers"]["model.layers.0.mlp.gate"]["selected_strategy"] in {
        "anchor",
        "weighted_average",
    }
    assert entry["router_strategy"] in {"anchor", "weighted_average"}
    assert "router_strategy_choice" in entry["diagnostics"]
