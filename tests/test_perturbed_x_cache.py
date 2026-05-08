import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    activation_cache_filename,
    capture_perturbed_activation_cache,
    stage_text_only_under_work_root,
)


class _TwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _load_cache(cache_dir: Path, name: str) -> torch.Tensor:
    blob = torch.load(
        cache_dir / activation_cache_filename(name),
        map_location="cpu",
        weights_only=False,
    )
    return blob["inputs"].to(torch.float32)


def test_perturbed_cache_captures_then_quantizes_for_forward(tmp_path):
    torch.manual_seed(0)
    model = _TwoLinear().eval()
    x = torch.randn(2, 64, dtype=torch.float32)
    fc1_w = model.fc1.weight.detach().clone()

    manifest = capture_perturbed_activation_cache(
        model,
        {"fc1": "NVFP4", "fc2": "BF16"},
        x,
        tmp_path,
        input_rows=8,
    )

    nvfp4 = fr.get_format("NVFP4")
    expected_fc2_input = F.linear(
        nvfp4.activation_quantize_dequantize(x),
        nvfp4.quantize_dequantize(fc1_w),
    )
    torch.testing.assert_close(_load_cache(tmp_path, "fc1"), x, rtol=0.01, atol=0.01)
    torch.testing.assert_close(
        _load_cache(tmp_path, "fc2"),
        expected_fc2_input,
        rtol=0.01,
        atol=0.01,
    )
    torch.testing.assert_close(model.fc1.weight, fc1_w)
    assert manifest["written"] == ["fc1", "fc2"]


class _SiblingInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(64, 64, bias=False)
        self.self_attn.k_proj = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.self_attn.q_proj(x) + self.self_attn.k_proj(x)


def test_perturbed_cache_shares_row_subsample_for_fused_siblings(tmp_path):
    model = _SiblingInputModel().eval()
    x = torch.arange(320, dtype=torch.float32).reshape(1, 5, 64)

    capture_perturbed_activation_cache(
        model,
        {"self_attn.q_proj": "BF16", "self_attn.k_proj": "BF16"},
        x,
        tmp_path,
        input_rows=2,
        cal_hash="fixed",
    )

    q_rows = _load_cache(tmp_path, "self_attn.q_proj")
    k_rows = _load_cache(tmp_path, "self_attn.k_proj")
    assert q_rows.shape == (2, 64)
    torch.testing.assert_close(q_rows, k_rows)


def test_perturbed_cache_can_skip_activation_quant_for_probe(tmp_path, monkeypatch):
    spec = fr.FormatSpec(
        name="ZERO_ACT_TEST",
        weight_bits=8,
        group_size=0,
        scale_bits=0,
        scale_dtype_name="none",
        weight_element_dtype="test",
        act_bits=4,
        quantize_dequantize=lambda w: w.clone(),
        activation_quantize_dequantize=lambda x: torch.zeros_like(x),
    )
    monkeypatch.setitem(fr.REGISTRY, spec.name, spec)
    model = nn.Sequential(nn.Linear(64, 64, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(64))
    x = torch.randn(2, 64)

    with_act = PerturbedActivationCache(
        model,
        {"0": spec.name},
        tmp_path / "with_act",
        input_rows=0,
        cal_hash="test",
        include_activation_quant=True,
    )
    with_act.install()
    try:
        torch.testing.assert_close(model(x), torch.zeros_like(x))
    finally:
        with_act.remove()

    without_act = PerturbedActivationCache(
        model,
        {"0": spec.name},
        tmp_path / "without_act",
        input_rows=0,
        cal_hash="test",
        include_activation_quant=False,
    )
    without_act.install()
    try:
        torch.testing.assert_close(model(x), x)
    finally:
        without_act.remove()


def test_stage_text_only_uses_work_root_for_tempdir(tmp_path):
    src = tmp_path / "model"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"placeholder")
    with open(src / "config.json", "w") as f:
        json.dump(
            {
                "vision_config": {},
                "text_config": {"hidden_size": 8, "model_type": "toy_text"},
                "architectures": ["ToyForConditionalGeneration"],
            },
            f,
        )
    work_root = tmp_path / "work"

    staged = Path(stage_text_only_under_work_root(str(src), work_root))

    assert staged.parent == work_root
    with open(staged / "config.json") as f:
        cfg = json.load(f)
    assert "vision_config" not in cfg
    assert cfg["hidden_size"] == 8
    assert cfg["architectures"] == ["ToyForCausalLM"]
