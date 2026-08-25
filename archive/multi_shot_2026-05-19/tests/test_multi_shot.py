"""Tests for multi-shot recalibration helpers in ``prismaquant.multi_shot``.

The cheap variant has three load-bearing properties we test here:

1. Per-Linear input activations are written in the on-disk format
   ``incremental_measure_quant_cost``'s ``ActivationIndex`` reads (a per-Linear
   ``.pt`` file with payload ``{"inputs": X, "name": name}`` and the file name
   produced by ``activation_cache_filename``).
2. A metadata sidecar is written that documents the calibration hash,
   assignment digest, model, profile, dtype, and input-rows policy. This is the
   guard against silently pairing a stale activation dir with the wrong probe.
3. Downstream activations actually change when an upstream Linear's assigned
   format is non-BF16 (otherwise the loop is a no-op and the design has no
   reason to exist).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.multi_shot import (
    _drain_snaps_to_probe_format,
    _write_metadata,
    recache_calibration_activations_for_cost,
)
from prismaquant.perturbed_x_cache import (
    PerturbedActivationCache,
    activation_cache_filename,
    calibration_data_hash,
    iter_calibration_forwards,
)
from prismaquant.production_weight_cache import ProductionWeightCache


class _TwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _load_inputs(cache_dir: Path, name: str) -> torch.Tensor:
    path = cache_dir / activation_cache_filename(name)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["inputs"]


def _make_production_cache(
    tmp_path: Path,
    *,
    name: str,
    fmt: str,
    rendered: torch.Tensor,
) -> ProductionWeightCache:
    """Build a ProductionWeightCache with one rendered weight on disk."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rendered_path = tmp_path / f"{name.replace('.', '__')}__{fmt}.pt"
    torch.save(rendered, rendered_path)
    return ProductionWeightCache(
        weights={(name, fr.canonical_format_name(fmt)): rendered_path.name},
        levers={"gptq": True},
        cache_dir=str(tmp_path),
    )


def test_drain_snaps_writes_probe_format(tmp_path):
    """A captured PerturbedActivationCache drains to per-Linear .pt files
    that ActivationIndex can read."""
    torch.manual_seed(0)
    model = _TwoLinear().eval()
    x = torch.randn(4, 64, dtype=torch.float32)

    builder = PerturbedActivationCache(
        model,
        {"fc1": "BF16", "fc2": "BF16"},
        tmp_path,
        input_rows=4,
        cal_hash="fixed",
        capture_inputs=True,
    )
    builder.install()
    try:
        with torch.no_grad():
            model(x)
    finally:
        builder.remove()

    written = _drain_snaps_to_probe_format(
        builder,
        tmp_path,
        activation_dtype=torch.float32,
        progress=False,
    )
    assert set(written) == {"fc1", "fc2"}

    fc1_blob = torch.load(
        tmp_path / activation_cache_filename("fc1"),
        map_location="cpu",
        weights_only=False,
    )
    assert set(fc1_blob.keys()) == {"inputs", "name"}
    assert fc1_blob["name"] == "fc1"
    assert fc1_blob["inputs"].dtype == torch.float32
    torch.testing.assert_close(
        fc1_blob["inputs"], x.reshape(-1, 64), rtol=0.0, atol=0.0,
    )


def test_metadata_records_recache_provenance(tmp_path):
    """metadata.json carries the fields needed to detect stale pairing."""
    target = _write_metadata(
        tmp_path,
        model="test-model",
        calibration_hash="cal-hash-abc",
        assignment_sha256="asn-sha-def",
        input_rows=128,
        activation_dtype=torch.float32,
        include_activation_quant=True,
        profile_name="test-profile",
        n_linears_written=42,
        seqlen=1024,
        n_samples=8,
        shot_index=2,
        source_layer_config="/work/shot_1/layer_config.json",
    )
    payload = json.loads(target.read_text())
    assert payload["schema"] == "prismaquant.multi_shot.activation_recache.v1"
    assert payload["model"] == "test-model"
    assert payload["calibration_hash"] == "cal-hash-abc"
    assert payload["assignment_sha256"] == "asn-sha-def"
    assert payload["input_rows"] == 128
    assert payload["activation_dtype"] == "float32"
    assert payload["include_activation_quant"] is True
    assert payload["profile"] == "test-profile"
    assert payload["n_linears_written"] == 42
    assert payload["calib_seqlen"] == 1024
    assert payload["n_calib_samples"] == 8
    assert payload["shot_index"] == 2
    assert payload["source_layer_config"] == "/work/shot_1/layer_config.json"


def test_recache_writes_full_pipeline_format(tmp_path):
    """End-to-end: recache_calibration_activations_for_cost produces files
    in the format incremental_measure_quant_cost expects."""
    torch.manual_seed(1)
    model = _TwoLinear().eval()
    x = torch.randn(8, 64, dtype=torch.float32)

    nvfp4 = fr.get_format("NVFP4")
    fc1_w_orig = model.fc1.weight.detach().clone()
    fc1_w_quantized = nvfp4.quantize_dequantize(fc1_w_orig)
    cache = _make_production_cache(
        tmp_path / "prod_cache",
        name="fc1",
        fmt="NVFP4",
        rendered=fc1_w_quantized,
    )

    out_dir = tmp_path / "act"
    manifest = recache_calibration_activations_for_cost(
        model,
        x,
        {"fc1": "NVFP4", "fc2": "BF16"},
        cache,
        out_dir,
        input_rows=8,
        progress=False,
        preload_production_cache=False,
        source_layer_config="/tmp/layer_config_shot_1.json",
        shot_index=2,
        n_samples=8,
        seqlen=1,
    )
    assert manifest["n_linears"] == 2
    assert set(manifest["written"]) == {"fc1", "fc2"}
    assert manifest["calibration_hash"] == calibration_data_hash(x)
    assert manifest["missing"] == []

    metadata_path = Path(manifest["metadata_path"])
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text())
    assert payload["n_linears_written"] == 2
    assert payload["shot_index"] == 2
    assert payload["source_layer_config"] == "/tmp/layer_config_shot_1.json"

    # ActivationIndex semantics: per-Linear file exists with the expected
    # name; load() returns the inputs tensor on CPU.
    fc1_inputs = _load_inputs(out_dir, "fc1")
    fc2_inputs = _load_inputs(out_dir, "fc2")
    assert fc1_inputs.shape == (8, 64)
    assert fc2_inputs.shape == (8, 64)


def test_recache_propagates_upstream_quantization(tmp_path):
    """When fc1 is rendered to NVFP4, fc2's captured input must reflect the
    quantized output of fc1, not the BF16 baseline. This is the property the
    multi-shot loop is for — without it, shot 2 would just see shot 1's costs."""
    torch.manual_seed(2)
    model = _TwoLinear().eval()
    x = torch.randn(8, 64, dtype=torch.float32)

    fc1_w_orig = model.fc1.weight.detach().clone()
    nvfp4 = fr.get_format("NVFP4")
    fc1_w_quantized = nvfp4.quantize_dequantize(fc1_w_orig)
    cache = _make_production_cache(
        tmp_path / "prod_cache",
        name="fc1",
        fmt="NVFP4",
        rendered=fc1_w_quantized,
    )

    out_dir = tmp_path / "act"
    recache_calibration_activations_for_cost(
        model,
        x,
        {"fc1": "NVFP4", "fc2": "BF16"},
        cache,
        out_dir,
        input_rows=8,
        progress=False,
        preload_production_cache=False,
    )

    fc1_inputs = _load_inputs(out_dir, "fc1").to(torch.float32)
    fc2_inputs = _load_inputs(out_dir, "fc2").to(torch.float32)

    # PerturbedActivationCache._capture snapshots the *raw* input before any
    # activation quantization the pre-hook may apply. The downstream cost step
    # decides for itself how to act-quantize each candidate format. So fc1's
    # captured inputs match the raw model input x.
    torch.testing.assert_close(fc1_inputs, x.reshape(-1, 64), rtol=0.0, atol=0.0)

    # fc2's captured inputs are fc1's output under (act_quant_x, quant_W_fc1).
    # This is the multi-shot-meaningful property: shot-2's cost for fc2 is
    # measured against activations that already reflect fc1's quantization.
    act_quantized_x = nvfp4.activation_quantize_dequantize(x)
    expected_fc2 = F.linear(act_quantized_x, fc1_w_quantized).reshape(-1, 64)
    torch.testing.assert_close(fc2_inputs, expected_fc2, rtol=0.01, atol=0.01)

    # And the headline property: fc2's captured inputs in this run differ
    # meaningfully from the BF16 baseline (i.e. recache actually does work).
    baseline_fc2 = F.linear(x, fc1_w_orig).reshape(-1, 64)
    diff = (fc2_inputs - baseline_fc2).abs().mean()
    assert diff > 1e-4, (
        f"fc2 inputs identical to BF16 baseline (mean abs diff {diff}); "
        "recache loop would be a no-op"
    )


def test_recache_rejects_empty_assignment(tmp_path):
    model = _TwoLinear().eval()
    cache = ProductionWeightCache(
        weights={},
        levers={"gptq": True},
        cache_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="assignment"):
        recache_calibration_activations_for_cost(
            model,
            torch.randn(2, 64),
            {},
            cache,
            tmp_path / "act",
            progress=False,
            preload_production_cache=False,
        )


def test_recache_rejects_missing_production_cache(tmp_path):
    model = _TwoLinear().eval()
    with pytest.raises(ValueError, match="production_weight_cache"):
        recache_calibration_activations_for_cost(
            model,
            torch.randn(2, 64),
            {"fc1": "NVFP4"},
            None,
            tmp_path / "act",
            progress=False,
            preload_production_cache=False,
        )


def test_recache_respects_input_rows_cap(tmp_path):
    """input_rows controls how many calibration rows land per Linear, matching
    the probe's SharedRowSubsampler semantics. Cap is enforced even when the
    forward processes more rows than the cap."""
    torch.manual_seed(3)
    model = _TwoLinear().eval()
    x = torch.randn(32, 64, dtype=torch.float32)

    nvfp4 = fr.get_format("NVFP4")
    fc1_w_quantized = nvfp4.quantize_dequantize(model.fc1.weight.detach().clone())
    cache = _make_production_cache(
        tmp_path / "prod_cache",
        name="fc1",
        fmt="NVFP4",
        rendered=fc1_w_quantized,
    )

    recache_calibration_activations_for_cost(
        model,
        x,
        {"fc1": "NVFP4", "fc2": "BF16"},
        cache,
        tmp_path / "act",
        input_rows=4,
        progress=False,
        preload_production_cache=False,
    )

    fc1_inputs = _load_inputs(tmp_path / "act", "fc1")
    fc2_inputs = _load_inputs(tmp_path / "act", "fc2")
    assert fc1_inputs.shape == (4, 64)
    assert fc2_inputs.shape == (4, 64)
