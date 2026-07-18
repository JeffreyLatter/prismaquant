"""Streaming NVFP4-CB exporter — CPU-only, tiny synthetic, compile-off.

Pins the streaming exporter (prismaquant.export_nvfp4_cb_streaming) against the
in-memory export_nvfp4_cb: byte-identical packed output, bounded peak
residency, per-expert->stacked bridging, fp8-source dequant-on-read, and the
stock-CT scope gate. No GPU, no torch.compile (PRISMAQUANT_CB_ENCODE_COMPILE=0).
"""
from __future__ import annotations

import json
import os
import shutil
import weakref
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

os.environ["PRISMAQUANT_CB_ENCODE_COMPILE"] = "0"

from prismaquant.export_nvfp4_cb import export_nvfp4_cb  # noqa: E402
from prismaquant.export_nvfp4_cb_streaming import (  # noqa: E402
    _LazySkeleton,
    export_nvfp4_cb_streaming,
)

_ROOT = Path("/home/rob/dq-runs/nvfp4-cb-phase0/stream-test/pytest")


@pytest.fixture
def workdir():
    _ROOT.mkdir(parents=True, exist_ok=True)
    import tempfile
    d = Path(tempfile.mkdtemp(dir=_ROOT))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _write_model(mdl: Path, tensors: dict, hid: int = 256):
    mdl.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(json.dumps({"hidden_size": hid}))


def _assign(path: Path, mapping: dict):
    path.write_text(json.dumps(mapping))


def _tensors_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


# --- byte-identity: dense CB + BF16 + stacked-3D experts + fp8_cb -----------

def test_streaming_byte_identical_dense_and_stacked(workdir):
    torch.manual_seed(0)
    mdl = workdir / "model"
    tens = {
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.experts.gate_up_proj.weight":
            (torch.randn(3, 64, 256) * 0.3).to(torch.bfloat16),   # stacked
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    }
    _write_model(mdl, tens)
    ap = workdir / "a.json"
    _assign(ap, {
        "model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb",
                                            "cb_k": 16},
        "model.layers.0.mlp.experts.gate_up_proj": {"data_type": "fp8_cb",
                                                    "cb_k": 40}})
    cw = {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05,
          "model.layers.0.mlp.experts.gate_up_proj": torch.rand(3, 1, 256)
          + 0.05}
    cm = export_nvfp4_cb(mdl, ap, workdir / "m", cw, device="cpu")
    cs = export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    assert dict(cm) == dict(cs)
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    assert _tensors_equal(tm, ts)
    qm = json.loads((workdir / "m" / "quant_config.json").read_text())
    qs = json.loads((workdir / "s" / "quant_config.json").read_text())
    assert qm["config_groups"] == qs["config_groups"]
    assert qm["ignore"] == qs["ignore"]
    assert qs["provenance"]["streaming"] is True
    # codebook sidecars identical
    cbm = load_file(str(workdir / "m" / "cm.pqcb")) if (
        workdir / "m" / "cm.pqcb").exists() else None
    if qs.get("codebook_file"):
        cb_s = load_file(str(workdir / "s" / qs["codebook_file"]))
        cb_m = load_file(str(workdir / "m" / qm["codebook_file"]))
        assert _tensors_equal(cb_m, cb_s)


# --- per-expert -> stacked bridging (Hy3 layout) ---------------------------

def _per_expert_model(mdl: Path, E=3, inter=256, hid=256, seed=1):
    torch.manual_seed(seed)
    pe = {}
    for e in range(E):
        pe[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = (
            torch.randn(inter, hid) * 0.3).to(torch.bfloat16)
        pe[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = (
            torch.randn(inter, hid) * 0.3).to(torch.bfloat16)
        pe[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = (
            torch.randn(hid, inter) * 0.3).to(torch.bfloat16)
    pe["model.norm.weight"] = torch.ones(hid, dtype=torch.bfloat16)
    _write_model(mdl, pe, hid)
    # equivalent pre-stacked model for the in-memory reference
    gu = torch.stack([
        torch.cat([pe[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"],
                   pe[f"model.layers.1.mlp.experts.{e}.up_proj.weight"]], 0)
        for e in range(E)])
    dn = torch.stack([pe[f"model.layers.1.mlp.experts.{e}.down_proj.weight"]
                      for e in range(E)])
    st = {"model.layers.1.mlp.experts.gate_up_proj.weight": gu,
          "model.layers.1.mlp.experts.down_proj.weight": dn,
          "model.norm.weight": pe["model.norm.weight"]}
    return st


def test_streaming_per_expert_bridging(workdir):
    E, inter, hid = 3, 256, 256
    st = _per_expert_model(workdir / "pe", E, inter, hid)
    _write_model(workdir / "st", st, hid)
    ap = workdir / "a.json"
    _assign(ap, {
        "model.layers.1.mlp.experts.gate_up_proj": {"data_type": "nvfp4_cb",
                                                    "cb_k": 16},
        "model.layers.1.mlp.experts.down_proj": {"data_type": "nvfp4_cb",
                                                 "cb_k": 16}})
    cw = {"model.layers.1.mlp.experts.gate_up_proj": torch.rand(E, 1, hid)
          + 0.05,
          "model.layers.1.mlp.experts.down_proj": torch.rand(E, 1, inter)
          + 0.05}
    export_nvfp4_cb(workdir / "st", ap, workdir / "m", cw, device="cpu")
    export_nvfp4_cb_streaming(workdir / "pe", ap, workdir / "s", cw,
                              device="cpu")
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    for key in ("model.layers.1.mlp.experts.gate_up_proj.cb_qweight",
                "model.layers.1.mlp.experts.down_proj.cb_qweight"):
        assert torch.equal(tm[key], ts[key]), key


# --- bounded peak residency ------------------------------------------------

def test_streaming_peak_residency(workdir, monkeypatch):
    torch.manual_seed(2)
    tens = {"model.norm.weight": torch.ones(256, dtype=torch.bfloat16)}
    # many passthrough tensors so full materialization would be obvious
    for i in range(20):
        tens[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(
            256, dtype=torch.bfloat16)
    tens["model.layers.0.self_attn.q_proj.weight"] = (
        torch.randn(128, 256) * 0.3).to(torch.bfloat16)
    mdl = workdir / "model"
    _write_model(mdl, tens)
    ap = workdir / "a.json"
    _assign(ap, {"model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb",
                                                     "cb_k": 16}})
    cw = {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05}

    live = {"n": 0, "peak": 0}
    orig = _LazySkeleton.load

    def counting(self, name):
        t = orig(self, name)
        live["n"] += 1
        live["peak"] = max(live["peak"], live["n"])
        weakref.finalize(t, lambda: live.__setitem__("n", live["n"] - 1))
        return t
    monkeypatch.setattr(_LazySkeleton, "load", counting)
    export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    # 22 source tensors total; peak resident must be a tiny constant, not ~22.
    assert live["peak"] <= 4, f"peak residency {live['peak']} too high"


# --- fp8-source dequant-on-read (DSv4 ingestion) ---------------------------

def test_streaming_fp8_source_dequant_on_read(workdir):
    from prismaquant.layer_streaming import _dequant_fp8_block_weight
    torch.manual_seed(3)
    out_f, in_f = 256, 256
    w_fp8 = (torch.randn(out_f, in_f) * 0.3).to(torch.float8_e4m3fn)
    scale_inv = (torch.rand(out_f // 128, in_f // 128) + 0.1).float()
    mdl = workdir / "model"
    _write_model(mdl, {
        "model.layers.0.mlp.down_proj.weight": w_fp8,
        "model.layers.0.mlp.down_proj.weight_scale_inv": scale_inv,
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16)},
        hid=256)
    (mdl / "config.json").write_text(json.dumps({
        "hidden_size": 256,
        "quantization_config": {"weight_block_size": [128, 128]}}))
    sk = _LazySkeleton(mdl)
    got = sk.dequant_weight("model.layers.0.mlp.down_proj.weight")
    ref = _dequant_fp8_block_weight(w_fp8, scale_inv, block=(128, 128)).float()
    assert torch.equal(got, ref)


# --- stock-CT scope gate ---------------------------------------------------

def test_streaming_rejects_stock_ct(workdir):
    torch.manual_seed(4)
    mdl = workdir / "model"
    _write_model(mdl, {
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16)})
    ap = workdir / "a.json"
    _assign(ap, {"model.layers.0.self_attn.q_proj":
                 {"data_type": "nv_fp", "bits": 4}})     # -> NVFP4 (stock CT)
    cw = {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05}
    with pytest.raises(ValueError, match="stock-CT"):
        export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")


# --- lazy skeleton: single-file + sharded, metadata without data load ------

def test_lazy_skeleton_metadata(workdir):
    torch.manual_seed(5)
    mdl = workdir / "model"
    _write_model(mdl, {
        "a.weight": torch.randn(16, 32),
        "b.weight": (torch.randn(8, 8)).to(torch.bfloat16)})
    sk = _LazySkeleton(mdl)
    assert "a.weight" in sk and "b.weight" in sk
    assert sk.get_shape("a.weight") == (16, 32)
    assert sk.get_dtype("b.weight") == torch.bfloat16
    assert torch.equal(sk.load("a.weight"), load_file(
        str(mdl / "model.safetensors"))["a.weight"])
