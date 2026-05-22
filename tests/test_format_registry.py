from __future__ import annotations

import torch

from prismaquant import format_registry as fr
from prismaquant.export_native_compressed import (
    _mxfp8_dequantize_2d,
    quantize_dequantize_mxfp8,
)


def test_plain_fp8_rtn_uses_eager_path(monkeypatch):
    compile_calls = []

    def fake_compile(fn, *args, **kwargs):
        compile_calls.append((fn, args, kwargs))

        def compiled(*_args, **_kwargs):
            raise AssertionError("plain FP8 RTN should not use torch.compile")

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    quantize = fr._make_rtn("fp8_e4m3", 0)
    x = torch.linspace(-3.1, 3.1, steps=64, dtype=torch.float32).reshape(2, 32)
    y = quantize(x)

    assert compile_calls == []
    assert y.shape == x.shape
    assert not torch.equal(y, x)


def test_mx_e8m0_rtn_matches_export_scale_rounding():
    w = torch.linspace(-3.7, 3.7, steps=64, dtype=torch.float32).reshape(2, 32)

    registry = fr.get_format("MXFP8_E4M3").quantize_dequantize(w)
    export_q, export_scales = quantize_dequantize_mxfp8(w)
    export = _mxfp8_dequantize_2d(export_q, export_scales)

    assert torch.allclose(registry, export, atol=0.0, rtol=0.0)
