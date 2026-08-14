"""The memoized NVFP4 codebook must be bit-identical to a freshly built one.

``_nvfp4_codebook`` is called once per column-quantize inside the GPTQ render,
and it was rebuilding a constant Python list into a CUDA tensor every time --
2.279 s of a 3.705 s single-Linear render. Caching it is a pure speed change,
which means the bar it has to clear is EXACTNESS, not accuracy: the production
render, the KL validation and the exported bytes must stay the same rendering
(principle 8), so a cached codebook that differed from a fresh one even in the
last bit would be a rendering confound rather than an optimization.

These tests pin three things: the tensor itself is identical, a whole rendered
Linear is identical, and the cache cannot serve a tensor from the wrong device.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prismaquant import export_native_compressed as enc  # noqa: E402


def _fresh(device, dtype=torch.float32):
    """What the function returned before it was memoized."""
    return torch.tensor(enc.FLOAT_TO_E2M1, device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cached_codebook_is_bit_identical_to_fresh(dtype):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cached = enc._nvfp4_codebook(torch.device(dev), dtype=dtype)
    assert torch.equal(cached, _fresh(dev, dtype))
    assert cached.dtype == dtype
    # Second call must hand back the SAME object -- otherwise it is not caching
    # and the measured win silently disappears.
    assert enc._nvfp4_codebook(torch.device(dev), dtype=dtype) is cached


def test_cache_is_keyed_by_device():
    """A cached tensor from the wrong device is a correctness bug, not a slow path."""
    if not torch.cuda.is_available():
        pytest.skip("needs a second device to distinguish")
    cpu = enc._nvfp4_codebook(torch.device("cpu"))
    gpu = enc._nvfp4_codebook(torch.device("cuda:0"))
    assert cpu.device.type == "cpu"
    assert gpu.device.type == "cuda"
    assert torch.equal(cpu, gpu.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the production render is a GPU path")
def test_full_render_is_unchanged_by_the_cache():
    """End-to-end: the same Linear renders identically with and without the cache.

    This is the assertion that actually matters. It renders through the real
    production path with the shipping levers (gptq + static_act_order + JSO),
    once with the cache primed and once with it cleared and monkeypatched back
    to the original uncached implementation, and requires bitwise equality.
    """
    from prismaquant.production_weight_cache import render_production_weight

    torch.manual_seed(0)
    qname = "probe.linear"
    w = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16) * 0.02
    acts = {qname: torch.randn(128, 1024, device="cuda", dtype=torch.float32)}
    levers = {"gptq": True, "static_act_order": True, "joint_scale_opt": True,
              "gptq_damp_sweep": False, "gptq_fixed_damp": 1.0,
              "nvfp4_scale_rule": "joint_mse", "scale_sweep": False}

    def render():
        return render_production_weight(
            w, "NVFP4", qname=qname, activations=acts, levers=levers)

    with_cache = render()

    original = enc._nvfp4_codebook
    try:
        enc._nvfp4_codebook = lambda device, dtype=torch.float32: _fresh(
            device, dtype)
        without_cache = render()
    finally:
        enc._nvfp4_codebook = original

    assert torch.equal(with_cache, without_cache), (
        "the codebook cache changed the rendered weight; that is a rendering "
        "confound, not a speedup")
