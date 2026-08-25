"""Regression for issue #4 — meta-init CUDA mask poisoning the fla fast path.

Several transformers modeling files (Qwen3.5/3.6 MoE, Qwen3-Next, OLMo-hybrid)
bind their gated-delta-rule FAST PATH at *module import time* behind a
`@lru_cache`d, CUDA-gated `if is_flash_linear_attention_available():`. PrismaQuant
builds the meta skeleton inside `_mask_cuda_queries_during_meta_init`, which sets
`torch.cuda.is_available = lambda: False`. If the modeling module is first
imported in that window, the availability check caches False and the fast path
is silently lost for the whole process.

The fix re-primes those lru_cached availability checks with CUDA visible BEFORE
masking. This test asserts the priming happens, clears the cache, and runs
*before* the mask takes effect — without needing fla/CUDA actually installed.
"""
import torch

from transformers.utils import import_utils as tiu

from prismaquant.streaming_model import _mask_cuda_queries_during_meta_init


def test_meta_init_mask_primes_fla_availability_before_masking(monkeypatch):
    # gb10 flips the mask's default to off (NVML probing during meta-init
    # hangs on some GB10 systems) — force it on so this test exercises the
    # priming mechanism regardless of that hardware-specific default.
    monkeypatch.setenv("PRISMAQUANT_MASK_CUDA_DURING_META_INIT", "1")
    original_is_available = torch.cuda.is_available
    record = {}

    def make(name):
        def fn():
            # `torch.cuda.is_available` is replaced by the masking lambda only
            # once masking is in effect — so identity-compare to the original.
            record[name + "_masked_at_call"] = (
                torch.cuda.is_available is not original_is_available)
            record[name + "_called"] = True
            return True
        fn.cache_clear = lambda: record.__setitem__(name + "_cleared", True)
        return fn

    monkeypatch.setattr(tiu, "is_flash_linear_attention_available",
                        make("fla"), raising=False)
    monkeypatch.setattr(tiu, "is_causal_conv1d_available",
                        make("conv"), raising=False)

    # The mask only activates when CUDA isn't already initialized — the
    # exact window the bug occurs in. In full-suite order on a GPU host an
    # earlier test has usually initialized CUDA; re-run this test in a
    # fresh interpreter so the precondition (and therefore the behavior
    # under test) is real rather than vacuously skipped. (Test-isolation
    # fix for the known full-suite failure, 2026-06-09 review.)
    if torch.cuda.is_initialized():
        import subprocess
        import sys
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", __file__ + "::"
             + "test_meta_init_mask_primes_fla_availability_before_masking",
             "-q", "-p", "no:cacheprovider"],
            capture_output=True, text=True,
            env={**__import__("os").environ,
                 "CUDA_VISIBLE_DEVICES":
                     __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "0")},
        )
        assert proc.returncode == 0, (
            "subprocess re-run failed:\n" + proc.stdout + proc.stderr)
        return
    assert not torch.cuda.is_initialized()
    with _mask_cuda_queries_during_meta_init("[test]"):
        pass

    # both availability checks were primed and their caches cleared ...
    assert record.get("fla_called") and record.get("conv_called"), \
        "fla/conv availability was not primed by the meta-init mask"
    assert record.get("fla_cleared") and record.get("conv_cleared"), \
        "lru_cache was not cleared before re-priming"
    # ... and crucially BEFORE torch.cuda was masked (so the real CUDA state
    # is cached, not the masked False).
    assert record.get("fla_masked_at_call") is False, \
        "fla availability primed AFTER the mask took effect (poisoning not fixed)"
    assert record.get("conv_masked_at_call") is False


def test_meta_init_mask_restores_cuda_queries():
    """The mask must restore the real torch.cuda functions on exit."""
    before = (torch.cuda.is_available, torch.cuda.device_count,
              torch.cuda.current_device)
    with _mask_cuda_queries_during_meta_init("[test]"):
        pass
    after = (torch.cuda.is_available, torch.cuda.device_count,
             torch.cuda.current_device)
    assert before == after
