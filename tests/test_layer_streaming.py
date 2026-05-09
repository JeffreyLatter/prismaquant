import unittest

import torch
import torch.nn as nn

from prismaquant.layer_streaming import LayerCache
from prismaquant.streaming_model import _init_rotary_inplace


class _ResettableRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = object()
        with torch.device("meta"):
            self.register_buffer("inv_freq", torch.empty(4), persistent=False)
            self.register_buffer("cos_cached", torch.empty(8, 8), persistent=False)
            self.register_buffer("sin_cached", torch.empty(8, 8), persistent=False)

    def compute_default_rope_parameters(self, config, device):
        del config
        return torch.arange(4, device=device, dtype=torch.float32), 1.0

    def reset_rope_cache(self, device=None):
        inv, self.attention_scaling = self.compute_default_rope_parameters(
            self.config,
            device,
        )
        self.register_buffer("inv_freq", inv, persistent=False)
        self.register_buffer("cos_cached", torch.ones(8, 8, device=device), persistent=False)
        self.register_buffer("sin_cached", torch.zeros(8, 8, device=device), persistent=False)


class _RotaryBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotary_emb = _ResettableRotary()


class TestLayerCache(unittest.TestCase):
    def test_rotary_reset_cache_materializes_all_buffers(self):
        base = _RotaryBase()

        _init_rotary_inplace(base, torch.device("cpu"), torch.float32)

        self.assertFalse(base.rotary_emb.inv_freq.is_meta)
        self.assertFalse(base.rotary_emb.cos_cached.is_meta)
        self.assertFalse(base.rotary_emb.sin_cached.is_meta)

    def test_eviction_and_residency_summary(self):
        cache = LayerCache(max_bytes=64)
        t0 = {"a": torch.zeros(8, dtype=torch.float32)}
        t1 = {"b": torch.zeros(8, dtype=torch.float32)}
        t2 = {"c": torch.zeros(8, dtype=torch.float32)}

        cache.put(0, t0)
        cache.put(1, t1)
        self.assertIs(cache.get(0), t0)
        cache.put(2, t2)

        # Accessing layer 0 made it MRU, so layer 1 should be evicted.
        self.assertTrue(cache.peek(0))
        self.assertFalse(cache.peek(1))
        self.assertTrue(cache.peek(2))
        self.assertIn("cpu:2", cache.summary())

    def test_clear_resets_contents_but_keeps_hit_counters(self):
        cache = LayerCache(max_bytes=128)
        cache.put(0, {"a": torch.zeros(4, dtype=torch.float32)})
        cache.get(0)
        cache.get(3)
        cache.clear()

        self.assertEqual(len(cache._cache), 0)
        self.assertEqual(cache.total_bytes, 0)
        self.assertEqual(cache.residency_summary(), "empty")
        self.assertIn("hits=1", cache.summary())
        self.assertIn("misses=1", cache.summary())

    def test_budget_eviction_under_tight_cap(self):
        """LayerCache eviction kicks in as soon as total bytes would
        exceed the cap — this is the path the large-model (122B)
        streaming probe depends on."""
        # Each entry is 4 floats * 4 bytes = 16 bytes. Cap = 40 bytes
        # admits at most 2 entries.
        cache = LayerCache(max_bytes=40)
        for i in range(5):
            cache.put(i, {"a": torch.zeros(4, dtype=torch.float32)})
        self.assertLessEqual(len(cache._cache), 2)
        self.assertLessEqual(cache.total_bytes, 40)
        # The two most-recent entries survive (strict LRU).
        self.assertTrue(cache.peek(4))
        self.assertTrue(cache.peek(3))
        self.assertFalse(cache.peek(0))

    def test_huge_budget_degenerates_to_no_eviction(self):
        """With a budget much larger than any realistic model, the
        cache must keep every layer resident. This is the no-op path
        small models take: incremental probe still calls put/get, but
        eviction never fires, so correctness matches the old
        whole-model-in-RAM branch."""
        # 1 GB budget vs. 32 bytes/entry — should comfortably hold all.
        cache = LayerCache(max_bytes=1 << 30)
        for i in range(32):
            cache.put(i, {"a": torch.zeros(8, dtype=torch.float32)})
        self.assertEqual(len(cache._cache), 32)
        for i in range(32):
            self.assertTrue(cache.peek(i))
        # Hit/miss counters still track access correctly.
        cache.get(0)
        cache.get(1)
        cache.get(999)  # miss
        self.assertIn("hits=2", cache.summary())
        self.assertIn("misses=1", cache.summary())


if __name__ == "__main__":
    unittest.main()
