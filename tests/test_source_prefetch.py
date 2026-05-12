import json

import pytest

from prismaquant.source_prefetch import prefetch_safetensors_checkpoint


def test_source_prefetch_requires_local_safetensors(tmp_path):
    with pytest.raises(RuntimeError, match="no local safetensors"):
        prefetch_safetensors_checkpoint(tmp_path, mode="require", progress=False)


def test_source_prefetch_reads_unique_index_shards(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"a" * 17)
    (tmp_path / "b.safetensors").write_bytes(b"b" * 19)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {
                "model.layers.0.weight": "a.safetensors",
                "model.layers.1.weight": "a.safetensors",
                "model.layers.2.weight": "b.safetensors",
            }
        })
    )

    stats = prefetch_safetensors_checkpoint(
        tmp_path,
        mode="require",
        max_resident_bytes=1024,
        workers=2,
        chunk_mb=1,
        progress=False,
    )

    assert stats["shards"] == 2
    assert stats["bytes"] == 36
    assert stats["prefetched_bytes"] == 36
    assert stats["skipped"] is False


def test_source_prefetch_budget_fail_fast(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 32)

    with pytest.raises(RuntimeError, match="budget"):
        prefetch_safetensors_checkpoint(
            tmp_path,
            mode="require",
            max_resident_bytes=16,
            progress=False,
        )

