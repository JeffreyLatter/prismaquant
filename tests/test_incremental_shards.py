from __future__ import annotations

import pickle

from prismaquant.incremental_shards import annotate_incremental_shard, read_pickle


def test_annotate_incremental_shard_preserves_existing_meta(tmp_path):
    path = tmp_path / "shard.pkl"
    with path.open("wb") as f:
        pickle.dump(
            {
                "stats": {"layer": {"h_trace": 1.0}},
                "meta": {
                    "model": "/tmp/model",
                    "incremental_shard": {"shard_idx": 0},
                },
            },
            f,
        )

    annotate_incremental_shard(path, {"linear_include": "re:layer"})

    payload = read_pickle(path)
    assert payload["stats"] == {"layer": {"h_trace": 1.0}}
    assert payload["meta"]["model"] == "/tmp/model"
    assert payload["meta"]["incremental_shard"] == {
        "shard_idx": 0,
        "linear_include": "re:layer",
    }
