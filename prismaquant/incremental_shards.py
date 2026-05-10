"""Shared helpers for incremental probe/cost shard pickles."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def read_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def annotate_incremental_shard(path: Path, extra_meta: dict[str, Any]) -> None:
    data = read_pickle(path)
    meta = dict(data.get("meta", {}))
    inc = dict(meta.get("incremental_shard", {}))
    inc.update(extra_meta)
    meta["incremental_shard"] = inc
    data["meta"] = meta
    with open(path, "wb") as f:
        pickle.dump(data, f)
