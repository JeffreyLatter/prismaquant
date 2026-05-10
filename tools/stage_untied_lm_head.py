#!/usr/bin/env python3
"""Stage a tied-embedding HF checkpoint with a materialized lm_head.

HALO needs separate embedding and lm_head tensors: embedding is right-rotated
to enter the rotated residual frame, while lm_head is right-rotated after
folding the final norm gamma. Tied checkpoints do not have two independent
storage locations, so this tool creates a lightweight sibling checkpoint that
hardlinks all existing files and adds one safetensors shard containing
``lm_head.weight = embed_tokens.weight``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def _set_tie_word_embeddings_false(cfg: dict) -> None:
    cfg["tie_word_embeddings"] = False
    for key in ("text_config", "language_model_config", "llm_config"):
        child = cfg.get(key)
        if isinstance(child, dict):
            child["tie_word_embeddings"] = False


def _find_embed_key(weight_map: dict[str, str], explicit: str | None) -> str:
    if explicit:
        if explicit not in weight_map:
            raise KeyError(f"embed key {explicit!r} not present in index")
        return explicit
    for candidate in (
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    ):
        if candidate in weight_map:
            return candidate
    matches = [k for k in weight_map if k.endswith(".embed_tokens.weight")]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        "could not infer embedding tensor key; pass --embed-key. "
        f"matches={matches[:8]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embed-key", default=None)
    parser.add_argument("--lm-head-key", default="lm_head.weight")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.is_dir():
        raise SystemExit(f"source is not a directory: {source}")
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists; pass --force to replace: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    cfg_path = source / "config.json"
    idx_path = source / "model.safetensors.index.json"
    if not cfg_path.exists():
        raise SystemExit(f"missing config.json: {cfg_path}")
    if not idx_path.exists():
        raise SystemExit(
            "only indexed safetensors checkpoints are supported by this "
            f"staging tool; missing {idx_path}"
        )

    cfg = json.loads(cfg_path.read_text())
    idx = json.loads(idx_path.read_text())
    weight_map = dict(idx.get("weight_map", {}))
    embed_key = _find_embed_key(weight_map, args.embed_key)
    lm_head_key = str(args.lm_head_key)

    for item in source.iterdir():
        if item.name in {"config.json", "model.safetensors.index.json"}:
            continue
        if item.is_file():
            _link_or_copy(item, output / item.name)
        elif item.is_dir():
            shutil.copytree(item, output / item.name, symlinks=True)

    _set_tie_word_embeddings_false(cfg)
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    added_lm_head = lm_head_key not in weight_map
    tensor_nbytes = 0
    if added_lm_head:
        shard_name = weight_map[embed_key]
        with safe_open(source / shard_name, framework="pt", device="cpu") as fh:
            embed = fh.get_tensor(embed_key).contiguous()
        tensor_nbytes = int(embed.numel() * embed.element_size())
        lm_head_shard = "prismaquant-untied-lm-head.safetensors"
        save_file({lm_head_key: embed}, output / lm_head_shard)
        weight_map[lm_head_key] = lm_head_shard

    idx["weight_map"] = weight_map
    metadata = dict(idx.get("metadata", {}) or {})
    try:
        metadata["total_size"] = str(
            int(metadata.get("total_size", 0)) + tensor_nbytes
        )
    except (TypeError, ValueError):
        metadata["total_size"] = str(tensor_nbytes)
    idx["metadata"] = metadata
    (output / "model.safetensors.index.json").write_text(
        json.dumps(idx, indent=2) + "\n"
    )

    manifest = {
        "source": str(source),
        "embed_key": embed_key,
        "lm_head_key": lm_head_key,
        "added_lm_head": added_lm_head,
        "lm_head_nbytes": tensor_nbytes,
    }
    (output / "prismaquant_untied_lm_head.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        "[stage-untied-lm-head] wrote "
        f"{output} embed={embed_key} lm_head={lm_head_key} "
        f"added={added_lm_head}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
