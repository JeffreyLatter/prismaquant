"""Shared layer-streaming execution adapter for cost stages.

This is intentionally a thin consumer of :class:`StreamingContext`.  It does
not own weights or maintain another cache: decoder residency, prefetch, and
unload all go through the existing streaming-model machinery.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Iterator

import torch

from prismaquant.layer_streaming import (
    _call_layer,
    _compute_attention_mask,
    _compute_position_embeddings,
    _get_final_norm,
)


STREAMED_MODEL_IDENTITY_SCHEMA = "prismaquant.streamed_model.identity.v1"
STREAMED_MODEL_IDENTITY_CACHE_SCHEMA = (
    "prismaquant.streamed_model.identity_cache.v1"
)


@dataclass
class StreamedForwardBoundaries:
    """One exact source-model forward, cut at decoder-layer boundaries."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    position_embeddings: object
    attention_mask: object
    activations_cpu: list[torch.Tensor]
    shared_pass_state: object


class StreamedCausalLM:
    """Causal-LM forward adapter over an existing ``StreamingContext``.

    ``pin_layer_for_qname`` keeps exactly one decoder layer installed while a
    caller temporarily mutates and restores a serving unit in it.  All other
    layers continue to stream through the context's cache.  This is the seam
    used by empirical expert KL; AURA additionally consumes the explicit
    boundary/isolated-layer methods for its streamed adjoint.
    """

    def __init__(self, context, profile, *, prefetch_lookahead: int = 2):
        self.context = context
        self.model = context.model
        self.base_model = context.base_model
        self.layers = context.layers
        self.layers_prefix = str(context.layers_prefix)
        self.num_layers = int(context.num_layers)
        self.device = torch.device(context.device)
        self.dtype = context.dtype
        self.profile = profile
        self.prefetch_lookahead = max(1, int(prefetch_lookahead))
        self._pinned_layer: int | None = None

    def layer_index_for_qname(self, qname: str) -> int:
        match = re.match(
            rf"^{re.escape(self.layers_prefix)}([0-9]+)(?:\.|$)",
            str(qname),
        )
        if match is None:
            raise RuntimeError(
                f"streamed cost unit {qname!r} is not under decoder prefix "
                f"{self.layers_prefix!r}"
            )
        layer = int(match.group(1))
        if not 0 <= layer < self.num_layers:
            raise RuntimeError(
                f"streamed cost unit {qname!r} resolved invalid layer {layer}"
            )
        return layer

    @contextmanager
    def pin_layer(self, layer: int) -> Iterator[None]:
        layer = int(layer)
        if self._pinned_layer is not None:
            raise RuntimeError(
                f"streamed cost already pins layer {self._pinned_layer}; "
                f"cannot also pin layer {layer}"
            )
        self.context.install(layer)
        self._pinned_layer = layer
        try:
            yield
        finally:
            self._pinned_layer = None
            self.context.unload(layer)

    @contextmanager
    def pin_layer_for_qname(self, qname: str) -> Iterator[None]:
        with self.pin_layer(self.layer_index_for_qname(qname)):
            yield

    def _head(self):
        name = str(self.profile.lm_head_name())
        try:
            return self.model.get_submodule(name)
        except (AttributeError, KeyError):
            head = getattr(self.model, "lm_head", None)
            if head is None:
                raise RuntimeError(
                    f"streamed cost could not resolve profile lm_head {name!r}"
                )
            return head

    def _prepare(self, input_ids: torch.Tensor):
        ids = input_ids.to(self.device)
        position_ids = torch.arange(
            ids.size(-1), device=self.device
        ).unsqueeze(0)
        hidden = self.base_model.embed_tokens(ids).to(self.dtype)
        position_embeddings = _compute_position_embeddings(
            self.base_model, hidden, position_ids
        )
        attention_mask = _compute_attention_mask(
            self.base_model, hidden, position_ids
        )
        hidden = self.profile.expand_hidden_for_layers(
            hidden, self.base_model
        )
        return ids, position_ids, hidden, position_embeddings, attention_mask

    def _call(self, layer: int, hidden: torch.Tensor, *, batch, pass_state):
        return _call_layer(
            self.layers[layer],
            hidden,
            position_embeddings=batch.position_embeddings,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            **self.profile.extra_layer_kwargs(input_ids=batch.input_ids),
            pass_state=pass_state,
        )

    def _finish(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.profile.collapse_hidden_after_layers(
            hidden, self.base_model
        )
        norm = _get_final_norm(self.base_model)
        if norm is not None:
            hidden = norm(hidden)
        return self._head()(hidden)

    def capture_boundaries(
        self, input_ids: torch.Tensor
    ) -> StreamedForwardBoundaries:
        """Stream a no-grad source forward and retain only boundary acts."""
        ids, position_ids, hidden, position_embeddings, attention_mask = (
            self._prepare(input_ids)
        )
        batch = StreamedForwardBoundaries(
            input_ids=ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            activations_cpu=[],
            shared_pass_state=None,
        )
        pass_state = self.profile.new_forward_pass_state()
        batch.activations_cpu.append(hidden.detach().to("cpu"))
        for depth in range(self.prefetch_lookahead):
            self.context.schedule_prefetch(depth)
        for layer in range(self.num_layers):
            if self._pinned_layer != layer:
                self.context.install(layer)
            self.context.schedule_prefetch(layer + self.prefetch_lookahead)
            try:
                with torch.no_grad():
                    hidden = self._call(
                        layer, hidden, batch=batch, pass_state=pass_state
                    )
                batch.activations_cpu.append(hidden.detach().to("cpu"))
            finally:
                if self._pinned_layer != layer:
                    self.context.unload(layer)
        batch.shared_pass_state = self.profile.capture_forward_pass_state(
            pass_state
        )
        return batch

    def isolated_layer(
        self,
        batch: StreamedForwardBoundaries,
        layer: int,
        hidden: torch.Tensor,
        *,
        pass_state: dict | None,
    ) -> torch.Tensor:
        return self._call(layer, hidden, batch=batch, pass_state=pass_state)

    def tail_logits(
        self, batch: StreamedForwardBoundaries, hidden: torch.Tensor
    ) -> torch.Tensor:
        return self._finish(hidden)

    def __call__(self, input_ids: torch.Tensor, **_kwargs: Any):
        """Run an end-to-end no-cache forward while streaming body layers."""
        ids, position_ids, hidden, position_embeddings, attention_mask = (
            self._prepare(input_ids)
        )
        batch = StreamedForwardBoundaries(
            input_ids=ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            activations_cpu=[],
            shared_pass_state=None,
        )
        pass_state = self.profile.new_forward_pass_state()
        for depth in range(self.prefetch_lookahead):
            self.context.schedule_prefetch(depth)
        for layer in range(self.num_layers):
            if self._pinned_layer != layer:
                self.context.install(layer)
            self.context.schedule_prefetch(layer + self.prefetch_lookahead)
            try:
                hidden = self._call(
                    layer, hidden, batch=batch, pass_state=pass_state
                )
            finally:
                if self._pinned_layer != layer:
                    self.context.unload(layer)
        return SimpleNamespace(logits=self.tail_logits(batch, hidden))

    def shutdown(self) -> None:
        self.context.shutdown()


def build_streamed_causal_lm(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    offload_folder: str,
    profile,
    cache_headroom_gb: float | None = None,
    prefetch_workers: int | str | None = None,
    prefetch_min_available_gb: float | str | None = None,
    prefetch_lookahead: int = 2,
) -> StreamedCausalLM:
    """Build the repository's existing streaming context and wrap it."""
    from prismaquant.streaming_model import _build_streaming_context

    context = _build_streaming_context(
        model_path,
        device=device,
        dtype=dtype,
        offload_folder=offload_folder,
        cache_headroom_gb=cache_headroom_gb,
        prefetch_workers=prefetch_workers,
        prefetch_min_available_gb=prefetch_min_available_gb,
        log_prefix="[cost-streaming]",
    )
    return StreamedCausalLM(
        context, profile, prefetch_lookahead=prefetch_lookahead
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_streamed_model_identity(
    runner: StreamedCausalLM,
    source_model: str,
    *,
    identity_cache_path: str | Path | None = None,
) -> dict[str, object]:
    """Hash the complete checkpoint backing a streamed cost run.

    End-to-end KL/adjoint values depend on every body/head/norm weight, so a
    path, index, or target-unit hash is insufficient.  This hashes each unique
    source shard exactly once and folds those digests together with the live
    checkpoint-key map and resolved config.  It is an initialization integrity
    pass, not a residency mechanism; decoder execution still uses the existing
    streaming cache.
    """
    from prismaquant.cost_stage_checkpoint import (
        canonical_json,
        canonical_json_sha256,
    )

    config = getattr(runner.model, "config", None)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
    mapping = {
        str(live): str(checkpoint)
        for live, checkpoint in sorted(runner.context.weight_ckpt.items())
    }
    shard_paths = sorted({
        Path(path).resolve()
        for path in runner.context.weight_shard.values()
    }, key=str)
    if not shard_paths:
        raise RuntimeError(
            "streamed model identity found no source checkpoint shards"
        )
    fingerprints: list[dict[str, object]] = []
    for path in shard_paths:
        stat = path.stat()
        fingerprints.append({
            "path": str(path),
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            # Unlike mtime, ctime cannot be restored with utime after a
            # same-size rewrite. It makes this a safe local hash-cache key:
            # any ordinary content mutation forces the exact rehash below.
            "ctime_ns": int(stat.st_ctime_ns),
        })
    cache_path = Path(identity_cache_path) if identity_cache_path else None
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
        except Exception as exc:
            raise RuntimeError(
                f"streamed model identity cache {cache_path} is corrupt; "
                "refusing identity reuse"
            ) from exc
        if (
            isinstance(cached, dict)
            and cached.get("schema") == STREAMED_MODEL_IDENTITY_CACHE_SCHEMA
            and cached.get("source") == str(source_model)
            and cached.get("fingerprints") == fingerprints
        ):
            identity = validate_streamed_model_identity(
                cached.get("identity"), where="streamed model identity cache"
            )
            if (
                identity.get("config") == canonical_json(
                    config_dict, where="streamed model config"
                )
                and identity.get("weight_map") == mapping
            ):
                return identity

    shards: list[dict[str, object]] = []
    for path in shard_paths:
        stat = path.stat()
        shards.append({
            "path": str(path),
            "size": int(stat.st_size),
            "sha256": _file_sha256(path),
        })
    value_bearing = {
        "config": canonical_json(config_dict, where="streamed model config"),
        "weight_map": mapping,
        "shards": shards,
    }
    identity = {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "source": str(source_model),
        "resolved_commit": getattr(config, "_commit_hash", None),
        "content_sha256": canonical_json_sha256(
            value_bearing, where="streamed model content identity"
        ),
        **value_bearing,
    }
    if cache_path is not None:
        from prismaquant.cost_stage_checkpoint import atomic_write_bytes

        atomic_write_bytes(
            cache_path,
            json.dumps(
                {
                    "schema": STREAMED_MODEL_IDENTITY_CACHE_SCHEMA,
                    "source": str(source_model),
                    "fingerprints": fingerprints,
                    "identity": identity,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
        )
    return identity


def validate_streamed_model_identity(
    identity: object, *, where: str
) -> dict[str, object]:
    """Require a value-bearing full-checkpoint identity, never a name stamp."""
    from collections.abc import Mapping
    from prismaquant.cost_stage_checkpoint import (
        canonical_json,
        canonical_json_sha256,
    )

    if not isinstance(identity, Mapping):
        raise RuntimeError(
            f"{where} requires a full streamed model identity object"
        )
    if identity.get("schema") != STREAMED_MODEL_IDENTITY_SCHEMA:
        raise RuntimeError(
            f"{where} requires model identity schema "
            f"{STREAMED_MODEL_IDENTITY_SCHEMA!r}"
        )
    digest = str(identity.get("content_sha256", "")).lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(
            f"{where} requires exact model content_sha256"
        )
    shards = identity.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError(f"{where} requires source shard content identities")
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise RuntimeError(f"{where} model shard {index} is malformed")
        shard_digest = str(shard.get("sha256", "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", shard_digest) is None:
            raise RuntimeError(
                f"{where} model shard {index} lacks content sha256"
            )
    canonical = canonical_json(identity, where=f"{where} model identity")
    expected = canonical_json_sha256(
        {
            "config": canonical.get("config"),
            "weight_map": canonical.get("weight_map"),
            "shards": canonical.get("shards"),
        },
        where=f"{where} model content identity",
    )
    if digest != expected:
        raise RuntimeError(
            f"{where} model content_sha256 does not match its source shard "
            "identity"
        )
    return canonical
