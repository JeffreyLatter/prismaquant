"""Shared primitives for layer-by-layer streaming of HF model weights
from safetensors.

Used by the unified incremental probe, cost, and export paths. Each
primitive is a pure move from the original monolithic streaming probe —
signatures and behavior are byte-identical. The goal is to share one
install/unload/cache implementation across every stage of the pipeline
so the allocator's view of layer materialization is the same regardless
of which script is driving it.

Notes:
  - The `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` hack must be
    set by the *entrypoint* module before torch.cuda initializes; it is
    deliberately NOT set here so this module can be imported lazily
    after cuda is already up.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import torch
import torch.nn as nn
try:
    from accelerate.utils.modeling import set_module_tensor_to_device
except ModuleNotFoundError:
    def set_module_tensor_to_device(
        module: nn.Module,
        tensor_name: str,
        device,
        *,
        value: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if "." in tensor_name:
            parent_name, attr = tensor_name.rsplit(".", 1)
            parent = module.get_submodule(parent_name)
        else:
            parent, attr = module, tensor_name
        target_device = torch.device(device)
        if attr in parent._parameters:
            old = parent._parameters[attr]
            if value is None:
                if old is None:
                    raise ValueError(f"missing parameter value for {tensor_name}")
                target = torch.empty(
                    tuple(old.shape),
                    dtype=old.dtype,
                    device=target_device,
                )
            else:
                target = value if value.device == target_device else value.to(target_device)
                if dtype is not None and target.is_floating_point():
                    target = target.to(dtype)
            parent._parameters[attr] = nn.Parameter(
                target,
                requires_grad=bool(getattr(old, "requires_grad", False)),
            )
            return
        if attr in parent._buffers:
            old = parent._buffers[attr]
            if value is None:
                if old is None:
                    raise ValueError(f"missing buffer value for {tensor_name}")
                target = torch.empty(
                    tuple(old.shape),
                    dtype=old.dtype,
                    device=target_device,
                )
            else:
                target = value if value.device == target_device else value.to(target_device)
                if dtype is not None and target.is_floating_point():
                    target = target.to(dtype)
            parent._buffers[attr] = target
            return
        raise AttributeError(f"{tensor_name!r} is not a parameter or buffer")
from safetensors import safe_open


# ---------------------------------------------------------------------------
# v21 #5: opt-in direct-to-CUDA safetensors load. Default path opens the
# safetensors file with framework="pt" (CPU mmap) and explicitly moves
# each tensor to device with `.to(device, non_blocking=True)`. That
# allocates a host-side torch.Tensor object even though the underlying
# bytes are mmapped, then issues a host→device cudaMemcpy.
#
# When PRISMAQUANT_DIRECT_CUDA_LOAD=1 is set, we instead pass
# `device=str(device)` to safe_open so safetensors materializes the
# tensor directly on the CUDA device. On UMA hardware (DGX Spark) the
# physical memory is shared, so the win is mostly the elision of the
# extra Python-side host tensor object and one redundant memcpy step;
# expected savings are 10–30 ms per layer load (modest but additive
# across 16 chunks × 2 phases × ~62 layers).
#
# Falls back to the host-stage path on any TypeError / RuntimeError to
# stay compatible with older safetensors releases that do not accept
# the device kwarg.
# ---------------------------------------------------------------------------
def _direct_cuda_enabled() -> bool:
    raw = os.environ.get("PRISMAQUANT_DIRECT_CUDA_LOAD")
    if raw is None:
        return True  # default on as of v26
    return raw not in ("0", "", "false", "False", "FALSE", "no", "NO")


def _safe_open_kwargs(device: torch.device) -> dict:
    if (_direct_cuda_enabled()
            and isinstance(device, torch.device)
            and device.type == "cuda"):
        # Fully qualified device string is what safetensors expects
        # (e.g. "cuda:0" — `str(torch.device("cuda"))` gives "cuda" but
        # we explicitly normalize to "cuda:<index>" so multi-GPU runs
        # don't accidentally land on the wrong card).
        idx = device.index if device.index is not None else 0
        return {"framework": "pt", "device": f"cuda:{idx}"}
    return {"framework": "pt"}


def _build_weight_map(model_path: str, *,
                      multimodal: bool = False
                      ) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({model_key: shard_path}, {model_key: checkpoint_key}).

    Multimodal umbrella checkpoints store tensors under
    `model.language_model.layers.X.*` (and similar visual/audio paths),
    but the staged text-only model has layers at `model.layers.X.*`.
    HF's `from_pretrained` applies a `WeightsMapper` to bridge the two;
    our streaming loader reads safetensors directly, so we apply the
    same rename up front and expose the model-side key to callers
    (while remembering the checkpoint-side key for the safetensors open).

    Also drops keys the text-only probe never needs (visual encoder,
    audio encoder, MTP — those follow their own code paths and would
    shadow real body tensors if they share suffixes).

    When `multimodal=True` the multimodal umbrella arch is used in the
    streaming skeleton (body at `model.language_model.layers.X.*`,
    visual at `model.visual.*`); no rename is applied and visual/audio
    keys are preserved so `_materialize` can load them onto the visual
    tower. MTP stays dropped — MTP has its own synthesis path."""
    # Rename strategy is owned by the model_profile (refactor #32).
    # The default ModelProfile.checkpoint_to_live_name preserves the
    # legacy `_rename_text_only` / `_rename_multimodal` behavior;
    # architecture-specific profiles (e.g. DeepseekV4Profile) override
    # to handle their own naming conventions.
    from .model_profiles import detect_profile
    profile = detect_profile(model_path)

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            raw = json.load(f)["weight_map"]
    else:
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(f"no safetensors under {model_path}")
        with safe_open(single, framework="pt") as f:
            raw = {k: single for k in f.keys()}
    model_to_shard: dict[str, str] = {}
    model_to_ckpt: dict[str, str] = {}
    for ck, shard in raw.items():
        mk = profile.checkpoint_to_live_name(ck, multimodal=multimodal)
        if mk is None:
            continue
        model_to_shard[mk] = os.path.join(model_path, shard)
        model_to_ckpt[mk] = ck
    return model_to_shard, model_to_ckpt


class Fp8ScaleInvMap(dict):
    """`{model_weight_key: (scale_shard_path, scale_ckpt_key)}` plus the
    checkpoint-declared dequant ``block`` size ``(rows, cols)``.

    Behaves as a plain dict everywhere (truthiness, lookups, iteration),
    so every existing caller is unchanged; the block size travels with
    the map so the dequant call sites never have to re-derive it — or
    worse, assume 128x128 for a checkpoint quantized at a different
    granularity.  ``block`` is None only for empty maps."""

    def __init__(self, data=None, block: tuple[int, int] | None = None):
        super().__init__(data or {})
        self.block = block


def _declared_weight_block_size(model_path: str) -> tuple[int, int]:
    """Read `quantization_config.weight_block_size` from the checkpoint
    config and validate it.

    Called only when fp8 block-scaled weights were actually found, so a
    missing/null/malformed declaration is a hard error: the dequant grid
    must be derived from the checkpoint, never assumed (the historical
    hardcoded 128x128 silently mis-scales any checkpoint quantized at a
    different block size, and per-tensor-scale fp8 checkpoints — e.g.
    Mistral-Medium's scalar `weight_scale_inv` — are not block-dequantable
    at all)."""
    cfg_path = os.path.join(model_path, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"checkpoint at {model_path!r} pairs fp8 weights with scale "
            f"tensors but its config.json could not be read ({exc!r}); "
            f"cannot derive the fp8 dequant block size"
        ) from exc
    qc = cfg.get("quantization_config") or {}
    if not qc.get("weight_block_size"):
        # Multimodal umbrellas may nest the quantization config.
        nested = (cfg.get("text_config") or {}).get("quantization_config") or {}
        if nested.get("weight_block_size"):
            qc = nested
    wbs = qc.get("weight_block_size")
    if not wbs:
        raise RuntimeError(
            f"checkpoint at {model_path!r} pairs fp8 weights with scale "
            f"tensors but config.json quantization_config.weight_block_size "
            f"is {'null/empty' if 'weight_block_size' in qc else 'absent'}; "
            f"refusing to assume a 128x128 dequant grid. Block-scaled fp8 "
            f"checkpoints must declare weight_block_size; per-tensor-scale "
            f"fp8 checkpoints are not supported by the block-dequant "
            f"streaming path."
        )
    try:
        pair = tuple(int(v) for v in wbs)
    except (TypeError, ValueError):
        pair = ()
    if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
        raise RuntimeError(
            f"checkpoint at {model_path!r} declares an unsupported "
            f"quantization_config.weight_block_size={wbs!r}; expected two "
            f"positive ints [out_block, in_block]."
        )
    return (pair[0], pair[1])


def _fp8_dequant_block(
    fp8_scale_inv_map: dict[str, tuple[str, str]] | None,
) -> tuple[int, int]:
    """Dequant block size carried by the scale map.

    Every map built by `_build_fp8_scale_inv_map` is an `Fp8ScaleInvMap`
    holding the checkpoint-declared block. Plain dicts (hand-built in
    tests/tools) keep the historical 128x128 default."""
    block = getattr(fp8_scale_inv_map, "block", None)
    if block is not None:
        return (int(block[0]), int(block[1]))
    return (128, 128)


def _build_fp8_scale_inv_map(model_path: str, *,
                             multimodal: bool = False
                             ) -> "Fp8ScaleInvMap":
    """Return `{model_weight_key: (scale_shard_path, scale_ckpt_key)}`
    for every native-FP8 weight tensor (fp8_e4m3fn + paired
    `.weight_scale_inv` fp32 block scale), as an `Fp8ScaleInvMap` whose
    `.block` carries the checkpoint-declared
    `quantization_config.weight_block_size`.

    The key space matches what `_build_weight_map` returns — i.e., the
    live model qname for the weight (`...something.weight`). Callers
    pair it with `_read_layer_to_device(..., fp8_scale_inv_map=...)`
    to apply the declared block dequant inline at load.

    Returns an empty map (block=None) for checkpoints that have no
    `.weight_scale_inv` tensors — load-time dequant is then a no-op and
    callers behave exactly as they did before this function existed.
    A non-empty map with no readable/declared weight_block_size raises
    (see `_declared_weight_block_size`).
    """
    # Profile-driven dispatch (refactor #32). Profiles that store FP8
    # scales under a non-standard path (DSv4 uses `.scale` siblings)
    # return a fully populated map from `fp8_scale_pairs`. Profiles
    # without overrides return None and we fall through to the legacy
    # `.weight_scale_inv`-suffix scan.
    from .model_profiles import detect_profile
    profile = detect_profile(model_path)
    explicit = profile.fp8_scale_pairs(model_path)
    if explicit is not None:
        return Fp8ScaleInvMap(
            explicit,
            _declared_weight_block_size(model_path) if explicit else None,
        )

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            raw = json.load(f)["weight_map"]
    else:
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single):
            return Fp8ScaleInvMap()
        with safe_open(single, framework="pt") as f:
            raw = {k: single for k in f.keys()}

    out: dict[str, tuple[str, str]] = {}
    for ck_key, shard in raw.items():
        if not ck_key.endswith(".weight_scale_inv"):
            continue
        # The legacy MiniMax / DSv3 path: `.weight_scale_inv` siblings
        # paired with `.weight`. Profile.checkpoint_to_live_name
        # always drops `.weight_scale_inv` from the body map; here we
        # rebuild the equivalent live qname by stripping the suffix
        # off the rewritten `.weight` name.
        weight_ck = ck_key[: -len("_scale_inv")]
        weight_live = profile.checkpoint_to_live_name(
            weight_ck, multimodal=multimodal)
        if weight_live is None:
            continue
        out[weight_live] = (os.path.join(model_path, shard), ck_key)
    return Fp8ScaleInvMap(
        out,
        _declared_weight_block_size(model_path) if out else None,
    )


def _check_fp8_scale_grid(
    name: str,
    weight_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    block: tuple[int, int],
) -> None:
    """Hard shape assertion for a block-scale grid vs its weight.

    A transposed `(in_blocks, out_blocks)` grid is numel-compatible with
    the expected `(out_blocks, in_blocks)` reshape, so without this check
    it reshapes silently and mis-scales every block."""
    out_dim, in_dim = int(weight_shape[0]), int(weight_shape[1])
    block_r, block_c = block
    expected = (-(-out_dim // block_r), -(-in_dim // block_c))
    if tuple(scale_shape) != expected:
        raise ValueError(
            f"fp8 weight_scale_inv for {name!r} has shape "
            f"{tuple(scale_shape)}; expected (out_blocks, in_blocks)="
            f"{expected} for weight {tuple(weight_shape)} at block "
            f"{tuple(block)}. A transposed (in_blocks, out_blocks) grid is "
            f"numel-compatible and would reshape silently, mis-scaling "
            f"every block — check the checkpoint's scale layout and its "
            f"declared quantization_config.weight_block_size."
        )


def _dequant_fp8_block_weight(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block: tuple[int, int] = (128, 128),
    name: str = "<weight>",
) -> torch.Tensor:
    """Apply the (ceil(out/block_r), ceil(in/block_c)) block scale to a 2D
    fp8-sourced weight and return bf16. Used by the fp8-aware streaming
    loader for native-FP8 checkpoints (MiniMax-M2/M2.7, DeepSeek-V3).

    `weight` is the fp8-sourced tensor as returned by the streaming
    loader — typically already cast to bf16 (each fp8 code maps
    losslessly to bf16). `scale_inv` is the fp32 `weight_scale_inv`
    block tensor; we cast it to bf16 here, since the fp8 code's 3-bit
    mantissa (≈12.5% precision) dominates the overall error budget —
    bf16's ~0.4% scale precision is well below that.

    Implementation: reshape to 4-D block tiles `(out_blocks, block_r,
    in_blocks, block_c)` and multiply by a broadcasted scale of shape
    `(out_blocks, 1, in_blocks, 1)`. Avoids materializing the full
    (out_dim, in_dim) expanded-scale intermediate — on MiniMax-M2.7's
    772 fp8 weights per layer this cuts allocation pressure ~25× vs a
    `repeat_interleave(128)`-pair expansion, and keeps everything in
    bf16 so no fp32 intermediate is ever allocated.
    """
    out_dim, in_dim = weight.shape
    block_r, block_c = block
    _check_fp8_scale_grid(
        name, tuple(weight.shape), tuple(scale_inv.shape), block)
    target_dtype = torch.bfloat16
    target_device = (scale_inv.device if scale_inv.device.type != "cpu"
                     else weight.device)
    if out_dim % block_r != 0 or in_dim % block_c != 0:
        # Unaligned tail: fall back to the expanded-scale path. Shouldn't
        # hit in practice on MiniMax/DeepSeek — both ship weights that
        # are exact multiples of 128 along both dims.
        scale = scale_inv.to(device=target_device, dtype=target_dtype)
        expanded = scale.repeat_interleave(block_r, dim=0)[:out_dim]
        expanded = expanded.repeat_interleave(block_c, dim=1)[:, :in_dim]
        return (weight.to(device=target_device, dtype=target_dtype)
                * expanded)
    out_blocks = out_dim // block_r
    in_blocks = in_dim // block_c
    scale_bf16 = scale_inv.to(device=target_device, dtype=target_dtype)
    scale_view = scale_bf16.reshape(out_blocks, 1, in_blocks, 1)
    w_bf16 = weight.to(device=target_device, dtype=target_dtype)
    w4 = w_bf16.reshape(out_blocks, block_r, in_blocks, block_c)
    return (w4 * scale_view).reshape(out_dim, in_dim)


def _is_fp8_scaled_tensor(
    model_name: str,
    fp8_scale_inv_map: dict[str, tuple[str, str]] | None,
) -> bool:
    return fp8_scale_inv_map is not None and model_name in fp8_scale_inv_map


_FLOAT8_DTYPES = frozenset(
    dt for dt in (
        getattr(torch, name, None)
        for name in (
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
            "float8_e8m0fnu",
        )
    ) if dt is not None
)


def _allow_unscaled_fp8() -> bool:
    raw = os.environ.get("PRISMAQUANT_ALLOW_UNSCALED_FP8")
    return raw not in (None, "", "0", "false", "False", "FALSE", "no", "NO")


def _require_fp8_scale(
    model_name: str,
    t: torch.Tensor,
    fp8_scale_inv_map: dict[str, tuple[str, str]] | None,
) -> None:
    """Fail fast on a float8 source tensor with no dequant scale mapping.

    Casting raw fp8 codes to bf16 installs values in the fp8 *code*
    range (±448 for e4m3) instead of true dequanted weights — the
    historical fp8-range bug that silently poisoned probe/cost passes on
    native-FP8 checkpoints. An unmapped fp8 tensor means the scale-map
    scan missed it, so raise instead of guessing."""
    if t.dtype not in _FLOAT8_DTYPES:
        return
    if _is_fp8_scaled_tensor(model_name, fp8_scale_inv_map):
        return
    if _allow_unscaled_fp8():
        return
    raise RuntimeError(
        f"native-FP8 tensor {model_name!r} (dtype {t.dtype}) has no entry "
        f"in fp8_scale_inv_map — casting raw fp8 codes to bf16 would "
        f"install values in the code range (±448) instead of true "
        f"dequanted weights (the historical fp8-range bug). The scale map "
        f"is built by _build_fp8_scale_inv_map from `.weight_scale_inv` "
        f"siblings (or the model profile's fp8_scale_pairs override, e.g. "
        f"DSv4's `.scale` siblings); check the checkpoint's scale tensor "
        f"naming against that scan. Set PRISMAQUANT_ALLOW_UNSCALED_FP8=1 "
        f"only if this tensor is genuinely scale-free."
    )


def _apply_fp8_dequant_inplace(
    out: dict[str, torch.Tensor],
    fp8_scale_inv_map: dict[str, tuple[str, str]],
    device: torch.device,
) -> int:
    """For each tensor in `out` whose key matches a `fp8_scale_inv_map`
    entry, read the scale_inv, apply the checkpoint-declared block
    dequant (`fp8_scale_inv_map.block`, see `_fp8_dequant_block`), and
    replace the loaded tensor with the dequanted bf16 weight.

    Tensors and scales are both grouped by shape and multiplied in a
    single batched 5-D broadcast op per shape-group. On MiniMax-M2.7
    that collapses 772 per-tensor kernel launches per layer-load
    (256 experts × 3 projs + 4 attention projs) into ~4-5 shape
    buckets — which is what the prefetch thread's CPU cost scales on,
    since each kernel launch has fixed Python + CUDA dispatch overhead.
    """
    if not fp8_scale_inv_map:
        return 0
    # Collect (name, scale_key) per shard so we can open each shard
    # once and pre-read every scale we need before batching.
    scale_reads: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for model_name in list(out.keys()):
        entry = fp8_scale_inv_map.get(model_name)
        if entry is None:
            continue
        shard, scale_key = entry
        scale_reads[shard].append((model_name, scale_key))
    if not scale_reads:
        return 0

    # Step 1: Read all scales from source safetensors once per shard.
    loaded_scales: dict[str, torch.Tensor] = {}  # name -> fp32 scale (cpu)
    for shard, reads in scale_reads.items():
        with safe_open(shard, framework="pt") as f:
            for model_name, scale_key in reads:
                loaded_scales[model_name] = f.get_tensor(scale_key)

    # Step 2: Group matched weights by (out_dim, in_dim) shape. We only
    # batch along exact-block-multiple shapes; odd-shaped tensors (rare)
    # fall back to the per-tensor path. The block size is the
    # checkpoint-declared quantization_config.weight_block_size carried
    # on the map, never assumed.
    block_r, block_c = _fp8_dequant_block(fp8_scale_inv_map)
    by_shape: dict[tuple[int, int], list[str]] = defaultdict(list)
    fallback: list[str] = []
    for name in loaded_scales:
        w = out[name]
        if w.dim() != 2:
            fallback.append(name)
            continue
        out_dim, in_dim = w.shape
        # Hard shape assertion (audit §3.7a): a transposed scale grid is
        # numel-compatible with the batched reshape below and would
        # silently mis-scale every block.
        _check_fp8_scale_grid(
            name, tuple(w.shape), tuple(loaded_scales[name].shape),
            (block_r, block_c))
        if out_dim % block_r != 0 or in_dim % block_c != 0:
            fallback.append(name)
            continue
        by_shape[(out_dim, in_dim)].append(name)

    dequanted = 0

    # Step 3: Batched multiply per shape-group. Stack all weights of
    # the same shape along a new outer dim, stack their scales the
    # same way, reshape both into block-tile form, one bf16 multiply,
    # split back.
    for (out_dim, in_dim), names in by_shape.items():
        out_blocks = out_dim // block_r
        in_blocks = in_dim // block_c
        E = len(names)
        # Stack weights: (E, out, in) bf16 on the execution device.
        # Native FP8 source tensors stay compressed until this point so
        # CPU-side reads and H2D/UMA traffic remain 1 byte/element.
        w_stack = torch.stack([out[n] for n in names], dim=0).to(
            device=device, dtype=torch.bfloat16)
        # Stack scales: (E, out_blocks, in_blocks) bf16 on device
        s_stack = torch.stack(
            [loaded_scales[n] for n in names], dim=0
        ).to(device=device, dtype=torch.bfloat16)
        # Reshape to block-tile form:
        #   w: (E, out_blocks, block_r, in_blocks, block_c)
        #   s: (E, out_blocks, 1, in_blocks, 1)
        w4 = w_stack.reshape(E, out_blocks, block_r, in_blocks, block_c)
        s4 = s_stack.reshape(E, out_blocks, 1, in_blocks, 1)
        dequanted_stack = (w4 * s4).reshape(E, out_dim, in_dim)
        # Split back — zero-copy via unbind.
        for i, n in enumerate(names):
            out[n] = dequanted_stack[i].contiguous()
        dequanted += E
        del w_stack, s_stack, w4, s4, dequanted_stack

    # Step 4: Fallback path for any shapes we didn't batch.
    for name in fallback:
        w = out[name]
        scale_fp = loaded_scales[name].to(device=device)
        out[name] = _dequant_fp8_block_weight(
            w, scale_fp, block=(block_r, block_c), name=name)
        dequanted += 1

    return dequanted


def _materialize(model: nn.Module, prefixes: list[str],
                 model_to_shard: dict[str, str],
                 model_to_ckpt: dict[str, str],
                 device: torch.device, dtype: torch.dtype,
                 fp8_scale_inv_map: dict[str, tuple[str, str]] | None = None,
                 ) -> int:
    """Load all tensors whose model-side name starts with any prefix in
    `prefixes` onto `device` as `dtype`. Uses the checkpoint-side key to
    read from safetensors but assigns to the model-side name.

    When `fp8_scale_inv_map` is provided, fp8-sourced weights get their
    128x128 block `weight_scale_inv` applied inline so the installed
    parameter holds the true dequanted weight, not the raw fp8 codes
    cast to bf16. See `_dequant_fp8_block_weight`.

    Returns count of tensors loaded."""
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for model_name, shard in model_to_shard.items():
        if any(model_name.startswith(p) for p in prefixes):
            by_shard[shard].append((model_name, model_to_ckpt[model_name]))
    # Collect loaded tensors first so we can batch the scale-read pass.
    out: dict[str, torch.Tensor] = {}
    open_kwargs = _safe_open_kwargs(device)
    for shard, pairs in by_shard.items():
        try:
            f_ctx = safe_open(shard, **open_kwargs)
        except (TypeError, RuntimeError):
            # Older safetensors / unsupported device combos: drop the
            # device kwarg and fall back to the host-stage path.
            f_ctx = safe_open(shard, framework="pt")
        with f_ctx as f:
            for model_name, ckpt_name in pairs:
                t = f.get_tensor(ckpt_name)
                _require_fp8_scale(model_name, t, fp8_scale_inv_map)
                if (t.is_floating_point()
                        and not _is_fp8_scaled_tensor(
                            model_name, fp8_scale_inv_map)):
                    t = t.to(dtype)
                out[model_name] = t
    if fp8_scale_inv_map:
        _apply_fp8_dequant_inplace(out, fp8_scale_inv_map, device)
    loaded = 0
    for model_name, t in out.items():
        install_dtype = t.dtype if t.is_floating_point() else None
        set_module_tensor_to_device(
            model, model_name, device, value=t, dtype=install_dtype)
        loaded += 1
    return loaded


def _pack_per_expert_into_packed(
    out: dict[str, torch.Tensor],
    *,
    per_expert_re: "re.Pattern",
    parent_for_projection,
    projection_names_for,
    live_param_shape,
) -> int:
    """Stack per-expert checkpoint tensors into packed 3D live params.

    Some MoE checkpoints store each routed expert's projections separately
    on disk (``…experts.{i}.{proj}.weight``) while the live module exposes a
    single packed parameter per projection group (``…experts.gate_up_proj``,
    a ``[num_experts, …]`` tensor). The install resolver is keyed by the
    *live* parameter names, so the per-expert disk tensors never match and
    the slow fallback walks a non-existent ``experts.{i}`` submodule.

    This bridges the two layouts generically: every structural decision —
    which projections fuse into which packed param, and in what order —
    comes from the supplied callables, which the caller wires from the model
    profile's packed-experts spec. No architecture names appear here. The
    assembled tensor's shape is checked against the live parameter so a
    layout mismatch fails loud instead of silently mis-packing.

    Mutates ``out`` in place: removes consumed per-expert keys and inserts
    the packed keys. Returns the number of packed params produced (0 = the
    checkpoint isn't per-expert, or the live module isn't packed)."""
    # packed_full_name -> {expert_idx -> {projection -> tensor}}
    groups: dict[str, dict[int, dict[str, torch.Tensor]]] = defaultdict(
        lambda: defaultdict(dict))
    consumed: list[str] = []
    for key, t in out.items():
        name = key[:-len(".weight")] if key.endswith(".weight") else key
        if not per_expert_re.match(name):
            continue
        head, proj = name.rsplit(".", 1)           # head = …experts.{idx}
        experts_path, idx_str = head.rsplit(".", 1)
        if not idx_str.isdigit():
            continue
        parent = parent_for_projection(proj)
        if parent is None:
            continue
        packed_full = f"{experts_path}.{parent}"
        if live_param_shape(packed_full) is None:
            continue  # live module isn't packed for this group — leave as-is
        groups[packed_full][int(idx_str)][proj] = t
        consumed.append(key)
    produced = 0
    for packed_full, by_expert in groups.items():
        parent = packed_full.rsplit(".", 1)[1]
        order = tuple(projection_names_for(parent))
        n_experts = max(by_expert) + 1
        slabs: list[torch.Tensor] = []
        for i in range(n_experts):
            projs = by_expert.get(i)
            if projs is None or any(p not in projs for p in order):
                raise ValueError(
                    f"per-expert pack: {packed_full} missing expert {i} "
                    f"projection(s) {order}")
            if len(order) == 1:
                slabs.append(projs[order[0]])
            else:
                # Fuse projections along the output axis (the transformers
                # packed-FusedMoE convention), then stack experts on a new
                # leading axis. The shape check below is the safety net.
                slabs.append(torch.cat([projs[p] for p in order], dim=0))
        packed = torch.stack(slabs, dim=0).contiguous()
        target = live_param_shape(packed_full)
        if tuple(packed.shape) != tuple(target):
            raise ValueError(
                f"per-expert pack: assembled {packed_full} shape "
                f"{tuple(packed.shape)} != live param {tuple(target)}")
        out[packed_full] = packed
        produced += 1
    for key in consumed:
        out.pop(key, None)
    return produced


def _build_expert_packer(model: nn.Module, weight_ckpt: dict[str, str]):
    """Return a callable that packs per-expert checkpoint tensors into the
    live module's packed 3D params, or None when not needed.

    Returns None (loader unchanged) unless ALL of:
      * the model profile declares packed-expert params + a per-expert regex,
      * the checkpoint actually stores experts per-expert on disk, and
      * the live module exposes the packed params (so there is a layout gap
        to bridge — a per-expert *live* layout needs no packing).

    Everything model-specific comes from the profile spec; the returned
    closure carries no architecture names. Used by both the streaming
    probe/cost context and the compressed-tensors exporter so a raw
    per-expert checkpoint loads identically on every path — no out-of-band
    pre-pack."""
    try:
        from .model_profiles import profile_from_model
        prof = profile_from_model(model)
    except Exception:
        return None
    packed_names = prof.packed_expert_param_names()
    regex = prof.per_expert_moe_regex()
    if not packed_names or not regex:
        return None
    pat = re.compile(regex[len("re:"):] if regex.startswith("re:") else regex)

    def _is_per_expert(k: str) -> bool:
        name = k[:-len(".weight")] if k.endswith(".weight") else k
        return bool(pat.match(name))

    if not any(_is_per_expert(k) for k in weight_ckpt):
        return None  # checkpoint already packed — nothing to do
    live_shapes = {
        n: tuple(p.shape) for n, p in model.named_parameters()
        if n.rsplit(".", 1)[-1] in packed_names
    }
    if not live_shapes:
        return None  # live module is per-expert too — no gap to bridge

    def _packer(out):
        _pack_per_expert_into_packed(
            out,
            per_expert_re=pat,
            parent_for_projection=prof.packed_expert_parent_for_projection,
            projection_names_for=prof.packed_expert_projection_names,
            live_param_shape=live_shapes.get,
        )

    return _packer


def _read_layer_to_device(prefix: str,
                          model_to_shard: dict[str, str],
                          model_to_ckpt: dict[str, str],
                          dtype: torch.dtype,
                          device: torch.device,
                          fp8_scale_inv_map: dict[str, tuple[str, str]]
                              | None = None,
                          pack_experts=None,
                          ) -> dict[str, torch.Tensor]:
    """Read all tensors under `prefix` from safetensors and place them
    on `device`. Returns {model_name: device_tensor}.

    When `fp8_scale_inv_map` is provided, native-FP8 block-scaled
    weights are kept compressed through the host-side read and moved to
    `device` before the 128x128 block dequant. That avoids CPU-side
    FP8→BF16 expansion and cuts the transfer/cache traffic for those
    tensors to the source checkpoint size until the final GPU multiply."""
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for model_name, shard in model_to_shard.items():
        if model_name.startswith(prefix):
            by_shard[shard].append((model_name, model_to_ckpt[model_name]))
    out: dict[str, torch.Tensor] = {}
    open_kwargs = _safe_open_kwargs(device)
    direct = "device" in open_kwargs
    for shard, pairs in by_shard.items():
        try:
            f_ctx = safe_open(shard, **open_kwargs)
            used_direct = direct
        except (TypeError, RuntimeError):
            f_ctx = safe_open(shard, framework="pt")
            used_direct = False
        with f_ctx as f:
            for model_name, ckpt_name in pairs:
                t = f.get_tensor(ckpt_name)
                _require_fp8_scale(model_name, t, fp8_scale_inv_map)
                if (t.is_floating_point()
                        and not _is_fp8_scaled_tensor(
                            model_name, fp8_scale_inv_map)):
                    t = t.to(dtype)
                if not used_direct:
                    t = t.to(device, non_blocking=True)
                if not t.is_contiguous():
                    t = t.contiguous()
                out[model_name] = t
    if fp8_scale_inv_map:
        _apply_fp8_dequant_inplace(out, fp8_scale_inv_map, device)
    if pack_experts is not None:
        # Generic per-expert -> packed-3D bridge for checkpoints that ship
        # MoE experts unfused while the live module is packed. No-op (None)
        # for every other checkpoint/model. Driven by the model profile.
        pack_experts(out)
    return out


# Back-compat alias. Historically callers treated this as a "CPU cache"
# reader, but the implementation has always returned tensors resident on
# the requested execution device.
_read_layer_to_cpu = _read_layer_to_device


def _install_cached_tensors(model: nn.Module,
                            cached_tensors: dict[str, torch.Tensor],
                            device: torch.device):
    """Install cached layer tensors into the model on `device`."""
    for model_name, t in cached_tensors.items():
        install_dtype = t.dtype if t.is_floating_point() else None
        set_module_tensor_to_device(
            model, model_name, device, value=t, dtype=install_dtype)


def _build_install_resolver(model: nn.Module,
                            layer_qname: str) -> dict[str, tuple]:
    """Pre-compute `(parent_module, attr, is_buffer)` for every
    Parameter / buffer under `layer_qname`. Letting us bypass
    accelerate's `set_module_tensor_to_device` at install time — 10×
    fewer Python frames per tensor, 90 s saved per phase at batch=32.

    The resolver maps full dotted names (e.g.
    `model.layers.3.linear_attn.in_proj_qkv.weight`) to the direct
    `nn.Module` + attribute that owns the storage slot."""
    layer_mod = model.get_submodule(layer_qname)
    resolver: dict[str, tuple] = {}
    for sub_name, param in layer_mod.named_parameters():
        full = f"{layer_qname}.{sub_name}"
        if "." in sub_name:
            parent_path, attr = sub_name.rsplit(".", 1)
            parent = layer_mod.get_submodule(parent_path)
        else:
            parent, attr = layer_mod, sub_name
        resolver[full] = (parent, attr, False)
    for sub_name, buf in layer_mod.named_buffers():
        # Skip non-persistent buffers (rotary inv_freq caches, attention
        # masks) — those aren't in our safetensors weight map anyway.
        if "." in sub_name:
            parent_path, attr = sub_name.rsplit(".", 1)
            parent = layer_mod.get_submodule(parent_path)
        else:
            parent, attr = layer_mod, sub_name
        if attr in getattr(parent, "_non_persistent_buffers_set", set()):
            continue
        full = f"{layer_qname}.{sub_name}"
        resolver[full] = (parent, attr, True)
    return resolver


def _fast_install(resolver: dict[str, tuple],
                  cached_tensors: dict[str, torch.Tensor],
                  device: torch.device,
                  model: nn.Module | None = None):
    """Direct install via `resolver` (built by `_build_install_resolver`).
    Swaps `Parameter.data` in place when the existing storage matches
    shape and isn't meta — otherwise allocates a fresh Parameter. On
    unified-memory systems `t.to(device)` for a same-device tensor is
    essentially a pointer rebind, so the hot path is a single attribute
    write per tensor."""
    import torch.nn as _nn
    for model_name, t in cached_tensors.items():
        slot = resolver.get(model_name)
        if slot is None:
            # Unknown key — fall back to the safe-but-slow path. Shouldn't
            # happen in practice; if we see it, the resolver-build logic
            # missed a branch of the module tree.
            if model is not None:
                install_dtype = t.dtype if t.is_floating_point() else None
                set_module_tensor_to_device(
                    model, model_name, device, value=t, dtype=install_dtype)
            continue
        parent, attr, is_buffer = slot
        target = t if t.device == device else t.to(device, non_blocking=True)
        if is_buffer:
            parent._buffers[attr] = target
            continue
        existing = parent._parameters.get(attr)
        if (existing is not None
                and not existing.is_meta
                and existing.shape == target.shape
                and existing.dtype == target.dtype):
            existing.data = target
        else:
            parent._parameters[attr] = _nn.Parameter(
                target, requires_grad=False)


def _unload(model: nn.Module, prefixes: list[str]) -> int:
    """Move all params/buffers under `prefixes` back to meta."""
    n = 0
    for name, _ in list(model.named_parameters()):
        if any(name.startswith(p) for p in prefixes):
            set_module_tensor_to_device(model, name, "meta")
            n += 1
    for name, _ in list(model.named_buffers()):
        if any(name.startswith(p) for p in prefixes):
            set_module_tensor_to_device(model, name, "meta")
            n += 1
    return n


class LayerCache:
    """LRU cache of decoded layer tensors keyed by layer index.

    Values are dicts `{model_name: tensor}` returned by the layer-read
    helper. In the current streaming path those tensors live on the
    execution device, not on a detached CPU-only cache. Cache size is
    bounded by bytes, not entries, so the same path degenerates to
    "keep everything resident" when enough memory is available.
    Eviction is LRU, which matches the forward-then-reverse access
    pattern used by the streaming probe.
    """

    def __init__(self, max_bytes: int):
        from collections import OrderedDict as _OD
        self._cache: "_OD[int, dict[str, torch.Tensor]]" = _OD()
        self._bytes: dict[int, int] = {}
        self.max_bytes = max_bytes
        self.total_bytes = 0
        self.hits = 0
        self.misses = 0
        # In-scope priority set (Task #4): layers in this set are protected
        # from LRU eviction when possible. The body shard runner registers
        # its tracked layers here so they stay hot through the reverse
        # sweep — they fire 768 hooks each and should not be evicted by
        # out-of-scope prefetches.
        self._priority_layers: set[int] = set()
        # Pressure-trigger threshold (Task #3): when MemAvailable falls
        # below this many bytes, an eviction call drops enough entries to
        # recover that floor. 0 disables. Set via configure_pressure_threshold().
        self._pressure_threshold_bytes: int = 0
        self.pressure_evictions = 0
        # Mark-done set (v20 step 2): layers the scheduler has declared
        # provably won't be requested again this chunk. put() refuses
        # to repopulate, get()/peek() see them as not-cached. Cleared
        # at chunk boundaries by clear_done() (called from
        # StreamingContext.reset_between_chunks).
        #
        # Within a single phase-3 sweep, NO body layer can be marked done
        # — every shard re-traverses all layers for backward gradient
        # propagation. The valid call sites are:
        #   1. After phase-1 completes, for layers not in any phase-3 shard.
        #   2. After phase-3 completes, for ALL body layers (only non-body
        #      shards remain, e.g. visual/lm_head, which don't load body
        #      layer tensors).
        #   3. Implicit at chunk teardown via clear_done().
        self._done_layers: set[int] = set()
        self.refused_puts = 0
        # Dynamic budget reserve (v20 step 3+4): when > 0, put()
        # recomputes the effective max as
        #   min(max_bytes, MemAvailable + total_bytes - reserve)
        # so the cache shrinks under host-memory pressure and grows
        # back up to max_bytes when other processes free RAM.
        # 0 = static max_bytes only (default).
        self._dynamic_reserve_bytes: int = 0

    def _sizeof(self, tensors: dict[str, torch.Tensor]) -> int:
        return sum(t.numel() * t.element_size() for t in tensors.values())

    def _residency(self, tensors: dict[str, torch.Tensor]) -> str:
        devices = {str(t.device) for t in tensors.values()}
        if not devices:
            return "empty"
        if len(devices) == 1:
            return next(iter(devices))
        return "mixed"

    def get(self, layer_idx: int):
        if layer_idx in self._cache:
            self._cache.move_to_end(layer_idx)
            self.hits += 1
            return self._cache[layer_idx]
        self.misses += 1
        return None

    def peek(self, layer_idx: int) -> bool:
        """Non-LRU-touching existence check — used by the prefetch
        scheduler so checking doesn't reshuffle eviction order."""
        return layer_idx in self._cache

    def put(self, layer_idx: int, tensors: dict[str, torch.Tensor],
            force: bool = True) -> bool:
        """Insert tensors into the cache. Returns True on success.

        force=True (default): always insert, even if the layer is
            larger than effective_max — required for cold ensure_loaded
            paths where the consumer needs the layer regardless of
            budget. After evict-all, the cache may be over budget for
            this one entry until the next put() naturally evicts it.

        force=False: refuse the insert if size > effective_max after
            evict-all. Used by the prefetch worker — a layer that
            won't fit shouldn't displace cache state speculatively.
            v20 fix #5.
        """
        # v20 step 2: refuse to populate a layer the scheduler said is
        # done. This catches stale in-flight prefetches and any policy
        # bug where a "won't be reused" claim turned out to be wrong —
        # the silent no-op + counter makes the bug visible without
        # crashing the run.
        if layer_idx in self._done_layers:
            self.refused_puts += 1
            return False
        if layer_idx in self._cache:
            return False
        size = self._sizeof(tensors)
        # Pressure-shrink check (Task #3): if MemAvailable is below the
        # configured floor, drop entries until the projected available
        # memory recovers that floor (or candidates are exhausted).
        self._maybe_pressure_shrink()
        # v20 step 3+4: dynamic budget — recompute effective_max from
        # current MemAvailable so the cache shrinks under load and
        # grows back when other processes free RAM. Bounded by static
        # max_bytes. When _dynamic_reserve_bytes is 0 (default), this
        # reduces to the static budget.
        effective_max = self._effective_max()
        # v20 fix #5: refuse over-budget prefetch BEFORE eviction.
        # If size > effective_max, no eviction makes room — and we
        # don't want to evict valuable entries just to refuse the new
        # one. Cold path (force=True) skips this check.
        if not force and size > effective_max:
            self.refused_puts += 1
            return False
        evicted = False
        # In-scope priority eviction (Task #4): when full, prefer evicting
        # out-of-scope (non-priority) entries before in-scope ones. Falls
        # back to LRU order if all candidates are in-scope.
        while (self.total_bytes + size > effective_max
               and len(self._cache) > 0):
            evict_idx = self._pick_evict_candidate()
            if evict_idx is None:
                break  # only priority entries left, can't evict any
            self._cache.pop(evict_idx, None)
            self.total_bytes -= self._bytes.pop(evict_idx, 0)
            evicted = True
        self._cache[layer_idx] = tensors
        self._bytes[layer_idx] = size
        self.total_bytes += size
        # On UMA the cuda caching allocator won't return freed blocks to
        # the OS on its own, so every eviction would otherwise leak into
        # the shared LPDDR5X pool. Force a release after each eviction.
        if evicted and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def _pick_evict_candidate(self) -> int | None:
        """Return the layer_idx of the LRU non-priority entry, or the
        LRU priority entry if no non-priority ones exist, or None if
        the cache is empty."""
        if not self._cache:
            return None
        # OrderedDict iteration is in insertion order; LRU is at front.
        for idx in self._cache:
            if idx not in self._priority_layers:
                return idx
        # All entries are priority — fall back to LRU
        return next(iter(self._cache))

    def _maybe_pressure_shrink(self) -> int:
        """If system MemAvailable is below the configured threshold,
        drop entries until projected MemAvailable recovers that
        headroom. Two-phase: non-priority first, then priority if
        still under pressure. v20 fix #4-B: previously returned early
        when only priority entries remained (e.g., unified-sweep
        marks every body layer priority), so pressure shrink was a
        no-op when it mattered most."""
        if self._pressure_threshold_bytes <= 0 or not self._cache:
            return 0
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except Exception:
            return 0
        if avail >= self._pressure_threshold_bytes:
            return 0
        needed = max(0, self._pressure_threshold_bytes - avail)
        freed = 0

        # Phase 1: non-priority LRU, no priority fallback.
        for idx in list(self._cache.keys()):
            if freed >= needed:
                break
            if idx in self._priority_layers:
                continue
            size = self._bytes.get(idx, 0)
            self._cache.pop(idx, None)
            self.total_bytes -= self._bytes.pop(idx, 0)
            freed += size
            self.pressure_evictions += 1

        # Phase 2: re-check pressure; if still tight, drop priority
        # entries in LRU order. Priority is a preference, not a hard
        # contract — when host memory is genuinely scarce, holding
        # cached weights is worse than re-loading them.
        if freed < needed:
            for idx in list(self._cache.keys()):
                if freed >= needed:
                    break
                size = self._bytes.get(idx, 0)
                self._cache.pop(idx, None)
                self.total_bytes -= self._bytes.pop(idx, 0)
                freed += size
                self.pressure_evictions += 1

        if freed and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Unified-memory systems can keep an evicted cache tensor alive
        # through another owner (for example, an installed model layer).
        # Re-check real MemAvailable after empty_cache and keep trimming
        # if the projected release did not materialize.
        if freed and self._cache:
            try:
                avail = psutil.virtual_memory().available
            except Exception:
                avail = self._pressure_threshold_bytes
            if avail < self._pressure_threshold_bytes:
                needed = self._pressure_threshold_bytes - avail
                extra_freed = 0
                while extra_freed < needed and self._cache:
                    evict_idx = self._pick_evict_candidate()
                    if evict_idx is None:
                        break
                    size = self._bytes.get(evict_idx, 0)
                    self._cache.pop(evict_idx, None)
                    self.total_bytes -= self._bytes.pop(evict_idx, 0)
                    extra_freed += size
                    self.pressure_evictions += 1
                if extra_freed and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                freed += extra_freed
        return freed

    def trim_for_memory_pressure(self) -> int:
        """Public pressure-trim hook for callers that just released large
        transient tensors. Returns the projected cache bytes evicted."""
        return self._maybe_pressure_shrink()

    def configure_pressure_threshold(self, available_bytes_floor: int):
        """Set the MemAvailable byte threshold below which the cache
        triggers proactive eviction. 0 disables (default)."""
        self._pressure_threshold_bytes = int(available_bytes_floor)

    def configure_dynamic_budget(self, reserve_bytes: int):
        """Enable dynamic cap: effective_max = min(max_bytes,
        MemAvailable + total_bytes - reserve_bytes). Each put()
        re-evaluates against current MemAvailable, so the cache
        breathes with host memory pressure. 0 disables (static max)."""
        self._dynamic_reserve_bytes = int(reserve_bytes)

    @property
    def dynamic_reserve_bytes(self) -> int:
        return int(self._dynamic_reserve_bytes)

    def prepare_for_load(self, size_hint: int) -> int:
        """Pre-evict entries to make room for an incoming layer load
        of approximately size_hint bytes. v20 fix #1: the dynamic
        budget cap in put() runs AFTER the load lands in memory,
        which on UMA can push the system into OOM during the load
        itself. Calling this before _read_layer_to_device gives the
        cache a chance to free bytes first.

        Eviction order: LRU non-priority first, then LRU priority.
        Returns the number of bytes actually freed (caller can decide
        whether to skip the load if 0 was freed and we're tight)."""
        self._maybe_pressure_shrink()
        effective_max = self._effective_max()
        target_total = max(0, effective_max - max(0, size_hint))
        freed = 0
        while self.total_bytes > target_total and self._cache:
            evict_idx = self._pick_evict_candidate()
            if evict_idx is None:
                break
            size = self._bytes.get(evict_idx, 0)
            self._cache.pop(evict_idx, None)
            self.total_bytes -= self._bytes.pop(evict_idx, 0)
            freed += size
        if freed and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return freed

    def _effective_max(self) -> int:
        """Compute the byte cap that put() should honor for this call.

        With dynamic budget disabled (default), returns static max_bytes.
        With dynamic budget on, caps such that completing this put will
        leave the system with at least reserve_bytes of MemAvailable.
        Falls back to static max if psutil is unavailable."""
        if self._dynamic_reserve_bytes <= 0:
            return self.max_bytes
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except Exception:
            return self.max_bytes
        # If we evicted everything, MemAvailable would rise by total_bytes.
        # We want post-eviction-and-put: avail_after >= reserve.
        # avail_after = avail + (current_total - new_cache_bytes), so
        # new_cache_bytes <= avail + current_total - reserve.
        room = avail + self.total_bytes - self._dynamic_reserve_bytes
        return max(0, min(self.max_bytes, room))

    def set_priority_layers(self, layers: "set[int] | list[int]"):
        """Mark these layer indices as in-scope/priority. They are
        protected from LRU eviction when possible (other entries are
        evicted first). Pass an empty set to clear."""
        self._priority_layers = set(layers)

    def discard(self, layer_idx: int):
        """Drop cache ownership for a layer that has been installed.

        Installed parameters/buffers keep tensor references alive until
        the model layer is unloaded. Removing the cache reference here
        prevents one-pass streaming from treating the just-consumed
        layer as MRU and evicting the next layer that prefetch prepared.
        """
        tensors = self._cache.pop(layer_idx, None)
        if tensors is None:
            return
        self.total_bytes -= self._bytes.pop(layer_idx, 0)

    def mark_done(self, layer_idx: int):
        """Declare layer_idx provably won't be requested again. Evicts
        any cached entry and refuses future put() until clear_done().

        See _done_layers comment for valid call sites."""
        self.discard(layer_idx)
        self._done_layers.add(layer_idx)

    def mark_layers_done(self, layer_indices) -> int:
        """Bulk mark_done. Returns the count actually transitioned (i.e.
        excluding layers that were already in the done set)."""
        before = len(self._done_layers)
        for L in layer_indices:
            self.discard(L)
            self._done_layers.add(L)
        return len(self._done_layers) - before

    def clear_done(self):
        """Reset the mark-done set. Called at chunk boundaries by
        StreamingContext.reset_between_chunks() so the next chunk's
        loads work normally."""
        self._done_layers.clear()
        self.refused_puts = 0

    def clear(self):
        self._cache.clear()
        self._bytes.clear()
        self.total_bytes = 0

    def residency_summary(self) -> str:
        counts: dict[str, int] = defaultdict(int)
        for tensors in self._cache.values():
            counts[self._residency(tensors)] += 1
        if not counts:
            return "empty"
        return ",".join(f"{k}:{counts[k]}" for k in sorted(counts))

    def summary(self) -> str:
        tot = self.hits + self.misses
        return (f"LayerCache: {len(self._cache)} layers, "
                f"{self.total_bytes / (1024**3):.1f} GB / "
                f"{self.max_bytes / (1024**3):.1f} GB, "
                f"residency={self.residency_summary()} "
                f"hits={self.hits} misses={self.misses} "
                f"hit_rate={(self.hits/tot*100 if tot else 0):.0f}%")


def _get_layer_list(model: nn.Module):
    """Return (base_model, layer_list) for a causal-LM model. Walks
    past any `ForConditionalGeneration` / `ForCausalLM` wrapper to
    find the decoder layers."""
    # Typical layouts:
    #   model.model.layers                              — text-only CausalLM
    #   model.language_model.model.layers               — pre-v5 multimodal
    #   model.model.language_model.layers               — v5 multimodal umbrella
    #                                                     (Qwen3_5MoeForConditionalGeneration)
    cand = getattr(model, "model", None)
    if cand is not None and hasattr(cand, "layers"):
        return cand, cand.layers
    # v5 multimodal: model.model wraps .visual + .language_model
    if cand is not None:
        lm = getattr(cand, "language_model", None)
        if lm is not None and hasattr(lm, "layers"):
            return lm, lm.layers
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", lm)
        if hasattr(inner, "layers"):
            return inner, inner.layers
    raise RuntimeError("could not locate model.layers in model tree")


def _get_rotary(base_model: nn.Module) -> nn.Module | None:
    """Find the rotary embedding module so we can compute
    position_embeddings once per sample."""
    for attr in ("rotary_emb", "rope", "rotary_embedding", "pos_emb"):
        r = getattr(base_model, attr, None)
        if r is not None:
            return r
    return None


def _get_final_norm(base_model: nn.Module) -> nn.Module | None:
    """Find the final pre-lm_head norm by trying the attribute names used
    across HF architectures, in priority order: ``norm`` (Llama/Qwen/most),
    then ``embedding_norm``, ``final_layernorm``, ``final_norm``, and
    ``ln_f`` (GPT-2 lineage). Returns the first present module, else None."""
    for attr in ("norm", "embedding_norm", "final_layernorm", "final_norm", "ln_f"):
        n = getattr(base_model, attr, None)
        if n is not None:
            return n
    return None


def _embed_prefix(base_model: nn.Module, full_path: str) -> str:
    """Return the full-dotted prefix to the embed_tokens param."""
    return f"{full_path}.embed_tokens." if full_path else "embed_tokens."


def _layer_attention_type(layer: nn.Module):
    return (
        getattr(layer, "layer_type", None)
        or getattr(getattr(layer, "self_attn", None), "layer_type", None)
        or getattr(getattr(layer, "attention", None), "layer_type", None)
    )


def _call_layer(layer: nn.Module, hidden: torch.Tensor, *,
                position_embeddings, attention_mask, position_ids,
                past_key_values=None, **extra) -> torch.Tensor:
    """Call a decoder layer with the common transformers v5 signature.
    Returns hidden output tensor.

    `extra` carries architecture-specific kwargs supplied by the
    profile's `extra_layer_kwargs(...)` (e.g. DSv4-Flash hash-routing
    layers consume `input_ids` for the `tid2eid` lookup). Layers that
    don't consume those kwargs ignore them via `**kwargs` absorption.

    When `position_embeddings` is a `{layer_type: (cos, sin)}` dict (produced
    by `_compute_position_embeddings` for multi-layer-type-rope models like
    Gemma3/Gemma4), select this layer's entry via its attention `layer_type`
    so sliding- and full-attention layers each get their own rope.

    When `attention_mask` is a `{layer_type: mask}` dict, select by the same
    layer type. This mirrors Gemma3/Gemma4 HF forwards, where sliding-window
    and full-attention layers receive different masks.
    """
    lt = None
    pe = position_embeddings
    if isinstance(pe, dict):
        lt = _layer_attention_type(layer)
        pe = pe.get(lt)
        if pe is None:
            if len(position_embeddings) == 1:
                pe = next(iter(position_embeddings.values()))
            else:
                raise RuntimeError(
                    "per-layer position_embeddings requires a known "
                    f"layer_type; got {lt!r} for {layer.__class__.__name__}"
                )
    am = attention_mask
    if isinstance(am, dict):
        if lt is None:
            lt = _layer_attention_type(layer)
        if lt not in am:
            raise RuntimeError(
                "per-layer attention mask requires a known layer_type; "
                f"got {lt!r} for {layer.__class__.__name__}"
            )
        am = am[lt]
    out = layer(
        hidden_states=hidden,
        attention_mask=am,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=False,
        position_embeddings=pe,
        **extra,
    )
    if isinstance(out, tuple):
        return out[0]
    return out


def _compute_position_embeddings(base_model: nn.Module,
                                 hidden: torch.Tensor,
                                 position_ids: torch.Tensor):
    """Call the rotary module to get position embeddings.

    Single-rope models return a `(cos, sin)` tuple. Multi-layer-type-rope
    models (Gemma3/Gemma4: separate rope per attention type, e.g.
    sliding vs full with different `rope_theta`) expose `rotary.layer_types`
    and a `forward(x, position_ids, layer_type=...)`; for those we return a
    `{layer_type: (cos, sin)}` dict and `_call_layer` selects the right entry
    per layer. Returns None if the model exposes no standalone rotary."""
    rotary = _get_rotary(base_model)
    if rotary is None:
        return None
    layer_types = getattr(rotary, "layer_types", None)
    with torch.no_grad():
        if layer_types:
            per_type: dict = {}
            for lt in layer_types:
                try:
                    per_type[lt] = tuple(rotary(hidden, position_ids,
                                                layer_type=lt))
                except TypeError:
                    # Rotary forward doesn't take layer_type (e.g. DSv4 uses
                    # one rope for all layers) — same embeddings for each.
                    per_type[lt] = tuple(rotary(hidden, position_ids))
            return per_type
        cos, sin = rotary(hidden, position_ids)
    return (cos, sin)


def _make_causal_mask(seqlen: int, device: torch.device, dtype: torch.dtype):
    """Build an additive causal mask [1, 1, T, T]. Standard
    upper-triangle -inf convention."""
    mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def _compute_attention_mask(
    base_model: nn.Module,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
):
    """Return the streaming attention mask for a full forward pass.

    Most models use one full causal mask. Gemma3/Gemma4-style hybrid models
    declare ``config.layer_types`` with both ``full_attention`` and
    ``sliding_attention``; those must receive the same per-type mask mapping
    that HuggingFace's model.forward builds.
    """
    cfg = getattr(base_model, "config", None)
    layer_types = tuple(getattr(cfg, "layer_types", ()) or ())
    if cfg is None or "sliding_attention" not in layer_types:
        return _make_causal_mask(hidden.size(1), hidden.device, hidden.dtype)

    try:
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
    except Exception as exc:
        raise RuntimeError(
            "sliding-window layer_types require transformers masking_utils"
        ) from exc

    mask_kwargs = {
        "config": cfg,
        "inputs_embeds": hidden,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    sliding_mask_kwargs = dict(mask_kwargs)
    if getattr(cfg, "use_bidirectional_attention", False):
        try:
            from transformers.models.gemma3.modeling_gemma3 import (
                _bidirectional_window_overlay,
            )
        except Exception as exc:
            raise RuntimeError(
                "Gemma3 bidirectional sliding masks require the Gemma3 "
                "transformers masking helper"
            ) from exc
        mask_kwargs["or_mask_function"] = (
            lambda *args: torch.tensor(True, dtype=torch.bool)
        )
        sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
            cfg.sliding_window
        )

    return {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(
            **sliding_mask_kwargs
        ),
    }


def _resolve_base_prefix(root: nn.Module, base: nn.Module) -> str:
    """Return the dotted name of `base` within `root`, or '' if it is root."""
    for name, mod in root.named_modules():
        if mod is base:
            return name
    return ""


def _head_prefixes(root: nn.Module, base_prefix: str) -> list[str]:
    """Prefixes for the always-resident pieces: embed + norm + lm_head +
    any rotary/position buffers under the base model.

    Architecture-specific names (e.g. an `embedding_norm` final norm or a
    `pos_emb` rotary) are contributed by the profile via
    `head_resident_extra_prefixes`, not hardcoded here (DSv4 adds
    `model.hc_head.` for the multi-stream→single-stream collapse module;
    LFM2.5 adds its `embedding_norm`/`pos_emb`)."""
    p = f"{base_prefix}." if base_prefix else ""
    prefixes = [
        f"{p}embed_tokens.",
        f"{p}norm.",
        "lm_head.",
        f"{p}rotary_emb.",
    ]
    # Profile-driven extension (refactor #32). Default profile returns
    # an empty list; architecture-specific profiles append their own
    # head-resident prefixes here.
    try:
        from .model_profiles import profile_from_model
        extra = profile_from_model(root).head_resident_extra_prefixes(root)
        for pref in extra:
            if pref not in prefixes:
                prefixes.append(pref)
    except Exception:
        # Defensive: fall back to the legacy hardcoded check if the
        # profile import path is unavailable for any reason.
        if hasattr(root, "model") and hasattr(root.model, "hc_head"):
            prefixes.append(f"{p}hc_head.")
        elif hasattr(root, "hc_head"):
            prefixes.append("hc_head.")
    # Some models put per-layer embeddings inputs (layer_scalar on Gemma 4,
    # per_layer_embeddings on multimodal umbrellas) at top level too —
    # not relevant to causal LM text-only path; skip.
    return prefixes
