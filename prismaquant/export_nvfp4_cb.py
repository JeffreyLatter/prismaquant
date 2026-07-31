"""Materialize a PrismaQuant recipe as an NVFP4-CB / FP8-CB checkpoint.

Sibling of :mod:`prismaquant.export_gguf` — the same skeleton-requantize
strategy, but the container is safetensors + a custom compressed-tensors-**style**
``quant_config.json`` whose scheme vocabulary (``nvfp4_cb`` / ``fp8_cb``) only
the out-of-tree vLLM plugin understands (docs/lanes/nvfp4-cb/serving-kernel.md
§2). It is explicitly **not** stock compressed-tensors (whose schemes cannot
express codebooks) — do not route a CB assignment through
:mod:`prismaquant.export_native_compressed`; that exporter hard-fails on CB.

Pipeline: read the bf16 HF skeleton (config.json + *.safetensors), VQ-pack each
target Linear with the **same** weighted closure the cost measured
(:func:`prismaquant.nvfp4_cb_formats.nvfp4_cb_pack`), copy every non-target
tensor verbatim (bf16 passthrough), and emit:

  * ``<name>.cb_qweight``  uint8 (rows, bytes_per_row) — the §1 superblock byte
    stream (index bits + fp4 versioned scale plane: production-v2 two-tier,
    explicit legacy-v1 E4M3-direct; fp8 index bits only);
  * ``<name>.weight_scale`` fp32 (out_features,) — fp8 families only (fp8 has no
    on-disk scale plane; the plane is per-output-channel);
  * ``cb_codebook.<ref>.<fmt>[.sub{i}]`` fp16 — the resolved codebook, shipped
    **once** per (ref, format): ``ref = "lattice"`` for the fixed lattice,
    ``ref = "<role>"`` for a shared per-(role) learned codebook;
  * ``config.json`` (verbatim + a ``quantization_config`` pointer) and
    ``quant_config.json`` (the custom scheme + provenance).

Bit-layout + tensor-naming + config-schema contract: docs/lanes/nvfp4-cb/LAYOUT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.layer_config import (
    _NVFP4_CB_FORMAT_NAMES,
    load_assignment,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    cb_payload_summary,
    cb_serialization_metadata_from_assignment_payload,
    cb_tensor_payload_breakdown,
    finalize_cb_export_artifact_inventory,
    whole_artifact_budget_from_assignment_payload,
    validate_cb_sidecar_tensors,
    validate_cb_assignment_serialization_stamps,
    validate_cb_serialization_context_stamp,
)

# This exporter's own declaration of what the mixed CB container can carry —
# exactly the coverage gate in `export_cb` below: the CB rung families, the two
# stock-CT schemes the plugin delegates to vLLM's CompressedTensors path
# (NVFP4, FP8_E4M3 <- FP8_DYNAMIC), the verbatim FP8_SOURCE passthrough and the
# BF16 container passthrough. The `nvfp4_cb` serving profile's export lane
# derives its format menu from this constant
# (serving_profile_specs/nvfp4_cb.json), so the allocator can never spend budget
# on a rung this exporter would hard-fail on.
EXPORTABLE_FORMATS = frozenset(_NVFP4_CB_FORMAT_NAMES) | frozenset(
    {"NVFP4", "FP8_E4M3", "FP8_SOURCE", "BF16"}
)

def _git_commit() -> str:
    from prismaquant.aura_cost import _git_commit as _aura_git_commit

    return _aura_git_commit() or "unknown"


def _parse_cb_format(fmt: str) -> tuple[str, str, int] | None:
    """``NVFP4_CB_K{k}`` -> (fp4, product, k); ``NVFP4_CB_S{k}`` -> (fp4,
    signed, k); ``FP8_CB_K{k}`` -> (fp8, product, k). None for non-CB."""
    up = str(fmt).strip().upper()
    if up not in _NVFP4_CB_FORMAT_NAMES:
        return None
    if up.startswith("NVFP4_CB_S"):
        return "fp4", "signed", int(up[len("NVFP4_CB_S"):])
    if up.startswith("NVFP4_CB_K"):
        return "fp4", "product", int(up[len("NVFP4_CB_K"):])
    if up.startswith("FP8_CB_K"):
        return "fp8", "product", int(up[len("FP8_CB_K"):])
    return None


def _role_of(qname: str) -> str:
    """Shared-codebook grouping key — the Linear's projection role (last qname
    component), e.g. ``model.layers.3.mlp.gate_proj`` -> ``gate_proj``."""
    return qname.split(".")[-1]


def _load_skeleton(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load every tensor from a HF safetensors dir (single file or sharded)."""
    index = model_dir / "model.safetensors.index.json"
    tensors: dict[str, torch.Tensor] = {}
    if index.exists():
        shards = sorted({
            v for v in json.loads(index.read_text())["weight_map"].values()
        })
        for shard in shards:
            tensors.update(load_file(str(model_dir / shard)))
        return tensors
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(
            f"no model.safetensors[.index.json] under {model_dir}")
    return load_file(str(single))


# --- Nested-prefix skeleton name resolution (hybrid Qwen3.6-27B / Hy3 / DSv4).
# The allocator's recipe qnames are the text-only-staged names
# (`model.layers.N.*`); the on-disk checkpoint nests the LM under an infix
# (`model.language_model.layers.N.*`). The profile knows the structure and maps
# both directions — never hard-code the infix. ---

def _pack_skeleton_experts(skeleton: dict, profile) -> int:
    """Per-expert-on-disk MoE checkpoints (Qwen3.5-MoE / Ornith): assemble
    the packed ``<experts>.gate_up_proj/.down_proj`` skeleton tensors the CB
    targets name, via layer_streaming's tested bridge. No-op for dense or
    already-packed checkpoints.

    Memory discipline: the bridge is invoked once PER packed group (the
    ``live_param_shape`` gate restricts each call), so the transient is one
    expert stack (~1 GB at 35B), not the whole model's expert bytes doubled.
    """
    if profile is None:
        return 0
    regex = getattr(profile, "per_expert_moe_regex", lambda: None)()
    pnames = getattr(profile, "packed_expert_param_names",
                     lambda: frozenset())()
    if not regex or not pnames:
        return 0
    from prismaquant.layer_streaming import _pack_per_expert_into_packed
    pat = re.compile(regex[len("re:"):] if regex.startswith("re:") else regex)

    def is_per_expert(name: str) -> bool:
        if pat.match(name):
            return True
        try:
            return bool(pat.match(profile.to_vllm_internal_name(name)))
        except Exception:
            return False

    # Pre-derive each packed group's expected shape from its per-expert
    # members (E, sum of fused projection out-dims, in).
    members: dict[str, dict[int, dict[str, tuple]]] = defaultdict(
        lambda: defaultdict(dict))
    for key, t in skeleton.items():
        name = key[:-len(".weight")] if key.endswith(".weight") else key
        if not is_per_expert(name):
            continue
        head, proj = name.rsplit(".", 1)
        experts_path, idx = head.rsplit(".", 1)
        if not idx.isdigit():
            continue
        parent = profile.packed_expert_parent_for_projection(proj)
        if parent is None:
            continue
        members[f"{experts_path}.{parent}"][int(idx)][proj] = tuple(t.shape)
    expected: dict[str, tuple] = {}
    for packed_full, by_e in members.items():
        parent = packed_full.rsplit(".", 1)[1]
        order = tuple(profile.packed_expert_projection_names(parent))
        shapes0 = by_e[min(by_e)]
        if any(p not in shapes0 for p in order):
            continue
        out_rows = sum(shapes0[p][0] for p in order)
        in_f = shapes0[order[0]][1]
        expected[packed_full] = (max(by_e) + 1, out_rows, in_f)

    produced = 0
    for packed_full, shape in expected.items():
        n = _pack_per_expert_into_packed(
            skeleton,
            is_per_expert=is_per_expert,
            parent_for_projection=profile.packed_expert_parent_for_projection,
            projection_names_for=profile.packed_expert_projection_names,
            live_param_shape=(
                lambda name, _t=packed_full, _s=shape:
                _s if name == _t else None),
        )
        if packed_full in skeleton:
            skeleton[packed_full + ".weight"] = skeleton.pop(packed_full)
        produced += n
    if produced:
        print(f"[export-cb] packed {produced} per-expert MoE groups into "
              f"stacked skeleton tensors")
    return produced


def _try_resolve_skeleton(qname, skeleton, profile, suffix=".weight"):
    """Recipe qname -> actual skeleton key, or None if neither the direct name
    nor the profile-mapped (checkpoint-convention) name is present."""
    direct = qname + suffix
    if direct in skeleton:
        return direct
    if profile is not None:
        mapped = profile.source_tensor_name(qname) + suffix
        if mapped in skeleton:
            return mapped
    return None


def _resolve_skeleton(qname, skeleton, profile, suffix=".weight"):
    """Strict `_try_resolve_skeleton`: raise listing both names tried."""
    key = _try_resolve_skeleton(qname, skeleton, profile, suffix)
    if key is not None:
        return key
    tried = [qname + suffix]
    if profile is not None:
        tried.append(profile.source_tensor_name(qname) + suffix)
    raise KeyError(
        f"{qname}: no skeleton tensor for {suffix!r} (tried {tried})")


def _export_base_name(qname, profile, skeleton=None):
    """Recipe qname -> the base name the EXPORTED tensor + its config_groups
    target must carry. The profile's checkpoint mapping is only TRUSTED when
    the mapped name actually resolves in the skeleton — a text-only snapshot
    inside a multimodal config shell (qwen35-0.8B: Qwen3_5ForConditional-
    Generation + text_config but model.layers.* keys) otherwise gets every
    config target mis-namespaced under model.language_model.* while the
    tensor writer's fallback uses the real names (2026-07-22 S-rung run:
    nothing resolved at serve, all layers loaded unquantized, crash)."""
    if profile is None:
        return qname
    mapped = profile.source_tensor_name(qname)
    if skeleton is not None and mapped != qname:
        if (mapped + ".weight") not in skeleton and mapped not in skeleton:
            return qname
    return mapped


def _canonical_qname(ckpt_qname, profile):
    """Skeleton (checkpoint) module qname -> canonical recipe qname, or None if
    the profile drops the key (visual/audio/`.weight_scale_inv`)."""
    if profile is None:
        return ckpt_qname
    live = profile.checkpoint_to_live_name(ckpt_qname + ".weight",
                                           multimodal=False)
    return live[:-len(".weight")] if live else None


def _vecs_and_wq(w: torch.Tensor, cw: torch.Tensor | None, grid: str):
    """One-shot scaled 8-dim vectors + per-vector weights for one Linear (the
    same scaling the encoder feeds the VQ search) — mirrors the exp1b driver's
    shared-codebook pooling."""
    w2d = w.reshape(-1, w.shape[-1]).to(torch.float32)
    vectors, _, _ = cb._scale_and_vectorize(w2d, grid)
    wq = None
    if cw is not None:
        # Broadcast against the ORIGINAL shape first so stacked-expert
        # per-expert weights ((E, 1, in) — the gguf _qw_blocks precedent)
        # slice correctly before the row flatten.
        cw2d = torch.broadcast_to(
            cw.to(w2d.device, torch.float32), tuple(w.shape)
        ).reshape(w2d.shape).contiguous()
        wq = cb._col_weight_vectors(cw2d)
    return vectors, wq


def _train_shared_codebook(weights, cws, *, grid, mode, k, seed, iters,
                           train_cap):
    """One learned codebook over a role's pooled scaled vectors (the exp1b
    shared-per-role logic): signed -> positive magnitude table; product ->
    n_sub grid-snapped sub-tables; full -> one (2^k, 8) table."""
    vlist, wlist = [], []
    for w, cw in zip(weights, cws):
        v, wq = _vecs_and_wq(w, cw, grid)
        vlist.append(v)
        wlist.append(wq if wq is not None else torch.ones_like(v))
    vec = torch.cat(vlist, 0)
    wq = torch.cat(wlist, 0)
    if vec.shape[0] > train_cap:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(vec.shape[0], generator=g)[:train_cap].to(
            vec.device)
        vec, wq = vec[idx], wq[idx]
    if mode == "signed":
        return cb.learn_codebook(vec.abs(), k - cb.VEC_DIM, grid=grid,
                                 col_weights=wq, positive=True, iters=iters,
                                 seed=seed).cpu()
    if mode == "product":
        n_sub = cb._product_n_sub(grid)
        sub_dim = cb.VEC_DIM // n_sub
        bits = cb._bit_split(k, n_sub)
        subs = []
        for i, b in enumerate(bits):
            xs = vec[:, i * sub_dim:(i + 1) * sub_dim]
            ws = wq[:, i * sub_dim:(i + 1) * sub_dim]
            init_i = cb.fixed_lattice(b, grid, sub_dim).to(vec.device)
            subs.append(cb.learn_codebook(xs, b, grid=grid, col_weights=ws,
                                          init=init_i, iters=iters,
                                          seed=seed).cpu())
        return tuple(subs)
    return cb.learn_codebook(vec, k, grid=grid, col_weights=wq, iters=iters,
                             seed=seed).cpu()


def _codebook_tensor_names(ref: str, fmt: str, codebook) -> tuple[str, ...]:
    """Physical sidecar tensor names for a resolved codebook object."""
    base = f"cb_codebook.{ref}.{fmt}"
    if isinstance(codebook, (tuple, list)):
        return tuple(f"{base}.sub{i}" for i in range(len(codebook)))
    return (base,)


def _codebook_tensors(ref: str, fmt: str, codebook) -> dict[str, torch.Tensor]:
    """Serialize a codebook (single table or product sub-table tuple) to fp16
    safetensors tensors under ``cb_codebook.<ref>.<fmt>[.sub{i}]`` (grid values
    are exact in fp16 for both the E2M1 and E4M3 grids)."""
    if isinstance(codebook, (tuple, list)):
        return {
            name: tensor.to(torch.float16).cpu().contiguous()
            for name, tensor in zip(
                _codebook_tensor_names(ref, fmt, codebook), codebook
            )
        }
    return {
        _codebook_tensor_names(ref, fmt, codebook)[0]:
        codebook.to(torch.float16).cpu().contiguous()
    }


def export_nvfp4_cb(
    model_dir: str | Path,
    layer_config_path: str | Path,
    out_dir: str | Path,
    col_weights: dict[str, torch.Tensor],
    *,
    shared_codebook_spec: dict | None = None,
    device: str | None = None,
    scale_sweep: bool = True,
    scale_coding: str = cb.SCALE_CODING_TWO_TIER,
) -> dict[str, int]:
    """Export a CB checkpoint. See module docstring / LAYOUT.md for the layout.

    ``col_weights`` maps each CB-target qname to its per-input-column importance
    (imatrix / Fisher). ``shared_codebook_spec`` (or None) selects the codebook
    source:

      * ``None`` / ``{"source": "lattice"}`` — the deterministic fixed lattice,
        shipped as one shared FP16 sidecar table set per format;
      * ``{"source": "learned", "train": True, "iters", "seed", "train_cap"}`` —
        a shared per-(role) learned codebook trained here on pooled vectors;
      * ``{"source": "learned", "codebooks": {role: cb_obj}}`` — use provided
        per-role codebooks (a missing role for a target hard-fails).

    ``scale_coding``: ``"two_tier"`` (production layout v2; fp4 targets write
    4k+9 bytes per superblock) or explicit legacy ``"v1"`` (4k+16). Readers
    remain backward compatible with v1; new artifacts default to v2.
    """
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if scale_coding not in (cb.SCALE_CODING_V1, cb.SCALE_CODING_TWO_TIER):
        raise ValueError(f"unknown scale_coding {scale_coding!r}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = shared_codebook_spec or {}
    source = str(spec.get("source", "lattice")).lower()
    if source not in ("lattice", "learned"):
        raise ValueError(f"shared_codebook_spec source must be lattice/learned,"
                         f" got {source!r}")

    assignment = load_assignment(layer_config_path)
    _recipe_payload = json.loads(Path(layer_config_path).read_text())
    _recipe_cb_context_stamp, _recipe_cb_tensor_stamps = (
        cb_serialization_metadata_from_assignment_payload(_recipe_payload)
    )
    _whole_artifact_budget = whole_artifact_budget_from_assignment_payload(
        _recipe_payload,
        where="export_nvfp4_cb layer config",
    )
    skeleton = _load_skeleton(model_dir)

    # Reuse the compressed-tensors codecs + scheme templates for stock rungs —
    # NEVER reimplement packing. M19 scale-fidelity: `_quantize_2d` renders and
    # packs from ONE scale selection, so the shipped scales ARE the render's.
    from copy import deepcopy as _deepcopy
    from prismaquant import format_registry as _fr
    from prismaquant.export_native_compressed import (
        _quantize_2d as _ct_quantize_2d,
        compute_nvfp4_global_real as _ct_nvfp4_global_real,
        _explicit_regex as _ct_explicit_regex,
        NVFP4_SCHEME as _NVFP4_SCHEME,
        FP8_E4M3_SCHEME as _FP8_E4M3_SCHEME,
        FP8_SOURCE_SCHEME as _FP8_SOURCE_SCHEME,
    )
    from prismaquant.model_profiles import detect_profile as _detect_profile
    # Stock rungs the mixed container carries CT-style (plugin delegates them to
    # vLLM's CompressedTensors path). FP8_DYNAMIC canonicalizes to FP8_E4M3.
    # FP8_SOURCE is a PASSTHROUGH scheme (verbatim fp8 weight + scale_inv copy),
    # not a _quantize_2d target — handled separately below.
    _STOCK_CT_SCHEMES = {"NVFP4": _NVFP4_SCHEME, "FP8_E4M3": _FP8_E4M3_SCHEME}
    # Profile drives nested-prefix skeleton name resolution (hybrid VLMs); None
    # for a flat checkpoint (recipe names == checkpoint names, resolver no-ops).
    try:
        _profile = _detect_profile(str(model_dir))
    except Exception:
        _profile = None
    # Per-expert-on-disk MoE checkpoints: assemble the packed expert stacks
    # the CB targets name BEFORE any skeleton resolution below.
    _pack_skeleton_experts(skeleton, _profile)

    # --- Coverage gate: classify every assigned format into CB / stock-CT /
    # BF16-passthrough (the mixed container, LAYOUT.md §4; "FP8 in every
    # recipe"). ---
    cb_targets: dict[str, tuple[str, str, int]] = {}   # qname -> (grid,mode,k)
    stock_targets: dict[str, str] = {}                 # qname -> "NVFP4"|"FP8_E4M3"
    source_targets: list[str] = []                     # FP8_SOURCE passthrough
    illegal = []
    for qname, fmt in assignment.items():
        if fmt == "BF16":
            continue
        parsed = _parse_cb_format(fmt)
        if parsed is not None:
            cb_targets[qname] = parsed
            continue
        canon = _fr.canonical_format_name(fmt)
        if canon == "FP8_SOURCE":
            source_targets.append(qname)
            continue
        if canon in _STOCK_CT_SCHEMES:
            stock_targets[qname] = canon
            continue
        illegal.append((qname, fmt))
    if illegal:
        raise ValueError(
            f"assignment contains formats the mixed CB container cannot carry: "
            f"{sorted({f for _, f in illegal})} — it carries the CB families "
            f"+ stock NVFP4/FP8_DYNAMIC (CT-delegated) + FP8_SOURCE "
            f"(verbatim fp8 passthrough) + BF16 passthrough only")

    # Sidecar stock targets (visual/audio — modules the profile's LM mapping
    # drops): ship WEIGHT-ONLY (W4A16). Text-only calibration has no visual
    # activations to derive a static input scale from, and vLLM's vision
    # tower builds the weight-only CT variant (no input_global_scale param —
    # the W4A4 tensor set failed to load, 2026-07-22).
    sidecar_stock = {q for q in stock_targets
                     if _canonical_qname(q, _profile) is None}

    # FP8_SOURCE is PASSTHROUGH-ONLY (PASSTHROUGH_SOURCE_REQUIREMENTS): legal
    # only where the source `.weight` is already fp8_e4m3fn with a
    # `.weight_scale_inv` sibling. The allocator's passthrough-integrity
    # filter should drop it otherwise — hard-fail here so a stale manifest
    # never ships a re-synthesized (8-bpp-wasting) FP8 tensor.
    for qname in source_targets:
        wname = _try_resolve_skeleton(qname, skeleton, _profile)
        sname = _try_resolve_skeleton(qname, skeleton, _profile,
                                      ".weight_scale_inv")
        w = skeleton.get(wname) if wname else None
        if w is None or w.dtype != torch.float8_e4m3fn or sname is None:
            raise ValueError(
                f"{qname}: assigned FP8_SOURCE but source is not native FP8 "
                f"(weight dtype={None if w is None else w.dtype}, "
                f"has scale_inv={sname is not None}). FP8_SOURCE is "
                f"passthrough-only — never synthesize it.")

    for qname, (grid, mode, k) in cb_targets.items():
        wname = _try_resolve_skeleton(qname, skeleton, _profile)
        if wname is None:
            raise ValueError(
                f"{qname}: assigned {grid}/{mode} k{k} but no weight tensor for "
                f"it in the skeleton (tried {qname}.weight + the "
                f"profile-mapped checkpoint name)")
        in_f = int(skeleton[wname].shape[-1])
        if in_f % cb.SUPERBLOCK != 0:
            raise ValueError(
                f"{qname}: in_features={in_f} is not a multiple of "
                f"{cb.SUPERBLOCK}; fall back to a coarser legal rung or BF16 "
                f"(no block-32 CB rung in Phase 0)")
        if qname not in col_weights:
            raise ValueError(
                f"{qname}: CB target has no col_weights entry — exporting "
                f"unweighted bytes would silently diverge from the "
                f"imatrix-weighted cost measurement (no silent RTN)")
        cwn = col_weights[qname].numel()
        n_exp = (int(skeleton[wname].shape[0])
                 if skeleton[wname].dim() == 3 else 1)
        if cwn not in (in_f, n_exp * in_f):
            raise ValueError(
                f"{qname}: col_weights has {cwn} elements but the weight "
                f"wants {in_f} (shared) or {n_exp}x{in_f} (per-expert, "
                f"(E,1,in)) — the imatrix does not describe this checkpoint")

    # --- Stock NVFP4 fused-sibling coherence: q/k/v (and gate/up) that all land
    # on NVFP4 MUST share one weight_global_scale, or vLLM's fused loader sees
    # inconsistent per-tensor global scales. Take the max over each fused group
    # and override every sibling's pack (mirrors export_native_compressed). ---
    for qname in stock_targets:
        if _try_resolve_skeleton(qname, skeleton, _profile) is None:
            raise ValueError(
                f"{qname}: assigned {stock_targets[qname]} but no weight tensor "
                f"for it in the skeleton (tried {qname}.weight + the "
                f"profile-mapped checkpoint name)")
    _nvfp4_shared_global: dict[str, torch.Tensor] = {}
    _nvfp4_groups: dict[str, list[str]] = {}
    for _q, _f in stock_targets.items():
        if _f != "NVFP4":
            continue
        _gk = (_profile.fused_sibling_group(_q)
               if _profile is not None else None) or _q
        _nvfp4_groups.setdefault(_gk, []).append(_q)
    for _members in _nvfp4_groups.values():
        _grs = [_ct_nvfp4_global_real(
                    skeleton[_resolve_skeleton(_m, skeleton, _profile)].to(device),
                    16)
                for _m in _members]
        _shared = torch.stack([g.reshape(()) for g in _grs]).max()
        for _m in _members:
            _nvfp4_shared_global[_m] = _shared

    # --- Resolve/train codebooks, grouped by (ref, format). ---
    provided = spec.get("codebooks", {}) if source == "learned" else {}
    train = bool(spec.get("train", False))
    iters = int(spec.get("iters", 4))
    seed = int(spec.get("seed", 0))
    train_cap = int(spec.get("train_cap", 1 << 20))

    # (ref, fmt) -> codebook object; ref = "lattice" or role.
    codebooks: dict[tuple[str, str], object] = {}
    # qname -> (ref, fmt, codebook, source_kind)
    target_cb: dict[str, tuple[str, str, object, str]] = {}
    by_group: dict[tuple[str, str], list[str]] = {}
    for qname, (grid, mode, k) in cb_targets.items():
        fmt = assignment[qname]
        ref = _role_of(qname) if source == "learned" else "lattice"
        by_group.setdefault((ref, fmt), []).append(qname)

    for (ref, fmt), qnames in by_group.items():
        grid, mode, k = cb_targets[qnames[0]]
        if source == "lattice":
            codebooks[(ref, fmt)] = cb._resolve_codebook(
                k, grid, mode, None, torch.device(device))
            kind = "lattice"
        else:
            role = ref
            if train:
                weights = [skeleton[_resolve_skeleton(q, skeleton, _profile)]
                           .to(device) for q in qnames]
                cws = [col_weights[q].to(device) for q in qnames]
                codebooks[(ref, fmt)] = _train_shared_codebook(
                    weights, cws, grid=grid, mode=mode, k=k, seed=seed,
                    iters=iters, train_cap=train_cap)
            elif role in provided:
                codebooks[(ref, fmt)] = provided[role]
            else:
                raise ValueError(
                    f"role {role!r} ({fmt}): codebook_source=learned but no "
                    f"codebook supplied and train=False — missing learned "
                    f"sidecar for {len(qnames)} tensor(s)")
            kind = "learned"
        for q in qnames:
            target_cb[q] = (ref, fmt, codebooks[(ref, fmt)], kind)

    # Bind byte pricing to the exact physical sidecar refs this artifact will
    # write.  This identity is shared by allocation/reporting/export checks;
    # no producer path silently assumes the legacy-v1 scale plane.
    materialized_codebook_tensors = {
        name: tensor
        for (ref, fmt), codebook in codebooks.items()
        for name, tensor in _codebook_tensors(ref, fmt, codebook).items()
    }
    materialized_codebook_digests = {
        name: hashlib.sha256(
            tensor.to(torch.float16).cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for name, tensor in materialized_codebook_tensors.items()
    }
    serialization_context = CBSerializationContext(
        scale_coding=scale_coding,
        codebook_source=source,
        codebook_refs={
            qname: _codebook_tensor_names(ref, fmt, codebook)
            for qname, (ref, fmt, codebook, _kind) in target_cb.items()
        },
        codebook_content_digests=materialized_codebook_digests,
    )
    validate_cb_serialization_context_stamp(
        _recipe_cb_context_stamp,
        serialization_context,
        where="export_nvfp4_cb",
    )

    # --- Pack targets; copy everything else verbatim. ---
    out_tensors: dict[str, torch.Tensor] = {}
    cb_tensor_blobs: dict[str, torch.Tensor] = {}
    cb_serialized_shapes: dict[str, tuple[int, ...]] = {}
    cb_output_tensor_names: set[str] = set()
    actual_cb_tensor_bytes = 0
    counts: Counter[str] = Counter()
    ignore: list[str] = []
    packed_qnames = set(cb_targets)
    source_qnames = set(source_targets)
    # scale_inv siblings of FP8_SOURCE targets are emitted verbatim in the
    # source branch below; skip them in the passthrough else-branch so they
    # are neither double-emitted nor added to the ignore list.
    _source_scale_keys = {
        _try_resolve_skeleton(q, skeleton, _profile, ".weight_scale_inv")
        for q in source_qnames}
    _source_scale_keys.discard(None)

    for name, tensor in skeleton.items():
        # `name` is the CHECKPOINT key; `ckpt_qname` its module base (drives the
        # EXPORTED tensor names — vLLM's convention, incl. the language_model
        # infix); `canon` is the canonical recipe qname that assignment /
        # col_weights / cb_targets are keyed by (nested -> canonical).
        ckpt_qname = name[:-len(".weight")] if name.endswith(".weight") else None
        canon = _canonical_qname(ckpt_qname, _profile) if ckpt_qname else None
        if canon is None and ckpt_qname is not None and (
                ckpt_qname in stock_targets or ckpt_qname in cb_targets):
            # Sidecar modules the profile's LM mapping drops (visual/audio)
            # but the recipe DOES assign (e.g. VISUAL_FORMAT=NVFP4): their
            # recipe qnames are already checkpoint-form, so classify by the
            # raw name. Without this the config pass promised a stock group
            # for 110 visual Linears while the write pass copied raw BF16 —
            # a split-brain artifact vLLM cannot load (2026-07-22 27B).
            canon = ckpt_qname
        if name in _source_scale_keys:
            continue
        if canon in source_qnames:
            # FP8_SOURCE passthrough: copy the native fp8 `.weight` verbatim
            # and rename `.weight_scale_inv` -> `.weight_scale` (bytes
            # verbatim, fp32) — EXACTLY as the CT streaming exporter does
            # (export_native_compressed:5711), so stock compressed-tensors
            # block-fp8 delegation reads it unchanged. No dequant/requant
            # round-trip; NOT added to ignore (it is an FP8_SOURCE group).
            out_tensors[ckpt_qname + ".weight"] = tensor.contiguous()
            sname = _resolve_skeleton(canon, skeleton, _profile,
                                      ".weight_scale_inv")
            out_tensors[ckpt_qname + ".weight_scale"] = skeleton[sname].to(
                torch.float32).contiguous()
            counts["FP8_SOURCE"] += 1
            continue
        if canon in packed_qnames:
            grid, mode, k = cb_targets[canon]
            ref, fmt, codebook, _ = target_cb[canon]
            cbook = _to_device(codebook, device)
            w = tensor.to(device)
            packed, fields = cb.nvfp4_cb_pack(
                w, k, grid=grid, mode=mode,
                col_weights=col_weights[canon].to(device),
                codebook=cbook, scale_sweep=scale_sweep,
                scale_coding=(scale_coding if grid == "fp4"
                              else cb.SCALE_CODING_V1))
            if w.dim() == 3:
                # Stacked packed experts: keep the expert axis explicit —
                # uint8 (E, out, bytes_per_row); fp8 per-channel scales
                # (E, out). LAYOUT.md §3 (stacked experts).
                packed = packed.reshape(w.shape[0], w.shape[1], -1)
            packed_out = packed.to(torch.uint8).cpu().contiguous()
            payload = cb_tensor_payload_breakdown(
                fmt,
                tuple(int(dim) for dim in w.shape),
                qname=canon,
                context=serialization_context,
            )
            packed_bytes = packed_out.numel() * packed_out.element_size()
            if packed_bytes != payload["packed_weight_bytes"]:
                raise AssertionError(
                    f"{canon}: serialized cb_qweight is {packed_bytes}B, "
                    f"accounting expected {payload['packed_weight_bytes']}B"
                )
            packed_name = ckpt_qname + ".cb_qweight"
            out_tensors[packed_name] = packed_out
            cb_output_tensor_names.add(packed_name)
            scale_bytes = 0
            if grid == "fp8":
                ws = fields["scales"].reshape(
                    *w.shape[:-1]).to(torch.float32).cpu().contiguous()
                scale_bytes = ws.numel() * ws.element_size()
                if scale_bytes != payload["fp8_row_scale_bytes"]:
                    raise AssertionError(
                        f"{canon}: serialized weight_scale is {scale_bytes}B, "
                        "accounting expected "
                        f"{payload['fp8_row_scale_bytes']}B"
                    )
                scale_name = ckpt_qname + ".weight_scale"
                out_tensors[scale_name] = ws
                cb_output_tensor_names.add(scale_name)
            elif payload["fp8_row_scale_bytes"]:
                raise AssertionError(
                    f"{canon}: FP4-CB unexpectedly priced an FP8 row scale"
                )
            actual = packed_bytes + scale_bytes
            if actual != payload["tensor_payload_bytes"]:
                raise AssertionError(
                    f"{canon}: emitted {actual}B of CB tensor payload, "
                    f"accounting expected {payload['tensor_payload_bytes']}B"
                )
            cb_serialized_shapes[canon] = tuple(int(dim) for dim in w.shape)
            actual_cb_tensor_bytes += actual
            counts[fmt] += 1
        elif canon in stock_targets:
            # Stock rung: CT-pack via the shared compressed-tensors codec
            # (RTN default levers = the render the allocator cost measured; the
            # packed scales are the render's, M19). Emit the CT suffix tensors
            # verbatim; NOT added to the ignore list (it is quantized).
            fmt = stock_targets[canon]
            override = (_nvfp4_shared_global.get(canon)
                        if fmt == "NVFP4" else None)
            packed = _ct_quantize_2d(
                tensor.to(device), fmt, nvfp4_global_real_override=override)
            for suffix, t in packed.items():
                if canon in sidecar_stock and "input" in suffix:
                    continue        # weight-only sidecar group (see above)
                out_tensors[f"{ckpt_qname}.{suffix}"] = t.cpu().contiguous()
            counts[assignment[canon]] += 1
        else:
            # Verbatim (BF16 passthrough, norms, embeddings, visual encoder,
            # lm_head) under the checkpoint name; 2-D unquantized Linears go to
            # the ignore list by their checkpoint/vLLM name.
            out_tensors[name] = tensor.contiguous()
            if ckpt_qname is not None and tensor.dim() >= 2:
                ignore.append(ckpt_qname)
            counts["copied"] += 1

    # --- Codebook tensors: shipped once per (ref, fmt) in a NON-safetensors-
    # globbed sidecar (cb_codebooks.pqcb) so vLLM's weight loader never sees
    # these non-parameter tensors. The plugin loads them explicitly via the
    # config's codebook_file pointer (plugins/gridbook config.py
    # get_codebooks -> load_file(model_dir/cb_codebooks.pqcb)), keyed by each
    # scheme's codebook_ref. Sidecar-only: NOT written into model.safetensors. ---
    cb_tensor_blobs.update(materialized_codebook_tensors)
    codebook_file = "cb_codebooks.pqcb" if cb_tensor_blobs else None

    if set(cb_serialized_shapes) != set(cb_targets):
        missing = sorted(set(cb_targets) - set(cb_serialized_shapes))
        extra = sorted(set(cb_serialized_shapes) - set(cb_targets))
        raise AssertionError(
            "CB serialized-payload coverage does not match assignment: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    serialized_payload = cb_assignment_payload_breakdown(
        {qname: assignment[qname] for qname in cb_targets},
        cb_serialized_shapes,
        context=serialization_context,
    )
    if _recipe_cb_context_stamp is not None or _recipe_cb_tensor_stamps:
        validate_cb_assignment_serialization_stamps(
            {qname: assignment[qname] for qname in cb_targets},
            cb_serialized_shapes,
            context=serialization_context,
            stamps=_recipe_cb_tensor_stamps,
            where="export_nvfp4_cb",
        )
    if actual_cb_tensor_bytes != serialized_payload["tensor_payload_bytes"]:
        raise AssertionError(
            f"emitted CB tensor payload is {actual_cb_tensor_bytes}B, "
            "assignment accounting expected "
            f"{serialized_payload['tensor_payload_bytes']}B"
        )
    validate_cb_sidecar_tensors(
        serialized_payload,
        cb_tensor_blobs,
        where="export_nvfp4_cb",
    )
    serialized_payload_summary = cb_payload_summary(serialized_payload)

    # --- Provenance hashes. ---
    assignment_sha = hashlib.sha256(json.dumps(
        dict(sorted(assignment.items())), separators=(",", ":")).encode(),
    ).hexdigest()
    ih = hashlib.sha256()
    for q in sorted(col_weights):
        ih.update(q.encode())
        ih.update(col_weights[q].to(torch.float32).cpu().numpy().tobytes())
    imatrix_sha = ih.hexdigest()
    codebook_sha = {
        tname: hashlib.sha256(
            blob.to(torch.float16).cpu().numpy().tobytes()).hexdigest()
        for tname, blob in cb_tensor_blobs.items()
    }

    # --- Custom quant config (config_groups keyed by scheme signature). ---
    config_groups: dict[str, dict] = {}
    for gi, ((ref, fmt), qnames) in enumerate(sorted(by_group.items())):
        grid, mode, k = cb_targets[qnames[0]]
        codebook = codebooks[(ref, fmt)]
        n_sub = len(codebook) if isinstance(codebook, (tuple, list)) else 1
        base = f"cb_codebook.{ref}.{fmt}"
        codebook_ref = ([f"{base}.sub{i}" for i in range(n_sub)]
                        if n_sub > 1 else base)
        group_coding = (scale_coding if grid == "fp4"
                        else cb.SCALE_CODING_V1)
        scheme = {
            "grid": grid,
            "mode": mode,
            "k": k,
            "superblock": cb.SUPERBLOCK,
            "group_size": cb.FP4_GROUP if grid == "fp4" else 0,
            "vec_dim": cb.VEC_DIM,
            "n_sub": n_sub,
            "type_size": cb.nvfp4_cb_type_size(k, grid, group_coding),
            "act_bits": 4 if grid == "fp4" else 8,
            "codebook_source": (
                "lattice" if ref == "lattice" else "learned"),
            "codebook_ref": codebook_ref,
            "codebook_group": None if ref == "lattice" else ref,
        }
        if group_coding == cb.SCALE_CODING_TWO_TIER:
            # Table entries asserted e4m3-exact by _two_tier_tables; ship the
            # 16 floats so the scheme is self-describing (spec §1.3/§5.1).
            table, _, _ = cb._two_tier_tables("cpu")
            scheme["scale_coding"] = {
                "kind": "two_tier",
                "sub_bits": 4,
                "super_bias": cb.TWO_TIER_SUPER_BIAS,
                "table": [float(t) for t in table.tolist()],
            }
        config_groups[f"group_{gi}"] = {
            # Targets are the CANONICAL qnames: vLLM's class mapper serves the
            # LM at model.layers.* regardless of the checkpoint's infix
            # convention (the 0.8B ships model.language_model.* on disk yet
            # serves at model.layers.* — checkpoint-namespace targets matched
            # nothing and every layer loaded unquantized, 2026-07-22).
            # Checkpoint names remain the TENSOR convention only.
            "targets": sorted(qnames),
            "format": fmt,
            "scheme": scheme,
        }
    # Stock CT config_groups: EXACT compressed-tensors vocabulary (weights/
    # input_activations/format at the group top, NO "scheme" key) so the plugin
    # hands them straight to CompressedTensorsConfig.from_config under
    # delegation. The presence of a "scheme" key is the CB-vs-stock dispatch
    # marker (LAYOUT.md §4): CB groups have "scheme"; stock CT groups do not.
    _stock_by_fmt: dict[str, list[str]] = {}
    for _q, _f in stock_targets.items():
        _key = f"{_f}//sidecar" if _q in sidecar_stock else _f
        _stock_by_fmt.setdefault(_key, []).append(_q)
    def _sidecar_serving_name(q: str) -> str:
        # Config-group targets must match vLLM's SERVING prefixes. The LM
        # keeps the checkpoint's `model.` prefix at serve time, but sidecar
        # towers do not: the Qwen VL mapper serves checkpoint
        # `model.visual.*` as module `visual.*` (demonstrated by the loader
        # itself — 'blocks.0.attn.proj ... in Qwen3_VisionTransformer').
        # Tensor NAMES stay checkpoint-form; only group targets strip the
        # leading `model.`.
        return q[len("model."):] if q.startswith("model.") else q

    for _key, _qnames in sorted(_stock_by_fmt.items()):
        _f = _key.split("//")[0]
        _group = _deepcopy(_STOCK_CT_SCHEMES[_f])
        _names = (_qnames if not _key.endswith("//sidecar")
                  else [_sidecar_serving_name(q) for q in _qnames])
        if _key.endswith("//sidecar"):
            # weight-only (W4A16/W8A16): no activation contract for sidecar
            # towers; CT vocabulary = input_activations null.
            _group["input_activations"] = None
        # Serving-namespace targets: canonical qnames (sidecars strip the
        # model. prefix per the class mapper) — see the CB-group note above.
        _group["targets"] = sorted(_ct_explicit_regex(q) for q in _names)
        config_groups[f"group_{len(config_groups)}"] = _group
    # FP8_SOURCE passthrough group: the stock CT `float-quantized` block-fp8
    # scheme (no "scheme" key -> the plugin delegates it to CompressedTensors,
    # exactly like the other stock rungs; the emitted `.weight`/`.weight_scale`
    # names match FP8_SOURCE_SCHEME).
    if source_targets:
        _src_group = _deepcopy(_FP8_SOURCE_SCHEME)
        _src_group["targets"] = sorted(
            _ct_explicit_regex(q) for q in source_targets)
        config_groups[f"group_{len(config_groups)}"] = _src_group
    quant_config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "config_groups": config_groups,
        "ignore": sorted(set(ignore)),
        **({"codebook_file": codebook_file} if codebook_file else {}),
        "provenance": {
            "git_commit": _git_commit(),
            "assignment_sha256": assignment_sha,
            "imatrix_sha256": imatrix_sha,
            "codebook_sha256": codebook_sha,
            "codebook_source": source,
            "scale_sweep": bool(scale_sweep),
            "scale_coding": scale_coding,
            "cb_targets": len(cb_targets),
            "stock_ct_targets": len(stock_targets),
            "fp8_source_targets": len(source_targets),
            "serialized_payload": serialized_payload_summary,
            "tensor_formats": {
                q: assignment[q]
                for q in sorted(set(cb_targets) | set(stock_targets)
                                | set(source_targets))},
        },
    }
    if scale_coding == cb.SCALE_CODING_TWO_TIER:
        # Absence of layout_version (and of any scheme scale_coding key)
        # means v1 — old artifacts parse unchanged, forever (spec §5.1).
        quant_config["layout_version"] = 2

    # --- Write safetensors (params only) + the codebook sidecar + configs. ---
    save_file(out_tensors, str(out_dir / "model.safetensors"),
              metadata={"format": "pt", "quant_method": "gridbook"})
    if codebook_file:
        # The .pqcb is a plain safetensors blob under a non-globbed extension:
        # the plugin reads it with safetensors.load_file, vLLM's *.safetensors
        # weight globber skips it (LAYOUT.md §3 codebook contract).
        save_file({k: v.contiguous() for k, v in cb_tensor_blobs.items()},
                  str(out_dir / codebook_file),
                  metadata={"format": "pt", "quant_method": "gridbook"})
    src_config = model_dir / "config.json"
    config = json.loads(src_config.read_text()) if src_config.exists() else {}
    config["quantization_config"] = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "config_file": "quant_config.json",
        **({"codebook_file": codebook_file} if codebook_file else {}),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    # Copy tokenizer / generation / multimodal sidecars verbatim (best effort).
    # The multimodal preprocessor configs are REQUIRED for VLM checkpoints
    # (e.g. Qwen3-VL): vLLM's input processor calls
    # `image_processor.from_pretrained(model_dir)` at load and hard-fails
    # without preprocessor_config.json — the artifact will not serve. Copy the
    # chat template too so chat/tool serving matches the source.
    for aux in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                "special_tokens_map.json", "generation_config.json",
                "vocab.json", "merges.txt",
                "preprocessor_config.json", "video_preprocessor_config.json",
                "processor_config.json", "chat_template.jinja",
                "chat_template.json"):
        p = model_dir / aux
        if p.exists():
            (out_dir / aux).write_bytes(p.read_bytes())
    # Final measured bytes are a separate scope from CB tensor-data pricing:
    # include safetensors headers, JSON, tokenizer files, and every other
    # regular file.  The helper embeds a self-consistent inventory in
    # quant_config.json and re-checks the exact CB spans in the final files.
    finalize_cb_export_artifact_inventory(
        out_dir,
        quant_config,
        serialized_payload=serialized_payload_summary,
        cb_tensor_names=sorted(cb_output_tensor_names),
        codebook_file=codebook_file,
        expected_model_files=["model.safetensors"],
        whole_artifact_budget_bytes=(
            int(_whole_artifact_budget["budget_bytes"])
            if _whole_artifact_budget is not None
            else None
        ),
    )
    return dict(counts)


def _to_device(codebook, device):
    if isinstance(codebook, (tuple, list)):
        return tuple(t.to(device) for t in codebook)
    return codebook.to(device)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True,
                    help="HF model dir (config.json + *.safetensors, bf16)")
    ap.add_argument("--layer-config", required=True,
                    help="assignment JSON (qname -> CB format)")
    ap.add_argument("--out", required=True, help="output checkpoint dir")
    ap.add_argument("--col-weights", required=True,
                    help="pickle: {qname: per-column importance tensor}")
    ap.add_argument("--codebook-source", default="lattice",
                    choices=["lattice", "learned"],
                    help="fixed lattice sidecar or shared per-role "
                    "learned codebooks trained at export time")
    ap.add_argument("--codebook-iters", type=int, default=4)
    ap.add_argument("--codebook-seed", type=int, default=0)
    ap.add_argument("--no-scale-sweep", action="store_true",
                    help="one-shot amax/grid-max scale (A/B only; default is "
                    "the joint scale sweep, IQ-rendering parity)")
    ap.add_argument("--scale-coding", default=cb.SCALE_CODING_TWO_TIER,
                    choices=[cb.SCALE_CODING_V1, cb.SCALE_CODING_TWO_TIER],
                    help="fp4 scale coding: production layout-v2 two-tier "
                    "super+sub coding (default), or explicit legacy v1 e4m3 "
                    "plane for backward-compatible artifacts")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("export_nvfp4_cb")

    with open(args.col_weights, "rb") as fh:
        col_weights = pickle.load(fh)
    col_weights = {k: torch.as_tensor(v) for k, v in col_weights.items()}
    spec = {"source": args.codebook_source}
    if args.codebook_source == "learned":
        spec.update(train=True, iters=args.codebook_iters,
                    seed=args.codebook_seed)
    counts = export_nvfp4_cb(
        args.model_dir, args.layer_config, args.out, col_weights,
        shared_codebook_spec=spec, device=args.device,
        scale_sweep=not args.no_scale_sweep,
        scale_coding=args.scale_coding,
    )
    size = sum(p.stat().st_size for p in Path(args.out).glob("*")) / 1e9
    print(f"wrote {args.out} ({size:.3f} GB)")
    for fmt, n in sorted(counts.items()):
        print(f"  {fmt}: {n}")


if __name__ == "__main__":
    main()
