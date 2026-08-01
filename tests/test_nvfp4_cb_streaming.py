"""Streaming NVFP4-CB exporter — CPU-only, tiny synthetic, compile-off.

Pins the streaming exporter (prismaquant.export_nvfp4_cb_streaming) against the
in-memory export_nvfp4_cb: byte-identical packed output, bounded peak
residency, per-expert->stacked bridging, fp8-source dequant-on-read, and the
stock-CT scope gate. No GPU, no torch.compile (PRISMAQUANT_CB_ENCODE_COMPILE=0).
"""
from __future__ import annotations

import json
import importlib
import os
import shutil
import struct
import weakref
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

os.environ["PRISMAQUANT_CB_ENCODE_COMPILE"] = "0"

from prismaquant.export_nvfp4_cb import (  # noqa: E402
    export_nvfp4_cb as _export_nvfp4_cb,
)
from prismaquant.export_nvfp4_cb_streaming import (  # noqa: E402
    _LazySkeleton,
    _StreamWriter,
    export_nvfp4_cb_streaming as _export_nvfp4_cb_streaming,
    main as _cb_stream_main,
)
from prismaquant.export_native_compressed import (  # noqa: E402
    _quantize_2d,
    build_quantization_config,
    compute_nvfp4_global_real,
)
from prismaquant.model_profiles import detect_profile  # noqa: E402


def export_nvfp4_cb(*args, **kwargs):
    """This module's synthetic direct calls are explicit research renders."""
    kwargs.setdefault("allow_unstamped_research", True)
    return _export_nvfp4_cb(*args, **kwargs)


def export_nvfp4_cb_streaming(*args, **kwargs):
    """This module's synthetic direct calls are explicit research renders."""
    kwargs.setdefault("allow_unstamped_research", True)
    return _export_nvfp4_cb_streaming(*args, **kwargs)


def _st_header(path: Path) -> tuple[dict, int]:
    """Parse a safetensors file's header dict and data-start offset."""
    raw = path.read_bytes()
    hlen = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8:8 + hlen]), 8 + hlen


def _assert_offsets_consistent(path: Path) -> dict:
    """The streaming header must lay tensors out gap-free, in order, with
    data_offsets matching dtype x shape (requirement a)."""
    header, _ = _st_header(path)
    _bytes = {"U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "F16": 2, "BF16": 2,
              "I32": 4, "F32": 4, "I64": 8}
    off = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        lo, hi = meta["data_offsets"]
        n = 1
        for d in meta["shape"]:
            n *= int(d)
        assert lo == off, f"{name}: gap/overlap at {lo} != {off}"
        assert hi - lo == n * _bytes[meta["dtype"]], f"{name}: nbytes mismatch"
        off = hi
    return header


def _stock_by_scheme(quant_config: dict) -> dict:
    """Config groups WITHOUT a 'scheme' key (stock CT / FP8_SOURCE), normalized
    by target-set so group-key ordering doesn't matter."""
    return {tuple(sorted(g["targets"])):
            {k: v for k, v in g.items() if k != "targets"}
            for g in quant_config["config_groups"].values() if "scheme" not in g}

@pytest.fixture
def workdir(tmp_path: Path):
    """Keep synthetic exports isolated and portable across CI runners."""
    return tmp_path


def _write_model(mdl: Path, tensors: dict, hid: int = 256):
    mdl.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(json.dumps({"hidden_size": hid}))


def _write_sharded_model(mdl: Path, tensors: dict, hid: int = 256):
    mdl.mkdir(parents=True, exist_ok=True)
    items = list(tensors.items())
    midpoint = max(1, len(items) // 2)
    shards = [items[:midpoint], items[midpoint:]]
    shards = [shard for shard in shards if shard]
    weight_map = {}
    for index, shard in enumerate(shards, start=1):
        name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        save_file(dict(shard), str(mdl / name))
        weight_map.update({tensor_name: name for tensor_name, _ in shard})
    (mdl / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {},
        "weight_map": weight_map,
    }))
    (mdl / "config.json").write_text(json.dumps({"hidden_size": hid}))


def _assign(path: Path, mapping: dict):
    path.write_text(json.dumps(mapping))


def _tensors_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_cb_exporters_reject_in_place_single_source_before_mutation(
    workdir,
    exporter,
):
    mdl = workdir / "model"
    _write_model(mdl, {"model.norm.weight": torch.ones(4)})
    assignment = workdir / "assignment.json"
    _assign(assignment, {})
    before = _tree_bytes(mdl)

    with pytest.raises(RuntimeError, match="resolve to the same path"):
        exporter(mdl, assignment, mdl, {}, device="cpu")

    assert _tree_bytes(mdl) == before


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_cb_exporters_reject_output_nested_under_source(
    workdir,
    exporter,
):
    mdl = workdir / "model"
    _write_model(mdl, {"model.norm.weight": torch.ones(4)})
    assignment = workdir / "assignment.json"
    _assign(assignment, {})
    output = mdl / "exported"
    before = _tree_bytes(mdl)

    with pytest.raises(RuntimeError, match="ancestor/descendant"):
        exporter(mdl, assignment, output, {}, device="cpu")

    assert not output.exists()
    assert _tree_bytes(mdl) == before


def test_streaming_rejects_in_place_sharded_source_before_mutation(workdir):
    mdl = workdir / "sharded-model"
    _write_sharded_model(
        mdl,
        {
            "model.embed_tokens.weight": torch.ones(2, 2),
            "model.norm.weight": torch.ones(2),
        },
        hid=2,
    )
    assignment = workdir / "assignment.json"
    _assign(assignment, {})
    before = _tree_bytes(mdl)

    with pytest.raises(RuntimeError, match="resolve to the same path"):
        export_nvfp4_cb_streaming(
            mdl,
            assignment,
            mdl,
            {},
            device="cpu",
        )

    assert _tree_bytes(mdl) == before


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_cb_exporters_reject_symlink_alias_and_stale_aux(
    workdir,
    exporter,
):
    mdl = workdir / "model"
    _write_model(mdl, {"model.norm.weight": torch.ones(4)})
    assignment = workdir / "assignment.json"
    _assign(assignment, {})
    before = _tree_bytes(mdl)

    alias = workdir / "source-alias"
    alias.symlink_to(mdl, target_is_directory=True)
    with pytest.raises(RuntimeError, match="resolve to the same path"):
        exporter(mdl, assignment, alias, {}, device="cpu")
    assert alias.is_symlink()
    assert _tree_bytes(mdl) == before

    output = workdir / "stale-output"
    output.mkdir()
    stale = output / "tokenizer_config.json"
    stale.write_text('{"old": true}')
    with pytest.raises(RuntimeError, match="is not empty"):
        exporter(mdl, assignment, output, {}, device="cpu")
    assert stale.read_text() == '{"old": true}'
    assert set(output.iterdir()) == {stale}


@pytest.mark.parametrize(
    ("module_name", "exporter"),
    [
        ("prismaquant.export_nvfp4_cb", export_nvfp4_cb),
        (
            "prismaquant.export_nvfp4_cb_streaming",
            export_nvfp4_cb_streaming,
        ),
    ],
    ids=["batch", "streaming"],
)
def test_cb_export_transaction_cleans_post_model_budget_failure(
    workdir,
    monkeypatch,
    module_name,
    exporter,
):
    mdl = workdir / "model"
    _write_model(mdl, {"model.norm.weight": torch.ones(4)})
    assignment = workdir / "assignment.json"
    _assign(assignment, {})
    output = workdir / "artifact"
    module = importlib.import_module(module_name)

    def fail_final_inventory(out_dir, *_args, **_kwargs):
        staged = Path(out_dir)
        assert staged != output
        assert (staged / "model.safetensors").is_file()
        raise RuntimeError("hard whole-artifact budget exceeded")

    monkeypatch.setattr(
        module,
        "finalize_cb_export_artifact_inventory",
        fail_final_inventory,
    )
    with pytest.raises(RuntimeError, match="hard whole-artifact budget"):
        exporter(mdl, assignment, output, {}, device="cpu")

    assert not output.exists()
    assert not list(workdir.glob(f".{output.name}.tmp-*"))


# --- byte-identity: dense CB + BF16 + stacked-3D experts + fp8_cb -----------

def test_streaming_byte_identical_dense_and_stacked(workdir):
    torch.manual_seed(0)
    mdl = workdir / "model"
    tens = {
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.experts.gate_up_proj.weight":
            (torch.randn(3, 64, 256) * 0.3).to(torch.bfloat16),   # stacked
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    }
    _write_model(mdl, tens)
    ap = workdir / "a.json"
    _assign(ap, {
        "model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb",
                                            "cb_k": 16},
        "model.layers.0.mlp.experts.gate_up_proj": {"data_type": "fp8_cb",
                                                    "cb_k": 40}})
    cw = {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05,
          "model.layers.0.mlp.experts.gate_up_proj": torch.rand(3, 1, 256)
          + 0.05}
    cm = export_nvfp4_cb(mdl, ap, workdir / "m", cw, device="cpu")
    cs = export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    assert dict(cm) == dict(cs)
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    assert _tensors_equal(tm, ts)
    qm = json.loads((workdir / "m" / "quant_config.json").read_text())
    qs = json.loads((workdir / "s" / "quant_config.json").read_text())
    assert qm["config_groups"] == qs["config_groups"]
    assert qm["ignore"] == qs["ignore"]
    assert qs["provenance"]["streaming"] is True
    for root, config in ((workdir / "m", qm), (workdir / "s", qs)):
        inventory = config["provenance"]["artifact_inventory"]
        files = {
            path.relative_to(root).as_posix(): path.stat().st_size
            for path in root.rglob("*") if path.is_file()
        }
        assert inventory["file_bytes"] == files
        assert inventory["export_directory_bytes"] == sum(files.values())
        assert inventory["cb_serialized_payload_bytes"] == (
            config["provenance"]["serialized_payload"]["total_bytes"]
        )
    # codebook sidecars identical
    cbm = load_file(str(workdir / "m" / "cm.pqcb")) if (
        workdir / "m" / "cm.pqcb").exists() else None
    if qs.get("codebook_file"):
        cb_s = load_file(str(workdir / "s" / qs["codebook_file"]))
        cb_m = load_file(str(workdir / "m" / qm["codebook_file"]))
        assert _tensors_equal(cb_m, cb_s)


def test_streaming_global_cb_stamp_does_not_bypass_missing_render_identity(
    workdir,
):
    from prismaquant.nvfp4_cb_footprint import (
        CBSerializationContext,
        cb_serialization_context_stamp,
    )

    qname = "model.layers.0.self_attn.o_proj"
    mdl = workdir / "model"
    _write_model(
        mdl,
        {f"{qname}.weight": torch.randn(2, 256).to(torch.bfloat16)},
    )
    ap = workdir / "a.json"
    _assign(ap, {
        qname: {"data_type": "nvfp4_cb", "cb_k": 16},
        "__prismaquant__": {
            "cb_serialized_payload": cb_serialization_context_stamp(
                CBSerializationContext.production(),
                formats=["NVFP4_CB_K16"],
            ),
        },
    })
    with pytest.raises(ValueError, match="missing its value-bearing render identity"):
        export_nvfp4_cb_streaming(
            mdl,
            ap,
            workdir / "s",
            {qname: torch.ones(256)},
            device="cpu",
        )


def test_streaming_rejects_unstamped_cb_recipe_by_default(workdir):
    qname = "model.layers.0.self_attn.o_proj"
    mdl = workdir / "model"
    _write_model(
        mdl,
        {f"{qname}.weight": torch.randn(2, 256).to(torch.bfloat16)},
    )
    ap = workdir / "a.json"
    _assign(ap, {qname: "NVFP4_CB_K16"})
    with pytest.raises(ValueError, match="value-bearing render identity"):
        _export_nvfp4_cb_streaming(
            mdl,
            ap,
            workdir / "s",
            {qname: torch.ones(256)},
            device="cpu",
        )


@pytest.mark.parametrize("failure_stage", ["producer", "before_publish"])
def test_stream_writer_removes_owned_temp_before_publish(workdir, failure_stage):
    output = workdir / "model.safetensors"
    writer = _StreamWriter()

    def producer():
        if failure_stage == "producer":
            raise RuntimeError("producer failed")
        return torch.ones(4, dtype=torch.float32)

    def before_publish():
        if failure_stage == "before_publish":
            raise RuntimeError("coverage failed")

    writer.add("value", torch.float32, (4,), producer)
    with pytest.raises(RuntimeError, match="failed"):
        writer.write(output, before_publish=before_publish)
    assert not output.exists()
    assert not (workdir / ".model.safetensors.tmp").exists()


# --- per-expert -> stacked bridging (Hy3 layout) ---------------------------

def _per_expert_model(mdl: Path, E=3, inter=256, hid=256, seed=1):
    torch.manual_seed(seed)
    pe = {}
    for e in range(E):
        pe[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = (
            torch.randn(inter, hid) * 0.3).to(torch.bfloat16)
        pe[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = (
            torch.randn(inter, hid) * 0.3).to(torch.bfloat16)
        pe[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = (
            torch.randn(hid, inter) * 0.3).to(torch.bfloat16)
    pe["model.norm.weight"] = torch.ones(hid, dtype=torch.bfloat16)
    _write_model(mdl, pe, hid)
    # equivalent pre-stacked model for the in-memory reference
    gu = torch.stack([
        torch.cat([pe[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"],
                   pe[f"model.layers.1.mlp.experts.{e}.up_proj.weight"]], 0)
        for e in range(E)])
    dn = torch.stack([pe[f"model.layers.1.mlp.experts.{e}.down_proj.weight"]
                      for e in range(E)])
    st = {"model.layers.1.mlp.experts.gate_up_proj.weight": gu,
          "model.layers.1.mlp.experts.down_proj.weight": dn,
          "model.norm.weight": pe["model.norm.weight"]}
    return st


def test_streaming_per_expert_bridging(workdir):
    E, inter, hid = 3, 256, 256
    st = _per_expert_model(workdir / "pe", E, inter, hid)
    _write_model(workdir / "st", st, hid)
    ap = workdir / "a.json"
    _assign(ap, {
        "model.layers.1.mlp.experts.gate_up_proj": {"data_type": "nvfp4_cb",
                                                    "cb_k": 16},
        "model.layers.1.mlp.experts.down_proj": {"data_type": "nvfp4_cb",
                                                 "cb_k": 16}})
    cw = {"model.layers.1.mlp.experts.gate_up_proj": torch.rand(E, 1, hid)
          + 0.05,
          "model.layers.1.mlp.experts.down_proj": torch.rand(E, 1, inter)
          + 0.05}
    export_nvfp4_cb(workdir / "st", ap, workdir / "m", cw, device="cpu")
    export_nvfp4_cb_streaming(workdir / "pe", ap, workdir / "s", cw,
                              device="cpu")
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    for key in ("model.layers.1.mlp.experts.gate_up_proj.cb_qweight",
                "model.layers.1.mlp.experts.down_proj.cb_qweight"):
        assert torch.equal(tm[key], ts[key]), key


def _lfm_per_expert_model(mdl: Path, layers=(2,), E=2, inter=256, hid=256,
                          seed=17):
    """Tiny LFM checkpoint layout: per-expert w1/w3 fuse to gate_up and w2
    maps to down. Dimensions stay superblock-legal for real CB packing."""
    torch.manual_seed(seed)
    tensors = {"model.embedding_norm.weight":
               torch.ones(hid, dtype=torch.bfloat16)}
    for layer in layers:
        prefix = f"model.layers.{layer}.feed_forward.experts"
        for e in range(E):
            tensors[f"{prefix}.{e}.w1.weight"] = (
                torch.randn(inter, hid) * 0.3).to(torch.bfloat16)
            tensors[f"{prefix}.{e}.w3.weight"] = (
                torch.randn(inter, hid) * 0.3).to(torch.bfloat16)
            tensors[f"{prefix}.{e}.w2.weight"] = (
                torch.randn(hid, inter) * 0.3).to(torch.bfloat16)
    mdl.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "lfm2_moe",
        "architectures": ["Lfm2MoeForCausalLM"],
        "hidden_size": hid,
        "intermediate_size": inter,
        "num_local_experts": E,
    }))
    return tensors


def _lfm_cb_assignment(layer: int) -> dict:
    prefix = f"model.layers.{layer}.feed_forward.experts"
    return {
        f"{prefix}.gate_up_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
        f"{prefix}.down_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    }


def _lfm_col_weights(layer: int, E=2, inter=256, hid=256) -> dict:
    prefix = f"model.layers.{layer}.feed_forward.experts"
    return {
        f"{prefix}.gate_up_proj": torch.rand(E, 1, hid) + 0.05,
        f"{prefix}.down_proj": torch.rand(E, 1, inter) + 0.05,
    }


def test_streaming_lfm_profile_declared_expert_projections(workdir):
    """Regression: streaming must discover LFM's w1/w2/w3 source tensors via
    the profile rather than a gate_proj/up_proj/down_proj-only regex."""
    mdl = workdir / "lfm"
    _lfm_per_expert_model(mdl)
    ap = workdir / "a.json"
    _assign(ap, _lfm_cb_assignment(2))
    cw = _lfm_col_weights(2)

    cm = export_nvfp4_cb(mdl, ap, workdir / "m", cw, device="cpu")
    cs = export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    assert cm == cs
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    assert _tensors_equal(tm, ts)
    prefix = "model.layers.2.feed_forward.experts"
    assert f"{prefix}.gate_up_proj.cb_qweight" in ts
    assert f"{prefix}.down_proj.cb_qweight" in ts
    assert not any(k.startswith(prefix + ".0.") for k in ts)


def _lfm_direct_packed_model(mdl: Path, layers=(2,), E=2, inter=256,
                             hid=256, seed=19):
    """LFM's live ``nn.Parameter`` save layout has no ``.weight`` suffix."""
    torch.manual_seed(seed)
    tensors = {"model.embedding_norm.weight":
               torch.ones(hid, dtype=torch.bfloat16)}
    for layer in layers:
        prefix = f"model.layers.{layer}.feed_forward.experts"
        tensors[f"{prefix}.gate_up_proj"] = (
            torch.randn(E, 2 * inter, hid) * 0.3).to(torch.bfloat16)
        tensors[f"{prefix}.down_proj"] = (
            torch.randn(E, hid, inter) * 0.3).to(torch.bfloat16)
    mdl.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "lfm2_moe",
        "architectures": ["Lfm2MoeForCausalLM"],
        "hidden_size": hid,
        "intermediate_size": inter,
        "num_local_experts": E,
    }))
    return tensors


def test_streaming_lfm_direct_packed_source_is_validated_and_exported(workdir):
    """Regression for packed LFM saves whose expert parameter keys are the
    direct qnames. Both exporters must quantize selected rank-3 tensors while
    preserving direct BF16 banks and marking them ignored."""
    mdl = workdir / "lfm-packed"
    original = _lfm_direct_packed_model(mdl, layers=(2, 3))
    ap = workdir / "a.json"
    assignment = _lfm_cb_assignment(2)
    bf16_prefix = "model.layers.3.feed_forward.experts"
    assignment.update({
        f"{bf16_prefix}.gate_up_proj": "BF16",
        f"{bf16_prefix}.down_proj": "BF16",
    })
    _assign(ap, assignment)
    cw = _lfm_col_weights(2)

    cm = export_nvfp4_cb(mdl, ap, workdir / "m", cw, device="cpu")
    cs = export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    assert cm == cs
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    assert _tensors_equal(tm, ts)
    cb_prefix = "model.layers.2.feed_forward.experts"
    assert f"{cb_prefix}.gate_up_proj.cb_qweight" in ts
    assert f"{cb_prefix}.down_proj.cb_qweight" in ts
    for projection in ("gate_up_proj", "down_proj"):
        key = f"{bf16_prefix}.{projection}"
        assert torch.equal(ts[key], original[key])
    quant_config = json.loads((workdir / "s" / "quant_config.json").read_text())
    assert {f"{bf16_prefix}.gate_up_proj", f"{bf16_prefix}.down_proj"} \
        <= set(quant_config["ignore"])


def test_streaming_lfm_direct_packed_source_rejects_non_3d(workdir):
    mdl = workdir / "lfm-packed-bad"
    tensors = _lfm_direct_packed_model(mdl)
    key = "model.layers.2.feed_forward.experts.gate_up_proj"
    tensors[key] = torch.randn(512, 256).to(torch.bfloat16)
    save_file(tensors, str(mdl / "model.safetensors"))
    ap = workdir / "a.json"
    _assign(ap, _lfm_cb_assignment(2))
    with pytest.raises(ValueError, match="direct packed expert source.*rank-3"):
        export_nvfp4_cb_streaming(
            mdl, ap, workdir / "s", _lfm_col_weights(2), device="cpu")


@pytest.mark.parametrize("streaming", [False, True])
def test_partial_lfm_mixed_export_preserves_bf16_per_expert(
        workdir, streaming):
    """A CB layer may coexist with an explicitly BF16 LFM layer. Only the CB
    parents are stacked; BF16 stays as the original per-expert checkpoint
    tensors expected by LFM's vLLM loader."""
    mdl = workdir / "lfm"
    original = _lfm_per_expert_model(mdl, layers=(2, 3))
    ap = workdir / "a.json"
    assignment = _lfm_cb_assignment(2)
    bf16_prefix = "model.layers.3.feed_forward.experts"
    assignment.update({
        f"{bf16_prefix}.gate_up_proj": {
            "data_type": "bfloat16", "bits": 16},
        f"{bf16_prefix}.down_proj": {
            "data_type": "bfloat16", "bits": 16},
    })
    _assign(ap, assignment)
    out = workdir / ("streaming" if streaming else "memory")
    exporter = export_nvfp4_cb_streaming if streaming else export_nvfp4_cb
    exporter(mdl, ap, out, _lfm_col_weights(2), device="cpu")

    exported = load_file(str(out / "model.safetensors"))
    cb_prefix = "model.layers.2.feed_forward.experts"
    assert f"{cb_prefix}.gate_up_proj.cb_qweight" in exported
    assert f"{cb_prefix}.down_proj.cb_qweight" in exported
    assert not any(k.startswith(cb_prefix + ".0.") for k in exported)
    assert f"{bf16_prefix}.gate_up_proj.weight" not in exported
    assert f"{bf16_prefix}.down_proj.weight" not in exported
    expected_ignore = set()
    for e in range(2):
        for proj in ("w1", "w2", "w3"):
            key = f"{bf16_prefix}.{e}.{proj}.weight"
            assert torch.equal(exported[key], original[key])
            expected_ignore.add(key[:-len(".weight")])
    quant_config = json.loads((out / "quant_config.json").read_text())
    assert expected_ignore <= set(quant_config["ignore"])


# --- bounded peak residency ------------------------------------------------

def test_streaming_peak_residency(workdir, monkeypatch):
    torch.manual_seed(2)
    tens = {"model.norm.weight": torch.ones(256, dtype=torch.bfloat16)}
    # many passthrough tensors so full materialization would be obvious
    for i in range(20):
        tens[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(
            256, dtype=torch.bfloat16)
    tens["model.layers.0.self_attn.q_proj.weight"] = (
        torch.randn(128, 256) * 0.3).to(torch.bfloat16)
    mdl = workdir / "model"
    _write_model(mdl, tens)
    ap = workdir / "a.json"
    _assign(ap, {"model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb",
                                                     "cb_k": 16}})
    cw = {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05}

    live = {"n": 0, "peak": 0}
    orig = _LazySkeleton.load

    def counting(self, name):
        t = orig(self, name)
        live["n"] += 1
        live["peak"] = max(live["peak"], live["n"])
        weakref.finalize(t, lambda: live.__setitem__("n", live["n"] - 1))
        return t
    monkeypatch.setattr(_LazySkeleton, "load", counting)
    export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    # 22 source tensors total; peak resident must be a tiny constant, not ~22.
    assert live["peak"] <= 4, f"peak residency {live['peak']} too high"


# --- fp8-source dequant-on-read (DSv4 ingestion) ---------------------------

def test_streaming_fp8_source_dequant_on_read(workdir):
    from prismaquant.layer_streaming import _dequant_fp8_block_weight
    torch.manual_seed(3)
    out_f, in_f = 256, 256
    w_fp8 = (torch.randn(out_f, in_f) * 0.3).to(torch.float8_e4m3fn)
    scale_inv = (torch.rand(out_f // 128, in_f // 128) + 0.1).float()
    mdl = workdir / "model"
    _write_model(mdl, {
        "model.layers.0.mlp.down_proj.weight": w_fp8,
        "model.layers.0.mlp.down_proj.weight_scale_inv": scale_inv,
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16)},
        hid=256)
    (mdl / "config.json").write_text(json.dumps({
        "hidden_size": 256,
        "quantization_config": {"weight_block_size": [128, 128]}}))
    sk = _LazySkeleton(mdl)
    got = sk.dequant_weight("model.layers.0.mlp.down_proj.weight")
    ref = _dequant_fp8_block_weight(w_fp8, scale_inv, block=(128, 128)).float()
    assert torch.equal(got, ref)


def test_fp8_cb_resident_and_streaming_exports_are_byte_identical(workdir):
    torch.manual_seed(31)
    qname = "model.layers.0.mlp.down_proj"
    w_fp8 = (torch.randn(256, 256) * 0.3).to(torch.float8_e4m3fn)
    scale_inv = (torch.rand(2, 2) + 0.1).float()
    mdl = workdir / "model"
    _write_model(mdl, {
        f"{qname}.weight": w_fp8,
        f"{qname}.weight_scale_inv": scale_inv,
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    })
    (mdl / "config.json").write_text(json.dumps({
        "hidden_size": 256,
        "quantization_config": {"weight_block_size": [128, 128]},
    }))
    ap = workdir / "a.json"
    _assign(ap, {qname: "FP8_CB_K36"})
    cw = {qname: torch.rand(256) + 0.05}

    export_nvfp4_cb(mdl, ap, workdir / "batch", cw, device="cpu")
    export_nvfp4_cb_streaming(
        mdl, ap, workdir / "stream", cw, device="cpu"
    )
    batch = load_file(str(workdir / "batch" / "model.safetensors"))
    stream = load_file(str(workdir / "stream" / "model.safetensors"))
    assert _tensors_equal(batch, stream)


def test_lazy_skeleton_rejects_unscaled_raw_fp8_codes(workdir, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_ALLOW_UNSCALED_FP8", raising=False)
    mdl = workdir / "model"
    _write_model(mdl, {
        "model.layers.0.mlp.down_proj.weight":
            torch.ones(128, 256).to(torch.float8_e4m3fn),
    })
    skeleton = _LazySkeleton(mdl)
    with pytest.raises(RuntimeError, match="has no entry"):
        skeleton.dequant_weight("model.layers.0.mlp.down_proj.weight")


def test_lazy_skeleton_uses_profile_defined_scale_pair(workdir):
    from prismaquant.layer_streaming import _dequant_fp8_block_weight

    torch.manual_seed(32)
    weight_key = "layers.0.attn.q_proj.weight"
    w_fp8 = (torch.randn(256, 256) * 0.3).to(torch.float8_e4m3fn)
    scale = (torch.rand(2, 2) + 0.1).float()
    mdl = workdir / "model"
    _write_model(mdl, {weight_key: w_fp8, "layers.0.attn.q_proj.scale": scale})
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "hidden_size": 256,
        "quantization_config": {"weight_block_size": [128, 128]},
    }))
    skeleton = _LazySkeleton(mdl)
    got = skeleton.dequant_weight(weight_key)
    ref = _dequant_fp8_block_weight(
        w_fp8, scale, block=(128, 128), name=weight_key
    ).to(torch.bfloat16).to(torch.float32)
    assert torch.equal(got, ref)


@pytest.mark.parametrize(
    "exporter",
    [export_nvfp4_cb, export_nvfp4_cb_streaming],
    ids=["batch", "streaming"],
)
def test_fp8_source_uses_profile_defined_scale_pair(workdir, exporter):
    torch.manual_seed(33)
    checkpoint_base = "layers.0.attn.q_proj"
    live_base = "model.layers.0.self_attn.q_proj"
    weight = (torch.randn(256, 256) * 0.3).to(torch.float8_e4m3fn)
    scale = (torch.rand(2, 2) + 0.1).float()
    mdl = workdir / "model"
    _write_model(mdl, {
        checkpoint_base + ".weight": weight,
        checkpoint_base + ".scale": scale,
    })
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "hidden_size": 256,
        "quantization_config": {"weight_block_size": [128, 128]},
    }))
    ap = workdir / "a.json"
    _assign(ap, {live_base: {
        "data_type": "fp8_e4m3", "bits": 8, "group_size": 128,
    }})
    out = workdir / "out"
    exporter(mdl, ap, out, {}, device="cpu")
    tensors = load_file(str(out / "model.safetensors"))
    assert torch.equal(
        tensors[checkpoint_base + ".weight"].view(torch.uint8),
        weight.view(torch.uint8),
    )
    assert torch.equal(tensors[checkpoint_base + ".weight_scale"], scale)
    assert checkpoint_base + ".scale" not in tensors


def test_resident_export_rejects_profile_scaled_per_expert_source(workdir):
    weight_key = "layers.0.ffn.experts.0.w1.weight"
    mdl = workdir / "model"
    _write_model(mdl, {
        weight_key: torch.ones(128, 256).to(torch.float8_e4m3fn),
        "layers.0.ffn.experts.0.w1.scale": torch.ones(1, 2),
    })
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "hidden_size": 256,
        "quantization_config": {"weight_block_size": [128, 128]},
    }))
    qname = "model.layers.0.mlp.experts.gate_up_proj"
    ap = workdir / "a.json"
    _assign(ap, {qname: "FP8_CB_K36"})
    with pytest.raises(ValueError, match="profile-scaled FP8/MXFP4"):
        export_nvfp4_cb(
            mdl,
            ap,
            workdir / "batch",
            {qname: torch.ones(256)},
            device="cpu",
        )


# --- mixed-menu: CB + stock NVFP4 + stock FP8_DYNAMIC + BF16 ----------------

def _mixed_menu_model(mdl: Path):
    """q/k on stock NVFP4 (fused siblings — shared global), gate on stock
    FP8_DYNAMIC, down on CB (nvfp4_cb), up on BF16 passthrough."""
    torch.manual_seed(7)
    tens = {
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.gate_proj.weight":
            (torch.randn(64, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight":
            (torch.randn(256, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight":
            (torch.randn(64, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    }
    _write_model(mdl, tens)
    return tens


_MIXED_ASSIGN = {
    "model.layers.0.self_attn.q_proj": {"data_type": "nv_fp", "bits": 4},
    "model.layers.0.self_attn.k_proj": {"data_type": "nv_fp", "bits": 4},
    "model.layers.0.mlp.gate_proj": {"data_type": "fp8_e4m3", "bits": 8,
                                     "group_size": 0},               # FP8_DYNAMIC
    "model.layers.0.mlp.down_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    "model.layers.0.mlp.up_proj": {"data_type": "bfloat16", "bits": 16},
}


def test_streaming_mixed_menu_byte_identical(workdir):
    mdl = workdir / "model"
    tens = _mixed_menu_model(mdl)
    ap = workdir / "a.json"
    _assign(ap, _MIXED_ASSIGN)
    cw = {"model.layers.0.mlp.down_proj": torch.rand(256) + 0.05}

    cm = export_nvfp4_cb(mdl, ap, workdir / "m", cw, device="cpu")
    cs = export_nvfp4_cb_streaming(mdl, ap, workdir / "s", cw, device="cpu")
    assert dict(cm) == dict(cs)
    assert cs["NVFP4"] == 2 and cs["FP8_E4M3"] == 1 and cs["NVFP4_CB_K16"] == 1

    # (a) header/offsets consistency of the streamed file.
    _assert_offsets_consistent(workdir / "s" / "model.safetensors")

    # (b) every tensor byte-identical to the in-memory exporter (which itself
    #     calls the export_native_compressed packers).
    tm = load_file(str(workdir / "m" / "model.safetensors"))
    ts = load_file(str(workdir / "s" / "model.safetensors"))
    assert _tensors_equal(tm, ts)

    # (b) stock tensor BYTES identical to the packers called directly. q/k are
    #     fused NVFP4 siblings -> they share the max global_real.
    gq = compute_nvfp4_global_real(
        tens["model.layers.0.self_attn.q_proj.weight"].float(), 16).reshape(())
    gk = compute_nvfp4_global_real(
        tens["model.layers.0.self_attn.k_proj.weight"].float(), 16).reshape(())
    shared = torch.stack([gq, gk]).max()
    for leaf in ("q_proj", "k_proj"):
        direct = _quantize_2d(
            tens[f"model.layers.0.self_attn.{leaf}.weight"].float(), "NVFP4",
            nvfp4_global_real_override=shared)
        for suffix, t in direct.items():
            assert torch.equal(
                ts[f"model.layers.0.self_attn.{leaf}.{suffix}"], t), \
                f"{leaf}.{suffix}"
    fp8 = _quantize_2d(
        tens["model.layers.0.mlp.gate_proj.weight"].float(), "FP8_E4M3")
    for suffix, t in fp8.items():
        assert torch.equal(ts[f"model.layers.0.mlp.gate_proj.{suffix}"], t), \
            suffix

    # stock groups have NO "scheme" key; CB groups DO (the dispatch marker).
    qs = json.loads((workdir / "s" / "quant_config.json").read_text())
    stock = _stock_by_scheme(qs)
    assert len(stock) == 2                       # one NVFP4 group, one FP8 group
    assert any(g["format"] == "nvfp4-pack-quantized" for g in stock.values())
    assert any(g["format"] == "float-quantized" for g in stock.values())
    assert "model.layers.0.mlp.up_proj" in qs["ignore"]      # BF16 passthrough

    # (c) stock config vocabulary equals build_quantization_config's (flat model
    #     -> DefaultProfile -> no greedy per-expert catch-all regex, so exact).
    qm = json.loads((workdir / "m" / "quant_config.json").read_text())
    assert qm["config_groups"] == qs["config_groups"]
    prof = detect_profile(str(mdl))
    bqc = build_quantization_config(
        {"model.layers.0.self_attn.q_proj": "NVFP4",
         "model.layers.0.self_attn.k_proj": "NVFP4",
         "model.layers.0.mlp.gate_proj": "FP8_E4M3"}, set(), profile=prof)
    assert _stock_by_scheme(qs) == _stock_by_scheme(
        {"config_groups": bqc["config_groups"]})


def test_streaming_resume_fails_closed_without_producer_identity(workdir):
    # A matching tensor header cannot prove that source/imatrix/codebook bytes
    # or exporter code are unchanged, even at a valid group boundary.
    mdl = workdir / "model"
    _mixed_menu_model(mdl)
    ap = workdir / "a.json"
    _assign(ap, _MIXED_ASSIGN)
    cw = {"model.layers.0.mlp.down_proj": torch.rand(256) + 0.05}
    out = workdir / "s"
    export_nvfp4_cb_streaming(mdl, ap, out, cw, device="cpu")
    ref = (out / "model.safetensors").read_bytes()

    header, data0 = _st_header(out / "model.safetensors")
    # cut mid-weight_scale of the stock NVFP4 q_proj group (after weight_packed,
    # before the group ends) so RESUME must re-enter the stock group.
    wp = header["model.layers.0.self_attn.q_proj.weight_packed"]["data_offsets"]
    ws = header["model.layers.0.self_attn.q_proj.weight_scale"]["data_offsets"]
    cut = data0 + (ws[0] + ws[1]) // 2
    assert wp[1] <= ws[0] <= (cut - data0) < ws[1]
    (out / "model.safetensors").write_bytes(ref[:cut])   # truncate mid-group

    with pytest.raises(RuntimeError, match="output directory .* is not empty"):
        export_nvfp4_cb_streaming(mdl, ap, out, cw, device="cpu")


# --- stock rungs on MoE expert stacks are gated off ------------------------

def test_streaming_rejects_stock_expert_stack(workdir):
    # Per-expert on-disk MoE, packed parent assigned a stock format -> gated.
    E, inter, hid = 3, 256, 256
    _per_expert_model(workdir / "pe", E, inter, hid)
    ap = workdir / "a.json"
    _assign(ap, {"model.layers.1.mlp.experts.gate_up_proj":
                 {"data_type": "nv_fp", "bits": 4}})     # NVFP4 on an expert stack
    cw = {"model.layers.1.mlp.experts.gate_up_proj":
          torch.rand(E, 1, hid) + 0.05}
    with pytest.raises(ValueError, match="expert-stack"):
        export_nvfp4_cb_streaming(workdir / "pe", ap, workdir / "s", cw,
                                  device="cpu")


def test_streaming_rejects_stock_stacked_3d_tensor(workdir):
    # Already-stacked 3-D expert tensor assigned a stock format -> gated too.
    torch.manual_seed(8)
    mdl = workdir / "model"
    _write_model(mdl, {
        "model.layers.0.mlp.experts.gate_up_proj.weight":
            (torch.randn(3, 64, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16)})
    ap = workdir / "a.json"
    _assign(ap, {"model.layers.0.mlp.experts.gate_up_proj":
                 {"data_type": "fp8_e4m3", "bits": 8, "group_size": 0}})
    with pytest.raises(ValueError, match="expert-stack"):
        export_nvfp4_cb_streaming(mdl, ap, workdir / "s", {}, device="cpu")


# --- hy_v3 shared_mlp: stock config target collapses via to_vllm_internal_name

def test_streaming_stock_shared_mlp_vllm_target(workdir):
    torch.manual_seed(9)
    mdl = workdir / "hy"
    mdl.mkdir(parents=True, exist_ok=True)
    save_file({
        "model.layers.5.mlp.shared_mlp.gate_up_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.layers.5.mlp.shared_mlp.down_proj.weight":
            (torch.randn(256, 64) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    }, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(
        json.dumps({"model_type": "hy_v3", "hidden_size": 256}))
    ap = workdir / "a.json"
    # recipe (live) names use `shared_experts`; checkpoint uses `shared_mlp`.
    _assign(ap, {
        "model.layers.5.mlp.shared_experts.gate_up_proj":
            {"data_type": "fp8_e4m3", "bits": 8, "group_size": 0},
        "model.layers.5.mlp.shared_experts.down_proj":
            {"data_type": "nv_fp", "bits": 4}})
    export_nvfp4_cb_streaming(mdl, ap, workdir / "s", {}, device="cpu")

    ts = load_file(str(workdir / "s" / "model.safetensors"))
    # tensors keep the CHECKPOINT name (params live under .shared_mlp.*).
    assert "model.layers.5.mlp.shared_mlp.gate_up_proj.weight" in ts
    assert "model.layers.5.mlp.shared_mlp.down_proj.weight_packed" in ts

    qs = json.loads((workdir / "s" / "quant_config.json").read_text())
    stock_targets = {t for g in qs["config_groups"].values()
                     if "scheme" not in g for t in g["targets"]}
    # config targets COLLAPSE .shared_mlp. -> .mlp. (to_vllm_internal_name),
    # matching vLLM's dispatch prefix and build_quantization_config (28b6862).
    assert "re:^model[.]layers[.]5[.]mlp[.]gate_up_proj$" in stock_targets
    assert "re:^model[.]layers[.]5[.]mlp[.]down_proj$" in stock_targets
    assert not any(".shared_mlp." in t or ".shared_experts." in t
                   for t in stock_targets)
    prof = detect_profile(str(mdl))
    bqc = build_quantization_config(
        {"model.layers.5.mlp.shared_experts.gate_up_proj": "FP8_E4M3",
         "model.layers.5.mlp.shared_experts.down_proj": "NVFP4"},
        set(), profile=prof)
    bqc_targets = {t for g in bqc["config_groups"].values()
                   for t in g["targets"]}
    # the collapsed explicit targets we emit are exactly the ones vLLM's own
    # config builder emits (modulo its greedy per-expert catch-all regex).
    assert stock_targets <= bqc_targets


# --- lazy skeleton: single-file + sharded, metadata without data load ------

def test_lazy_skeleton_metadata(workdir):
    torch.manual_seed(5)
    mdl = workdir / "model"
    _write_model(mdl, {
        "a.weight": torch.randn(16, 32),
        "b.weight": (torch.randn(8, 8)).to(torch.bfloat16)})
    sk = _LazySkeleton(mdl)
    assert "a.weight" in sk and "b.weight" in sk
    assert sk.get_shape("a.weight") == (16, 32)
    assert sk.get_dtype("b.weight") == torch.bfloat16
    assert torch.equal(sk.load("a.weight"), load_file(
        str(mdl / "model.safetensors"))["a.weight"])


# --- DELTA-EXPORT reuse (PRISMAQUANT_EXPORT_REUSE_PRIOR) --------------------
#
# Two dense CB Linears + a BF16 passthrough norm. On a re-allocation most CB
# targets keep their (format, scheme, codebook), so a re-encode reproduces the
# exact bytes — the reuse path byte-copies them from a prior artifact instead.

_REUSE_ASSIGN_A = {
    "model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    "model.layers.0.mlp.down_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
}
# down_proj alone moves K16 -> K20 (a different CB format string) in B.
_REUSE_ASSIGN_B = {
    "model.layers.0.self_attn.q_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    "model.layers.0.mlp.down_proj": {"data_type": "nvfp4_cb", "cb_k": 20},
}


def _reuse_model(mdl: Path, seed: int = 11) -> dict:
    torch.manual_seed(seed)
    tens = {
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight":
            (torch.randn(256, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
    }
    _write_model(mdl, tens)
    return tens


def _reuse_cw() -> dict:
    torch.manual_seed(99)
    return {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05,
            "model.layers.0.mlp.down_proj": torch.rand(256) + 0.05}


def _reshard(src: Path, dst: Path):
    """Re-serialize a single-file artifact as a 2-shard artifact + index."""
    dst.mkdir(parents=True, exist_ok=True)
    tens = load_file(str(src / "model.safetensors"))
    keys = list(tens)
    half = max(1, len(keys) // 2)
    g1 = {k: tens[k] for k in keys[:half]}
    g2 = {k: tens[k] for k in keys[half:]}
    save_file(g1, str(dst / "model-00001-of-00002.safetensors"),
              metadata={"format": "pt"})
    save_file(g2, str(dst / "model-00002-of-00002.safetensors"),
              metadata={"format": "pt"})
    wm = {**{k: "model-00001-of-00002.safetensors" for k in g1},
          **{k: "model-00002-of-00002.safetensors" for k in g2}}
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": wm}))
    shutil.copy(src / "quant_config.json", dst / "quant_config.json")
    qp = json.loads((src / "quant_config.json").read_text())
    if qp.get("codebook_file"):
        shutil.copy(src / qp["codebook_file"], dst / qp["codebook_file"])
    if (src / "config.json").exists():
        shutil.copy(src / "config.json", dst / "config.json")


# (1) reuse disabled == today: byte-identical + no reuse_* keys leak.
def test_reuse_disabled_is_noop(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    cw = _reuse_cw()
    c0 = export_nvfp4_cb_streaming(mdl, ap, workdir / "s0", cw, device="cpu")
    c1 = export_nvfp4_cb_streaming(mdl, ap, workdir / "s1", cw, device="cpu",
                                   reuse_prior=None)
    assert (workdir / "s0" / "model.safetensors").read_bytes() == \
        (workdir / "s1" / "model.safetensors").read_bytes()
    assert dict(c0) == dict(c1)
    assert not any(str(k).startswith("reuse_") for k in c0)
    assert not any(str(k).startswith("reuse_") for k in c1)


# (2) reuse is fail-closed until every copied tensor has an immutable producer
# identity (source bytes + imatrix + codebook + scheme + exporter ABI).
def test_reuse_prior_is_blocked_before_any_output(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    cw = _reuse_cw()
    prior = workdir / "prior"
    export_nvfp4_cb_streaming(mdl, ap, prior, cw, device="cpu")   # fresh
    out = workdir / "delta"
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(
            mdl, ap, out, cw, device="cpu", reuse_prior=prior, reuse_verify=2
        )
    assert not (out / "model.safetensors").exists()


# (3) changed-format target re-encodes; unchanged one still copies.
def test_reuse_changed_format_is_still_blocked_globally(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    cw = _reuse_cw()
    apA = workdir / "a.json"
    _assign(apA, _REUSE_ASSIGN_A)
    prior = workdir / "prior"
    export_nvfp4_cb_streaming(mdl, apA, prior, cw, device="cpu")
    apB = workdir / "b.json"
    _assign(apB, _REUSE_ASSIGN_B)
    fresh_b = workdir / "freshB"
    export_nvfp4_cb_streaming(mdl, apB, fresh_b, cw, device="cpu")   # reference
    out = workdir / "delta"
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(
            mdl, apB, out, cw, device="cpu", reuse_prior=prior, reuse_verify=5
        )


# (4) codebook byte-mismatch makes every CB target on that group ineligible.
def test_reuse_codebook_mismatch_does_not_reenable_copy(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    cw = _reuse_cw()
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    prior = workdir / "prior"
    export_nvfp4_cb_streaming(mdl, ap, prior, cw, device="cpu")
    qp = json.loads((prior / "quant_config.json").read_text())
    cbf = prior / qp["codebook_file"]
    cbt = load_file(str(cbf))
    cbt = {k: (v + 1.0).to(v.dtype).contiguous() for k, v in cbt.items()}
    save_file(cbt, str(cbf), metadata={"format": "pt"})   # perturb codebook
    out = workdir / "delta"
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(
            mdl, ap, out, cw, device="cpu", reuse_prior=prior
        )


# (5) even a positive sample count cannot turn an unbound prior into proof.
def test_reuse_sampling_cannot_override_missing_identity(workdir):
    torch.manual_seed(13)
    mdl = workdir / "model"
    _write_model(mdl, {
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, 256) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(256, dtype=torch.bfloat16)})
    cw = {"model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05}
    ap = workdir / "a.json"
    _assign(ap, {"model.layers.0.self_attn.q_proj":
                 {"data_type": "nvfp4_cb", "cb_k": 16}})
    prior = workdir / "prior"
    export_nvfp4_cb_streaming(mdl, ap, prior, cw, device="cpu")
    header, data0 = _st_header(prior / "model.safetensors")
    off = header["model.layers.0.self_attn.q_proj.cb_qweight"]["data_offsets"]
    raw = bytearray((prior / "model.safetensors").read_bytes())
    raw[data0 + off[0]] ^= 0xFF                   # flip one packed-code byte
    (prior / "model.safetensors").write_bytes(bytes(raw))
    out = workdir / "delta"
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(mdl, ap, out, cw, device="cpu",
                                  reuse_prior=prior, reuse_verify=1)
    assert not (out / "model.safetensors").exists()   # nothing shipped


# (6) sharded prior artifact (index.json + model-XXXXX-of-XXXXX) read path.
def test_reuse_sharded_prior_is_blocked_before_index_read(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    cw = _reuse_cw()
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    single = workdir / "prior_single"
    export_nvfp4_cb_streaming(mdl, ap, single, cw, device="cpu")
    sharded = workdir / "prior_sharded"
    _reshard(single, sharded)
    out = workdir / "delta"
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(
            mdl, ap, out, cw, device="cpu", reuse_prior=sharded, reuse_verify=2
        )


# (7) main()/CLI env fallback (PRISMAQUANT_EXPORT_REUSE_PRIOR) — the exact path
# run-pipeline.sh drives tonight.
def test_reuse_main_env_fallback(workdir, monkeypatch):
    import pickle
    import prismaquant.gpu_guard as gpu_guard

    # The production CLI is intentionally GPU-only. This test exercises only
    # argument/environment plumbing on tiny CPU tensors, so isolate that policy
    # guard instead of weakening it in production code.
    monkeypatch.setattr(
        gpu_guard,
        "require_cuda_hot_path",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    mdl = workdir / "model"
    _reuse_model(mdl)
    cwp = workdir / "cw.pkl"
    with open(cwp, "wb") as f:
        pickle.dump(_reuse_cw(), f)
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    base = ["--model-dir", str(mdl), "--layer-config", str(ap),
            "--col-weights", str(cwp), "--device", "cpu",
            "--allow-unstamped-research"]
    prior = workdir / "prior"
    monkeypatch.delenv("PRISMAQUANT_EXPORT_REUSE_PRIOR", raising=False)
    _cb_stream_main(base + ["--out", str(prior)])                  # fresh
    out = workdir / "delta"
    monkeypatch.setenv("PRISMAQUANT_EXPORT_REUSE_PRIOR", str(prior))
    monkeypatch.setenv("PRISMAQUANT_EXPORT_REUSE_VERIFY", "2")
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        _cb_stream_main(base + ["--out", str(out)])


# (8) reuse is rejected before an existing output can be mistaken for a bound
# resume journal.
def test_reuse_does_not_bypass_resume_gate(workdir):
    mdl = workdir / "model"
    _reuse_model(mdl)
    cw = _reuse_cw()
    ap = workdir / "a.json"
    _assign(ap, _REUSE_ASSIGN_A)
    prior = workdir / "prior"
    export_nvfp4_cb_streaming(mdl, ap, prior, cw, device="cpu")
    out = workdir / "delta"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"unbound-partial")
    with pytest.raises(RuntimeError, match="DELTA-EXPORT reuse is disabled"):
        export_nvfp4_cb_streaming(
            mdl, ap, out, cw, device="cpu", reuse_prior=prior, reuse_verify=1
        )
