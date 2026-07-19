"""Correctness gate for the CUDA FP8_CB decode-GEMV (prototype ii) against the
Triton decode-GEMM it replaces and the fp64 reconstruct reference.

Needs nvcc (JIT build) — runs in the serving container, skips in the build
venv:

  docker run --rm --gpus all -v /home/rob/prismaquant:/repo \\
    -v /home/rob/dq-runs/nvfp4-cb-phase0/serve:/artifacts \\
    --entrypoint bash vllm-node:latest -c \\
    'PYTHONPATH=/repo:/repo/plugins/vllm_prismaquant python3 -m pytest \\
     /repo/plugins/vllm_prismaquant/tests/test_cuda_gemv.py -v'

The KL-preservation contract: identical weight rounding (bf16(val*scale)),
bit-exact activation QDQ, fp32 accumulation — only summation order may differ
from Triton's tl.dot, so CUDA-vs-Triton tolerances are reassociation-level.
"""
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

codec = pytest.importorskip(
    "vllm_prismaquant.codec",
    reason="vllm_prismaquant plugin not importable")
kernels = pytest.importorskip("vllm_prismaquant.kernels")
from vllm_prismaquant.cuda_ext import get_ext  # noqa: E402

ext = get_ext()
if ext is None:
    pytest.skip("CUDA extension unavailable (no nvcc?)",
                allow_module_level=True)

cb_decode_linear = kernels.cb_decode_linear
DEV = "cuda"
ART = "fp8cb_k44"
PICK = ["model.layers.5.mlp.down_proj", "model.layers.5.mlp.gate_proj",
        "model.layers.0.self_attn.q_proj"]
_REF_REL = 1e-2        # vs fp64 reconstruct (matches test_cb_kernels gate)


def _assert_triton_close(y_cuda, y_triton, tag):
    """CUDA vs Triton: identical weights + inputs, fp32 accumulation — only
    summation ORDER differs, so the bf16 outputs may differ by at most one
    output-rounding step (verified live: fp32 truth lands mid-ULP and the two
    round to the adjacent bf16 neighbours). Elementwise: |Δ| <= 1 bf16 ULP
    (7 mantissa bits -> 2^-7 relative) + tiny abs; plus a norm backstop."""
    a, b = y_cuda.float(), y_triton.float()
    d = (a - b).abs()
    tol = torch.maximum(a.abs(), b.abs()) * 2.0 ** -7 + 1e-5
    nbad = int((d > tol).sum())
    assert nbad == 0, (
        f"{tag}: {nbad} elements beyond 1 bf16 output ULP "
        f"(max Δ {d.max():.3e} vs tol {tol.flatten()[d.argmax()]:.3e})")
    rel = d.norm() / b.norm().clamp_min(1e-6)
    assert rel <= 1e-3, f"{tag}: norm backstop rel {rel:.3e}"


def _serve_root() -> Path:
    for p in (os.environ.get("CB_SERVE_ROOT"),
              "/home/rob/dq-runs/nvfp4-cb-phase0/serve", "/artifacts"):
        if p and (Path(p) / ART / "model.safetensors").exists():
            return Path(p)
    pytest.skip("CB serve artifacts (fp8cb_k44) not found")


def _prep(qname):
    d = _serve_root() / ART
    cfg = json.loads((d / "config.json").read_text())["quantization_config"]
    tensors = load_file(str(d / "model.safetensors"))
    codebooks = load_file(str(d / cfg.get("codebook_file", "cb_codebooks.pqcb")))
    q2s = {}
    for g in cfg["config_groups"].values():
        for t in g["targets"]:
            q2s[t] = g["scheme"]
    if qname not in q2s:
        pytest.skip(f"{qname} not a CB target in {ART}")
    sch = q2s[qname]
    assert sch["grid"] == "fp8"
    packed = tensors[qname + ".cb_qweight"].to(DEV)
    N = packed.shape[0]
    K = (packed.shape[1] // sch["type_size"]) * codec.SUPERBLOCK
    ws = tensors[qname + ".weight_scale"].to(DEV).float().reshape(-1)
    ref = sch["codebook_ref"]
    names = ref if isinstance(ref, list) else [ref]
    subs = [codebooks[n].to(DEV).float() for n in names]
    cb_flat = codec.build_flat_codebook(subs)
    return dict(qwp=codec.pad_qweight(packed), cb_flat=cb_flat,
                cb8=cb_flat.to(torch.float8_e4m3fn).view(
                    torch.uint8).contiguous(),
                row_off=torch.zeros(N, dtype=torch.int32, device=DEV),
                N=N, K=K, k=int(sch["k"]), n_sub=int(sch["n_sub"]),
                ts=int(sch["type_size"]), ws=ws)


def _synth(k, N=96, K=768, seed=0):
    """Synthetic rung: random packed bytes + a random codebook SNAPPED to the
    e4m3 grid (the FP8_CB contract — and what makes the byte-codebook gather
    value-identical to the bf16 one), at realistic weight magnitudes."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ts = 4 * k
    n_sb = K // 256
    packed = torch.randint(0, 256, (N, n_sb * ts), generator=g,
                           dtype=torch.uint8).to(DEV)
    sub_w = k // 4
    subs = [(torch.randn(1 << sub_w, 2, generator=g) * 4.0)
            .to(torch.float8_e4m3fn).float().to(DEV) for _ in range(4)]
    ws = (torch.rand(N, generator=g).to(DEV) + 0.5) * 0.02
    cb_flat = codec.build_flat_codebook(subs)
    return dict(qwp=codec.pad_qweight(packed), cb_flat=cb_flat,
                cb8=cb_flat.to(torch.float8_e4m3fn).view(
                    torch.uint8).contiguous(),
                row_off=torch.zeros(N, dtype=torch.int32, device=DEV),
                N=N, K=K, k=k, n_sub=4, ts=ts, ws=ws.float())


def _triton_y(p, xq):
    return cb_decode_linear(
        xq, p["qwp"], p["cb_flat"], p["row_off"], p["ws"],
        torch.zeros(1, device=DEV), N=p["N"], K=p["K"], k_bits=p["k"],
        n_sub=p["n_sub"], type_size=p["ts"], is_fp4=False)


def _cuda_y(p, xq):
    return ext.cb_gemv_fp8(xq, p["qwp"], p["cb8"], p["row_off"], p["ws"],
                           p["N"], p["K"], p["k"], p["n_sub"], p["ts"], False)


def _rel(a, b):
    return ((a.float() - b.float()).norm()
            / b.float().norm().clamp_min(1e-6)).item()


# --------------------------------------------------------------------------- #
def test_qdq_bitexact():
    """Bit-exact to codec.fp8_dynamic_act_qdq across many draws. This caught
    two real 1-ULP traps: __nv_cvt_float_to_fp8 double-rounding (fixed by the
    c10 conversion port) and torch's tensor/scalar division being a
    reciprocal MULTIPLY (fixed by matching it in the scale)."""
    torch.manual_seed(0)
    for M, K in ((1, 1024), (7, 5120), (16, 17408)):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        x[0, :4] = 0.0                        # exercise tiny-amax clamp path
        got = ext.fp8_act_qdq(x)
        want = codec.fp8_dynamic_act_qdq(x)
        assert torch.equal(got.view(torch.uint16), want.view(torch.uint16)), (
            f"QDQ not bit-exact at M={M} K={K}")
    for seed in range(24):                    # tie-hunting: many draws/scales
        torch.manual_seed(seed)
        x = (torch.randn(4, 2048, dtype=torch.bfloat16, device=DEV)
             * (10.0 ** (seed % 5 - 2)))
        got = ext.fp8_act_qdq(x)
        want = codec.fp8_dynamic_act_qdq(x)
        neq = int((got.view(torch.uint16) != want.view(torch.uint16)).sum())
        assert neq == 0, f"QDQ not bit-exact at seed={seed}: {neq} mismatches"


def test_qdq_min_scale_clamp():
    x = torch.full((2, 512), 1e-9, dtype=torch.bfloat16, device=DEV)
    got = ext.fp8_act_qdq(x)
    want = codec.fp8_dynamic_act_qdq(x)
    assert torch.equal(got.view(torch.uint16), want.view(torch.uint16))


@pytest.mark.parametrize("qname", PICK)
@pytest.mark.parametrize("M", [1, 3, 16])
def test_gemv_matches_triton_real_artifact(qname, M):
    p = _prep(qname)
    torch.manual_seed(0)
    x = torch.randn(M, p["K"], dtype=torch.bfloat16, device=DEV)
    xq = codec.fp8_dynamic_act_qdq(x)
    _assert_triton_close(_cuda_y(p, xq), _triton_y(p, xq),
                         f"{qname} M={M}")


@pytest.mark.parametrize("k", [36, 40, 44, 48])
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
def test_gemv_matches_triton_all_rungs(k, M):
    p = _synth(k, seed=k)
    torch.manual_seed(k)
    x = torch.randn(M, p["K"], dtype=torch.bfloat16, device=DEV)
    xq = codec.fp8_dynamic_act_qdq(x)
    _assert_triton_close(_cuda_y(p, xq), _triton_y(p, xq), f"k={k} M={M}")


def test_gemv_matches_reference():
    """vs fp64 dequant reference on the real artifact (same gate style as
    test_cb_kernels.test_gemm_matches_reconstruct)."""
    try:
        from prismaquant.nvfp4_cb_formats import (
            nvfp4_cb_reconstruct, nvfp4_cb_unpack)
    except Exception:
        pytest.skip("prismaquant not importable for the reference")
    qname = PICK[0]
    p = _prep(qname)
    d = _serve_root() / ART
    cfg = json.loads((d / "config.json").read_text())["quantization_config"]
    tensors = load_file(str(d / "model.safetensors"))
    codebooks = load_file(str(d / cfg.get("codebook_file", "cb_codebooks.pqcb")))
    sch = next(g["scheme"] for g in cfg["config_groups"].values()
               if qname in g["targets"])
    ref = sch["codebook_ref"]
    subs = [codebooks[n].to(DEV).float()
            for n in (ref if isinstance(ref, list) else [ref])]
    packed = tensors[qname + ".cb_qweight"].to(DEV)
    fields = nvfp4_cb_unpack(packed, p["k"], "fp8", "product",
                             (p["N"], p["K"]), codebook=subs,
                             scales=p["ws"].reshape(-1, 1))
    w_ref = nvfp4_cb_reconstruct(fields, p["k"], grid="fp8", mode="product",
                                 codebook=subs).to(torch.bfloat16)
    torch.manual_seed(1)
    x = torch.randn(4, p["K"], dtype=torch.bfloat16, device=DEV)
    xq = codec.fp8_dynamic_act_qdq(x)
    y = _cuda_y(p, xq)
    y_ref = xq.float() @ w_ref.float().t()
    r = _rel(y, y_ref)
    assert r <= _REF_REL, f"CUDA vs reconstruct rel {r:.3e}"


def test_fused_row_offset_two_roles():
    """Two roles, distinct codebooks, concatenated rows (the qkv/gate_up
    fusion mechanism) — CUDA must honor cb_row_offset exactly as Triton."""
    pa, pb = _synth(44, N=64, K=512, seed=1), _synth(44, N=32, K=512, seed=2)
    qwp = codec.pad_qweight(torch.cat(
        [pa["qwp"][:, :-8], pb["qwp"][:, :-8]], dim=0))
    cb_flat = torch.cat([pa["cb_flat"], pb["cb_flat"]])
    off = torch.cat([
        torch.zeros(64, dtype=torch.int32, device=DEV),
        torch.full((32,), pa["cb_flat"].numel(), dtype=torch.int32,
                   device=DEV)])
    ws = torch.cat([pa["ws"], pb["ws"]])
    p = dict(qwp=qwp, cb_flat=cb_flat,
             cb8=cb_flat.to(torch.float8_e4m3fn).view(torch.uint8).contiguous(),
             row_off=off, N=96, K=512, k=44, n_sub=4, ts=176, ws=ws)
    torch.manual_seed(3)
    x = torch.randn(2, 512, dtype=torch.bfloat16, device=DEV)
    xq = codec.fp8_dynamic_act_qdq(x)
    _assert_triton_close(_cuda_y(p, xq), _triton_y(p, xq), "fused row-offset")


@pytest.mark.parametrize("k", [36, 40, 44, 48])
def test_cuda_expand_bitexact_vs_triton(k):
    """The CUDA transient expander must produce byte-identical tiles to the
    Triton expand_cb_to_fp8 (which is itself pinned to the bf16-expand+cast
    reference)."""
    from vllm_prismaquant.expand import expand_cb_to_fp8
    p = _synth(k, N=96, K=768, seed=100 + k)
    ref = expand_cb_to_fp8(p["qwp"], p["cb8"], p["row_off"],
                           p["N"], p["K"], p["k"], 4, p["ts"])
    got = ext.cb_expand_fp8(p["qwp"], p["cb8"], p["row_off"],
                            p["N"], p["K"], p["k"], 4, p["ts"])
    assert got.dtype == torch.float8_e4m3fn
    assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8)), (
        f"k={k}: CUDA expand bytes != Triton expand bytes")


def test_cuda_expand_bitexact_real_artifact():
    from vllm_prismaquant.expand import expand_cb_to_fp8
    p = _prep(PICK[0])
    ref = expand_cb_to_fp8(p["qwp"], p["cb8"], p["row_off"],
                           p["N"], p["K"], p["k"], 4, p["ts"])
    got = ext.cb_expand_fp8(p["qwp"], p["cb8"], p["row_off"],
                            p["N"], p["K"], p["k"], 4, p["ts"])
    assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))


def test_full_op_raw_x_matches_triton_path():
    """The registered custom op (raw x in, QDQ fused) equals the Triton path
    (torch QDQ then decode-GEMM) — the exact serving-dispatch equivalence."""
    from vllm_prismaquant.ops import cb_gemv_fp8 as op
    p = _prep(PICK[1])
    torch.manual_seed(2)
    x = torch.randn(1, p["K"], dtype=torch.bfloat16, device=DEV)
    y_op = op(x, p["qwp"], p["cb8"], p["row_off"], p["ws"],
              p["N"], p["K"], p["k"], p["n_sub"], p["ts"])
    y_t = _triton_y(p, codec.fp8_dynamic_act_qdq(x))
    _assert_triton_close(y_op, y_t, "full-op raw-x")


def test_moe_grouped_gemv_matches_loop_numerics():
    """The grouped MoE decode path (one launch over routed pairs) must match
    the per-expert loop's numerics chain: per-token fp8 QDQ of input AND
    intermediate, W = bf16(val*scale), bf16 GEMMs (fp32 accum), bf16-rounded
    router weight, expert-ascending per-add bf16 combine."""
    torch.manual_seed(21)
    E, out13, hidden, inter, k = 8, 64, 512, 32, 44
    ts = 4 * k
    assert out13 == 2 * inter
    g = torch.Generator(device="cpu").manual_seed(21)
    w13 = torch.randint(0, 256, (E, out13, (hidden // 256) * ts),
                        generator=g, dtype=torch.uint8).to(DEV)
    # down: in=inter=32 < 256 superblock -> use in=256 for the test
    inter2 = 256
    w2 = torch.randint(0, 256, (E, hidden, (inter2 // 256) * ts),
                       generator=g, dtype=torch.uint8).to(DEV)
    subs = [(torch.randn(1 << (k // 4), 2, generator=g) * 4.0)
            .to(torch.float8_e4m3fn).float().to(DEV) for _ in range(4)]
    cb = codec.build_flat_codebook(subs)
    cb8 = cb.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    s13 = ((torch.rand(E, out13, generator=g) + 0.5) * 0.02).float().to(DEV)
    s2 = ((torch.rand(E, hidden, generator=g) + 0.5) * 0.02).float().to(DEV)

    T, topk = 2, 3
    x = torch.randn(T, hidden, dtype=torch.bfloat16, device=DEV)
    topk_ids = torch.stack([torch.randperm(E, generator=g)[:topk]
                            for _ in range(T)]).to(DEV)
    topk_w = torch.rand(T, topk, generator=g).float().to(DEV)

    def expand_expert(stack, scales, e, in_f):
        qwp = codec.pad_qweight(stack[e].contiguous())
        off = torch.zeros(stack.shape[1], dtype=torch.int32, device=DEV)
        val = ext.cb_expand_fp8(qwp, cb8, off, stack.shape[1], in_f,
                                k, 4, ts).float()
        return (val * scales[e][:, None]).to(torch.bfloat16)

    # ---- reference: the per-expert loop chain, expert-ascending ----
    out_ref = torch.zeros(T, hidden, dtype=torch.bfloat16, device=DEV)
    for e in range(E):
        sel = (topk_ids == e)
        if not bool(sel.any()):
            continue
        tok_idx, slot = torch.where(sel)
        xe = codec.fp8_dynamic_act_qdq(x[tok_idx]).to(torch.bfloat16)
        W13 = expand_expert(w13, s13, e, hidden)
        gu = torch.nn.functional.linear(xe, W13)
        gate, up = gu.chunk(2, dim=-1)
        a = (torch.nn.functional.silu(gate) * up)
        # widen intermediate to the 256-superblock width for down
        a256 = torch.zeros(a.shape[0], inter2, dtype=a.dtype, device=DEV)
        a256[:, :inter] = a
        aq = codec.fp8_dynamic_act_qdq(a256).to(torch.bfloat16)
        W2 = expand_expert(w2, s2, e, inter2)
        oe = torch.nn.functional.linear(aq, W2)
        oe = oe * topk_w[tok_idx, slot][:, None].to(oe.dtype)
        out_ref.index_add_(0, tok_idx, oe)

    # ---- grouped kernels (mirrors moe._apply_grouped_decode) ----
    ids_sorted, order = torch.sort(topk_ids.to(torch.int32), dim=-1)
    w_sorted = torch.gather(topk_w, -1, order.to(torch.int64))
    pair_expert = ids_sorted.reshape(-1).contiguous()
    pair_xrow = torch.arange(T, device=DEV,
                             dtype=torch.int32).repeat_interleave(topk)
    pair_w = (w_sorted.reshape(-1).to(torch.bfloat16)
              .to(torch.float32).contiguous())
    tok_start = torch.arange(T + 1, device=DEV, dtype=torch.int32) * topk
    xq = ext.fp8_act_qdq(x)
    gu = ext.cb_moe_gemv_fp8(xq, w13, cb8, s13, pair_expert, pair_xrow,
                             k, 4, ts)
    gate, up = gu.chunk(2, dim=-1)
    a = torch.nn.functional.silu(gate) * up
    a256 = torch.zeros(a.shape[0], inter2, dtype=a.dtype, device=DEV)
    a256[:, :inter] = a
    aq = ext.fp8_act_qdq(a256)
    pair_self = torch.arange(pair_expert.numel(), device=DEV,
                             dtype=torch.int32)
    y_down = ext.cb_moe_gemv_fp8(aq, w2, cb8, s2, pair_expert, pair_self,
                                 k, 4, ts)
    out_got = ext.cb_moe_combine(y_down, pair_w, tok_start, T)

    _assert_triton_close(out_got, out_ref, "moe grouped vs loop")
