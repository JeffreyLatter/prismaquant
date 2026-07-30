# Verbatim export branches removed from prismaquant/export_native_compressed.py
# on 2026-07-30 (re-vet R25). Three sites: the per-layer FP16 snapshot capture,
# the per-Linear deferred-pack branch, and the post-loop refine+finalize block
# reproduced below, plus _finalize_compute_only (in ./prismaquant/).
# NOT importable; a record for reconstruction.

# ---- site 1: per-layer snapshot capture (immediately before the
#              `for sub_name, mod in layer_mod.named_modules():` loop) ----
#         # #12 Block-output match deferred-pack list. Per-layer scope.
#         _BLOCK_COMPUTE_PENDING: list[dict] = []
#         # Capture FP16 snapshots of the layer's standard block Linears
#         # so we can run a reference (pre-quantization) forward pass for
#         # block-output match. Cheap: a layer's q/k/v/o + gate/up/down at
#         # FP32 ≈ 64-128 MB.
#         _FP16_BLOCK_SNAPSHOTS: dict[str, torch.Tensor] = {}
#         if os.environ.get("PRISMAQUANT_BLOCK_OUTPUT_MATCH", "1") != "0":
#             for _sn, _m in layer_mod.named_modules():
#                 if not isinstance(_m, nn.Linear):
#                     continue
#                 _leaf = _sn.rsplit(".", 1)[-1] if _sn else ""
#                 if _leaf in (
#                     "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
#                     "gate_proj", "up_proj", "down_proj",
#                 ):
#                     _FP16_BLOCK_SNAPSHOTS[_sn] = _m.weight.detach().clone()

# ---- site 3: post-loop refinement + finalize ----

        # 3c'. Block-output match (#12). When PRISMAQUANT_BLOCK_OUTPUT_MATCH=1
        # the per-Linear loop above deferred packing for standard block
        # Linears (q/k/v/o, gate/up/down). Now run greedy refinement of
        # per-Linear scale perturbations against an FP16 reference forward,
        # then finalize the pack. Skipped if no compute-only entries
        # accumulated (env flag off, or no eligible Linears in this layer).
        if _BLOCK_COMPUTE_PENDING:
            try:
                from .block_output_match import (
                    block_output_mse,
                    make_attention_block_spec, make_mlp_block_spec,
                    refine_block_scales,
                )
                # Group pending entries by sub_leaf so we can index
                # them when applying refined scales. Also recover
                # the FP16 reference weights from _FP16_BLOCK_SNAPSHOTS.
                pending_by_sub = {p["sub_leaf"]: p
                                  for p in _BLOCK_COMPUTE_PENDING}

                # Use a small calibration input drawn from the cached
                # activation of q_proj (its input == post-norm of the
                # residual stream, which is the natural attn-block
                # input). For MLP block, gate_proj input is the same
                # post-norm residual after attention. If activations
                # aren't cached for this layer, skip refinement —
                # there's no reference signal.
                cal_input_attn = None
                cal_input_mlp = None
                if _CACHED_ACTIVATIONS is not None:
                    # cached keys are recipe_keys; pull from any
                    # block-Linear that's pending so naming variation
                    # across profiles still works.
                    for p in _BLOCK_COMPUTE_PENDING:
                        if p["sub_leaf"] in ("q_proj",) and cal_input_attn is None:
                            cal_input_attn = _CACHED_ACTIVATIONS.get(
                                profile.live_to_recipe_name(p["full"]))
                        if p["sub_leaf"] in ("gate_proj",) and cal_input_mlp is None:
                            cal_input_mlp = _CACHED_ACTIVATIONS.get(
                                profile.live_to_recipe_name(p["full"]))

                # Run refinement for each block we have a cal input for.
                # Candidates are simple multiplicative perturbations of
                # the current dequantized weight; refine_block_scales
                # picks the per-Linear scale that minimizes block MSE.
                cands = [torch.tensor(s) for s in (0.95, 1.0, 1.05)]

                block_logs: list[str] = []

                def _apply_refined_scales(label: str, spec_factory, cal_input):
                    if cal_input is None:
                        block_logs.append(f"{label}=no_cal")
                        return
                    ref_spec = spec_factory(layer_mod, layer_qname)
                    if ref_spec is None:
                        block_logs.append(f"{label}=no_spec")
                        return
                    # Cap the cal_input to a small batch to keep refinement fast.
                    ci = cal_input.to(layer_mod.input_layernorm.weight.device
                                      if hasattr(layer_mod, "input_layernorm")
                                      else next(iter(layer_mod.parameters())).device)
                    if ci.dim() == 2:
                        ci = ci[:32]
                    elif ci.dim() == 3:
                        ci = ci[:8]
                    run_dtype = next(
                        (p["mod"].weight.dtype for p in _BLOCK_COMPUTE_PENDING
                         if p["mod"].weight.dtype.is_floating_point),
                        torch.float32,
                    )
                    ci_run = ci.to(dtype=run_dtype)
                    # Full-precision reference first, while the live layer
                    # still holds original weights. Earlier code built the
                    # reference and candidates from the same live weights,
                    # making scale=1.0 perfect and the pass a silent no-op.
                    with torch.no_grad():
                        ref = ref_spec.forward_fn(ci_run).float().clone()

                    touched: list[dict] = []
                    for ln in ref_spec.linears:
                        p = pending_by_sub.get(ln)
                        if p is None:
                            continue
                        mod = p["mod"]
                        touched.append(p)
                        q_weight = p["compute_dict"]["_w_dq"].to(
                            device=mod.weight.device, dtype=mod.weight.dtype)
                        mod.weight.data.copy_(q_weight)

                    if not touched:
                        block_logs.append(f"{label}=no_pending")
                        return

                    try:
                        spec = spec_factory(layer_mod, layer_qname)
                        if spec is None:
                            block_logs.append(f"{label}=lost_spec")
                            return
                        candidates = {
                            ln: cands for ln in spec.linears
                            if ln in pending_by_sub
                        }
                        before = block_output_mse(spec, ci_run, ref)
                        final = refine_block_scales(
                            spec, ci_run, ref, candidates, max_passes=2)
                        n_changed = 0
                        n_eval = 0
                        for ln in spec.linears:
                            p = pending_by_sub.get(ln)
                            if p is None:
                                continue
                            n_eval += len(cands) * 2
                            s = float(spec.scale_getter(ln))
                            if abs(s - 1.0) < 1e-8:
                                continue
                            p["compute_dict"]["_w_dq"] = (
                                p["compute_dict"]["_w_dq"] * s)
                            n_changed += 1
                        block_logs.append(
                            f"{label}=spec evals={n_eval} "
                            f"changed={n_changed} "
                            f"mse={before:.3e}->{final:.3e}")
                    finally:
                        for p in touched:
                            snap = _FP16_BLOCK_SNAPSHOTS.get(p["sub_name"])
                            if snap is not None:
                                p["mod"].weight.data.copy_(
                                    snap.to(device=p["mod"].weight.device,
                                            dtype=p["mod"].weight.dtype))

                _apply_refined_scales(
                    "attn", make_attention_block_spec, cal_input_attn)
                _apply_refined_scales(
                    "mlp", make_mlp_block_spec, cal_input_mlp)
                print(
                    f"[block-output-match] {layer_qname}: "
                    f"pending={len(_BLOCK_COMPUTE_PENDING)} "
                    + " ".join(block_logs),
                    flush=True,
                )

            except Exception as e:
                print(f"[block-output-match] WARN refinement failed for "
                      f"{layer_qname}: {e}", flush=True)

            # Finalize the pack for every pending Linear (refined or not).
            for p in _BLOCK_COMPUTE_PENDING:
                compressed = _finalize_compute_only(p["compute_dict"])
                emit_full = p["emit_full"]
                for suffix, t in compressed.items():
                    out[f"{emit_full}.{suffix}"] = t.cpu()
                if p["mod"].bias is not None:
                    out[f"{emit_full}.bias"], _ = _passthrough_tensor(
                        f"{p['full']}.bias", p["mod"].bias,
                        source_dtype_by_name)
                hist[("linear", "NVFP4_block_match")] += 1
                covered.add(p["full"])

            del _BLOCK_COMPUTE_PENDING, _FP16_BLOCK_SNAPSHOTS


# ---- _finalize_compute_only (removed from export_native_compressed.py) ----

def _finalize_compute_only(compute_dict: dict, *,
                           weight_override: torch.Tensor | None = None
                           ) -> dict[str, torch.Tensor]:
    """Pack a compute_only result from `_quantize_2d` into the final
    on-disk tensor dict. When `weight_override` is supplied (e.g. after
    block-output match modified the dequantized weight), pack that
    instead of the original `_w_dq`.

    Currently only NVFP4 is supported in compute_only mode. Other
    formats fall through to a clear error so a misuse fails loudly
    rather than silently silently corrupting the artifact.
    """
    fmt = compute_dict.get("_fmt")
    if fmt != "NVFP4":
        raise ValueError(
            f"_finalize_compute_only: only NVFP4 is supported "
            f"(got fmt={fmt}). Other formats should not be in "
            f"compute_only mode.")
    w = compute_dict["_w_dq"] if weight_override is None else weight_override
    nvfp4_global_real = compute_dict["_nvfp4_global_real"]
    input_scale = compute_dict["_input_scale"]

    wp, ws, wg = quantize_dequantize_nvfp4(
        w, group_size=16,
        global_real_override=nvfp4_global_real,
    )
    return {
        "weight_packed": wp,
        "weight_scale": ws,
        "weight_global_scale": wg,
        "input_global_scale": torch.tensor(
            [float(input_scale)], dtype=torch.float32,
        ),
    }
