"""CLI entry point for the Hadamard-DuQuant joint search.

Drives :func:`prismaquant.joint_hadamard_format_search.run_joint_search`
end-to-end:

  1. Load the source model on GPU.
  2. Build insertion specs via
     :func:`prismaquant.hadamard_duquant.default_insertion_specs`.
  3. Capture per-cluster input activations via forward pre-hooks on each
     cluster's first consumer Linear (sibling consumers share the input,
     so one hook per cluster is enough).
  4. Run a small calibration forward to populate the activation buffers.
  5. Construct :class:`ClusterInputs` per cluster from the captured
     activations + the consumer Linears' weights.
  6. Invoke ``run_joint_search`` and emit sidecar JSON + rotation
     safetensors + decision-log JSONL.

The CLI mirrors :mod:`prismaquant.build_production_cache` for calibration
arg conventions. Typical pipeline invocation runs between Phase 2 (cost
measurement) and Phase 3 (allocator); see ``run-pipeline.sh``.

Output paths (all required):

  - ``--sidecar-output``: ``{cluster_key: {candidates, ...}}`` cost table
    consumed by ``--hadamard-duquant-cost`` on the allocator.
  - ``--rotations-output``: safetensors of per-cluster composed matrices.
  - ``--decision-log-output``: JSONL of per-cluster decision records.

Default group size is the NVFP4 microscale block (16). For MXFP8-only
allocators, pass ``--group-size 32``. The joint search itself measures
both microscale formats per cluster regardless; ``--group-size`` only
controls which clusters are *discovered* (e.g., a cluster whose input dim
is not divisible by 32 won't be searched at group_size=32).
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant.activation_sampling import update_priority_reservoir
from prismaquant.build_rtn_cache import stage_multimodal
from prismaquant.calibration_data import (
    _dtype_from_name,
    load_wikitext_calibration_windowed,
)
from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.hadamard_duquant import (
    ClusterRotationTarget,
    HadamardDuQuantSpec,
    NVFP4_GROUP_SIZE,
    default_insertion_specs,
)
from prismaquant.joint_hadamard_format_search import (
    ClusterInputs,
    DEFAULT_FORMAT_MENU,
    DEFAULT_FORMATS_WITH_ROTATION,
    run_joint_search,
)
from prismaquant.sensitivity_probe import load_calibration


def _resolve_module(model: nn.Module, qname: str) -> nn.Module | None:
    parts = [p for p in qname.split(".") if p]
    cur: object = model
    for p in parts:
        try:
            cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
        except (AttributeError, IndexError, KeyError, TypeError):
            return None
    return cur if isinstance(cur, nn.Module) else None


class _ActivationCapture:
    """Forward pre-hook recording the input activation of one Linear.

    Used to capture each cluster's shared input. We attach one capture per
    cluster (on the first consumer Linear); sibling consumers share the
    same input by construction so additional hooks would be redundant.
    """

    def __init__(self, max_rows: int = 256, *, sample_seed: int = 42):
        self.max_rows = int(max_rows)
        self.captured: torch.Tensor | None = None
        self._priorities: torch.Tensor | None = None
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(sample_seed))

    def __call__(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if not inputs:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return
        # Keep activations on whatever device the forward produced them on
        # (typically CUDA). Solver runs on the same device as the weight,
        # which is also CUDA after the model.to(device) above — so keeping
        # activations on CUDA avoids host/device transfers in the solver
        # inner loop.
        flat = x.detach().reshape(-1, x.shape[-1]).to(dtype=torch.float32)
        self.captured, self._priorities = update_priority_reservoir(
            self.captured,
            self._priorities,
            flat,
            max_rows=self.max_rows,
            generator=self._generator,
        )


def _build_cluster_inputs(
    model: nn.Module,
    specs: Sequence[HadamardDuQuantSpec],
    captures: dict[str, _ActivationCapture],
) -> dict[str, ClusterInputs]:
    """Bundle each cluster's captured activations + per-consumer weights."""
    cluster_inputs: dict[str, ClusterInputs] = {}
    for spec in specs:
        capture = captures.get(spec.cluster_key)
        if capture is None or capture.captured is None:
            continue
        activations = capture.captured
        targets: list[ClusterRotationTarget] = []
        for qname in spec.consumer_qnames:
            mod = _resolve_module(model, qname)
            if mod is None or not hasattr(mod, "weight"):
                continue
            # Keep weights on their native device (CUDA after model.to(device)
            # in main()). The solver inside solve_cluster_rotation infers the
            # device from targets[0].weight.device — keeping everything on
            # GPU makes the Adam/Cayley loop run with GPU matmul throughput.
            targets.append(
                ClusterRotationTarget(
                    qname=qname,
                    weight=mod.weight.detach().to(dtype=torch.float32),
                    activations=activations,
                )
            )
        if not targets:
            continue
        cluster_inputs[spec.cluster_key] = ClusterInputs(
            cluster_key=spec.cluster_key, targets=targets
        )
    return cluster_inputs


def _capture_cluster_inputs_for_samples(
    model: nn.Module,
    specs: Sequence[HadamardDuQuantSpec],
    samples: torch.Tensor,
    *,
    device: torch.device,
    max_act_rows: int,
    sample_seed: int,
) -> dict[str, ClusterInputs]:
    """Capture one activation reservoir per cluster for a sample tensor."""
    captures: dict[str, _ActivationCapture] = {}
    handles = []
    for spec in specs:
        if not spec.consumer_qnames:
            continue
        first = spec.consumer_qnames[0]
        mod = _resolve_module(model, first)
        if mod is None:
            continue
        cap = _ActivationCapture(
            max_rows=max_act_rows,
            sample_seed=sample_seed,
        )
        captures[spec.cluster_key] = cap
        handles.append(mod.register_forward_pre_hook(cap))

    try:
        with torch.no_grad():
            for sample in samples:
                if sample.dim() == 1:
                    sample = sample.unsqueeze(0)
                sample = sample.to(device=device)
                model(sample)
    finally:
        for h in handles:
            h.remove()

    return _build_cluster_inputs(model, specs, captures)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hadamard-DuQuant joint search")
    p.add_argument("--model", required=True)
    p.add_argument("--sidecar-output", required=True)
    p.add_argument("--rotations-output", required=True)
    p.add_argument("--decision-log-output", default=None)
    p.add_argument(
        "--group-size",
        type=int,
        default=NVFP4_GROUP_SIZE,
        help="Microscale block size for insertion-point discovery. "
        "16 = NVFP4 default; 32 picks up MXFP8 clusters only.",
    )
    p.add_argument("--n-calib-samples", type=int, default=8)
    p.add_argument(
        "--n-validation-samples",
        type=int,
        default=0,
        help="Optional held-out calibration windows. When >0, rotations are "
        "accepted only if they improve both train and held-out renderer "
        "scores by the configured margins. Sidecar fisher_mse uses held-out "
        "scores so the allocator sees the validation estimate.",
    )
    p.add_argument("--calib-seqlen", type=int, default=256)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--validation-seed", type=int, default=4242)
    p.add_argument(
        "--dataset",
        default=None,
        help="Calibration source (HF id / .jsonl / .txt). When omitted, "
        "uses the wikitext-2 windowed loader (matches build_production_cache).",
    )
    p.add_argument("--dtype", default="bf16")
    p.add_argument(
        "--max-act-rows",
        type=int,
        default=256,
        help="Rows of captured activations per cluster.",
    )
    p.add_argument(
        "--solver-iters",
        type=int,
        default=500,
        help="Maximum Adam steps per cluster. Default 500 leaves ample "
        "headroom for the slow-converging V→O clusters; in practice most "
        "clusters early-stop well before the cap (see "
        "--solver-early-stop-patience).",
    )
    p.add_argument(
        "--solver-lr", type=float, default=1e-3,
        help="Adam learning rate for the rotation solver. Default 1e-3 "
        "is conservative enough that V_O cluster gains stabilize; higher "
        "lr (e.g. 5e-3) tends to converge faster but ends in shallower "
        "local minima for the V→O rotations especially.",
    )
    p.add_argument(
        "--solver-loss",
        choices=["w_only", "w4a4"],
        default="w4a4",
        help="Solver MSE objective. 'w_only': legacy ||x (W - "
        "Q(W M^T) M)^T||^2 — weight quantization error only. 'w4a4' "
        "(default): joint "
        "||x W^T - Q_a(x M^T) @ Q_w(W M^T)^T||^2 — models runtime W4A4 "
        "exactly, includes the activation quantization error term. The "
        "joint loss costs ~2x compute per iter (extra (N, out) matmul) "
        "but is the right objective for NVFP4-A4 / MXFP4-A4 deployment.",
    )
    p.add_argument(
        "--score-loss",
        choices=["w_only", "w4a4"],
        default=None,
        help="Sidecar/adaptive-probe scoring objective. Defaults to "
        "--solver-loss so W4A4 solver runs also score candidates with "
        "activation quantization included.",
    )
    p.add_argument(
        "--solver-weight-decay",
        type=float,
        default=0.0,
        help="L2 weight decay on A_skew (rotation parameter). Biases the "
        "solver toward smaller rotations, trading some calib-set fit for "
        "better generalization on small calibration sets. Default 0.0 = "
        "vanilla Adam (legacy). Try 1e-4 or 1e-3 if rotations look "
        "overfit to the calibration sample.",
    )
    p.add_argument(
        "--solver-multi-init",
        default="",
        help="Comma-separated extra init candidates to try per cluster, "
        "selecting the W4A4 winner. E.g. "
        "'sylvester_t0p3,sylvester_t0p5,svd_v'. Empty (default) skips "
        "multi-init and uses only --solver-init. Costs ~N× joint-search "
        "wall-clock where N is the number of inits.",
    )
    p.add_argument(
        "--solver-n-random-probes",
        type=int,
        default=0,
        help="Per-cluster adaptive probe: run N random-orthogonal-init "
        "Adams alongside the named init, commit argmin W4A4 score. The "
        "random-orthogonal diagnostic on Qwen3-4B showed ~17%% of "
        "clusters have multimodal W4A4 landscape where random inits "
        "find basins identity-Adam can't reach (best cluster: +6.5pp "
        "additional Fisher-MSE reduction). N=4 catches ~50%% of stuck "
        "basins; N=8 catches ~75%%; N=16 catches ~94%%. Cost: (1+N)× "
        "joint-search wall-clock. Default 0 = legacy single-init path.",
    )
    p.add_argument(
        "--solver-init",
        default="identity",
        help="Initial rotation for the Adam STE solver. 'identity' (default) "
        "starts at the no-rotation baseline; Adam finds small local "
        "refinements. 'sylvester' starts at the normalized Sylvester "
        "Hadamard (DuQuant's spreading rotation) and the STE-bounded Adam "
        "loop refines from there — historically regressed under W-only "
        "loss but useful with --solver-loss=w4a4 on larger models where "
        "outlier suppression actually pays off.",
    )
    p.add_argument(
        "--solver-early-stop-patience", type=int, default=200,
        help="Break the Adam loop after this many consecutive iters "
        "without a new best loss. Default 200 is intentionally "
        "conservative: it gives slow-converging clusters a long plateau "
        "before declaring convergence, at the cost of running fast "
        "clusters past their optimum. Best-so-far tracking means the "
        "extra iters cost wall-clock but never quality. Pass 0 to "
        "disable early-stop entirely (run all --solver-iters iters).",
    )
    p.add_argument(
        "--format-menu",
        default=",".join(DEFAULT_FORMAT_MENU),
        help="Comma-separated format menu to score per cluster.",
    )
    p.add_argument(
        "--formats-with-rotation",
        default=",".join(DEFAULT_FORMATS_WITH_ROTATION),
        help="Comma-separated formats for which to attempt rotation.",
    )
    p.add_argument(
        "--body-layer-prefix",
        default="model.layers",
        help="Dotted path to the layer container (default: model.layers).",
    )
    p.add_argument(
        "--rotation-scope",
        choices=["folded_only", "online", "all"],
        default="all",
        help="Insertion-point scope. all (default) emits both folded "
        "producer/consumer clusters and vLLM-native online Hadamard input "
        "transform clusters. folded_only avoids runtime transforms. online "
        "emits runtime input transform clusters only.",
    )
    p.add_argument(
        "--cluster-kind-filter",
        default="",
        help="Optional comma-separated insertion kinds to keep after discovery "
        "(residual,v_o,attn_out,down_proj). Useful for ablations such as "
        "online down_proj only.",
    )
    p.add_argument(
        "--online-rotation-mode",
        choices=["hadamard", "learned"],
        default="hadamard",
        help="Runtime transform family for ONLINE clusters. hadamard "
        "(default) emits vLLM-native compressed-tensors Hadamard "
        "transforms and scores a fixed normalized Sylvester matrix. "
        "learned keeps the dense random-matrix solver path, which is "
        "research-only for NVFP4 online transforms on vLLM 0.20.",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Override residual-stream hidden dim. When unset, inferred "
        "from the first layer's standard projection — fallback chain "
        "covers self_attn.q_proj, linear_attn.in_proj_qkv, and mlp.gate/"
        "up_proj for models with mixed attention types.",
    )
    p.add_argument(
        "--rotation-min-train-gain",
        type=float,
        default=None,
        help="Minimum relative train renderer-score gain required to accept "
        "a solved rotation. Defaults to disabled without validation and 0.0 "
        "when --n-validation-samples > 0.",
    )
    p.add_argument(
        "--rotation-min-validation-gain",
        type=float,
        default=None,
        help="Minimum relative held-out renderer-score gain required to "
        "accept a solved rotation. Defaults to 0.0 when validation is enabled.",
    )
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    sidecar_path = Path(args.sidecar_output)
    rotations_path = Path(args.rotations_output)
    decision_log_path = Path(args.decision_log_output) if args.decision_log_output else None

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    rotations_path.parent.mkdir(parents=True, exist_ok=True)
    if decision_log_path is not None:
        decision_log_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = _dtype_from_name(args.dtype)
    staged, _cleanup = stage_multimodal(args.model)
    device = require_cuda_hot_path("run_joint_hadamard_search")
    local_only = Path(staged).exists()

    tokenizer = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True, local_files_only=local_only,
    )
    n_validation_samples = max(0, int(args.n_validation_samples))
    if args.dataset:
        calib_all = load_calibration(
            tokenizer,
            args.dataset,
            int(args.n_calib_samples) + n_validation_samples,
            args.calib_seqlen,
        )
        calib_ids = calib_all[: int(args.n_calib_samples)]
        validation_ids = (
            calib_all[int(args.n_calib_samples):]
            if n_validation_samples > 0 else None
        )
    else:
        calib_ids = load_wikitext_calibration_windowed(
            tokenizer,
            args.n_calib_samples,
            args.calib_seqlen,
            split=args.calib_split,
            seed=args.calib_seed,
        )
        validation_ids = (
            load_wikitext_calibration_windowed(
                tokenizer,
                n_validation_samples,
                args.calib_seqlen,
                split=args.calib_split,
                seed=args.validation_seed,
            )
            if n_validation_samples > 0 else None
        )
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": local_only,
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = "cuda"
    try:
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
    except ValueError as exc:
        if "accelerate" not in str(exc):
            raise
        load_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        model.to(device)
    if device.type != "cuda":
        model.to(device)
    model.eval()

    include_folded = args.rotation_scope in ("folded_only", "all")
    include_online = args.rotation_scope in ("online", "all")
    include_residual_online = include_online
    specs = default_insertion_specs(
        model,
        group_size=int(args.group_size),
        body_layer_prefix=args.body_layer_prefix,
        hidden_dim=int(args.hidden_dim) if args.hidden_dim else None,
        include_folded=include_folded,
        include_online=include_online,
        include_residual_online=include_residual_online,
    )
    if args.cluster_kind_filter.strip():
        allowed_kinds = {
            k.strip() for k in args.cluster_kind_filter.split(",") if k.strip()
        }
        specs = [s for s in specs if s.kind.value in allowed_kinds]
    print(
        f"[joint-search] {len(specs)} insertion points discovered at "
        f"group_size={args.group_size} scope={args.rotation_scope} "
        f"online_rotation_mode={args.online_rotation_mode}",
        flush=True,
    )
    if not specs:
        print(
            "[joint-search] no insertion points — nothing to search; "
            "writing empty sidecar.",
            flush=True,
        )
        sidecar_path.write_text('{"version": "1", "clusters": {}}\n')
        return 0

    cluster_inputs = _capture_cluster_inputs_for_samples(
        model,
        specs,
        calib_ids,
        device=device,
        max_act_rows=int(args.max_act_rows),
        sample_seed=int(args.calib_seed),
    )
    print(
        f"[joint-search] {len(cluster_inputs)}/{len(specs)} clusters "
        f"populated with activations + weights",
        flush=True,
    )

    validation_cluster_inputs = None
    if validation_ids is not None and int(validation_ids.numel()) > 0:
        validation_cluster_inputs = _capture_cluster_inputs_for_samples(
            model,
            specs,
            validation_ids,
            device=device,
            max_act_rows=int(args.max_act_rows),
            sample_seed=int(args.validation_seed),
        )
        print(
            f"[joint-search] {len(validation_cluster_inputs)}/{len(specs)} "
            "clusters populated with held-out activations + weights",
            flush=True,
        )

    format_menu = tuple(s.strip() for s in args.format_menu.split(",") if s.strip())
    formats_with_rotation = tuple(
        s.strip() for s in args.formats_with_rotation.split(",") if s.strip()
    )

    run_joint_search(
        specs,
        cluster_inputs,
        validation_cluster_inputs=validation_cluster_inputs,
        sidecar_path=sidecar_path,
        rotation_safetensors_path=rotations_path,
        decision_log_path=decision_log_path,
        format_menu=format_menu,
        formats_with_rotation=formats_with_rotation,
        solver_n_iters=int(args.solver_iters),
        solver_lr=float(args.solver_lr),
        solver_loss=str(args.solver_loss),
        score_loss=str(args.score_loss) if args.score_loss else None,
        solver_init=str(args.solver_init),
        solver_weight_decay=float(args.solver_weight_decay),
        solver_multi_init=tuple(
            s.strip() for s in args.solver_multi_init.split(",") if s.strip()
        ),
        solver_n_random_probes=int(args.solver_n_random_probes),
        solver_early_stop_patience=(
            None if int(args.solver_early_stop_patience) <= 0
            else int(args.solver_early_stop_patience)
        ),
        online_rotation_mode=str(args.online_rotation_mode),
        rotation_min_train_gain=(
            float(args.rotation_min_train_gain)
            if args.rotation_min_train_gain is not None
            else (0.0 if validation_cluster_inputs is not None else float("-inf"))
        ),
        rotation_min_validation_gain=(
            float(args.rotation_min_validation_gain)
            if args.rotation_min_validation_gain is not None
            else (0.0 if validation_cluster_inputs is not None else float("-inf"))
        ),
    )
    print(
        f"[joint-search] wrote sidecar={sidecar_path} rotations={rotations_path}"
        + (f" decisions={decision_log_path}" if decision_log_path else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
