"""Compare per-G-block weight outlier reduction across two HDQ artifacts."""
import json, statistics, sys, os
import torch
from safetensors.torch import load_file
import prismaquant  # noqa: F401 — polyfills
from transformers import AutoModelForCausalLM
from prismaquant.hadamard_duquant import apply_block_rotation_input

MODEL = os.environ.get("HFMODEL", "/hfcache/qwen35-0p8b-bf16-untied")


def per_block_stats(w_chunked: torch.Tensor) -> dict:
    max_abs = w_chunked.abs().amax(dim=-1)
    median_abs = w_chunked.abs().median(dim=-1).values
    peak = (max_abs / median_abs.clamp_min(1e-12)).mean().item()
    return {
        "max_abs_p99": float(max_abs.quantile(0.99).item()),
        "peakiness": float(peak),
    }


def _get(model, qname):
    cur = model
    for p in qname.split("."):
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    return cur


def analyze(model, run_dir: str) -> dict:
    if os.path.exists(f"{run_dir}/artifacts/sidecar.json"):
        sc = json.load(open(f"{run_dir}/artifacts/sidecar.json"))
        rotations = load_file(f"{run_dir}/artifacts/rotations.safetensors", device="cuda")
    else:
        sc = json.load(open(f"{run_dir}/artifacts/hadamard_duquant_sidecar.json"))
        rotations = load_file(
            f"{run_dir}/artifacts/hadamard_duquant_rotations.safetensors", device="cuda")
    results: dict[str, dict[str, list]] = {}
    for cluster_key, cdata in sc["clusters"].items():
        if not cdata.get("consumer_qnames"):
            continue
        rot_key = f"{cluster_key}/NVFP4/composed_matrix"
        if rot_key not in rotations:
            continue
        M = rotations[rot_key]
        G = M.shape[0]
        kind = cdata["insertion_kind"]
        for qname in cdata["consumer_qnames"]:
            try:
                lin = _get(model, qname)
                W = lin.weight.detach().to(torch.float32)
            except Exception:
                continue
            in_features = W.shape[1]
            if in_features % G != 0:
                continue
            W_g = W.view(W.shape[0], in_features // G, G)
            pre = per_block_stats(W_g)
            W_rot = apply_block_rotation_input(W, M.t())
            W_rot_g = W_rot.view(W_rot.shape[0], in_features // G, G)
            post = per_block_stats(W_rot_g)
            bucket = results.setdefault(kind, {"pre": [], "post": []})
            bucket["pre"].append(pre)
            bucket["post"].append(post)
    return results


def fmt(name: str, results: dict):
    print(f"=== {name} ===")
    hdr = (
        "  kind        n_lin   pre_peak  post_peak  Δpeak%    "
        "pre_p99   post_p99  Δp99%"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for kind in sorted(results):
        b = results[kind]
        pre_pk = statistics.mean(s["peakiness"] for s in b["pre"])
        post_pk = statistics.mean(s["peakiness"] for s in b["post"])
        pre_p99 = statistics.mean(s["max_abs_p99"] for s in b["pre"])
        post_p99 = statistics.mean(s["max_abs_p99"] for s in b["post"])
        d_pk = (post_pk / pre_pk - 1) * 100
        d_p99 = (post_p99 / pre_p99 - 1) * 100
        print(
            f"  {kind:<10}  {len(b['pre']):<5}   "
            f"{pre_pk:8.4f}  {post_pk:9.4f}  {d_pk:+7.2f}%  "
            f"{pre_p99:7.4f}  {post_p99:8.4f}  {d_p99:+6.2f}%"
        )


if __name__ == "__main__":
    runs = sys.argv[1:]
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    m.to("cuda"); m.eval()
    for r in runs:
        out = analyze(m, r)
        fmt(r, out)
        print()
