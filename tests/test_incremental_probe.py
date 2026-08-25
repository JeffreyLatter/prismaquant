import pickle
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.incremental_probe import (
    build_layer_shard_regexes,
    merge_probe_pickles,
    _set_minimax_fast_moe,
)


class TestIncrementalProbe(unittest.TestCase):
    def test_build_layer_shard_regexes_groups_layers(self):
        regexes = build_layer_shard_regexes(5, 2)
        self.assertEqual(regexes, [
            r"model\.layers\.(?:0|1)\.",
            r"model\.layers\.(?:2|3)\.",
            r"model\.layers\.4\.",
        ])

    def test_merge_probe_pickles_sums_router_counts(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p1 = td / "a.pkl"
            p2 = td / "b.pkl"
            out = td / "merged.pkl"
            with open(p1, "wb") as f:
                pickle.dump({
                    "stats": {"layer.0": {"h_trace": 1.0}},
                    "router_counts": {"r": {"0": 1.5}},
                    "router_totals": {"r": 3},
                    "expert_info": {"layer.0": ("r", "0")},
                    "meta": {"model": "toy"},
                }, f)
            with open(p2, "wb") as f:
                pickle.dump({
                    "stats": {"layer.1": {"h_trace": 2.0}},
                    "router_counts": {"r": {"0": 0.5, "1": 2.0}},
                    "router_totals": {"r": 5},
                    "expert_info": {"layer.1": ("r", "1")},
                    "meta": {"model": "toy"},
                }, f)

            merge_probe_pickles([p1, p2], out)
            with open(out, "rb") as f:
                merged = pickle.load(f)
            self.assertEqual(set(merged["stats"]), {"layer.0", "layer.1"})
            self.assertEqual(merged["router_counts"]["r"]["0"], 2.0)
            self.assertEqual(merged["router_counts"]["r"]["1"], 2.0)
            self.assertEqual(merged["router_totals"]["r"], 8)
            self.assertEqual(merged["meta"]["n_shards"], 2)

    def test_minimax_fast_moe_matches_modulelist_forward_and_backward(self):
        class ToyMLP(nn.Module):
            def __init__(self, hidden=5, ffn=7):
                super().__init__()
                self.w1 = nn.Linear(hidden, ffn, bias=False)
                self.w2 = nn.Linear(ffn, hidden, bias=False)
                self.w3 = nn.Linear(hidden, ffn, bias=False)
                self.act_fn = F.silu

            def forward(self, hidden_states):
                return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))

        class MiniMaxM2Experts(nn.ModuleList):
            def __init__(self, n_experts=6, top_k=3):
                super().__init__([ToyMLP() for _ in range(n_experts)])
                self.num_experts = n_experts
                self.top_k = top_k

            def forward(self, hidden_states, top_k_index, top_k_weights):
                final_hidden_states = torch.zeros_like(hidden_states)
                expert_mask = torch.nn.functional.one_hot(
                    top_k_index, num_classes=self.num_experts
                ).permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                for expert_idx in expert_hit:
                    idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                    current_state = hidden_states[None, top_x].reshape(
                        -1, hidden_states.shape[-1])
                    expert_id = int(expert_idx.item())
                    current_hidden_states = (
                        self[expert_id](current_state)
                        * top_k_weights[top_x, idx, None]
                    )
                    final_hidden_states.index_add_(0, top_x, current_hidden_states)
                return final_hidden_states

        torch.manual_seed(123)
        ref = MiniMaxM2Experts()
        fast = MiniMaxM2Experts()
        fast.load_state_dict(ref.state_dict())
        # `class_names` now comes from the model profile
        # (`packed_expert_module_class_names()` -> the spec's
        # `packed_experts.module_class_names`), replacing the literal
        # "MiniMaxM2Experts" that used to live in incremental_probe.py.
        self.assertEqual(
            _set_minimax_fast_moe(
                fast, True, chunk_size=2,
                class_names=("MiniMaxM2Experts",)),
            1,
        )

        hidden_ref = torch.randn(11, 5, requires_grad=True)
        hidden_fast = hidden_ref.detach().clone().requires_grad_(True)
        top_k_index = torch.tensor([
            [0, 1, 3], [1, 4, 5], [2, 3, 0], [5, 4, 1],
            [3, 2, 1], [0, 5, 4], [2, 4, 3], [1, 0, 5],
            [4, 2, 0], [3, 5, 1], [2, 0, 4],
        ])
        weights = torch.rand(11, 3)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        upstream = torch.randn(11, 5)

        out_ref = ref(hidden_ref, top_k_index, weights)
        out_fast = fast(hidden_fast, top_k_index, weights)
        self.assertTrue(torch.allclose(out_fast, out_ref, atol=1e-6, rtol=1e-6))

        out_ref.backward(upstream)
        out_fast.backward(upstream)
        self.assertTrue(torch.allclose(
            hidden_fast.grad, hidden_ref.grad, atol=1e-6, rtol=1e-6))
        for (_, p_ref), (_, p_fast) in zip(ref.named_parameters(), fast.named_parameters()):
            self.assertTrue(torch.allclose(
                p_fast.grad, p_ref.grad, atol=1e-6, rtol=1e-6))


def _build_tiny_model(path: Path) -> int:
    """Write a tiny Llama checkpoint + fast tokenizer to `path`.
    Returns the number of decoder layers."""
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers

    n_layers = 6
    cfg = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=n_layers, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=128,
        tie_word_embeddings=False)
    LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(
        path, safe_serialization=True)

    words = ("the quick brown fox jumps over a lazy dog and then runs far "
             "away into deep dark woods").split()
    vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3}
    for w in sorted(set(words)):
        vocab[w] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>",
        bos_token="<s>", eos_token="</s>", model_max_length=128,
    ).save_pretrained(path)
    return n_layers


class TestPhase3GradCarry(unittest.TestCase):
    """Part 1 of the pipeline-parallel work: the phase-3 reverse sweep
    carries `grad_out` shard→shard instead of restarting from the model
    tail every shard. That is a pure dead-code elimination — the chain
    rule split at a layer boundary and resumed with the saved
    intermediate is identical to the full sweep. This test proves it:
    the merged probe.pkl Fisher stats must be bit-identical regardless
    of `--layers-per-shard` (which controls how many shards, hence how
    much carrying happens). lps=6 sweeps the whole model in one shard
    with NO carry (the baseline); lps=1 carries between all 6 shards."""

    def test_grad_carry_is_lps_invariant(self):
        try:
            import transformers  # noqa: F401
            import tokenizers  # noqa: F401
        except Exception:
            self.skipTest("transformers/tokenizers not available")
        if not torch.cuda.is_available():
            # `incremental_probe.main()` calls `gpu_guard.require_cuda_hot_path`
            # (GPU-or-bust, no override — see tests/test_gpu_guard.py). This
            # test exercises the real CLI subprocess, so it needs a real CUDA
            # device even though the model is tiny and the check is pure math.
            self.skipTest("CUDA not available")

        import json
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "tiny"
            _build_tiny_model(model_dir)

            calib = root / "calib.jsonl"
            with open(calib, "w") as f:
                f.write(json.dumps({"__manifest__": {
                    "schema": "prismaquant.calibration.diverse_v1",
                    "version": "test"}}) + "\n")
                for _ in range(4):
                    f.write(json.dumps({"text": " ".join(
                        ["the quick brown fox jumps over a lazy dog"] * 6)})
                        + "\n")

            def run(lps: int) -> dict:
                out = root / f"run{lps}"
                subprocess.run(
                    [sys.executable, "-m", "prismaquant.incremental_probe",
                     "--model", str(model_dir), "--dataset", str(calib),
                     "--nsamples", "4", "--seqlen", "32",
                     "--device", "cuda", "--dtype", "bf16",
                     "--output", str(out / "probe.pkl"),
                     "--activation-cache-dir", str(out / "act"),
                     "--work-dir", str(out / "work"),
                     "--layers-per-shard", str(lps)],
                    check=True, capture_output=True, text=True)
                with open(out / "probe.pkl", "rb") as f:
                    return pickle.load(f)["stats"]

            baseline = run(6)   # one shard, full sweep, no carry
            carried = run(1)    # six shards, carry between each

            self.assertEqual(set(baseline), set(carried),
                             "stat key sets differ between lps=1 and lps=6")
            self.assertTrue(baseline, "probe produced no stats")
            for fqn in baseline:
                for k in ("h_trace", "h_w2_sum", "h_trace_raw",
                          "h_w2_sum_raw", "n_tokens_seen"):
                    vb = baseline[fqn].get(k)
                    vc = carried[fqn].get(k)
                    if vb is None or vc is None:
                        continue
                    self.assertEqual(
                        float(vb), float(vc),
                        f"{fqn}.{k}: carry (lps=1) {vc} != baseline "
                        f"(lps=6) {vb} — grad carry changed the math")


if __name__ == "__main__":
    unittest.main()
