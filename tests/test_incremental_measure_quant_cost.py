import pickle
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant.incremental_measure_quant_cost import merge_cost_pickles
from prismaquant.measure_quant_cost import (
    ActivationIndex,
    HDetailIndex,
    measure_batched_gpu,
    measure_unbatched,
)
from prismaquant.sensitivity_probe import FisherAccumulator
from prismaquant import format_registry as fr


class TestIncrementalMeasureQuantCost(unittest.TestCase):
    def test_merge_cost_pickles_combines_disjoint_shards(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p1 = td / "a.pkl"
            p2 = td / "b.pkl"
            out = td / "merged.pkl"
            with open(p1, "wb") as f:
                pickle.dump({
                    "costs": {"layer.0": {"NVFP4": {"output_mse": 1.0}}},
                    "formats": ["NVFP4"],
                    "meta": {"part": 1},
                }, f)
            with open(p2, "wb") as f:
                pickle.dump({
                    "costs": {"layer.1": {"BF16": {"output_mse": 0.0}}},
                    "formats": ["NVFP4"],
                    "meta": {"part": 2},
                }, f)

            merge_cost_pickles([p1, p2], out)
            with open(out, "rb") as f:
                merged = pickle.load(f)
            self.assertEqual(set(merged["costs"]), {"layer.0", "layer.1"})
            self.assertEqual(merged["formats"], ["NVFP4"])
            self.assertEqual(merged["meta"]["n_shards"], 2)

    def test_batched_cost_matches_unbatched_for_grouped_linears(self):
        torch.manual_seed(0)

        model = nn.Module()
        model.a = nn.Linear(16, 4, bias=False)
        model.b = nn.Linear(16, 4, bias=False)
        model.c = nn.Linear(16, 4, bias=False)
        target_names = {"a", "b", "c"}
        specs = [fr.get_format("BF16"), fr.get_format("NVFP4")]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            act_dir = root / "act"
            h_dir = root / "h"
            act_dir.mkdir()
            h_dir.mkdir()

            for name in sorted(target_names):
                safe = ActivationIndex._FNAME_SUB.sub("__", name) + ".pt"
                torch.save(
                    {
                        "inputs": torch.randn(7, 16),
                        "row_indices": torch.arange(7, dtype=torch.long),
                        "name": name,
                    },
                    act_dir / safe,
                )
                torch.save(
                    {
                        "H": torch.rand(4, 16),
                        "g2_per_token": torch.linspace(0.25, 2.0, steps=7),
                        "name": name,
                    },
                    h_dir / safe,
                )

            act_cache = ActivationIndex(act_dir, target_names)
            h_detail = HDetailIndex(h_dir, target_names)
            unbatched = measure_unbatched(
                model,
                act_cache,
                target_names,
                specs,
                device="cpu",
                dtype=torch.float32,
                h_detail=h_detail,
            )
            batched = measure_batched_gpu(
                model,
                act_cache,
                target_names,
                specs,
                device="cpu",
                dtype=torch.float32,
                chunk_size=2,
                h_detail=h_detail,
            )

        self.assertEqual(set(batched), target_names)
        self.assertEqual(set(unbatched), target_names)
        for name in sorted(target_names):
            self.assertEqual(set(batched[name]), {s.name for s in specs})
            for spec in specs:
                fmt = spec.name
                for field in (
                    "weight_mse",
                    "output_mse",
                    "rel_output_mse",
                    "predicted_dloss",
                    "fisher_output_mse",
                ):
                    self.assertAlmostEqual(
                        batched[name][fmt][field],
                        unbatched[name][fmt][field],
                        places=6,
                        msg=f"{name} {fmt} {field}",
                    )

    def test_fisher_output_mse_uses_activation_row_indices(self):
        model = nn.Module()
        model.a = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.a.weight.copy_(torch.tensor([[1.0, 2.0]]))

        target_names = {"a"}
        spec = fr.get_format("FP8_E4M3")
        X = torch.tensor(
            [
                [0.2, -0.3],
                [3.0, 0.5],
                [-0.7, 1.2],
            ],
            dtype=torch.float32,
        )
        row_indices = torch.tensor([5, 2, 9], dtype=torch.long)
        g2 = torch.ones(10, dtype=torch.float32)
        g2[2] = 10.0
        g2[5] = 1.0
        g2[9] = 4.0

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            act_dir = root / "act"
            h_dir = root / "h"
            act_dir.mkdir()
            h_dir.mkdir()
            safe = ActivationIndex._FNAME_SUB.sub("__", "a") + ".pt"
            torch.save(
                {"inputs": X, "row_indices": row_indices, "name": "a"},
                act_dir / safe,
            )
            torch.save(
                {"h_diag": torch.ones(1, 2), "g2_per_token": g2, "name": "a"},
                h_dir / safe,
            )

            act_cache = ActivationIndex(act_dir, target_names)
            h_detail = HDetailIndex(h_dir, target_names)
            got = measure_unbatched(
                model,
                act_cache,
                target_names,
                [spec],
                device="cpu",
                dtype=torch.float32,
                h_detail=h_detail,
            )["a"][spec.name]

        W = model.a.weight.detach()
        W_hat = spec.quantize_dequantize(W.clone())
        y_err_sq = (X @ W.T - spec.activation_quantize_dequantize(X.clone()) @ W_hat.T).pow(2)
        weights = g2.index_select(0, row_indices)
        weights = weights / weights.mean()
        expected = float((y_err_sq * weights.unsqueeze(1)).mean().item())
        self.assertAlmostEqual(got["fisher_output_mse"], expected, places=6)

    def test_fisher_accumulator_writes_mtp_h_detail_and_row_indices(self):
        class TinyMtpWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.mtp = nn.Module()
                self.mtp.fc = nn.Linear(2, 1, bias=False)

            def forward(self, x):
                return self.mtp.fc(x)

        torch.manual_seed(0)
        model = TinyMtpWrapper()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            act_dir = root / "act"
            h_dir = root / "h"
            acc = FisherAccumulator(
                model,
                ["mtp.fc"],
                {},
                act_cache_dir=act_dir,
                input_rows=3,
                h_detail_dir=h_dir,
            )
            try:
                x = torch.tensor(
                    [[[0.25, -0.5], [1.0, 0.75], [-0.125, 0.5]]],
                    dtype=torch.float32,
                    requires_grad=True,
                )
                loss = model(x).pow(2).sum()
                loss.backward()
                acc.finalize(tracker=None)
            finally:
                acc.remove_hooks()

            safe = ActivationIndex._FNAME_SUB.sub("__", "mtp.fc") + ".pt"
            act_payload = torch.load(act_dir / safe, map_location="cpu")
            h_payload = torch.load(h_dir / safe, map_location="cpu")

        self.assertEqual(act_payload["name"], "mtp.fc")
        self.assertEqual(tuple(act_payload["inputs"].shape), (3, 2))
        self.assertTrue(torch.equal(
            act_payload["row_indices"],
            torch.tensor([0, 1, 2], dtype=torch.long),
        ))
        self.assertEqual(h_payload["name"], "mtp.fc")
        self.assertEqual(h_payload["kind"], "linear")
        self.assertEqual(tuple(h_payload["h_diag"].shape), (1, 2))
        self.assertEqual(tuple(h_payload["g2_per_token"].shape), (3,))
        self.assertEqual(
            acc.stats["mtp.fc"]["h_detail_path"],
            safe,
        )


if __name__ == "__main__":
    unittest.main()
