import unittest

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.allocator import Candidate, build_candidates, filter_candidates_for_profile
from prismaquant.calibrate_allocator import (
    install_activation_hooks,
    per_format_predicted_breakdown,
    select_targets,
)
from prismaquant.sensitivity_probe import discover_moe_structure


class TestPrismaQuantFormatRegistry(unittest.TestCase):
    def test_block_formats_have_expected_shape_aware_bits(self):
        shape = (128, 128)
        self.assertAlmostEqual(fr.get_format("NVFP4").effective_bits_for_shape(shape), 4.5)
        self.assertAlmostEqual(fr.get_format("MXFP4").effective_bits_for_shape(shape), 4.25)
        self.assertAlmostEqual(fr.get_format("MXFP8").effective_bits_for_shape(shape), 8.25)
        self.assertAlmostEqual(fr.get_format("FP8_SOURCE").effective_bits_for_shape(shape), 8.001953125)
        self.assertAlmostEqual(fr.get_format("BF16").effective_bits_for_shape(shape), 16.0)

    def test_mxfp8_short_name_is_input_alias_only(self):
        self.assertEqual(fr.canonical_format_name("MXFP8"), "MXFP8_E4M3")
        self.assertEqual(fr.get_format("MXFP8").name, "MXFP8_E4M3")
        self.assertIn("MXFP8", fr.aliases_for("MXFP8_E4M3"))

    def test_low_bit_custom_kernel_formats_are_not_registered(self):
        for name in ("INT2", "INT3", "NVINT2", "NVINT3", "NVFP3"):
            with self.assertRaises(KeyError):
                fr.get_format(name)

    def test_source_fp8_uses_2d_scale_blocks(self):
        spec = fr.get_format("FP8_SOURCE")
        shape = (256, 256)
        expected_bytes = 256 * 256 + 4 * 4  # fp8 weights + four fp32 128x128 scales

        self.assertAlmostEqual(spec.effective_bits, 8.001953125)
        self.assertEqual(spec.scale_count_for_shape(shape), 4)
        self.assertEqual(spec.memory_bytes_for_shape(shape), expected_bytes)
        self.assertAlmostEqual(
            spec.effective_bits_for_shape(shape),
            8.0 * expected_bytes / (256 * 256),
        )

    def test_per_channel_formats_use_row_scale_count(self):
        shape = (5, 7)
        spec = fr.get_format("FP8_E4M3")
        expected_bytes = 5 * 7 + 5 * 4  # one byte per weight, fp32 row scales
        self.assertEqual(spec.scale_count_for_shape(shape), 5)
        self.assertEqual(spec.memory_bytes_for_shape(shape), expected_bytes)
        self.assertAlmostEqual(spec.effective_bits_for_shape(shape), 8.0 * expected_bytes / 35.0)

    def test_activation_quantization_changes_native_a4_a8_formats(self):
        x = torch.tensor([[0.13, -0.51, 1.77, -3.25]], dtype=torch.float32)
        for fmt in ("NVFP4", "MXFP8", "FP8_E4M3"):
            y = fr.get_format(fmt).activation_quantize_dequantize(x.clone())
            self.assertFalse(torch.equal(x, y), msg=fmt)

    def test_activation_quantization_skips_a16_formats(self):
        x = torch.tensor([[0.13, -0.51, 1.77, -3.25]], dtype=torch.float32)
        for fmt in ("MXFP8A16", "NVFP4A16", "BF16"):
            y = fr.get_format(fmt).activation_quantize_dequantize(x.clone())
            self.assertTrue(torch.equal(x, y), msg=fmt)


class TestPrismaQuantAllocatorMath(unittest.TestCase):
    def test_build_candidates_uses_shape_aware_bits(self):
        # Predicted Δloss = 0.5 · h_trace · output_mse  (joint W·X
        # perturbation under diagonal-Fisher curvature; output_mse is
        # the cost step's W*A*-aware error metric).
        stats = {
            "layer.weight": {
                "h_trace": 2.0,
                "out_features": 5,
                "in_features": 7,
                "n_params": 35,
            }
        }
        costs = {
            "layer.weight": {
                "FP8_E4M3": {"weight_mse": 0.10, "output_mse": 0.25},
                "BF16": {"weight_mse": 0.0, "output_mse": 0.0},
            }
        }
        cands = build_candidates(stats, costs, [fr.get_format("FP8_E4M3"), fr.get_format("BF16")])
        by_fmt = {cand.fmt: cand for cand in cands["layer.weight"]}
        self.assertAlmostEqual(
            by_fmt["FP8_E4M3"].bits_per_param,
            fr.get_format("FP8_E4M3").effective_bits_for_shape((5, 7)),
        )
        self.assertEqual(
            by_fmt["FP8_E4M3"].memory_bytes,
            fr.get_format("FP8_E4M3").memory_bytes_for_shape((5, 7)),
        )
        self.assertAlmostEqual(by_fmt["FP8_E4M3"].predicted_dloss, 0.5 * 2.0 * 0.25)
        self.assertAlmostEqual(by_fmt["BF16"].predicted_dloss, 0.0)

    def test_build_candidates_prices_source_fp8_below_mxfp8(self):
        stats = {
            "layer.weight": {
                "h_trace": 2.0,
                "out_features": 128,
                "in_features": 128,
                "n_params": 128 * 128,
            }
        }
        costs = {
            "layer.weight": {
                "FP8_SOURCE": {"weight_mse": 0.0},
                "MXFP8": {"weight_mse": 0.01},
            }
        }
        cands = build_candidates(
            stats,
            costs,
            [fr.get_format("FP8_SOURCE"), fr.get_format("MXFP8")],
            source_manifest={"layer.weight": "fp8"},
        )
        by_fmt = {cand.fmt: cand for cand in cands["layer.weight"]}

        self.assertLess(
            by_fmt["FP8_SOURCE"].bits_per_param,
            by_fmt["MXFP8_E4M3"].bits_per_param,
        )
        self.assertEqual(by_fmt["FP8_SOURCE"].memory_bytes, 128 * 128 + 4)
        self.assertAlmostEqual(by_fmt["FP8_SOURCE"].bits_per_param, 8.001953125)
        self.assertAlmostEqual(by_fmt["MXFP8_E4M3"].bits_per_param, 8.25)

    def test_build_candidates_applies_calibrated_gains(self):
        stats = {
            "layer.weight": {
                "h_trace": 2.0,
                "out_features": 128,
                "in_features": 128,
                "n_params": 128 * 128,
            }
        }
        costs = {
            "layer.weight": {
                "NVFP4": {"weight_mse": 0.10},
                "MXFP8": {"weight_mse": 0.02},
            }
        }
        # Without calibration: NVFP4 = 0.10, MXFP8 = 0.02 (per-element MSE).
        # With α_NVFP4=2 the NVFP4 candidate's predicted Δloss should double.
        cands = build_candidates(
            stats, costs,
            [fr.get_format("NVFP4"), fr.get_format("MXFP8")],
            calibrated_gains={"NVFP4": 2.0, "MXFP8": 1.0},
        )
        by_fmt = {c.fmt: c for c in cands["layer.weight"]}
        self.assertAlmostEqual(by_fmt["NVFP4"].predicted_dloss, 0.5 * 2.0 * 0.10 * 2.0)
        self.assertAlmostEqual(
            by_fmt["MXFP8_E4M3"].predicted_dloss,
            0.5 * 2.0 * 0.02 * 1.0,
        )

    def test_build_candidates_ignores_unmeasured_packed_output_mse(self):
        stats = {
            "model.layers.0.mlp.experts.gate_up_proj": {
                "h_trace": 3.0,
                "out_features": 8,
                "in_features": 16,
                "n_params": 2 * 8 * 16,
                "num_experts": 2,
                "_packed_experts_module": "model.layers.0.mlp.experts",
                "_packed_param": "gate_up_proj",
            }
        }
        costs = {
            "model.layers.0.mlp.experts.gate_up_proj": {
                "NVFP4": {
                    "weight_mse": 0.20,
                    "output_mse": 0.0,
                    "output_mse_measured": False,
                    "predicted_dloss": 7.0,
                },
                # Defensive path for old packed artifacts that wrote the
                # placeholder zero but did not carry the explicit measured flag.
                "MXFP4": {
                    "weight_mse": 0.10,
                    "output_mse": 0.0,
                    "predicted_dloss": 2.0,
                },
                "BF16": {
                    "weight_mse": 0.25,
                    "output_mse": 0.0,
                    "output_mse_measured": False,
                },
            }
        }
        cands = build_candidates(
            stats,
            costs,
            [fr.get_format("NVFP4"), fr.get_format("MXFP4"), fr.get_format("BF16")],
        )
        by_fmt = {
            c.fmt: c
            for c in cands["model.layers.0.mlp.experts.gate_up_proj"]
        }
        self.assertAlmostEqual(by_fmt["NVFP4"].predicted_dloss, 7.0)
        self.assertAlmostEqual(by_fmt["MXFP4"].predicted_dloss, 2.0)
        self.assertAlmostEqual(by_fmt["BF16"].predicted_dloss, 0.5 * 3.0 * 0.25)
        expected_nvfp4_bytes = fr.get_format("NVFP4").memory_bytes_for_shape(
            (2, 8, 16)
        )
        self.assertEqual(by_fmt["NVFP4"].memory_bytes, expected_nvfp4_bytes)

    def test_build_candidates_prices_packed_expert_memory_with_expert_dim(self):
        name = "model.layers.0.mlp.experts.down_proj"
        stats = {
            name: {
                "h_trace": 1.0,
                "out_features": 8,
                "in_features": 16,
                "num_experts": 4,
                "n_params": 4 * 8 * 16,
                "_packed_experts_module": "model.layers.0.mlp.experts",
                "_packed_param": "down_proj",
            }
        }
        costs = {
            name: {
                "NVFP4": {"weight_mse": 0.1, "predicted_dloss": 0.1},
                "BF16": {"weight_mse": 0.0, "predicted_dloss": 0.0},
            }
        }

        cands = build_candidates(
            stats, costs, [fr.get_format("NVFP4"), fr.get_format("BF16")]
        )
        by_fmt = {c.fmt: c for c in cands[name]}

        self.assertEqual(
            by_fmt["NVFP4"].memory_bytes,
            fr.get_format("NVFP4").memory_bytes_for_shape((4, 8, 16)),
        )
        self.assertEqual(
            by_fmt["BF16"].memory_bytes,
            fr.get_format("BF16").memory_bytes_for_shape((4, 8, 16)),
        )

    def test_calibration_breakdown_uses_allocator_candidate_basis(self):
        assignment = {
            "layer.output_measured": "NVFP4",
            "layer.packed_expert": "MXFP4",
        }
        stats = {
            "layer.output_measured": {
                "h_trace": 2.0,
                "out_features": 8,
                "in_features": 16,
                "n_params": 128,
            },
            "layer.packed_expert": {
                "h_trace": 3.0,
                "out_features": 8,
                "in_features": 16,
                "n_params": 256,
                "num_experts": 2,
            },
        }
        costs = {
            "layer.output_measured": {
                "NVFP4": {"weight_mse": 10.0, "output_mse": 0.25},
            },
            "layer.packed_expert": {
                "MXFP4": {
                    "weight_mse": 0.20,
                    "output_mse": 0.0,
                    "output_mse_measured": False,
                    "predicted_dloss": 5.0,
                },
            },
        }

        breakdown = per_format_predicted_breakdown(assignment, stats, costs)

        self.assertAlmostEqual(breakdown["NVFP4"], 0.5 * 2.0 * 0.25)
        self.assertAlmostEqual(breakdown["MXFP4"], 5.0)

    def test_qwen_packed_moe_profile_allows_only_exportable_expert_formats(self):
        name = "model.layers.0.mlp.experts.gate_up_proj"
        candidates = {
            name: [
                Candidate("NVFP4", 4.5, 100, 1.0),
                Candidate("MXFP8", 8.25, 180, 0.2),
                Candidate("MXFP8_E4M3", 8.25, 180, 0.2),
                Candidate("MXFP4", 4.25, 96, 0.9),
                Candidate("FP8_E4M3", 8.5, 190, 0.1),
                Candidate("BF16", 16.0, 512, 0.0),
            ],
            "model.layers.0.self_attn.q_proj": [
                Candidate("MXFP4", 4.25, 96, 0.9),
            ],
        }

        filtered = filter_candidates_for_profile(
            candidates, "vllm_qwen3_5_packed_moe"
        )

        self.assertEqual(
            [c.fmt for c in filtered[name]],
            ["NVFP4", "MXFP8", "MXFP8_E4M3", "MXFP4", "BF16"],
        )
        self.assertNotIn("model.layers.0.self_attn.q_proj", filtered)

    def test_select_targets_returns_baseline_knee_high(self):
        curve = [
            {"feasible": True, "achieved_bits": 4.5, "predicted_dloss": 10.0},
            {"feasible": True, "achieved_bits": 5.0, "predicted_dloss": 4.0},
            {"feasible": True, "achieved_bits": 6.0, "predicted_dloss": 3.0},
            {"feasible": True, "achieved_bits": 8.0, "predicted_dloss": 2.8},
        ]
        picks = select_targets(curve, "baseline,knee,high")
        self.assertEqual(picks[0], 0)
        self.assertEqual(picks[-1], 3)
        self.assertEqual(len(picks), 3)


class TestCalibrationHooks(unittest.TestCase):
    def test_install_activation_hooks_skips_conflicting_module_formats(self):
        linear = torch.nn.Linear(4, 4, bias=False)
        quant_map = {
            "a.weight": (linear, "weight"),
            "b.weight": (linear, "weight"),
        }
        handles, active, skipped = install_activation_hooks(
            {"a.weight": "NVFP4", "b.weight": "MXFP8"},
            quant_map,
        )
        try:
            self.assertEqual(active, [])
            self.assertEqual(len(skipped), 1)
            self.assertEqual(set(skipped[0]["formats"]), {"MXFP8_E4M3", "NVFP4"})
        finally:
            for handle in handles:
                handle.remove()

    def test_install_activation_hooks_quantizes_input(self):
        linear = torch.nn.Linear(4, 4, bias=False)
        quant_map = {"a.weight": (linear, "weight")}
        handles, active, skipped = install_activation_hooks({"a.weight": "NVFP4"}, quant_map)
        seen = {}

        def recorder(_mod, args):
            seen["input"] = args[0].detach().clone()

        capture = linear.register_forward_pre_hook(recorder)
        x = torch.tensor([[0.13, -0.51, 1.77, -3.25]], dtype=torch.float32)
        try:
            linear(x)
            self.assertEqual(len(active), 1)
            self.assertEqual(skipped, [])
            self.assertFalse(torch.equal(seen["input"], x))
        finally:
            capture.remove()
            for handle in handles:
                handle.remove()


class _ToyExpertsLinearLoop(nn.Module):
    def __init__(self, num_experts=3, hidden=4, intermediate=6):
        super().__init__()
        self.gate_up_proj = nn.ModuleList(
            [nn.Linear(hidden, 2 * intermediate, bias=False) for _ in range(num_experts)]
        )
        self.down_proj = nn.ModuleList(
            [nn.Linear(intermediate, hidden, bias=False) for _ in range(num_experts)]
        )


class _ToyMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 3, bias=False)
        self.experts = _ToyExpertsLinearLoop()


class _ToyRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, 4))


class _ToyMoeBlockCustomRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _ToyRouter()
        self.experts = _ToyExpertsLinearLoop()


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = _ToyMoeBlock()


class TestMoeDiscovery(unittest.TestCase):
    def test_discover_moe_structure_handles_linear_loop_projection_lists(self):
        toy = _ToyModel()
        info = discover_moe_structure(toy)
        self.assertEqual(info["model.layers.0.mlp.experts.gate_up_proj.0"], ("model.layers.0.mlp.gate", "0"))
        self.assertEqual(info["model.layers.0.mlp.experts.down_proj.2"], ("model.layers.0.mlp.gate", "2"))
        self.assertEqual(len(info), 6)

    def test_discover_moe_structure_handles_router_modules_with_weight(self):
        toy = _ToyModel()
        toy.model.layers[0].mlp = _ToyMoeBlockCustomRouter()
        info = discover_moe_structure(toy)
        self.assertEqual(info["model.layers.0.mlp.experts.gate_up_proj.1"], ("model.layers.0.mlp.gate", "1"))
        self.assertEqual(len(info), 6)


if __name__ == "__main__":
    unittest.main()
