from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from scripts.merge_e120_transfer_balanced_deltas_20260712 import main as merge_main
from src.stage2_preference.transfer_balanced_composition import (
    CATEGORIES,
    combine_tensor,
    solve_maximin_weights,
)


class TransferBalancedCompositionTest(unittest.TestCase):
    def test_maximin_balances_conflicting_experts(self) -> None:
        masked = {
            "a": {"x": 2.0, "y": -1.0},
            "b": {"x": -1.0, "y": 2.0},
        }
        full = {
            "a": {"x": 0.2, "y": 0.1},
            "b": {"x": 0.1, "y": 0.2},
        }

        result = solve_maximin_weights(
            masked, full, experts=("a", "b"), categories=("x", "y")
        )

        self.assertAlmostEqual(result["weights"]["a"], 0.5, places=8)
        self.assertAlmostEqual(result["weights"]["b"], 0.5, places=8)
        self.assertAlmostEqual(
            result["maximin_masked_margin_delta"], 0.5, places=8
        )
        self.assertGreater(
            min(result["predicted_full_response_margin_deltas"].values()), 0
        )

    def test_maximin_rejects_infeasible_full_response_floor(self) -> None:
        masked = {"a": {"x": 1.0}, "b": {"x": 1.0}}
        full = {"a": {"x": -1.0}, "b": {"x": -2.0}}

        with self.assertRaisesRegex(ValueError, "infeasible"):
            solve_maximin_weights(
                masked, full, experts=("a", "b"), categories=("x",)
            )

    def test_maximin_reports_nonpositive_optimum(self) -> None:
        masked = {
            "a": {"x": 1.0, "y": -2.0},
            "b": {"x": -2.0, "y": 1.0},
        }
        full = {
            "a": {"x": 1.0, "y": 1.0},
            "b": {"x": 1.0, "y": 1.0},
        }

        result = solve_maximin_weights(
            masked, full, experts=("a", "b"), categories=("x", "y")
        )

        self.assertAlmostEqual(
            result["maximin_masked_margin_delta"], -0.5, places=8
        )

    def test_combine_tensor_applies_scaled_convex_delta(self) -> None:
        base = torch.tensor([1.0, 2.0], dtype=torch.float32)
        first = torch.tensor([2.0, 2.0], dtype=torch.float32)
        second = torch.tensor([1.0, 4.0], dtype=torch.float32)

        merged = combine_tensor(base, [first, second], [0.25, 0.75], 2.0)

        self.assertTrue(torch.equal(merged, torch.tensor([1.5, 5.0])))

    def test_combine_tensor_requires_identical_nonfloating_values(self) -> None:
        base = torch.tensor([1, 2], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "non-floating"):
            combine_tensor(base, [torch.tensor([1, 3])], [1.0], 1.0)

    def test_sharded_checkpoint_composition_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            base.mkdir()
            base_tensors = {
                "first": torch.tensor([1.0, 2.0]),
                "second": torch.tensor([3.0]),
            }
            save_file(base_tensors, base / "model-00001-of-00001.safetensors")
            index = {
                "metadata": {"total_size": 12},
                "weight_map": {
                    key: "model-00001-of-00001.safetensors" for key in base_tensors
                },
            }
            (base / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            (base / "config.json").write_text("{}\n", encoding="utf-8")
            experts = {}
            for offset, category in enumerate(CATEGORIES, start=1):
                expert = root / category
                expert.mkdir()
                save_file(
                    {key: tensor + offset for key, tensor in base_tensors.items()},
                    expert / "model-00001-of-00001.safetensors",
                )
                (expert / "model.safetensors.index.json").write_text(
                    json.dumps(index), encoding="utf-8"
                )
                experts[category] = expert
            weights_path = root / "weights.json"
            weights_path.write_text(
                json.dumps(
                    {
                        "weights": {category: 0.2 for category in CATEGORIES},
                        "composition_scale": 5.0,
                        "composition_authorized": False,
                        "frozen": True,
                        "test_data_access": False,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            arguments = [
                "merge",
                "--base_model",
                str(base),
                "--weights_json",
                str(weights_path),
                "--output_dir",
                str(output),
                "--composition_scale",
                "5.0",
            ]
            for category in CATEGORIES:
                arguments.extend(["--expert", f"{category}={experts[category]}"])

            with patch.object(sys, "argv", arguments):
                with self.assertRaisesRegex(ValueError, "did not authorize"):
                    merge_main()

            weights_path.write_text(
                json.dumps(
                    {
                        "weights": {category: 0.2 for category in CATEGORIES},
                        "composition_scale": 5.0,
                        "composition_authorized": True,
                        "frozen": True,
                        "test_data_access": False,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(sys, "argv", arguments):
                self.assertEqual(merge_main(), 0)

            merged = load_file(output / "model-00001-of-00001.safetensors")
            self.assertTrue(torch.equal(merged["first"], torch.tensor([16.0, 17.0])))
            self.assertTrue(torch.equal(merged["second"], torch.tensor([18.0])))
            manifest = json.loads(
                (output / "composition_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["composition_scale"], 5.0)
            self.assertFalse(manifest["test_data_access"])


if __name__ == "__main__":
    unittest.main()
