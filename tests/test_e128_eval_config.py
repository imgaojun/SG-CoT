import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class E128EvaluationConfigTests(unittest.TestCase):
    def test_frozen_evaluation_contract(self):
        config = json.loads(
            (
                ROOT
                / "configs/generated/stage2_confirmation/e128e_strict_confirmation_eval.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["methods"], ["direct", "sgcot"])
        self.assertEqual(config["seeds"], [42, 8322, 8333])
        self.assertEqual(
            config["splits"], ["test_seen", "test_unseen", "pooled_unseen"]
        )
        self.assertEqual(
            config["decode"],
            {"batch_size": 4, "temperature": 0.0, "max_new_tokens": 1024},
        )
        self.assertEqual(
            config["expected_examples"],
            {"test_seen": 386, "test_unseen": 63, "pooled_unseen": 272},
        )
        self.assertEqual(config["expected_pooled_unique_wnd_ids"], 272)
        self.assertEqual(config["expected_pooled_gold_events"], 317)
        self.assertEqual(
            config["bootstrap"],
            {
                "unit": "wnd_id",
                "paired_by_seed": True,
                "samples": 10000,
                "seed": 20260712,
            },
        )
        self.assertTrue(config["freeze_required"])

    def test_runner_reads_decode_and_bootstrap_settings(self):
        runner = (
            ROOT / "scripts/run_e128_strict_training_eval_20260712.sh"
        ).read_text(encoding="utf-8")
        for name in (
            "EVAL_BATCH_SIZE",
            "EVAL_TEMPERATURE",
            "EVAL_MAX_NEW_TOKENS",
            "BOOTSTRAP_SAMPLES",
            "BOOTSTRAP_SEED",
        ):
            self.assertIn(name, runner)

    def test_runner_enforces_complete_models_and_smoke_before_freeze(self):
        runner = (
            ROOT / "scripts/run_e128_strict_training_eval_20260712.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("assert_completed_model", runner)
        self.assertGreaterEqual(runner.count("assert_smoke_passed"), 3)
        self.assertIn("validate_training_artifact_20260712.py", runner)


if __name__ == "__main__":
    unittest.main()
