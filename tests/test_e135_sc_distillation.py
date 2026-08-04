import json
import unittest

from scripts.build_e135_sc_distillation_smoke_20260713 import (
    select_rows,
    selection_key,
)
from scripts.build_e135_sc_distillation_targets_20260713 import (
    canonical_structure,
    choose_carrier,
    evaluate_gate,
    strict_parse_response,
)
from scripts.generate_e135_sc_paths_20260713 import (
    validate_existing_prefix,
    validate_existing_shard_prefix,
)


def source_row(index: int) -> dict:
    return {
        "instruction": "extract",
        "input": f"document {index}",
        "output": "teacher",
        "gold_output": {"events": []},
        "meta": {"wnd_id": f"train-{index}", "source_part": "train"},
    }


def event(argument_end: int = 4) -> dict:
    return {
        "events": [
            {
                "event_type": "Conflict:Attack",
                "trigger": {"start": 1, "end": 2},
                "arguments": [
                    {"role": "Target", "start": 3, "end": argument_end}
                ],
            }
        ]
    }


class E135SCDistillationTest(unittest.TestCase):
    def test_hash_selection_is_deterministic_and_train_only(self):
        rows = [source_row(index) for index in range(10)]
        first = select_rows(rows, seed=1350, count=4)
        second = select_rows(rows, seed=1350, count=4)
        self.assertEqual(
            [row["meta"]["wnd_id"] for row in first],
            [row["meta"]["wnd_id"] for row in second],
        )
        for rank, row in enumerate(first):
            meta = row["meta"]
            self.assertEqual(meta["source_part"], "train")
            self.assertEqual(meta["e135_selection_rank"], rank)
            self.assertEqual(
                meta["e135_selection_key"],
                selection_key(1350, meta["e135_source_index"], meta["wnd_id"]),
            )

    def test_hash_selection_rejects_non_train_rows(self):
        rows = [source_row(0)]
        rows[0]["meta"]["source_part"] = "test"
        with self.assertRaisesRegex(ValueError, "non-train"):
            select_rows(rows, seed=1350, count=1)

    def test_disjoint_selection_excludes_parent_ids_and_scopes_metadata(self):
        rows = [source_row(index) for index in range(20)]
        excluded = {"train-2", "train-7", "train-11"}
        selected = select_rows(
            rows,
            seed=1400,
            count=8,
            excluded_wnd_ids=excluded,
            prefix="e140",
        )
        self.assertFalse({row["meta"]["wnd_id"] for row in selected} & excluded)
        for rank, row in enumerate(selected):
            self.assertEqual(row["meta"]["e140_selection_rank"], rank)
            self.assertNotIn("e135_selection_rank", row["meta"])

    def test_strict_tag_parser_accepts_one_complete_response(self):
        payload = {"events": []}
        text = f"<thinking>audit</thinking><final>{json.dumps(payload)}</final>"
        parsed, error = strict_parse_response(text)
        self.assertIsNone(error)
        self.assertEqual(parsed, payload)

    def test_strict_tag_parser_rejects_text_outside_or_duplicate_tags(self):
        payload = json.dumps({"events": []})
        for text in (
            f"prefix<thinking>audit</thinking><final>{payload}</final>",
            f"<thinking>audit</thinking><thinking>x</thinking><final>{payload}</final>",
            f"<thinking></thinking><final>{payload}</final>",
        ):
            parsed, error = strict_parse_response(text)
            self.assertIsNone(parsed)
            self.assertIsNotNone(error)

    def test_carrier_requires_exact_recovered_structure_and_validity(self):
        voted = event()
        samples = [event(), event(argument_end=5), event()]
        self.assertEqual(choose_carrier(voted, samples, [True, True, False]), [0])
        self.assertEqual(canonical_structure(samples[0]), canonical_structure(voted))
        self.assertNotEqual(canonical_structure(samples[1]), canonical_structure(voted))

    def test_gate_requires_yield_and_corrections(self):
        protocol = {
            "gate": {
                "required_generation_rows": 64,
                "minimum_vote_carrier_rows": 36,
                "minimum_vote_gold_exact_rows": 12,
                "minimum_eligible_distillation_rows": 12,
                "minimum_vote_corrects_greedy_rows": 3,
                "maximum_selected_target_parse_errors": 0,
                "maximum_duplicate_wnd_ids": 0,
            }
        }
        counts = {
            "generation_rows": 64,
            "sample_count_errors": 0,
            "rows_below_min_valid_samples": 0,
            "vote_carrier_rows": 36,
            "vote_gold_exact_rows": 12,
            "eligible_distillation_rows": 12,
            "vote_corrects_greedy_rows": 3,
            "selected_target_parse_errors": 0,
            "duplicate_wnd_ids": 0,
        }
        self.assertTrue(evaluate_gate(counts, protocol)["passed"])
        counts["vote_corrects_greedy_rows"] = 2
        failed = evaluate_gate(counts, protocol)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["sc_correction_yield"])

    def test_selection_safety_gate_uses_aggregate_parse_coverage(self):
        protocol = {
            "report_ids": {"target_gate": "e140_gate"},
            "gate": {
                "mode": "selection_safety_aggregate_v1",
                "required_generation_rows": 128,
                "maximum_strict_greedy_parse_errors": 0,
                "minimum_strict_sample_valid_rate": 0.94,
                "maximum_rows_below_min_valid_samples": 6,
                "minimum_vote_carrier_rows": 72,
                "minimum_vote_gold_exact_rows": 20,
                "minimum_eligible_distillation_rows": 20,
                "minimum_vote_corrects_greedy_rows": 6,
                "maximum_selected_target_parse_errors": 0,
                "maximum_selected_target_recovery_errors": 0,
                "maximum_duplicate_wnd_ids": 0,
            },
        }
        counts = {
            "generation_rows": 128,
            "sample_count_errors": 0,
            "strict_greedy_parse_errors": 0,
            "strict_sample_valid_rate": 0.94,
            "rows_below_min_valid_samples": 6,
            "vote_carrier_rows": 72,
            "vote_gold_exact_rows": 20,
            "eligible_distillation_rows": 20,
            "vote_corrects_greedy_rows": 6,
            "selected_target_parse_errors": 0,
            "selected_target_recovery_errors": 0,
            "duplicate_wnd_ids": 0,
        }
        passed = evaluate_gate(counts, protocol)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["id"], "e140_gate")
        counts["selected_target_recovery_errors"] = 1
        failed = evaluate_gate(counts, protocol)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["selected_targets_recovery"])

    def test_training_boundary_gate_reports_greedy_parse_without_gating_on_it(self):
        protocol = {
            "report_ids": {"target_gate": "e142_gate"},
            "gate": {
                "mode": "training_boundary_aggregate_v1",
                "required_generation_rows": 128,
                "minimum_strict_sample_valid_rate": 0.94,
                "maximum_rows_below_min_valid_samples": 6,
                "minimum_vote_carrier_rows": 72,
                "minimum_vote_gold_exact_rows": 20,
                "minimum_eligible_distillation_rows": 20,
                "minimum_vote_corrects_strict_greedy_rows": 6,
                "maximum_selected_target_parse_errors": 0,
                "maximum_selected_target_recovery_errors": 0,
                "maximum_duplicate_wnd_ids": 0,
            },
        }
        counts = {
            "generation_rows": 128,
            "sample_count_errors": 0,
            "strict_greedy_parse_errors": 2,
            "strict_sample_valid_rate": 0.95,
            "rows_below_min_valid_samples": 4,
            "vote_carrier_rows": 80,
            "vote_gold_exact_rows": 30,
            "eligible_distillation_rows": 28,
            "vote_corrects_greedy_rows": 9,
            "vote_corrects_strict_greedy_rows": 7,
            "selected_target_parse_errors": 0,
            "selected_target_recovery_errors": 0,
            "duplicate_wnd_ids": 0,
        }
        passed = evaluate_gate(counts, protocol)
        self.assertTrue(passed["passed"])
        self.assertNotIn("greedy_parse_reliability", passed["checks"])
        counts["vote_corrects_strict_greedy_rows"] = 5
        failed = evaluate_gate(counts, protocol)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["strict_greedy_correction_yield"])

    def test_resume_requires_an_exact_generation_prefix(self):
        manifest = [source_row(0), source_row(1)]
        for index, row in enumerate(manifest):
            row["meta"]["e135_source_index"] = index
        generated = [
            {
                "selection_rank": 0,
                "source_index": 0,
                "wnd_id": "train-0",
                "greedy_text": "greedy",
                "sampled_texts": [f"sample-{index}" for index in range(8)],
            }
        ]
        validate_existing_prefix(generated, manifest, n_samples=8)
        generated[0]["wnd_id"] = "train-1"
        with self.assertRaisesRegex(ValueError, "wnd_id mismatch"):
            validate_existing_prefix(generated, manifest, n_samples=8)

    def test_shard_resume_requires_the_frozen_global_rank_sequence(self):
        manifest = [source_row(index) for index in range(6)]
        for index, row in enumerate(manifest):
            row["meta"]["e143_source_index"] = index
        generated = [
            {
                "selection_rank": rank,
                "source_index": rank,
                "wnd_id": f"train-{rank}",
                "greedy_text": "greedy",
                "sampled_texts": [f"sample-{index}" for index in range(8)],
            }
            for rank in (1, 3)
        ]
        validate_existing_shard_prefix(
            generated, manifest, 8, [1, 3, 5], "e143"
        )
        generated[1]["selection_rank"] = 5
        with self.assertRaisesRegex(ValueError, "non-prefix shard"):
            validate_existing_shard_prefix(
                generated, manifest, 8, [1, 3, 5], "e143"
            )


if __name__ == "__main__":
    unittest.main()
