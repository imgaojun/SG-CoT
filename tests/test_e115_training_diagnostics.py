from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_e115_training_diagnostics_20260712 import (
    document_id,
    select_balanced_style_diagnostic,
    select_doc_diverse_smoke,
    validate_selection,
)
from scripts.compare_e115_margin_gate_20260712 import compare_margin_rows
from scripts.compare_e118_difference_masked_gate_20260712 import runtime_mask_evidence
from src.stage2_preference.atomic_counterfactual import ATOMIC_CATEGORIES
from src.stage2_preference.difference_masking import (
    divergent_token_indices,
    mask_pair_labels,
)


def preference_row(category: str, doc_index: int, window_index: int = 0):
    wnd_id = f"doc-{category}-{doc_index:03d}-{window_index}"
    return {
        "instruction": "instruction",
        "input": "input",
        "chosen": "<thinking>x</thinking><final>{}</final>",
        "rejected": "<thinking>y</thinking><final>{}</final>",
        "meta": {"wnd_id": wnd_id, "error_category": category},
    }


def style_row(category: str, doc_index: int, window_index: int = 0):
    row = preference_row(category, doc_index, window_index)
    row["canonical"] = row.pop("chosen")
    row["native"] = row.pop("rejected")
    row["meta"]["document_id"] = document_id(row["meta"]["wnd_id"])
    return row


class E115TrainingDiagnosticTests(unittest.TestCase):
    def test_runtime_mask_evidence_requires_nontrivial_single_marker(self):
        marker = {
            "event": "e118_difference_mask_active",
            "batch_pairs": 1,
            "context_tokens": 1,
            "chosen_kept_tokens": 12,
            "chosen_response_tokens": 100,
            "rejected_kept_tokens": 11,
            "rejected_response_tokens": 98,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text("prefix\n" + json.dumps(marker) + "\n")
            valid, evidence = runtime_mask_evidence(path)
        self.assertTrue(valid)
        self.assertEqual(evidence["marker_count"], 1)

    def test_difference_mask_keeps_only_changed_tokens_and_context(self):
        chosen = [10, 11, 12, 13, 14, 15]
        rejected = [10, 11, 99, 13, 14, 15]
        chosen_keep, rejected_keep = divergent_token_indices(
            chosen, rejected, context_tokens=1
        )
        self.assertEqual(chosen_keep, [1, 2, 3])
        self.assertEqual(rejected_keep, [1, 2, 3])

    def test_pair_label_mask_preserves_prompt_mask_and_nonempty_decisions(self):
        chosen = [-100, -100, 10, 11, 12, 13, 14]
        rejected = [-100, -100, 10, 11, 99, 13, 14]
        masked_chosen, masked_rejected, stats = mask_pair_labels(
            chosen, rejected, context_tokens=1
        )
        self.assertEqual(masked_chosen, [-100, -100, -100, 11, 12, 13, -100])
        self.assertEqual(masked_rejected, [-100, -100, -100, 11, 99, 13, -100])
        self.assertEqual(stats["chosen_kept_tokens"], 3)
        self.assertEqual(stats["rejected_kept_tokens"], 3)

    def test_difference_mask_handles_insertion_with_context_on_both_sides(self):
        chosen_keep, rejected_keep = divergent_token_indices(
            [1, 2, 3, 4], [1, 2, 8, 9, 3, 4], context_tokens=1
        )
        self.assertTrue(chosen_keep)
        self.assertTrue(rejected_keep)
        self.assertIn(2, chosen_keep)
        self.assertIn(2, rejected_keep)
        self.assertIn(3, rejected_keep)

    def test_doc_diverse_smoke_is_balanced_deterministic_and_globally_unique(self):
        rows = [
            preference_row(category, doc_index)
            for category in ATOMIC_CATEGORIES
            for doc_index in range(12)
        ]
        first = select_doc_diverse_smoke(rows, per_category=8, seed=1150)
        second = select_doc_diverse_smoke(list(reversed(rows)), per_category=8, seed=1150)
        self.assertEqual(
            [row["meta"]["wnd_id"] for row in first],
            [row["meta"]["wnd_id"] for row in second],
        )
        self.assertEqual(
            Counter(row["meta"]["error_category"] for row in first),
            Counter({category: 8 for category in ATOMIC_CATEGORIES}),
        )
        docs = [document_id(row["meta"]["wnd_id"]) for row in first]
        self.assertEqual(len(docs), 40)
        self.assertEqual(len(set(docs)), 40)

    def test_style_selection_balances_and_deduplicates_documents_within_category(self):
        rows = [
            style_row(category, doc_index, window_index)
            for category in ATOMIC_CATEGORIES
            for doc_index in range(45)
            for window_index in range(2)
        ]
        selected = select_balanced_style_diagnostic(rows, per_category=40, seed=1150)
        self.assertEqual(len(selected), 200)
        for category in ATOMIC_CATEGORIES:
            category_rows = [
                row for row in selected if row["meta"]["error_category"] == category
            ]
            self.assertEqual(len(category_rows), 40)
            self.assertEqual(
                len({row["meta"]["document_id"] for row in category_rows}), 40
            )

    def test_validate_selection_rejects_duplicate_smoke_documents(self):
        smoke = [
            preference_row(category, index)
            for category in ATOMIC_CATEGORIES
            for index in range(2)
        ]
        style = [
            style_row(category, index)
            for category in ATOMIC_CATEGORIES
            for index in range(2)
        ]
        smoke[1]["meta"]["wnd_id"] = (
            smoke[0]["meta"]["wnd_id"].rsplit("-", 1)[0] + "-9"
        )
        with self.assertRaisesRegex(ValueError, "duplicate documents"):
            validate_selection(smoke, style, smoke_per_category=2, style_per_category=2)

    def test_margin_comparison_is_paired_and_category_aware(self):
        before = {
            "rows": [
                {"wnd_id": "a-1", "error_category": "extra_frame", "margin": -0.2},
                {"wnd_id": "b-1", "error_category": "trigger_drift", "margin": 0.1},
            ]
        }
        after = {
            "rows": [
                {"wnd_id": "a-1", "error_category": "extra_frame", "margin": -0.1},
                {"wnd_id": "b-1", "error_category": "trigger_drift", "margin": 0.15},
            ]
        }
        rows, category_deltas = compare_margin_rows(before, after)
        self.assertAlmostEqual(rows[0]["margin_delta"], 0.1)
        self.assertAlmostEqual(rows[1]["margin_delta"], 0.05)
        self.assertAlmostEqual(category_deltas["extra_frame"], 0.1)
        self.assertAlmostEqual(category_deltas["trigger_drift"], 0.05)


if __name__ == "__main__":
    unittest.main()
