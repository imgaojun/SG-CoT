import unittest

from scripts.audit_e136_rejected_retry_20260713 import (
    choose_final_record,
    compute_combined_gate,
)
from scripts.build_e136_rejected_retry_manifest_20260713 import select_retry_rows


def source(index, mode="gold_present"):
    return {
        "instruction": "extract",
        "input": "text",
        "output": '{"events":[]}',
        "meta": {
            "e40_source_index": index,
            "e40_sample_id": f"row-{index}",
            "e130_target_mode": mode,
            "candidate_types": ["Life:Die"],
        },
    }


def record(index, accepted=False, error_stage=None):
    return {
        "source_index": index,
        "accepted": accepted,
        "error_stage": error_stage,
        "final_obj": {"events": []},
    }


def protocol_for(rows=3):
    return {
        "source_row_count": rows,
        "parent_accepted_rows": 1,
        "parent_accepted_mode_counts": {
            "gold_present": 1,
            "partial_supported": 0,
            "abstain": 0,
        },
        "retry_rows": rows - 1,
        "retry_mode_counts": {
            "gold_present": rows - 1,
            "partial_supported": 0,
            "abstain": 0,
        },
        "parent_final_verifier_parse_indices": [1],
        "gate": {
            "required_locked_parent_rows": 1,
            "required_retry_coverage": rows - 1,
            "minimum_accepted_total": 2,
            "minimum_gold_present_accepted": 2,
            "minimum_partial_supported_accepted": 0,
            "minimum_abstain_accepted": 0,
            "maximum_selected_final_verifier_parse_errors": 0,
            "maximum_accepted_candidate_inconsistent": 0,
        },
    }


class E136RejectedRetryTest(unittest.TestCase):
    def test_manifest_contains_every_and_only_parent_reject(self):
        rows = [source(i) for i in range(3)]
        raw = [record(0, True), record(1, False, "verifier_parse"), record(2)]
        selected, audit = select_retry_rows(rows, raw, protocol_for())
        self.assertEqual(
            [row["meta"]["e136_original_source_index"] for row in selected],
            [1, 2],
        )
        self.assertEqual(audit["parent_accepted_rows"], 1)
        self.assertEqual(audit["test_rows_read"], 0)

    def test_parent_accept_is_always_locked(self):
        parent = record(0, True)
        retry = record(0, True)
        self.assertIs(choose_final_record(parent, retry), parent)

    def test_parse_clean_reject_can_clear_parent_interface_error(self):
        parent = record(1, False, "verifier_parse")
        retry = record(1, False, None)
        self.assertIs(choose_final_record(parent, retry), retry)

    def test_retry_accept_recovers_yield_without_changing_locked_row(self):
        rows = [source(i) for i in range(3)]
        parent = {
            0: record(0, True),
            1: record(1, False, "verifier_parse"),
            2: record(2),
        }
        retry = {1: record(1, True), 2: record(2)}
        result, accepted = compute_combined_gate(
            rows, parent, retry, {1, 2}, protocol_for()
        )
        self.assertEqual(accepted, {0, 1})
        self.assertEqual(result["retry_accepted_rows"], 1)
        self.assertEqual(result["selected_final_verifier_parse_errors"], 0)
        self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
