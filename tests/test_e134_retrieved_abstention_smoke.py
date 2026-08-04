import unittest

from scripts.audit_e134_retrieved_abstention_smoke_20260713 import compute_gate


class RetrievedAbstentionSmokeGateTest(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "smoke_rows": 2,
            "smoke_changed_rows": 1,
            "smoke_unchanged_rows": 1,
            "smoke_min_accepted": 2,
            "smoke_min_changed_accepted": 1,
            "smoke_min_unchanged_accepted": 1,
        }
        self.rows = [
            {"meta": {"e130_target_mode": "abstain", "candidate_types": ["A"]}},
            {"meta": {"e130_target_mode": "gold_present", "candidate_types": ["B"]}},
        ]

    def test_passes_stratified_candidate_consistent_rows(self):
        raw = [
            {"sample_id": "run_0000", "accepted": True, "final_obj": {"events": []}},
            {
                "sample_id": "run_0001",
                "accepted": True,
                "final_obj": {"events": [{"event_type": "B"}]},
            },
        ]
        self.assertTrue(compute_gate(self.rows, raw, self.protocol)["passed"])

    def test_rejects_candidate_inconsistent_final(self):
        raw = [
            {
                "sample_id": "run_0000",
                "accepted": True,
                "final_obj": {"events": [{"event_type": "Z"}]},
            },
            {"sample_id": "run_0001", "accepted": True, "final_obj": {"events": []}},
        ]
        gate = compute_gate(self.rows, raw, self.protocol)
        self.assertFalse(gate["checks"]["candidate_consistent"])

    def test_rejects_final_verifier_parse_error(self):
        raw = [
            {"sample_id": "run_0000", "accepted": False, "error_stage": "verifier_parse"},
            {"sample_id": "run_0001", "accepted": True, "final_obj": {"events": []}},
        ]
        gate = compute_gate(self.rows, raw, self.protocol)
        self.assertFalse(gate["checks"]["zero_final_verifier_parse_errors"])


if __name__ == "__main__":
    unittest.main()
