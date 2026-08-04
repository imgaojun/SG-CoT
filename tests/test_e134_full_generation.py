import unittest

from scripts.audit_e134_full_generation_20260713 import compute_full_gate


class FullGenerationGateTest(unittest.TestCase):
    def test_balanced_synthetic_full_gate(self):
        protocol = {
            "full_rows": 3,
            "full_gold_present_rows": 1,
            "full_partial_supported_rows": 1,
            "full_abstain_rows": 1,
            "full_min_accepted": 3,
            "full_min_gold_present_accepted": 1,
            "full_min_partial_supported_accepted": 1,
            "full_min_abstain_accepted": 1,
        }
        modes = ("gold_present", "partial_supported", "abstain")
        rows = [
            {"meta": {"e130_target_mode": mode, "candidate_types": ["A"]}}
            for mode in modes
        ]
        raw = [
            {"sample_id": f"run_{index:04d}", "accepted": True, "final_obj": {"events": []}}
            for index in range(3)
        ]
        self.assertTrue(compute_full_gate(rows, raw, protocol)["passed"])


if __name__ == "__main__":
    unittest.main()
