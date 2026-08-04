import unittest

from scripts.analyze_e131_selective_sc_20260713 import canonical_structure, evaluate_gate


class SelectiveSelfConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "max_mean_paths": 3.5,
            "positive_gain_retention": 0.8,
            "max_macro_regression_from_greedy": 0.005,
            "nonpositive_full_sc_tolerance": 0.002,
        }

    def test_canonical_structure_ignores_event_and_argument_order(self):
        first = {
            "events": [
                {
                    "event_type": "A",
                    "trigger": {"start": 4, "end": 5},
                    "arguments": [
                        {"role": "z", "start": 8, "end": 9},
                        {"role": "a", "start": 1, "end": 2},
                    ],
                }
            ]
        }
        second = {
            "events": [
                {
                    "event_type": "A",
                    "trigger": {"start": 4, "end": 5},
                    "arguments": [
                        {"role": "a", "start": 1, "end": 2},
                        {"role": "z", "start": 8, "end": 9},
                    ],
                }
            ]
        }
        self.assertEqual(canonical_structure(first), canonical_structure(second))

    def test_gate_passes_cost_and_positive_gain_retention(self):
        greedy = {metric: 0.4 for metric in ("argument", "event", "trigger")}
        full = {metric: 0.5 for metric in greedy}
        adaptive = {metric: 0.48 for metric in greedy}
        gate = evaluate_gate(greedy, full, adaptive, 3.0, self.protocol)
        self.assertTrue(gate["passed"])

    def test_gate_rejects_excess_cost(self):
        values = {metric: 0.5 for metric in ("argument", "event", "trigger")}
        gate = evaluate_gate(values, values, values, 3.6, self.protocol)
        self.assertFalse(gate["passed"])

    def test_gate_handles_nonpositive_full_sc_metric(self):
        greedy = {metric: 0.5 for metric in ("argument", "event", "trigger")}
        full = dict(greedy, event=0.49)
        adaptive = dict(greedy, event=0.487)
        gate = evaluate_gate(greedy, full, adaptive, 3.0, self.protocol)
        self.assertFalse(gate["metrics"]["event"]["reference_pass"])


if __name__ == "__main__":
    unittest.main()
