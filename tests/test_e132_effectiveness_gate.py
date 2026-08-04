import unittest

from scripts.build_e132_unseen_after_dev_gate_20260713 import require_dev_gate
from scripts.evaluate_e132_effectiveness_gate_20260713 import build_gate


def summary(argument, event, trigger, valid=1.0, recovery=0.95):
    return {
        "argument_f1": argument,
        "event_f1": event,
        "trigger_f1": trigger,
        "final_json_valid_rate": valid,
        "offset_recovery_full_rate": recovery,
    }


class E132EffectivenessGateTest(unittest.TestCase):
    def test_dev_gate_passes_at_all_frozen_floors(self):
        rules = {
            "maximum_macro_regression_per_metric": 0.015,
            "maximum_json_valid_rate_regression": 0.01,
            "maximum_offset_recovery_rate_regression": 0.01,
        }
        baseline = summary(0.44, 0.33, 0.72)
        candidate = summary(0.425, 0.315, 0.705, valid=0.99, recovery=0.94)
        self.assertTrue(build_gate("dev_seen", baseline, candidate, rules)["passed"])

    def test_dev_gate_fails_one_metric_below_floor(self):
        rules = {
            "maximum_macro_regression_per_metric": 0.015,
            "maximum_json_valid_rate_regression": 0.01,
            "maximum_offset_recovery_rate_regression": 0.01,
        }
        gate = build_gate(
            "dev_seen", summary(0.44, 0.33, 0.72), summary(0.424, 0.33, 0.72), rules
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["argument_f1_retention"])

    def test_unseen_gate_requires_absolute_and_relative_trigger_gain(self):
        rules = {
            "minimum_trigger_f1": 0.285,
            "minimum_trigger_delta": 0.03,
            "maximum_argument_regression": 0.015,
            "maximum_event_regression": 0.015,
            "maximum_json_valid_rate_regression": 0.01,
            "maximum_offset_recovery_rate_regression": 0.01,
        }
        baseline = summary(0.1935, 0.1443, 0.2508, recovery=0.9024)
        candidate = summary(0.18, 0.13, 0.285, valid=0.99, recovery=0.895)
        self.assertTrue(build_gate("test_unseen", baseline, candidate, rules)["passed"])
        candidate["trigger_f1"] = 0.28
        self.assertFalse(build_gate("test_unseen", baseline, candidate, rules)["passed"])

    def test_unseen_builder_rejects_failed_dev_gate(self):
        class GatePath:
            def read_text(self, encoding):
                del encoding
                return '{"id":"e132_dev_seen_effectiveness_gate_v1","passed":false,"test_rows_read":0}'

        with self.assertRaisesRegex(ValueError, "passing frozen"):
            require_dev_gate(GatePath())


if __name__ == "__main__":
    unittest.main()
