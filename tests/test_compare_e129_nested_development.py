import unittest

from scripts.compare_e129_nested_development import evaluate_gate, summarize


def comparison(macro_delta, *, precision=-0.005, event_ratio=1.04, json_delta=0.0):
    return {
        "macro_delta": dict(zip(("argument", "event", "trigger"), macro_delta)),
        "trigger_precision_delta": precision,
        "event_ratio_multiplier": event_ratio,
        "json_valid_rate_delta": json_delta,
    }


class CompareE129NestedDevelopmentTest(unittest.TestCase):
    def test_frozen_positive_gate_passes(self):
        splits = {
            "seen": {"mixed_vs_direct": comparison((0.0, -0.005, 0.002))},
            "unseen": {"mixed_vs_direct": comparison((0.01, 0.02, 0.03))},
        }
        gate = evaluate_gate(splits)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["nonnegative_macro_cells"], 5)

    def test_event_overproposal_fails_even_when_f1_improves(self):
        splits = {
            "seen": {"mixed_vs_direct": comparison((0.01, 0.01, 0.01))},
            "unseen": {
                "mixed_vs_direct": comparison((0.01, 0.01, 0.01), event_ratio=1.051)
            },
        }
        gate = evaluate_gate(splits)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["unseen_event_ratio_inflation_at_most_1_05x"])

    def test_summary_computes_event_inflation_from_predictions(self):
        rows = [
            {
                "gold": {
                    "events": [
                        {
                            "event_type": "Justice:Trial-Hearing",
                            "trigger": {"start": 1, "end": 2},
                            "arguments": [],
                        }
                    ]
                },
                "predicted": {
                    "events": [
                        {
                            "event_type": "Justice:Trial-Hearing",
                            "trigger": {"start": 1, "end": 2},
                            "arguments": [],
                        },
                        {
                            "event_type": "Justice:Convict",
                            "trigger": {"start": 3, "end": 4},
                            "arguments": [],
                        },
                    ]
                },
                "argument_f1": 0.0,
                "event_f1": 2 / 3,
                "trigger_f1": 2 / 3,
                "valid_json": True,
                "candidate_types_valid": True,
            }
        ]
        result = summarize(rows)
        self.assertEqual(result["predicted_events"], 2)
        self.assertEqual(result["gold_events"], 1)
        self.assertEqual(result["predicted_to_gold_event_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
