import json
import unittest

from scripts.build_e130_retrieved_abstention_data_20260713 import adapt_row, select_smoke
from scripts.generate_strategy_variants_cot_e47_20260606 import generator_prompt


def row(wnd_id, candidates, event_types):
    return {
        "instruction": "extract",
        "input": "Text:\nx\n\nCandidate event types:\n" + ", ".join(candidates),
        "output": {"events": [{"event_type": value} for value in event_types]},
        "meta": {"wnd_id": wnd_id, "candidate_types": candidates},
    }


class E130RetrievedAbstentionDataTest(unittest.TestCase):
    def test_missing_only_becomes_empty_abstention(self):
        adapted = adapt_row(row("w1", ["Life:Die"], ["Conflict:Attack"]), 0)
        self.assertEqual(adapted["meta"]["e130_target_mode"], "abstain")
        self.assertEqual(adapted["meta"]["e130_missing_gold_types"], ["Conflict:Attack"])
        self.assertEqual(adapted["output"], '{"events":[]}')

    def test_mixed_window_keeps_only_supported_events(self):
        adapted = adapt_row(
            row("w2", ["Life:Die"], ["Life:Die", "Conflict:Attack"]), 1
        )
        self.assertEqual(adapted["meta"]["e130_target_mode"], "partial_supported")
        self.assertEqual(adapted["meta"]["e130_supported_target_types"], ["Life:Die"])
        self.assertNotIn("Conflict:Attack", adapted["output"])

    def test_smoke_is_stratified(self):
        rows = []
        for index in range(6):
            candidates = ["Life:Die"]
            gold = ["Conflict:Attack"] if index < 3 else ["Life:Die"]
            rows.append(adapt_row(row(f"w{index}", candidates, gold), index))
        selected = select_smoke(rows, seed=7, changed_count=2, unchanged_count=2)
        modes = [item["meta"]["e130_target_mode"] for item in selected]
        self.assertEqual(sum(mode != "gold_present" for mode in modes), 2)
        self.assertEqual(sum(mode == "gold_present" for mode in modes), 2)

    def test_generation_prompt_preserves_candidate_consistency(self):
        adapted = adapt_row(row("w7", ["Life:Die"], ["Conflict:Attack"]), 7)
        prompt = json.loads(
            generator_prompt(adapted, "e130_retrieved_abstention", "xml_tags")
        )
        self.assertEqual(prompt["retrieved_support_contract"]["target_mode"], "abstain")
        self.assertEqual(prompt["input"]["target_surface_events_to_copy"], {"events": []})


if __name__ == "__main__":
    unittest.main()
