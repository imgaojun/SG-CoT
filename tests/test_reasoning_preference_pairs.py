import json
import tempfile
import unittest
from pathlib import Path

from scripts.mine_reasoning_preferences_e110_20260711 import (
    load_completed_window_ids,
    load_sample_records,
    select_balanced_pairs,
    teacher_fallback_allowed,
    windows_requiring_topup,
)
from scripts.build_surface_evidence_dataset_20260712 import shortest_unique_evidence
from src.stage2_data.build_formal_stage2_dataset import resolve_candidate_types
from src.stage2_preference.reasoning_preference import (
    classify_single_error,
    extract_final_json,
    find_heldout_leaks,
    has_complete_reasoning_response,
    is_exact,
    recover_offsets_from_evidence,
    valid_length_pair,
)


def event(event_type, start, end, arguments=None):
    return {
        "event_type": event_type,
        "trigger": {"text": "trigger", "start": start, "end": end},
        "arguments": arguments or [],
    }


def argument(role, start, end):
    return {"role": role, "text": "argument", "start": start, "end": end}


class ReasoningPreferenceTests(unittest.TestCase):
    def test_case_insensitive_final_parsing(self):
        payload = extract_final_json(
            '<THINKING>x</THINKING><FINAL>{"events": []}</FINAL>'
        )
        self.assertEqual(payload, {"events": []})
        self.assertIsNone(extract_final_json("no tagged answer"))
        self.assertTrue(
            has_complete_reasoning_response(
                '<thinking>x</thinking><final>{"events": []}</final>'
            )
        )
        self.assertFalse(
            has_complete_reasoning_response(
                '<THINKING>x</THINKING><FINAL>{"events": []}</FINAL>'
            )
        )
        self.assertFalse(
            has_complete_reasoning_response(
                '<thinking>x</thinking><final>{"events": []}'
            )
        )

    def test_evidence_offset_recovery_uses_local_context(self):
        input_text = (
            "Text:\nJohn called Mary and John left.\n\n"
            "Tokens:\nJohn called Mary and John left .\n\n"
            "Candidate event types:\nContact:Contact\n\nSchema cards:\n..."
        )
        surface = {
            "events": [
                {
                    "event_type": "Contact:Contact",
                    "trigger": {"text": "called", "evidence": "John called Mary"},
                    "arguments": [
                        {"role": "Participant", "text": "John", "evidence": "John called Mary"},
                        {"role": "Participant", "text": "Mary", "evidence": "called Mary and"},
                    ],
                }
            ]
        }
        recovered, diagnostics = recover_offsets_from_evidence(surface, input_text)
        self.assertEqual(diagnostics["missing_offsets"], 0)
        self.assertEqual(recovered["events"][0]["trigger"]["start"], 1)
        self.assertEqual(recovered["events"][0]["arguments"][0]["start"], 0)

    def test_single_error_categories(self):
        gold = {
            "events": [
                event("Conflict:Attack", 1, 2, [argument("Attacker", 0, 1)]),
                event("Life:Die", 5, 6),
            ]
        }
        wrong_type = {
            "events": [
                event("Life:Injure", 1, 2, [argument("Attacker", 0, 1)]),
                event("Life:Die", 5, 6),
            ]
        }
        drift = {
            "events": [
                event("Conflict:Attack", 2, 3, [argument("Attacker", 0, 1)]),
                event("Life:Die", 5, 6),
            ]
        }
        arg_omission = {
            "events": [event("Conflict:Attack", 1, 2), event("Life:Die", 5, 6)]
        }
        event_omission = {
            "events": [
                event("Conflict:Attack", 1, 2, [argument("Attacker", 0, 1)])
            ]
        }
        extra_frame = {
            "events": gold["events"] + [event("Contact:Contact", 8, 9)]
        }
        self.assertTrue(is_exact(gold, gold))
        self.assertEqual(classify_single_error(wrong_type, gold), "wrong_type")
        self.assertEqual(classify_single_error(drift, gold), "trigger_drift")
        self.assertEqual(classify_single_error(arg_omission, gold), "argument_omission")
        self.assertEqual(classify_single_error(event_omission, gold), "event_omission")
        self.assertEqual(classify_single_error(extra_frame, gold), "extra_frame")

    def test_length_filter(self):
        self.assertTrue(valid_length_pair(100, 90, 1000, 990, 1536))
        self.assertFalse(valid_length_pair(100, 50, 1000, 950, 1536))
        self.assertFalse(valid_length_pair(100, 90, 1600, 990, 1536))

    def test_duplicate_samples_are_deduplicated(self):
        record = {
            "wnd_id": "w1",
            "sample_round": 0,
            "samples": [{"sample_seed": 11, "sample_index": 0, "raw_response": "x"}],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            first = Path(temporary_dir) / "samples-1.jsonl"
            second = Path(temporary_dir) / "samples-2.jsonl"
            line = json.dumps(record) + "\n"
            first.write_text(line, encoding="utf-8")
            second.write_text(line, encoding="utf-8")
            loaded = load_sample_records(str(Path(temporary_dir) / "samples-*.jsonl"))
        self.assertEqual(len(loaded["w1"]), 1)
        self.assertEqual(loaded["w1"][0]["sample_round"], 0)

    def test_teacher_fallback_requires_k8_topup_round(self):
        self.assertFalse(teacher_fallback_allowed([{"sample_round": 0}], True))
        self.assertTrue(
            teacher_fallback_allowed(
                [{"sample_round": 0}, {"sample_round": 1}], True
            )
        )
        self.assertFalse(teacher_fallback_allowed([{"sample_round": 1}], False))

    def test_topup_targets_unpaired_before_global_caps(self):
        candidates = [
            {
                "meta": {
                    "wnd_id": "w1",
                    "error_category": "wrong_type",
                    "chosen_source": "sample_exact",
                    "rejected_quality": 0.5,
                }
            }
        ]
        self.assertEqual(
            windows_requiring_topup(["w0", "w1", "w2"], candidates),
            ["w0", "w2"],
        )

    def test_resume_completed_ids_respect_sample_round(self):
        records = [
            {"wnd_id": "w0", "sample_round": 0, "samples": []},
            {"wnd_id": "w1", "sample_round": 1, "samples": []},
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "samples.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            self.assertEqual(load_completed_window_ids({path}, 0), {"w0"})
            self.assertEqual(load_completed_window_ids({path}, 1), {"w1"})

    def test_build_sample_loader_can_freeze_k4_round(self):
        records = [
            {
                "wnd_id": "w0",
                "sample_round": 0,
                "samples": [{"sample_seed": 1, "sample_index": 0}],
            },
            {
                "wnd_id": "w0",
                "sample_round": 1,
                "samples": [{"sample_seed": 2, "sample_index": 0}],
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "samples.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            loaded = load_sample_records(str(path), max_sample_round=0)
        self.assertEqual(len(loaded["w0"]), 1)
        self.assertEqual(loaded["w0"][0]["sample_round"], 0)

    def test_e81_category_cap(self):
        pairs = []
        categories = ["wrong_type"] * 6 + ["trigger_drift"] * 4 + ["extra_frame"] * 4
        for index, category in enumerate(categories):
            pairs.append(
                {
                    "meta": {
                        "wnd_id": f"w{index}",
                        "error_category": category,
                        "chosen_source": "sample_exact",
                        "rejected_quality": 0.5,
                    }
                }
            )
        selected = select_balanced_pairs(pairs, "e81")
        counts = {}
        for pair in selected:
            category = pair["meta"]["error_category"]
            counts[category] = counts.get(category, 0) + 1
        self.assertTrue(selected)
        self.assertLessEqual(max(counts.values()) / len(selected), 0.4)

    def test_teacher_fraction_cap(self):
        pairs = []
        categories = ["wrong_type", "trigger_drift", "extra_frame"]
        for index in range(30):
            pairs.append(
                {
                    "meta": {
                        "wnd_id": f"w{index}",
                        "error_category": categories[index % len(categories)],
                        "chosen_source": "sample_exact" if index < 20 else "verified_teacher_trace",
                        "rejected_quality": 0.5,
                    }
                }
            )
        selected = select_balanced_pairs(pairs, "e81")
        teacher_count = sum(
            pair["meta"]["chosen_source"] == "verified_teacher_trace" for pair in selected
        )
        self.assertLessEqual(teacher_count / len(selected), 0.3)

    def test_heldout_leak_scan(self):
        row = {
            "instruction": "Do extraction",
            "input": "Schema for Justice:Sentence",
            "meta": {"cluster": ["Life:Die"]},
        }
        leaks = find_heldout_leaks(row, ["Justice:Sentence", "Contact:Broadcast"])
        self.assertEqual(leaks, [{"path": "$.input", "event_type": "Justice:Sentence"}])

    def test_shortest_unique_evidence_disambiguates_repeated_surface(self):
        tokens = ["John", "called", "Mary", "and", "John", "left", "."]
        self.assertEqual(shortest_unique_evidence(tokens, 4, 5), "and John")

    def test_seen_only_scope_filters_prediction_leaks(self):
        row = {
            "wnd_id": "w1",
            "event_mentions": [event("Life:Die", 1, 2)],
        }
        resolved = resolve_candidate_types(
            row=row,
            schema_by_type={},
            candidate_universe=["Life:Die", "Conflict:Attack"],
            candidate_source="oracle_anchor_predicted",
            top_k=2,
            prediction_map={
                "w1": {
                    "ranked_types": ["Justice:Sentence", "Conflict:Attack", "Life:Die"]
                }
            },
            seed=13,
            candidate_order_mode="as_is",
        )
        self.assertNotIn("Justice:Sentence", resolved["candidate_types"])
        self.assertNotIn("Justice:Sentence", resolved["raw_predicted_topk"])


if __name__ == "__main__":
    unittest.main()
