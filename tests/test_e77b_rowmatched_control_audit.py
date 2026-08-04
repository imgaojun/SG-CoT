from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_e77b_rowmatched_control_20260712 import audit_split


class E77bRowMatchedControlAuditTest(unittest.TestCase):
    def make_rows(self) -> tuple[dict, dict]:
        source = {
            "instruction": (
                "Task. First output `<thinking>...</thinking>` with reasoning. "
                "Then output `<final>{...}</final>` now."
            ),
            "input": "Example text",
            "output": '<thinking>reason</thinking><final>{"events":[]}</final>',
            "gold_output": "[]",
            "meta": {"doc_id": "d1", "wnd_id": "d1-0", "e40_source_index": 7},
        }
        control = {
            "instruction": "Task. Output `<final>{...}</final>` now.",
            "input": "Example text",
            "output": '<final>{"events":[]}</final>',
            "gold_output": "[]",
            "meta": {
                "doc_id": "d1",
                "wnd_id": "d1-0",
                "e40_source_index": 7,
                "control_changed_variable": "remove_thinking_keep_final",
            },
        }
        return source, control

    def write_jsonl(self, path: Path, row: dict) -> None:
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def test_exact_pair_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "source.jsonl"
            control_path = Path(temporary) / "control.jsonl"
            source, control = self.make_rows()
            self.write_jsonl(source_path, source)
            self.write_jsonl(control_path, control)

            report = audit_split(source_path, control_path, expected_count=1)

            self.assertTrue(report["passed"])
            self.assertEqual(report["exact_rows"], 1)
            self.assertEqual(
                report["source_final_sequence_sha256"],
                report["control_output_sequence_sha256"],
            )

    def test_final_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "source.jsonl"
            control_path = Path(temporary) / "control.jsonl"
            source, control = self.make_rows()
            control["output"] = '<final>{"events":[1]}</final>'
            self.write_jsonl(source_path, source)
            self.write_jsonl(control_path, control)

            report = audit_split(source_path, control_path, expected_count=1)

            self.assertFalse(report["passed"])
            self.assertTrue(
                any("byte-identical" in problem for problem in report["problems"])
            )


if __name__ == "__main__":
    unittest.main()
