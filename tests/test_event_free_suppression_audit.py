from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_event_free_suppression_20260712 import audit_run


class EventFreeSuppressionAuditTest(unittest.TestCase):
    def write_rows(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_parse_failure_is_not_successful_suppression(self) -> None:
        rows = [
            {
                "input": "negative one",
                "gold": {"events": []},
                "predicted": {"events": []},
                "valid_final_json": False,
            },
            {
                "input": "negative two",
                "gold": {"events": []},
                "predicted": {"events": []},
                "valid_final_json": True,
            },
            {
                "input": "positive ignored",
                "gold": {"events": [{"event_type": "X"}]},
                "predicted": {"events": []},
                "valid_final_json": True,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            self.write_rows(path, rows)
            result = audit_run(path, expected_negative_rows=2)

        self.assertEqual(result["parsed_empty_rows"], 2)
        self.assertEqual(result["strict_valid_empty_rows"], 1)
        self.assertEqual(result["positive_rows_ignored"], 1)

    def test_spurious_events_are_counted(self) -> None:
        rows = [
            {
                "input": "negative",
                "gold": {"events": []},
                "predicted": {"events": [{"event_type": "X"}, {"event_type": "Y"}]},
                "valid_json": True,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            self.write_rows(path, rows)
            result = audit_run(path, expected_negative_rows=1)

        self.assertEqual(result["predicted_event_total"], 2)
        self.assertEqual(result["mean_predicted_events"], 2.0)
        self.assertEqual(result["strict_valid_empty_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
