import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_e129_nested_strict_development import (
    assert_no_leaks,
    assert_unique_wnd_ids,
    event_types,
    filtered_events,
)


class E129NestedStrictDevelopmentTest(unittest.TestCase):
    def test_event_filter_keeps_only_pseudo_unseen_type(self):
        row = {
            "wnd_id": "w1",
            "event_mentions": [
                {"event_type": "Justice:Trial-Hearing"},
                {"event_type": "Life:Die"},
            ],
        }
        filtered = filtered_events(row, {"Justice:Trial-Hearing"})
        self.assertEqual(event_types(filtered), {"Justice:Trial-Hearing"})
        self.assertEqual(event_types(row), {"Justice:Trial-Hearing", "Life:Die"})

    def test_duplicate_window_ids_fail(self):
        with self.assertRaises(ValueError):
            assert_unique_wnd_ids([{"wnd_id": "w1"}, {"wnd_id": "w1"}], "test")

    def test_canonical_type_leak_fails(self):
        with self.assertRaises(ValueError):
            assert_no_leaks(
                [{"instruction": "contrast Justice:Trial-Hearing", "input": "x", "output": "y"}],
                ["Justice:Trial-Hearing"],
                "test",
            )


if __name__ == "__main__":
    unittest.main()
