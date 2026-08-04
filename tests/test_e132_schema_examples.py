import unittest

from scripts.generate_e132_unseen_schema_examples_20260713 import (
    contains_trigger,
    hard_verify,
)


class SchemaExampleVerificationTest(unittest.TestCase):
    def setUp(self):
        self.request = {
            "event_type": "Life:Injure",
            "requested_trigger_cues_min": 2,
            "requested_trigger_cues_max": 4,
            "requested_examples": 2,
        }

    def test_trigger_matching_uses_phrase_boundaries(self):
        self.assertTrue(contains_trigger("Two people were seriously wounded.", "wounded"))
        self.assertFalse(contains_trigger("The unwounded group left.", "wounded"))

    def test_valid_payload_passes(self):
        payload = {
            "event_type": "Life:Injure",
            "trigger_cues": ["wounded", "injured"],
            "examples": [
                {"sentence": "Two people were wounded.", "trigger": "wounded"},
                {"sentence": "The driver was injured.", "trigger": "injured"},
            ],
        }
        self.assertEqual(hard_verify(self.request, payload), [])

    def test_unlocatable_trigger_fails(self):
        payload = {
            "event_type": "Life:Injure",
            "trigger_cues": ["wounded", "injured"],
            "examples": [
                {"sentence": "Two people were hurt.", "trigger": "wounded"},
                {"sentence": "The driver was injured.", "trigger": "injured"},
            ],
        }
        self.assertIn("example_0_trigger_not_locatable", hard_verify(self.request, payload))


if __name__ == "__main__":
    unittest.main()
