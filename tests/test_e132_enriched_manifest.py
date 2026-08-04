import unittest

from scripts.build_e132_enriched_train_manifest_20260713 import (
    learned_cues,
    render_cards,
    render_cards_compact,
)


class EnrichedManifestTest(unittest.TestCase):
    def test_learned_cues_deduplicates_surface_and_lemma(self):
        entry = {
            "surface_cues": [{"cue": "attacks"}, {"cue": "attack"}],
            "lemma_cues": [{"cue": "attack"}, {"cue": "war"}],
            "fallback_schema_cues": [],
        }
        self.assertEqual(learned_cues(entry, 3), ["attacks", "attack", "war"])

    def test_seen_and_unseen_cards_use_distinct_enrichment_sources(self):
        schema = {
            "Seen:A": {"definition": "seen", "trigger_cues": ["a"], "core_roles": ["X"]},
            "Unseen:B": {"definition": "unseen", "trigger_cues": ["b"], "core_roles": ["Y"]},
        }
        lexicon = {
            "Seen:A": {
                "surface_cues": [{"cue": "acted"}],
                "lemma_cues": [{"cue": "act"}],
                "fallback_schema_cues": [],
            }
        }
        unseen = {
            "Unseen:B": {
                "trigger_cues": ["built"],
                "examples": [{"trigger": "built", "sentence": "They built it."}],
            }
        }
        rendered = render_cards(["Seen:A", "Unseen:B"], schema, lexicon, unseen, 6, 2)
        self.assertIn("Learned train trigger forms: acted, act", rendered)
        self.assertIn("Synthetic trigger examples: built -> They built it.", rendered)
        self.assertNotIn("Learned train trigger forms: built", rendered)

    def test_compact_renderer_merges_cues_without_seen_extra_line(self):
        schema = {
            "Seen:A": {"definition": "seen", "trigger_cues": ["act"], "core_roles": ["X"]},
            "Unseen:B": {"definition": "unseen", "trigger_cues": ["make"], "core_roles": ["Y"]},
        }
        lexicon = {
            "Seen:A": {
                "surface_cues": [{"cue": "acted"}],
                "lemma_cues": [{"cue": "act"}],
                "fallback_schema_cues": [],
            }
        }
        unseen = {
            "Unseen:B": {
                "trigger_cues": ["built"],
                "examples": [{"trigger": "built", "sentence": "They built it."}],
            }
        }
        rendered = render_cards_compact(
            ["Seen:A", "Unseen:B"], schema, lexicon, unseen, 6, 8, 2
        )
        self.assertIn("Trigger cues: act, acted", rendered)
        self.assertIn("Trigger cues: make, built", rendered)
        self.assertNotIn("Learned train trigger forms", rendered)


if __name__ == "__main__":
    unittest.main()
