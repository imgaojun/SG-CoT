import collections
import unittest

from scripts.build_e132_trigger_lexicon_20260713 import (
    normalize_surface,
    rank_counts,
    simple_lemma,
)


class TriggerLexiconTest(unittest.TestCase):
    def test_surface_normalization(self):
        self.assertEqual(normalize_surface("  Filed   For Bankruptcy "), "filed for bankruptcy")

    def test_deterministic_lemma_handles_common_inflections(self):
        self.assertEqual(simple_lemma("attacks"), "attack")
        self.assertEqual(simple_lemma("died"), "die")
        self.assertEqual(simple_lemma("fighting"), "fight")

    def test_rank_counts_is_frequency_then_lexical_and_thresholded(self):
        counts = collections.Counter({"war": 4, "attack": 4, "bombing": 1})
        self.assertEqual(
            rank_counts(counts, minimum=2, top_k=2),
            [{"cue": "attack", "count": 4}, {"cue": "war", "count": 4}],
        )


if __name__ == "__main__":
    unittest.main()
