import json
import unittest
from pathlib import Path

import scripts.generate_strategy_variants_cot_e47_20260606 as generator
from src.stage2_preference.reasoning_preference import find_heldout_leaks


REPO = Path(__file__).resolve().parents[1]


def strict_prompt_row() -> dict:
    return {
        "input": (
            "Text:\nThe blast killed two workers.\n\n"
            "Tokens:\nThe blast killed two workers .\n\n"
            "Candidate event types:\nConflict:Attack, Life:Die\n\n"
            "Schema cards:\n"
            "[1] Event type: Conflict:Attack\n"
            "Definition: A violent physical attack is carried out against a target.\n"
            "Trigger cues: attack, blast\n"
            "Core roles: Attacker, Target, Place, Instrument\n\n"
            "[2] Event type: Life:Die\n"
            "Definition: A person dies.\n"
            "Trigger cues: died, killed, death\n"
            "Core roles: Victim, Agent, Place, Instrument\n\n"
            "Return JSON only."
        ),
        "gold_output": json.dumps(
            {
                "events": [
                    {
                        "event_type": "Life:Die",
                        "trigger": {"text": "killed", "start": 2, "end": 3},
                        "arguments": [
                            {"role": "Victim", "text": "workers", "start": 4, "end": 5}
                        ],
                    }
                ]
            }
        ),
        "meta": {"e40_sample_id": "e111_strict_sgcot_0000"},
    }


class StrictSgCotPromptTests(unittest.TestCase):
    def test_e111_autocluster_prompt_has_no_heldout_or_contact_hardcoding(self):
        generator.AUTO_CLUSTER_MAP_PATH = str(
            REPO / "data/schema/richere-en.auto_cluster_map.json"
        )
        generator.AUTO_CLUSTER_MAP_CACHE = None
        prompt = generator.generator_prompt(
            strict_prompt_row(), prompt_profile="e95_trigger_locked_autocluster"
        )
        heldout_types = json.loads(
            (
                REPO
                / "data/processed/type_holdout/richere-en/balanced-subtype-v1/split1/unseen_types.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(generator.variant_key_from_name("e111_strict_sgcot"), "e83")
        self.assertFalse(find_heldout_leaks(prompt, heldout_types))
        self.assertNotIn("contact", prompt.lower())


if __name__ == "__main__":
    unittest.main()
