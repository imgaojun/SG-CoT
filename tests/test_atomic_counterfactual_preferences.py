import unittest
from collections import Counter

from src.stage2_preference.atomic_counterfactual import (
    ATOMIC_CATEGORIES,
    aggregate_observed_proposals,
    apply_atomic_proposal,
    fallback_proposal,
    label_leaks,
    render_canonical_pair,
    select_quota_assignment,
    shortest_unique_evidence,
)
from src.stage2_preference.reasoning_preference import (
    classify_single_error,
    extract_final_json,
    is_exact,
    recover_offsets_from_evidence,
)


TOKENS = [
    "John",
    "attacked",
    "Mary",
    "and",
    "later",
    "died",
    "after",
    "officials",
    "moved",
    "him",
    "to",
    "Rome",
    ".",
    "They",
    "met",
    ".",
]
CANDIDATES = [
    "Conflict:Attack",
    "Conflict:Demonstrate",
    "Life:Die",
    "Life:Injure",
    "Movement:Transport-Person",
    "Movement:Transport-Artifact",
    "Contact:Meet",
    "Contact:Contact",
]


def argument(role, start, end):
    return {"role": role, "text": " ".join(TOKENS[start:end]), "start": start, "end": end}


def event(event_type, start, end, arguments=None):
    return {
        "event_type": event_type,
        "trigger": {"text": " ".join(TOKENS[start:end]), "start": start, "end": end},
        "arguments": arguments or [],
    }


GOLD = {
    "events": [
        event(
            "Conflict:Attack",
            1,
            2,
            [argument("Attacker", 0, 1), argument("Target", 2, 3)],
        ),
        event("Life:Die", 5, 6, [argument("Victim", 2, 3)]),
        event(
            "Movement:Transport-Person",
            8,
            9,
            [argument("Person", 9, 10), argument("Destination", 11, 12)],
        ),
    ]
}


def input_text():
    return (
        "Text:\nJohn attacked Mary and later died after officials moved him to Rome. "
        "They met.\n\nTokens:\n"
        + " ".join(TOKENS)
        + "\n\nCandidate event types:\n"
        + ", ".join(CANDIDATES)
        + "\n\nSchema cards:\n..."
    )


class AtomicCounterfactualTests(unittest.TestCase):
    def test_five_fallback_operations_are_atomic(self):
        for category in ATOMIC_CATEGORIES:
            with self.subTest(category=category):
                proposal = fallback_proposal(category, GOLD, CANDIDATES, TOKENS)
                self.assertIsNotNone(proposal)
                mutated = apply_atomic_proposal(GOLD, proposal, TOKENS)
                self.assertEqual(classify_single_error(mutated, GOLD), category)

    def test_multiple_error_sample_is_factorized_into_atomic_proposals(self):
        predicted = {
            "events": [
                event("Conflict:Attack", 1, 2, [argument("Attacker", 0, 1)]),
                event("Life:Injure", 5, 6),
                event("Movement:Transport-Person", 9, 10),
                event("Contact:Meet", 14, 15),
            ]
        }
        proposals = aggregate_observed_proposals(
            GOLD,
            [{"recovered": predicted, "sample_seed": 1104, "sample_round": 0, "sample_index": 2}],
            CANDIDATES,
            TOKENS,
        )
        self.assertEqual(set(proposals), set(ATOMIC_CATEGORIES))
        for category, category_proposals in proposals.items():
            self.assertTrue(category_proposals)
            mutated = apply_atomic_proposal(GOLD, category_proposals[0], TOKENS)
            self.assertEqual(classify_single_error(mutated, GOLD), category)

    def test_partial_recovery_offsets_are_skipped_without_crashing(self):
        partial = {
            "events": [
                {
                    "event_type": "Contact:Meet",
                    "trigger": {"text": "met", "start": None, "end": None},
                    "arguments": [
                        {"role": "Entity", "text": "John", "start": None, "end": 1}
                    ],
                }
            ]
        }
        proposals = aggregate_observed_proposals(
            GOLD,
            [{"recovered": partial, "sample_seed": 1104, "sample_round": 0, "sample_index": 0}],
            CANDIDATES,
            TOKENS,
        )
        for category_proposals in proposals.values():
            for proposal in category_proposals:
                mutated = apply_atomic_proposal(GOLD, proposal, TOKENS)
                self.assertEqual(
                    classify_single_error(mutated, GOLD), proposal["category"]
                )

    def test_renderer_is_paired_recoverable_and_label_free(self):
        for category in ATOMIC_CATEGORIES:
            with self.subTest(category=category):
                proposal = fallback_proposal(category, GOLD, CANDIDATES, TOKENS)
                rejected_numeric = apply_atomic_proposal(GOLD, proposal, TOKENS)
                chosen, rejected = render_canonical_pair(
                    GOLD, rejected_numeric, CANDIDATES, TOKENS
                )
                character_ratio = len(chosen) / len(rejected)
                self.assertGreaterEqual(character_ratio, 0.85)
                self.assertLessEqual(character_ratio, 1.15)
                for response in (chosen, rejected):
                    for step in range(1, 7):
                        self.assertIn(f"Step {step}:", response)
                    self.assertEqual(label_leaks(response, input_text()), [])
                self.assertIn("not emitted", chosen + rejected)
                chosen_recovered, chosen_diag = recover_offsets_from_evidence(
                    extract_final_json(chosen), input_text()
                )
                rejected_recovered, rejected_diag = recover_offsets_from_evidence(
                    extract_final_json(rejected), input_text()
                )
                self.assertEqual(chosen_diag["missing_offsets"], 0)
                self.assertEqual(rejected_diag["missing_offsets"], 0)
                self.assertTrue(is_exact(chosen_recovered, GOLD))
                self.assertEqual(classify_single_error(rejected_recovered, GOLD), category)

    def test_shortest_unique_evidence_disambiguates_repeated_surface(self):
        tokens = ["John", "left", ".", "John", "stayed", "."]
        evidence = shortest_unique_evidence(tokens, 3, 4)
        self.assertIn("stayed", evidence)

    def test_min_cost_flow_is_quota_exact_unique_and_deterministic(self):
        options = {}
        for index in range(8):
            options[f"w{index}"] = {}
            for category in ATOMIC_CATEGORIES:
                options[f"w{index}"][category] = {
                    "proposal_source": "observed_atomic" if index < 5 else "deterministic_fallback",
                    "frequency": 2 if index == 0 else 1,
                }
        quotas = {category: 1 for category in ATOMIC_CATEGORIES}
        first = select_quota_assignment(options, quotas, 1140)
        second = select_quota_assignment(options, quotas, 1140)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len({item["wnd_id"] for item in first}), 5)
        self.assertEqual(Counter(item["error_category"] for item in first), Counter(quotas))
        self.assertTrue(all(item["option"]["proposal_source"] == "observed_atomic" for item in first))

    def test_min_cost_flow_fails_instead_of_relaxing_quota(self):
        options = {"w0": {"wrong_type": {"proposal_source": "observed_atomic"}}}
        quotas = {category: 1 for category in ATOMIC_CATEGORIES}
        with self.assertRaisesRegex(ValueError, "infeasible"):
            select_quota_assignment(options, quotas, 1140)

    def test_label_scanner_flags_training_only_hints(self):
        self.assertEqual(label_leaks("This is the chosen path.", "plain source"), ["chosen"])


if __name__ == "__main__":
    unittest.main()
