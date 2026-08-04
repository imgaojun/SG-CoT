import unittest

from src.stage2_quality_validation.generation_diagnostics import (
    completion_token_diagnostics,
    output_contract_diagnostics,
)


class CompletionTokenDiagnosticsTests(unittest.TestCase):
    def test_eos_excludes_eos_and_batch_padding(self):
        result = completion_token_diagnostics(
            [10, 11, 2, 2], eos_token_id=2, max_new_tokens=4
        )
        self.assertEqual(result["generated_token_count"], 2)
        self.assertTrue(result["generation_ended_with_eos"])
        self.assertFalse(result["hit_max_new_tokens"])

    def test_cap_without_eos_is_reported(self):
        result = completion_token_diagnostics(
            [10, 11, 12, 13], eos_token_id=2, max_new_tokens=4
        )
        self.assertEqual(result["generated_token_count"], 4)
        self.assertFalse(result["generation_ended_with_eos"])
        self.assertTrue(result["hit_max_new_tokens"])

    def test_short_non_eos_completion_is_not_labeled_truncated(self):
        result = completion_token_diagnostics(
            [10, 11], eos_token_id=[2, 3], max_new_tokens=4
        )
        self.assertEqual(result["generated_token_count"], 2)
        self.assertFalse(result["generation_ended_with_eos"])
        self.assertFalse(result["hit_max_new_tokens"])

    def test_output_contract_requires_complete_lowercase_tags_and_candidate_types(self):
        result = output_contract_diagnostics(
            '<thinking>reason</thinking><final>{"events": []}</final>',
            {"events": [{"event_type": "Seen:Type"}]},
            candidate_types=["Seen:Type"],
            expects_reasoning=True,
        )
        self.assertTrue(all(value is True for value in result.values()))

    def test_output_contract_rejects_open_final_and_out_of_candidate_type(self):
        result = output_contract_diagnostics(
            '<thinking>reason</thinking><final>{"events": []}',
            {"events": [{"event_type": "Other:Type"}]},
            candidate_types=["Seen:Type"],
            expects_reasoning=True,
        )
        self.assertFalse(result["final_tag_complete"])
        self.assertTrue(result["reasoning_tag_complete"])
        self.assertTrue(result["surface_event_list_valid"])
        self.assertFalse(result["candidate_types_valid"])


if __name__ == "__main__":
    unittest.main()
