import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.generate_strategy_variants_cot_e47_20260606 as generator
from scripts.audit_e121_confirmation_data_20260712 import candidates
from scripts.build_surface_evidence_dataset_20260712 import (
    offset_aligned_surface_text,
    shortest_unique_evidence,
)
from scripts.build_e121_type_holdout_20260712 import derive_types
from scripts.compare_e121_confirmation_n3_20260712 import apply_gate
from scripts.e121_freeze_manifest_20260712 import build as build_manifest
from scripts.e121_freeze_manifest_20260712 import verify as verify_manifest
from scripts.audit_e122_verifier_budget_smoke_20260712 import evaluate_smoke
from scripts.build_e124_disjoint_train_manifest_20260712 import select_disjoint_rows
from scripts.audit_e124b_full_generation_20260712 import evaluate_full
from scripts.audit_e124b_rejection_causes_20260712 import (
    audit as audit_e124b_rejections,
    classify_rejection,
)
from scripts.reverify_e125_role_alias_20260712 import (
    select_last_hard_valid_attempt,
    summarize as summarize_e125,
    verify_one as verify_one_e125,
)
from scripts.run_e124c_independent_deepseek_audit_20260712 import (
    audit_one,
    normalized_surface_hard_verify,
    select_audit_rows,
)
from scripts.build_e126_deterministic_trace_dataset_20260712 import (
    build_dataset as build_e126_dataset,
    reconstructed_output as reconstructed_e126_output,
)
from scripts.build_e127_deterministic_exact_trace_dataset_20260712 import (
    build_dataset as build_e127_dataset,
)
from src.data_preprocessing.type_holdout.generate_type_holdout import load_jsonl
from src.stage2_preference.reasoning_preference import find_heldout_leaks


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "balanced-subtype-v2-confirmation"
EXPECTED = [
    "Business:Start-Org",
    "Conflict:Demonstrate",
    "Justice:Charge-Indict",
    "Life:Marry",
    "Personnel:End-Position",
    "Transaction:Transfer-Ownership",
]


class E121ConfirmationTests(unittest.TestCase):
    def test_preregistered_selection_rule_derives_exact_types(self):
        protocol_map = json.loads(
            (REPO / "configs/seen_unseen_type_holdout_protocols.json").read_text()
        )["richere-en"]
        prior = {
            event_type
            for name, event_types in protocol_map.items()
            if name != PROTOCOL
            for event_type in event_types
        }
        schema = json.loads((REPO / "data/schema/richere-en.event_schema.json").read_text())
        train = load_jsonl(REPO / "data/processed/textee/richere-en/split1/train.jsonl")
        selected, _ = derive_types(
            train, [entry["event_type"] for entry in schema], prior, 20, 250
        )
        self.assertEqual(selected, EXPECTED)

    def test_raw_pooled_cardinality_is_frozen(self):
        heldout = set(EXPECTED)
        raw_rows = 0
        unique = {}
        for split in range(1, 6):
            rows = load_jsonl(
                REPO / f"data/processed/textee/richere-en/split{split}/test.jsonl"
            )
            for row in rows:
                events = [
                    event
                    for event in row["event_mentions"]
                    if event["event_type"] in heldout
                ]
                if not events:
                    continue
                raw_rows += 1
                signature = (
                    row["text"],
                    tuple(
                        sorted(
                            (
                                event["event_type"],
                                event["trigger"]["start"],
                                event["trigger"]["end"],
                            )
                            for event in events
                        )
                    ),
                )
                if row["wnd_id"] in unique:
                    self.assertEqual(unique[row["wnd_id"]], signature)
                unique[row["wnd_id"]] = signature
        mentions = sum(len(signature[1]) for signature in unique.values())
        self.assertEqual((raw_rows, len(unique), mentions), (332, 272, 317))

    def test_surface_text_falls_back_to_the_gold_offset_span(self):
        tokens = "the ECB # 2 or # 3 take over".split()
        self.assertEqual(
            offset_aligned_surface_text(tokens, 2, 7, "#2 or #3"),
            "# 2 or # 3",
        )
        self.assertEqual(
            offset_aligned_surface_text(tokens, 7, 9, "take over"),
            "take over",
        )

    def test_evidence_disambiguates_adjacent_repeated_surface(self):
        tokens = "Iran Iran said they would respond".split()
        evidence = shortest_unique_evidence(tokens, 1, 2, "Iran")
        self.assertNotEqual(evidence, "Iran Iran")
        self.assertIn("Iran", evidence)

    def test_e121_autocluster_prompt_uses_only_row_candidates(self):
        row = {
            "input": (
                "Text:\nThe blast killed two workers.\n\n"
                "Tokens:\nThe blast killed two workers .\n\n"
                "Candidate event types:\nConflict:Attack, Life:Die\n\n"
                "Schema cards:\n[1] Event type: Conflict:Attack\nDefinition: attack\n"
                "Trigger cues: blast\nCore roles: Target\n\n"
                "[2] Event type: Life:Die\nDefinition: death\n"
                "Trigger cues: killed\nCore roles: Victim\n\nReturn JSON only."
            ),
            "gold_output": json.dumps(
                {
                    "events": [
                        {
                            "event_type": "Life:Die",
                            "trigger": {"text": "killed", "start": 2, "end": 3},
                            "arguments": [],
                        }
                    ]
                }
            ),
            "meta": {"e40_sample_id": "e121_test_0000"},
        }
        generator.AUTO_CLUSTER_MAP_PATH = str(
            REPO / "data/schema/richere-en.auto_cluster_map.json"
        )
        generator.AUTO_CLUSTER_MAP_CACHE = None
        prompt = generator.generator_prompt(
            row, prompt_profile="e95_trigger_locked_autocluster"
        )
        self.assertEqual(generator.variant_key_from_name("e121_confirmation"), "e83")
        self.assertEqual(generator.variant_key_from_name("e122a_verifier4096"), "e83")
        self.assertEqual(generator.variant_key_from_name("e123a_glm51_high4096"), "e83")
        self.assertEqual(generator.variant_key_from_name("e124a_glm51_selfverify"), "e83")
        self.assertEqual(generator.variant_key_from_name("e126_deterministic_hard_valid"), "e83")
        self.assertEqual(generator.variant_key_from_name("e127_deterministic_exact"), "e83")
        self.assertFalse(find_heldout_leaks(prompt, EXPECTED))
        self.assertEqual(candidates(row["input"]), ["Conflict:Attack", "Life:Die"])

    def test_seeded_priority_sampling_is_reproducible_and_not_a_prefix(self):
        rows = load_jsonl(
            REPO
            / "data/stage2_confirmation_e121/"
            "richere_v2confirm_split1_strict_seenonly_oracle_mixed_noise_top10_shuffle_"
            "sgcot_target_train_pos.jsonl"
        )
        first = generator.e40.sample_rows(rows, 1500, 1111, "e121_test")
        second = generator.e40.sample_rows(rows, 1500, 1111, "e121_test")
        first_ids = [row["meta"]["wnd_id"] for row in first]
        second_ids = [row["meta"]["wnd_id"] for row in second]
        prefix_ids = [row["meta"]["wnd_id"] for row in rows[:1500]]
        self.assertEqual(first_ids, second_ids)
        self.assertNotEqual(first_ids, prefix_ids)
        self.assertEqual(len(set(first_ids)), 1500)

    def test_litellm_key_resolution_prefers_native_then_gateway_then_legacy(self):
        with patch.dict(
            os.environ,
            {
                "LITELLM_API_KEY": "native",
                "LLM_API_KEY": "gateway",
                "OPENAI_API_KEY": "legacy",
            },
            clear=True,
        ):
            self.assertEqual(generator.resolve_api_key(), "native")
        with patch.dict(os.environ, {"LLM_API_KEY": "gateway"}, clear=True):
            self.assertEqual(generator.resolve_api_key(), "gateway")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "legacy"}, clear=True):
            self.assertEqual(generator.resolve_api_key(), "legacy")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(generator.resolve_api_key())

    def test_isolated_generation_directory_gets_an_empty_registry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = generator.ensure_dataset_registry(Path(temp_dir) / "datasets")
            self.assertEqual(json.loads(registry.read_text()), {})
            registry.write_text('{"existing": {}}\n', encoding="utf-8")
            self.assertEqual(generator.ensure_dataset_registry(registry.parent), registry)
            self.assertEqual(json.loads(registry.read_text()), {"existing": {}})

    def test_generation_exception_consumes_an_attempt_and_retries(self):
        row = {"meta": {"e40_sample_id": "sample", "e40_source_index": 0}}
        args = SimpleNamespace(
            output_protocol="xml_tags",
            prompt_profile="e95_trigger_locked_autocluster",
            max_attempts=2,
            base_url="https://example.invalid/v1",
            model="generator",
            verifier_model="verifier",
            gen_max_tokens=10,
            verify_max_tokens=10,
            timeout=1,
            reasoning_effort=None,
            verifier_reasoning_effort=None,
            repair_profile="strict_full",
        )
        responses = [
            json.JSONDecodeError("empty", "", 0),
            {"content": "generated"},
            {"content": "verified"},
        ]
        with (
            patch.object(generator, "generator_prompt", return_value="prompt"),
            patch.object(
                generator,
                "hard_verify",
                return_value=("thinking", {"events": []}, []),
            ),
            patch.object(generator, "verifier_prompt", return_value="verify") as prompt,
            patch.object(generator, "semantic_pass", return_value=(True, [])),
            patch.object(generator.e40, "extract_json_obj", return_value={}),
            patch.object(generator, "call_model", side_effect=responses) as call,
        ):
            result = generator.process_one(row, args, "secret")
        self.assertTrue(result["accepted"])
        self.assertEqual(call.call_count, 3)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertIn("JSONDecodeError", result["attempts"][0]["error"])
        self.assertEqual(result["attempts"][0]["error_stage"], "generator_call")
        self.assertNotIn("error", result)

    def test_verifier_parse_exception_preserves_raw_response(self):
        row = {"meta": {"e40_sample_id": "sample", "e40_source_index": 0}}
        args = SimpleNamespace(
            output_protocol="xml_tags",
            prompt_profile="e95_trigger_locked_autocluster",
            max_attempts=1,
            base_url="https://example.invalid/v1",
            model="generator",
            verifier_model="verifier",
            gen_max_tokens=10,
            verify_max_tokens=10,
            timeout=1,
            reasoning_effort=None,
            verifier_reasoning_effort="max",
            repair_profile="strict_full",
        )
        responses = [
            {"content": "generated"},
            {"content": "", "finish_reason": "length", "usage": {"completion_tokens": 10}},
        ]
        with (
            patch.object(generator, "generator_prompt", return_value="prompt"),
            patch.object(generator, "hard_verify", return_value=("thinking", {"events": []}, [])),
            patch.object(generator, "verifier_prompt", return_value="verify") as prompt,
            patch.object(generator, "call_model", side_effect=responses),
        ):
            result = generator.process_one(row, args, "secret")
        attempt = result["attempts"][0]
        self.assertFalse(result["accepted"])
        self.assertEqual(attempt["error_stage"], "verifier_parse")
        self.assertEqual(attempt["verifier"]["finish_reason"], "length")
        self.assertEqual(attempt["verifier"]["usage"]["completion_tokens"], 10)

    def test_e122_gate_requires_acceptance_and_verifier_headroom(self):
        verifier = {
            "content": '{"pass": true}',
            "finish_reason": "stop",
            "usage": {"completion_tokens": 1200},
        }
        rows = [
            {
                "sample_id": f"sample_{index}",
                "accepted": True,
                "hard_ok": True,
                "semantic_ok": True,
                "attempts": [{"attempt": 1, "verifier": verifier}],
            }
            for index in range(20)
        ]
        summary = {"sampled": 20, "accepted": 20}
        result = evaluate_smoke(
            rows,
            summary,
            expected_rows=20,
            min_accepted=19,
            max_attempts=3,
            verify_max_tokens=4096,
            min_headroom_tokens=96,
        )
        self.assertTrue(result["passed"])
        rows[0]["attempts"][0] = {
            "attempt": 1,
            "verifier": {"content": "", "finish_reason": "length", "usage": {"completion_tokens": 4096}},
            "error_stage": "verifier_parse",
        }
        self.assertFalse(
            evaluate_smoke(
                rows,
                summary,
                expected_rows=20,
                min_accepted=19,
                max_attempts=3,
                verify_max_tokens=4096,
                min_headroom_tokens=96,
            )["passed"]
        )

    def test_e124_manifest_selection_is_deterministic_and_disjoint(self):
        rows = [{"meta": {"wnd_id": f"wnd-{index:03d}"}} for index in range(20)]
        excluded = {"wnd-002", "wnd-007", "wnd-011"}
        first = select_disjoint_rows(rows, excluded, count=8, seed=1240)
        second = select_disjoint_rows(rows, excluded, count=8, seed=1240)
        first_ids = [row["meta"]["wnd_id"] for row in first]
        self.assertEqual(first_ids, [row["meta"]["wnd_id"] for row in second])
        self.assertEqual(len(first_ids), len(set(first_ids)))
        self.assertFalse(set(first_ids) & excluded)

    def test_generation_metadata_records_active_models(self):
        old_generator = generator.ACTIVE_GENERATOR_MODEL
        old_verifier = generator.ACTIVE_VERIFIER_MODEL
        try:
            generator.ACTIVE_GENERATOR_MODEL = "glm-5.1"
            generator.ACTIVE_VERIFIER_MODEL = "glm-5.1"
            with patch.object(
                generator,
                "BASE_MAKE_EVIDENCE_ROW",
                return_value={"meta": {}},
            ):
                row = generator.make_evidence_row(
                    {}, "thinking", {"events": []}, "train", "e124b_glm51_selfverifier"
                )
            self.assertEqual(row["meta"]["e40_generator_model"], "glm-5.1")
            self.assertEqual(row["meta"]["e47_verifier_model"], "glm-5.1")
        finally:
            generator.ACTIVE_GENERATOR_MODEL = old_generator
            generator.ACTIVE_VERIFIER_MODEL = old_verifier

    def test_e124b_full_gate_checks_rates_and_p99_headroom(self):
        verifier = {
            "content": '{"pass": true}',
            "finish_reason": "stop",
            "usage": {"completion_tokens": 2000},
        }
        rows = [
            {
                "sample_id": f"sample_{index}",
                "accepted": True,
                "hard_ok": True,
                "semantic_ok": True,
                "attempts": [{"attempt": 1, "verifier": verifier}],
            }
            for index in range(100)
        ]
        result = evaluate_full(
            rows,
            {"sampled": 100, "accepted": 100},
            expected_rows=100,
            min_accepted=94,
            max_attempts=3,
            max_verifier_failure_rate=0.01,
            verify_max_tokens=4096,
            min_p99_headroom_tokens=256,
        )
        self.assertTrue(result["passed"])
        rows[0]["attempts"][0] = {
            "attempt": 1,
            "verifier": {
                "content": "",
                "finish_reason": "length",
                "usage": {"completion_tokens": 4096},
            },
            "error_stage": "verifier_parse",
        }
        rows[1]["attempts"][0] = rows[0]["attempts"][0]
        self.assertFalse(
            evaluate_full(
                rows,
                {"sampled": 100, "accepted": 100},
                expected_rows=100,
                min_accepted=94,
                max_attempts=3,
                max_verifier_failure_rate=0.01,
                verify_max_tokens=4096,
                min_p99_headroom_tokens=256,
            )["passed"]
        )

    def test_e124c_audit_selection_is_stable_and_unique(self):
        rows = [{"meta": {"wnd_id": f"wnd-{index:03d}"}} for index in range(120)]
        first = select_audit_rows(rows, count=100, seed=1242)
        second = select_audit_rows(rows, count=100, seed=1242)
        first_ids = [row["meta"]["wnd_id"] for row in first]
        self.assertEqual(first_ids, [row["meta"]["wnd_id"] for row in second])
        self.assertEqual(len(first_ids), len(set(first_ids)))

    def test_e128_audit_selection_excludes_prior_sample(self):
        rows = [{"meta": {"wnd_id": f"wnd-{index:03d}"}} for index in range(120)]
        excluded = {"wnd-001", "wnd-007", "wnd-099"}
        selected = select_audit_rows(rows, count=100, seed=1280, excluded_wnd_ids=excluded)
        selected_ids = {row["meta"]["wnd_id"] for row in selected}
        self.assertEqual(len(selected_ids), 100)
        self.assertFalse(selected_ids & excluded)

    def test_e124c_valid_semantic_reject_is_not_retried(self):
        row = {"meta": {"wnd_id": "wnd-001"}, "output": "trace"}
        args = SimpleNamespace(
            max_attempts=3,
            base_url="https://example.invalid/v1",
            model="deepseek-v4-pro",
            max_tokens=4096,
            timeout=1,
            reasoning_effort="high",
        )
        with (
            patch.object(
                generator,
                "hard_verify",
                return_value=("thinking", {"events": []}, []),
            ),
            patch.object(generator, "verifier_prompt", return_value="verify") as prompt,
            patch.object(generator, "call_model", return_value={"content": "{}"}) as call,
            patch.object(generator.e40, "extract_json_obj", return_value={"pass": False}),
            patch.object(generator, "semantic_pass", return_value=(False, ["semantic_pass_false"])),
        ):
            result = audit_one(row, args, "secret")
        self.assertFalse(result["semantic_ok"])
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(
            prompt.call_args.kwargs["verifier_profile"], "strict_schema_labels"
        )

    def test_e124c_forwards_optional_alias_aware_verifier_profile(self):
        row = {"meta": {"wnd_id": "wnd-002"}, "output": "trace"}
        args = SimpleNamespace(
            max_attempts=1,
            base_url="https://example.invalid/v1",
            model="deepseek-v4-pro",
            max_tokens=8192,
            timeout=1,
            reasoning_effort="high",
            verifier_profile="target_role_alias_v1",
        )
        with (
            patch.object(
                generator,
                "hard_verify",
                return_value=("thinking", {"events": []}, []),
            ),
            patch.object(generator, "verifier_prompt", return_value="verify") as prompt,
            patch.object(generator, "call_model", return_value={"content": "{}"}),
            patch.object(generator.e40, "extract_json_obj", return_value={"pass": True}),
            patch.object(generator, "semantic_pass", return_value=(True, [])),
        ):
            result = audit_one(row, args, "secret")
        self.assertTrue(result["semantic_ok"])
        self.assertEqual(
            prompt.call_args.kwargs["verifier_profile"], "target_role_alias_v1"
        )

    def test_e127_normalized_surface_hard_profile_allows_short_unique_evidence(self):
        row = {
            "input": "Text: Alice resigned.",
            "gold_output": '{"events": []}',
            "output": '<thinking>Check the exact surface final.</thinking><final>{"events": []}</final>',
        }
        thinking, final_obj, errors = normalized_surface_hard_verify(row)
        self.assertEqual(thinking, "Check the exact surface final.")
        self.assertEqual(final_obj, {"events": []})
        self.assertEqual(errors, [])

    def test_e127_normalized_surface_hard_profile_rejects_missing_thinking(self):
        row = {
            "input": "Text: Alice resigned.",
            "gold_output": '{"events": []}',
            "output": '<final>{"events": []}</final>',
        }
        _, _, errors = normalized_surface_hard_verify(row)
        self.assertIn("missing_thinking", errors)

    def test_e126_reconstructs_and_keeps_existing_hard_valid_trace(self):
        attempt = {
            "attempt": 3,
            "thinking": "Ground the trigger and align the final frame.",
            "final_obj": {"events": []},
        }
        rendered = reconstructed_e126_output(attempt)
        self.assertEqual(
            rendered,
            '<thinking>Ground the trigger and align the final frame.</thinking>'
            '<final>{"events": []}</final>',
        )
        item = {
            "sample_id": "sample-1",
            "source_index": 7,
            "source_row": {"meta": {"wnd_id": "wnd-1"}},
            "selected_attempt": attempt,
            "selected_attempt_number": 3,
            "originally_accepted": False,
        }
        with (
            patch.object(
                generator,
                "hard_verify",
                return_value=(attempt["thinking"], attempt["final_obj"], []),
            ),
            patch.object(
                generator,
                "make_evidence_row",
                return_value={"output": rendered, "meta": {"wnd_id": "wnd-1"}},
            ),
        ):
            rows, manifest, summary = build_e126_dataset(
                [item], "e126_deterministic_hard_valid"
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(manifest[0]["selected_attempt_number"], 3)
        self.assertEqual(summary["counts"]["selected_hard_valid"], 1)
        self.assertEqual(summary["counts"]["originally_rejected"], 1)
        self.assertFalse(summary["passed"])

    def test_e127_requires_exact_raw_final_and_zero_heldout_leak(self):
        def item(sample_id, wnd_id):
            return {
                "sample_id": sample_id,
                "source_index": 1,
                "source_row": {
                    "input": "input",
                    "gold_output": '{"events": []}',
                    "meta": {"wnd_id": wnd_id},
                },
                "selected_attempt": {
                    "attempt": 1,
                    "thinking": "thinking",
                    "final_obj": {"events": []},
                },
                "selected_attempt_number": 1,
                "originally_accepted": True,
            }

        items = [item("keep", "wnd-keep"), item("not-exact", "wnd-exact"), item("leak", "wnd-leak")]
        output_rows = [
            {"output": "keep", "meta": {"wnd_id": "wnd-keep"}},
            {"output": "leak", "meta": {"wnd_id": "wnd-leak"}},
        ]
        with (
            patch.object(
                generator,
                "hard_verify",
                return_value=("thinking", {"events": []}, []),
            ),
            patch(
                "scripts.build_e127_deterministic_exact_trace_dataset_20260712.recover_offsets_from_evidence",
                side_effect=[({}, {"missing_offsets": 0})] * 3,
            ),
            patch(
                "scripts.build_e127_deterministic_exact_trace_dataset_20260712.is_exact",
                side_effect=[True, False, True],
            ),
            patch.object(generator, "make_evidence_row", side_effect=output_rows),
            patch(
                "scripts.build_e127_deterministic_exact_trace_dataset_20260712.find_heldout_leaks",
                side_effect=[[], [{"path": "$.output", "event_type": "Held:Out"}]],
            ),
        ):
            rows, manifest, excluded, summary = build_e127_dataset(
                items, ["Held:Out"], "e127_deterministic_exact", minimum_rows=1
            )
        self.assertEqual([row["sample_id"] for row in manifest], ["keep"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            [row["reason"] for row in excluded],
            ["raw_final_not_exact_gold", "heldout_string_leak"],
        )
        self.assertEqual(summary["counts"]["kept_rows"], 1)
        self.assertFalse(summary["passed"])

    def test_e125_role_alias_contract_is_explicit_and_generic(self):
        row = {"meta": {"e40_sample_id": "e125_sample"}, "input": "input"}
        with (
            patch.object(generator.e40, "extract_schema", return_value=([], [])),
            patch.object(generator.e40, "extract_text", return_value="text"),
            patch.object(generator.e40, "surface_gold_json", return_value={"events": []}),
        ):
            strict = json.loads(generator.verifier_prompt(row, "thinking", {"events": []}))
            alias = json.loads(
                generator.verifier_prompt(
                    row,
                    "thinking",
                    {"events": []},
                    verifier_profile="target_role_alias_v1",
                )
            )
        self.assertNotIn("role_label_contract", strict)
        self.assertIn("role_label_contract", alias)
        contract = " ".join(alias["role_label_contract"])
        self.assertIn("authoritative dataset output labels", contract)
        self.assertIn("does not relax local grounding", contract)
        self.assertNotIn("Contact", contract)

    def test_e128_core_reasoning_contract_excludes_final_evidence_quality(self):
        row = {"meta": {"e40_sample_id": "e128_sample"}, "input": "input"}
        with (
            patch.object(generator.e40, "extract_schema", return_value=([], [])),
            patch.object(generator.e40, "extract_text", return_value="text"),
            patch.object(generator.e40, "surface_gold_json", return_value={"events": []}),
        ):
            prompt = json.loads(
                generator.verifier_prompt(
                    row,
                    "thinking",
                    {"events": []},
                    verifier_profile="target_role_alias_core_reasoning_v1",
                )
            )
        self.assertNotIn("evidence_informativeness", prompt["return_contract"]["scores"])
        requirements = " ".join(prompt["pass_requirements"])
        self.assertIn("Do not judge or score the length", requirements)
        self.assertIn("shortest_unique_evidence_v1", requirements)
        self.assertIn("role_label_contract", prompt)

    def test_e128_core_semantic_profile_ignores_absent_evidence_scores(self):
        scores = {
            "type_discrimination": 5,
            "trigger_boundary_control": 5,
            "argument_role_grounding": 4,
            "extraction_style_control": 5,
            "candidate_coverage": 5,
            "minimal_trigger_separation": 4,
            "role_abstention": 5,
            "no_extra_event_gate": 5,
            "final_structure_consistency": 5,
        }
        passed, errors = generator.semantic_pass(
            {"pass": True, "scores": scores, "errors": []},
            semantic_profile="core_reasoning_v1",
        )
        self.assertTrue(passed)
        self.assertEqual(errors, [])
        full_passed, full_errors = generator.semantic_pass(
            {"pass": True, "scores": scores, "errors": []},
            semantic_profile="full_v1",
        )
        self.assertFalse(full_passed)
        self.assertIn("low_evidence_informativeness:0", full_errors)

    def test_e125_selects_last_existing_hard_valid_attempt(self):
        attempts = [
            {"attempt": 1, "hard_ok": True, "thinking": "first", "final_obj": {}},
            {"attempt": 2, "hard_ok": False, "thinking": None, "final_obj": None},
            {"attempt": 3, "hard_ok": True, "thinking": "last", "final_obj": {}},
        ]
        selected = select_last_hard_valid_attempt({"attempts": attempts})
        self.assertEqual(selected["attempt"], 3)
        self.assertIsNone(select_last_hard_valid_attempt({"attempts": attempts[1:2]}))

    def test_e125_valid_semantic_reject_is_not_retried(self):
        item = {
            "sample_id": "sample-1",
            "source_index": 1,
            "originally_accepted": False,
            "selected_attempt_number": 2,
            "selected_attempt": {"thinking": "thinking", "final_obj": {"events": []}},
            "source_row": {"input": "input", "meta": {"e40_sample_id": "e125_sample"}},
        }
        args = SimpleNamespace(
            verifier_profile="target_role_alias_v1",
            max_attempts=3,
            base_url="https://example.invalid/v1",
            model="glm-5.1",
            max_tokens=6144,
            timeout=1,
            reasoning_effort=None,
        )
        with (
            patch.object(generator, "verifier_prompt", return_value="verify"),
            patch.object(generator, "call_model", return_value={"content": "{}"}) as call,
            patch.object(generator.e40, "extract_json_obj", return_value={"pass": False}),
            patch.object(generator, "semantic_pass", return_value=(False, ["semantic_pass_false"])),
        ):
            result = verify_one_e125(item, args, "secret")
        self.assertFalse(result["semantic_ok"])
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(call.call_count, 1)

    def test_e125_summary_requires_yield_and_token_headroom(self):
        args = SimpleNamespace(
            min_hard_valid=4,
            min_valid_judgments=4,
            min_semantic_pass=4,
            max_attempts=3,
            max_failure_rate=0.01,
            max_tokens=6144,
            min_p99_headroom_tokens=512,
        )
        verifier = {
            "content": '{"pass": true}',
            "finish_reason": "stop",
            "usage": {"completion_tokens": 2000},
        }
        results = [
            {
                "sample_id": f"sample-{index}",
                "hard_ok": True,
                "semantic_ok": True,
                "verifier_obj": {"pass": True},
                "attempts": [{"attempt": 1, "verifier": verifier}],
            }
            for index in range(4)
        ]
        self.assertTrue(
            summarize_e125(results, [row["sample_id"] for row in results], args)["passed"]
        )
        results[-1]["attempts"][0]["verifier"] = {
            "content": "",
            "finish_reason": "length",
            "usage": {"completion_tokens": 6144},
        }
        self.assertFalse(
            summarize_e125(results, [row["sample_id"] for row in results], args)["passed"]
        )

    def test_e124b_rejection_audit_partitions_role_alias_failures(self):
        rows = [
            {"sample_id": "accepted", "accepted": True},
            {
                "sample_id": "alias",
                "accepted": False,
                "semantic_errors": ["schema_role_mismatch: Entity vs Participant"],
            },
            {
                "sample_id": "long",
                "accepted": False,
                "hard_errors": ["thinking_too_long"],
            },
            {
                "sample_id": "hard",
                "accepted": False,
                "hard_errors": ["missing_final"],
            },
            {
                "sample_id": "semantic",
                "accepted": False,
                "semantic_errors": ["weak_trigger"],
            },
        ]
        self.assertEqual(
            classify_rejection(rows[1]), "role_label_schema_name_conflict"
        )
        result = audit_e124b_rejections(rows, "sha")
        self.assertEqual(result["counts"]["rejected"], 4)
        self.assertTrue(result["checks"]["categories_partition_rejections"])

    def test_generation_runner_preserves_e99_verifier_effort(self):
        config = json.loads(
            (
                REPO
                / "configs/generated/stage2_confirmation/"
                "e121c_autocluster_generation.json"
            ).read_text()
        )
        runner = (
            REPO / "scripts/run_e121_strict_confirmation_20260712.sh"
        ).read_text()
        self.assertEqual(config["verifier_reasoning_effort"], "max")
        self.assertIn("--verifier_reasoning_effort max", runner)

    def test_freeze_manifest_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.txt"
            source.write_text("frozen\n", encoding="utf-8")
            manifest = root / "manifest.json"
            args = argparse.Namespace(
                root=root,
                path=["source.txt"],
                glob=[],
                manifest=manifest,
            )
            self.assertEqual(build_manifest(args), 0)
            self.assertEqual(verify_manifest(args), 0)
            source.write_text("changed\n", encoding="utf-8")
            self.assertEqual(verify_manifest(args), 6)

    def test_registered_gate_is_config_driven(self):
        integrity = {"json": 1.0, "offset": 1.0}
        seen = {
            "macro_mean_delta": {"argument": -0.01, "event": -0.01, "trigger": 0.0},
            "baseline_integrity_mean": integrity,
            "candidate_integrity_mean": integrity,
        }
        pooled = {
            "macro_mean_delta": {"argument": 0.03, "event": 0.02, "trigger": 0.04},
            "micro_mean_delta": {"argument": 0.03, "event": 0.02, "trigger": 0.04},
            "macro_micro_directions_match": True,
            "paired_bootstrap": {
                "argument": {"point": 0.03, "lower_95": 0.01, "upper_95": 0.05},
                "event": {"point": 0.02, "lower_95": -0.01, "upper_95": 0.04},
                "trigger": {"point": 0.04, "lower_95": 0.01, "upper_95": 0.06},
            },
            "seed_runs": [
                {"macro_delta": {"argument": 0.03, "event": 0.02, "trigger": 0.04}}
                for _ in range(3)
            ],
            "baseline_integrity_mean": integrity,
            "candidate_integrity_mean": integrity,
        }
        per_type = {
            event_type: {"argument": 0.01, "event": 0.01, "trigger": 0.01, "mean": 0.01}
            for event_type in EXPECTED
        }
        config = json.loads(
            (REPO / "configs/generated/stage2_confirmation/e121e_confirmation_gate.json").read_text()
        )
        result = apply_gate(seen, pooled, pooled, per_type, set(EXPECTED), config)
        self.assertTrue(result["passed"])

        degraded_split1 = json.loads(json.dumps(pooled))
        degraded_split1["candidate_integrity_mean"]["offset"] = 0.98
        result = apply_gate(
            seen, degraded_split1, pooled, per_type, set(EXPECTED), config
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["split1_unseen_offset_registered_drop_limit"])


if __name__ == "__main__":
    unittest.main()
