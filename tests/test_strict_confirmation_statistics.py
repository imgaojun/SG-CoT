import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.compare_e121_confirmation_n3_20260712 import (
    completion_diagnostics,
    main as confirmation_main,
    protocol_integrity_checks,
)
from scripts.compare_preference_run_gate_20260712 import verify_pairing


def row(wnd_id="wnd-1", input_text="input", event_type="Held:Type"):
    return {
        "input": input_text,
        "gold": {"events": [{"event_type": event_type}]},
        "predicted": {"events": []},
        "meta": {"wnd_id": wnd_id} if wnd_id else {},
    }


class StrictPairingTests(unittest.TestCase):
    @staticmethod
    def _write_run(
        directory: Path,
        *,
        rows: list[dict],
        exact_predictions: bool,
        expects_reasoning: bool,
    ) -> None:
        directory.mkdir(parents=True)
        output_rows = []
        for source in rows:
            value = json.loads(json.dumps(source))
            value["predicted"] = value["gold"] if exact_predictions else {"events": []}
            value.update(
                {
                    "generated_token_count": 12,
                    "generation_ended_with_eos": True,
                    "hit_max_new_tokens": False,
                    "final_tag_complete": True,
                    "reasoning_tag_complete": expects_reasoning,
                    "surface_event_list_valid": True,
                    "candidate_types_valid": True,
                }
            )
            output_rows.append(value)
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for value in output_rows:
                handle.write(json.dumps(value) + "\n")
        score = 1.0 if exact_predictions else 0.0
        summary = {
            "argument_f1": score,
            "event_f1": score,
            "trigger_f1": score,
            "final_json_valid_rate": 1.0,
            "offset_recovery_full_rate": 1.0,
            "max_new_tokens": 32,
            "generated_token_count_mean": 12.0,
            "generated_token_count_p95": 12,
            "generated_token_count_max": 12,
            "hit_max_new_tokens_count": 0,
            "hit_max_new_tokens_rate": 0.0,
            "final_tag_complete_rate": 1.0,
            "reasoning_tag_complete_rate": 1.0 if expects_reasoning else None,
            "surface_event_list_valid_rate": 1.0,
            "candidate_type_valid_rate": 1.0,
        }
        (directory / "summary.json").write_text(
            json.dumps(summary) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _synthetic_rows(split: str) -> list[dict]:
        rows = []
        for index, event_type in enumerate(("Held:One", "Held:Two")):
            rows.append(
                {
                    "input": f"{split}-input-{index}",
                    "gold": {
                        "events": [
                            {
                                "event_type": event_type,
                                "trigger": {"start": index, "end": index + 1},
                                "arguments": [
                                    {
                                        "role": "Entity",
                                        "start": index + 1,
                                        "end": index + 2,
                                    }
                                ],
                            }
                        ]
                    },
                    "meta": {
                        "wnd_id": f"{split}-wnd-{index}",
                        "source_protocol": "synthetic-strict-v1",
                    },
                }
            )
        return rows

    def test_confirmation_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = {}
            for split in ("seen", "unseen", "pooled"):
                rows = self._synthetic_rows(split)
                for method, exact, reasoning in (
                    ("baseline", False, False),
                    ("candidate", True, True),
                ):
                    directories[(method, split)] = []
                    for seed in (42, 8322, 8333):
                        directory = root / f"{method}-{split}-{seed}"
                        self._write_run(
                            directory,
                            rows=rows,
                            exact_predictions=exact,
                            expects_reasoning=reasoning,
                        )
                        directories[(method, split)].append(directory)

            heldout = root / "heldout.json"
            heldout.write_text('["Held:One", "Held:Two"]\n', encoding="utf-8")
            gate = root / "gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "bootstrap_samples": 100,
                        "bootstrap_seed": 128,
                        "pooled_macro_mean_delta_min": 0.015,
                        "required_positive_seed_metric_cells": 9,
                        "required_nonnegative_types": 2,
                        "minimum_per_type_mean_delta": -0.02,
                        "minimum_seen_mean_delta": -0.025,
                        "minimum_seen_metric_delta": -0.05,
                        "maximum_integrity_drop": 0.01,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            eval_config = root / "eval.json"
            eval_config.write_text(
                json.dumps(
                    {
                        "id": "synthetic_confirmation",
                        "protocol": "synthetic-strict-v1",
                        "expected_examples": {
                            "test_seen": 2,
                            "test_unseen": 2,
                            "pooled_unseen": 2,
                        },
                        "expected_pooled_unique_wnd_ids": 2,
                        "expected_pooled_gold_events": 2,
                        "expected_heldout_types": 2,
                        "decode": {"max_new_tokens": 32},
                        "completion_diagnostics": [
                            "generated_token_count",
                            "generation_ended_with_eos",
                            "hit_max_new_tokens",
                        ],
                        "output_contract_diagnostics": [
                            "final_tag_complete",
                            "reasoning_tag_complete",
                            "surface_event_list_valid",
                            "candidate_types_valid",
                        ],
                        "bootstrap": {"samples": 100, "seed": 128},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "comparison"

            argv = ["compare_e121_confirmation_n3_20260712.py"]
            for option, key in (
                ("--baseline_seen", ("baseline", "seen")),
                ("--baseline_unseen", ("baseline", "unseen")),
                ("--baseline_pooled", ("baseline", "pooled")),
                ("--candidate_seen", ("candidate", "seen")),
                ("--candidate_unseen", ("candidate", "unseen")),
                ("--candidate_pooled", ("candidate", "pooled")),
            ):
                argv.extend([option, *(str(path) for path in directories[key])])
            argv.extend(
                [
                    "--heldout_types_json",
                    str(heldout),
                    "--gate_config",
                    str(gate),
                    "--eval_config",
                    str(eval_config),
                    "--output_dir",
                    str(output),
                    "--bootstrap_samples",
                    "100",
                    "--bootstrap_seed",
                    "128",
                ]
            )
            with patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(confirmation_main(), 0)
            report = json.loads((output / "comparison.json").read_text())
            self.assertTrue(report["gate"]["passed"])
            self.assertTrue(all(report["protocol_integrity_checks"].values()))
            self.assertTrue((output / "comparison.md").is_file())

    def test_completion_diagnostics_aggregate_cap_hits(self):
        pairs = []
        for index in range(3):
            summary = {
                "max_new_tokens": 1024,
                "generated_token_count_mean": 100 + index,
                "generated_token_count_p95": 200 + index,
                "generated_token_count_max": 300 + index,
                "hit_max_new_tokens_count": index,
                "hit_max_new_tokens_rate": index / 10,
            }
            pairs.append(
                {"baseline_summary": dict(summary), "candidate_summary": dict(summary)}
            )
        result = completion_diagnostics(pairs)
        self.assertTrue(result["candidate"]["available"])
        self.assertEqual(result["candidate"]["max_new_tokens"], [1024])
        self.assertEqual(
            result["candidate"]["hit_max_new_tokens_count_across_seeds"], 3
        )
        self.assertEqual(
            result["candidate"]["generated_token_count_max_across_seeds"], 302
        )

    def test_strict_pairing_requires_wnd_id(self):
        with self.assertRaisesRegex(ValueError, "missing wnd_id"):
            verify_pairing(
                [row(wnd_id=None)],
                [row(wnd_id=None)],
                require_wnd_id=True,
                require_input_match=True,
            )

    def test_strict_pairing_requires_identical_input(self):
        with self.assertRaisesRegex(ValueError, "input mismatch"):
            verify_pairing(
                [row(input_text="left")],
                [row(input_text="right")],
                require_wnd_id=True,
                require_input_match=True,
            )

    def test_protocol_cardinality_checks(self):
        def pair(rows):
            return {"baseline_rows": rows, "candidate_rows": list(rows)}

        split_pairs = {
            "test_seen": [pair([row("seen")])] * 3,
            "test_unseen": [pair([row("unseen")])] * 3,
            "pooled_unseen": [pair([row("pool-1"), row("pool-2")])] * 3,
        }
        config = {
            "protocol": "strict-v1",
            "expected_examples": {
                "test_seen": 1,
                "test_unseen": 1,
                "pooled_unseen": 2,
            },
            "expected_pooled_unique_wnd_ids": 2,
            "expected_pooled_gold_events": 2,
            "expected_heldout_types": 1,
            "completion_diagnostics": [
                "generated_token_count",
                "generation_ended_with_eos",
                "hit_max_new_tokens",
            ],
            "output_contract_diagnostics": [
                "final_tag_complete",
                "reasoning_tag_complete",
                "surface_event_list_valid",
                "candidate_types_valid",
            ],
            "decode": {"max_new_tokens": 1024},
        }
        for pairs in split_pairs.values():
            for item in pairs:
                for side in ("baseline_rows", "candidate_rows"):
                    for value in item[side]:
                        value["meta"]["source_protocol"] = "strict-v1"
                        value.update(
                            {
                                "generated_token_count": 10,
                                "generation_ended_with_eos": True,
                                "hit_max_new_tokens": False,
                                "final_tag_complete": True,
                                "reasoning_tag_complete": True,
                                "surface_event_list_valid": True,
                                "candidate_types_valid": True,
                            }
                        )
                item["baseline_summary"] = {"max_new_tokens": 1024}
                item["candidate_summary"] = {"max_new_tokens": 1024}
        checks = protocol_integrity_checks(split_pairs, {"Held:Type"}, config)
        self.assertTrue(all(checks.values()))

        split_pairs["pooled_unseen"][0]["baseline_rows"][1]["meta"]["wnd_id"] = "pool-1"
        checks = protocol_integrity_checks(split_pairs, {"Held:Type"}, config)
        self.assertFalse(checks["pooled_unique_wnd_ids_exact"])


if __name__ == "__main__":
    unittest.main()
