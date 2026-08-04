#!/usr/bin/env python3
"""Build and gate gold-safe, on-policy E135 SC-distillation targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.rescore_self_consistency_samples_20260712 import (  # noqa: E402
    as_payload,
    normalize,
    recover_offsets_from_evidence,
    vote,
)
from scripts.generate_e135_sc_paths_20260713 import experiment_prefix  # noqa: E402


TAGGED_RESPONSE = re.compile(
    r"\A\s*<thinking>(?P<thinking>.*?)</thinking>\s*"
    r"<final>(?P<final>\{.*\})</final>\s*\Z",
    re.DOTALL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def strict_parse_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(text, str):
        return None, "not_string"
    if any(
        text.count(tag) != 1
        for tag in ("<thinking>", "</thinking>", "<final>", "</final>")
    ):
        return None, "tag_count"
    match = TAGGED_RESPONSE.fullmatch(text)
    if match is None:
        return None, "tag_structure"
    if not match.group("thinking").strip():
        return None, "empty_thinking"
    try:
        payload = json.loads(match.group("final"))
    except json.JSONDecodeError:
        return None, "final_json"
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return None, "final_schema"
    return payload, None


def canonical_structure(payload: dict[str, Any]) -> tuple[Any, ...]:
    events = []
    for event_key, arguments in normalize(payload):
        events.append((event_key, tuple(sorted(arguments, key=repr))))
    return tuple(sorted(events, key=repr))


def choose_carrier(
    voted: dict[str, Any],
    recovered_samples: list[dict[str, Any]],
    strict_valid: list[bool],
) -> list[int]:
    target = canonical_structure(voted)
    return [
        index
        for index, (sample, valid) in enumerate(zip(recovered_samples, strict_valid))
        if valid and canonical_structure(sample) == target
    ]


def recover_response(text: str, input_text: str) -> dict[str, Any]:
    surface, error = strict_parse_response(text)
    result: dict[str, Any] = {
        "surface": surface,
        "parse_error": error,
        "strict_valid": False,
        "recovered": {"events": []},
        "missing_offsets": 0,
    }
    if surface is None:
        return result
    try:
        recovered, diagnostics = recover_offsets_from_evidence(surface, input_text)
    except Exception as exc:  # recorded as data, never silently accepted
        result["parse_error"] = f"recovery_error:{type(exc).__name__}"
        return result
    missing = int(diagnostics.get("missing_offsets", 0))
    result.update(
        {
            "recovered": recovered,
            "missing_offsets": missing,
            "strict_valid": missing == 0,
            "parse_error": None if missing == 0 else "missing_offsets",
        }
    )
    return result


def evaluate_gate(counts: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    frozen = protocol["gate"]
    mode = frozen.get("mode", "all_rows_minimum_v1")
    checks = {
        "generation_rows_exact": counts["generation_rows"]
        == int(frozen["required_generation_rows"]),
        "sample_count_exact": counts["sample_count_errors"] == 0,
    }
    if mode == "all_rows_minimum_v1":
        checks["minimum_valid_samples_per_row"] = (
            counts["rows_below_min_valid_samples"] == 0
        )
    elif mode in {
        "selection_safety_aggregate_v1",
        "training_boundary_aggregate_v1",
    }:
        checks.update(
            {
                "aggregate_strict_sample_rate": counts["strict_sample_valid_rate"]
                >= float(frozen["minimum_strict_sample_valid_rate"]),
                "rows_below_min_valid_samples_bounded": counts[
                    "rows_below_min_valid_samples"
                ]
                <= int(frozen["maximum_rows_below_min_valid_samples"]),
            }
        )
        if mode == "selection_safety_aggregate_v1":
            checks["greedy_parse_reliability"] = counts[
                "strict_greedy_parse_errors"
            ] <= int(frozen["maximum_strict_greedy_parse_errors"])
    else:
        raise ValueError(f"unsupported SC-distillation gate mode: {mode}")
    checks.update(
        {
            "vote_carrier_yield": counts["vote_carrier_rows"]
            >= int(frozen["minimum_vote_carrier_rows"]),
            "vote_gold_exact_yield": counts["vote_gold_exact_rows"]
            >= int(frozen["minimum_vote_gold_exact_rows"]),
            "eligible_yield": counts["eligible_distillation_rows"]
            >= int(frozen["minimum_eligible_distillation_rows"]),
            "selected_targets_parse": counts["selected_target_parse_errors"]
            <= int(frozen["maximum_selected_target_parse_errors"]),
            "duplicate_wnd_ids": counts["duplicate_wnd_ids"]
            <= int(frozen["maximum_duplicate_wnd_ids"]),
        }
    )
    if mode == "training_boundary_aggregate_v1":
        checks["strict_greedy_correction_yield"] = counts[
            "vote_corrects_strict_greedy_rows"
        ] >= int(frozen["minimum_vote_corrects_strict_greedy_rows"])
    else:
        checks["sc_correction_yield"] = counts[
            "vote_corrects_greedy_rows"
        ] >= int(frozen["minimum_vote_corrects_greedy_rows"])
    if mode in {
        "selection_safety_aggregate_v1",
        "training_boundary_aggregate_v1",
    }:
        checks["selected_targets_recovery"] = counts[
            "selected_target_recovery_errors"
        ] <= int(frozen["maximum_selected_target_recovery_errors"])
    return {
        "id": protocol.get("report_ids", {}).get(
            "target_gate", "e135_sc_distillation_smoke64_gate_v1"
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "thresholds": frozen,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest_jsonl", type=Path, required=True)
    parser.add_argument("--generations_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--require_pass", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    prefix = experiment_prefix(protocol)
    manifest = load_jsonl(args.manifest_jsonl)
    generated = load_jsonl(args.generations_jsonl)
    required_rows = int(protocol["gate"]["required_generation_rows"])
    required_samples = int(protocol["gate"]["required_samples_per_row"])
    threshold = int(protocol["vote"]["event_and_argument_threshold"])
    minimum_valid = int(protocol["gate"]["minimum_strictly_parsed_samples_per_row"])
    aggregate_mode = protocol["gate"].get("mode") in {
        "selection_safety_aggregate_v1",
        "training_boundary_aggregate_v1",
    }
    if len(manifest) != required_rows:
        raise ValueError("manifest row count mismatch")

    counts = {
        "manifest_rows": len(manifest),
        "generation_rows": len(generated),
        "sample_count_errors": 0,
        "rows_below_min_valid_samples": 0,
        "strict_sample_parse_errors": 0,
        "strict_greedy_parse_errors": 0,
        "vote_carrier_rows": 0,
        "vote_gold_exact_rows": 0,
        "eligible_distillation_rows": 0,
        "vote_corrects_greedy_rows": 0,
        "vote_corrects_strict_greedy_rows": 0,
        "selected_target_parse_errors": 0,
        "duplicate_wnd_ids": 0,
        "test_rows_read": 0,
    }
    if aggregate_mode:
        counts.update(
            {
                "strict_sample_total": 0,
                "strict_sample_valid": 0,
                "selected_target_recovery_errors": 0,
            }
        )
    seen_wnd_ids = set()
    targets = []
    row_audits = []
    for rank, source in enumerate(manifest):
        wnd_id = source["meta"]["wnd_id"]
        if wnd_id in seen_wnd_ids:
            counts["duplicate_wnd_ids"] += 1
        seen_wnd_ids.add(wnd_id)
        if rank >= len(generated):
            continue
        raw = generated[rank]
        if (
            int(raw.get("selection_rank", -1)) != rank
            or raw.get("wnd_id") != wnd_id
            or int(raw.get("source_index", -1))
            != int(source["meta"][f"{prefix}_source_index"])
        ):
            raise ValueError(f"generation pairing mismatch at rank {rank}")
        sampled_texts = raw.get("sampled_texts")
        if not isinstance(sampled_texts, list) or len(sampled_texts) != required_samples:
            counts["sample_count_errors"] += 1
            row_audits.append({"selection_rank": rank, "wnd_id": wnd_id, "paired": True})
            continue

        greedy = recover_response(raw.get("greedy_text"), source["input"])
        samples = [recover_response(text, source["input"]) for text in sampled_texts]
        valid_samples = sum(int(sample["strict_valid"]) for sample in samples)
        if aggregate_mode:
            counts["strict_sample_total"] += required_samples
            counts["strict_sample_valid"] += valid_samples
        counts["strict_sample_parse_errors"] += required_samples - valid_samples
        counts["strict_greedy_parse_errors"] += int(not greedy["strict_valid"])
        counts["rows_below_min_valid_samples"] += int(valid_samples < minimum_valid)
        recovered_samples = [sample["recovered"] for sample in samples]
        voted = as_payload(vote([normalize(sample) for sample in recovered_samples], threshold))
        carriers = choose_carrier(
            voted, recovered_samples, [bool(sample["strict_valid"]) for sample in samples]
        )
        has_carrier = bool(carriers)
        counts["vote_carrier_rows"] += int(has_carrier)

        gold_raw = source.get("gold_output", source["output"])
        gold = json.loads(gold_raw) if isinstance(gold_raw, str) else gold_raw
        vote_gold_exact = canonical_structure(voted) == canonical_structure(gold)
        greedy_gold_exact = (
            bool(greedy["strict_valid"])
            and canonical_structure(greedy["recovered"]) == canonical_structure(gold)
        )
        counts["vote_gold_exact_rows"] += int(vote_gold_exact)
        eligible = has_carrier and vote_gold_exact
        corrects_greedy = eligible and not greedy_gold_exact
        corrects_strict_greedy = (
            eligible and bool(greedy["strict_valid"]) and not greedy_gold_exact
        )
        counts["eligible_distillation_rows"] += int(eligible)
        counts["vote_corrects_greedy_rows"] += int(corrects_greedy)
        counts["vote_corrects_strict_greedy_rows"] += int(corrects_strict_greedy)

        chosen_index = carriers[0] if eligible else None
        if chosen_index is not None:
            chosen_text = sampled_texts[chosen_index]
            chosen_surface, chosen_error = strict_parse_response(chosen_text)
            selected_error = chosen_error is not None or chosen_surface is None
            counts["selected_target_parse_errors"] += int(selected_error)
            if aggregate_mode:
                counts["selected_target_recovery_errors"] += int(
                    not samples[chosen_index]["strict_valid"]
                )
            target = json.loads(json.dumps(source, ensure_ascii=False))
            target["output"] = chosen_text
            target["meta"][f"{prefix}_distillation"] = {
                "selection_rank": rank,
                "sample_index": chosen_index,
                "vote_k": threshold,
                "vote_gold_exact": True,
                "vote_corrects_greedy": bool(corrects_greedy),
                "target_kind": "on_policy_sample_carrying_gold_exact_sc_vote",
            }
            targets.append(target)
        row_audits.append(
            {
                "selection_rank": rank,
                "wnd_id": wnd_id,
                "valid_samples": valid_samples,
                "greedy_valid": bool(greedy["strict_valid"]),
                "vote_carrier_indices": carriers,
                "vote_gold_exact": vote_gold_exact,
                "greedy_gold_exact": greedy_gold_exact,
                "eligible": eligible,
                "vote_corrects_greedy": corrects_greedy,
                "vote_corrects_strict_greedy": corrects_strict_greedy,
                "chosen_sample_index": chosen_index,
            }
        )

    if aggregate_mode:
        counts["strict_sample_valid_rate"] = (
            counts["strict_sample_valid"] / counts["strict_sample_total"]
            if counts["strict_sample_total"]
            else 0.0
        )
        counts["strict_greedy_valid_rate"] = (
            (counts["generation_rows"] - counts["strict_greedy_parse_errors"])
            / counts["generation_rows"]
            if counts["generation_rows"]
            else 0.0
        )
    gate = evaluate_gate(counts, protocol)
    args.output_dir.mkdir(parents=True)
    target_path = args.output_dir / "distillation_targets.jsonl"
    with target_path.open("w", encoding="utf-8") as handle:
        for target in targets:
            handle.write(json.dumps(target, ensure_ascii=False) + "\n")
    with (args.output_dir / "row_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in row_audits:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    gate["input_sha256"] = {
        "protocol": sha256_file(args.protocol),
        "manifest": sha256_file(args.manifest_jsonl),
        "generations": sha256_file(args.generations_jsonl),
    }
    gate["output_sha256"] = {
        "targets": sha256_file(target_path),
    }
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if args.require_pass and not gate["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
