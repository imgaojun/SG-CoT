#!/usr/bin/env python3
"""Audit and freeze E114 atomic-counterfactual preference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.mine_reasoning_preferences_e110_20260711 import (  # noqa: E402
    conversation_token_count,
    load_jsonl,
    load_sample_records,
    parse_gold,
    row_id,
    token_count,
)
from src.stage2_preference.atomic_counterfactual import (  # noqa: E402
    ATOMIC_CATEGORIES,
    apply_atomic_proposal,
    label_leaks,
    render_canonical_pair,
)
from src.stage2_preference.reasoning_preference import (  # noqa: E402
    classify_single_error,
    event_types_within_candidates,
    extract_final_json,
    has_complete_reasoning_response,
    is_exact,
    offsets_complete,
    parse_prompt_tokens,
    recover_offsets_from_evidence,
    valid_length_pair,
)


E81_QUOTAS = {category: 180 for category in ATOMIC_CATEGORIES}
G9_QUOTAS = {
    "wrong_type": 135,
    "trigger_drift": 90,
    "argument_omission": 225,
    "event_omission": 270,
    "extra_frame": 180,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_quotas(value: str | None, profile: str) -> dict[str, int]:
    if value is None:
        return dict(E81_QUOTAS if profile == "e81" else G9_QUOTAS)
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(value)
    quotas = {str(category): int(count) for category, count in payload.items()}
    if set(quotas) != set(ATOMIC_CATEGORIES):
        raise ValueError(f"quota categories do not match: {quotas}")
    return quotas


def add_problem(
    problems: list[dict[str, Any]], dataset: str, index: int, wnd_id: str, check: str
) -> None:
    problems.append(
        {"dataset": dataset, "row_index": index, "wnd_id": wnd_id, "check": check}
    )


def audit_dataset(
    *,
    name: str,
    pairs: list[dict[str, Any]],
    source_by_wnd: dict[str, dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    deterministic_only: bool,
    samples_by_wnd: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []
    windows: set[str] = set()
    categories: Counter[str] = Counter()
    proposal_sources: Counter[str] = Counter()
    chosen_exact = 0
    rejected_atomic = 0
    offsets_recovered = 0
    candidate_legal = 0
    canonical_replays = 0
    tag_complete = 0
    self_exact_qualifications = 0
    maximum_total = 0
    ratios: list[float] = []

    for index, pair in enumerate(pairs):
        meta = pair.get("meta") if isinstance(pair.get("meta"), dict) else {}
        wnd_id = str(meta.get("wnd_id") or "")
        row = source_by_wnd.get(wnd_id)
        if row is None:
            add_problem(problems, name, index, wnd_id, "unknown_or_missing_wnd_id")
            continue
        if wnd_id in windows:
            add_problem(problems, name, index, wnd_id, "duplicate_wnd_id")
        windows.add(wnd_id)
        if pair.get("instruction") != row.get("instruction") or pair.get("input") != row.get("input"):
            add_problem(problems, name, index, wnd_id, "prompt_mismatch")

        category = str(meta.get("error_category") or "")
        categories[category] += 1
        proposal_source = str(meta.get("proposal_source") or "")
        proposal_sources[proposal_source] += 1
        if category not in ATOMIC_CATEGORIES:
            add_problem(problems, name, index, wnd_id, "unknown_error_category")
        if deterministic_only and proposal_source != "deterministic_fallback":
            add_problem(problems, name, index, wnd_id, "non_deterministic_control_proposal")
        if not deterministic_only and proposal_source not in {
            "observed_atomic",
            "deterministic_fallback",
        }:
            add_problem(problems, name, index, wnd_id, "invalid_main_proposal_source")
        if meta.get("chosen_source") != "canonical_gold_from_self_exact_window":
            add_problem(problems, name, index, wnd_id, "invalid_chosen_source")
        if meta.get("external_teacher") is not False:
            add_problem(problems, name, index, wnd_id, "external_teacher_not_false")
        if meta.get("renderer_version") != args.renderer_version:
            add_problem(problems, name, index, wnd_id, "renderer_version_mismatch")
        eligibility = meta.get("eligibility_exact_sample")
        if not isinstance(eligibility, dict) or int(eligibility.get("total_tokens", args.cutoff_len + 1)) > args.cutoff_len:
            add_problem(problems, name, index, wnd_id, "invalid_eligibility_exact_sample")
        elif samples_by_wnd is not None:
            matching_samples = [
                sample
                for sample in samples_by_wnd.get(wnd_id, [])
                if sample.get("sample_seed") == eligibility.get("sample_seed")
                and sample.get("sample_round") == eligibility.get("sample_round")
                and sample.get("sample_index") == eligibility.get("sample_index")
            ]
            qualification_ok = False
            gold_for_qualification = parse_gold(row)
            for sample in matching_samples:
                raw_response = sample.get("raw_response")
                payload = extract_final_json(raw_response) if isinstance(raw_response, str) else None
                if payload is None or not has_complete_reasoning_response(raw_response):
                    continue
                recovered, diagnostics = recover_offsets_from_evidence(payload, row["input"])
                exact_total = conversation_token_count(
                    tokenizer, row["instruction"], row["input"], raw_response
                )
                if (
                    not diagnostics.get("missing_offsets")
                    and offsets_complete(recovered)
                    and event_types_within_candidates(
                        recovered, row.get("meta", {}).get("candidate_types", [])
                    )
                    and is_exact(recovered, gold_for_qualification)
                    and exact_total <= args.cutoff_len
                    and exact_total == int(eligibility.get("total_tokens", -1))
                ):
                    qualification_ok = True
                    break
            if qualification_ok:
                self_exact_qualifications += 1
            else:
                add_problem(problems, name, index, wnd_id, "self_exact_qualification_replay_failed")

        chosen = pair.get("chosen")
        rejected = pair.get("rejected")
        if not isinstance(chosen, str) or not isinstance(rejected, str):
            add_problem(problems, name, index, wnd_id, "response_not_string")
            continue
        if not has_complete_reasoning_response(chosen) or not has_complete_reasoning_response(rejected):
            add_problem(problems, name, index, wnd_id, "incomplete_lowercase_tags")
            continue
        tag_complete += 1
        if label_leaks(chosen, row["input"]) or label_leaks(rejected, row["input"]):
            add_problem(problems, name, index, wnd_id, "label_leak")

        chosen_payload = extract_final_json(chosen)
        rejected_payload = extract_final_json(rejected)
        if chosen_payload is None or rejected_payload is None:
            add_problem(problems, name, index, wnd_id, "invalid_final_json")
            continue
        chosen_recovered, chosen_diag = recover_offsets_from_evidence(chosen_payload, row["input"])
        rejected_recovered, rejected_diag = recover_offsets_from_evidence(
            rejected_payload, row["input"]
        )
        complete_offsets = (
            not chosen_diag.get("missing_offsets")
            and not rejected_diag.get("missing_offsets")
            and offsets_complete(chosen_recovered)
            and offsets_complete(rejected_recovered)
        )
        if complete_offsets:
            offsets_recovered += 1
        else:
            add_problem(problems, name, index, wnd_id, "offset_recovery_failed")

        candidates = row.get("meta", {}).get("candidate_types", [])
        legal_types = event_types_within_candidates(
            chosen_recovered, candidates
        ) and event_types_within_candidates(rejected_recovered, candidates)
        if legal_types:
            candidate_legal += 1
        else:
            add_problem(problems, name, index, wnd_id, "candidate_type_violation")

        gold = parse_gold(row)
        if is_exact(chosen_recovered, gold):
            chosen_exact += 1
        else:
            add_problem(problems, name, index, wnd_id, "chosen_not_exact")
        if classify_single_error(rejected_recovered, gold) == category:
            rejected_atomic += 1
        else:
            add_problem(problems, name, index, wnd_id, "rejected_not_assigned_atomic_error")

        operation = meta.get("atomic_operation")
        try:
            replay_numeric = apply_atomic_proposal(
                gold,
                {"category": category, "operation": operation},
                parse_prompt_tokens(row["input"]),
            )
            replay_chosen, replay_rejected = render_canonical_pair(
                gold,
                replay_numeric,
                list(candidates),
                parse_prompt_tokens(row["input"]),
            )
            replay_ok = (
                is_exact(replay_numeric, rejected_recovered)
                and replay_chosen == chosen
                and replay_rejected == rejected
            )
        except (KeyError, TypeError, ValueError):
            replay_ok = False
        if replay_ok:
            canonical_replays += 1
        else:
            add_problem(problems, name, index, wnd_id, "atomic_operation_or_renderer_replay_failed")

        chosen_tokens = token_count(tokenizer, chosen)
        rejected_tokens = token_count(tokenizer, rejected)
        chosen_total = conversation_token_count(
            tokenizer, row["instruction"], row["input"], chosen
        )
        rejected_total = conversation_token_count(
            tokenizer, row["instruction"], row["input"], rejected
        )
        if not valid_length_pair(
            chosen_tokens,
            rejected_tokens,
            chosen_total,
            rejected_total,
            args.cutoff_len,
            args.min_length_ratio,
            args.max_length_ratio,
        ):
            add_problem(problems, name, index, wnd_id, "length_or_cutoff_violation")
        ratios.append(chosen_tokens / rejected_tokens)
        maximum_total = max(maximum_total, chosen_total, rejected_total)

    return {
        "pairs": len(pairs),
        "unique_windows": len(windows),
        "error_categories": dict(sorted(categories.items())),
        "proposal_sources": dict(sorted(proposal_sources.items())),
        "external_teacher_pairs": sum(
            1 for pair in pairs if pair.get("meta", {}).get("external_teacher") is not False
        ),
        "chosen_exact": chosen_exact,
        "rejected_single_target_error": rejected_atomic,
        "offset_recovered": offsets_recovered,
        "candidate_type_legal": candidate_legal,
        "complete_tags": tag_complete,
        "self_exact_qualifications": self_exact_qualifications,
        "canonical_replays": canonical_replays,
        "response_length_ratio_min": min(ratios) if ratios else None,
        "response_length_ratio_max": max(ratios) if ratios else None,
        "maximum_prompt_plus_response_tokens": maximum_total,
        "problems": problems,
    }


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--preference_jsonl", type=Path, required=True)
    parser.add_argument("--deterministic_preference_jsonl", type=Path, required=True)
    parser.add_argument("--assignment_manifest", type=Path, required=True)
    parser.add_argument("--build_summary", type=Path, required=True)
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--samples_glob", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--profile", choices=["e81", "g9"], required=True)
    parser.add_argument("--target_pairs", type=int, default=900)
    parser.add_argument("--category_quotas")
    parser.add_argument("--renderer_version", default="ac_rpo_v1")
    parser.add_argument("--cutoff_len", type=int, required=True)
    parser.add_argument("--min_length_ratio", type=float, default=0.9)
    parser.add_argument("--max_length_ratio", type=float, default=1.1)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--freeze_json", type=Path, required=True)
    args = parser.parse_args()

    quotas = parse_quotas(args.category_quotas, args.profile)
    if sum(quotas.values()) != args.target_pairs:
        raise ValueError("quota total does not equal target_pairs")
    source_rows = load_jsonl(args.input_jsonl)
    source_by_wnd = {row_id(row, index): row for index, row in enumerate(source_rows)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    main_pairs = load_jsonl(args.preference_jsonl)
    deterministic_pairs = load_jsonl(args.deterministic_preference_jsonl)
    manifest = load_jsonl(args.assignment_manifest)
    samples_by_wnd = load_sample_records(args.samples_glob, None)

    main_report = audit_dataset(
        name="observed_first",
        pairs=main_pairs,
        source_by_wnd=source_by_wnd,
        tokenizer=tokenizer,
        args=args,
        deterministic_only=False,
        samples_by_wnd=samples_by_wnd,
    )
    deterministic_report = audit_dataset(
        name="deterministic_only",
        pairs=deterministic_pairs,
        source_by_wnd=source_by_wnd,
        tokenizer=tokenizer,
        args=args,
        deterministic_only=True,
        samples_by_wnd=None,
    )
    global_problems: list[dict[str, Any]] = []
    expected_assignment = [
        (str(item.get("wnd_id")), str(item.get("error_category"))) for item in manifest
    ]
    main_assignment = [
        (str(pair.get("meta", {}).get("wnd_id")), str(pair.get("meta", {}).get("error_category")))
        for pair in main_pairs
    ]
    deterministic_assignment = [
        (str(pair.get("meta", {}).get("wnd_id")), str(pair.get("meta", {}).get("error_category")))
        for pair in deterministic_pairs
    ]
    if main_assignment != expected_assignment:
        add_problem(global_problems, "global", -1, "*", "main_assignment_manifest_mismatch")
    if deterministic_assignment != expected_assignment:
        add_problem(
            global_problems,
            "global",
            -1,
            "*",
            "deterministic_assignment_manifest_mismatch",
        )
    counts = Counter(category for _, category in main_assignment)
    if len(main_pairs) != args.target_pairs or len(set(main_assignment)) != args.target_pairs:
        add_problem(global_problems, "global", -1, "*", "target_or_unique_window_count_mismatch")
    if counts != Counter(quotas):
        add_problem(global_problems, "global", -1, "*", "category_quota_mismatch")

    build_summary = json.loads(args.build_summary.read_text(encoding="utf-8"))
    required_artifacts = {
        "preference",
        "deterministic_preference",
        "chosen_sft",
        "finalonly",
        "smoke16",
        "assignment_manifest",
    }
    paths = {
        name: Path(path)
        for name, path in build_summary.get("paths", {}).items()
        if name in required_artifacts
    }
    if set(paths) != required_artifacts:
        add_problem(global_problems, "global", -1, "*", "build_artifact_paths_incomplete")
    if paths.get("preference") != args.preference_jsonl.resolve():
        add_problem(global_problems, "global", -1, "*", "preference_path_mismatch")
    if paths.get("deterministic_preference") != args.deterministic_preference_jsonl.resolve():
        add_problem(global_problems, "global", -1, "*", "deterministic_path_mismatch")
    if paths.get("assignment_manifest") != args.assignment_manifest.resolve():
        add_problem(global_problems, "global", -1, "*", "assignment_path_mismatch")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, digest in hashes.items():
        if build_summary.get("sha256", {}).get(name) != digest:
            add_problem(global_problems, "global", -1, "*", f"build_hash_mismatch:{name}")
    if build_summary.get("valid") is not True:
        add_problem(global_problems, "global", -1, "*", "build_summary_not_valid")

    problems = main_report["problems"] + deterministic_report["problems"] + global_problems
    report = {
        "valid": not problems,
        "profile": args.profile,
        "target_pairs": args.target_pairs,
        "category_quotas": quotas,
        "renderer_version": args.renderer_version,
        "observed_first": {key: value for key, value in main_report.items() if key != "problems"},
        "deterministic_only": {
            key: value for key, value in deterministic_report.items() if key != "problems"
        },
        "sha256": hashes,
        "problem_count": len(problems),
        "problems": problems,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if report["valid"]:
        freeze = {
            "frozen": True,
            "preference_protocol": "E114 AC-RPO",
            "profile": args.profile,
            "target_pairs": args.target_pairs,
            "category_quotas": quotas,
            "renderer_version": args.renderer_version,
            "cutoff_len": args.cutoff_len,
            "length_ratio": [args.min_length_ratio, args.max_length_ratio],
            "artifacts": {
                name: {"path": str(path.resolve()), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "audit": {"path": str(args.output_json.resolve()), "sha256": sha256_file(args.output_json)},
        }
        args.freeze_json.parent.mkdir(parents=True, exist_ok=True)
        args.freeze_json.write_text(
            json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    elif args.freeze_json.exists():
        args.freeze_json.unlink()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
