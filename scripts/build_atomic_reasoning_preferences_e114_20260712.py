#!/usr/bin/env python3
"""Build paired canonical atomic-counterfactual preference datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
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
    register_dataset,
    row_id,
    token_count,
    write_jsonl,
)
from src.stage2_preference.atomic_counterfactual import (  # noqa: E402
    ATOMIC_CATEGORIES,
    aggregate_observed_proposals,
    apply_atomic_proposal,
    fallback_proposal,
    label_leaks,
    render_canonical_pair,
    select_quota_assignment,
)
from src.stage2_preference.reasoning_preference import (  # noqa: E402
    classify_single_error,
    event_types_within_candidates,
    extract_final_json,
    final_only_response,
    has_complete_reasoning_response,
    is_exact,
    offsets_complete,
    parse_prompt_tokens,
    recover_offsets_from_evidence,
    valid_length_pair,
)


DEFAULT_E81_QUOTAS = {category: 180 for category in ATOMIC_CATEGORIES}
DEFAULT_G9_QUOTAS = {
    "wrong_type": 135,
    "trigger_drift": 90,
    "argument_omission": 225,
    "event_omission": 270,
    "extra_frame": 180,
}


def parse_quotas(value: str | None, profile: str) -> dict[str, int]:
    if value is None:
        return dict(DEFAULT_E81_QUOTAS if profile == "e81" else DEFAULT_G9_QUOTAS)
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("category quotas must be a JSON object")
    quotas = {str(category): int(count) for category, count in payload.items()}
    if set(quotas) != set(ATOMIC_CATEGORIES) or any(count < 0 for count in quotas.values()):
        raise ValueError(f"invalid category quotas: {quotas}")
    return quotas


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structurally_valid_exact_sample(
    sample: dict[str, Any], row: dict[str, Any], gold: dict[str, Any]
) -> bool:
    response = sample.get("raw_response", "")
    complete = sample.get("complete_reasoning_response", has_complete_reasoning_response(response))
    payload = extract_final_json(response)
    if not isinstance(payload, dict) or not complete:
        return False
    recovered, diagnostics = recover_offsets_from_evidence(payload, row["input"])
    return bool(
        not diagnostics.get("missing_offsets")
        and offsets_complete(recovered)
        and event_types_within_candidates(recovered, row.get("meta", {}).get("candidate_types", []))
        and is_exact(recovered, gold)
    )


def eligibility_sample(
    samples: list[dict[str, Any]],
    row: dict[str, Any],
    gold: dict[str, Any],
    tokenizer: Any,
    cutoff_len: int,
) -> dict[str, Any] | None:
    eligible = []
    for sample in samples:
        if not structurally_valid_exact_sample(sample, row, gold):
            continue
        total_tokens = conversation_token_count(
            tokenizer, row["instruction"], row["input"], sample["raw_response"]
        )
        if total_tokens > cutoff_len:
            continue
        eligible.append(
            {
                "sample_seed": sample.get("sample_seed"),
                "sample_round": sample.get("sample_round"),
                "sample_index": sample.get("sample_index"),
                "response_tokens": token_count(tokenizer, sample["raw_response"]),
                "total_tokens": total_tokens,
            }
        )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item["total_tokens"],
            item["response_tokens"],
            int(item.get("sample_round") or 0),
            int(item.get("sample_index") or 0),
            int(item.get("sample_seed") or 0),
        ),
    )


def verify_rendered_pair(
    row: dict[str, Any],
    gold: dict[str, Any],
    category: str,
    chosen: str,
    rejected: str,
    tokenizer: Any,
    cutoff_len: int,
    minimum_ratio: float,
    maximum_ratio: float,
) -> dict[str, Any] | None:
    if not has_complete_reasoning_response(chosen) or not has_complete_reasoning_response(rejected):
        return None
    if label_leaks(chosen, row["input"]) or label_leaks(rejected, row["input"]):
        return None
    chosen_payload = extract_final_json(chosen)
    rejected_payload = extract_final_json(rejected)
    if chosen_payload is None or rejected_payload is None:
        return None
    chosen_recovered, chosen_diagnostics = recover_offsets_from_evidence(chosen_payload, row["input"])
    rejected_recovered, rejected_diagnostics = recover_offsets_from_evidence(
        rejected_payload, row["input"]
    )
    candidates = row.get("meta", {}).get("candidate_types", [])
    if (
        chosen_diagnostics.get("missing_offsets")
        or rejected_diagnostics.get("missing_offsets")
        or not offsets_complete(chosen_recovered)
        or not offsets_complete(rejected_recovered)
        or not event_types_within_candidates(chosen_recovered, candidates)
        or not event_types_within_candidates(rejected_recovered, candidates)
        or not is_exact(chosen_recovered, gold)
        or classify_single_error(rejected_recovered, gold) != category
    ):
        return None
    chosen_response_tokens = token_count(tokenizer, chosen)
    rejected_response_tokens = token_count(tokenizer, rejected)
    chosen_total_tokens = conversation_token_count(
        tokenizer, row["instruction"], row["input"], chosen
    )
    rejected_total_tokens = conversation_token_count(
        tokenizer, row["instruction"], row["input"], rejected
    )
    if not valid_length_pair(
        chosen_response_tokens,
        rejected_response_tokens,
        chosen_total_tokens,
        rejected_total_tokens,
        cutoff_len,
        minimum_ratio,
        maximum_ratio,
    ):
        return None
    return {
        "chosen_response_tokens": chosen_response_tokens,
        "rejected_response_tokens": rejected_response_tokens,
        "chosen_total_tokens": chosen_total_tokens,
        "rejected_total_tokens": rejected_total_tokens,
        "response_length_ratio": chosen_response_tokens / rejected_response_tokens,
    }


def build_option(
    row: dict[str, Any],
    gold: dict[str, Any],
    proposal: dict[str, Any],
    tokenizer: Any,
    cutoff_len: int,
    minimum_ratio: float,
    maximum_ratio: float,
    renderer_version: str,
) -> dict[str, Any] | None:
    category = str(proposal["category"])
    tokens = parse_prompt_tokens(row["input"])
    try:
        rejected_numeric = apply_atomic_proposal(gold, proposal, tokens)
        if classify_single_error(rejected_numeric, gold) != category:
            return None
        chosen, rejected = render_canonical_pair(
            gold,
            rejected_numeric,
            list(row.get("meta", {}).get("candidate_types", [])),
            tokens,
        )
    except (KeyError, TypeError, ValueError):
        return None
    lengths = verify_rendered_pair(
        row,
        gold,
        category,
        chosen,
        rejected,
        tokenizer,
        cutoff_len,
        minimum_ratio,
        maximum_ratio,
    )
    if lengths is None:
        return None
    return {
        "category": category,
        "chosen": chosen,
        "rejected": rejected,
        "proposal_source": proposal["proposal_source"],
        "frequency": int(proposal.get("frequency", 1)),
        "operation": proposal["operation"],
        "source_sample_seed": proposal.get("source_sample_seed"),
        "source_sample_round": proposal.get("source_sample_round"),
        "source_sample_index": proposal.get("source_sample_index"),
        "renderer_version": renderer_version,
        **lengths,
    }


def choose_observed_first_option(
    row: dict[str, Any],
    gold: dict[str, Any],
    category: str,
    observed: list[dict[str, Any]],
    fallback: dict[str, Any] | None,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    proposals = list(observed) if args.proposal_mode == "observed_first" else []
    if fallback is not None:
        proposals.append(fallback)
    for proposal in proposals:
        option = build_option(
            row,
            gold,
            proposal,
            tokenizer,
            args.cutoff_len,
            args.min_length_ratio,
            args.max_length_ratio,
            args.renderer_version,
        )
        if option is not None:
            return option
    return None


def pair_meta(
    wnd_id: str,
    option: dict[str, Any],
    eligibility: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    return {
        "wnd_id": wnd_id,
        "error_category": option["category"],
        "chosen_source": "canonical_gold_from_self_exact_window",
        "external_teacher": False,
        "proposal_source": option["proposal_source"],
        "atomic_operation": option["operation"],
        "source_sample_seed": option.get("source_sample_seed"),
        "source_sample_round": option.get("source_sample_round"),
        "source_sample_index": option.get("source_sample_index"),
        "proposal_frequency": option.get("frequency", 1),
        "renderer_version": option["renderer_version"],
        "eligibility_exact_sample": eligibility,
        "chosen_response_tokens": option["chosen_response_tokens"],
        "rejected_response_tokens": option["rejected_response_tokens"],
        "chosen_total_tokens": option["chosen_total_tokens"],
        "rejected_total_tokens": option["rejected_total_tokens"],
        "response_length_ratio": option["response_length_ratio"],
        "profile": profile,
    }


def make_pair(
    row: dict[str, Any],
    wnd_id: str,
    option: dict[str, Any],
    eligibility: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    return {
        "instruction": row["instruction"],
        "input": row["input"],
        "chosen": option["chosen"],
        "rejected": option["rejected"],
        "meta": pair_meta(wnd_id, option, eligibility, profile),
    }


def selected_smoke_pairs(pairs: list[dict[str, Any]], count: int = 16) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_category[pair["meta"]["error_category"]].append(pair)
    output = []
    index = 0
    categories = list(ATOMIC_CATEGORIES)
    while len(output) < min(count, len(pairs)):
        progressed = False
        for category in categories:
            if index < len(by_category[category]) and len(output) < count:
                output.append(by_category[category][index])
                progressed = True
        if not progressed:
            break
        index += 1
    return output


def finalonly_pairs(
    pairs: list[dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
    gold_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for pair in pairs:
        wnd_id = pair["meta"]["wnd_id"]
        prompt_row = source_rows.get(wnd_id)
        if prompt_row is None:
            raise ValueError(f"final-only prompt missing wnd_id: {wnd_id}")
        if prompt_row.get("input") != pair.get("input"):
            raise ValueError(f"final-only input mismatch: {wnd_id}")
        if parse_gold(prompt_row) != parse_gold(gold_rows[wnd_id]):
            raise ValueError(f"final-only gold mismatch: {wnd_id}")
        chosen = final_only_response(pair["chosen"])
        rejected = final_only_response(pair["rejected"])
        if chosen is None or rejected is None:
            raise ValueError(f"final-only extraction failed: {wnd_id}")
        converted = dict(pair)
        converted.update(
            {
                "instruction": prompt_row["instruction"],
                "input": prompt_row["input"],
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        output.append(converted)
    return output


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--samples_glob", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--dataset_info", type=Path, required=True)
    parser.add_argument("--preference_name", required=True)
    parser.add_argument("--deterministic_preference_name", required=True)
    parser.add_argument("--smoke_name", required=True)
    parser.add_argument("--finalonly_prompt_jsonl", type=Path, required=True)
    parser.add_argument("--target_pairs", type=int, default=900)
    parser.add_argument("--profile", choices=["e81", "g9"], default="e81")
    parser.add_argument("--category_quotas")
    parser.add_argument("--proposal_mode", choices=["observed_first", "deterministic_only"], default="observed_first")
    parser.add_argument("--renderer_version", default="ac_rpo_v1")
    parser.add_argument("--cutoff_len", type=int, default=1536)
    parser.add_argument("--min_length_ratio", type=float, default=0.9)
    parser.add_argument("--max_length_ratio", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=1140)
    args = parser.parse_args()

    quotas = parse_quotas(args.category_quotas, args.profile)
    if sum(quotas.values()) != args.target_pairs:
        raise ValueError(f"quota total {sum(quotas.values())} != target {args.target_pairs}")
    rows = load_jsonl(args.input_jsonl)
    by_wnd = {row_id(row, index): row for index, row in enumerate(rows)}
    if len(by_wnd) != len(rows):
        raise ValueError("duplicate wnd_id in input")
    samples_by_wnd = load_sample_records(args.samples_glob, None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    options_by_window: dict[str, dict[str, dict[str, Any]]] = {}
    control_options: dict[str, dict[str, dict[str, Any]]] = {}
    eligibility_by_window: dict[str, dict[str, Any]] = {}
    supply_counts: Counter[str] = Counter()
    observed_supply_counts: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()

    for wnd_id, row in by_wnd.items():
        gold = parse_gold(row)
        samples = samples_by_wnd.get(wnd_id, [])
        eligibility = eligibility_sample(samples, row, gold, tokenizer, args.cutoff_len)
        if eligibility is None:
            rejected_reasons["no_self_exact_within_cutoff"] += 1
            continue
        tokens = parse_prompt_tokens(row["input"])
        candidate_types = list(row.get("meta", {}).get("candidate_types", []))
        observed = aggregate_observed_proposals(gold, samples, candidate_types, tokens)
        main_window_options = {}
        control_window_options = {}
        for category in ATOMIC_CATEGORIES:
            fallback = fallback_proposal(category, gold, candidate_types, tokens)
            control = (
                build_option(
                    row,
                    gold,
                    fallback,
                    tokenizer,
                    args.cutoff_len,
                    args.min_length_ratio,
                    args.max_length_ratio,
                    args.renderer_version,
                )
                if fallback is not None
                else None
            )
            if control is None:
                continue
            main_option = choose_observed_first_option(
                row,
                gold,
                category,
                observed.get(category, []),
                fallback,
                tokenizer,
                args,
            )
            if main_option is None:
                continue
            main_window_options[category] = main_option
            control_window_options[category] = control
            supply_counts[category] += 1
            if main_option["proposal_source"] == "observed_atomic":
                observed_supply_counts[category] += 1
        if main_window_options:
            options_by_window[wnd_id] = main_window_options
            control_options[wnd_id] = control_window_options
            eligibility_by_window[wnd_id] = eligibility

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = args.output_dir / "build_failure.json"
    try:
        assignment = select_quota_assignment(options_by_window, quotas, args.seed)
    except ValueError as error:
        failure = {
            "valid": False,
            "reason": "quota_assignment_infeasible",
            "message": str(error),
            "target_pairs": args.target_pairs,
            "category_quotas": quotas,
            "self_exact_eligible_windows": len(eligibility_by_window),
            "windows_with_any_pair_option": len(options_by_window),
            "supply_counts": dict(sorted(supply_counts.items())),
            "observed_supply_counts": dict(sorted(observed_supply_counts.items())),
            "rejected_window_reasons": dict(sorted(rejected_reasons.items())),
            "renderer_version": args.renderer_version,
            "cutoff_len": args.cutoff_len,
            "length_ratio": [args.min_length_ratio, args.max_length_ratio],
            "seed": args.seed,
        }
        failure_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 3
    main_pairs = []
    deterministic_pairs = []
    manifest = []
    for item in assignment:
        wnd_id = item["wnd_id"]
        category = item["error_category"]
        row = by_wnd[wnd_id]
        eligibility = eligibility_by_window[wnd_id]
        main_option = item["option"]
        control_option = control_options[wnd_id][category]
        main_pair = make_pair(row, wnd_id, main_option, eligibility, args.profile)
        control_pair = make_pair(row, wnd_id, control_option, eligibility, args.profile)
        main_pairs.append(main_pair)
        deterministic_pairs.append(control_pair)
        manifest.append(
            {
                "wnd_id": wnd_id,
                "error_category": category,
                "main_proposal_source": main_option["proposal_source"],
                "main_atomic_operation": main_option["operation"],
                "deterministic_atomic_operation": control_option["operation"],
                "eligibility_exact_sample": eligibility,
            }
        )

    failure_path.unlink(missing_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    main_path = args.dataset_dir / f"{args.preference_name}.jsonl"
    deterministic_path = args.dataset_dir / f"{args.deterministic_preference_name}.jsonl"
    chosen_name = f"{args.preference_name}_chosen_sft"
    chosen_path = args.dataset_dir / f"{chosen_name}.jsonl"
    finalonly_name = f"{args.preference_name}_finalonly"
    finalonly_path = args.dataset_dir / f"{finalonly_name}.jsonl"
    smoke_path = args.dataset_dir / f"{args.smoke_name}.jsonl"
    manifest_path = args.output_dir / "assignment_manifest.jsonl"

    write_jsonl(main_path, main_pairs)
    write_jsonl(deterministic_path, deterministic_pairs)
    chosen_rows = [
        {
            "instruction": pair["instruction"],
            "input": pair["input"],
            "output": pair["chosen"],
            "meta": pair["meta"],
        }
        for pair in main_pairs
    ]
    write_jsonl(chosen_path, chosen_rows)
    final_prompt_rows = load_jsonl(args.finalonly_prompt_jsonl)
    final_prompts = {row_id(row, index): row for index, row in enumerate(final_prompt_rows)}
    final_rows = finalonly_pairs(main_pairs, final_prompts, by_wnd)
    write_jsonl(finalonly_path, final_rows)
    smoke_pairs = selected_smoke_pairs(main_pairs, 16)
    write_jsonl(smoke_path, smoke_pairs)
    write_jsonl(manifest_path, manifest)

    register_dataset(args.dataset_info, args.preference_name, main_path.name, ranking=True)
    register_dataset(
        args.dataset_info,
        args.deterministic_preference_name,
        deterministic_path.name,
        ranking=True,
    )
    register_dataset(args.dataset_info, chosen_name, chosen_path.name, ranking=False)
    register_dataset(args.dataset_info, finalonly_name, finalonly_path.name, ranking=True)
    register_dataset(args.dataset_info, args.smoke_name, smoke_path.name, ranking=True)

    category_counts = Counter(pair["meta"]["error_category"] for pair in main_pairs)
    proposal_counts = Counter(pair["meta"]["proposal_source"] for pair in main_pairs)
    hashes = {
        "preference": sha256_file(main_path),
        "deterministic_preference": sha256_file(deterministic_path),
        "chosen_sft": sha256_file(chosen_path),
        "finalonly": sha256_file(finalonly_path),
        "smoke16": sha256_file(smoke_path),
        "assignment_manifest": sha256_file(manifest_path),
    }
    summary = {
        "valid": len(main_pairs) == args.target_pairs and category_counts == Counter(quotas),
        "input_windows": len(rows),
        "sampled_windows": len(samples_by_wnd),
        "self_exact_eligible_windows": len(eligibility_by_window),
        "windows_with_any_pair_option": len(options_by_window),
        "target_pairs": args.target_pairs,
        "pairs": len(main_pairs),
        "category_quotas": quotas,
        "error_categories": dict(sorted(category_counts.items())),
        "proposal_sources": dict(sorted(proposal_counts.items())),
        "external_teacher_pairs": 0,
        "supply_counts": dict(sorted(supply_counts.items())),
        "observed_supply_counts": dict(sorted(observed_supply_counts.items())),
        "rejected_window_reasons": dict(sorted(rejected_reasons.items())),
        "profile": args.profile,
        "renderer_version": args.renderer_version,
        "proposal_mode": args.proposal_mode,
        "cutoff_len": args.cutoff_len,
        "length_ratio": [args.min_length_ratio, args.max_length_ratio],
        "seed": args.seed,
        "paths": {
            "preference": str(main_path.resolve()),
            "deterministic_preference": str(deterministic_path.resolve()),
            "chosen_sft": str(chosen_path.resolve()),
            "finalonly": str(finalonly_path.resolve()),
            "smoke16": str(smoke_path.resolve()),
            "assignment_manifest": str(manifest_path.resolve()),
        },
        "sha256": hashes,
    }
    summary_path = args.output_dir / "build_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
