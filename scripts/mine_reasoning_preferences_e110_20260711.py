#!/usr/bin/env python3
"""Mine reasoning-path preferences and emit LLaMAFactory ORPO datasets."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_preference.reasoning_preference import (  # noqa: E402
    ERROR_CATEGORIES,
    classify_single_error,
    event_types_within_candidates,
    extract_final_json,
    final_only_response,
    has_complete_reasoning_response,
    is_exact,
    metric_f1s,
    offsets_complete,
    recover_offsets_from_evidence,
    valid_length_pair,
    weighted_quality,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_prompt(tokenizer: Any, instruction: str, input_text: str) -> str:
    content = f"{instruction}\n{input_text}"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("meta", {}).get("wnd_id") or row.get("wnd_id") or f"row-{index:06d}")


def parse_gold(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("gold_output", row.get("output", "{}"))
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict):
        raise ValueError("gold output is not a JSON object")
    return payload


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def conversation_token_count(
    tokenizer: Any, instruction: str, input_text: str, response: str
) -> int:
    user_content = f"{instruction}\n{input_text}"
    if getattr(tokenizer, "chat_template", None):
        return len(
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
        )
    return token_count(tokenizer, user_content + "\n" + response)


def load_completed_window_ids(paths: set[Path], sample_round: int) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        completed.update(
            str(record["wnd_id"])
            for record in load_jsonl(path)
            if record.get("sample_round") == sample_round
        )
    return completed


def diagnose_response(
    response: str,
    row: dict[str, Any],
    gold: dict[str, Any],
    tokenizer: Any,
    prompt_tokens: int,
    sample_seed: int,
    sample_index: int,
) -> dict[str, Any]:
    final_payload = extract_final_json(response)
    recovered = None
    recovery = {"missing_offsets": None}
    metrics = {"argument": 0.0, "event": 0.0, "trigger": 0.0}
    error_category = "invalid_json"
    candidate_ok = False
    if final_payload is not None:
        recovered, recovery = recover_offsets_from_evidence(final_payload, row["input"])
        metrics = metric_f1s(recovered, gold)
        candidate_ok = event_types_within_candidates(
            recovered, row.get("meta", {}).get("candidate_types", [])
        )
        category = classify_single_error(recovered, gold)
        error_category = category or "multiple_or_other"
        if not offsets_complete(recovered):
            error_category = "unrecoverable_offset"
        elif not candidate_ok:
            error_category = "out_of_candidates"
    response_tokens = token_count(tokenizer, response)
    return {
        "sample_index": sample_index,
        "sample_seed": sample_seed,
        "raw_response": response,
        "final_json": final_payload,
        "recovered": recovered,
        "recovery": recovery,
        "metrics": metrics,
        "error_category": error_category,
        "complete_reasoning_response": has_complete_reasoning_response(response),
        "is_exact": bool(recovered is not None and is_exact(recovered, gold)),
        "offsets_complete": bool(recovered is not None and offsets_complete(recovered)),
        "candidate_types_valid": candidate_ok,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
    }


def sample(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".shard-{args.shard_index:02d}-of-{args.num_shards:02d}" if args.num_shards > 1 else ""
    sample_path = output_dir / f"samples{suffix}.jsonl"
    rows = load_jsonl(Path(args.input_jsonl))
    selected = [
        (index, row)
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
    ]
    if args.wnd_ids_json:
        requested_wnd_ids = set(json.loads(Path(args.wnd_ids_json).read_text(encoding="utf-8")))
        selected = [
            (index, row) for index, row in selected if row_id(row, index) in requested_wnd_ids
        ]
    if args.max_examples is not None:
        selected = selected[: args.max_examples]

    completed: set[str] = set()
    if not args.overwrite:
        completed_paths = {sample_path} if sample_path.exists() else set()
        if args.completed_samples_glob:
            completed_paths.update(Path(path) for path in glob.glob(args.completed_samples_glob))
        completed = load_completed_window_ids(completed_paths, args.sample_round)
    elif sample_path.exists():
        sample_path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    started = time.time()
    written = 0
    with sample_path.open("a", encoding="utf-8") as handle:
        for local_position, (index, row) in enumerate(selected):
            wnd_id = row_id(row, index)
            if wnd_id in completed:
                continue
            prompt = build_prompt(tokenizer, row["instruction"], row["input"])
            prompt_tokens = token_count(tokenizer, prompt)
            encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
            sample_seed = args.seed + args.sample_round * 1_000_003 + index * 17
            random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_width = encoded["input_ids"].shape[1]
            gold = parse_gold(row)
            samples = []
            for sample_index, output_ids in enumerate(generated):
                response = tokenizer.decode(output_ids[prompt_width:], skip_special_tokens=True)
                prefix = row.get("response_prefix", "")
                if prefix:
                    response = f"{prefix}{response}"
                samples.append(
                    diagnose_response(
                        response,
                        row,
                        gold,
                        tokenizer,
                        prompt_tokens,
                        sample_seed,
                        sample_index,
                    )
                )
            record = {
                "row_index": index,
                "wnd_id": wnd_id,
                "sample_round": args.sample_round,
                "profile": args.profile,
                "prompt_tokens": prompt_tokens,
                "samples": samples,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if written % args.log_every == 0 or local_position + 1 == len(selected):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    json.dumps(
                        {
                            "written": written,
                            "shard_rows": len(selected),
                            "rows_per_minute": round(written * 60 / elapsed, 2),
                            "last_wnd_id": wnd_id,
                        }
                    ),
                    flush=True,
                )
    summary = summarize_samples(load_jsonl(sample_path))
    summary.update(
        {
            "input_jsonl": str(Path(args.input_jsonl).resolve()),
            "model_path": str(args.model_path),
            "num_samples": args.num_samples,
            "sample_round": args.sample_round,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        }
    )
    dump_json(output_dir / f"sample_summary{suffix}.json", summary)
    return sample_path


def summarize_samples(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [sample for record in records for sample in record.get("samples", [])]
    categories = Counter(sample.get("error_category", "missing") for sample in samples)
    windows_exact = sum(any(sample.get("is_exact") for sample in record.get("samples", [])) for record in records)
    return {
        "windows": len(records),
        "samples": len(samples),
        "windows_with_exact": windows_exact,
        "windows_with_exact_rate": windows_exact / len(records) if records else 0.0,
        "valid_json_rate": sum(sample.get("final_json") is not None for sample in samples) / len(samples)
        if samples
        else 0.0,
        "offset_complete_rate": sum(bool(sample.get("offsets_complete")) for sample in samples) / len(samples)
        if samples
        else 0.0,
        "complete_reasoning_response_rate": sum(
            has_complete_reasoning_response(sample.get("raw_response", "")) for sample in samples
        )
        / len(samples)
        if samples
        else 0.0,
        "error_categories": dict(sorted(categories.items())),
    }


def load_sample_records(
    pattern: str, max_sample_round: int | None = None
) -> dict[str, list[dict[str, Any]]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"no sample files matched: {pattern}")
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_keys = set()
    for path in paths:
        for record in load_jsonl(path):
            wnd_id = str(record["wnd_id"])
            sample_round = int(record.get("sample_round", 0))
            if max_sample_round is not None and sample_round > max_sample_round:
                continue
            for sample_record in record.get("samples", []):
                key = (
                    wnd_id,
                    sample_record.get("sample_seed"),
                    sample_round,
                    sample_record.get("sample_index"),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                enriched = dict(sample_record)
                enriched["sample_round"] = sample_round
                by_window[wnd_id].append(enriched)
    return by_window


def teacher_fallback_allowed(samples: list[dict[str, Any]], allow_teacher: bool) -> bool:
    return allow_teacher and any(int(sample.get("sample_round", 0)) >= 1 for sample in samples)


def make_pair_candidate(
    row: dict[str, Any],
    wnd_id: str,
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    chosen_source: str,
    profile: str,
) -> dict[str, Any]:
    category = str(rejected["error_category"])
    return {
        "instruction": row["instruction"],
        "input": row["input"],
        "chosen": chosen["raw_response"],
        "rejected": rejected["raw_response"],
        "meta": {
            "wnd_id": wnd_id,
            "error_category": category,
            "chosen_source": chosen_source,
            "sample_seed": rejected.get("sample_seed"),
            "chosen_sample_seed": chosen.get("sample_seed"),
            "chosen_sample_round": chosen.get("sample_round"),
            "rejected_sample_round": rejected.get("sample_round"),
            "rejected_quality": rejected["weighted_quality"],
            "chosen_response_tokens": chosen["response_tokens"],
            "rejected_response_tokens": rejected["response_tokens"],
            "chosen_total_tokens": chosen["total_tokens"],
            "rejected_total_tokens": rejected["total_tokens"],
            "profile": profile,
        },
    }


def prepare_window_candidate(
    row: dict[str, Any],
    wnd_id: str,
    samples: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    gold = parse_gold(row)
    candidates = row.get("meta", {}).get("candidate_types", [])
    valid_samples = []
    for sample in samples:
        if not (
            sample.get("final_json") is not None
            and sample.get(
                "complete_reasoning_response",
                has_complete_reasoning_response(sample.get("raw_response", "")),
            )
            and sample.get("offsets_complete")
            and sample.get("candidate_types_valid")
        ):
            continue
        enriched = dict(sample)
        enriched["total_tokens"] = conversation_token_count(
            tokenizer, row["instruction"], row["input"], sample["raw_response"]
        )
        valid_samples.append(enriched)
    chosen_samples = [sample for sample in valid_samples if sample.get("is_exact")]
    rejected_samples = []
    for sample in valid_samples:
        category = sample.get("error_category")
        if category not in ERROR_CATEGORIES:
            continue
        recovered = sample.get("recovered")
        if not isinstance(recovered, dict):
            continue
        enriched = dict(sample)
        enriched["weighted_quality"] = weighted_quality(recovered, gold, args.profile)
        rejected_samples.append(enriched)

    chosen_source = "sample_exact"
    if not chosen_samples and teacher_fallback_allowed(samples, args.allow_teacher):
        teacher_response = row.get("output", "")
        teacher_payload = extract_final_json(teacher_response)
        if teacher_payload is not None and has_complete_reasoning_response(teacher_response):
            teacher_recovered, teacher_recovery = recover_offsets_from_evidence(teacher_payload, row["input"])
            teacher_tokens = token_count(tokenizer, teacher_response)
            teacher_total_tokens = conversation_token_count(
                tokenizer, row["instruction"], row["input"], teacher_response
            )
            if (
                teacher_recovery["missing_offsets"] == 0
                and offsets_complete(teacher_recovered)
                and event_types_within_candidates(teacher_recovered, candidates)
                and is_exact(teacher_recovered, gold)
            ):
                chosen_samples = [
                    {
                        "raw_response": teacher_response,
                        "sample_seed": None,
                        "response_tokens": teacher_tokens,
                        "total_tokens": teacher_total_tokens,
                    }
                ]
                chosen_source = "verified_teacher_trace"
    if not chosen_samples or not rejected_samples:
        return None

    possible_pairs = []
    for chosen in sorted(chosen_samples, key=lambda item: item["response_tokens"]):
        for rejected in rejected_samples:
            if valid_length_pair(
                chosen["response_tokens"],
                rejected["response_tokens"],
                chosen["total_tokens"],
                rejected["total_tokens"],
                args.cutoff_len,
                args.min_length_ratio,
                args.max_length_ratio,
            ):
                possible_pairs.append((chosen, rejected))
    if not possible_pairs:
        return None
    chosen, rejected = max(
        possible_pairs,
        key=lambda pair: (
            pair[1]["weighted_quality"],
            -abs(pair[0]["response_tokens"] - pair[1]["response_tokens"]),
            -pair[1]["response_tokens"],
        ),
    )
    pair = make_pair_candidate(row, wnd_id, chosen, rejected, chosen_source, args.profile)
    pair["meta"]["sample_rounds_observed"] = sorted(
        {int(sample.get("sample_round", 0)) for sample in samples}
    )
    return pair


def category_group(pair: dict[str, Any], profile: str) -> str:
    category = pair["meta"]["error_category"]
    if profile == "g9" and category in {"argument_omission", "event_omission"}:
        return "omission"
    return category


def category_cap(group: str, profile: str) -> float:
    if profile == "e81":
        return 0.40
    if profile == "g9" and group == "omission":
        return 0.60
    return 1.0


def select_balanced_pairs(
    candidates: list[dict[str, Any]], profile: str, maximum_pairs: int | None = None
) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in candidates:
        by_group[category_group(pair, profile)].append(pair)
    for pairs in by_group.values():
        pairs.sort(
            key=lambda pair: (
                pair["meta"]["chosen_source"] != "sample_exact",
                -pair["meta"]["rejected_quality"],
                pair["meta"]["wnd_id"],
            )
        )

    total_candidates = min(len(candidates), maximum_pairs or len(candidates))
    for target in range(total_candidates, 0, -1):
        quotas = {
            group: min(len(pairs), math.floor(category_cap(group, profile) * target + 1e-9))
            for group, pairs in by_group.items()
        }
        if sum(quotas.values()) < target:
            continue
        self_selected = []
        selected_per_group = Counter()
        for group, pairs in sorted(by_group.items()):
            group_self = [
                pair for pair in pairs if pair["meta"]["chosen_source"] == "sample_exact"
            ][: quotas[group]]
            self_selected.extend(group_self)
            selected_per_group[group] = len(group_self)
        self_selected.sort(
            key=lambda pair: (
                -pair["meta"]["rejected_quality"],
                pair["meta"]["wnd_id"],
            )
        )
        if len(self_selected) >= target:
            return self_selected[:target]
        teacher_needed = target - len(self_selected)
        if teacher_needed > math.floor(0.30 * target + 1e-9):
            continue
        teacher_candidates = []
        for group, pairs in sorted(by_group.items()):
            available = quotas[group] - selected_per_group[group]
            teacher_candidates.extend(
                [
                    pair
                    for pair in pairs
                    if pair["meta"]["chosen_source"] == "verified_teacher_trace"
                ][:available]
            )
        teacher_candidates.sort(
            key=lambda pair: (
                -pair["meta"]["rejected_quality"],
                pair["meta"]["wnd_id"],
            )
        )
        if len(teacher_candidates) >= teacher_needed:
            return self_selected + teacher_candidates[:teacher_needed]
    return []


def windows_requiring_topup(
    window_ids: list[str], candidates: list[dict[str, Any]]
) -> list[str]:
    paired_windows = {str(pair["meta"]["wnd_id"]) for pair in candidates}
    return [wnd_id for wnd_id in window_ids if wnd_id not in paired_windows]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def register_dataset(dataset_info_path: Path, name: str, file_name: str, ranking: bool) -> None:
    dataset_info = (
        json.loads(dataset_info_path.read_text(encoding="utf-8"))
        if dataset_info_path.exists()
        else {}
    )
    columns = {"prompt": "instruction", "query": "input"}
    if ranking:
        columns.update({"chosen": "chosen", "rejected": "rejected"})
    else:
        columns["response"] = "output"
    entry: dict[str, Any] = {"file_name": file_name, "columns": columns}
    if ranking:
        entry["ranking"] = True
    existing = dataset_info.get(name)
    if existing is not None and existing != entry:
        raise ValueError(f"dataset_info already contains a different entry for {name}")
    dataset_info[name] = entry
    dump_json(dataset_info_path, dataset_info)


def build(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    from transformers import AutoTokenizer

    input_path = Path(args.input_jsonl)
    rows = load_jsonl(input_path)
    by_wnd = {row_id(row, index): row for index, row in enumerate(rows)}
    if len(by_wnd) != len(rows):
        raise ValueError("duplicate wnd_id values in input_jsonl")
    sample_pattern = args.samples_glob or str(Path(args.output_dir) / "samples*.jsonl")
    samples = load_sample_records(sample_pattern, args.max_sample_round)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    candidates = []
    missing_windows = 0
    for wnd_id, row in by_wnd.items():
        candidate = prepare_window_candidate(
            row, wnd_id, samples.get(wnd_id, []), tokenizer, args
        )
        if candidate is None:
            missing_windows += 1
        else:
            candidates.append(candidate)
    pairs = select_balanced_pairs(candidates, args.profile, args.max_pairs)
    pairs.sort(key=lambda pair: pair["meta"]["wnd_id"])
    topup_wnd_ids = windows_requiring_topup(list(by_wnd), candidates)

    dataset_dir = Path(args.dataset_dir)
    dataset_path = dataset_dir / f"{args.preference_name}.jsonl"
    write_jsonl(dataset_path, pairs)
    register_dataset(
        Path(args.dataset_info), args.preference_name, dataset_path.name, ranking=True
    )

    chosen_name = f"{args.preference_name}_chosen_sft"
    chosen_rows = [
        {
            "instruction": pair["instruction"],
            "input": pair["input"],
            "output": pair["chosen"],
            "meta": pair["meta"],
        }
        for pair in pairs
    ]
    chosen_path = dataset_dir / f"{chosen_name}.jsonl"
    write_jsonl(chosen_path, chosen_rows)
    register_dataset(Path(args.dataset_info), chosen_name, chosen_path.name, ranking=False)

    finalonly_prompt_rows: dict[str, dict[str, Any]] = {}
    if args.finalonly_prompt_jsonl:
        prompt_rows = load_jsonl(Path(args.finalonly_prompt_jsonl))
        finalonly_prompt_rows = {
            row_id(row, index): row for index, row in enumerate(prompt_rows)
        }
        if len(finalonly_prompt_rows) != len(prompt_rows):
            raise ValueError("duplicate wnd_id values in finalonly_prompt_jsonl")

    final_name = f"{args.preference_name}_finalonly"
    final_pairs = []
    for pair in pairs:
        chosen_final = final_only_response(pair["chosen"])
        rejected_final = final_only_response(pair["rejected"])
        if chosen_final and rejected_final:
            final_pair = dict(pair)
            if finalonly_prompt_rows:
                wnd_id = str(pair["meta"]["wnd_id"])
                prompt_row = finalonly_prompt_rows.get(wnd_id)
                if prompt_row is None:
                    raise ValueError(f"final-only prompt row missing wnd_id: {wnd_id}")
                if prompt_row.get("input") != pair.get("input"):
                    raise ValueError(f"final-only input mismatch for wnd_id: {wnd_id}")
                if parse_gold(prompt_row) != parse_gold(by_wnd[wnd_id]):
                    raise ValueError(f"final-only gold mismatch for wnd_id: {wnd_id}")
                final_pair["instruction"] = prompt_row["instruction"]
                final_pair["input"] = prompt_row["input"]
            final_pair["chosen"] = chosen_final
            final_pair["rejected"] = rejected_final
            final_pairs.append(final_pair)
    final_path = dataset_dir / f"{final_name}.jsonl"
    write_jsonl(final_path, final_pairs)
    register_dataset(Path(args.dataset_info), final_name, final_path.name, ranking=True)

    category_counts = Counter(pair["meta"]["error_category"] for pair in pairs)
    source_counts = Counter(pair["meta"]["chosen_source"] for pair in pairs)
    summary = {
        "input_rows": len(rows),
        "sampled_windows": len(samples),
        "candidate_pairs_before_caps": len(candidates),
        "pairs": len(pairs),
        "missing_or_filtered_windows": missing_windows,
        "topup_windows": len(topup_wnd_ids),
        "minimum_required_pairs": args.min_pairs,
        "maximum_pairs": args.max_pairs,
        "meets_minimum": len(pairs) >= args.min_pairs,
        "requires_k8_topup": len(pairs) < args.min_pairs,
        "teacher_fraction": source_counts.get("verified_teacher_trace", 0) / len(pairs)
        if pairs
        else 0.0,
        "chosen_sources": dict(sorted(source_counts.items())),
        "error_categories": dict(sorted(category_counts.items())),
        "profile": args.profile,
        "teacher_requires_sample_round": 1,
        "maximum_sample_round_loaded": args.max_sample_round,
        "cutoff_len": args.cutoff_len,
        "length_ratio": [args.min_length_ratio, args.max_length_ratio],
        "preference_dataset": str(dataset_path.resolve()),
        "chosen_sft_dataset": str(chosen_path.resolve()),
        "finalonly_dataset": str(final_path.resolve()),
        "finalonly_prompt_jsonl": str(Path(args.finalonly_prompt_jsonl).resolve())
        if args.finalonly_prompt_jsonl
        else None,
    }
    summary_path = Path(args.output_dir) / "pair_summary.json"
    dump_json(summary_path, summary)
    dump_json(Path(args.output_dir) / "topup_wnd_ids.json", topup_wnd_ids)
    return dataset_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "build", "all"], default="all")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1104)
    parser.add_argument("--profile", choices=["e81", "g9"], required=True)
    parser.add_argument("--sample_round", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--wnd_ids_json")
    parser.add_argument("--completed_samples_glob")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--samples_glob")
    parser.add_argument("--max_sample_round", type=int)
    parser.add_argument("--dataset_dir", default="data/stage2_adaptive_datasets")
    parser.add_argument(
        "--dataset_info", default="data/stage2_adaptive_datasets/dataset_info.json"
    )
    parser.add_argument("--preference_name", default="e110_reasoning_path_preferences")
    parser.add_argument("--finalonly_prompt_jsonl")
    parser.add_argument("--cutoff_len", type=int, default=1536)
    parser.add_argument("--min_pairs", type=int, default=900)
    parser.add_argument("--max_pairs", type=int)
    parser.add_argument("--min_length_ratio", type=float, default=0.7)
    parser.add_argument("--max_length_ratio", type=float, default=1.3)
    parser.add_argument("--allow_teacher", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num_samples must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard_index must be in [0, num_shards)")
    return args


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.mode in {"sample", "all"}:
        sample_path = sample(args)
        print(f"samples={sample_path}")
    if args.mode in {"build", "all"}:
        dataset_path, summary = build(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"preference_dataset={dataset_path}")
        if not summary["meets_minimum"]:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
