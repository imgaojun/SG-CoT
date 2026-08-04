#!/usr/bin/env python3
import argparse
import collections
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORMAL_ROOT = REPO / "outputs/stage2_1_7b_paired_augmentation/e27_formal_20260527"
OUT_ROOT = REPO / "outputs/stage2_error_diagnosis_e34/gold_conditioned_20260531"
REPORT_PATH = REPO / "reports/2026-05-31_e34_gold_conditioned_error_diagnosis.md"
DEFAULT_BASE_URL = "${LLM_BASE_URL}"

VARIANTS = {
    "e32a": "trigger_preserving_tail_natural_step",
    "e32c": "trigger_role_ground_direct_anchor",
}
SPLITS = ["test_seen", "test_unseen"]
ERROR_CATEGORIES = [
    "missing_event",
    "spurious_event",
    "wrong_event_type",
    "wrong_trigger",
    "missing_argument",
    "spurious_argument",
    "wrong_role",
    "argument_boundary",
    "invalid_json",
    "format_error",
    "over_generation",
    "under_generation",
    "schema_confusion",
    "other",
]
ROOT_CAUSES = [
    "trigger_detection",
    "event_type_confusion",
    "argument_role_grounding",
    "argument_boundary",
    "over_generation",
    "under_generation",
    "schema_generalization",
    "multi_event_confusion",
    "format_following",
    "other",
]
DATA_RECOMMENDATIONS = [
    "add_hard_negative",
    "add_role_contrast",
    "add_argument_specificity",
    "add_multi_event",
    "add_unseen_schema_generalization",
    "add_trigger_disambiguation",
    "add_event_count_contrast",
    "add_negative_no_event",
    "other",
]


def load_jsonl(path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_text(input_text):
    match = re.search(r"Text:\n(.*?)\n\nTokens:", input_text, flags=re.S)
    return match.group(1).strip() if match else input_text.strip()


def extract_schema_cards(input_text):
    match = re.search(r"Candidate event types:\n(.*?)\n\nSchema cards:\n(.*?)\n\nReturn JSON only\.", input_text, flags=re.S)
    if not match:
        return [], ""
    candidate_types = [x.strip() for x in match.group(1).split(",") if x.strip()]
    return candidate_types, match.group(2).strip()


def pred_path(variant, budget, split):
    return FORMAL_ROOT / variant / f"forced_{budget}" / split / "predictions.jsonl"


def score_sum(row):
    return float(row.get("argument_f1", 0.0)) + float(row.get("event_f1", 0.0)) + float(row.get("trigger_f1", 0.0))


def bucket_case(none_row, standard_row, improve_threshold=0.2, wrong_threshold=0.5, correct_threshold=2.5):
    none_sum = score_sum(none_row)
    std_sum = score_sum(standard_row)
    if std_sum - none_sum >= improve_threshold:
        return "reason_improves"
    if none_sum - std_sum >= improve_threshold:
        return "reason_hurts"
    if none_sum <= wrong_threshold and std_sum <= wrong_threshold:
        return "both_wrong"
    if none_sum >= correct_threshold and std_sum >= correct_threshold:
        return "both_correct"
    return "mixed_or_tie"


def make_case(variant, split, index, none_row, standard_row):
    candidate_types, schema_cards = extract_schema_cards(none_row["input"])
    case_id = f"{variant}_{split}_{index:04d}"
    direct_metrics = {
        "argument_f1": none_row.get("argument_f1"),
        "event_f1": none_row.get("event_f1"),
        "trigger_f1": none_row.get("trigger_f1"),
        "sum": score_sum(none_row),
    }
    reason_metrics = {
        "argument_f1": standard_row.get("argument_f1"),
        "event_f1": standard_row.get("event_f1"),
        "trigger_f1": standard_row.get("trigger_f1"),
        "sum": score_sum(standard_row),
    }
    return {
        "case_id": case_id,
        "variant": variant,
        "variant_label": VARIANTS[variant],
        "split": split,
        "index": index,
        "comparison_bucket": bucket_case(none_row, standard_row),
        "text": extract_text(none_row["input"]),
        "candidate_event_types": candidate_types,
        "schema_cards": schema_cards,
        "gold_events": none_row.get("gold", {}).get("events", []),
        "direct_prediction": none_row.get("final_predicted", {}).get("events", []),
        "reason_prediction": standard_row.get("final_predicted", {}).get("events", []),
        "direct_metrics": direct_metrics,
        "reason_metrics": reason_metrics,
        "direct_valid_json": bool(none_row.get("valid_final_json")),
        "reason_valid_json": bool(standard_row.get("valid_final_json")),
        "direct_generated_text": none_row.get("generated_text", ""),
        "reason_generated_text": standard_row.get("generated_text", ""),
    }


def build_cases():
    cases = []
    for variant in VARIANTS:
        for split in SPLITS:
            none_rows = load_jsonl(pred_path(variant, "none", split))
            standard_rows = load_jsonl(pred_path(variant, "standard", split))
            if len(none_rows) != len(standard_rows):
                raise ValueError(f"row count mismatch for {variant}/{split}: {len(none_rows)} vs {len(standard_rows)}")
            for i, (none_row, standard_row) in enumerate(zip(none_rows, standard_rows)):
                cases.append(make_case(variant, split, i, none_row, standard_row))
    return cases


def sample_cases(cases, per_bucket, seed, include_buckets):
    rng = random.Random(seed)
    grouped = collections.defaultdict(list)
    for case in cases:
        key = (case["variant"], case["split"], case["comparison_bucket"])
        if case["comparison_bucket"] in include_buckets:
            grouped[key].append(case)

    sampled = []
    for key in sorted(grouped):
        items = grouped[key]
        items = sorted(items, key=lambda x: abs(x["reason_metrics"]["sum"] - x["direct_metrics"]["sum"]), reverse=True)
        if len(items) > per_bucket:
            head = items[: max(1, per_bucket // 2)]
            rest = items[max(1, per_bucket // 2) :]
            rng.shuffle(rest)
            picked = head + rest[: per_bucket - len(head)]
        else:
            picked = items
        sampled.extend(picked)
    sampled.sort(key=lambda x: (x["variant"], x["split"], x["comparison_bucket"], x["case_id"]))
    return sampled


def diagnosis_prompt(case):
    contract = {
        "case_id": case["case_id"],
        "comparison_summary": {
            "better_system": "direct|reason|tie|both_bad",
            "reason_effect": "fixes_error|introduces_error|mixed|no_change",
            "one_sentence": "short summary",
        },
        "direct_errors": [
            {
                "category": "|".join(ERROR_CATEGORIES),
                "severity": "major|minor",
                "gold_reference": "short text or JSON pointer",
                "prediction_reference": "short text or JSON pointer",
                "explanation": "why this is an error",
            }
        ],
        "reason_errors": [],
        "reason_fixed_errors": [
            {"category": "|".join(ERROR_CATEGORIES), "explanation": "what reason fixed vs direct"}
        ],
        "reason_introduced_errors": [
            {"category": "|".join(ERROR_CATEGORIES), "explanation": "what reason made worse vs direct"}
        ],
        "root_cause": ROOT_CAUSES,
        "data_recommendation": DATA_RECOMMENDATIONS,
    }
    payload = {
        "task": "Gold-conditioned error diagnosis for event extraction. Do not repair predictions. Compare predictions against gold and label errors.",
        "rules": [
            "Use the gold_events as the ground truth.",
            "Do not infer new gold labels unless noting obvious spurious prediction risk.",
            "Direct prediction is forced none. Reason prediction is forced standard reasoning.",
            "Mark missing gold items as missing_event or missing_argument.",
            "Mark extra predicted items not in gold as spurious_event or spurious_argument.",
            "If text span is close but too wide/narrow, use argument_boundary or wrong_trigger.",
            "Return JSON only, matching the output contract.",
        ],
        "allowed_error_categories": ERROR_CATEGORIES,
        "allowed_root_causes": ROOT_CAUSES,
        "allowed_data_recommendations": DATA_RECOMMENDATIONS,
        "case": {
            "case_id": case["case_id"],
            "variant": case["variant"],
            "split": case["split"],
            "comparison_bucket": case["comparison_bucket"],
            "text": case["text"],
            "candidate_event_types": case["candidate_event_types"],
            "schema_cards": case["schema_cards"],
            "gold_events": case["gold_events"],
            "direct_prediction": case["direct_prediction"],
            "reason_prediction": case["reason_prediction"],
            "direct_metrics": case["direct_metrics"],
            "reason_metrics": case["reason_metrics"],
            "direct_valid_json": case["direct_valid_json"],
            "reason_valid_json": case["reason_valid_json"],
        },
        "output_contract": contract,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def extract_json(text):
    if text is None:
        raise ValueError("empty content")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def call_model(base_url, api_key, model, prompt, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful evaluator for event extraction. You output strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    latency = time.time() - started
    data = json.loads(raw)
    choice = data["choices"][0]
    return {
        "latency_sec": latency,
        "finish_reason": choice.get("finish_reason"),
        "content": choice.get("message", {}).get("content"),
        "usage": data.get("usage", {}),
        "raw_response": data,
    }


def validate_annotation(obj):
    errors = []
    if not isinstance(obj, dict):
        return ["annotation_not_object"]
    summary = obj.get("comparison_summary")
    if not isinstance(summary, dict):
        errors.append("missing_comparison_summary")
    else:
        if summary.get("better_system") not in {"direct", "reason", "tie", "both_bad"}:
            errors.append("bad_better_system")
        if summary.get("reason_effect") not in {"fixes_error", "introduces_error", "mixed", "no_change"}:
            errors.append("bad_reason_effect")
    for field in ["direct_errors", "reason_errors", "reason_fixed_errors", "reason_introduced_errors"]:
        if not isinstance(obj.get(field), list):
            errors.append(f"{field}_not_list")
    for field in ["root_cause", "data_recommendation"]:
        if not isinstance(obj.get(field), list):
            errors.append(f"{field}_not_list")
    return errors


def run_diagnosis(cases, args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required unless --prepare-only is used")
    out_path = args.output_dir / "llm_error_annotations.jsonl"
    existing = {}
    if args.reuse_existing and out_path.exists():
        for row in load_jsonl(out_path):
            if row.get("annotation_valid"):
                existing[row["case_id"]] = row
    rows = []
    for case in cases:
        if case["case_id"] in existing:
            rows.append(existing[case["case_id"]])
            continue
        rec = {
            "case_id": case["case_id"],
            "variant": case["variant"],
            "split": case["split"],
            "comparison_bucket": case["comparison_bucket"],
            "api_ok": False,
            "json_ok": False,
            "annotation_valid": False,
        }
        try:
            response = call_model(
                args.base_url,
                api_key,
                args.model,
                diagnosis_prompt(case),
                args.max_tokens,
                args.timeout,
            )
            rec.update(response)
            rec["api_ok"] = True
            obj = extract_json(response["content"])
            rec["json_ok"] = True
            rec["annotation"] = obj
            validation_errors = validate_annotation(obj)
            rec["annotation_validation_errors"] = validation_errors
            rec["annotation_valid"] = not validation_errors
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, KeyError) as exc:
            rec["error"] = repr(exc)
        rows.append(rec)
        with out_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rows


def summarize(cases, annotations):
    case_by_id = {c["case_id"]: c for c in cases}
    summary = {
        "total_cases": len(cases),
        "annotated": len(annotations),
        "api_success": sum(1 for r in annotations if r.get("api_ok")),
        "json_success": sum(1 for r in annotations if r.get("json_ok")),
        "annotation_valid": sum(1 for r in annotations if r.get("annotation_valid")),
        "by_variant_split_bucket": {},
        "better_system": collections.Counter(),
        "reason_effect": collections.Counter(),
        "direct_error_categories": collections.Counter(),
        "reason_error_categories": collections.Counter(),
        "reason_fixed_categories": collections.Counter(),
        "reason_introduced_categories": collections.Counter(),
        "root_cause": collections.Counter(),
        "data_recommendation": collections.Counter(),
        "avg_latency_sec": None,
    }
    latencies = []
    for case in cases:
        key = f"{case['variant']}/{case['split']}/{case['comparison_bucket']}"
        summary["by_variant_split_bucket"][key] = summary["by_variant_split_bucket"].get(key, 0) + 1
    for row in annotations:
        if row.get("latency_sec") is not None:
            latencies.append(row["latency_sec"])
        ann = row.get("annotation") or {}
        comp = ann.get("comparison_summary") or {}
        summary["better_system"][comp.get("better_system", "missing")] += 1
        summary["reason_effect"][comp.get("reason_effect", "missing")] += 1
        for err in ann.get("direct_errors") or []:
            summary["direct_error_categories"][err.get("category", "missing")] += 1
        for err in ann.get("reason_errors") or []:
            summary["reason_error_categories"][err.get("category", "missing")] += 1
        for err in ann.get("reason_fixed_errors") or []:
            summary["reason_fixed_categories"][err.get("category", "missing")] += 1
        for err in ann.get("reason_introduced_errors") or []:
            summary["reason_introduced_categories"][err.get("category", "missing")] += 1
        for item in ann.get("root_cause") or []:
            summary["root_cause"][item] += 1
        for item in ann.get("data_recommendation") or []:
            summary["data_recommendation"][item] += 1
    if latencies:
        summary["avg_latency_sec"] = sum(latencies) / len(latencies)
    for key in [
        "better_system",
        "reason_effect",
        "direct_error_categories",
        "reason_error_categories",
        "reason_fixed_categories",
        "reason_introduced_categories",
        "root_cause",
        "data_recommendation",
    ]:
        summary[key] = dict(summary[key])
    summary["sample_metric_means"] = {}
    grouped = collections.defaultdict(list)
    for case in cases:
        grouped[f"{case['variant']}/{case['split']}"].append(case)
    for key, items in grouped.items():
        summary["sample_metric_means"][key] = {
            "direct_sum": sum(x["direct_metrics"]["sum"] for x in items) / len(items),
            "reason_sum": sum(x["reason_metrics"]["sum"] for x in items) / len(items),
            "n": len(items),
        }
    return summary


def write_report(summary, args):
    lines = [
        "# E34 Gold-conditioned LLM Error Diagnosis",
        "",
        f"- model: `{args.model}`",
        f"- output: `{args.output_dir}`",
        f"- total cases: `{summary['total_cases']}`",
        f"- API/JSON/annotation-valid: `{summary['api_success']} / {summary['json_success']} / {summary['annotation_valid']}`",
        f"- avg latency sec: `{summary['avg_latency_sec']:.2f}`" if summary["avg_latency_sec"] else "- avg latency sec: pending",
        "",
        "## Sample Buckets",
        "",
    ]
    for key, value in sorted(summary["by_variant_split_bucket"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Better System", ""]
    for key, value in sorted(summary["better_system"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Reason Effect", ""]
    for key, value in sorted(summary["reason_effect"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Error Categories", "", "### Direct", ""]
    for key, value in sorted(summary["direct_error_categories"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "### Reason", ""]
    for key, value in sorted(summary["reason_error_categories"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Reason Fixed", ""]
    for key, value in sorted(summary["reason_fixed_categories"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Reason Introduced", ""]
    for key, value in sorted(summary["reason_introduced_categories"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Root Causes", ""]
    for key, value in sorted(summary["root_cause"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Data Recommendations", ""]
    for key, value in sorted(summary["data_recommendation"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Reading",
        "",
        "Pending manual reading of representative cases.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--per-bucket", type=int, default=2)
    ap.add_argument("--sample-seed", type=int, default=34)
    ap.add_argument("--include-buckets", nargs="+", default=["reason_improves", "reason_hurts", "both_wrong", "mixed_or_tie"])
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_cases = build_cases()
    sampled = sample_cases(all_cases, args.per_bucket, args.sample_seed, set(args.include_buckets))
    write_jsonl(args.output_dir / "sampled_cases.jsonl", sampled)
    (args.output_dir / "sampling_summary.json").write_text(
        json.dumps(
            {
                "total_available_cases": len(all_cases),
                "sampled_cases": len(sampled),
                "per_bucket": args.per_bucket,
                "include_buckets": args.include_buckets,
                "sample_seed": args.sample_seed,
                "bucket_counts": dict(collections.Counter(c["comparison_bucket"] for c in sampled)),
                "variant_split_counts": dict(collections.Counter(f"{c['variant']}/{c['split']}" for c in sampled)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.prepare_only:
        print(json.dumps({"sampled_cases": len(sampled), "output_dir": str(args.output_dir)}, indent=2))
        return
    annotations = run_diagnosis(sampled, args)
    summary = summarize(sampled, annotations)
    (args.output_dir / "error_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_report(summary, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
