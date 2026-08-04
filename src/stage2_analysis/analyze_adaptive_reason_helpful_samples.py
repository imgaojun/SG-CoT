import argparse
from collections import Counter, defaultdict
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import (  # noqa: E402
    DIRECT_EVAL_JSONL,
    HARDCONF_FOCUS_RUNS,
    SPLITS,
    build_feature_map,
    prediction_key,
    prediction_path,
    score,
)
from src.stage2_analysis.analyze_adaptive_route_case_studies import (  # noqa: E402
    argument_items,
    categorize_fn_arg,
    categorize_fp_arg,
    compact_events,
    event_sets,
    extract_text_block,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


FEATURE_KEYS = [
    "hardconf_score",
    "confusion_score",
    "confusion_norm",
    "role_signature_rarity",
    "role_density_norm",
    "multi_event_or_multi_trigger",
    "core_role_absence_risk",
]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def payload(row):
    return row.get("final_predicted") or row.get("predicted") or {"events": []}


def metric(row):
    return {
        "trigger_f1": float(row.get("trigger_f1", 0.0) or 0.0),
        "argument_f1": float(row.get("argument_f1", 0.0) or 0.0),
        "event_f1": float(row.get("event_f1", 0.0) or 0.0),
        "score": score(row),
        "valid_json": bool(row.get("valid_final_json", row.get("valid_json", False))),
    }


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def feature_means(samples):
    if not samples:
        return {key: 0.0 for key in FEATURE_KEYS}
    return {
        key: mean([(sample.get("features") or {}).get(key, 0.0) for sample in samples])
        for key in FEATURE_KEYS
    }


def feature_auc(samples, key):
    positives = [
        (sample.get("features") or {}).get(key, 0.0)
        for sample in samples
        if sample["reason_helpful"]
    ]
    negatives = [
        (sample.get("features") or {}).get(key, 0.0)
        for sample in samples
        if not sample["reason_helpful"]
    ]
    if not positives or not negatives:
        return 0.0
    wins = ties = total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1
            elif pos == neg:
                ties += 1
    return (wins + 0.5 * ties) / total if total else 0.0


def top_bucket_rate(samples, key, frac=0.2):
    if not samples:
        return {"top_rate": 0.0, "rest_rate": 0.0, "top_count": 0, "rest_count": 0}
    ranked = sorted(samples, key=lambda sample: (sample.get("features") or {}).get(key, 0.0), reverse=True)
    cap = max(1, round(len(ranked) * frac))
    top = ranked[:cap]
    rest = ranked[cap:]
    return {
        "top_rate": mean([1.0 if sample["reason_helpful"] else 0.0 for sample in top]),
        "rest_rate": mean([1.0 if sample["reason_helpful"] else 0.0 for sample in rest]),
        "top_count": len(top),
        "rest_count": len(rest),
    }


def first_by_key(items):
    return {item["key"]: item for item in items}


def trigger_exact_correct(row):
    gold_sets = event_sets(row.get("gold") or {})
    pred_sets = event_sets(payload(row))
    return bool(gold_sets["triggers"] & pred_sets["triggers"])


def pair_diagnosis(direct_row, reason_row):
    gold_payload = direct_row.get("gold") or {"events": []}
    direct_payload = payload(direct_row)
    reason_payload = payload(reason_row)

    gold_sets = event_sets(gold_payload)
    direct_sets = event_sets(direct_payload)
    reason_sets = event_sets(reason_payload)

    gold_args = argument_items(gold_payload)
    direct_args = argument_items(direct_payload)
    reason_args = argument_items(reason_payload)
    gold_by_key = first_by_key(gold_args)
    direct_by_key = first_by_key(direct_args)
    reason_by_key = first_by_key(reason_args)

    gold_keys = set(gold_by_key)
    direct_keys = set(direct_by_key)
    reason_keys = set(reason_by_key)

    recovered_arg_keys = (gold_keys - direct_keys) & reason_keys
    lost_arg_keys = (gold_keys & direct_keys) - reason_keys
    removed_fp_keys = (direct_keys - gold_keys) - reason_keys
    added_fp_keys = (reason_keys - gold_keys) - direct_keys

    recovered_fn_categories = Counter()
    removed_fp_categories = Counter()
    lost_categories = Counter()
    added_fp_categories = Counter()
    recovered_roles = Counter()
    lost_roles = Counter()
    removed_fp_roles = Counter()
    added_fp_roles = Counter()

    for key in recovered_arg_keys:
        gold_arg = gold_by_key[key]
        recovered_fn_categories[categorize_fn_arg(gold_arg, direct_args)] += 1
        recovered_roles[gold_arg["role"]] += 1
    for key in lost_arg_keys:
        gold_arg = gold_by_key[key]
        lost_categories[categorize_fn_arg(gold_arg, reason_args)] += 1
        lost_roles[gold_arg["role"]] += 1
    for key in removed_fp_keys:
        pred_arg = direct_by_key[key]
        removed_fp_categories[categorize_fp_arg(pred_arg, gold_args)] += 1
        removed_fp_roles[pred_arg["role"]] += 1
    for key in added_fp_keys:
        pred_arg = reason_by_key[key]
        added_fp_categories[categorize_fp_arg(pred_arg, gold_args)] += 1
        added_fp_roles[pred_arg["role"]] += 1

    direct_trigger_correct = bool(gold_sets["triggers"] & direct_sets["triggers"])
    reason_trigger_correct = bool(gold_sets["triggers"] & reason_sets["triggers"])

    return {
        "recovered_arg_count": len(recovered_arg_keys),
        "lost_arg_count": len(lost_arg_keys),
        "removed_fp_arg_count": len(removed_fp_keys),
        "added_fp_arg_count": len(added_fp_keys),
        "recovered_trigger_count": len((gold_sets["triggers"] - direct_sets["triggers"]) & reason_sets["triggers"]),
        "lost_trigger_count": len((gold_sets["triggers"] & direct_sets["triggers"]) - reason_sets["triggers"]),
        "recovered_event_count": len((gold_sets["events"] - direct_sets["events"]) & reason_sets["events"]),
        "lost_event_count": len((gold_sets["events"] & direct_sets["events"]) - reason_sets["events"]),
        "direct_trigger_correct": direct_trigger_correct,
        "reason_trigger_correct": reason_trigger_correct,
        "trigger_correct_arg_repair": direct_trigger_correct and len(recovered_arg_keys) > 0,
        "recovered_fn_categories": recovered_fn_categories,
        "removed_fp_categories": removed_fp_categories,
        "lost_categories": lost_categories,
        "added_fp_categories": added_fp_categories,
        "recovered_roles": recovered_roles,
        "lost_roles": lost_roles,
        "removed_fp_roles": removed_fp_roles,
        "added_fp_roles": added_fp_roles,
    }


def compact_counter(counter, limit=12):
    return counter.most_common(limit)


def add_counter(dst, src):
    for key, value in src.items():
        dst[key] += value


def extract_plan(row, max_chars=800):
    text = row.get("generated_payload") or row.get("generated_text") or ""
    for tag in ["PLAN", "REASON"]:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            continue
        end = text.find(end_tag, start + len(start_tag))
        if end == -1:
            return text[start : start + max_chars]
        return text[start + len(start_tag) : end][:max_chars].strip()
    return ""


def build_features_for_split(split: str, schema_path: Path):
    return build_feature_map(Path(DIRECT_EVAL_JSONL[split]), schema_path)


def analyze_run_split(root: Path, run, split: str, schema_path: Path):
    paths = {
        mode: prediction_path(root, run, mode, split)
        for mode in ["free_route", "forced_direct", "forced_reason"]
    }
    missing = [path.as_posix() for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing predictions for {run['branch']}/{run['selection']}/{split}: {missing}")

    free = load_prediction_map(paths["free_route"])
    direct = load_prediction_map(paths["forced_direct"])
    reason = load_prediction_map(paths["forced_reason"])
    features = build_features_for_split(split, schema_path)

    samples = []
    aggregate = {
        "recovered_fn_categories": Counter(),
        "removed_fp_categories": Counter(),
        "lost_categories": Counter(),
        "added_fp_categories": Counter(),
        "recovered_roles": Counter(),
        "lost_roles": Counter(),
        "removed_fp_roles": Counter(),
        "added_fp_roles": Counter(),
    }
    aggregate_counts = Counter()
    helpful_aggregate = {
        "recovered_fn_categories": Counter(),
        "removed_fp_categories": Counter(),
        "lost_categories": Counter(),
        "added_fp_categories": Counter(),
        "recovered_roles": Counter(),
        "lost_roles": Counter(),
        "removed_fp_roles": Counter(),
        "added_fp_roles": Counter(),
    }
    helpful_counts = Counter()

    for key in sorted(set(free) & set(direct) & set(reason)):
        free_row = free[key]
        direct_row = direct[key]
        reason_row = reason[key]
        direct_metric = metric(direct_row)
        reason_metric = metric(reason_row)
        free_metric = metric(free_row)
        diagnosis = pair_diagnosis(direct_row, reason_row)
        reason_gain = reason_metric["score"] - direct_metric["score"]
        argument_gain = reason_metric["argument_f1"] - direct_metric["argument_f1"]
        event_gain = reason_metric["event_f1"] - direct_metric["event_f1"]
        trigger_gain = reason_metric["trigger_f1"] - direct_metric["trigger_f1"]
        sample = {
            "key": key,
            "split": split,
            "branch": run["branch"],
            "selection": run["selection"],
            "free_route_pred": free_row.get("route_pred", "unknown"),
            "free_reason_used": bool(free_row.get("reason_used")),
            "direct_metric": direct_metric,
            "reason_metric": reason_metric,
            "free_metric": free_metric,
            "reason_gain": reason_gain,
            "argument_gain": argument_gain,
            "event_gain": event_gain,
            "trigger_gain": trigger_gain,
            "reason_helpful": reason_gain > 1e-9,
            "argument_helpful": argument_gain > 1e-9,
            "event_helpful": event_gain > 1e-9,
            "reason_harmful": reason_gain < -1e-9,
            "features": features.get(key, {}),
            "diagnosis": {
                key_name: value
                for key_name, value in diagnosis.items()
                if not isinstance(value, Counter)
            },
        }
        samples.append(sample)

        for count_key in [
            "recovered_arg_count",
            "lost_arg_count",
            "removed_fp_arg_count",
            "added_fp_arg_count",
            "recovered_trigger_count",
            "lost_trigger_count",
            "recovered_event_count",
            "lost_event_count",
            "trigger_correct_arg_repair",
        ]:
            aggregate_counts[count_key] += int(diagnosis[count_key])
            if sample["reason_helpful"]:
                helpful_counts[count_key] += int(diagnosis[count_key])
        for counter_key in aggregate:
            add_counter(aggregate[counter_key], diagnosis[counter_key])
            if sample["reason_helpful"]:
                add_counter(helpful_aggregate[counter_key], diagnosis[counter_key])

    helpful = [sample for sample in samples if sample["reason_helpful"]]
    harmful = [sample for sample in samples if sample["reason_harmful"]]
    ties = [sample for sample in samples if not sample["reason_helpful"] and not sample["reason_harmful"]]
    free_reason = [sample for sample in samples if sample["free_route_pred"] == "reason"]
    captured = [sample for sample in helpful if sample["free_route_pred"] == "reason"]

    return {
        "num_examples": len(samples),
        "reason_helpful_count": len(helpful),
        "reason_helpful_rate": len(helpful) / len(samples) if samples else 0.0,
        "reason_harmful_count": len(harmful),
        "tie_count": len(ties),
        "argument_helpful_count": sum(1 for sample in samples if sample["argument_helpful"]),
        "event_helpful_count": sum(1 for sample in samples if sample["event_helpful"]),
        "free_reason_count": len(free_reason),
        "capture_recall": len(captured) / len(helpful) if helpful else 0.0,
        "capture_precision": len(captured) / len(free_reason) if free_reason else 0.0,
        "mean_gains": {
            "score": mean([sample["reason_gain"] for sample in samples]),
            "argument_f1": mean([sample["argument_gain"] for sample in samples]),
            "event_f1": mean([sample["event_gain"] for sample in samples]),
            "trigger_f1": mean([sample["trigger_gain"] for sample in samples]),
        },
        "feature_means": {
            "reason_helpful": feature_means(helpful),
            "reason_not_helpful": feature_means([sample for sample in samples if not sample["reason_helpful"]]),
            "reason_harmful": feature_means(harmful),
        },
        "feature_auc": {key: feature_auc(samples, key) for key in FEATURE_KEYS},
        "top20_feature_helpful_rate": {key: top_bucket_rate(samples, key) for key in FEATURE_KEYS},
        "diagnosis_all": {
            "counts": dict(aggregate_counts),
            "recovered_fn_categories": compact_counter(aggregate["recovered_fn_categories"]),
            "removed_fp_categories": compact_counter(aggregate["removed_fp_categories"]),
            "lost_categories": compact_counter(aggregate["lost_categories"]),
            "added_fp_categories": compact_counter(aggregate["added_fp_categories"]),
            "recovered_roles": compact_counter(aggregate["recovered_roles"]),
            "lost_roles": compact_counter(aggregate["lost_roles"]),
            "removed_fp_roles": compact_counter(aggregate["removed_fp_roles"]),
            "added_fp_roles": compact_counter(aggregate["added_fp_roles"]),
        },
        "diagnosis_reason_helpful": {
            "counts": dict(helpful_counts),
            "recovered_fn_categories": compact_counter(helpful_aggregate["recovered_fn_categories"]),
            "removed_fp_categories": compact_counter(helpful_aggregate["removed_fp_categories"]),
            "lost_categories": compact_counter(helpful_aggregate["lost_categories"]),
            "added_fp_categories": compact_counter(helpful_aggregate["added_fp_categories"]),
            "recovered_roles": compact_counter(helpful_aggregate["recovered_roles"]),
            "lost_roles": compact_counter(helpful_aggregate["lost_roles"]),
            "removed_fp_roles": compact_counter(helpful_aggregate["removed_fp_roles"]),
            "added_fp_roles": compact_counter(helpful_aggregate["added_fp_roles"]),
        },
        "samples": samples,
    }


def summarize_all_samples(run_payloads):
    out = {}
    for split in SPLITS:
        samples = []
        for run in run_payloads:
            samples.extend(run["splits"][split]["samples"])
        helpful = [sample for sample in samples if sample["reason_helpful"]]
        out[split] = {
            "num_examples": len(samples),
            "reason_helpful_count": len(helpful),
            "reason_helpful_rate": len(helpful) / len(samples) if samples else 0.0,
            "mean_gains": {
                "score": mean([sample["reason_gain"] for sample in samples]),
                "argument_f1": mean([sample["argument_gain"] for sample in samples]),
                "event_f1": mean([sample["event_gain"] for sample in samples]),
                "trigger_f1": mean([sample["trigger_gain"] for sample in samples]),
            },
            "feature_means": {
                "reason_helpful": feature_means(helpful),
                "reason_not_helpful": feature_means([sample for sample in samples if not sample["reason_helpful"]]),
            },
            "feature_auc": {key: feature_auc(samples, key) for key in FEATURE_KEYS},
            "top20_feature_helpful_rate": {key: top_bucket_rate(samples, key) for key in FEATURE_KEYS},
        }
    return out


def build_case(run, split, key, direct_row, reason_row, free_row, sample):
    diagnosis = pair_diagnosis(direct_row, reason_row)
    return {
        "branch": run["branch"],
        "selection": run["selection"],
        "split": split,
        "key": key,
        "text": extract_text_block(direct_row.get("input", "")),
        "candidate_types": (direct_row.get("meta") or {}).get("candidate_types"),
        "gold_event_types": (direct_row.get("meta") or {}).get("gold_event_types"),
        "free_route_pred": free_row.get("route_pred"),
        "direct_metric": metric(direct_row),
        "reason_metric": metric(reason_row),
        "reason_gain": sample["reason_gain"],
        "argument_gain": sample["argument_gain"],
        "event_gain": sample["event_gain"],
        "features": sample.get("features") or {},
        "diagnosis": {
            key_name: value
            for key_name, value in diagnosis.items()
            if not isinstance(value, Counter)
        },
        "recovered_fn_categories": compact_counter(diagnosis["recovered_fn_categories"], 6),
        "removed_fp_categories": compact_counter(diagnosis["removed_fp_categories"], 6),
        "recovered_roles": compact_counter(diagnosis["recovered_roles"], 8),
        "gold": compact_events(direct_row.get("gold") or {}),
        "direct_pred": compact_events(payload(direct_row)),
        "reason_pred": compact_events(payload(reason_row)),
        "reason_plan": extract_plan(reason_row),
    }


def collect_cases(root: Path, schema_path: Path, focus_runs, per_split=3):
    primary = [
        run
        for run in focus_runs
        if run["branch"] == "hardconf10_type_role_hint_plan_lite"
        and run["selection"] == "seen_stable_best"
    ][0]
    cases = []
    for split in SPLITS:
        free = load_prediction_map(prediction_path(root, primary, "free_route", split))
        direct = load_prediction_map(prediction_path(root, primary, "forced_direct", split))
        reason = load_prediction_map(prediction_path(root, primary, "forced_reason", split))
        features = build_features_for_split(split, schema_path)
        scored = []
        for key in sorted(set(free) & set(direct) & set(reason)):
            direct_row = direct[key]
            reason_row = reason[key]
            sample = {
                "reason_gain": score(reason_row) - score(direct_row),
                "argument_gain": metric(reason_row)["argument_f1"] - metric(direct_row)["argument_f1"],
                "event_gain": metric(reason_row)["event_f1"] - metric(direct_row)["event_f1"],
                "features": features.get(key, {}),
            }
            if sample["reason_gain"] <= 1e-9:
                continue
            scored.append((sample["reason_gain"], sample["argument_gain"], sample["event_gain"], key, sample))
        scored.sort(reverse=True)
        for _, _, _, key, sample in scored[:per_split]:
            cases.append(build_case(primary, split, key, direct[key], reason[key], free[key], sample))
    return cases


def analyze(root: Path, schema_path: Path, focus_runs):
    runs = []
    for run in focus_runs:
        split_payload = {}
        for split in SPLITS:
            split_payload[split] = analyze_run_split(root, run, split, schema_path)
        runs.append({**run, "splits": split_payload})

    aggregate = summarize_all_samples(runs)
    cases = collect_cases(root, schema_path, focus_runs)

    # Keep artifact compact by removing per-sample rows from the run table. Case studies retain examples.
    compact_runs = []
    for run in runs:
        split_payload = {}
        for split, payload_dict in run["splits"].items():
            split_payload[split] = {k: v for k, v in payload_dict.items() if k != "samples"}
        compact_runs.append({**{k: v for k, v in run.items() if k != "splits"}, "splits": split_payload})
    return {
        "definition": "reason_helpful iff forced_reason_score > forced_direct_score, where score = argument_f1 + event_f1 + 0.25 * trigger_f1",
        "runs": compact_runs,
        "aggregate_by_split": aggregate,
        "case_studies": cases,
    }


def fmt(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def counter_text(items, limit=6):
    return ", ".join(f"{name}:{count}" for name, count in items[:limit]) or "-"


def feature_diff_text(features):
    helpful = features["reason_helpful"]
    not_helpful = features["reason_not_helpful"]
    return ", ".join(
        f"{key} {helpful.get(key, 0.0):.3f}/{not_helpful.get(key, 0.0):.3f}"
        for key in ["hardconf_score", "confusion_norm", "role_signature_rarity", "role_density_norm", "core_role_absence_risk"]
    )


def markdown_report(payload):
    lines = [
        "# Adaptive Reason-Helpful Sample Deep Analysis",
        "",
        "## Definition",
        "",
        "- `reason-helpful`: same sample/checkpoint where `forced_reason_score > forced_direct_score`.",
        "- `score = argument_f1 + event_f1 + 0.25 * trigger_f1`.",
        "- This is post-hoc analysis from formal predictions; it is not used as a training label in the completed runs.",
        "",
        "## Aggregate Feature Reading",
        "",
        "| split | examples | helpful | helpful_rate | mean gain arg/event/score | helpful vs non-helpful features | top20 hardconf helpful rate | hardconf AUC |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for split in SPLITS:
        row = payload["aggregate_by_split"][split]
        hardconf_top = row["top20_feature_helpful_rate"]["hardconf_score"]
        lines.append(
            "| `{split}` | {n} | {h} | {hr:.3f} | {ag:+.4f}/{eg:+.4f}/{sg:+.4f} | {features} | {top:.3f} vs {rest:.3f} | {auc:.3f} |".format(
                split=split,
                n=row["num_examples"],
                h=row["reason_helpful_count"],
                hr=row["reason_helpful_rate"],
                ag=row["mean_gains"]["argument_f1"],
                eg=row["mean_gains"]["event_f1"],
                sg=row["mean_gains"]["score"],
                features=feature_diff_text(row["feature_means"]),
                top=hardconf_top["top_rate"],
                rest=hardconf_top["rest_rate"],
                auc=row["feature_auc"]["hardconf_score"],
            )
        )

    lines.extend(
        [
            "",
            "## Run-Level Reason Helpfulness",
            "",
            "| branch | selection | split | helpful/harmful/tie | arg-help | event-help | capture R/P | mean gain arg/event | feature contrast |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in payload["runs"]:
        for split in SPLITS:
            row = run["splits"][split]
            lines.append(
                "| `{branch}` | `{selection}` | `{split}` | {helpful}/{harmful}/{tie} | {arg_help} | {event_help} | {rec:.3f}/{prec:.3f} | {ag:+.4f}/{eg:+.4f} | {features} |".format(
                    branch=run["branch"],
                    selection=run["selection"],
                    split=split,
                    helpful=row["reason_helpful_count"],
                    harmful=row["reason_harmful_count"],
                    tie=row["tie_count"],
                    arg_help=row["argument_helpful_count"],
                    event_help=row["event_helpful_count"],
                    rec=row["capture_recall"],
                    prec=row["capture_precision"],
                    ag=row["mean_gains"]["argument_f1"],
                    eg=row["mean_gains"]["event_f1"],
                    features=feature_diff_text(row["feature_means"]),
                )
            )

    lines.extend(["", "## Error-Repair Mechanisms", ""])
    lines.append("| branch | selection | split | recovered args | lost args | removed FP args | added FP args | trigger-correct arg repairs | recovered roles | recovered FN categories |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for run in payload["runs"]:
        if run["selection"] != "seen_stable_best":
            continue
        for split in SPLITS:
            diag = run["splits"][split]["diagnosis_reason_helpful"]
            counts = diag["counts"]
            lines.append(
                "| `{}` | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                    run["branch"],
                    run["selection"],
                    split,
                    counts.get("recovered_arg_count", 0),
                    counts.get("lost_arg_count", 0),
                    counts.get("removed_fp_arg_count", 0),
                    counts.get("added_fp_arg_count", 0),
                    counts.get("trigger_correct_arg_repair", 0),
                    counter_text(diag["recovered_roles"]),
                    counter_text(diag["recovered_fn_categories"]),
                )
            )

    lines.extend(
        [
            "",
            "## Case Studies",
            "",
            "Top reason-helpful samples are selected from `hardconf10_type_role_hint_plan_lite / seen_stable_best` by score gain.",
        ]
    )
    for idx, case in enumerate(payload["case_studies"], 1):
        feature_bits = ", ".join(
            f"{key}={case['features'].get(key, 0.0):.3f}"
            for key in ["hardconf_score", "confusion_norm", "role_signature_rarity", "role_density_norm", "core_role_absence_risk"]
        )
        lines.extend(
            [
                "",
                f"### Case {idx}: `{case['split']}` `{case['key']}`",
                "",
                f"- gain score/arg/event: `{case['reason_gain']:+.4f}` / `{case['argument_gain']:+.4f}` / `{case['event_gain']:+.4f}`",
                f"- features: {feature_bits}",
                f"- direct metric: `{case['direct_metric']}`",
                f"- reason metric: `{case['reason_metric']}`",
                f"- diagnosis: `{case['diagnosis']}`",
                f"- recovered roles: `{case['recovered_roles']}`",
                f"- recovered FN categories: `{case['recovered_fn_categories']}`",
                f"- text: {case['text'][:700]}",
                "",
                "Gold:",
                "```json",
                json.dumps(case["gold"], ensure_ascii=False, indent=2),
                "```",
                "Direct:",
                "```json",
                json.dumps(case["direct_pred"], ensure_ascii=False, indent=2),
                "```",
                "Reason:",
                "```json",
                json.dumps(case["reason_pred"], ensure_ascii=False, indent=2),
                "```",
            ]
        )
        if case["reason_plan"]:
            lines.extend(["Plan:", "```text", case["reason_plan"][:800], "```"])

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Reason-helpful samples are not an unseen-only phenomenon; they appear in `test_seen` and `test` as well.",
            "- The strongest actionable signal is sample hardness: high hardconf/confusion/role-risk samples have higher reason-helpfulness rates, but the signal is imperfect.",
            "- Most useful reason gains come from argument recovery and role/span repair after the event trigger is already partly correct; this supports a role-aware router rather than a seen/unseen router.",
            "- Free-route capture remains the bottleneck when capture recall/precision is near zero; the next route labels need outcome-aware or explicitly route-balanced supervision.",
            "- Directdup controls still show path diversity gains, so claims should separate reasoning-specific repairs from checkpoint/decoding diversity.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_root", default="outputs/stage2_adaptive_runs_user_formal_clean")
    parser.add_argument("--schema_path", default="data/schema/richere-en.event_schema.json")
    parser.add_argument("--output_md", default="reports/2026-05-12_stage2_adaptive_reason_helpful_samples_analysis.md")
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-12_stage2_adaptive_reason_helpful_samples_analysis.json")
    args = parser.parse_args()

    payload = analyze(Path(args.formal_root), Path(args.schema_path), HARDCONF_FOCUS_RUNS)
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), markdown_report(payload))
    print(json.dumps({"output_md": args.output_md, "output_json": args.output_json}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
