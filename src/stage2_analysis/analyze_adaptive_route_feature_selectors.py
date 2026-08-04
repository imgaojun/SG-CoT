import argparse
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
from src.stage2_analysis.analyze_adaptive_reason_helpful_samples import (  # noqa: E402
    extract_plan,
    extract_text_block,
    metric,
    payload,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


BUDGETS = [0.05, 0.10, 0.15, 0.20]
FOCUS_RUN_KEYS = {
    ("hardconf10_type_role_hint_plan_lite", "seen_stable_best"),
    ("hardconf10_type_role_hint_plan_lite", "hard_reason_best"),
    ("hardconf10_calibrated_type_role_hint_plan_lite", "seen_stable_best"),
    ("hardconf10_directdup", "seen_stable_best"),
}


SELECTORS = {
    "hardconf": lambda f: f.get("hardconf_score", 0.0),
    "confusion": lambda f: f.get("confusion_norm", 0.0),
    "role_signature": lambda f: f.get("role_signature_rarity", 0.0),
    "role_density": lambda f: f.get("role_density_norm", 0.0),
    "multi_event": lambda f: f.get("multi_event_or_multi_trigger", 0.0),
    "core_absence_high": lambda f: f.get("core_role_absence_risk", 0.0),
    "core_absence_low": lambda f: -f.get("core_role_absence_risk", 0.0),
    "hardconf_no_absence": lambda f: (
        0.35 * f.get("confusion_norm", 0.0)
        + 0.20 * f.get("role_signature_rarity", 0.0)
        + 0.20 * f.get("role_density_norm", 0.0)
        + 0.15 * f.get("multi_event_or_multi_trigger", 0.0)
    ),
    "role_boundary": lambda f: (
        0.30 * f.get("confusion_norm", 0.0)
        + 0.30 * f.get("role_signature_rarity", 0.0)
        + 0.25 * f.get("role_density_norm", 0.0)
        + 0.15 * f.get("multi_event_or_multi_trigger", 0.0)
    ),
    "role_sig_density": lambda f: (
        0.55 * f.get("role_signature_rarity", 0.0)
        + 0.45 * f.get("role_density_norm", 0.0)
    ),
    "conf_role_sig": lambda f: (
        0.60 * f.get("confusion_norm", 0.0)
        + 0.40 * f.get("role_signature_rarity", 0.0)
    ),
}


def write_json(path: Path, payload_obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload_obj, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def summarize_metric(samples, selected_keys):
    rows = []
    for sample in samples:
        use_reason = sample["key"] in selected_keys
        rows.append(sample["reason_metric"] if use_reason else sample["direct_metric"])
    return {
        "trigger_f1": mean([row["trigger_f1"] for row in rows]),
        "argument_f1": mean([row["argument_f1"] for row in rows]),
        "event_f1": mean([row["event_f1"] for row in rows]),
        "score": mean([row["score"] for row in rows]),
        "reason_rate": len(selected_keys) / len(samples) if samples else 0.0,
    }


def direct_summary(samples):
    return summarize_metric(samples, set())


def forced_reason_summary(samples):
    return summarize_metric(samples, {sample["key"] for sample in samples})


def selected_by_selector(samples, selector_name, budget):
    cap = round(len(samples) * budget)
    ranked = sorted(
        samples,
        key=lambda sample: (SELECTORS[selector_name](sample["features"]), sample["key"]),
        reverse=True,
    )
    return {sample["key"] for sample in ranked[:cap]}


def selected_by_oracle(samples, budget):
    cap = round(len(samples) * budget)
    ranked = [
        sample for sample in samples
        if sample["reason_gain"] > 1e-9
    ]
    ranked.sort(key=lambda sample: (sample["reason_gain"], sample["key"]), reverse=True)
    return {sample["key"] for sample in ranked[:cap]}


def route_eval(samples, selected_keys):
    metric_row = summarize_metric(samples, selected_keys)
    direct = direct_summary(samples)
    helpful = {sample["key"] for sample in samples if sample["reason_helpful"]}
    harmful = {sample["key"] for sample in samples if sample["reason_harmful"]}
    selected_helpful = selected_keys & helpful
    selected_harmful = selected_keys & harmful
    selected = [sample for sample in samples if sample["key"] in selected_keys]
    rest = [sample for sample in samples if sample["key"] not in selected_keys]
    metric_row.update(
        {
            "argument_gain_vs_direct": metric_row["argument_f1"] - direct["argument_f1"],
            "event_gain_vs_direct": metric_row["event_f1"] - direct["event_f1"],
            "score_gain_vs_direct": metric_row["score"] - direct["score"],
            "selected_count": len(selected_keys),
            "helpful_count": len(helpful),
            "selected_helpful_count": len(selected_helpful),
            "selected_harmful_count": len(selected_harmful),
            "precision": len(selected_helpful) / len(selected_keys) if selected_keys else 0.0,
            "recall": len(selected_helpful) / len(helpful) if helpful else 0.0,
            "selected_mean_reason_gain": mean([sample["reason_gain"] for sample in selected]),
            "rest_mean_reason_gain": mean([sample["reason_gain"] for sample in rest]),
        }
    )
    return metric_row


def build_samples(root: Path, run, split: str, schema_path: Path):
    direct = load_prediction_map(prediction_path(root, run, "forced_direct", split))
    reason = load_prediction_map(prediction_path(root, run, "forced_reason", split))
    free = load_prediction_map(prediction_path(root, run, "free_route", split))
    features = build_feature_map(Path(DIRECT_EVAL_JSONL[split]), schema_path)
    samples = []
    for key in sorted(set(direct) & set(reason) & set(free)):
        direct_row = direct[key]
        reason_row = reason[key]
        direct_metric = metric(direct_row)
        reason_metric = metric(reason_row)
        reason_gain = reason_metric["score"] - direct_metric["score"]
        samples.append(
            {
                "key": key,
                "direct_row": direct_row,
                "reason_row": reason_row,
                "free_row": free[key],
                "features": features.get(key, {}),
                "direct_metric": direct_metric,
                "reason_metric": reason_metric,
                "reason_gain": reason_gain,
                "argument_gain": reason_metric["argument_f1"] - direct_metric["argument_f1"],
                "event_gain": reason_metric["event_f1"] - direct_metric["event_f1"],
                "reason_helpful": reason_gain > 1e-9,
                "reason_harmful": reason_gain < -1e-9,
            }
        )
    return samples


def safe_compact_events(events_payload):
    events = events_payload.get("events", []) if isinstance(events_payload, dict) else []
    out = []
    if not isinstance(events, list):
        return out
    for event in events:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") or {}
        if not isinstance(trigger, dict):
            trigger = {"text": str(trigger), "start": None, "end": None}
        args = []
        raw_args = event.get("arguments") or []
        if not isinstance(raw_args, list):
            raw_args = []
        for arg in raw_args:
            if not isinstance(arg, dict):
                continue
            args.append(
                {
                    "role": arg.get("role"),
                    "text": arg.get("text"),
                    "span": [arg.get("start"), arg.get("end")],
                }
            )
        out.append(
            {
                "type": event.get("event_type"),
                "trigger": trigger.get("text"),
                "trigger_span": [trigger.get("start"), trigger.get("end")],
                "args": args,
            }
        )
    return out


def analyze_run_split(root: Path, run, split: str, schema_path: Path):
    samples = build_samples(root, run, split, schema_path)
    direct = direct_summary(samples)
    forced_reason = forced_reason_summary(samples)
    oracle = {
        f"cap{int(budget * 100)}": route_eval(samples, selected_by_oracle(samples, budget))
        for budget in BUDGETS
    }
    selectors = {}
    for selector_name in SELECTORS:
        selectors[selector_name] = {
            f"cap{int(budget * 100)}": route_eval(samples, selected_by_selector(samples, selector_name, budget))
            for budget in BUDGETS
        }
    best_by_budget = {}
    for budget in BUDGETS:
        cap_name = f"cap{int(budget * 100)}"
        best_name, best_row = max(
            ((name, rows[cap_name]) for name, rows in selectors.items()),
            key=lambda item: (item[1]["argument_gain_vs_direct"] + item[1]["event_gain_vs_direct"], item[1]["score_gain_vs_direct"]),
        )
        best_by_budget[cap_name] = {"selector": best_name, **best_row}

    hardconf_cap15 = selected_by_selector(samples, "hardconf", 0.15)
    oracle_cap15 = selected_by_oracle(samples, 0.15)
    false_positives = [
        sample for sample in samples
        if sample["key"] in hardconf_cap15 and sample["reason_harmful"]
    ]
    false_negatives = [
        sample for sample in samples
        if sample["key"] not in hardconf_cap15 and sample["reason_helpful"]
    ]
    false_positives.sort(key=lambda sample: sample["reason_gain"])
    false_negatives.sort(key=lambda sample: sample["reason_gain"], reverse=True)

    return {
        "num_examples": len(samples),
        "reason_helpful_count": sum(1 for sample in samples if sample["reason_helpful"]),
        "reason_harmful_count": sum(1 for sample in samples if sample["reason_harmful"]),
        "direct": direct,
        "forced_reason": forced_reason,
        "oracle": oracle,
        "selectors": selectors,
        "best_by_budget": best_by_budget,
        "hardconf_cap15_overlap_with_oracle": {
            "hardconf_selected": len(hardconf_cap15),
            "oracle_selected": len(oracle_cap15),
            "overlap": len(hardconf_cap15 & oracle_cap15),
            "overlap_rate_vs_hardconf": len(hardconf_cap15 & oracle_cap15) / len(hardconf_cap15) if hardconf_cap15 else 0.0,
            "overlap_rate_vs_oracle": len(hardconf_cap15 & oracle_cap15) / len(oracle_cap15) if oracle_cap15 else 0.0,
        },
        "hardconf_cap15_false_positive_cases": [case_payload(run, split, sample) for sample in false_positives[:3]],
        "hardconf_cap15_false_negative_cases": [case_payload(run, split, sample) for sample in false_negatives[:3]],
    }


def case_payload(run, split, sample):
    direct_row = sample["direct_row"]
    reason_row = sample["reason_row"]
    return {
        "branch": run["branch"],
        "selection": run["selection"],
        "split": split,
        "key": sample["key"],
        "reason_gain": sample["reason_gain"],
        "argument_gain": sample["argument_gain"],
        "event_gain": sample["event_gain"],
        "features": sample["features"],
        "text": extract_text_block(direct_row.get("input", "")),
        "gold": safe_compact_events(direct_row.get("gold") or {}),
        "direct_pred": safe_compact_events(payload(direct_row)),
        "reason_pred": safe_compact_events(payload(reason_row)),
        "reason_plan": extract_plan(reason_row),
    }


def analyze(root: Path, schema_path: Path):
    focus_runs = [
        run for run in HARDCONF_FOCUS_RUNS
        if (run["branch"], run["selection"]) in FOCUS_RUN_KEYS
    ]
    runs = []
    for run in focus_runs:
        split_payload = {}
        for split in SPLITS:
            split_payload[split] = analyze_run_split(root, run, split, schema_path)
        runs.append({**run, "splits": split_payload})
    return {
        "definition": "feature-selector routing routes top-k samples by feature to forced_reason and all others to forced_direct; oracle routes top-k positive reason_gain samples.",
        "selectors": sorted(SELECTORS),
        "budgets": BUDGETS,
        "runs": runs,
    }


def fmt(value):
    return f"{value:.4f}"


def markdown_report(payload_obj):
    lines = [
        "# Adaptive Route Feature Selector Analysis",
        "",
        "## Purpose",
        "",
        "- Separate two failure modes: router execution collapse vs selector quality.",
        "- For each selector, route top-k samples to `forced_reason` and all other samples to `forced_direct`.",
        "- Compare feature selectors with the post-hoc oracle that routes top-k positive `reason_gain` samples.",
        "",
        "## Primary Branch: Feature Route vs Oracle",
        "",
        "| branch | selection | split | budget | direct arg/event | oracle gain arg/event P/R | hardconf gain arg/event P/R | best selector gain arg/event P/R |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in payload_obj["runs"]:
        if run["branch"] not in {
            "hardconf10_type_role_hint_plan_lite",
            "hardconf10_calibrated_type_role_hint_plan_lite",
        }:
            continue
        if run["selection"] != "seen_stable_best":
            continue
        for split in SPLITS:
            row = run["splits"][split]
            direct = row["direct"]
            for budget in [0.10, 0.15]:
                cap_name = f"cap{int(budget * 100)}"
                oracle = row["oracle"][cap_name]
                hardconf = row["selectors"]["hardconf"][cap_name]
                best = row["best_by_budget"][cap_name]
                lines.append(
                    "| `{branch}` | `{selection}` | `{split}` | {budget:.0%} | {darg}/{devent} | {ogarg}/{ogevent} {op}/{or_} | {hgarg}/{hgevent} {hp}/{hr} | `{best_name}` {bgarg}/{bgevent} {bp}/{br} |".format(
                        branch=run["branch"],
                        selection=run["selection"],
                        split=split,
                        budget=budget,
                        darg=fmt(direct["argument_f1"]),
                        devent=fmt(direct["event_f1"]),
                        ogarg=fmt(oracle["argument_gain_vs_direct"]),
                        ogevent=fmt(oracle["event_gain_vs_direct"]),
                        op=fmt(oracle["precision"]),
                        or_=fmt(oracle["recall"]),
                        hgarg=fmt(hardconf["argument_gain_vs_direct"]),
                        hgevent=fmt(hardconf["event_gain_vs_direct"]),
                        hp=fmt(hardconf["precision"]),
                        hr=fmt(hardconf["recall"]),
                        best_name=best["selector"],
                        bgarg=fmt(best["argument_gain_vs_direct"]),
                        bgevent=fmt(best["event_gain_vs_direct"]),
                        bp=fmt(best["precision"]),
                        br=fmt(best["recall"]),
                    )
                )

    lines.extend(
        [
            "",
            "## Selector Diagnostics At 15% Budget",
            "",
            "| branch | selection | split | selector | gain arg/event | precision | recall | selected helpful/harmful | selected/rest mean reason_gain |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    key_selectors = [
        "hardconf",
        "hardconf_no_absence",
        "role_boundary",
        "role_signature",
        "role_density",
        "core_absence_low",
        "core_absence_high",
    ]
    for run in payload_obj["runs"]:
        if run["selection"] != "seen_stable_best":
            continue
        for split in SPLITS:
            for selector in key_selectors:
                row = run["splits"][split]["selectors"][selector]["cap15"]
                lines.append(
                    "| `{}` | `{}` | `{}` | `{}` | {:+.4f}/{:+.4f} | {:.3f} | {:.3f} | {}/{} | {:+.4f}/{:+.4f} |".format(
                        run["branch"],
                        run["selection"],
                        split,
                        selector,
                        row["argument_gain_vs_direct"],
                        row["event_gain_vs_direct"],
                        row["precision"],
                        row["recall"],
                        row["selected_helpful_count"],
                        row["selected_harmful_count"],
                        row["selected_mean_reason_gain"],
                        row["rest_mean_reason_gain"],
                    )
                )

    lines.extend(["", "## Hardconf 15% Overlap With Oracle 15%", ""])
    lines.append("| branch | selection | split | hardconf selected | oracle selected | overlap | overlap/hardconf | overlap/oracle |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for run in payload_obj["runs"]:
        if run["selection"] != "seen_stable_best":
            continue
        for split in SPLITS:
            row = run["splits"][split]["hardconf_cap15_overlap_with_oracle"]
            lines.append(
                f"| `{run['branch']}` | `{run['selection']}` | `{split}` | {row['hardconf_selected']} | {row['oracle_selected']} | {row['overlap']} | {fmt(row['overlap_rate_vs_hardconf'])} | {fmt(row['overlap_rate_vs_oracle'])} |"
            )

    primary = next(
        run for run in payload_obj["runs"]
        if run["branch"] == "hardconf10_type_role_hint_plan_lite" and run["selection"] == "seen_stable_best"
    )
    lines.extend(
        [
            "",
            "## Primary Branch Error Cases",
            "",
            "False positive: selected by hardconf15 but forced_reason is worse than forced_direct. False negative: not selected by hardconf15 but forced_reason is better.",
        ]
    )
    for split in SPLITS:
        split_row = primary["splits"][split]
        for label, cases_key in [
            ("False Positive", "hardconf_cap15_false_positive_cases"),
            ("False Negative", "hardconf_cap15_false_negative_cases"),
        ]:
            for idx, case in enumerate(split_row[cases_key], 1):
                features = ", ".join(
                    f"{k}={case['features'].get(k, 0.0):.3f}"
                    for k in ["hardconf_score", "confusion_norm", "role_signature_rarity", "role_density_norm", "core_role_absence_risk"]
                )
                lines.extend(
                    [
                        "",
                        f"### {split} {label} {idx}: `{case['key']}`",
                        "",
                        f"- gain score/arg/event: `{case['reason_gain']:+.4f}` / `{case['argument_gain']:+.4f}` / `{case['event_gain']:+.4f}`",
                        f"- features: {features}",
                        f"- text: {case['text'][:500]}",
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
                    lines.extend(["Plan:", "```text", case["reason_plan"][:700], "```"])

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Current model free-route collapse is one issue, but not the only one: the hardconf heuristic itself has limited precision/recall against reason-helpful samples.",
            "- Feature routing is sometimes negative even when oracle routing is strongly positive, so simply forcing the model to follow hardconf top-k will not be sufficient.",
            "- The best feature differs by split and checkpoint; this supports learning an outcome-aware hardness model from train/dev forced-direct vs forced-reason comparisons.",
            "- `core_role_absence_risk` is not a monotonic positive signal. In several cases lower absence risk or no-absence composites are better than raw hardconf.",
            "- Next optimization should separate route supervision from final extraction supervision: train the route head/token with a balanced outcome-aware set, while preserving direct extraction stability.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_root", default="outputs/stage2_adaptive_runs_user_formal_clean")
    parser.add_argument("--schema_path", default="data/schema/richere-en.event_schema.json")
    parser.add_argument("--output_md", default="reports/2026-05-12_stage2_adaptive_route_feature_selector_analysis.md")
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-12_stage2_adaptive_route_feature_selector_analysis.json")
    args = parser.parse_args()

    result = analyze(Path(args.formal_root), Path(args.schema_path))
    write_json(Path(args.output_json), result)
    write_text(Path(args.output_md), markdown_report(result))
    print(json.dumps({"output_md": args.output_md, "output_json": args.output_json}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
