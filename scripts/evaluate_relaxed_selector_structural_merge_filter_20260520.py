#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.diagnose_sampled_k2_formal_unseen_false_positives_20260519 import (  # noqa: E402
    events_from_row,
    key_for,
)
from scripts.diagnose_sampled_k2_goldfree_harmful_cases_20260519 import (  # noqa: E402
    load_exec_rows,
    pair_features,
)
from scripts.summarize_sampled_confident_router_dev_20260518 import pct, signed, write_json, write_text  # noqa: E402
from scripts.summarize_sampled_k2_structural_proxy_locked_validation_20260519 import (  # noqa: E402
    DEFAULT_FRESH_NLL_ROOT,
    DEFAULT_NEW_NLL_ROOT,
    DEFAULT_OLD_NLL_ROOT,
    DEFAULT_SAMPLE_ROOT,
    build_cases,
)


OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_structural_merge_filter_20260520"
REPORT_MD = REPO / "reports/2026-05-20_stage2_relaxed_selector_structural_merge_filter.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_relaxed_selector_structural_merge_filter.json"
SPLITS = ["test_seen", "test_unseen"]
METRICS = ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json"]
STRATEGIES = [
    "direct_only",
    "reason_full",
    "direct_plus_safe_args",
    "direct_plus_safe_events",
    "direct_plus_safe_events_args",
    "reason_filtered",
]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def normalize_events(events_payload):
    events = events_payload.get("events", []) if isinstance(events_payload, dict) else []
    trigger_set = set()
    argument_set = set()
    event_set = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("event_type")
        trigger = event.get("trigger", {})
        if not isinstance(trigger, dict):
            trigger = {}
        trig = (etype, trigger.get("start"), trigger.get("end"))
        trigger_set.add(trig)
        args = []
        raw_args = event.get("arguments", [])
        if not isinstance(raw_args, list):
            raw_args = []
        for arg in raw_args:
            if not isinstance(arg, dict):
                continue
            argument_set.add(
                (
                    etype,
                    trigger.get("start"),
                    trigger.get("end"),
                    arg.get("role"),
                    arg.get("start"),
                    arg.get("end"),
                )
            )
            args.append((arg.get("role"), arg.get("start"), arg.get("end")))
        sorted_args = tuple(
            sorted(
                args,
                key=lambda item: (
                    item[0] or "",
                    -1 if item[1] is None else item[1],
                    -1 if item[2] is None else item[2],
                ),
            )
        )
        event_set.add((etype, trigger.get("start"), trigger.get("end"), sorted_args))
    return trigger_set, argument_set, event_set


def prf(pred_set, gold_set):
    if not pred_set and not gold_set:
        return {"p": 1.0, "r": 1.0, "f1": 1.0}
    if not pred_set or not gold_set:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"p": precision, "r": recall, "f1": f1}


def score(metrics):
    return metrics["argument_f1"] + metrics["event_f1"] + 0.25 * metrics["trigger_f1"]


def avg_metrics(rows):
    items = list(rows)
    return {
        metric: mean(row[metric] for row in items)
        for metric in METRICS
    }


def relaxed_selector(case):
    avg_margin = mean([case["fresh_margin"], case["old17_18_margin"], case["new19_20_margin"]])
    return (
        case["fresh_margin"] >= 0.25
        and case["margin_range"] <= 0.75
        and case["num_margins_ge_0p25"] >= 1
        and case["sample_arg_text_jaccard_mean"] >= 0.40
        and case["sample_event_count_delta_mean"] <= 0.0
        and avg_margin >= 0.0
    )


def event_list(row):
    return deepcopy(events_from_row(row))


def event_type(event):
    return str(event.get("event_type") or "") if isinstance(event, dict) else ""


def trigger_span(event):
    trigger = event.get("trigger") if isinstance(event, dict) else {}
    if not isinstance(trigger, dict):
        return None
    start = trigger.get("start")
    end = trigger.get("end")
    return (start, end) if start is not None and end is not None else None


def span_overlap(left, right):
    if left is None or right is None:
        return False
    left_start, left_end = left
    right_start, right_end = right
    return max(left_start, right_start) < min(left_end, right_end)


def event_key(event):
    return (event_type(event), trigger_span(event))


def args_of(event):
    args = event.get("arguments") if isinstance(event, dict) else []
    return args if isinstance(args, list) else []


def arg_key(arg):
    return (arg.get("role"), arg.get("start"), arg.get("end"))


def arg_span(arg):
    start = arg.get("start")
    end = arg.get("end")
    return (start, end) if start is not None and end is not None else None


def arg_text(arg):
    return str(arg.get("text") or "").strip().lower()


def parse_allowed_roles(input_text):
    allowed = defaultdict(set)
    current = None
    for line in input_text.splitlines():
        match = re.match(r"\[\d+\]\s+Event type:\s+(.+)", line.strip())
        if match:
            current = match.group(1).strip()
            continue
        if current and line.strip().startswith("Core roles:"):
            roles = line.split(":", 1)[1]
            for role in roles.split(","):
                role = role.strip()
                if role:
                    allowed[current].add(role)
    return allowed


def load_source_inputs():
    out = {}
    for split in SPLITS:
        path = REPO / f"data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_{split}_pos.jsonl"
        out[split] = {key_for(row): row for row in load_jsonl(path)}
    return out


def output_payload(events):
    return {"events": events}


def metric_from_payload(payload, gold_payload):
    pred_trig, pred_arg, pred_event = normalize_events(payload)
    gold_trig, gold_arg, gold_event = normalize_events(gold_payload)
    trigger = prf(pred_trig, gold_trig)["f1"]
    argument = prf(pred_arg, gold_arg)["f1"]
    event = prf(pred_event, gold_event)["f1"]
    return {
        "argument_f1": argument,
        "event_f1": event,
        "trigger_f1": trigger,
        "score": argument + event + 0.25 * trigger,
        "valid_json": 1.0,
    }


def safe_arg(reason_arg, direct_event, allowed_roles):
    role = reason_arg.get("role")
    if allowed_roles and role not in allowed_roles:
        return False
    if arg_span(reason_arg) is None:
        return False
    if not arg_text(reason_arg):
        return False
    existing = {arg_key(arg) for arg in args_of(direct_event)}
    if arg_key(reason_arg) in existing:
        return False
    for direct_arg in args_of(direct_event):
        if direct_arg.get("role") == role and span_overlap(arg_span(reason_arg), arg_span(direct_arg)):
            return False
    return True


def add_safe_args(direct_event, reason_event, allowed_roles, max_new_args=1):
    merged = deepcopy(direct_event)
    args = list(args_of(merged))
    added = 0
    for reason_arg in args_of(reason_event):
        if added >= max_new_args:
            break
        if safe_arg(reason_arg, merged, allowed_roles):
            args.append(deepcopy(reason_arg))
            added += 1
    merged["arguments"] = args
    return merged, added


def matching_reason_event(direct_event, reason_events):
    d_type = event_type(direct_event)
    d_span = trigger_span(direct_event)
    for reason_event in reason_events:
        if event_type(reason_event) != d_type:
            continue
        if trigger_span(reason_event) == d_span or span_overlap(trigger_span(reason_event), d_span):
            return reason_event
    return None


def safe_new_event(reason_event, direct_events, allowed_roles, pair_feat):
    direct_types = {event_type(event) for event in direct_events}
    if event_type(reason_event) not in direct_types:
        return False
    if trigger_span(reason_event) is None:
        return False
    if pair_feat["event_count_delta"] > 1:
        return False
    args = args_of(reason_event)
    if len(args) > 2:
        return False
    roles = allowed_roles.get(event_type(reason_event), set())
    for arg in args:
        if roles and arg.get("role") not in roles:
            return False
        if arg_span(arg) is None:
            return False
    return True


def global_safe(case, pair_feat):
    return (
        case["sample_arg_text_jaccard_mean"] >= 0.40
        and case["sample_event_count_delta_mean"] <= 0.0
        and pair_feat["event_count_delta"] <= 1
        and pair_feat["reason_new_event_type_count"] <= 0
    )


def merge_events(strategy, direct_row, reason_row, source_row, case):
    direct_events = event_list(direct_row)
    reason_events = event_list(reason_row)
    pair_feat = pair_features(direct_row, reason_row)
    allowed = parse_allowed_roles(source_row["input"])
    if strategy == "direct_only":
        return direct_events, {"accepted_args": 0, "accepted_events": 0, "global_safe": True}
    if strategy == "reason_full":
        return reason_events, {"accepted_args": 0, "accepted_events": 0, "global_safe": True}
    if not global_safe(case, pair_feat):
        return direct_events, {"accepted_args": 0, "accepted_events": 0, "global_safe": False}
    if strategy == "reason_filtered":
        filtered = []
        direct_keys = {event_key(event) for event in direct_events}
        direct_types = {event_type(event) for event in direct_events}
        for event in reason_events:
            if event_key(event) in direct_keys or event_type(event) in direct_types:
                filtered.append(deepcopy(event))
        return filtered if filtered else direct_events, {
            "accepted_args": 0,
            "accepted_events": max(0, len(filtered) - len(direct_events)),
            "global_safe": True,
        }

    merged = []
    accepted_args = 0
    accepted_events = 0
    for direct_event in direct_events:
        if strategy in {"direct_plus_safe_args", "direct_plus_safe_events_args"}:
            reason_event = matching_reason_event(direct_event, reason_events)
            if reason_event is not None:
                new_event, added = add_safe_args(direct_event, reason_event, allowed.get(event_type(direct_event), set()))
                accepted_args += added
                merged.append(new_event)
                continue
        merged.append(deepcopy(direct_event))
    if strategy in {"direct_plus_safe_events", "direct_plus_safe_events_args"}:
        merged_keys = {event_key(event) for event in merged}
        for reason_event in reason_events:
            if event_key(reason_event) in merged_keys:
                continue
            if len(merged) >= len(direct_events) + 1:
                break
            if safe_new_event(reason_event, direct_events, allowed, pair_feat):
                merged.append(deepcopy(reason_event))
                merged_keys.add(event_key(reason_event))
                accepted_events += 1
    return merged, {"accepted_args": accepted_args, "accepted_events": accepted_events, "global_safe": True}


def summarize_strategy(strategy, cases_by_split, direct_rows, reason_rows, source_rows, output_root):
    prediction_rows = []
    eval_rows = []
    selected_gains = []
    original_gains = []
    selected_details = []
    for split in SPLITS:
        for case in cases_by_split[split]:
            key = case["key"]
            direct_row = direct_rows[split][key]
            reason_row = reason_rows[split][key]
            source_row = source_rows[split][key]
            gold = direct_row.get("gold") or {}
            if relaxed_selector(case):
                events, merge_info = merge_events(strategy, direct_row, reason_row, source_row, case)
                routed_payload = output_payload(events)
                selected = True
            else:
                routed_payload = output_payload(event_list(direct_row))
                merge_info = {"accepted_args": 0, "accepted_events": 0, "global_safe": None}
                selected = False
            direct_metrics = case["single_gen_execution_direct"]
            reason_metrics = case["single_gen_execution_reason"]
            routed_metrics = metric_from_payload(routed_payload, gold)
            eval_rows.append(
                {
                    "split": split,
                    "case_id": case["case_id"],
                    "key": key,
                    "selected": selected,
                    "direct": direct_metrics,
                    "original_reason": reason_metrics,
                    "routed": routed_metrics,
                    "merge_info": merge_info,
                }
            )
            prediction_rows.append(
                {
                    "split": split,
                    "key": key,
                    "case_id": case["case_id"],
                    "selected": selected,
                    "strategy": strategy,
                    "predicted": routed_payload,
                    "gold": gold,
                    "metrics": routed_metrics,
                    "merge_info": merge_info,
                }
            )
            if selected:
                gain = routed_metrics["score"] - direct_metrics["score"]
                selected_gains.append(gain)
                original_gains.append(reason_metrics["score"] - direct_metrics["score"])
                selected_details.append({**eval_rows[-1], "gain": gain})
    write_jsonl(output_root / f"{strategy}_predictions.jsonl", prediction_rows)
    return eval_rows, selected_gains, original_gains, selected_details


def consolidate(strategy, eval_rows, selected_gains, original_gains, selected_details):
    rows = []
    for split in ["test", "test_seen", "test_unseen"]:
        split_rows = eval_rows if split == "test" else [row for row in eval_rows if row["split"] == split]
        selected = selected_details if split == "test" else [row for row in selected_details if row["split"] == split]
        direct_avg = avg_metrics(row["direct"] for row in split_rows)
        routed_avg = avg_metrics(row["routed"] for row in split_rows)
        gains = [row["gain"] for row in selected]
        original = [row["original_reason"]["score"] - row["direct"]["score"] for row in selected]
        buckets = Counter("helpful" if gain > 0 else "harmful" if gain < 0 else "neutral" for gain in gains)
        rows.append(
            {
                "strategy": strategy,
                "split": split,
                "num_examples": len(split_rows),
                "pred_reason_count": len(selected),
                "pred_reason_rate": len(selected) / len(split_rows) if split_rows else 0.0,
                "selected_helpful": buckets["helpful"],
                "selected_harmful": buckets["harmful"],
                "selected_neutral": buckets["neutral"],
                "selected_harm_rate": buckets["harmful"] / len(selected) if selected else 0.0,
                "selected_gain_mean": mean(gains),
                "selected_original_reason_gain_mean": mean(original),
                "accepted_args_mean": mean(row["merge_info"]["accepted_args"] for row in selected),
                "accepted_events_mean": mean(row["merge_info"]["accepted_events"] for row in selected),
                "routed": routed_avg,
                "routed_minus_direct": {
                    metric: routed_avg[metric] - direct_avg[metric]
                    for metric in METRICS
                },
            }
        )
    test = next(row for row in rows if row["split"] == "test")
    seen = next(row for row in rows if row["split"] == "test_seen")
    unseen = next(row for row in rows if row["split"] == "test_unseen")
    return {
        "strategy": strategy,
        "test_score_delta": test["routed_minus_direct"]["score"],
        "seen_score_delta": seen["routed_minus_direct"]["score"],
        "unseen_score_delta": unseen["routed_minus_direct"]["score"],
        "test_reason_rate": test["pred_reason_rate"],
        "seen_reason_rate": seen["pred_reason_rate"],
        "unseen_reason_rate": unseen["pred_reason_rate"],
        "test_harm_rate": test["selected_harm_rate"],
        "seen_harm_rate": seen["selected_harm_rate"],
        "unseen_harm_rate": unseen["selected_harm_rate"],
        "test_selected_gain_mean": test["selected_gain_mean"],
        "test_original_reason_gain_mean": test["selected_original_reason_gain_mean"],
        "test_accepted_args_mean": test["accepted_args_mean"],
        "test_accepted_events_mean": test["accepted_events_mean"],
        "passes_target": (
            test["routed_minus_direct"]["score"] > 0.0085
            and seen["routed_minus_direct"]["score"] >= 0.0042
            and test["selected_harm_rate"] <= 0.16
            and seen["selected_harm_rate"] <= 0.20
            and unseen["routed_minus_direct"]["score"] >= 0.0200
        ),
        "rows": rows,
    }


def render_leaderboard(rows):
    lines = [
        "| strategy | pass | reason test/seen/unseen | score test/seen/unseen | harm test/seen/unseen | selected gain | orig gain | accepted args/events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['strategy']}` | `{row['passes_target']}` | "
            f"{pct(row['test_reason_rate'])}/{pct(row['seen_reason_rate'])}/{pct(row['unseen_reason_rate'])} | "
            f"{signed(row['test_score_delta'])}/{signed(row['seen_score_delta'])}/{signed(row['unseen_score_delta'])} | "
            f"{pct(row['test_harm_rate'])}/{pct(row['seen_harm_rate'])}/{pct(row['unseen_harm_rate'])} | "
            f"{signed(row['test_selected_gain_mean'])} | {signed(row['test_original_reason_gain_mean'])} | "
            f"{row['test_accepted_args_mean']:.2f}/{row['test_accepted_events_mean']:.2f} |"
        )
    return "\n".join(lines)


def render_report(payload):
    best = payload["leaderboard"][0]
    lines = [
        "# Relaxed Selector Structural Merge Filter",
        "",
        "This evaluates offline structural merge/filter policies for the top relaxed no-training selector.",
        "",
        "Selector:",
        "",
        "```text",
        "fresh_margin >= 0.25",
        "margin_range <= 0.75",
        "num_margins_ge_0p25 >= 1",
        "sample_arg_text_jaccard_mean >= 0.40",
        "sample_event_count_delta_mean <= 0.0",
        "avg_margin >= 0.0",
        "```",
        "",
        "## Leaderboard",
        "",
        render_leaderboard(payload["leaderboard"]),
        "",
        "## Reading",
        "",
        f"- Best strategy by target screen: `{best['strategy']}` with score delta `{best['test_score_delta']:+.4f}/{best['seen_score_delta']:+.4f}/{best['unseen_score_delta']:+.4f}` and harm `{best['test_harm_rate']:.1%}/{best['seen_harm_rate']:.1%}/{best['unseen_harm_rate']:.1%}`.",
        "- `reason_full` should reproduce the relaxed selector baseline; `direct_only` should reproduce direct baseline under the same selection mask.",
        "- Merge filters are useful only if they retain meaningful score gain while reducing selected harm.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['report_json']}`",
        f"- output root: `{payload['output_root']}`",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    build_args = argparse.Namespace(
        seeds=args.seeds,
        sample_root=args.sample_root,
        fresh_nll_root=args.fresh_nll_root,
        old_nll_root=args.old_nll_root,
        new_nll_root=args.new_nll_root,
        checkpoint=args.checkpoint,
    )
    cases_by_split = build_cases(build_args)
    direct_rows = {split: load_exec_rows(split, "direct") for split in SPLITS}
    reason_rows = {split: load_exec_rows(split, "reason") for split in SPLITS}
    source_rows = load_source_inputs()
    args.output_root.mkdir(parents=True, exist_ok=True)

    consolidated = []
    details = {}
    for strategy in STRATEGIES:
        eval_rows, gains, original_gains, selected_details = summarize_strategy(
            strategy,
            cases_by_split,
            direct_rows,
            reason_rows,
            source_rows,
            args.output_root,
        )
        summary = consolidate(strategy, eval_rows, gains, original_gains, selected_details)
        consolidated.append(summary)
        details[strategy] = {
            "rows": summary.pop("rows"),
            "selected_case_count": len(selected_details),
        }
    consolidated.sort(
        key=lambda row: (
            row["passes_target"],
            row["test_score_delta"],
            -row["test_harm_rate"],
            row["seen_score_delta"],
        ),
        reverse=True,
    )
    payload = {
        "checkpoint": args.checkpoint,
        "seeds": args.seeds,
        "output_root": args.output_root.as_posix(),
        "strategies": STRATEGIES,
        "leaderboard": consolidated,
        "details": details,
        "report_md": args.report_md.as_posix(),
        "report_json": args.report_json.as_posix(),
    }
    write_json(args.report_json, payload)
    write_json(args.output_root / "summary.json", payload)
    write_text(args.report_md, render_report(payload))
    print(json.dumps({"report_md": args.report_md.as_posix(), "report_json": args.report_json.as_posix()}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 24])
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--fresh-nll-root", type=Path, default=DEFAULT_FRESH_NLL_ROOT)
    parser.add_argument("--old-nll-root", type=Path, default=DEFAULT_OLD_NLL_ROOT)
    parser.add_argument("--new-nll-root", type=Path, default=DEFAULT_NEW_NLL_ROOT)
    parser.add_argument("--checkpoint", default="checkpoint-50")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
