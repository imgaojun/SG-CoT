#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


METRICS = ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json_rate"]
SPLITS = ["test_seen", "test_unseen"]
ROUTES = ["direct", "reason"]
SOURCE_SPECS = {
    "k2_expected_17_18": {"type": "sampled", "seeds": [17, 18]},
    "k2_expected_19_20": {"type": "sampled", "seeds": [19, 20]},
    "k2_expected_17_20": {"type": "sampled", "seeds": [17, 18, 19, 20]},
    "single_gen_execution": {"type": "execution"},
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def key_for(row):
    meta = row.get("meta") or {}
    return row.get("sample_key") or meta.get("wnd_id") or row.get("wnd_id")


def case_id(case):
    return f"{case['split']}::{case['key']}"


def metric_score(row):
    return row.get("argument_f1", 0.0) + row.get("event_f1", 0.0) + 0.25 * row.get("trigger_f1", 0.0)


def metric_dict(row):
    return {
        "argument_f1": row.get("argument_f1", 0.0),
        "event_f1": row.get("event_f1", 0.0),
        "trigger_f1": row.get("trigger_f1", 0.0),
        "score": metric_score(row),
        "valid_json_rate": 1.0 if row.get("valid_final_json", row.get("valid_json")) else 0.0,
    }


def avg_metric_dict(rows):
    if not rows:
        return {metric: 0.0 for metric in METRICS}
    return {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}


def load_sampled_metrics(sample_root: Path, split: str, route: str, seeds):
    grouped = defaultdict(list)
    for seed in seeds:
        path = sample_root / split / route / f"seed-{seed}" / "predictions.jsonl"
        for row in load_jsonl(path):
            key = key_for(row)
            if key:
                grouped[key].append(metric_dict(row))
    out = {}
    for key, rows in grouped.items():
        if len(rows) != len(seeds):
            raise ValueError(f"{split}/{route}/{key}: expected {len(seeds)} rows, got {len(rows)}")
        out[key] = avg_metric_dict(rows)
        out[key]["sample_count"] = len(rows)
    return out


def load_execution_metrics(execution_root: Path, split: str, route: str):
    path = execution_root / f"forced_{route}" / split / "predictions.jsonl"
    out = {}
    for row in load_jsonl(path):
        key = key_for(row)
        if key:
            out[key] = metric_dict(row)
    return out


def load_margins(root: Path, split: str, checkpoint: str = "checkpoint-50"):
    path = root / checkpoint / split / "scores.jsonl"
    out = {}
    for row in load_jsonl(path):
        key = key_for(row)
        if key:
            out[key] = row.get("delta_direct_minus_reason_route_nll")
    return out


def build_cases(config):
    sample_root = repo_path(config["sample_root"])
    execution_root = repo_path(config["execution_root"])
    old_root = repo_path(config["route_nll_roots"]["seedpair17_18"])
    new_root = repo_path(config["route_nll_roots"]["seedpair19_20"])
    cases = {}
    for split in SPLITS:
        source_metrics = {}
        for source, spec in SOURCE_SPECS.items():
            source_metrics[source] = {}
            if spec["type"] == "sampled":
                for route in ROUTES:
                    source_metrics[source][route] = load_sampled_metrics(
                        sample_root, split, route, spec["seeds"]
                    )
            else:
                for route in ROUTES:
                    source_metrics[source][route] = load_execution_metrics(
                        execution_root, split, route
                    )

        old_margins = load_margins(old_root, split)
        new_margins = load_margins(new_root, split)
        keys = set(old_margins) & set(new_margins)
        for source in SOURCE_SPECS:
            keys &= set(source_metrics[source]["direct"])
            keys &= set(source_metrics[source]["reason"])
        split_cases = []
        for key in sorted(keys):
            case = {
                "split": split,
                "key": key,
                "old17_18_margin": old_margins[key],
                "new19_20_margin": new_margins[key],
            }
            case["avg_margin"] = (case["old17_18_margin"] + case["new19_20_margin"]) / 2
            case["min_margin"] = min(case["old17_18_margin"], case["new19_20_margin"])
            case["max_margin"] = max(case["old17_18_margin"], case["new19_20_margin"])
            for source in SOURCE_SPECS:
                for route in ROUTES:
                    case[f"{source}_{route}"] = source_metrics[source][route][key]
            split_cases.append(case)
        if not split_cases:
            raise ValueError(f"no common cases for {split}")
        cases[split] = split_cases
    return cases


def route_reason(case, policy_id):
    old = case["old17_18_margin"]
    new = case["new19_20_margin"]
    avg = case["avg_margin"]
    if policy_id == "old17_18_main":
        return old >= 0.25
    if policy_id == "new19_20_main":
        return new >= 0.25
    if policy_id == "both_seedpairs_main":
        return old >= 0.25 and new >= 0.25
    if policy_id == "avg_margin_main":
        return avg >= 0.25
    if policy_id == "avg_margin_both_positive":
        return avg >= 0.25 and old > 0.0 and new > 0.0
    if policy_id == "either_seedpair_main":
        return old >= 0.25 or new >= 0.25
    raise KeyError(policy_id)


def route_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def avg_metrics(items):
    if not items:
        return {metric: 0.0 for metric in METRICS}
    return {metric: sum(item[metric] for item in items) / len(items) for metric in METRICS}


def summarize_cases(cases, source, policy_id):
    selected = []
    selected_set = set()
    helpful = set()
    direct_rows = []
    reason_rows = []
    routed_rows = []
    for case in cases:
        cid = case_id(case)
        direct = case[f"{source}_direct"]
        reason = case[f"{source}_reason"]
        direct_rows.append(direct)
        reason_rows.append(reason)
        gain = reason["score"] - direct["score"]
        if gain > 0:
            helpful.add(cid)
        if route_reason(case, policy_id):
            selected_set.add(cid)
            selected.append(
                {
                    "case_id": cid,
                    "split": case["split"],
                    "key": case["key"],
                    "score_gain": gain,
                    "argument_gain": reason["argument_f1"] - direct["argument_f1"],
                    "event_gain": reason["event_f1"] - direct["event_f1"],
                    "trigger_gain": reason["trigger_f1"] - direct["trigger_f1"],
                    "old17_18_margin": case["old17_18_margin"],
                    "new19_20_margin": case["new19_20_margin"],
                    "avg_margin": case["avg_margin"],
                    "min_margin": case["min_margin"],
                    "max_margin": case["max_margin"],
                }
            )
            routed_rows.append(reason)
        else:
            routed_rows.append(direct)

    n = len(cases)
    direct_summary = avg_metrics(direct_rows)
    reason_summary = avg_metrics(reason_rows)
    routed_summary = avg_metrics(routed_rows)
    tp = len(selected_set & helpful)
    fp = len(selected_set - helpful)
    fn = len(helpful - selected_set)
    harm_count = sum(1 for row in selected if row["score_gain"] < 0)
    return {
        "source": source,
        "policy": policy_id,
        "split": cases[0]["split"] if cases else "unknown",
        "num_examples": n,
        "pred_reason_count": len(selected_set),
        "pred_reason_rate": len(selected_set) / n if n else 0.0,
        "reason_helpful_count": len(helpful),
        "reason_helpful_rate": len(helpful) / n if n else 0.0,
        "route_vs_helpful": route_prf(tp, fp, fn),
        "selected_reason_score_gain_mean": (
            sum(row["score_gain"] for row in selected) / len(selected) if selected else 0.0
        ),
        "selected_reason_harm_count": harm_count,
        "selected_reason_harm_rate": harm_count / len(selected) if selected else 0.0,
        "direct": direct_summary,
        "reason_all": reason_summary,
        "routed": routed_summary,
        "routed_minus_direct": {
            metric: routed_summary[metric] - direct_summary[metric] for metric in METRICS
        },
        "routed_minus_reason_all": {
            metric: routed_summary[metric] - reason_summary[metric] for metric in METRICS
        },
        "selected_reason_best": sorted(selected, key=lambda row: row["score_gain"], reverse=True)[:10],
        "selected_reason_worst": sorted(selected, key=lambda row: row["score_gain"])[:10],
    }


def selected_ids(cases, policy_id):
    return {case_id(case) for case in cases if route_reason(case, policy_id)}


def summarize_policy_overlap(cases, split):
    policies = [
        "old17_18_main",
        "new19_20_main",
        "both_seedpairs_main",
        "avg_margin_main",
        "avg_margin_both_positive",
        "either_seedpair_main",
    ]
    sets = {policy: selected_ids(cases, policy) for policy in policies}
    rows = []
    for left in policies:
        for right in policies:
            if left >= right:
                continue
            inter = sets[left] & sets[right]
            union = sets[left] | sets[right]
            rows.append(
                {
                    "split": split,
                    "left": left,
                    "right": right,
                    "left_count": len(sets[left]),
                    "right_count": len(sets[right]),
                    "intersection_count": len(inter),
                    "jaccard": len(inter) / len(union) if union else 1.0,
                }
            )
    return rows


def summarize_margin_buckets(cases, split, source):
    buckets = [
        ("both_ge_0p25", lambda case: case["old17_18_margin"] >= 0.25 and case["new19_20_margin"] >= 0.25),
        ("old_only_ge_0p25", lambda case: case["old17_18_margin"] >= 0.25 and case["new19_20_margin"] < 0.25),
        ("new_only_ge_0p25", lambda case: case["old17_18_margin"] < 0.25 and case["new19_20_margin"] >= 0.25),
        ("neither_ge_0p25", lambda case: case["old17_18_margin"] < 0.25 and case["new19_20_margin"] < 0.25),
        ("avg_ge_0p25", lambda case: case["avg_margin"] >= 0.25),
        ("avg_ge_0p25_both_positive", lambda case: case["avg_margin"] >= 0.25 and case["old17_18_margin"] > 0 and case["new19_20_margin"] > 0),
    ]
    rows = []
    for name, fn in buckets:
        selected = [case for case in cases if fn(case)]
        gains = [
            case[f"{source}_reason"]["score"] - case[f"{source}_direct"]["score"]
            for case in selected
        ]
        rows.append(
            {
                "bucket": name,
                "source": source,
                "split": split,
                "count": len(selected),
                "rate": len(selected) / len(cases) if cases else 0.0,
                "score_gain_mean": sum(gains) / len(gains) if gains else 0.0,
                "harm_rate": sum(1 for gain in gains if gain < 0) / len(gains) if gains else 0.0,
            }
        )
    return rows


def metric_cell(row):
    return f"{fmt(row['argument_f1'])}/{fmt(row['event_f1'])}/{fmt(row['trigger_f1'])}/{fmt(row['score'])}"


def delta_cell(row):
    return f"{signed(row['argument_f1'])}/{signed(row['event_f1'])}/{signed(row['trigger_f1'])}/{signed(row['score'])}"


def split_order(split):
    return {"test": 0, "test_seen": 1, "test_unseen": 2}.get(split, 99)


def policy_order(policy):
    order = {
        "old17_18_main": 0,
        "new19_20_main": 1,
        "both_seedpairs_main": 2,
        "avg_margin_main": 3,
        "avg_margin_both_positive": 4,
        "either_seedpair_main": 5,
    }
    return order.get(policy, 99)


def render_result_table(rows, source):
    lines = [
        "| split | policy | reason rate | routed A/E/T/Score | delta vs Direct A/E/T/Score | selected gain | harm rate | P/R/F1 vs helpful |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    selected_rows = [row for row in rows if row["source"] == source]
    selected_rows.sort(key=lambda row: (split_order(row["split"]), policy_order(row["policy"])))
    for row in selected_rows:
        prf = row["route_vs_helpful"]
        lines.append(
            f"| `{row['split']}` | `{row['policy']}` | {pct(row['pred_reason_rate'])} | "
            f"{metric_cell(row['routed'])} | {delta_cell(row['routed_minus_direct'])} | "
            f"{signed(row['selected_reason_score_gain_mean'])} | {pct(row['selected_reason_harm_rate'])} | "
            f"{fmt(prf['precision'], 3)}/{fmt(prf['recall'], 3)}/{fmt(prf['f1'], 3)} |"
        )
    return "\n".join(lines)


def render_baseline_table(rows):
    lines = [
        "| source | split | Direct A/E/T/Score | Reason-all A/E/T/Score | Reason-all delta Score |",
        "|---|---|---:|---:|---:|",
    ]
    seen = set()
    for row in sorted(rows, key=lambda item: (item["source"], split_order(item["split"]))):
        key = (row["source"], row["split"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| `{row['source']}` | `{row['split']}` | {metric_cell(row['direct'])} | "
            f"{metric_cell(row['reason_all'])} | {signed(row['reason_all']['score'] - row['direct']['score'])} |"
        )
    return "\n".join(lines)


def render_bucket_table(rows, source):
    lines = [
        "| split | bucket | count | rate | score gain mean | harm rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(
        [item for item in rows if item["source"] == source],
        key=lambda item: (split_order(item["split"]), item["bucket"]),
    ):
        lines.append(
            f"| `{row['split']}` | `{row['bucket']}` | {row['count']} | {pct(row['rate'])} | "
            f"{signed(row['score_gain_mean'])} | {pct(row['harm_rate'])} |"
        )
    return "\n".join(lines)


def render_reading(payload):
    rows = payload["results"]
    lines = []
    for source in ["k2_expected_17_20", "single_gen_execution"]:
        lines.append(f"- `{source}` aggregated `test`:")
        for policy in ["old17_18_main", "new19_20_main", "both_seedpairs_main", "avg_margin_main", "avg_margin_both_positive"]:
            row = next(item for item in rows if item["source"] == source and item["split"] == "test" and item["policy"] == policy)
            delta = row["routed_minus_direct"]
            lines.append(
                f"  - `{policy}`: reason `{row['pred_reason_rate']:.1%}`, score delta `{delta['score']:+.4f}`, "
                f"A/E/T `{delta['argument_f1']:+.4f}/{delta['event_f1']:+.4f}/{delta['trigger_f1']:+.4f}`, "
                f"selected gain `{row['selected_reason_score_gain_mean']:+.4f}`."
            )
    return "\n".join(lines)


def render_report(payload):
    lines = [
        "# Sampled K2 Formal Seedpair Consensus",
        "",
        "This offline analysis combines the existing formal K2 route-NLL scores from seedpairs `17/18` and `19/20`. It does not generate new model outputs.",
        "",
        f"- config: `{payload['config']}`",
        f"- output root: `{payload['output_root']}`",
        "",
        "## Baselines",
        "",
        render_baseline_table(payload["results"]),
        "",
        "## K2 Expected Utility over Seeds 17-20",
        "",
        render_result_table(payload["results"], "k2_expected_17_20"),
        "",
        "## Reused Single-Generation Execution",
        "",
        render_result_table(payload["results"], "single_gen_execution"),
        "",
        "## Margin Buckets under K2 Expected 17-20",
        "",
        render_bucket_table(payload["margin_buckets"], "k2_expected_17_20"),
        "",
        "## Reading",
        "",
        render_reading(payload),
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['report_json']}`",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    config_path = repo_path(args.config)
    config = load_json(config_path)
    cases_by_split = build_cases(config)
    all_cases = [case for split in SPLITS for case in cases_by_split[split]]
    case_sets = {"test": all_cases, **cases_by_split}
    policies = [policy["id"] for policy in config["policies"]]

    results = []
    overlap = []
    margin_buckets = []
    for split, cases in case_sets.items():
        for policy in policies:
            for source in SOURCE_SPECS:
                row = summarize_cases(cases, source, policy)
                row["split"] = split
                results.append(row)
        overlap.extend(summarize_policy_overlap(cases, split))
        for source in SOURCE_SPECS:
            margin_buckets.extend(summarize_margin_buckets(cases, split, source))

    output_root = repo_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    report_md = repo_path(config["reports"]["markdown"])
    report_json = repo_path(config["reports"]["json"])
    payload = {
        "config": config_path.as_posix(),
        "output_root": output_root.as_posix(),
        "splits": {split: len(cases) for split, cases in cases_by_split.items()},
        "policies": config["policies"],
        "sources": SOURCE_SPECS,
        "results": results,
        "overlap": overlap,
        "margin_buckets": margin_buckets,
        "report_md": report_md.as_posix(),
        "report_json": report_json.as_posix(),
    }
    write_json(report_json, payload)
    write_json(output_root / "summary.json", payload)
    write_text(report_md, render_report(payload))
    print(json.dumps({"report_md": report_md.as_posix(), "report_json": report_json.as_posix()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
