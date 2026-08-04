#!/usr/bin/env python3
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


SAMPLE_ROOT = (
    REPO
    / "outputs/stage2_modular_dualexpert/formal_k2_counterfactual_utility_20260518"
    / "sampled_reason_expert_forcedreason_from_noaux_20260517_checkpoint-258"
)
NLL_ROOT = (
    REPO
    / "outputs/stage2_adaptive_route_formal_nll_20260518"
    / "sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
)
REPORT_MD = REPO / "reports/2026-05-18_stage2_sampled_k2_formal_route_decision_diagnosis.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-18_stage2_sampled_k2_formal_route_decision_diagnosis.json"
SPLITS = ["test_seen", "test_unseen"]
SEEDS = [17, 18]
MAIN = ("checkpoint-50", "margin_ge_0p25", 0.25)
COMPARATOR = ("checkpoint-75", "margin_ge_0p05", 0.05)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def key_for(row):
    meta = row.get("meta") or {}
    return row.get("sample_key") or meta.get("wnd_id") or row.get("wnd_id")


def score_value(row):
    return row.get("argument_f1", 0.0) + row.get("event_f1", 0.0) + 0.25 * row.get("trigger_f1", 0.0)


def mean(xs):
    vals = list(xs)
    return sum(vals) / len(vals) if vals else 0.0


def summarize_numbers(values):
    vals = sorted(values)
    if not vals:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(vals),
        "mean": mean(vals),
        "median": statistics.median(vals),
        "min": vals[0],
        "max": vals[-1],
    }


def load_route_metrics(split, route):
    grouped = defaultdict(list)
    for seed in SEEDS:
        path = SAMPLE_ROOT / split / route / f"seed-{seed}" / "predictions.jsonl"
        for row in load_jsonl(path):
            grouped[key_for(row)].append(row)
    out = {}
    for key, rows in grouped.items():
        out[key] = {
            "trigger_f1": mean(row.get("trigger_f1", 0.0) for row in rows),
            "argument_f1": mean(row.get("argument_f1", 0.0) for row in rows),
            "event_f1": mean(row.get("event_f1", 0.0) for row in rows),
            "score": mean(score_value(row) for row in rows),
            "sample_count": len(rows),
        }
    return out


def load_scores(checkpoint, split):
    path = NLL_ROOT / checkpoint / split / "scores.jsonl"
    return {key_for(row): row for row in load_jsonl(path)}


def policy_route(score_row, threshold):
    delta = score_row.get("delta_direct_minus_reason_route_nll")
    return "reason" if delta is not None and delta >= threshold else "direct"


def build_cases():
    cases = []
    for split in SPLITS:
        direct = load_route_metrics(split, "direct")
        reason = load_route_metrics(split, "reason")
        main_scores = load_scores(MAIN[0], split)
        comp_scores = load_scores(COMPARATOR[0], split)
        for key in sorted(set(direct) & set(reason) & set(main_scores) & set(comp_scores)):
            gain = {
                metric: reason[key][metric] - direct[key][metric]
                for metric in ["argument_f1", "event_f1", "trigger_f1", "score"]
            }
            main_delta = main_scores[key]["delta_direct_minus_reason_route_nll"]
            comp_delta = comp_scores[key]["delta_direct_minus_reason_route_nll"]
            cases.append(
                {
                    "case_id": f"{split}::{key}",
                    "split": split,
                    "wnd_id": key,
                    "direct": direct[key],
                    "reason": reason[key],
                    "gain_reason_minus_direct": gain,
                    "main_margin": main_delta,
                    "comparator_margin": comp_delta,
                    "main_route": policy_route(main_scores[key], MAIN[2]),
                    "comparator_route": policy_route(comp_scores[key], COMPARATOR[2]),
                }
            )
    return cases


def prf(selected, positives):
    tp = len(selected & positives)
    fp = len(selected - positives)
    fn = len(positives - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def split_summary(cases, route_field):
    out = []
    for split in ["test", "test_seen", "test_unseen"]:
        rows = cases if split == "test" else [row for row in cases if row["split"] == split]
        selected = {row["case_id"] for row in rows if row[route_field] == "reason"}
        positives = {row["case_id"] for row in rows if row["gain_reason_minus_direct"]["score"] > 0}
        selected_rows = [row for row in rows if row[route_field] == "reason"]
        harmful_rows = [row for row in selected_rows if row["gain_reason_minus_direct"]["score"] < 0]
        out.append(
            {
                "split": split,
                "num_examples": len(rows),
                "positive_gain_count": len(positives),
                "positive_gain_rate": len(positives) / len(rows) if rows else 0.0,
                "selected_reason_count": len(selected),
                "selected_reason_rate": len(selected) / len(rows) if rows else 0.0,
                "positive_gain_prf": prf(selected, positives),
                "selected_score_gain": summarize_numbers(
                    [row["gain_reason_minus_direct"]["score"] for row in selected_rows]
                ),
                "selected_argument_gain_mean": mean(
                    row["gain_reason_minus_direct"]["argument_f1"] for row in selected_rows
                ),
                "selected_event_gain_mean": mean(
                    row["gain_reason_minus_direct"]["event_f1"] for row in selected_rows
                ),
                "selected_trigger_gain_mean": mean(
                    row["gain_reason_minus_direct"]["trigger_f1"] for row in selected_rows
                ),
                "selected_harm_count": len(harmful_rows),
                "selected_harm_rate": len(harmful_rows) / len(selected_rows) if selected_rows else 0.0,
            }
        )
    return out


def margin_bucket(value):
    if value < 0.0:
        return "<0.00"
    if value < 0.10:
        return "0.00-0.10"
    if value < 0.25:
        return "0.10-0.25"
    if value < 0.50:
        return "0.25-0.50"
    return ">=0.50"


def margin_buckets(cases):
    grouped = defaultdict(list)
    for row in cases:
        grouped[(row["split"], margin_bucket(row["main_margin"]))].append(row)
        grouped[("test", margin_bucket(row["main_margin"]))].append(row)
    order = ["<0.00", "0.00-0.10", "0.10-0.25", "0.25-0.50", ">=0.50"]
    out = []
    for split in ["test", "test_seen", "test_unseen"]:
        for bucket in order:
            rows = grouped.get((split, bucket), [])
            if not rows:
                continue
            out.append(
                {
                    "split": split,
                    "main_margin_bucket": bucket,
                    "count": len(rows),
                    "avg_score_gain": mean(row["gain_reason_minus_direct"]["score"] for row in rows),
                    "avg_argument_gain": mean(row["gain_reason_minus_direct"]["argument_f1"] for row in rows),
                    "avg_event_gain": mean(row["gain_reason_minus_direct"]["event_f1"] for row in rows),
                    "avg_trigger_gain": mean(row["gain_reason_minus_direct"]["trigger_f1"] for row in rows),
                    "positive_gain_rate": mean(1.0 if row["gain_reason_minus_direct"]["score"] > 0 else 0.0 for row in rows),
                }
            )
    return out


def overlap_summary(cases):
    out = []
    for split in ["test", "test_seen", "test_unseen"]:
        rows = cases if split == "test" else [row for row in cases if row["split"] == split]
        groups = defaultdict(list)
        for row in rows:
            if row["main_route"] == "reason" and row["comparator_route"] == "reason":
                key = "both"
            elif row["main_route"] == "reason":
                key = "main_only"
            elif row["comparator_route"] == "reason":
                key = "comparator_only"
            else:
                key = "neither"
            groups[key].append(row)
        for key in ["both", "main_only", "comparator_only", "neither"]:
            group = groups.get(key, [])
            out.append(
                {
                    "split": split,
                    "group": key,
                    "count": len(group),
                    "rate": len(group) / len(rows) if rows else 0.0,
                    "avg_score_gain": mean(row["gain_reason_minus_direct"]["score"] for row in group),
                    "avg_argument_gain": mean(row["gain_reason_minus_direct"]["argument_f1"] for row in group),
                    "avg_event_gain": mean(row["gain_reason_minus_direct"]["event_f1"] for row in group),
                    "avg_trigger_gain": mean(row["gain_reason_minus_direct"]["trigger_f1"] for row in group),
                }
            )
    return out


def top_cases(cases, route_field, n=12):
    selected = [row for row in cases if row[route_field] == "reason"]
    best = sorted(selected, key=lambda row: row["gain_reason_minus_direct"]["score"], reverse=True)[:n]
    worst = sorted(selected, key=lambda row: row["gain_reason_minus_direct"]["score"])[:n]
    keep = []
    for label, rows in [("best_selected", best), ("worst_selected", worst)]:
        for row in rows:
            keep.append(
                {
                    "bucket": label,
                    "split": row["split"],
                    "wnd_id": row["wnd_id"],
                    "main_margin": row["main_margin"],
                    "comparator_margin": row["comparator_margin"],
                    "main_route": row["main_route"],
                    "comparator_route": row["comparator_route"],
                    "score_gain": row["gain_reason_minus_direct"]["score"],
                    "argument_gain": row["gain_reason_minus_direct"]["argument_f1"],
                    "event_gain": row["gain_reason_minus_direct"]["event_f1"],
                    "trigger_gain": row["gain_reason_minus_direct"]["trigger_f1"],
                }
            )
    return keep


def render_policy_summary(rows, title):
    lines = [
        f"### {title}",
        "",
        "| split | reason-helpful rate | selected reason | P/R/F1 vs K2 gain>0 | selected gain mean/median/min/max | harm rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        prf_row = row["positive_gain_prf"]
        gains = row["selected_score_gain"]
        lines.append(
            f"| `{row['split']}` | {pct(row['positive_gain_rate'])} | "
            f"{row['selected_reason_count']} ({pct(row['selected_reason_rate'])}) | "
            f"{fmt(prf_row['precision'], 3)}/{fmt(prf_row['recall'], 3)}/{fmt(prf_row['f1'], 3)} | "
            f"{signed(gains['mean'])}/{signed(gains['median'])}/{signed(gains['min'])}/{signed(gains['max'])} | "
            f"{pct(row['selected_harm_rate'])} |"
        )
    return "\n".join(lines)


def render_margin_table(rows):
    lines = [
        "| split | main margin bucket | count | positive gain rate | avg gain A/E/T/Score |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['split']}` | `{row['main_margin_bucket']}` | {row['count']} | {pct(row['positive_gain_rate'])} | "
            f"{signed(row['avg_argument_gain'])}/{signed(row['avg_event_gain'])}/"
            f"{signed(row['avg_trigger_gain'])}/{signed(row['avg_score_gain'])} |"
        )
    return "\n".join(lines)


def render_overlap_table(rows):
    lines = [
        "| split | group | count | rate | avg gain A/E/T/Score |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['split']}` | `{row['group']}` | {row['count']} | {pct(row['rate'])} | "
            f"{signed(row['avg_argument_gain'])}/{signed(row['avg_event_gain'])}/"
            f"{signed(row['avg_trigger_gain'])}/{signed(row['avg_score_gain'])} |"
        )
    return "\n".join(lines)


def render_cases_table(rows):
    lines = [
        "| bucket | split | wnd_id | main margin | comparator margin | routes | gain A/E/T/Score |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['bucket']}` | `{row['split']}` | `{row['wnd_id']}` | "
            f"{fmt(row['main_margin'])} | {fmt(row['comparator_margin'])} | "
            f"{row['main_route']}/{row['comparator_route']} | "
            f"{signed(row['argument_gain'])}/{signed(row['event_gain'])}/"
            f"{signed(row['trigger_gain'])}/{signed(row['score_gain'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    main_test = next(row for row in payload["main_summary"] if row["split"] == "test")
    comp_test = next(row for row in payload["comparator_summary"] if row["split"] == "test")
    lines = [
        "# Sampled K2 Formal Route Decision Diagnosis",
        "",
        "This report diagnoses the formal route decisions behind the fixed route-NLL policies using K2 direct/reason sampled utility. A sample is treated as K2 reason-helpful when `reason_score - direct_score > 0`.",
        "",
        "## Policy Quality",
        "",
        render_policy_summary(payload["main_summary"], "Main: checkpoint-50 / margin >= 0.25"),
        "",
        render_policy_summary(payload["comparator_summary"], "Comparator: checkpoint-75 / margin >= 0.05"),
        "",
        "## Main Margin Buckets",
        "",
        render_margin_table(payload["margin_buckets"]),
        "",
        "## Main vs Comparator Selection Overlap",
        "",
        render_overlap_table(payload["overlap_summary"]),
        "",
        "## Selected Case Extremes",
        "",
        render_cases_table(payload["main_selected_extremes"]),
        "",
        "## Reading",
        "",
        f"- Main policy selects `{main_test['selected_reason_count']}` / `{main_test['num_examples']}` formal samples as Reason, with selected-score-gain mean `{main_test['selected_score_gain']['mean']:+.4f}` and harm rate `{main_test['selected_harm_rate']:.1%}`.",
        f"- Comparator selects more Reason samples (`{comp_test['selected_reason_count']}`) but has weaker selected-score-gain mean `{comp_test['selected_score_gain']['mean']:+.4f}` and higher harm rate `{comp_test['selected_harm_rate']:.1%}`.",
        "- The main policy is conservative: it misses many K2-positive samples, but the samples it routes to Reason are net beneficial on both seen and unseen formal splits.",
        "",
        "## Inputs",
        "",
        f"- sample root: `{SAMPLE_ROOT}`",
        f"- route-NLL root: `{NLL_ROOT}`",
        f"- artifact JSON: `{REPORT_JSON}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    cases = build_cases()
    payload = {
        "sample_root": SAMPLE_ROOT.as_posix(),
        "route_nll_root": NLL_ROOT.as_posix(),
        "main_policy": {"checkpoint": MAIN[0], "policy": MAIN[1], "threshold": MAIN[2]},
        "comparator_policy": {"checkpoint": COMPARATOR[0], "policy": COMPARATOR[1], "threshold": COMPARATOR[2]},
        "num_cases": len(cases),
        "main_summary": split_summary(cases, "main_route"),
        "comparator_summary": split_summary(cases, "comparator_route"),
        "margin_buckets": margin_buckets(cases),
        "overlap_summary": overlap_summary(cases),
        "main_selected_extremes": top_cases(cases, "main_route"),
    }
    write_json(REPORT_JSON, payload)
    write_text(REPORT_MD, render_report(payload))
    print(json.dumps({"output_json": REPORT_JSON.as_posix(), "output_md": REPORT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
