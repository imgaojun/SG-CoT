#!/usr/bin/env python3
import json
import math
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402
from scripts.summarize_sampled_k2_structural_proxy_locked_validation_20260519 import METRICS  # noqa: E402


INPUT_JSONL = REPO / "outputs/stage2_adaptive_rewrite_safety_train_dev_locked_replay_20260520/formal_scored_candidates.jsonl"
OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_rewrite_safety_margin_augmented_diagnostic_20260520"
REPORT_MD = REPO / "reports/2026-05-20_stage2_rewrite_safety_margin_augmented_diagnostic.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_rewrite_safety_margin_augmented_diagnostic.json"
SCORED_JSONL = OUTPUT_ROOT / "margin_augmented_formal_candidates.jsonl"

RANKERS = [
    "train_dev_nb_safe",
    "train_dev_gain_corr",
    "fresh_margin",
    "avg_margin",
    "num_margins_ge_0p25",
    "margin_combo",
    "nb_plus_margin",
    "gain_plus_margin",
    "oracle_safe",
]
BUDGETS = [5, 10, 15, 20, 25, 32]
FEATURES = [
    "train_dev_nb_safe",
    "train_dev_gain_corr",
    "fresh_margin",
    "old17_18_margin",
    "new19_20_margin",
    "avg_margin",
    "margin_range",
    "num_margins_ge_0p25",
    "margin_combo",
    "nb_plus_margin",
    "gain_plus_margin",
]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def stdev(values):
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return math.sqrt(mean((value - mu) ** 2 for value in vals))


def safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def zscore(value, values):
    sd = stdev(values)
    return 0.0 if sd <= 1e-12 else (value - mean(values)) / sd


def avg_metrics(rows):
    items = list(rows)
    return {metric: mean(row[metric] for row in items) for metric in METRICS}


def add_scores(rows):
    selected = [row for row in rows if row["relaxed_selected"]]
    nb_vals = [safe_float(row["train_dev_scores"]["nb_safe"]) for row in selected]
    gain_vals = [safe_float(row["train_dev_scores"]["gain_corr"]) for row in selected]
    fresh_vals = [safe_float(row.get("fresh_margin")) for row in selected]
    avg_vals = [safe_float(row.get("avg_margin")) for row in selected]
    ge_vals = [safe_float(row.get("num_margins_ge_0p25")) for row in selected]
    range_vals = [safe_float(row.get("margin_range")) for row in selected]
    for row in rows:
        nb = safe_float(row.get("train_dev_scores", {}).get("nb_safe"))
        gain = safe_float(row.get("train_dev_scores", {}).get("gain_corr"))
        fresh = safe_float(row.get("fresh_margin"))
        avg = safe_float(row.get("avg_margin"))
        ge = safe_float(row.get("num_margins_ge_0p25"))
        margin_range = safe_float(row.get("margin_range"))
        margin_combo = (
            zscore(fresh, fresh_vals)
            + zscore(avg, avg_vals)
            + zscore(ge, ge_vals)
            - zscore(margin_range, range_vals)
        )
        row["diagnostic_scores"] = {
            "train_dev_nb_safe": nb,
            "train_dev_gain_corr": gain,
            "fresh_margin": fresh,
            "avg_margin": avg,
            "num_margins_ge_0p25": ge,
            "margin_combo": margin_combo,
            "nb_plus_margin": zscore(nb, nb_vals) + margin_combo,
            "gain_plus_margin": zscore(gain, gain_vals) + margin_combo,
            "oracle_safe": 1.0 if row["label_safe"] else 0.0,
        }
        for name, value in row["diagnostic_scores"].items():
            row[name] = value


def selected_set(rows, ranker, budget):
    selected = [row for row in rows if row["relaxed_selected"]]
    ordered = sorted(selected, key=lambda row: (row["diagnostic_scores"][ranker], row["case_id"]), reverse=True)
    return {row["case_id"] for row in ordered[:budget]}


def summarize_ranker_budget(rows, ranker, budget):
    out = []
    for split in ["test", "test_seen", "test_unseen"]:
        split_rows = rows if split == "test" else [row for row in rows if row["split"] == split]
        selected_ids = selected_set(split_rows, ranker, budget)
        direct = avg_metrics(row["direct"] for row in split_rows)
        routed = avg_metrics(row["reason"] if row["case_id"] in selected_ids else row["direct"] for row in split_rows)
        selected = [row for row in split_rows if row["case_id"] in selected_ids]
        buckets = Counter("helpful" if row["label_helpful"] else "harmful" if row["label_harmful"] else "neutral" for row in selected)
        out.append(
            {
                "split": split,
                "num_examples": len(split_rows),
                "pred_reason_count": len(selected),
                "pred_reason_rate": len(selected) / len(split_rows) if split_rows else 0.0,
                "selected_helpful": buckets["helpful"],
                "selected_harmful": buckets["harmful"],
                "selected_neutral": buckets["neutral"],
                "selected_harm_rate": buckets["harmful"] / len(selected) if selected else 0.0,
                "selected_gain_mean": mean(row["single_gen_score_gain"] for row in selected),
                "direct": direct,
                "routed": routed,
                "routed_minus_direct": {metric: routed[metric] - direct[metric] for metric in METRICS},
            }
        )
    test = next(row for row in out if row["split"] == "test")
    seen = next(row for row in out if row["split"] == "test_seen")
    unseen = next(row for row in out if row["split"] == "test_unseen")
    return {
        "ranker": ranker,
        "budget": budget,
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
        "test_selected_helpful": test["selected_helpful"],
        "test_selected_harmful": test["selected_harmful"],
        "test_selected_neutral": test["selected_neutral"],
        "passes_target": (
            test["routed_minus_direct"]["score"] > 0.0085
            and seen["routed_minus_direct"]["score"] >= 0.0042
            and test["selected_harm_rate"] <= 0.16
            and seen["selected_harm_rate"] <= 0.20
            and unseen["routed_minus_direct"]["score"] >= 0.0200
        ),
        "rows": out,
    }


def correlation(rows, feature):
    selected = [row for row in rows if row["relaxed_selected"]]
    xs = [safe_float(row.get(feature)) for row in selected]
    ys = [safe_float(row["single_gen_score_gain"]) for row in selected]
    if not xs or stdev(xs) <= 1e-12 or stdev(ys) <= 1e-12:
        return 0.0
    xmu = mean(xs)
    ymu = mean(ys)
    return mean((x - xmu) * (y - ymu) for x, y in zip(xs, ys)) / (stdev(xs) * stdev(ys))


def feature_audit(rows):
    selected = [row for row in rows if row["relaxed_selected"]]
    safe = [row for row in selected if row["label_safe"]]
    harm = [row for row in selected if row["label_harmful"]]
    out = []
    for feat in FEATURES:
        safe_vals = [safe_float(row.get(feat)) for row in safe]
        harm_vals = [safe_float(row.get(feat)) for row in harm]
        pooled = stdev(safe_vals + harm_vals)
        out.append(
            {
                "feature": feat,
                "corr_gain": correlation(rows, feat),
                "safe_mean": mean(safe_vals),
                "harm_mean": mean(harm_vals),
                "diff_safe_minus_harm": mean(safe_vals) - mean(harm_vals),
                "abs_standardized_diff": abs(mean(safe_vals) - mean(harm_vals)) / pooled if pooled else 0.0,
            }
        )
    out.sort(key=lambda row: (-abs(row["corr_gain"]), -row["abs_standardized_diff"]))
    return out


def render_leaderboard(rows, limit=24):
    lines = [
        "| ranker | budget | pass | reason test/seen/unseen | score test/seen/unseen | harm test/seen/unseen | H/h/N | gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['ranker']}` | {row['budget']} | `{row['passes_target']}` | "
            f"{pct(row['test_reason_rate'])}/{pct(row['seen_reason_rate'])}/{pct(row['unseen_reason_rate'])} | "
            f"{signed(row['test_score_delta'])}/{signed(row['seen_score_delta'])}/{signed(row['unseen_score_delta'])} | "
            f"{pct(row['test_harm_rate'])}/{pct(row['seen_harm_rate'])}/{pct(row['unseen_harm_rate'])} | "
            f"{row['test_selected_helpful']}/{row['test_selected_harmful']}/{row['test_selected_neutral']} | "
            f"{signed(row['test_selected_gain_mean'])} |"
        )
    return "\n".join(lines)


def render_feature_table(rows):
    lines = [
        "| feature | corr gain | safe mean | harm mean | diff | std diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['feature']}` | {fmt(row['corr_gain'])} | {fmt(row['safe_mean'])} | "
            f"{fmt(row['harm_mean'])} | {signed(row['diff_safe_minus_harm'])} | {fmt(row['abs_standardized_diff'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    best = payload["leaderboard"][0]
    pass_rows = [row for row in payload["leaderboard"] if row["passes_target"]]
    lines = [
        "# Rewrite Safety Margin-Augmented Diagnostic",
        "",
        "This is a diagnostic over formal relaxed candidates. It uses formal labels to measure correlations and should not be reported as a locked policy.",
        "",
        "## Feature Audit",
        "",
        render_feature_table(payload["feature_audit"]),
        "",
        "## Ranking Sweep",
        "",
        render_leaderboard(payload["leaderboard"]),
        "",
        "## Reading",
        "",
        f"- Best diagnostic row: `{best['ranker']}` budget `{best['budget']}`, score `{best['test_score_delta']:+.4f}/{best['seen_score_delta']:+.4f}/{best['unseen_score_delta']:+.4f}`, harm `{best['test_harm_rate']:.1%}/{best['seen_harm_rate']:.1%}/{best['unseen_harm_rate']:.1%}`.",
        f"- Diagnostic rows passing target: `{len(pass_rows)}`.",
        "- A margin-combination win here would justify running train route-NLL scoring and adding margins to the locked train/dev selector.",
        "",
        "## Artifacts",
        "",
        f"- scored candidates: `{payload['scored_jsonl']}`",
        f"- JSON: `{payload['report_json']}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    rows = load_jsonl(INPUT_JSONL)
    add_scores(rows)
    leaderboard = []
    for ranker in RANKERS:
        for budget in BUDGETS:
            leaderboard.append(summarize_ranker_budget(rows, ranker, budget))
    leaderboard.sort(
        key=lambda row: (
            row["passes_target"],
            row["test_score_delta"],
            -row["test_harm_rate"],
            row["seen_score_delta"],
        ),
        reverse=True,
    )
    details = {row["ranker"] + f"_top{row['budget']:02d}": {"rows": row.pop("rows")} for row in leaderboard}
    payload = {
        "input_jsonl": INPUT_JSONL.as_posix(),
        "output_root": OUTPUT_ROOT.as_posix(),
        "scored_jsonl": SCORED_JSONL.as_posix(),
        "feature_audit": feature_audit(rows),
        "leaderboard": leaderboard,
        "details": details,
        "report_md": REPORT_MD.as_posix(),
        "report_json": REPORT_JSON.as_posix(),
    }
    write_jsonl(SCORED_JSONL, rows)
    write_json(REPORT_JSON, payload)
    write_json(OUTPUT_ROOT / "summary.json", payload)
    write_text(REPORT_MD, render_report(payload))
    print(json.dumps({"report_md": REPORT_MD.as_posix(), "report_json": REPORT_JSON.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
