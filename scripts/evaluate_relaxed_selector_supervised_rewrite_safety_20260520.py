#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.diagnose_sampled_k2_goldfree_harmful_cases_20260519 import (  # noqa: E402
    load_exec_rows,
    pair_features,
)
from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402
from scripts.summarize_sampled_k2_structural_proxy_locked_validation_20260519 import (  # noqa: E402
    DEFAULT_FRESH_NLL_ROOT,
    DEFAULT_NEW_NLL_ROOT,
    DEFAULT_OLD_NLL_ROOT,
    DEFAULT_SAMPLE_ROOT,
    METRICS,
    build_cases,
)


SPLITS = ["test_seen", "test_unseen"]
OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_rewrite_safety_replay_20260520"
REPORT_MD = REPO / "reports/2026-05-20_stage2_relaxed_selector_supervised_rewrite_safety_replay.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_relaxed_selector_supervised_rewrite_safety_replay.json"
CANDIDATES_JSONL = OUTPUT_ROOT / "rewrite_safety_candidates.jsonl"

FEATURES = [
    "fresh_margin",
    "old17_18_margin",
    "new19_20_margin",
    "avg_margin",
    "margin_range",
    "num_margins_ge_0p25",
    "sample_arg_text_jaccard_mean",
    "sample_arg_span_jaccard_mean",
    "sample_arg_role_jaccard_mean",
    "sample_event_type_jaccard_mean",
    "sample_event_type_role_jaccard_mean",
    "sample_trigger_text_jaccard_mean",
    "sample_reason_arg_text_retention_mean",
    "sample_reason_event_type_retention_mean",
    "sample_reason_new_arg_text_count_mean",
    "sample_reason_new_event_type_count_mean",
    "sample_argument_count_delta_mean",
    "sample_event_count_delta_mean",
    "exec_arg_text_jaccard",
    "exec_arg_span_jaccard",
    "exec_arg_role_jaccard",
    "exec_event_type_jaccard",
    "exec_event_type_role_jaccard",
    "exec_trigger_text_jaccard",
    "exec_reason_arg_text_retention",
    "exec_reason_event_type_retention",
    "exec_reason_new_arg_text_count",
    "exec_reason_new_event_type_count",
    "exec_argument_count_delta",
    "exec_event_count_delta",
]

POSITIVE_FEATURES = {
    "fresh_margin",
    "old17_18_margin",
    "new19_20_margin",
    "avg_margin",
    "num_margins_ge_0p25",
    "sample_arg_text_jaccard_mean",
    "sample_arg_span_jaccard_mean",
    "sample_arg_role_jaccard_mean",
    "sample_event_type_jaccard_mean",
    "sample_event_type_role_jaccard_mean",
    "sample_trigger_text_jaccard_mean",
    "sample_reason_arg_text_retention_mean",
    "sample_reason_event_type_retention_mean",
    "exec_arg_text_jaccard",
    "exec_arg_span_jaccard",
    "exec_arg_role_jaccard",
    "exec_event_type_jaccard",
    "exec_event_type_role_jaccard",
    "exec_trigger_text_jaccard",
    "exec_reason_arg_text_retention",
    "exec_reason_event_type_retention",
}

POLICIES = [
    "direct_only",
    "relaxed_full_reason",
    "oracle_safe",
    "loo_naive_bayes_top05",
    "loo_naive_bayes_top10",
    "loo_naive_bayes_top15",
    "loo_naive_bayes_top20",
    "loo_linear_z_top05",
    "loo_linear_z_top10",
    "loo_linear_z_top15",
    "loo_linear_z_top20",
    "seen_train_nb_top05",
    "seen_train_nb_top10",
    "seen_train_nb_top15",
    "seen_train_nb_top20",
    "seen_train_linear_top05",
    "seen_train_linear_top10",
    "seen_train_linear_top15",
    "seen_train_linear_top20",
]


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def stdev(values):
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return math.sqrt(mean((value - mu) ** 2 for value in vals))


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def stable_numeric(value):
    if value is None:
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def add_exec_pair_features(cases_by_split):
    direct_rows = {split: load_exec_rows(split, "direct") for split in SPLITS}
    reason_rows = {split: load_exec_rows(split, "reason") for split in SPLITS}
    for split in SPLITS:
        for case in cases_by_split[split]:
            key = case["key"]
            for name, value in pair_features(direct_rows[split][key], reason_rows[split][key]).items():
                case[f"exec_{name}"] = value


def build_candidate_rows(cases_by_split):
    rows = []
    for split in SPLITS:
        for case in cases_by_split[split]:
            direct = case["single_gen_execution_direct"]
            reason = case["single_gen_execution_reason"]
            gain = reason["score"] - direct["score"]
            features = {name: stable_numeric(case.get(name)) for name in FEATURES}
            row = {
                "split": split,
                "case_id": case["case_id"],
                "key": case["key"],
                "relaxed_selected": relaxed_selector(case),
                "single_gen_score_gain": gain,
                "label_safe": gain >= 0.0,
                "label_helpful": gain > 0.0,
                "label_harmful": gain < 0.0,
                "direct": direct,
                "reason": reason,
                "features": features,
            }
            for name, value in features.items():
                row[name] = value
            rows.append(row)
    return rows


def fit_stats(rows, features=FEATURES):
    stats = {}
    for feat in features:
        vals = [row["features"][feat] for row in rows]
        mu = mean(vals)
        sd = stdev(vals)
        stats[feat] = {"mean": mu, "std": sd if sd > 1e-12 else 1.0}
    return stats


def linear_feature_weights(rows, features=FEATURES):
    safe = [row for row in rows if row["label_safe"]]
    harm = [row for row in rows if row["label_harmful"]]
    all_stats = fit_stats(rows, features)
    weights = {}
    for feat in features:
        safe_mean = mean(row["features"][feat] for row in safe)
        harm_mean = mean(row["features"][feat] for row in harm)
        direction = 1.0 if feat in POSITIVE_FEATURES else -1.0
        raw = direction * (safe_mean - harm_mean) / all_stats[feat]["std"]
        weights[feat] = raw
    return weights, all_stats


def score_linear(row, weights, stats):
    out = 0.0
    for feat, weight in weights.items():
        val = row["features"][feat]
        out += weight * ((val - stats[feat]["mean"]) / stats[feat]["std"])
    return out


def fit_gaussian_nb(rows, features=FEATURES):
    classes = {
        True: [row for row in rows if row["label_safe"]],
        False: [row for row in rows if row["label_harmful"]],
    }
    model = {"features": list(features), "prior": {}, "params": {}}
    total = len(rows)
    for label, label_rows in classes.items():
        model["prior"][label] = (len(label_rows) + 1.0) / (total + 2.0)
        model["params"][label] = {}
        source = label_rows if label_rows else rows
        for feat in features:
            vals = [row["features"][feat] for row in source]
            mu = mean(vals)
            var = mean((value - mu) ** 2 for value in vals)
            model["params"][label][feat] = {"mean": mu, "var": max(var, 1e-4)}
    return model


def score_gaussian_nb(row, model):
    scores = {}
    for label in [True, False]:
        score = math.log(model["prior"][label])
        for feat in model["features"]:
            val = row["features"][feat]
            params = model["params"][label][feat]
            var = params["var"]
            score += -0.5 * math.log(2 * math.pi * var) - ((val - params["mean"]) ** 2 / (2 * var))
        scores[label] = score
    return scores[True] - scores[False]


def add_loo_scores(candidates):
    selected = [row for row in candidates if row["relaxed_selected"]]
    for row in candidates:
        row["scores"] = {}
    for idx, row in enumerate(selected):
        train = selected[:idx] + selected[idx + 1 :]
        if len({item["label_safe"] for item in train}) < 2:
            row["scores"]["loo_naive_bayes"] = 0.0
            row["scores"]["loo_linear_z"] = 0.0
            continue
        nb = fit_gaussian_nb(train)
        weights, stats = linear_feature_weights(train)
        row["scores"]["loo_naive_bayes"] = score_gaussian_nb(row, nb)
        row["scores"]["loo_linear_z"] = score_linear(row, weights, stats)
    for row in candidates:
        if not row["relaxed_selected"]:
            row["scores"]["loo_naive_bayes"] = float("-inf")
            row["scores"]["loo_linear_z"] = float("-inf")


def add_seen_train_scores(candidates):
    train = [row for row in candidates if row["split"] == "test_seen" and row["relaxed_selected"]]
    if len({row["label_safe"] for row in train}) < 2:
        return
    nb = fit_gaussian_nb(train)
    weights, stats = linear_feature_weights(train)
    for row in candidates:
        if row["relaxed_selected"]:
            row["scores"]["seen_train_nb"] = score_gaussian_nb(row, nb)
            row["scores"]["seen_train_linear"] = score_linear(row, weights, stats)
        else:
            row["scores"]["seen_train_nb"] = float("-inf")
            row["scores"]["seen_train_linear"] = float("-inf")


def policy_selected_set(policy, rows):
    selected = [row for row in rows if row["relaxed_selected"]]
    if policy == "direct_only":
        return set()
    if policy == "relaxed_full_reason":
        return {row["case_id"] for row in selected}
    if policy == "oracle_safe":
        return {row["case_id"] for row in selected if row["label_safe"]}
    if policy.endswith("_top05"):
        budget = 5
    elif policy.endswith("_top10"):
        budget = 10
    elif policy.endswith("_top15"):
        budget = 15
    elif policy.endswith("_top20"):
        budget = 20
    else:
        raise KeyError(policy)
    if policy.startswith("loo_naive_bayes"):
        score_name = "loo_naive_bayes"
    elif policy.startswith("loo_linear_z"):
        score_name = "loo_linear_z"
    elif policy.startswith("seen_train_nb"):
        score_name = "seen_train_nb"
    elif policy.startswith("seen_train_linear"):
        score_name = "seen_train_linear"
    else:
        raise KeyError(policy)
    ordered = sorted(selected, key=lambda row: (row["scores"].get(score_name, float("-inf")), row["case_id"]), reverse=True)
    return {row["case_id"] for row in ordered[:budget]}


def summarize_policy(policy, candidates):
    rows = []
    for split in ["test", "test_seen", "test_unseen"]:
        split_rows = candidates if split == "test" else [row for row in candidates if row["split"] == split]
        selected_set = policy_selected_set(policy, split_rows)
        direct_metrics = avg_metrics(row["direct"] for row in split_rows)
        routed_metrics = avg_metrics(row["reason"] if row["case_id"] in selected_set else row["direct"] for row in split_rows)
        selected = [row for row in split_rows if row["case_id"] in selected_set]
        buckets = Counter("helpful" if row["single_gen_score_gain"] > 0 else "harmful" if row["single_gen_score_gain"] < 0 else "neutral" for row in selected)
        rows.append(
            {
                "policy": policy,
                "split": split,
                "num_examples": len(split_rows),
                "pred_reason_count": len(selected),
                "pred_reason_rate": len(selected) / len(split_rows) if split_rows else 0.0,
                "selected_helpful": buckets["helpful"],
                "selected_harmful": buckets["harmful"],
                "selected_neutral": buckets["neutral"],
                "selected_harm_rate": buckets["harmful"] / len(selected) if selected else 0.0,
                "selected_gain_mean": mean(row["single_gen_score_gain"] for row in selected),
                "direct": direct_metrics,
                "routed": routed_metrics,
                "routed_minus_direct": {
                    metric: routed_metrics[metric] - direct_metrics[metric]
                    for metric in METRICS
                },
            }
        )
    test = next(row for row in rows if row["split"] == "test")
    seen = next(row for row in rows if row["split"] == "test_seen")
    unseen = next(row for row in rows if row["split"] == "test_unseen")
    if policy == "oracle_safe":
        protocol = "oracle_upper_bound"
    elif policy.startswith("seen_train"):
        protocol = "seen_train_in_sample_seen_transfer_unseen"
    elif policy.startswith("loo"):
        protocol = "leave_one_out_candidate_replay"
    elif policy == "relaxed_full_reason":
        protocol = "baseline_relaxed"
    else:
        protocol = "baseline_direct"
    passes_target = (
        test["routed_minus_direct"]["score"] > 0.0085
        and seen["routed_minus_direct"]["score"] >= 0.0042
        and test["selected_harm_rate"] <= 0.16
        and seen["selected_harm_rate"] <= 0.20
        and unseen["routed_minus_direct"]["score"] >= 0.0200
    )
    return {
        "policy": policy,
        "protocol": protocol,
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
        "passes_target": passes_target,
        "passes_target_unbiased": passes_target and policy.startswith("loo"),
        "rows": rows,
    }


def feature_audit(candidates):
    selected = [row for row in candidates if row["relaxed_selected"]]
    safe = [row for row in selected if row["label_safe"]]
    harm = [row for row in selected if row["label_harmful"]]
    out = []
    for feat in FEATURES:
        safe_vals = [row["features"][feat] for row in safe]
        harm_vals = [row["features"][feat] for row in harm]
        pooled = stdev(safe_vals + harm_vals)
        out.append(
            {
                "feature": feat,
                "safe_mean": mean(safe_vals),
                "harm_mean": mean(harm_vals),
                "diff_safe_minus_harm": mean(safe_vals) - mean(harm_vals),
                "abs_standardized_diff": abs(mean(safe_vals) - mean(harm_vals)) / pooled if pooled else 0.0,
            }
        )
    out.sort(key=lambda row: (-row["abs_standardized_diff"], row["feature"]))
    return out


def label_summary(candidates):
    out = []
    for split in ["test", "test_seen", "test_unseen"]:
        split_rows = candidates if split == "test" else [row for row in candidates if row["split"] == split]
        selected = [row for row in split_rows if row["relaxed_selected"]]
        buckets = Counter("helpful" if row["label_helpful"] else "harmful" if row["label_harmful"] else "neutral" for row in selected)
        out.append(
            {
                "split": split,
                "num_examples": len(split_rows),
                "relaxed_selected": len(selected),
                "relaxed_reason_rate": len(selected) / len(split_rows) if split_rows else 0.0,
                "helpful": buckets["helpful"],
                "harmful": buckets["harmful"],
                "neutral": buckets["neutral"],
                "harm_rate": buckets["harmful"] / len(selected) if selected else 0.0,
                "selected_gain_mean": mean(row["single_gen_score_gain"] for row in selected),
            }
        )
    return out


def render_leaderboard(rows):
    lines = [
        "| policy | protocol | pass | unbiased pass | reason test/seen/unseen | score test/seen/unseen | harm test/seen/unseen | H/h/N | selected gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | `{row['protocol']}` | `{row['passes_target']}` | `{row['passes_target_unbiased']}` | "
            f"{pct(row['test_reason_rate'])}/{pct(row['seen_reason_rate'])}/{pct(row['unseen_reason_rate'])} | "
            f"{signed(row['test_score_delta'])}/{signed(row['seen_score_delta'])}/{signed(row['unseen_score_delta'])} | "
            f"{pct(row['test_harm_rate'])}/{pct(row['seen_harm_rate'])}/{pct(row['unseen_harm_rate'])} | "
            f"{row['test_selected_helpful']}/{row['test_selected_harmful']}/{row['test_selected_neutral']} | "
            f"{signed(row['test_selected_gain_mean'])} |"
        )
    return "\n".join(lines)


def render_label_summary(rows):
    lines = [
        "| split | relaxed selected | reason rate | helpful/harmful/neutral | harm | gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['split']}` | {row['relaxed_selected']} / {row['num_examples']} | {pct(row['relaxed_reason_rate'])} | "
            f"{row['helpful']}/{row['harmful']}/{row['neutral']} | {pct(row['harm_rate'])} | {signed(row['selected_gain_mean'])} |"
        )
    return "\n".join(lines)


def render_feature_audit(rows, limit=14):
    lines = [
        "| feature | safe mean | harm mean | diff | std diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['feature']}` | {fmt(row['safe_mean'])} | {fmt(row['harm_mean'])} | "
            f"{signed(row['diff_safe_minus_harm'])} | {fmt(row['abs_standardized_diff'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    best = payload["leaderboard"][0]
    lines = [
        "# Relaxed Selector Supervised Rewrite Safety Replay",
        "",
        "This exploratory replay uses execution gain labels only to fit lightweight rewrite-safety selectors inside the relaxed-selector candidate set.",
        "",
        "Important: `oracle_safe` and `seen_train_*` are optimistic diagnostics. The only less-biased replay here is `loo_*`, and a deployable selector still needs train/dev supervision outside this formal test set.",
        "",
        "## Candidate Label Audit",
        "",
        render_label_summary(payload["label_summary"]),
        "",
        "## Feature Separability",
        "",
        render_feature_audit(payload["feature_audit"]),
        "",
        "## Replay Leaderboard",
        "",
        render_leaderboard(payload["leaderboard"]),
        "",
        "## Reading",
        "",
        f"- Best by target screen/sort is `{best['policy']}`: score delta `{best['test_score_delta']:+.4f}/{best['seen_score_delta']:+.4f}/{best['unseen_score_delta']:+.4f}`, harm `{best['test_harm_rate']:.1%}/{best['seen_harm_rate']:.1%}/{best['unseen_harm_rate']:.1%}`.",
        "- `oracle_safe` is an upper bound that uses gold execution labels and is not deployable.",
        "- LOOCV rows estimate whether the relaxed candidate set is separable without scoring an example by a model trained on its own label.",
        "- Seen-train rows test whether seen-selected labels can rank unseen candidates, but their seen/test aggregate numbers are in-sample and should not be treated as final policy validation.",
        "",
        "## Artifacts",
        "",
        f"- candidates: `{payload['candidates_jsonl']}`",
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
    add_exec_pair_features(cases_by_split)
    candidates = build_candidate_rows(cases_by_split)
    add_loo_scores(candidates)
    add_seen_train_scores(candidates)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(CANDIDATES_JSONL, candidates)

    consolidated = []
    details = {}
    for policy in POLICIES:
        summary = summarize_policy(policy, candidates)
        consolidated.append(summary)
        details[policy] = {"rows": summary.pop("rows")}
    consolidated.sort(
        key=lambda row: (
            row["passes_target_unbiased"],
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
        "output_root": OUTPUT_ROOT.as_posix(),
        "candidates_jsonl": CANDIDATES_JSONL.as_posix(),
        "num_candidates": len(candidates),
        "num_relaxed_selected": sum(1 for row in candidates if row["relaxed_selected"]),
        "features": FEATURES,
        "label_summary": label_summary(candidates),
        "feature_audit": feature_audit(candidates),
        "leaderboard": consolidated,
        "details": details,
        "report_md": REPORT_MD.as_posix(),
        "report_json": REPORT_JSON.as_posix(),
    }
    write_json(REPORT_JSON, payload)
    write_json(OUTPUT_ROOT / "summary.json", payload)
    write_text(REPORT_MD, render_report(payload))
    print(json.dumps({"report_md": REPORT_MD.as_posix(), "report_json": REPORT_JSON.as_posix()}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 24])
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--fresh-nll-root", type=Path, default=DEFAULT_FRESH_NLL_ROOT)
    parser.add_argument("--old-nll-root", type=Path, default=DEFAULT_OLD_NLL_ROOT)
    parser.add_argument("--new-nll-root", type=Path, default=DEFAULT_NEW_NLL_ROOT)
    parser.add_argument("--checkpoint", default="checkpoint-50")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
