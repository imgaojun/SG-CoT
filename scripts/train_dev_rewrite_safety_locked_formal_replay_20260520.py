#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.diagnose_sampled_k2_goldfree_harmful_cases_20260519 import (  # noqa: E402
    aggregate_pair_features,
    load_sample_rows,
)
from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402
from scripts.summarize_sampled_k2_structural_proxy_locked_validation_20260519 import METRICS  # noqa: E402


SPLITS = ["train", "dev_seen"]
SEEDS = [17, 18, 19, 20, 21, 22, 23, 24]
SAMPLE_ROOT = REPO / (
    "outputs/stage2_modular_dualexpert/sampled_counterfactual_utility_20260517/"
    "sampled_reason_expert_forcedreason_from_noaux_20260517_checkpoint-258"
)
LABEL_ROOT = REPO / "data/stage2_adaptive_datasets/labels"
LABEL_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_sampled_counterfactual_utility_k8_checkpoint-258"
FORMAL_CANDIDATES = REPO / "outputs/stage2_adaptive_rewrite_safety_replay_20260520/rewrite_safety_candidates.jsonl"

OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_rewrite_safety_train_dev_locked_replay_20260520"
REPORT_MD = REPO / "reports/2026-05-20_stage2_train_dev_rewrite_safety_selector_locked_formal_replay.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_train_dev_rewrite_safety_selector_locked_formal_replay.json"

FEATURES = [
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
]
POSITIVE_FEATURES = {
    "sample_arg_text_jaccard_mean",
    "sample_arg_span_jaccard_mean",
    "sample_arg_role_jaccard_mean",
    "sample_event_type_jaccard_mean",
    "sample_event_type_role_jaccard_mean",
    "sample_trigger_text_jaccard_mean",
    "sample_reason_arg_text_retention_mean",
    "sample_reason_event_type_retention_mean",
}
MODELS = ["nb_safe", "gain_corr"]
BUDGETS = [5, 10, 15, 20]


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


def stdev(values):
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return math.sqrt(mean((value - mu) ** 2 for value in vals))


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


def avg_metrics(rows):
    items = list(rows)
    return {metric: mean(row[metric] for row in items) for metric in METRICS}


def load_labels(split):
    path = LABEL_ROOT / f"{LABEL_PREFIX}_{split}_labels.jsonl"
    return {row["wnd_id"]: row for row in load_jsonl(path)}


def build_split_rows(split):
    labels = load_labels(split)
    sample_rows = {
        route: load_sample_rows(SAMPLE_ROOT, split, route, SEEDS)
        for route in ["direct", "reason"]
    }
    keys = sorted(set(labels) & set(sample_rows["direct"]) & set(sample_rows["reason"]))
    rows = []
    for key in keys:
        label = labels[key]
        pair = aggregate_pair_features(sample_rows["direct"][key], sample_rows["reason"][key])
        features = {f"sample_{name}": stable_numeric(value) for name, value in pair.items()}
        gain = stable_numeric(label.get("mean_gain"))
        row = {
            "split": split,
            "key": key,
            "case_id": f"{split}::{key}",
            "mean_gain": gain,
            "label_safe": gain >= 0.0,
            "label_helpful": gain > 0.0,
            "label_harmful": gain < 0.0,
            "route_label": label.get("route_label"),
            "utility_label": label.get("utility_label"),
            "direct_score": stable_numeric(label.get("direct_mean_score")),
            "reason_score": stable_numeric(label.get("reason_mean_score")),
            "features": {feat: stable_numeric(features.get(feat)) for feat in FEATURES},
        }
        for feat in FEATURES:
            row[feat] = row["features"][feat]
        rows.append(row)
    return rows


def fit_stats(rows):
    stats = {}
    for feat in FEATURES:
        vals = [row["features"][feat] for row in rows]
        mu = mean(vals)
        sd = stdev(vals)
        stats[feat] = {"mean": mu, "std": sd if sd > 1e-12 else 1.0}
    return stats


def fit_gain_corr(rows):
    stats = fit_stats(rows)
    gains = [row["mean_gain"] for row in rows]
    gain_mu = mean(gains)
    gain_sd = stdev(gains) or 1.0
    weights = {}
    for feat in FEATURES:
        vals = [row["features"][feat] for row in rows]
        feat_mu = stats[feat]["mean"]
        feat_sd = stats[feat]["std"]
        corr = mean(((value - feat_mu) / feat_sd) * ((gain - gain_mu) / gain_sd) for value, gain in zip(vals, gains))
        weights[feat] = corr
    return {"stats": stats, "weights": weights}


def score_gain_corr(row, model):
    out = 0.0
    for feat, weight in model["weights"].items():
        stat = model["stats"][feat]
        out += weight * ((row["features"][feat] - stat["mean"]) / stat["std"])
    return out


def fit_gaussian_nb(rows):
    classes = {
        True: [row for row in rows if row["label_safe"]],
        False: [row for row in rows if row["label_harmful"]],
    }
    model = {"prior": {}, "params": {}}
    total = len(rows)
    for label, label_rows in classes.items():
        model["prior"][label] = (len(label_rows) + 1.0) / (total + 2.0)
        model["params"][label] = {}
        source = label_rows if label_rows else rows
        for feat in FEATURES:
            vals = [row["features"][feat] for row in source]
            mu = mean(vals)
            var = mean((value - mu) ** 2 for value in vals)
            model["params"][label][feat] = {"mean": mu, "var": max(var, 1e-4)}
    return model


def score_gaussian_nb(row, model):
    scores = {}
    for label in [True, False]:
        score = math.log(model["prior"][label])
        for feat in FEATURES:
            val = row["features"][feat]
            params = model["params"][label][feat]
            var = params["var"]
            score += -0.5 * math.log(2 * math.pi * var) - ((val - params["mean"]) ** 2 / (2 * var))
        scores[label] = score
    return scores[True] - scores[False]


def fit_models(train_rows):
    return {
        "nb_safe": fit_gaussian_nb(train_rows),
        "gain_corr": fit_gain_corr(train_rows),
    }


def score_rows(rows, models):
    for row in rows:
        row["scores"] = {
            "nb_safe": score_gaussian_nb(row, models["nb_safe"]),
            "gain_corr": score_gain_corr(row, models["gain_corr"]),
        }


def summarize_label_rows(rows, split):
    selected = rows
    buckets = Counter("helpful" if row["label_helpful"] else "harmful" if row["label_harmful"] else "neutral" for row in selected)
    return {
        "split": split,
        "num_rows": len(rows),
        "helpful": buckets["helpful"],
        "harmful": buckets["harmful"],
        "neutral": buckets["neutral"],
        "harm_rate": buckets["harmful"] / len(selected) if selected else 0.0,
        "mean_gain": mean(row["mean_gain"] for row in selected),
    }


def select_top(rows, model_name, budget):
    return sorted(rows, key=lambda row: (row["scores"][model_name], row["case_id"]), reverse=True)[:budget]


def summarize_dev_policy(rows, model_name, budget):
    selected = select_top(rows, model_name, budget)
    selected_ids = {row["case_id"] for row in selected}
    buckets = Counter("helpful" if row["label_helpful"] else "harmful" if row["label_harmful"] else "neutral" for row in selected)
    return {
        "model": model_name,
        "budget": budget,
        "num_examples": len(rows),
        "pred_reason_count": len(selected),
        "pred_reason_rate": len(selected) / len(rows) if rows else 0.0,
        "sampled_score_delta": sum(row["mean_gain"] for row in rows if row["case_id"] in selected_ids) / len(rows) if rows else 0.0,
        "selected_gain_mean": mean(row["mean_gain"] for row in selected),
        "selected_helpful": buckets["helpful"],
        "selected_harmful": buckets["harmful"],
        "selected_neutral": buckets["neutral"],
        "selected_harm_rate": buckets["harmful"] / len(selected) if selected else 0.0,
    }


def pick_dev_policy(dev_rows):
    rows = []
    for model_name in MODELS:
        for budget in BUDGETS:
            rows.append(summarize_dev_policy(dev_rows, model_name, budget))
    rows.sort(
        key=lambda row: (
            row["sampled_score_delta"] > 0,
            row["selected_harm_rate"] <= 0.20,
            row["sampled_score_delta"],
            -row["selected_harm_rate"],
            row["budget"],
        ),
        reverse=True,
    )
    return rows[0], rows


def load_formal_candidates():
    rows = load_jsonl(FORMAL_CANDIDATES)
    for row in rows:
        row["features"] = {feat: stable_numeric(row.get("features", {}).get(feat, row.get(feat))) for feat in FEATURES}
    return rows


def score_formal_rows(rows, models):
    for row in rows:
        row["train_dev_scores"] = {
            "nb_safe": score_gaussian_nb(row, models["nb_safe"]),
            "gain_corr": score_gain_corr(row, models["gain_corr"]),
        }


def formal_selected_set(policy, rows, locked_policy=None):
    selected = [row for row in rows if row["relaxed_selected"]]
    if policy == "direct_only":
        return set()
    if policy == "relaxed_full_reason":
        return {row["case_id"] for row in selected}
    if policy == "oracle_safe":
        return {row["case_id"] for row in selected if row["label_safe"]}
    if policy == "locked_train_dev":
        model_name = locked_policy["model"]
        budget = locked_policy["budget"]
        ordered = sorted(
            selected,
            key=lambda row: (row["train_dev_scores"][model_name], row["case_id"]),
            reverse=True,
        )
        return {row["case_id"] for row in ordered[:budget]}
    if policy.startswith("train_dev_"):
        body = policy.removeprefix("train_dev_")
        model_name, budget_text = body.rsplit("_", 1)
        budget = int(budget_text.replace("top", ""))
        ordered = sorted(
            selected,
            key=lambda row: (row["train_dev_scores"][model_name], row["case_id"]),
            reverse=True,
        )
        return {row["case_id"] for row in ordered[:budget]}
    raise KeyError(policy)


def summarize_formal_policy(policy, formal_rows, locked_policy=None):
    out = []
    for split in ["test", "test_seen", "test_unseen"]:
        rows = formal_rows if split == "test" else [row for row in formal_rows if row["split"] == split]
        selected_ids = formal_selected_set(policy, rows, locked_policy)
        direct = avg_metrics(row["direct"] for row in rows)
        routed = avg_metrics(row["reason"] if row["case_id"] in selected_ids else row["direct"] for row in rows)
        selected = [row for row in rows if row["case_id"] in selected_ids]
        buckets = Counter("helpful" if row["label_helpful"] else "harmful" if row["label_harmful"] else "neutral" for row in selected)
        out.append(
            {
                "split": split,
                "policy": policy,
                "num_examples": len(rows),
                "pred_reason_count": len(selected),
                "pred_reason_rate": len(selected) / len(rows) if rows else 0.0,
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
        "policy": policy,
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


def render_dev_table(rows):
    lines = [
        "| model | budget | reason rate | sampled delta | harm | helpful/harmful/neutral | selected gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | {row['budget']} | {pct(row['pred_reason_rate'])} | "
            f"{signed(row['sampled_score_delta'])} | {pct(row['selected_harm_rate'])} | "
            f"{row['selected_helpful']}/{row['selected_harmful']}/{row['selected_neutral']} | "
            f"{signed(row['selected_gain_mean'])} |"
        )
    return "\n".join(lines)


def render_formal_table(rows):
    lines = [
        "| policy | pass | reason test/seen/unseen | score test/seen/unseen | harm test/seen/unseen | H/h/N | selected gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | `{row['passes_target']}` | "
            f"{pct(row['test_reason_rate'])}/{pct(row['seen_reason_rate'])}/{pct(row['unseen_reason_rate'])} | "
            f"{signed(row['test_score_delta'])}/{signed(row['seen_score_delta'])}/{signed(row['unseen_score_delta'])} | "
            f"{pct(row['test_harm_rate'])}/{pct(row['seen_harm_rate'])}/{pct(row['unseen_harm_rate'])} | "
            f"{row['test_selected_helpful']}/{row['test_selected_harmful']}/{row['test_selected_neutral']} | "
            f"{signed(row['test_selected_gain_mean'])} |"
        )
    return "\n".join(lines)


def render_formal_diagnostic_table(rows):
    lines = [
        "| policy | reason test/seen/unseen | score test/seen/unseen | harm test/seen/unseen | H/h/N | selected gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | "
            f"{pct(row['test_reason_rate'])}/{pct(row['seen_reason_rate'])}/{pct(row['unseen_reason_rate'])} | "
            f"{signed(row['test_score_delta'])}/{signed(row['seen_score_delta'])}/{signed(row['unseen_score_delta'])} | "
            f"{pct(row['test_harm_rate'])}/{pct(row['seen_harm_rate'])}/{pct(row['unseen_harm_rate'])} | "
            f"{row['test_selected_helpful']}/{row['test_selected_harmful']}/{row['test_selected_neutral']} | "
            f"{signed(row['test_selected_gain_mean'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    locked = payload["locked_policy"]
    locked_formal = next(row for row in payload["formal_leaderboard"] if row["policy"] == "locked_train_dev")
    lines = [
        "# Train/Dev Rewrite Safety Selector Locked Formal Replay",
        "",
        "This experiment fits lightweight safety rankers on train sampled counterfactual labels, selects a model/budget on dev_seen, then replays the locked policy on formal relaxed-selector candidates.",
        "",
        "## Label Summary",
        "",
        "| split | rows | helpful/harmful/neutral | harm | mean gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["label_summary"]:
        lines.append(
            f"| `{row['split']}` | {row['num_rows']} | {row['helpful']}/{row['harmful']}/{row['neutral']} | "
            f"{pct(row['harm_rate'])} | {signed(row['mean_gain'])} |"
        )
    lines.extend(
        [
            "",
            "## Dev Selection",
            "",
            render_dev_table(payload["dev_rankings"]),
            "",
            f"Locked policy: `{locked['model']}` with budget `{locked['budget']}`.",
            "",
            "## Formal Replay",
            "",
            render_formal_table(payload["formal_leaderboard"]),
            "",
            "## Formal Diagnostic Sweep",
            "",
            "These rows use the train-fitted rankers with different budgets on formal candidates. They are diagnostics, not the locked policy selected by dev.",
            "",
            render_formal_diagnostic_table(payload["formal_diagnostics"]),
            "",
            "## Reading",
            "",
            f"- Locked train/dev policy formal score delta: `{locked_formal['test_score_delta']:+.4f}/{locked_formal['seen_score_delta']:+.4f}/{locked_formal['unseen_score_delta']:+.4f}`.",
            f"- Locked train/dev policy formal harm: `{locked_formal['test_harm_rate']:.1%}/{locked_formal['seen_harm_rate']:.1%}/{locked_formal['unseen_harm_rate']:.1%}`.",
            "- This is cleaner than fitting on formal labels, but it still uses sampled train/dev labels from a different distribution than formal relaxed candidates.",
            "",
            "## Artifacts",
            "",
            f"- train candidates: `{payload['train_candidates_jsonl']}`",
            f"- dev candidates: `{payload['dev_candidates_jsonl']}`",
            f"- formal scored candidates: `{payload['formal_scored_jsonl']}`",
            f"- JSON: `{payload['report_json']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(_args):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    train_rows = build_split_rows("train")
    dev_rows = build_split_rows("dev_seen")
    models = fit_models(train_rows)
    score_rows(train_rows, models)
    score_rows(dev_rows, models)
    locked_policy, dev_rankings = pick_dev_policy(dev_rows)

    formal_rows = load_formal_candidates()
    score_formal_rows(formal_rows, models)
    formal_summaries = [
        summarize_formal_policy("oracle_safe", formal_rows),
        summarize_formal_policy("relaxed_full_reason", formal_rows),
        summarize_formal_policy("locked_train_dev", formal_rows, locked_policy),
        summarize_formal_policy("direct_only", formal_rows),
    ]
    formal_diagnostics = []
    for model_name in MODELS:
        for budget in BUDGETS:
            policy = f"train_dev_{model_name}_top{budget:02d}"
            formal_diagnostics.append(summarize_formal_policy(policy, formal_rows))
    formal_diagnostics.sort(
        key=lambda row: (
            row["test_score_delta"],
            -row["test_harm_rate"],
            row["seen_score_delta"],
        ),
        reverse=True,
    )
    formal_summaries.sort(
        key=lambda row: (
            row["policy"] == "oracle_safe",
            row["policy"] == "relaxed_full_reason",
            row["policy"] == "locked_train_dev",
            row["test_score_delta"],
        ),
        reverse=True,
    )

    train_path = OUTPUT_ROOT / "train_candidates.jsonl"
    dev_path = OUTPUT_ROOT / "dev_seen_candidates.jsonl"
    formal_path = OUTPUT_ROOT / "formal_scored_candidates.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)
    write_jsonl(formal_path, formal_rows)

    details = {row["policy"]: {"rows": row.pop("rows")} for row in formal_summaries}
    diagnostic_details = {row["policy"]: {"rows": row.pop("rows")} for row in formal_diagnostics}
    payload = {
        "features": FEATURES,
        "sample_root": SAMPLE_ROOT.as_posix(),
        "label_prefix": LABEL_PREFIX,
        "formal_candidates": FORMAL_CANDIDATES.as_posix(),
        "output_root": OUTPUT_ROOT.as_posix(),
        "train_candidates_jsonl": train_path.as_posix(),
        "dev_candidates_jsonl": dev_path.as_posix(),
        "formal_scored_jsonl": formal_path.as_posix(),
        "label_summary": [
            summarize_label_rows(train_rows, "train"),
            summarize_label_rows(dev_rows, "dev_seen"),
        ],
        "locked_policy": locked_policy,
        "dev_rankings": dev_rankings,
        "formal_leaderboard": formal_summaries,
        "formal_diagnostics": formal_diagnostics,
        "details": details,
        "diagnostic_details": diagnostic_details,
        "report_md": REPORT_MD.as_posix(),
        "report_json": REPORT_JSON.as_posix(),
    }
    write_json(REPORT_JSON, payload)
    write_json(OUTPUT_ROOT / "summary.json", payload)
    write_text(REPORT_MD, render_report(payload))
    print(json.dumps({"report_md": REPORT_MD.as_posix(), "report_json": REPORT_JSON.as_posix()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
