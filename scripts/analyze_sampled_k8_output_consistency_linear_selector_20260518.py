#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.analyze_sampled_k8_output_consistency_selector_20260518 import (  # noqa: E402
    DATA_PREFIX,
    LABEL_SOURCE,
    METRIC_KEYS,
    SAMPLE_COUNTS,
    auc,
    average_precision,
    build_rows,
    evaluate_selection,
    fmt_delta,
    label_summary,
    labels_cell,
    load_jsonl,
    mean,
)


L2_VALUES = [0.1, 1.0, 10.0, 100.0]
MODEL_SPECS = [
    {
        "name": "binary_confident_balanced",
        "target": "binary",
        "train_filter": "confident",
        "balanced_weights": True,
    },
    {
        "name": "binary_allneg_balanced",
        "target": "binary",
        "train_filter": "all",
        "balanced_weights": True,
    },
    {
        "name": "gain_regression_all",
        "target": "gain",
        "train_filter": "all",
        "balanced_weights": False,
    },
    {
        "name": "gain_regression_confident",
        "target": "gain",
        "train_filter": "confident",
        "balanced_weights": False,
    },
]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def feature_names_from(rows):
    names = sorted(rows[0]["features"])
    return [
        name for name in names
        if not name.endswith("_std") or not name.startswith("reason_plan")
    ]


def row_target(row, model_spec):
    if model_spec["target"] == "gain":
        return float(row["sampled_gain"])
    return 1.0 if row["is_stable_reason"] else 0.0


def row_in_train(row, model_spec):
    if model_spec["train_filter"] == "all":
        return True
    return row["utility_label"] in {"stable_reason", "stable_direct"}


def matrix(rows, feature_names):
    return np.asarray(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=np.float64,
    )


def standardize_train(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


def apply_standardize(X, mu, sigma):
    return (X - mu) / sigma


def sample_weights(y, model_spec):
    if not model_spec.get("balanced_weights"):
        return np.ones_like(y, dtype=np.float64)
    pos = y > 0.5
    neg = ~pos
    weights = np.ones_like(y, dtype=np.float64)
    if pos.any():
        weights[pos] = len(y) / (2.0 * pos.sum())
    if neg.any():
        weights[neg] = len(y) / (2.0 * neg.sum())
    return weights


def fit_ridge(X, y, weights, l2):
    X_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    sqrt_w = np.sqrt(weights)[:, None]
    Xw = X_aug * sqrt_w
    yw = y * sqrt_w[:, 0]
    penalty = np.eye(X_aug.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    lhs = Xw.T @ Xw + penalty
    rhs = Xw.T @ yw
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    return beta


def score_rows(rows, feature_names, mu, sigma, beta):
    X = apply_standardize(matrix(rows, feature_names), mu, sigma)
    X_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    scores = X_aug @ beta
    return {row["key"]: float(score) for row, score in zip(rows, scores)}


def ranked_selection(rows, scores, count):
    ranked = sorted(rows, key=lambda row: (scores[row["key"]], row["key"]), reverse=True)
    return {row["key"] for row in ranked[:count]}


def selected_by_threshold(rows, scores, threshold):
    return {row["key"] for row in rows if scores[row["key"]] >= threshold}


def prf_for_scores(rows, scores):
    return {
        "auc_stable_reason_vs_stable_direct": auc(rows, scores, "stable_direct"),
        "auc_stable_reason_vs_all_other": auc(rows, scores, "all_other"),
        "average_precision_all": average_precision(rows, scores),
    }


def evaluate_model(rows, scores, budget_name, budget_count):
    out = evaluate_selection(rows, ranked_selection(rows, scores, budget_count))
    out.update(prf_for_scores(rows, scores))
    out["budget"] = budget_name
    out["budget_count"] = budget_count
    return out


def evaluate_threshold(rows, scores, threshold, budget_name):
    out = evaluate_selection(rows, selected_by_threshold(rows, scores, threshold))
    out.update(prf_for_scores(rows, scores))
    out["budget"] = budget_name
    out["threshold"] = float(threshold)
    out["budget_count"] = out["selected_count"]
    return out


def stable_count(rows):
    return sum(1 for row in rows if row["is_stable_reason"])


def train_rate_count(train_rows, target_rows):
    return round(len(target_rows) * stable_count(train_rows) / len(train_rows))


def threshold_for_top_count(rows, scores, count):
    ranked_scores = sorted((scores[row["key"]] for row in rows), reverse=True)
    if not ranked_scores:
        return float("inf")
    if count <= 0:
        return ranked_scores[0] + 1e-9
    if count >= len(ranked_scores):
        return ranked_scores[-1]
    return ranked_scores[count - 1]


def train_one(sample_count, train_rows, dev_rows, feature_names, model_spec, l2):
    fit_rows = [row for row in train_rows if row_in_train(row, model_spec)]
    y = np.asarray([row_target(row, model_spec) for row in fit_rows], dtype=np.float64)
    X = matrix(fit_rows, feature_names)
    mu, sigma = standardize_train(X)
    Xs = apply_standardize(X, mu, sigma)
    beta = fit_ridge(Xs, y, sample_weights(y, model_spec), l2)
    train_scores = score_rows(train_rows, feature_names, mu, sigma, beta)
    dev_scores = score_rows(dev_rows, feature_names, mu, sigma, beta)

    train_exact_count = stable_count(train_rows)
    dev_exact_count = stable_count(dev_rows)
    dev_train_rate_count = train_rate_count(train_rows, dev_rows)
    train_threshold = threshold_for_top_count(train_rows, train_scores, train_exact_count)

    result = {
        "model": f"{model_spec['name']}_l2{l2:g}",
        "model_family": model_spec["name"],
        "sample_count": sample_count,
        "l2": l2,
        "num_fit_rows": len(fit_rows),
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "train": {
            "exact_stable_reason_count": evaluate_model(
                train_rows,
                train_scores,
                "exact_stable_reason_count",
                train_exact_count,
            ),
            "threshold_from_train_exact_count": evaluate_threshold(
                train_rows,
                train_scores,
                train_threshold,
                "threshold_from_train_exact_count",
            ),
        },
        "dev_seen": {
            "exact_stable_reason_count": evaluate_model(
                dev_rows,
                dev_scores,
                "exact_stable_reason_count",
                dev_exact_count,
            ),
            "train_rate_budget": evaluate_model(
                dev_rows,
                dev_scores,
                "train_rate_budget",
                dev_train_rate_count,
            ),
            "threshold_from_train_exact_count": evaluate_threshold(
                dev_rows,
                dev_scores,
                train_threshold,
                "threshold_from_train_exact_count",
            ),
        },
    }
    return result


def add_model_fields(row, result, split, budget):
    out = dict(row)
    out.update(
        {
            "model": result["model"],
            "model_family": result["model_family"],
            "sample_count": result["sample_count"],
            "l2": result["l2"],
            "split": split,
            "budget": budget,
        }
    )
    return out


def sort_key(row):
    delta = row["sampled_expected_routed_minus_direct"]
    prf = row["route_vs_stable_reason"]
    return (
        delta["score"],
        prf["f1"],
        row.get("average_precision_all") or 0.0,
    )


def best_by_train(results, sample_count):
    candidates = [
        result for result in results
        if result["sample_count"] == sample_count
    ]
    return max(candidates, key=lambda result: sort_key(result["train"]["exact_stable_reason_count"]))


def best_by_dev(results, sample_count, budget):
    candidates = [
        result for result in results
        if result["sample_count"] == sample_count
    ]
    return max(candidates, key=lambda result: sort_key(result["dev_seen"][budget]))


def labels_summary_cell(labels):
    return (
        f"R={labels.get('stable_reason', 0)}, "
        f"D={labels.get('stable_direct', 0)}, "
        f"A={labels.get('ambiguous', 0)}"
    )


def render_row(row):
    prf = row["route_vs_stable_reason"]
    return (
        f"{prf['precision']:.3f}/{prf['recall']:.3f}/{prf['f1']:.3f}",
        labels_summary_cell(row["selected_labels"]),
        fmt_delta(row["sampled_expected_routed_minus_direct"]),
        f"{row['selected_avg_sampled_gain']:.4f}",
    )


def render_report(payload):
    results = payload["results"]
    train_summary = payload["label_summary"]["train"]["8"]
    dev_summary = payload["label_summary"]["dev_seen"]["8"]
    lines = [
        "# Sampled K8 Output-Consistency Linear Selector",
        "",
        "This report trains lightweight ridge linear scorers on output-consistency features. It is not an LLM training run; it tests whether the sampling evidence can support a learned selector.",
        "",
        "## Label Summary",
        "",
        "| split | n | stable_reason | stable_direct | ambiguous | stable_reason rate |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| train | {train_summary['num_examples']} | {train_summary['stable_reason']} | "
            f"{train_summary['stable_direct']} | {train_summary['ambiguous']} | {train_summary['stable_reason_rate']:.1%} |"
        ),
        (
            f"| dev_seen | {dev_summary['num_examples']} | {dev_summary['stable_reason']} | "
            f"{dev_summary['stable_direct']} | {dev_summary['ambiguous']} | {dev_summary['stable_reason_rate']:.1%} |"
        ),
        "",
        "## Train-Selected Transfer",
        "",
        "Model selection is done on train exact stable_reason budget. The same model is then evaluated on dev with both exact dev budget and train-rate budget.",
        "",
        "| samples/route | train-selected model | train P/R/F1 | dev exact P/R/F1 | dev exact labels | dev exact delta A/E/T/Score | dev train-rate labels | dev train-rate delta A/E/T/Score |",
        "|---:|---|---:|---:|---|---:|---|---:|",
    ]
    for sample_count in SAMPLE_COUNTS:
        result = best_by_train(results, sample_count)
        train_prf, _train_labels, _train_delta, _train_gain = render_row(result["train"]["exact_stable_reason_count"])
        dev_exact = result["dev_seen"]["exact_stable_reason_count"]
        dev_rate = result["dev_seen"]["train_rate_budget"]
        dev_prf, dev_labels, dev_delta, _dev_gain = render_row(dev_exact)
        _rate_prf, rate_labels, rate_delta, _rate_gain = render_row(dev_rate)
        lines.append(
            f"| {sample_count} | `{result['model']}` | {train_prf} | {dev_prf} | "
            f"{dev_labels} | {dev_delta} | {rate_labels} | {rate_delta} |"
        )

    lines.extend(
        [
            "",
            "## Dev-Oracle Model Selection",
            "",
            "This table is diagnostic only because it selects the model on dev.",
            "",
            "| samples/route | budget | dev-selected model | P/R/F1 | selected labels | delta A/E/T/Score | avg gain | AP |",
            "|---:|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for sample_count in SAMPLE_COUNTS:
        for budget in ["exact_stable_reason_count", "train_rate_budget"]:
            result = best_by_dev(results, sample_count, budget)
            row = result["dev_seen"][budget]
            prf_text, label_text, delta_text, gain_text = render_row(row)
            lines.append(
                f"| {sample_count} | `{budget}` | `{result['model']}` | {prf_text} | "
                f"{label_text} | {delta_text} | {gain_text} | "
                f"{(row.get('average_precision_all') or 0.0):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "If the train-selected models beat the hand-written consistency rules, the next route experiment should distill these features into a learned selector. If not, keep the consistency features as a diagnostic and avoid another plain route-tag LLM branch.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['output_json']}`",
            f"- report: `{payload['output_md']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    labels = {
        split: load_jsonl(REPO / f"data/stage2_adaptive_datasets/labels/{DATA_PREFIX}_{LABEL_SOURCE}_{split}_labels.jsonl")
        for split in ["train", "dev_seen"]
    }
    rows = {
        split: {
            str(sample_count): build_rows(split, sample_count, labels[split])
            for sample_count in SAMPLE_COUNTS
        }
        for split in ["train", "dev_seen"]
    }
    results = []
    for sample_count in SAMPLE_COUNTS:
        train_rows = rows["train"][str(sample_count)]
        dev_rows = rows["dev_seen"][str(sample_count)]
        feature_names = feature_names_from(train_rows)
        for model_spec in MODEL_SPECS:
            for l2 in L2_VALUES:
                results.append(train_one(sample_count, train_rows, dev_rows, feature_names, model_spec, l2))

    payload = {
        "inputs": {
            "label_source": LABEL_SOURCE,
            "model_specs": MODEL_SPECS,
            "l2_values": L2_VALUES,
            "sample_counts": SAMPLE_COUNTS,
        },
        "label_summary": {
            split: {
                sample_count: label_summary(split_rows)
                for sample_count, split_rows in split_map.items()
            }
            for split, split_map in rows.items()
        },
        "result_count": len(results),
        "results": results,
        "output_json": args.output_json,
        "output_md": args.output_md,
    }
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), render_report(payload))
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_json",
        default="reports/artifacts/2026-05-18_stage2_sampled_k8_output_consistency_linear_selector_checkpoint258.json",
    )
    parser.add_argument(
        "--output_md",
        default="reports/2026-05-18_stage2_sampled_k8_output_consistency_linear_selector_checkpoint258.md",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
