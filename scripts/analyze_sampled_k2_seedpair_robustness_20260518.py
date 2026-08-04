#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.analyze_sampled_k8_output_consistency_linear_selector_20260518 import (  # noqa: E402
    L2_VALUES,
    MODEL_SPECS,
    feature_names_from,
    train_one,
)
from scripts.analyze_sampled_k8_output_consistency_selector_20260518 import (  # noqa: E402
    DATA_PREFIX,
    LABEL_SOURCE,
    METRIC_KEYS,
    ROUTES,
    SAMPLE_ROOT,
    auc,
    average_precision,
    evaluate_selection,
    fmt_delta,
    group_features,
    labels_cell,
    load_jsonl,
    mean,
    pair_disagreement,
    route_score,
    score_selector,
    selector_specs,
)
from src.stage2_analysis.build_sampled_counterfactual_utility_labels import (  # noqa: E402
    classify,
    metric_row,
)


SEED_PAIRS = [
    ("17_18", [17, 18]),
    ("19_20", [19, 20]),
    ("21_22", [21, 22]),
    ("23_24", [23, 24]),
]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def label_path(split: str):
    return REPO / f"data/stage2_adaptive_datasets/labels/{DATA_PREFIX}_{LABEL_SOURCE}_{split}_labels.jsonl"


def key_for(row):
    meta = row.get("meta") or {}
    return row.get("sample_key") or meta.get("wnd_id") or meta.get("doc_id")


def load_samples(split: str, route: str, seeds):
    grouped = defaultdict(list)
    for seed in seeds:
        path = SAMPLE_ROOT / split / route / f"seed-{seed}" / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in load_jsonl(path):
            grouped[key_for(row)].append(row)
    return grouped


def build_feature_rows(split: str, seeds, labels):
    samples = {route: load_samples(split, route, seeds) for route in ROUTES}
    rows = []
    for label in labels:
        key = label["wnd_id"]
        if key not in samples["direct"] or key not in samples["reason"]:
            raise KeyError(f"missing sampled rows for {split}/{key}")
        direct_rows = samples["direct"][key]
        reason_rows = samples["reason"][key]
        direct_feat = group_features(direct_rows)
        reason_feat = group_features(reason_rows)
        features = {}
        for prefix, feat in [("direct", direct_feat), ("reason", reason_feat)]:
            for name, value in feat.items():
                features[f"{prefix}_{name}"] = value
        for name in sorted(set(direct_feat) & set(reason_feat)):
            features[f"reason_minus_direct_{name}"] = reason_feat[name] - direct_feat[name]
            features[f"abs_reason_minus_direct_{name}"] = abs(reason_feat[name] - direct_feat[name])

        direct_consensus = mean(
            [
                direct_feat["full_consensus"],
                direct_feat["trigger_consensus"],
                direct_feat["argument_consensus"],
                direct_feat["event_type_consensus"],
            ]
        )
        reason_consensus = mean(
            [
                reason_feat["full_consensus"],
                reason_feat["trigger_consensus"],
                reason_feat["argument_consensus"],
                reason_feat["event_type_consensus"],
            ]
        )
        direct_count_instability = mean(
            [
                direct_feat["event_count_std"],
                direct_feat["trigger_count_std"],
                direct_feat["argument_count_std"],
                direct_feat["type_count_std"],
            ]
        )
        reason_count_instability = mean(
            [
                reason_feat["event_count_std"],
                reason_feat["trigger_count_std"],
                reason_feat["argument_count_std"],
                reason_feat["type_count_std"],
            ]
        )
        features.update(
            {
                "direct_consensus_avg": direct_consensus,
                "reason_consensus_avg": reason_consensus,
                "direct_instability": (1.0 - direct_consensus) + direct_count_instability,
                "reason_instability": (1.0 - reason_consensus) + reason_count_instability,
                "reason_stability": reason_feat["valid_rate"] + reason_consensus - reason_count_instability,
                "direct_unstable_reason_stable": (1.0 - direct_consensus) + direct_count_instability + reason_consensus,
                "reason_consistency_advantage": reason_consensus - direct_consensus - reason_count_instability,
                "direct_sparse_reason_rich": (
                    reason_feat["event_count_mean"]
                    + 0.35 * reason_feat["argument_count_mean"]
                    - direct_feat["event_count_mean"]
                    - 0.35 * direct_feat["argument_count_mean"]
                ),
                "reason_plan_signal": (
                    reason_feat["plan_contrast_count_mean"]
                    + reason_feat["plan_role_present_count_mean"]
                    - 0.2 * reason_feat["plan_role_absent_count_mean"]
                ),
                "route_full_disagreement": pair_disagreement(direct_rows, reason_rows, "full"),
                "route_trigger_disagreement": pair_disagreement(direct_rows, reason_rows, "trigger"),
                "route_argument_disagreement": pair_disagreement(direct_rows, reason_rows, "argument"),
                "route_event_type_disagreement": pair_disagreement(direct_rows, reason_rows, "event_type"),
            }
        )
        utility_label = label["utility_label"]
        rows.append(
            {
                "key": key,
                "split": split,
                "sample_count": len(seeds),
                "seed_pair": "_".join(str(seed) for seed in seeds),
                "utility_label": utility_label,
                "is_stable_reason": utility_label == "stable_reason",
                "is_stable_direct": utility_label == "stable_direct",
                "sampled_gain": float(label["mean_gain"]),
                "direct_metric": route_score(label, "direct"),
                "reason_metric": route_score(label, "reason"),
                "features": features,
            }
        )
    return rows


def default_label_args():
    return SimpleNamespace(
        pair_score_margin=0.2,
        trigger_harm_tolerance=0.02,
        reason_valid_rate_min=0.875,
        mean_gain_min=0.35,
        p_win_min=0.70,
        p_trigger_noharm_min=0.75,
        direct_reason_valid_rate_max=0.75,
        direct_mean_gain_max=-0.20,
        direct_p_win_max=0.25,
        direct_p_trigger_noharm_min=0.50,
    )


def build_pair_labels(split: str, seeds, k8_labels):
    samples = {route: load_samples(split, route, seeds) for route in ROUTES}
    args = default_label_args()
    labels = []
    for k8_label in k8_labels:
        key = k8_label["wnd_id"]
        direct_rows = samples["direct"][key]
        reason_rows = samples["reason"][key]
        pair_label = classify(
            [metric_row(row) for row in direct_rows],
            [metric_row(row) for row in reason_rows],
            args,
        )
        pair_label.update(
            {
                "wnd_id": key,
                "label_source": f"{LABEL_SOURCE}_seedpair_{'_'.join(str(seed) for seed in seeds)}",
                "expected_samples_per_route": len(seeds),
                "k8_utility_label": k8_label["utility_label"],
                "k8_mean_gain": k8_label["mean_gain"],
                "direct_sample_ids": sorted(str(row.get("sample_id")) for row in direct_rows),
                "reason_sample_ids": sorted(str(row.get("sample_id")) for row in reason_rows),
            }
        )
        labels.append(pair_label)
    return labels


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def pair_label_summary(feature_rows, pair_labels):
    pair_by_id = {row["wnd_id"]: row for row in pair_labels}
    pair_counts = Counter(row["utility_label"] for row in pair_labels)
    selected = {row["wnd_id"] for row in pair_labels if row["utility_label"] == "stable_reason"}
    eval_row = evaluate_selection(feature_rows, selected)
    k8_positive = {row["key"] for row in feature_rows if row["is_stable_reason"]}
    tp = len(selected & k8_positive)
    fp = len(selected - k8_positive)
    fn = len(k8_positive - selected)
    eval_row.update(
        {
            "pair_label_counts": dict(pair_counts),
            "pair_stable_reason_rate": pair_counts["stable_reason"] / len(pair_labels) if pair_labels else 0.0,
            "pair_vs_k8_stable_reason": prf(tp, fp, fn),
            "selected_pair_stable_reason": len(selected),
            "pair_label_by_id_preview": {
                key: pair_by_id[key]["utility_label"]
                for key in sorted(pair_by_id)[:5]
            },
        }
    )
    return eval_row


def stable_count(rows):
    return sum(1 for row in rows if row["is_stable_reason"])


def ranked_selection(rows, scores, count):
    ranked = sorted(rows, key=lambda row: (scores[row["key"]], row["key"]), reverse=True)
    return {row["key"] for row in ranked[:count]}


def sort_eval(row):
    delta = row["sampled_expected_routed_minus_direct"]
    prf_row = row["route_vs_stable_reason"]
    return (
        delta["score"],
        prf_row["f1"],
        row.get("average_precision_all") or 0.0,
    )


def evaluate_selector_spec(rows, spec, budget_name, budget_count):
    scores = score_selector(rows, spec)
    out = evaluate_selection(rows, ranked_selection(rows, scores, budget_count))
    out.update(
        {
            "selector": spec["name"],
            "budget": budget_name,
            "budget_count": budget_count,
            "auc_stable_reason_vs_stable_direct": auc(rows, scores, "stable_direct"),
            "auc_stable_reason_vs_all_other": auc(rows, scores, "all_other"),
            "average_precision_all": average_precision(rows, scores),
        }
    )
    return out


def hand_selector_transfer(train_rows, dev_rows):
    feature_names = sorted(train_rows[0]["features"]) if train_rows else []
    specs = selector_specs(feature_names)
    train_budget = stable_count(train_rows)
    dev_budget = stable_count(dev_rows)
    candidates = []
    for spec in specs:
        train_eval = evaluate_selector_spec(train_rows, spec, "exact_stable_reason_count", train_budget)
        dev_eval = evaluate_selector_spec(dev_rows, spec, "exact_stable_reason_count", dev_budget)
        candidates.append({"selector": spec["name"], "train": train_eval, "dev_seen": dev_eval})
    train_selected = max(candidates, key=lambda row: sort_eval(row["train"]))
    dev_oracle = max(candidates, key=lambda row: sort_eval(row["dev_seen"]))
    return {
        "num_selectors": len(candidates),
        "train_selected": train_selected,
        "dev_oracle": dev_oracle,
    }


def linear_selector_transfer(train_rows, dev_rows):
    feature_names = feature_names_from(train_rows)
    results = []
    for model_spec in MODEL_SPECS:
        for l2 in L2_VALUES:
            results.append(train_one(2, train_rows, dev_rows, feature_names, model_spec, l2))
    train_selected = max(results, key=lambda row: sort_eval(row["train"]["exact_stable_reason_count"]))
    dev_oracle = max(results, key=lambda row: sort_eval(row["dev_seen"]["exact_stable_reason_count"]))
    return {
        "num_models": len(results),
        "feature_count": len(feature_names),
        "train_selected": train_selected,
        "dev_oracle": dev_oracle,
    }


def label_counts(rows):
    counts = Counter(row["utility_label"] for row in rows)
    total = len(rows)
    return {
        "num_examples": total,
        "stable_reason": counts["stable_reason"],
        "stable_direct": counts["stable_direct"],
        "ambiguous": counts["ambiguous"],
        "stable_reason_rate": counts["stable_reason"] / total if total else 0.0,
    }


def route_text(row):
    route = row["route_vs_stable_reason"]
    return f"{route['precision']:.3f}/{route['recall']:.3f}/{route['f1']:.3f}"


def pair_counts_text(counts):
    return (
        f"R={counts.get('stable_reason', 0)}, "
        f"D={counts.get('stable_direct', 0)}, "
        f"A={counts.get('ambiguous', 0)}"
    )


def render_label_table(payload, split):
    lines = [
        f"### {split}",
        "",
        "| seed pair | K2 labels | K2 stable_reason rate | K2 vs K8 P/R/F1 | selected K8 labels | delta A/E/T/Score |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for pair_name, _seeds in SEED_PAIRS:
        row = payload["pairs"][pair_name]["pair_label_policy"][split]
        prf_row = row["pair_vs_k8_stable_reason"]
        lines.append(
            f"| `{pair_name}` | {pair_counts_text(row['pair_label_counts'])} | "
            f"{row['pair_stable_reason_rate']:.1%} | "
            f"{prf_row['precision']:.3f}/{prf_row['recall']:.3f}/{prf_row['f1']:.3f} | "
            f"{labels_cell(row['selected_labels'])} | "
            f"{fmt_delta(row['sampled_expected_routed_minus_direct'])} |"
        )
    return lines


def render_transfer_table(payload, selector_kind):
    title = "Hand-Written Evidence Selector" if selector_kind == "hand_selector" else "Linear Evidence Selector"
    lines = [
        f"## {title} Transfer",
        "",
        "Selection is done on train with exact K8 stable_reason budget, then transferred to dev_seen with exact dev stable_reason budget.",
        "",
        "| seed pair | train-selected selector/model | train P/R/F1 | dev P/R/F1 | dev selected K8 labels | dev delta A/E/T/Score | dev-oracle P/R/F1 | dev-oracle delta A/E/T/Score |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for pair_name, _seeds in SEED_PAIRS:
        block = payload["pairs"][pair_name][selector_kind]
        selected = block["train_selected"]
        oracle = block["dev_oracle"]
        if selector_kind == "hand_selector":
            name = selected["selector"]
            train_eval = selected["train"]
            dev_eval = selected["dev_seen"]
            oracle_eval = oracle["dev_seen"]
        else:
            name = selected["model"]
            train_eval = selected["train"]["exact_stable_reason_count"]
            dev_eval = selected["dev_seen"]["exact_stable_reason_count"]
            oracle_eval = oracle["dev_seen"]["exact_stable_reason_count"]
        lines.append(
            f"| `{pair_name}` | `{name}` | "
            f"{route_text(train_eval)} | {route_text(dev_eval)} | "
            f"{labels_cell(dev_eval['selected_labels'])} | "
            f"{fmt_delta(dev_eval['sampled_expected_routed_minus_direct'])} | "
            f"{route_text(oracle_eval)} | "
            f"{fmt_delta(oracle_eval['sampled_expected_routed_minus_direct'])} |"
        )
    return lines


def summarize_scores(payload):
    label_scores = [
        payload["pairs"][pair_name]["pair_label_policy"]["dev_seen"]["sampled_expected_routed_minus_direct"]["score"]
        for pair_name, _seeds in SEED_PAIRS
    ]
    hand_scores = [
        payload["pairs"][pair_name]["hand_selector"]["train_selected"]["dev_seen"]["sampled_expected_routed_minus_direct"]["score"]
        for pair_name, _seeds in SEED_PAIRS
    ]
    linear_scores = [
        payload["pairs"][pair_name]["linear_selector"]["train_selected"]["dev_seen"]["exact_stable_reason_count"]["sampled_expected_routed_minus_direct"]["score"]
        for pair_name, _seeds in SEED_PAIRS
    ]
    return {
        "pair_label_dev_score_delta_min": min(label_scores),
        "pair_label_dev_score_delta_mean": mean(label_scores),
        "hand_selector_dev_score_delta_min": min(hand_scores),
        "hand_selector_dev_score_delta_mean": mean(hand_scores),
        "linear_selector_dev_score_delta_min": min(linear_scores),
        "linear_selector_dev_score_delta_mean": mean(linear_scores),
    }


def render_report(payload):
    score_summary = summarize_scores(payload)
    lines = [
        "# Sampled K2 Seed-Pair Robustness Diagnostic",
        "",
        "This diagnostic reuses the completed K=8 direct/reason samples and asks whether the K=2 evidence result depends on the specific seed pair. The supervision target remains the K=8 stable_reason/stable_direct label when evaluating evidence selectors; a separate table shows what would happen if labels were mined from only the two samples.",
        "",
        "## Inputs",
        "",
        f"- label source: `{LABEL_SOURCE}`",
        f"- sample root: `{SAMPLE_ROOT.relative_to(REPO)}`",
        f"- seed pairs: `{', '.join(pair for pair, _ in SEED_PAIRS)}`",
        "",
        "## K8 Label Summary",
        "",
        "| split | n | stable_reason | stable_direct | ambiguous | stable_reason rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "dev_seen"]:
        row = payload["k8_label_summary"][split]
        lines.append(
            f"| `{split}` | {row['num_examples']} | {row['stable_reason']} | "
            f"{row['stable_direct']} | {row['ambiguous']} | {row['stable_reason_rate']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## K2 Label-Mining Robustness",
            "",
            "These rows apply the same confident-only thresholds to only two samples per route. Deltas are still measured against the K=8 mean direct/reason metrics so the rows are comparable.",
            "",
        ]
    )
    lines.extend(render_label_table(payload, "train"))
    lines.append("")
    lines.extend(render_label_table(payload, "dev_seen"))
    lines.append("")
    lines.extend(render_transfer_table(payload, "hand_selector"))
    lines.append("")
    lines.extend(render_transfer_table(payload, "linear_selector"))
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- K2-only label mining dev score delta: min `{score_summary['pair_label_dev_score_delta_min']:+.4f}`, mean `{score_summary['pair_label_dev_score_delta_mean']:+.4f}`.",
            f"- train-selected hand selector dev score delta: min `{score_summary['hand_selector_dev_score_delta_min']:+.4f}`, mean `{score_summary['hand_selector_dev_score_delta_mean']:+.4f}`.",
            f"- train-selected linear selector dev score delta: min `{score_summary['linear_selector_dev_score_delta_min']:+.4f}`, mean `{score_summary['linear_selector_dev_score_delta_mean']:+.4f}`.",
            "- If K2-only label mining is volatile but K2 evidence selectors stay positive, the safer path is still K8-label supervision with cheap K2 evidence at inference time.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['output_json']}`",
            f"- report: `{payload['output_md']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    k8_labels = {split: load_jsonl(label_path(split)) for split in ["train", "dev_seen"]}
    payload = {
        "inputs": {
            "label_source": LABEL_SOURCE,
            "sample_root": SAMPLE_ROOT.as_posix(),
            "seed_pairs": [{"name": name, "seeds": seeds} for name, seeds in SEED_PAIRS],
            "label_thresholds": vars(default_label_args()),
        },
        "k8_label_summary": {
            split: label_counts(
                [
                    {
                        "utility_label": row["utility_label"],
                    }
                    for row in rows
                ]
            )
            for split, rows in k8_labels.items()
        },
        "pairs": {},
        "output_json": args.output_json,
        "output_md": args.output_md,
    }

    for pair_name, seeds in SEED_PAIRS:
        feature_rows = {
            split: build_feature_rows(split, seeds, k8_labels[split])
            for split in ["train", "dev_seen"]
        }
        pair_labels = {
            split: build_pair_labels(split, seeds, k8_labels[split])
            for split in ["train", "dev_seen"]
        }
        payload["pairs"][pair_name] = {
            "seeds": seeds,
            "feature_label_summary": {
                split: label_counts(rows)
                for split, rows in feature_rows.items()
            },
            "pair_label_policy": {
                split: pair_label_summary(feature_rows[split], pair_labels[split])
                for split in ["train", "dev_seen"]
            },
            "hand_selector": hand_selector_transfer(feature_rows["train"], feature_rows["dev_seen"]),
            "linear_selector": linear_selector_transfer(feature_rows["train"], feature_rows["dev_seen"]),
        }

    payload["score_summary"] = summarize_scores(payload)
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), render_report(payload))
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md, "score_summary": payload["score_summary"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_json",
        default="outputs/stage2_analysis/sampledk2_seedpair_robustness_20260518/results.json",
    )
    parser.add_argument(
        "--output_md",
        default="reports/2026-05-18_stage2_sampled_k2_seedpair_robustness_checkpoint258.md",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
