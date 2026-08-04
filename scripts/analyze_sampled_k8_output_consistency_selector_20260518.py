#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
LABEL_SOURCE = "sampled_counterfactual_utility_k8_checkpoint-258"
SAMPLE_ROOT = (
    REPO
    / "outputs/stage2_modular_dualexpert/sampled_counterfactual_utility_20260517/"
    / "sampled_reason_expert_forcedreason_from_noaux_20260517_checkpoint-258"
)
SPLITS = ["train", "dev_seen"]
ROUTES = ["direct", "reason"]
SAMPLE_COUNTS = [1, 2, 4, 8]
SEEDS = [f"seed-{seed}" for seed in range(17, 25)]
METRIC_KEYS = ["argument_f1", "event_f1", "trigger_f1", "score"]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def key_for(row):
    meta = row.get("meta") or {}
    return row.get("sample_key") or meta.get("wnd_id") or meta.get("doc_id")


def mean(values):
    vals = [value for value in values if value is not None and math.isfinite(value)]
    return sum(vals) / len(vals) if vals else 0.0


def std(values):
    vals = [value for value in values if value is not None and math.isfinite(value)]
    if len(vals) <= 1:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((value - mu) ** 2 for value in vals) / len(vals))


def safe_ratio(num, den):
    return num / den if den else 0.0


def route_score(label, route):
    return {
        "trigger_f1": float(label[f"{route}_mean_trigger_f1"]),
        "argument_f1": float(label[f"{route}_mean_argument_f1"]),
        "event_f1": float(label[f"{route}_mean_event_f1"]),
        "score": float(label[f"{route}_mean_score"]),
    }


def compact_events(payload):
    events = payload.get("events") if isinstance(payload, dict) else []
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def normalized_event_parts(row):
    events = []
    for event in compact_events(row.get("predicted") or row.get("final_predicted") or {}):
        event_type = event.get("event_type")
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        trigger_tuple = (event_type, trigger.get("start"), trigger.get("end"))
        arguments = []
        for arg in event.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            arguments.append((arg.get("role"), arg.get("start"), arg.get("end")))
        events.append((trigger_tuple, tuple(sorted(arguments))))
    events.sort()
    triggers = tuple(trigger for trigger, _args in events)
    args = tuple(
        sorted(
            (trigger[0], trigger[1], trigger[2], role, start, end)
            for trigger, arguments in events
            for role, start, end in arguments
        )
    )
    event_types = tuple(sorted(trigger[0] for trigger, _args in events if trigger[0]))
    return {
        "full": tuple(events),
        "trigger": triggers,
        "argument": args,
        "event_type": event_types,
        "event_count": len(events),
        "trigger_count": len(triggers),
        "argument_count": len(args),
        "type_count": len(event_types),
        "empty": 1.0 if not events else 0.0,
    }


def consensus_rate(signatures):
    if not signatures:
        return 0.0
    counts = Counter(signatures)
    return max(counts.values()) / len(signatures)


def unique_rate(signatures):
    if not signatures:
        return 0.0
    return len(set(signatures)) / len(signatures)


def plan_features(text):
    plan = ""
    if "<PLAN>" in text and "</PLAN>" in text:
        plan = text.split("<PLAN>", 1)[1].split("</PLAN>", 1)[0]
    lines = [line.strip() for line in plan.splitlines() if line.strip()]
    return {
        "plan_line_count": float(len(lines)),
        "plan_contrast_count": float(sum(1 for line in lines if "CONTRAST" in line)),
        "plan_role_present_count": float(sum(1 for line in lines if re.search(r"\bROLE\b.*\bpresent\b", line))),
        "plan_role_absent_count": float(sum(1 for line in lines if re.search(r"\bROLE\b.*\babsent\b", line))),
        "plan_type_count": float(sum(1 for line in lines if re.search(r"\bTYPE\b", line))),
    }


def group_features(rows):
    parts = [normalized_event_parts(row) for row in rows]
    valid = [1.0 if row.get("valid_final_json", row.get("valid_json", False)) else 0.0 for row in rows]
    generated_lengths = [len((row.get("generated_text") or row.get("generated_payload") or "").split()) for row in rows]
    plan_rows = [plan_features(row.get("generated_text") or row.get("generated_payload") or "") for row in rows]
    out = {
        "valid_rate": mean(valid),
        "generated_len_mean": mean(generated_lengths),
        "generated_len_std": std(generated_lengths),
    }
    for count_key in ["event_count", "trigger_count", "argument_count", "type_count", "empty"]:
        values = [float(part[count_key]) for part in parts]
        out[f"{count_key}_mean"] = mean(values)
        out[f"{count_key}_std"] = std(values)
    for sig_key in ["full", "trigger", "argument", "event_type"]:
        sigs = [part[sig_key] for part in parts]
        out[f"{sig_key}_consensus"] = consensus_rate(sigs)
        out[f"{sig_key}_unique_rate"] = unique_rate(sigs)
    for key in ["plan_line_count", "plan_contrast_count", "plan_role_present_count", "plan_role_absent_count", "plan_type_count"]:
        out[f"{key}_mean"] = mean([row[key] for row in plan_rows])
        out[f"{key}_std"] = std([row[key] for row in plan_rows])
    return out


def pair_disagreement(direct_rows, reason_rows, sig_key):
    total = 0
    disagree = 0
    direct_by_seed = {row.get("sample_id"): normalized_event_parts(row)[sig_key] for row in direct_rows}
    reason_by_seed = {row.get("sample_id"): normalized_event_parts(row)[sig_key] for row in reason_rows}
    for seed in sorted(set(direct_by_seed) & set(reason_by_seed)):
        total += 1
        if direct_by_seed[seed] != reason_by_seed[seed]:
            disagree += 1
    return safe_ratio(disagree, total)


def load_samples(split, route, sample_count):
    grouped = defaultdict(list)
    for seed in SEEDS[:sample_count]:
        path = SAMPLE_ROOT / split / route / seed / "predictions.jsonl"
        for row in load_jsonl(path):
            grouped[key_for(row)].append(row)
    return grouped


def build_rows(split, sample_count, labels):
    samples = {
        route: load_samples(split, route, sample_count)
        for route in ROUTES
    }
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
                "sample_count": sample_count,
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


def normalize(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [0.0 for _ in values]
    lo = min(finite)
    hi = max(finite)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) if math.isfinite(value) else 0.0 for value in values]


def selector_specs(feature_names):
    focused_features = [
        "direct_consensus_avg",
        "reason_consensus_avg",
        "direct_instability",
        "reason_instability",
        "reason_stability",
        "direct_unstable_reason_stable",
        "reason_consistency_advantage",
        "direct_sparse_reason_rich",
        "reason_plan_signal",
        "route_full_disagreement",
        "route_trigger_disagreement",
        "route_argument_disagreement",
        "route_event_type_disagreement",
        "direct_valid_rate",
        "reason_valid_rate",
        "direct_full_consensus",
        "reason_full_consensus",
        "direct_trigger_consensus",
        "reason_trigger_consensus",
        "direct_argument_consensus",
        "reason_argument_consensus",
        "direct_event_type_consensus",
        "reason_event_type_consensus",
        "direct_argument_count_mean",
        "reason_argument_count_mean",
        "direct_event_count_mean",
        "reason_event_count_mean",
        "direct_argument_count_std",
        "reason_argument_count_std",
        "direct_event_count_std",
        "reason_event_count_std",
        "reason_plan_line_count_mean",
        "reason_plan_contrast_count_mean",
        "reason_plan_role_present_count_mean",
        "reason_plan_role_absent_count_mean",
        "abs_reason_minus_direct_argument_count_mean",
        "abs_reason_minus_direct_event_count_mean",
        "abs_reason_minus_direct_full_consensus",
        "abs_reason_minus_direct_trigger_consensus",
        "abs_reason_minus_direct_argument_consensus",
        "abs_reason_minus_direct_event_type_consensus",
    ]
    feature_names = [name for name in focused_features if name in feature_names]
    specs = []
    for name in feature_names:
        specs.append({"name": f"+{name}", "terms": [(name, 1.0)], "normalize": False})
        specs.append({"name": f"-{name}", "terms": [(name, -1.0)], "normalize": False})
    specs.extend(
        [
            {
                "name": "direct_unstable_reason_stable",
                "terms": [
                    ("direct_instability", 1.0),
                    ("reason_stability", 1.0),
                    ("route_trigger_disagreement", 0.6),
                    ("route_argument_disagreement", 0.4),
                ],
                "normalize": True,
            },
            {
                "name": "direct_sparse_reason_rich",
                "terms": [
                    ("direct_sparse_reason_rich", 1.0),
                    ("reason_plan_signal", 0.4),
                    ("reason_consistency_advantage", 0.4),
                ],
                "normalize": True,
            },
            {
                "name": "reason_stable_not_verbose",
                "terms": [
                    ("reason_stability", 1.0),
                    ("reason_generated_len_std", -0.4),
                    ("reason_full_unique_rate", -0.6),
                    ("direct_instability", 0.5),
                ],
                "normalize": True,
            },
            {
                "name": "route_disagreement_plus_reason_stability",
                "terms": [
                    ("route_full_disagreement", 1.0),
                    ("route_argument_disagreement", 0.6),
                    ("reason_stability", 0.7),
                    ("direct_consensus_avg", -0.4),
                ],
                "normalize": True,
            },
            {
                "name": "reason_rich_consensus",
                "terms": [
                    ("reason_event_count_mean", 0.7),
                    ("reason_argument_count_mean", 0.5),
                    ("reason_consensus_avg", 0.8),
                    ("direct_argument_count_mean", -0.3),
                ],
                "normalize": True,
            },
        ]
    )
    return specs


def score_selector(rows, spec):
    if spec.get("normalize"):
        normalized = []
        for key, _weight in spec["terms"]:
            normalized.append(normalize([float(row["features"].get(key, 0.0)) for row in rows]))
        return {
            row["key"]: sum(weight * normalized[idx][row_idx] for idx, (_key, weight) in enumerate(spec["terms"]))
            for row_idx, row in enumerate(rows)
        }
    return {
        row["key"]: sum(weight * float(row["features"].get(key, 0.0)) for key, weight in spec["terms"])
        for row in rows
    }


def auc(rows, scores, negative_mode):
    positives = [scores[row["key"]] for row in rows if row["is_stable_reason"]]
    if negative_mode == "stable_direct":
        negatives = [scores[row["key"]] for row in rows if row["is_stable_direct"]]
    else:
        negatives = [scores[row["key"]] for row in rows if not row["is_stable_reason"]]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else None


def average_precision(rows, scores):
    positives = {row["key"] for row in rows if row["is_stable_reason"]}
    if not positives:
        return None
    ranked = sorted(rows, key=lambda row: (scores[row["key"]], row["key"]), reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, row in enumerate(ranked, start=1):
        if row["key"] in positives:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(positives)


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate(metrics):
    return {key: mean([metric[key] for metric in metrics]) for key in METRIC_KEYS}


def evaluate_selection(rows, selected):
    direct = aggregate([row["direct_metric"] for row in rows])
    routed = aggregate([row["reason_metric"] if row["key"] in selected else row["direct_metric"] for row in rows])
    stable_reason = {row["key"] for row in rows if row["is_stable_reason"]}
    labels = Counter(row["utility_label"] for row in rows if row["key"] in selected)
    selected_rows = [row for row in rows if row["key"] in selected]
    tp = len(selected & stable_reason)
    fp = len(selected - stable_reason)
    fn = len(stable_reason - selected)
    return {
        "selected_count": len(selected),
        "selected_reason_rate": len(selected) / len(rows) if rows else 0.0,
        "selected_labels": dict(labels),
        "route_vs_stable_reason": prf(tp, fp, fn),
        "selected_avg_sampled_gain": mean([row["sampled_gain"] for row in selected_rows]),
        "sampled_expected_direct": direct,
        "sampled_expected_routed": routed,
        "sampled_expected_routed_minus_direct": {key: routed[key] - direct[key] for key in METRIC_KEYS},
    }


def evaluate_selector(rows, spec, budget_name, budget_count):
    scores = score_selector(rows, spec)
    ranked = sorted(rows, key=lambda row: (scores[row["key"]], row["key"]), reverse=True)
    selected = {row["key"] for row in ranked[:budget_count]}
    out = evaluate_selection(rows, selected)
    out.update(
        {
            "selector": spec["name"],
            "sample_count": rows[0]["sample_count"] if rows else 0,
            "budget": budget_name,
            "budget_count": budget_count,
            "auc_stable_reason_vs_stable_direct": auc(rows, scores, "stable_direct"),
            "auc_stable_reason_vs_all_other": auc(rows, scores, "all_other"),
            "average_precision_all": average_precision(rows, scores),
        }
    )
    return out


def label_summary(rows):
    counts = Counter(row["utility_label"] for row in rows)
    total = len(rows)
    return {
        "num_examples": total,
        "stable_reason": counts["stable_reason"],
        "stable_direct": counts["stable_direct"],
        "ambiguous": counts["ambiguous"],
        "stable_reason_rate": counts["stable_reason"] / total if total else 0.0,
    }


def fmt_float(value, digits=3):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def fmt_delta(delta):
    return (
        f"{delta['argument_f1']:+.4f}/"
        f"{delta['event_f1']:+.4f}/"
        f"{delta['trigger_f1']:+.4f}/"
        f"{delta['score']:+.4f}"
    )


def labels_cell(labels):
    return (
        f"R={labels.get('stable_reason', 0)}, "
        f"D={labels.get('stable_direct', 0)}, "
        f"A={labels.get('ambiguous', 0)}"
    )


def best_rows(evals, split, sample_count, budget):
    rows = evals[split][str(sample_count)][budget]
    return sorted(
        rows,
        key=lambda row: (
            row["sampled_expected_routed_minus_direct"]["score"],
            row["route_vs_stable_reason"]["f1"],
            row["average_precision_all"] or 0.0,
        ),
        reverse=True,
    )


def render_report(payload):
    evals = payload["evaluations"]
    stable_budget = "exact_stable_reason_count"
    train_summary = payload["label_summary"]["train"]["8"]
    dev_summary = payload["label_summary"]["dev_seen"]["8"]
    lines = [
        "# Sampled K8 Output-Consistency Selector Diagnostic",
        "",
        "This report evaluates route selectors that use only repeated direct/reason model outputs, not gold metrics. It tests whether sampling-time consistency can recover K=8 stable_reason examples better than raw prompt/schema features.",
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
        "## Dev Best By Sample Count",
        "",
        "Budget is exactly the dev stable_reason count, so each row routes `15/197` dev examples to Reason.",
        "",
        "| samples/route | selector | AUC vs stable_direct | AP | P/R/F1 | selected labels | delta A/E/T/Score | avg gain |",
        "|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for sample_count in SAMPLE_COUNTS:
        row = best_rows(evals, "dev_seen", sample_count, stable_budget)[0]
        prf_row = row["route_vs_stable_reason"]
        lines.append(
            f"| {sample_count} | `{row['selector']}` | "
            f"{fmt_float(row['auc_stable_reason_vs_stable_direct'])} | "
            f"{fmt_float(row['average_precision_all'])} | "
            f"{prf_row['precision']:.3f}/{prf_row['recall']:.3f}/{prf_row['f1']:.3f} | "
            f"{labels_cell(row['selected_labels'])} | "
            f"{fmt_delta(row['sampled_expected_routed_minus_direct'])} | "
            f"{row['selected_avg_sampled_gain']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Train-Selected Transfer",
            "",
            "Each row chooses the best selector on train at the same sample count and stable-reason budget, then reports that selector on dev_seen.",
            "",
            "| samples/route | train-selected selector | train P/R/F1 | dev P/R/F1 | dev selected labels | dev delta A/E/T/Score |",
            "|---:|---|---:|---:|---|---:|",
        ]
    )
    for sample_count in SAMPLE_COUNTS:
        train_best = best_rows(evals, "train", sample_count, stable_budget)[0]
        dev_match = [
            row for row in evals["dev_seen"][str(sample_count)][stable_budget]
            if row["selector"] == train_best["selector"]
        ][0]
        train_prf = train_best["route_vs_stable_reason"]
        dev_prf = dev_match["route_vs_stable_reason"]
        lines.append(
            f"| {sample_count} | `{train_best['selector']}` | "
            f"{train_prf['precision']:.3f}/{train_prf['recall']:.3f}/{train_prf['f1']:.3f} | "
            f"{dev_prf['precision']:.3f}/{dev_prf['recall']:.3f}/{dev_prf['f1']:.3f} | "
            f"{labels_cell(dev_match['selected_labels'])} | "
            f"{fmt_delta(dev_match['sampled_expected_routed_minus_direct'])} |"
        )
    lines.extend(
        [
            "",
            "## Top Dev Selectors At K=8",
            "",
            "| selector | AUC vs stable_direct | AP | P/R/F1 | selected labels | delta A/E/T/Score | avg gain |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in best_rows(evals, "dev_seen", 8, stable_budget)[:16]:
        prf_row = row["route_vs_stable_reason"]
        lines.append(
            f"| `{row['selector']}` | "
            f"{fmt_float(row['auc_stable_reason_vs_stable_direct'])} | "
            f"{fmt_float(row['average_precision_all'])} | "
            f"{prf_row['precision']:.3f}/{prf_row['recall']:.3f}/{prf_row['f1']:.3f} | "
            f"{labels_cell(row['selected_labels'])} | "
            f"{fmt_delta(row['sampled_expected_routed_minus_direct'])} | "
            f"{row['selected_avg_sampled_gain']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "If train-selected output-consistency features transfer better than raw prompt/schema features, the next route experiment should expose sampled self-consistency evidence to a selector or train a lightweight scorer on those features. If they do not transfer, then K=8 labels are mostly outcome-oracle supervision and are not recoverable from inexpensive pre-execution signals.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['output_json']}`",
            f"- sample root: `{SAMPLE_ROOT.as_posix()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    labels = {
        split: load_jsonl(REPO / f"data/stage2_adaptive_datasets/labels/{DATA_PREFIX}_{LABEL_SOURCE}_{split}_labels.jsonl")
        for split in SPLITS
    }
    rows = {
        split: {
            str(sample_count): build_rows(split, sample_count, labels[split])
            for sample_count in SAMPLE_COUNTS
        }
        for split in SPLITS
    }
    feature_names = sorted(rows["train"]["8"][0]["features"])
    specs = selector_specs(feature_names)
    budgets = {
        split: {
            str(sample_count): {
                "exact_stable_reason_count": sum(1 for row in split_rows if row["is_stable_reason"]),
            }
            for sample_count, split_rows in sample_rows.items()
        }
        for split, sample_rows in rows.items()
    }
    evaluations = defaultdict(lambda: defaultdict(dict))
    for split, sample_rows in rows.items():
        for sample_count, split_rows in sample_rows.items():
            for budget_name, budget_count in budgets[split][sample_count].items():
                evaluations[split][sample_count][budget_name] = [
                    evaluate_selector(split_rows, spec, budget_name, budget_count)
                    for spec in specs
                ]
    payload = {
        "inputs": {
            "sample_root": SAMPLE_ROOT.as_posix(),
            "label_source": LABEL_SOURCE,
            "sample_counts": SAMPLE_COUNTS,
        },
        "label_summary": {
            split: {
                sample_count: label_summary(split_rows)
                for sample_count, split_rows in sample_rows.items()
            }
            for split, sample_rows in rows.items()
        },
        "budgets": budgets,
        "feature_names": feature_names,
        "evaluations": evaluations,
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
        default="reports/artifacts/2026-05-18_stage2_sampled_k8_output_consistency_selector_checkpoint258.json",
    )
    parser.add_argument(
        "--output_md",
        default="reports/2026-05-18_stage2_sampled_k8_output_consistency_selector_checkpoint258.md",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
