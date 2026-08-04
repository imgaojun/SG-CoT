import argparse
from collections import Counter, defaultdict
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import (  # noqa: E402
    DIRECT_EVAL_JSONL,
    build_feature_map,
    prediction_key,
    score,
)
from src.stage2_analysis.analyze_adaptive_route_case_studies import (  # noqa: E402
    argument_items,
    categorize_fn_arg,
    categorize_fp_arg,
    event_sets,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


LIKELIHOOD_RUNS = [
    {
        "branch": "likelihood10_raw",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_raw",
        "selection": "seen_stable_best",
        "checkpoint": "checkpoint-1806",
    },
    {
        "branch": "likelihood10_raw",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_raw",
        "selection": "hard_reason_best",
        "checkpoint": "checkpoint-1161",
    },
    {
        "branch": "likelihood10_bal30",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_bal30",
        "selection": "seen_stable_best",
        "checkpoint": "checkpoint-2656",
    },
    {
        "branch": "likelihood10_bal30",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_bal30",
        "selection": "hard_reason_best",
        "checkpoint": "checkpoint-332",
    },
    {
        "branch": "likelihood15_raw",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_raw",
        "selection": "seen_stable_best",
        "checkpoint": "checkpoint-1935",
    },
    {
        "branch": "likelihood15_raw",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_raw",
        "selection": "hard_reason_best",
        "checkpoint": "checkpoint-1032",
    },
    {
        "branch": "likelihood15_bal30",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30",
        "selection": "seen_stable_best",
        "checkpoint": "checkpoint-2355",
    },
    {
        "branch": "likelihood15_bal30",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30",
        "selection": "hard_reason_best",
        "checkpoint": "checkpoint-942",
    },
    {
        "branch": "likelihood10_pairdirect_bal30",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_pairdirect_bal30",
        "selection": "seen_stable_best",
        "checkpoint": "checkpoint-2760",
    },
    {
        "branch": "likelihood10_pairdirect_bal30",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_pairdirect_bal30",
        "selection": "hard_reason_best",
        "checkpoint": "checkpoint-1288",
    },
    {
        "branch": "likelihood10_pairdirect_bal30",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_pairdirect_bal30",
        "selection": "balanced_hardroute_best",
        "checkpoint": "checkpoint-2208",
    },
]

SPLITS = ["test", "test_seen", "test_unseen"]
FEATURE_KEYS = [
    "hardconf_score",
    "confusion_norm",
    "role_signature_rarity",
    "role_density_norm",
    "multi_event_or_multi_trigger",
    "core_role_absence_risk",
]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def prediction_path(root: Path, run, mode: str, split: str):
    return root / run["formal_slug"] / f"frontier_{run['selection']}" / mode / split / "predictions.jsonl"


def summary_path(root: Path, run, mode: str, split: str):
    return root / run["formal_slug"] / f"frontier_{run['selection']}" / mode / split / "summary.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def metric(row):
    return {
        "trigger_f1": float(row.get("trigger_f1", 0.0) or 0.0),
        "argument_f1": float(row.get("argument_f1", 0.0) or 0.0),
        "event_f1": float(row.get("event_f1", 0.0) or 0.0),
        "score": score(row),
    }


def feature_means(samples):
    return {
        key: mean([(sample.get("features") or {}).get(key, 0.0) for sample in samples])
        for key in FEATURE_KEYS
    }


def feature_auc(samples, feature_key):
    pos = [(s.get("features") or {}).get(feature_key, 0.0) for s in samples if s["reason_helpful"]]
    neg = [(s.get("features") or {}).get(feature_key, 0.0) for s in samples if not s["reason_helpful"]]
    if not pos or not neg:
        return 0.0
    wins = ties = total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / total


def route_capture(samples):
    helpful = [sample for sample in samples if sample["reason_helpful"]]
    free_reason = [sample for sample in samples if sample["free_route_pred"] == "reason"]
    captured = [sample for sample in helpful if sample["free_route_pred"] == "reason"]
    harmful_routed = [sample for sample in free_reason if not sample["reason_helpful"]]
    return {
        "reason_helpful_count": len(helpful),
        "reason_helpful_rate": len(helpful) / len(samples) if samples else 0.0,
        "free_reason_count": len(free_reason),
        "free_reason_rate": len(free_reason) / len(samples) if samples else 0.0,
        "captured_count": len(captured),
        "harmful_reason_routed_count": len(harmful_routed),
        "capture_recall": len(captured) / len(helpful) if helpful else 0.0,
        "capture_precision": len(captured) / len(free_reason) if free_reason else 0.0,
    }


def prf(pred_set, gold_set):
    if not pred_set and not gold_set:
        return {"p": 1.0, "r": 1.0, "f1": 1.0}
    if not pred_set:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}
    if not gold_set:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set)
    r = tp / len(gold_set)
    return {"p": p, "r": r, "f1": 0.0 if p + r == 0 else 2 * p * r / (p + r)}


def payload(row):
    return row.get("final_predicted") or row.get("predicted") or {"events": []}


def first_by_key(items):
    return {item["key"]: item for item in items}


def error_delta(direct_row, reason_row):
    gold = direct_row.get("gold") or {"events": []}
    direct_payload = payload(direct_row)
    reason_payload = payload(reason_row)
    gold_sets = event_sets(gold)
    direct_sets = event_sets(direct_payload)
    reason_sets = event_sets(reason_payload)

    gold_args = argument_items(gold)
    direct_args = argument_items(direct_payload)
    reason_args = argument_items(reason_payload)
    gold_by_key = first_by_key(gold_args)
    direct_by_key = first_by_key(direct_args)
    reason_by_key = first_by_key(reason_args)

    gold_keys = set(gold_by_key)
    direct_keys = set(direct_by_key)
    reason_keys = set(reason_by_key)
    recovered = (gold_keys - direct_keys) & reason_keys
    lost = (gold_keys & direct_keys) - reason_keys
    removed_fp = (direct_keys - gold_keys) - reason_keys
    added_fp = (reason_keys - gold_keys) - direct_keys

    recovered_fn_categories = Counter()
    lost_categories = Counter()
    removed_fp_categories = Counter()
    added_fp_categories = Counter()
    recovered_roles = Counter()
    lost_roles = Counter()
    for key in recovered:
        arg = gold_by_key[key]
        recovered_fn_categories[categorize_fn_arg(arg, direct_args)] += 1
        recovered_roles[arg["role"]] += 1
    for key in lost:
        arg = gold_by_key[key]
        lost_categories[categorize_fn_arg(arg, reason_args)] += 1
        lost_roles[arg["role"]] += 1
    for key in removed_fp:
        removed_fp_categories[categorize_fp_arg(direct_by_key[key], gold_args)] += 1
    for key in added_fp:
        added_fp_categories[categorize_fp_arg(reason_by_key[key], gold_args)] += 1

    return {
        "recovered_arg_count": len(recovered),
        "lost_arg_count": len(lost),
        "removed_fp_arg_count": len(removed_fp),
        "added_fp_arg_count": len(added_fp),
        "recovered_trigger_count": len((gold_sets["triggers"] - direct_sets["triggers"]) & reason_sets["triggers"]),
        "lost_trigger_count": len((gold_sets["triggers"] & direct_sets["triggers"]) - reason_sets["triggers"]),
        "trigger_correct_arg_repair": bool(gold_sets["triggers"] & direct_sets["triggers"]) and len(recovered) > 0,
        "recovered_fn_categories": recovered_fn_categories,
        "lost_categories": lost_categories,
        "removed_fp_categories": removed_fp_categories,
        "added_fp_categories": added_fp_categories,
        "recovered_roles": recovered_roles,
        "lost_roles": lost_roles,
    }


def add_counter(dst, src):
    for key, value in src.items():
        dst[key] += value


def aggregate_errors(samples):
    helpful = [sample for sample in samples if sample["reason_helpful"]]
    out = {
        "recovered_arg_count": 0,
        "lost_arg_count": 0,
        "removed_fp_arg_count": 0,
        "added_fp_arg_count": 0,
        "recovered_trigger_count": 0,
        "lost_trigger_count": 0,
        "trigger_correct_arg_repair_count": 0,
        "recovered_fn_categories": Counter(),
        "lost_categories": Counter(),
        "removed_fp_categories": Counter(),
        "added_fp_categories": Counter(),
        "recovered_roles": Counter(),
        "lost_roles": Counter(),
    }
    for sample in helpful:
        delta = sample["error_delta"]
        for key in [
            "recovered_arg_count",
            "lost_arg_count",
            "removed_fp_arg_count",
            "added_fp_arg_count",
            "recovered_trigger_count",
            "lost_trigger_count",
        ]:
            out[key] += delta[key]
        if delta["trigger_correct_arg_repair"]:
            out["trigger_correct_arg_repair_count"] += 1
        for key in [
            "recovered_fn_categories",
            "lost_categories",
            "removed_fp_categories",
            "added_fp_categories",
            "recovered_roles",
            "lost_roles",
        ]:
            add_counter(out[key], delta[key])
    for key, value in list(out.items()):
        if isinstance(value, Counter):
            out[key] = value.most_common(12)
    out["num_helpful_samples"] = len(helpful)
    return out


def selected_by_feature(samples, feature_key, budget):
    cap = round(len(samples) * budget)
    ranked = sorted(samples, key=lambda s: ((s.get("features") or {}).get(feature_key, 0.0), s["key"]), reverse=True)
    return {sample["key"] for sample in ranked[:cap]}


def selected_oracle(samples, budget):
    cap = round(len(samples) * budget)
    ranked = [sample for sample in samples if sample["reason_gain"] > 1e-9]
    ranked.sort(key=lambda s: (s["reason_gain"], s["key"]), reverse=True)
    return {sample["key"] for sample in ranked[:cap]}


def summarize_routed(samples, reason_ids):
    routed = []
    direct = []
    helpful = {sample["key"] for sample in samples if sample["reason_helpful"]}
    for sample in samples:
        use_reason = sample["key"] in reason_ids
        metric_key = "reason_metric" if use_reason else "direct_metric"
        routed.append(sample[metric_key])
        direct.append(sample["direct_metric"])
    chosen_helpful = reason_ids & helpful
    return {
        "selected_count": len(reason_ids),
        "precision": len(chosen_helpful) / len(reason_ids) if reason_ids else 0.0,
        "recall": len(chosen_helpful) / len(helpful) if helpful else 0.0,
        "trigger_f1": mean([m["trigger_f1"] for m in routed]),
        "argument_f1": mean([m["argument_f1"] for m in routed]),
        "event_f1": mean([m["event_f1"] for m in routed]),
        "argument_gain": mean([m["argument_f1"] for m in routed]) - mean([m["argument_f1"] for m in direct]),
        "event_gain": mean([m["event_f1"] for m in routed]) - mean([m["event_f1"] for m in direct]),
    }


def analyze_run_split(root: Path, schema_path: Path, run, split: str):
    paths = {mode: prediction_path(root, run, mode, split) for mode in ["free_route", "forced_direct", "forced_reason"]}
    summaries = {mode: load_json(summary_path(root, run, mode, split)) for mode in paths}
    maps = {mode: load_prediction_map(path) for mode, path in paths.items()}
    features = build_feature_map(Path(DIRECT_EVAL_JSONL[split]), schema_path)
    samples = []
    for key in sorted(set(maps["free_route"]) & set(maps["forced_direct"]) & set(maps["forced_reason"])):
        free_row = maps["free_route"][key]
        direct_row = maps["forced_direct"][key]
        reason_row = maps["forced_reason"][key]
        direct_metric = metric(direct_row)
        reason_metric = metric(reason_row)
        gain = reason_metric["score"] - direct_metric["score"]
        sample = {
            "key": key,
            "split": split,
            "free_route_pred": free_row.get("route_pred", "unknown"),
            "reason_used": bool(free_row.get("reason_used")),
            "direct_metric": direct_metric,
            "reason_metric": reason_metric,
            "free_metric": metric(free_row),
            "reason_gain": gain,
            "reason_helpful": gain > 1e-9,
            "features": features.get(key, {}),
            "error_delta": error_delta(direct_row, reason_row),
        }
        samples.append(sample)

    routed_selectors = {
        "oracle15": summarize_routed(samples, selected_oracle(samples, 0.15)),
        "hardconf15": summarize_routed(samples, selected_by_feature(samples, "hardconf_score", 0.15)),
        "role_density15": summarize_routed(samples, selected_by_feature(samples, "role_density_norm", 0.15)),
        "role_signature15": summarize_routed(samples, selected_by_feature(samples, "role_signature_rarity", 0.15)),
    }
    helpful = [sample for sample in samples if sample["reason_helpful"]]
    not_helpful = [sample for sample in samples if not sample["reason_helpful"]]
    free_reason = [sample for sample in samples if sample["free_route_pred"] == "reason"]
    free_direct = [sample for sample in samples if sample["free_route_pred"] == "direct"]
    return {
        "summaries": summaries,
        "num_samples": len(samples),
        "route_capture": route_capture(samples),
        "feature_means": {
            "reason_helpful": feature_means(helpful),
            "reason_not_helpful": feature_means(not_helpful),
            "free_reason": feature_means(free_reason),
            "free_direct": feature_means(free_direct),
        },
        "feature_auc": {key: feature_auc(samples, key) for key in FEATURE_KEYS},
        "selector_oracle": routed_selectors,
        "error_aggregate_helpful": aggregate_errors(samples),
    }


def analyze_dev_likelihood_alignment(scores_path: Path, labels_path: Path, direct_path: Path, reason_path: Path, schema_path: Path):
    scores = {row["wnd_id"]: row for row in load_jsonl(scores_path)}
    labels = {row["wnd_id"]: row for row in load_jsonl(labels_path)}
    direct = load_prediction_map(direct_path)
    reason = load_prediction_map(reason_path)
    features = build_feature_map(Path("data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_dev_seen_pos.jsonl"), schema_path)
    rows = []
    for key in sorted(set(scores) & set(labels) & set(direct) & set(reason)):
        dm = metric(direct[key])
        rm = metric(reason[key])
        gain = rm["score"] - dm["score"]
        rows.append(
            {
                "key": key,
                "delta_final_nll": scores[key].get("delta_final_nll"),
                "route_label": labels[key].get("route_label"),
                "reason_helpful": gain > 1e-9,
                "reason_gain": gain,
                "features": features.get(key, {}),
            }
        )
    selected = {row["key"] for row in rows if row["route_label"] == "reason"}
    helpful = {row["key"] for row in rows if row["reason_helpful"]}
    selected_helpful = selected & helpful
    auc_rows = []
    for row in rows:
        auc_rows.append({"reason_helpful": row["reason_helpful"], "features": {"delta_final_nll": row["delta_final_nll"] or 0.0}})
    return {
        "num_rows": len(rows),
        "selected_count": len(selected),
        "helpful_count": len(helpful),
        "selected_helpful_count": len(selected_helpful),
        "precision": len(selected_helpful) / len(selected) if selected else 0.0,
        "recall": len(selected_helpful) / len(helpful) if helpful else 0.0,
        "reason_helpful_rate": len(helpful) / len(rows) if rows else 0.0,
        "auc_delta_final_nll": feature_auc(auc_rows, "delta_final_nll"),
        "selected_feature_means": feature_means([row for row in rows if row["key"] in selected]),
        "helpful_feature_means": feature_means([row for row in rows if row["reason_helpful"]]),
        "nonhelpful_feature_means": feature_means([row for row in rows if not row["reason_helpful"]]),
    }


def analyze(args):
    root = Path(args.formal_root)
    schema_path = Path(args.schema_path)
    runs = []
    for run in LIKELIHOOD_RUNS:
        split_payload = {}
        for split in SPLITS:
            split_payload[split] = analyze_run_split(root, schema_path, run, split)
        runs.append({**run, "splits": split_payload})
    dev_align = {}
    dev_direct_base = Path(args.devpick_root)
    for cap, labels_path in [
        ("goldplan10", Path(args.likelihood10_dev_labels)),
        ("goldplan15", Path(args.likelihood15_dev_labels)),
    ]:
        # Use final scorer labels against the strongest likelihood15_bal30 hard-reason checkpoint on dev.
        direct_path = dev_direct_base / "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30_full_forced_direct_dev_seen_max512" / "checkpoint-942" / "predictions.jsonl"
        reason_path = dev_direct_base / "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30_full_forced_reason_dev_seen_max512" / "checkpoint-942" / "predictions.jsonl"
        if direct_path.exists() and reason_path.exists():
            dev_align[cap] = analyze_dev_likelihood_alignment(
                Path(args.dev_scores), labels_path, direct_path, reason_path, schema_path
            )
    return {"runs": runs, "dev_likelihood_alignment": dev_align}


def fmt(x):
    return "-" if x is None else f"{x:.4f}"


def markdown(payload):
    lines = [
        "# Adaptive Likelihood Router Deep Analysis",
        "",
        "Date: 2026-05-13",
        "",
        "## Executive Reading",
        "",
        "- The likelihood-router wave confirms reason has local action value, but the learned free router still misses most reason-helpful samples.",
        "- The best free-route models are mainly strong because their direct path is strong, not because adaptive routing is working.",
        "- Gold-plan FINAL-NLL is too optimistic/noisy as route supervision: it can identify some reason-positive regions, but it does not produce a reliable route boundary.",
        "- The next optimization should prioritize router supervision mechanics: outcome-aware labels, route-token loss balancing, and route-only warmup/evaluation.",
        "",
        "## Route Capture And Oracle Headroom",
        "",
        "| branch | selection | split | free arg/event/rr | direct arg/event | reason arg/event | helpful rate | capture R/P | oracle15 gain arg/event | hardconf15 gain arg/event |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        for split in SPLITS:
            row = run["splits"][split]
            free = row["summaries"]["free_route"]
            direct = row["summaries"]["forced_direct"]
            reason = row["summaries"]["forced_reason"]
            cap = row["route_capture"]
            oracle15 = row["selector_oracle"]["oracle15"]
            hard15 = row["selector_oracle"]["hardconf15"]
            lines.append(
                "| `{}` | `{}` | `{}` | {}/{}/{} | {}/{} | {}/{} | {} | {}/{} | {:+.4f}/{:+.4f} | {:+.4f}/{:+.4f} |".format(
                    run["branch"],
                    run["selection"],
                    split,
                    fmt(free.get("final_argument_f1", free.get("argument_f1"))),
                    fmt(free.get("final_event_f1", free.get("event_f1"))),
                    fmt(free.get("route_reason_rate")),
                    fmt(direct.get("final_argument_f1", direct.get("argument_f1"))),
                    fmt(direct.get("final_event_f1", direct.get("event_f1"))),
                    fmt(reason.get("final_argument_f1", reason.get("argument_f1"))),
                    fmt(reason.get("final_event_f1", reason.get("event_f1"))),
                    fmt(cap["reason_helpful_rate"]),
                    fmt(cap["capture_recall"]),
                    fmt(cap["capture_precision"]),
                    oracle15["argument_gain"],
                    oracle15["event_gain"],
                    hard15["argument_gain"],
                    hard15["event_gain"],
                )
            )
    lines.extend([
        "",
        "## Feature Contrast",
        "",
        "| branch | selection | split | helpful hardconf | nonhelp hardconf | helpful role-density | nonhelp role-density | hardconf AUC | role-density AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run in payload["runs"]:
        for split in SPLITS:
            row = run["splits"][split]
            fm = row["feature_means"]
            auc = row["feature_auc"]
            lines.append(
                "| `{}` | `{}` | `{}` | {} | {} | {} | {} | {} | {} |".format(
                    run["branch"],
                    run["selection"],
                    split,
                    fmt(fm["reason_helpful"]["hardconf_score"]),
                    fmt(fm["reason_not_helpful"]["hardconf_score"]),
                    fmt(fm["reason_helpful"]["role_density_norm"]),
                    fmt(fm["reason_not_helpful"]["role_density_norm"]),
                    fmt(auc["hardconf_score"]),
                    fmt(auc["role_density_norm"]),
                )
            )
    lines.extend(["", "## Dev Likelihood Label Alignment", ""])
    lines.append("| labels | rows | selected | helpful | precision | recall | AUC delta-nll |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, row in payload["dev_likelihood_alignment"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                name,
                row["num_rows"],
                row["selected_count"],
                row["helpful_count"],
                fmt(row["precision"]),
                fmt(row["recall"]),
                fmt(row["auc_delta_final_nll"]),
            )
        )
    lines.extend([
        "",
        "## Error Repairs In Reason-Helpful Samples",
        "",
        "The aggregate counters below are computed only on samples where forced-reason beats forced-direct under the task score.",
        "",
    ])
    for focus_branch, focus_sel in [
        ("likelihood15_bal30", "hard_reason_best"),
        ("likelihood15_raw", "seen_stable_best"),
        ("likelihood10_pairdirect_bal30", "balanced_hardroute_best"),
    ]:
        for run in payload["runs"]:
            if run["branch"] == focus_branch and run["selection"] == focus_sel:
                lines.append(f"### `{focus_branch}/{focus_sel}`")
                lines.append("")
                for split in SPLITS:
                    err = run["splits"][split]["error_aggregate_helpful"]
                    lines.append(
                        "- `{}`: helpful `{}`, recovered args `{}`, lost args `{}`, removed FP args `{}`, added FP args `{}`, trigger-correct arg repairs `{}`.".format(
                            split,
                            err["num_helpful_samples"],
                            err["recovered_arg_count"],
                            err["lost_arg_count"],
                            err["removed_fp_arg_count"],
                            err["added_fp_arg_count"],
                            err["trigger_correct_arg_repair_count"],
                        )
                    )
                    lines.append(f"  - recovered FN categories: `{err['recovered_fn_categories'][:6]}`")
                    lines.append(f"  - recovered roles: `{err['recovered_roles'][:8]}`")
                lines.append("")
    lines.extend([
        "## Optimization Implications",
        "",
        "1. Do not rely on gold-plan FINAL-NLL alone as the next route label. It is a useful proxy but does not align tightly enough with actual forced-reason wins.",
        "2. Add a route-only or route-weighted training stage. The current route token is overwhelmed by the long FINAL JSON loss, so even 30% route balancing can still collapse to direct.",
        "3. Build outcome-aware labels on train/dev with split discipline. The label should be `reason` only when forced-reason improves argument/event/trigger score, otherwise `direct`.",
        "4. Use hardconf/role-density as priors, not labels. They separate helpful from non-helpful samples weakly to moderately, so they are good candidate-pool filters before outcome labeling.",
        "5. Improve the reason action value only after route learning is measurable. The current reason path can help, especially on unseen and some seen hard samples, but global forced-reason is not stable enough.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_root", default="outputs/stage2_adaptive_runs_user_formal_clean")
    parser.add_argument("--devpick_root", default="outputs/stage2_adaptive_runs_user_devpick_frontier")
    parser.add_argument("--schema_path", default="data/schema/richere-en.event_schema.json")
    parser.add_argument("--dev_scores", default="outputs/stage2_adaptive_likelihood_scores/pairall_type_role_hint_plan_lite_scorer/dev_seen.jsonl")
    parser.add_argument("--likelihood10_dev_labels", default="data/stage2_adaptive_datasets/labels/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood_goldplan10_dev_seen_labels.jsonl")
    parser.add_argument("--likelihood15_dev_labels", default="data/stage2_adaptive_datasets/labels/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood_goldplan15_dev_seen_labels.jsonl")
    parser.add_argument("--output_md", default="reports/2026-05-13_stage2_adaptive_likelihood_router_deep_analysis.md")
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-13_stage2_adaptive_likelihood_router_deep_analysis.json")
    args = parser.parse_args()

    payload = analyze(args)
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), markdown(payload))
    print(json.dumps({"output_md": args.output_md, "output_json": args.output_json}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
