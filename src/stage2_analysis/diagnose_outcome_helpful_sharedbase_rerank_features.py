import argparse
from collections import defaultdict
import itertools
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import prediction_key, score  # noqa: E402
from src.stage2_analysis.analyze_adaptive_outcome_router_execution import (  # noqa: E402
    route_prf,
    row_metric,
    summarize_metrics,
    write_json,
    write_text,
)
from src.stage2_cot.build_selective_aux_reasoning_dataset import (  # noqa: E402
    build_confrare_stats,
    hardconf_score_row,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl, load_schema_map  # noqa: E402


RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BRANCH = "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux"
CKPT = "checkpoint-2058"
SCHEMA = Path("data/schema/richere-en.event_schema.json")
SPLITS = ["dev_seen", "test", "test_seen", "test_unseen"]
BUDGETS = [0.10, 0.15, 0.20, 0.25, 0.30]


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def paths_for(split: str):
    if split == "dev_seen":
        direct = REPO_ROOT / (
            f"outputs/stage2_adaptive_runs_user_devpick_frontier/{RUN_PREFIX}_{BRANCH}_full_forced_direct_dev_seen_max512/{CKPT}/predictions.jsonl"
        )
        reason = REPO_ROOT / (
            f"outputs/stage2_adaptive_runs_user_devpick_frontier/{RUN_PREFIX}_{BRANCH}_full_forced_reason_dev_seen_max512/{CKPT}/predictions.jsonl"
        )
        scores = REPO_ROOT / f"outputs/stage2_adaptive_route_likelihood_probe/outcome_helpful_sharedbase_20260515/{BRANCH}/{CKPT}/dev_seen_scores.jsonl"
        eval_rows = REPO_ROOT / f"data/stage2_adaptive_datasets/{DATA_PREFIX}_{BRANCH}_dev_seen_pos.jsonl"
        return direct, reason, scores, eval_rows
    direct = REPO_ROOT / (
        f"outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/richere_split1_qwen3_1_7b_adaptive_{BRANCH}/{CKPT}/forced_direct/{split}/predictions.jsonl"
    )
    reason = REPO_ROOT / (
        f"outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/richere_split1_qwen3_1_7b_adaptive_{BRANCH}/{CKPT}/forced_reason/{split}/predictions.jsonl"
    )
    scores = REPO_ROOT / f"outputs/stage2_adaptive_route_likelihood_probe/outcome_helpful_sharedbase_formal_20260515/{BRANCH}/{CKPT}/{split}/scores.jsonl"
    eval_rows = REPO_ROOT / f"data/stage2_adaptive_datasets/{DATA_PREFIX}_{BRANCH}_{split}_pos.jsonl"
    return direct, reason, scores, eval_rows


def row_id(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or prediction_key(row)


def static_feature_map(eval_jsonl: Path):
    rows = load_jsonl(eval_jsonl)
    feature_rows = []
    for row in rows:
        feature_row = dict(row)
        if row.get("gold_output"):
            feature_row["output"] = row["gold_output"]
        feature_rows.append(feature_row)
    schema_by_type = load_schema_map(SCHEMA)
    stats = build_confrare_stats(feature_rows)
    out = {}
    for idx, row in enumerate(feature_rows):
        features = hardconf_score_row(idx, row, schema_by_type, stats)
        out[row_id(row)] = {key: value for key, value in features.items() if key != "idx"}
    return out


def compact_count(payload):
    events = payload.get("events", []) if isinstance(payload, dict) else []
    if not isinstance(events, list):
        return {"event_count": 0, "argument_count": 0, "trigger_count": 0}
    triggers = set()
    args = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        triggers.add((event.get("event_type"), trigger.get("start"), trigger.get("end")))
        raw_args = event.get("arguments") if isinstance(event.get("arguments"), list) else []
        args += sum(1 for arg in raw_args if isinstance(arg, dict))
    return {"event_count": len(events), "argument_count": args, "trigger_count": len(triggers)}


def text_len(row):
    return len((row.get("generated_payload") or row.get("generated_text") or "").split())


def feature_rows(split: str):
    direct_path, reason_path, score_path, eval_path = paths_for(split)
    direct = load_prediction_map(direct_path)
    reason = load_prediction_map(reason_path)
    scores = load_prediction_map(score_path)
    static = static_feature_map(eval_path)
    rows = []
    for key in sorted(set(direct) & set(reason) & set(scores)):
        drow = direct[key]
        rrow = reason[key]
        srow = scores[key]
        delta = srow.get("delta_direct_minus_reason_route_nll")
        dcount = compact_count(drow.get("predicted") or drow.get("final_predicted") or {})
        rcount = compact_count(rrow.get("predicted") or rrow.get("final_predicted") or {})
        reason_gain = score(rrow) - score(drow)
        features = {
            "nll_delta": float(delta) if delta is not None else float("-inf"),
            "nll_delta_neg": -(float(delta) if delta is not None else float("-inf")),
            "direct_len": text_len(drow),
            "reason_len": text_len(rrow),
            "len_diff": text_len(rrow) - text_len(drow),
            "direct_event_count": dcount["event_count"],
            "reason_event_count": rcount["event_count"],
            "event_count_diff": rcount["event_count"] - dcount["event_count"],
            "direct_argument_count": dcount["argument_count"],
            "reason_argument_count": rcount["argument_count"],
            "argument_count_diff": rcount["argument_count"] - dcount["argument_count"],
            "direct_trigger_count": dcount["trigger_count"],
            "reason_trigger_count": rcount["trigger_count"],
            "trigger_count_diff": rcount["trigger_count"] - dcount["trigger_count"],
            "abs_event_count_diff": abs(rcount["event_count"] - dcount["event_count"]),
            "abs_argument_count_diff": abs(rcount["argument_count"] - dcount["argument_count"]),
            "hardconf_score": (static.get(key) or {}).get("hardconf_score", 0.0),
            "confusion_norm": (static.get(key) or {}).get("confusion_norm", 0.0),
            "role_signature_rarity": (static.get(key) or {}).get("role_signature_rarity", 0.0),
            "role_density_norm": (static.get(key) or {}).get("role_density_norm", 0.0),
            "multi_event_or_multi_trigger": (static.get(key) or {}).get("multi_event_or_multi_trigger", 0.0),
            "core_role_absence_risk": (static.get(key) or {}).get("core_role_absence_risk", 0.0),
        }
        rows.append(
            {
                "key": key,
                "split": split,
                "direct": drow,
                "reason": rrow,
                "reason_gain": reason_gain,
                "helpful": reason_gain > 1e-9,
                "features": features,
                "meta": drow.get("meta") or {},
            }
        )
    return rows


def normalize(values):
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return [0.0 for _ in values]
    lo = min(vals)
    hi = max(vals)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) if math.isfinite(v) else 0.0 for v in values]


def add_scores(rows, feature_names, weights):
    values_by_feature = {name: normalize([row["features"].get(name, 0.0) for row in rows]) for name in feature_names}
    for idx, row in enumerate(rows):
        row["rerank_score"] = sum(weight * values_by_feature[name][idx] for name, weight in zip(feature_names, weights))


def selected_by_score(rows, budget):
    cap = round(len(rows) * budget)
    ranked = sorted(rows, key=lambda row: (row["rerank_score"], row["key"]), reverse=True)
    return {row["key"] for row in ranked[:cap]}


def evaluate(rows, selected, rule_name):
    direct_metrics = []
    routed_metrics = []
    helpful_tp = helpful_fp = helpful_fn = 0
    positive_helpful_count = 0
    for row in rows:
        direct_metrics.append(row_metric(row["direct"]))
        use_reason = row["key"] in selected
        routed_metrics.append(row_metric(row["reason"] if use_reason else row["direct"]))
        helpful = row["helpful"]
        if helpful:
            positive_helpful_count += 1
        if use_reason and helpful:
            helpful_tp += 1
        elif use_reason and not helpful:
            helpful_fp += 1
        elif not use_reason and helpful:
            helpful_fn += 1
    direct = summarize_metrics(direct_metrics)
    routed = summarize_metrics(routed_metrics)
    return {
        "rule": rule_name,
        "selected_reason_count": len(selected),
        "selected_reason_rate": len(selected) / len(rows) if rows else 0.0,
        "positive_reason_helpful_count": positive_helpful_count,
        "route_vs_positive_reason_helpful": route_prf(helpful_tp, helpful_fp, helpful_fn),
        "direct": direct,
        "routed": routed,
        "routed_delta_vs_direct": {
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
        },
    }


def candidate_specs():
    deployable = [
        ("nll_delta", ["nll_delta"], [1.0], "deployable"),
        ("nll_plus_static", ["nll_delta", "role_signature_rarity", "multi_event_or_multi_trigger", "role_density_norm"], [1.0, 0.5, 0.3, 0.2], "deployable"),
        ("static_hardness", ["hardconf_score", "role_signature_rarity", "multi_event_or_multi_trigger", "role_density_norm"], [1.0, 0.5, 0.4, 0.2], "deployable"),
        ("low_absence_static", ["role_signature_rarity", "multi_event_or_multi_trigger", "core_role_absence_risk"], [0.7, 0.5, -0.3], "deployable"),
    ]
    diagnostic = [
        ("nll_plus_disagreement", ["nll_delta", "abs_event_count_diff", "abs_argument_count_diff", "len_diff"], [1.0, 0.5, 0.4, 0.1], "two_pass_diagnostic"),
        ("reason_more_events", ["reason_event_count", "reason_argument_count", "event_count_diff", "argument_count_diff"], [0.7, 0.5, 0.4, 0.3], "two_pass_diagnostic"),
        ("reason_output_richness", ["reason_len", "reason_event_count", "reason_argument_count", "abs_argument_count_diff"], [0.2, 0.6, 0.6, 0.4], "two_pass_diagnostic"),
        ("disagreement_static", ["abs_event_count_diff", "abs_argument_count_diff", "role_signature_rarity", "multi_event_or_multi_trigger"], [0.6, 0.5, 0.4, 0.3], "two_pass_diagnostic"),
    ]
    # Small grid for transparent linear rerankers over diagnostic features.
    base_features = ["nll_delta", "abs_event_count_diff", "abs_argument_count_diff", "role_signature_rarity", "multi_event_or_multi_trigger"]
    diagnostic_feature_set = {"abs_event_count_diff", "abs_argument_count_diff"}
    grid = []
    for weights in itertools.product([0.0, 0.5, 1.0], repeat=len(base_features)):
        if sum(weights) == 0:
            continue
        used_features = {feature for feature, weight in zip(base_features, weights) if weight > 0}
        family = "two_pass_grid" if used_features & diagnostic_feature_set else "deployable_grid"
        grid.append((f"grid_{'_'.join(str(int(w * 10)) for w in weights)}", base_features, list(weights), family))
    return deployable + diagnostic + grid


def robustness_ok(result):
    delta = result["routed_delta_vs_direct"]
    helpful = result["route_vs_positive_reason_helpful"]
    return (
        delta["argument_f1"] >= 0.0
        and delta["event_f1"] >= 0.005
        and delta["trigger_f1"] >= -0.012
        and helpful["recall"] >= 0.20
    )


def score_result(result):
    delta = result["routed_delta_vs_direct"]
    helpful = result["route_vs_positive_reason_helpful"]
    return (
        int(robustness_ok(result)),
        min(delta["argument_f1"], 0.03) + min(delta["event_f1"], 0.03),
        helpful["f1"],
        helpful["recall"],
        -max(0.0, -delta["trigger_f1"]),
        -abs(result["selected_reason_rate"] - 0.20),
    )


def run_candidates(rows_by_split):
    dev_rows = rows_by_split["dev_seen"]
    results = []
    for name, features, weights, family in candidate_specs():
        for budget in BUDGETS:
            local_rows = [dict(row, features=dict(row["features"])) for row in dev_rows]
            add_scores(local_rows, features, weights)
            selected = selected_by_score(local_rows, budget)
            result = evaluate(dev_rows, selected, f"{name}_top{int(budget * 100):02d}")
            result["family"] = family
            result["features"] = features
            result["weights"] = weights
            result["budget"] = budget
            result["robustness_ok"] = robustness_ok(result)
            result["selector_score"] = score_result(result)
            results.append(result)
    ranked = sorted(results, key=lambda row: row["selector_score"], reverse=True)
    selected_by_family = {}
    for family in ["deployable", "deployable_grid", "two_pass_diagnostic", "two_pass_grid"]:
        selected_by_family[family] = next((row for row in ranked if row["family"] == family), None)
    formal = {}
    for family, dev_result in selected_by_family.items():
        if dev_result is None:
            continue
        formal[family] = {}
        for split in ["test", "test_seen", "test_unseen"]:
            split_rows = [dict(row, features=dict(row["features"])) for row in rows_by_split[split]]
            add_scores(split_rows, dev_result["features"], dev_result["weights"])
            selected = selected_by_score(split_rows, dev_result["budget"])
            formal[family][split] = evaluate(rows_by_split[split], selected, dev_result["rule"])
    return ranked, selected_by_family, formal


def render(payload):
    lines = [
        "# Outcome-Helpful Shared-Base Rerank Feature Diagnosis",
        "",
        "## Decision",
        "",
    ]
    for family, row in payload["selected_by_family"].items():
        if row is None:
            continue
        lines.append(f"- `{family}` selected on dev: `{row['rule']}` score `{row['selector_score']}`")
    lines.extend(["", "## Dev Top Candidates", ""])
    lines.append("| rule | family | selected | arg/event delta | trigger delta | helpful P/R/F1 | robust |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["dev_ranked"][:20]:
        delta = row["routed_delta_vs_direct"]
        helpful = row["route_vs_positive_reason_helpful"]
        lines.append(
            "| `{}` | `{}` | {:.1%} | {:+.4f}/{:+.4f} | {:+.4f} | {:.3f}/{:.3f}/{:.3f} | {} |".format(
                row["rule"],
                row["family"],
                row["selected_reason_rate"],
                delta["argument_f1"],
                delta["event_f1"],
                delta["trigger_f1"],
                helpful["precision"],
                helpful["recall"],
                helpful["f1"],
                "yes" if row["robustness_ok"] else "no",
            )
        )
    lines.extend(["", "## Formal Transfer", ""])
    lines.append("| family | split | rule | selected | delta arg/event | helpful P/R/F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for family, split_rows in payload["formal"].items():
        for split, row in split_rows.items():
            delta = row["routed_delta_vs_direct"]
            helpful = row["route_vs_positive_reason_helpful"]
            lines.append(
                "| `{}` | `{}` | `{}` | {:.1%} | {:+.4f}/{:+.4f} | {:.3f}/{:.3f}/{:.3f} |".format(
                    family,
                    split,
                    row["rule"],
                    row["selected_reason_rate"],
                    delta["argument_f1"],
                    delta["event_f1"],
                    helpful["precision"],
                    helpful["recall"],
                    helpful["f1"],
                )
            )
    lines.extend(["", "## Reading", ""])
    lines.append("- Deployable-ish static/direct-feature rerankers test whether cheap features can improve over raw NLL without forced-reason execution.")
    lines.append("- Two-pass diagnostic rerankers are not deployment-ready; they estimate whether richer execution/disagreement features contain a learnable signal.")
    lines.append("- If diagnostic rerankers transfer but deployable rerankers do not, the next training objective should learn a route score from outcome/disagreement signals rather than hand-tuning static rules.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-15_stage2_adaptive_outcome_helpful_sharedbase_rerank_feature_diagnosis.json")
    parser.add_argument("--output_md", default="reports/2026-05-15_stage2_adaptive_outcome_helpful_sharedbase_rerank_feature_diagnosis.md")
    args = parser.parse_args()
    rows_by_split = {split: feature_rows(split) for split in SPLITS}
    dev_ranked, selected_by_family, formal = run_candidates(rows_by_split)
    payload = {
        "branch": BRANCH,
        "checkpoint": CKPT,
        "selected_by_family": selected_by_family,
        "dev_ranked": dev_ranked,
        "formal": formal,
        "candidate_count": len(dev_ranked),
    }
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), render(payload))
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md, "candidate_count": len(dev_ranked)}, indent=2))


if __name__ == "__main__":
    main()
