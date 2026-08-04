#!/usr/bin/env python3
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.diagnose_sampled_k2_goldfree_harmful_cases_20260519 import pair_features  # noqa: E402
from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


SPLITS = ["test_seen", "test_unseen"]
ALL_SPLITS = ["test", "test_seen", "test_unseen"]
OUTPUT_ROOT = REPO / "outputs/stage2_strong_system_v0_supervised_rerank_20260520"
REPORT_MD = REPO / "reports/2026-05-20_stage2_strong_system_v0_supervised_rerank.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_strong_system_v0_supervised_rerank.json"
CASE_JSONL = OUTPUT_ROOT / "candidate_cases.jsonl"

FORMAL_GATED = REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated"
FORMAL_CLEAN = REPO / "outputs/stage2_adaptive_runs_user_formal_clean"
DEVPICK_FRONTIER = REPO / "outputs/stage2_adaptive_runs_user_devpick_frontier"

DIRECT_DEV = (
    DEVPICK_FRONTIER
    / "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux_full_forced_direct_dev_seen_max512"
    / "checkpoint-1930/predictions.jsonl"
)
REASON_DEV = (
    DEVPICK_FRONTIER
    / "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux_full_forced_reason_dev_seen_max512"
    / "checkpoint-2058/predictions.jsonl"
)


def candidate_specs(split):
    return [
        {
            "name": "direct_modular_d1930",
            "role": "direct",
            "path": (
                FORMAL_GATED
                / "outcome_helpful_sharedbase_balrouteaux_20260516"
                / "richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux"
                / f"checkpoint-1930/forced_direct/{split}/predictions.jsonl"
            ),
        },
        {
            "name": "reason_modular_r2058",
            "role": "reason",
            "path": (
                FORMAL_GATED
                / "outcome_helpful_sharedbase_20260515"
                / "richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux"
                / f"checkpoint-2058/forced_reason/{split}/predictions.jsonl"
            ),
        },
        {
            "name": "free_type_plan_reason_best",
            "role": "free_route",
            "path": (
                FORMAL_CLEAN
                / "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_plan_lite"
                / f"frontier_reason_expert_best/free_route/{split}/predictions.jsonl"
            ),
        },
        {
            "name": "free_type_plan_direct_anchor",
            "role": "free_route",
            "path": (
                FORMAL_CLEAN
                / "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_plan_lite"
                / f"frontier_direct_anchor_best/free_route/{split}/predictions.jsonl"
            ),
        },
    ]


def dev_candidate_specs():
    return [
        {"name": "direct_modular_d1930", "role": "direct", "path": DIRECT_DEV},
        {"name": "reason_modular_r2058", "role": "reason", "path": REASON_DEV},
    ]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def key_for(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("wnd_id")


def valid_json(row):
    return bool(row.get("valid_final_json", row.get("valid_json", False)))


def metric_score(row):
    return row.get("argument_f1", 0.0) + row.get("event_f1", 0.0) + 0.25 * row.get("trigger_f1", 0.0)


def row_metrics(row):
    return {
        "json_valid_rate": 1.0 if valid_json(row) else 0.0,
        "trigger_f1": float(row.get("trigger_f1", 0.0) or 0.0),
        "argument_f1": float(row.get("argument_f1", 0.0) or 0.0),
        "event_f1": float(row.get("event_f1", 0.0) or 0.0),
        "score": metric_score(row),
    }


def load_predictions(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    rows = {}
    for row in load_jsonl(path):
        key = key_for(row)
        if key:
            rows[key] = row
    return rows


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def stdev(values):
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return math.sqrt(mean((value - mu) ** 2 for value in vals))


def summarize_rows(rows):
    items = list(rows)
    if not items:
        return {
            "num_examples": 0,
            "json_valid_rate": 0.0,
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
            "score": 0.0,
        }
    return {
        "num_examples": len(items),
        "json_valid_rate": mean(row_metrics(row)["json_valid_rate"] for row in items),
        "trigger_f1": mean(row_metrics(row)["trigger_f1"] for row in items),
        "argument_f1": mean(row_metrics(row)["argument_f1"] for row in items),
        "event_f1": mean(row_metrics(row)["event_f1"] for row in items),
        "score": mean(row_metrics(row)["score"] for row in items),
    }


def safe_pair_features(base_row, cand_row):
    feats = pair_features(base_row, cand_row)
    out = {}
    for name, value in feats.items():
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            out[name] = 0.0
    return out


def candidate_feature_row(base_row, cand_row, spec):
    feats = safe_pair_features(base_row, cand_row)
    role = spec["role"]
    output = {
        "bias": 1.0,
        "is_direct": 1.0 if role == "direct" else 0.0,
        "is_reason": 1.0 if role == "reason" else 0.0,
        "is_free_route": 1.0 if role == "free_route" else 0.0,
        "valid_json": 1.0 if valid_json(cand_row) else 0.0,
        "trigger_f1_self_proxy": 0.0,
        "arg_text_jaccard": feats.get("arg_text_jaccard", 0.0),
        "arg_role_jaccard": feats.get("arg_role_jaccard", 0.0),
        "event_type_jaccard": feats.get("event_type_jaccard", 0.0),
        "trigger_text_jaccard": feats.get("trigger_text_jaccard", 0.0),
        "reason_new_arg_text_count": feats.get("reason_new_arg_text_count", 0.0),
        "reason_new_event_type_count": feats.get("reason_new_event_type_count", 0.0),
        "argument_count_delta": feats.get("argument_count_delta", 0.0),
        "event_count_delta": feats.get("event_count_delta", 0.0),
    }
    return output


FEATURES = [
    "bias",
    "is_direct",
    "is_reason",
    "is_free_route",
    "valid_json",
    "arg_text_jaccard",
    "arg_role_jaccard",
    "event_type_jaccard",
    "trigger_text_jaccard",
    "reason_new_arg_text_count",
    "reason_new_event_type_count",
    "argument_count_delta",
    "event_count_delta",
]


def fit_linear(rows):
    stats = {}
    for feat in FEATURES:
        vals = [row["features"][feat] for row in rows]
        stats[feat] = {"mean": mean(vals), "std": max(stdev(vals), 1e-6)}

    weights = {}
    labels = [row["target_gain"] for row in rows]
    label_mu = mean(labels)
    label_sd = max(stdev(labels), 1e-6)
    for feat in FEATURES:
        xs = [(row["features"][feat] - stats[feat]["mean"]) / stats[feat]["std"] for row in rows]
        ys = [(row["target_gain"] - label_mu) / label_sd for row in rows]
        weights[feat] = mean(x * y for x, y in zip(xs, ys))
    return {"stats": stats, "weights": weights, "label_mean": label_mu, "label_std": label_sd}


def predict_linear(row, model, safety=False):
    score = model["label_mean"]
    for feat, weight in model["weights"].items():
        stat = model["stats"][feat]
        score += model["label_std"] * weight * ((row["features"][feat] - stat["mean"]) / stat["std"])
    if safety:
        if row["features"]["valid_json"] < 0.5:
            score -= 1.0
        score -= 0.05 * max(0.0, row["features"]["reason_new_arg_text_count"] - 2.0)
        score -= 0.05 * max(0.0, row["features"]["reason_new_event_type_count"] - 1.0)
        score -= 0.05 * max(0.0, row["features"]["argument_count_delta"] - 2.0)
        score -= 0.10 * max(0.0, row["features"]["event_count_delta"] - 1.0)
    return score


def choose_dev_threshold(dev_rows, model, safety=False):
    by_key = defaultdict(dict)
    for row in dev_rows:
        by_key[row["key"]][row["candidate"]] = row
    scored = []
    for key, candidates in by_key.items():
        direct = candidates["direct_modular_d1930"]
        reason = candidates["reason_modular_r2058"]
        margin = predict_linear(reason, model, safety=safety) - predict_linear(direct, model, safety=safety)
        scored.append(
            {
                "key": key,
                "margin": margin,
                "gain": reason["target_gain"],
            }
        )
    thresholds = sorted({row["margin"] for row in scored})
    thresholds = [max(thresholds) + 1.0] + thresholds + [min(thresholds) - 1.0]
    best = None
    for threshold in thresholds:
        selected = [row for row in scored if row["margin"] >= threshold]
        delta = mean(row["gain"] if row["margin"] >= threshold else 0.0 for row in scored)
        harm_rate = (
            sum(1 for row in selected if row["gain"] < -1e-12) / len(selected)
            if selected
            else 0.0
        )
        reason_rate = len(selected) / len(scored) if scored else 0.0
        # Keep this as a performance-first strong-system selector, but avoid selecting everything.
        feasible = reason_rate <= 0.35
        key = (
            feasible,
            delta,
            -harm_rate,
            -abs(reason_rate - 0.15),
            -reason_rate,
        )
        if best is None or key > best["key"]:
            best = {
                "threshold": threshold,
                "dev_delta": delta,
                "dev_harm_rate": harm_rate,
                "dev_reason_rate": reason_rate,
                "key": key,
            }
    best.pop("key", None)
    return best


def build_dev_rows():
    loaded = [(spec, load_predictions(spec["path"])) for spec in dev_candidate_specs()]
    base = loaded[0][1]
    keys = sorted(set.intersection(*(set(rows) for _, rows in loaded)))
    rows = []
    for key in keys:
        base_row = base[key]
        base_score = metric_score(base_row)
        for spec, pred_rows in loaded:
            cand = pred_rows[key]
            rows.append(
                {
                    "split": "dev_seen",
                    "key": key,
                    "candidate": spec["name"],
                    "role": spec["role"],
                    "target_gain": metric_score(cand) - base_score,
                    "features": candidate_feature_row(base_row, cand, spec),
                }
            )
    return rows


def build_formal_cases():
    cases = []
    all_candidate_rows = []
    for split in SPLITS:
        loaded = [(spec, load_predictions(spec["path"])) for spec in candidate_specs(split)]
        base = loaded[0][1]
        keys = sorted(set.intersection(*(set(rows) for _, rows in loaded)))
        for key in keys:
            candidates = []
            base_row = base[key]
            base_score = metric_score(base_row)
            for spec, pred_rows in loaded:
                row = pred_rows[key]
                features = candidate_feature_row(base_row, row, spec)
                cand = {
                    "name": spec["name"],
                    "role": spec["role"],
                    "row": row,
                    "metrics": row_metrics(row),
                    "target_gain": metric_score(row) - base_score,
                    "features": features,
                }
                candidates.append(cand)
                all_candidate_rows.append(
                    {
                        "split": split,
                        "key": key,
                        "candidate": spec["name"],
                        "role": spec["role"],
                        "target_gain": cand["target_gain"],
                        "features": features,
                        "metrics": cand["metrics"],
                    }
                )
            cases.append({"split": split, "key": key, "base": base_row, "candidates": candidates})
    write_jsonl(CASE_JSONL, all_candidate_rows)
    return cases


def choose_candidate(case, policy, model=None):
    candidates = case["candidates"]
    by_name = {cand["name"]: cand for cand in candidates}
    if policy == "direct_only":
        return by_name["direct_modular_d1930"]
    if policy == "reason_only":
        return by_name["reason_modular_r2058"]
    if policy in by_name:
        return by_name[policy]
    if policy == "oracle_best":
        return max(candidates, key=lambda cand: (cand["metrics"]["score"], cand["metrics"]["event_f1"], cand["metrics"]["argument_f1"]))
    if policy == "dev_linear_rerank":
        return max(candidates, key=lambda cand: predict_linear(cand, model["linear"], safety=False))
    if policy == "dev_safety_linear_rerank":
        return max(candidates, key=lambda cand: predict_linear(cand, model["linear"], safety=True))
    if policy == "dev_linear_threshold":
        direct = by_name["direct_modular_d1930"]
        best = max(
            [cand for cand in candidates if cand["name"] != "direct_modular_d1930"],
            key=lambda cand: predict_linear(cand, model["linear"], safety=False),
        )
        margin = predict_linear(best, model["linear"], safety=False) - predict_linear(direct, model["linear"], safety=False)
        return best if margin >= model["thresholds"]["linear"]["threshold"] else direct
    if policy == "dev_safety_linear_threshold":
        direct = by_name["direct_modular_d1930"]
        best = max(
            [cand for cand in candidates if cand["name"] != "direct_modular_d1930"],
            key=lambda cand: predict_linear(cand, model["linear"], safety=True),
        )
        margin = predict_linear(best, model["linear"], safety=True) - predict_linear(direct, model["linear"], safety=True)
        return best if margin >= model["thresholds"]["safety"]["threshold"] else direct
    if policy.startswith("dev_linear_top"):
        budget = int(policy.removeprefix("dev_linear_top"))
        return choose_budget_candidate(case, model, safety=False, budget=budget)
    if policy.startswith("dev_safety_linear_top"):
        budget = int(policy.removeprefix("dev_safety_linear_top"))
        return choose_budget_candidate(case, model, safety=True, budget=budget)
    raise ValueError(policy)


def candidate_margin(case, model, safety=False):
    candidates = case["candidates"]
    by_name = {cand["name"]: cand for cand in candidates}
    direct = by_name["direct_modular_d1930"]
    best = max(
        [cand for cand in candidates if cand["name"] != "direct_modular_d1930"],
        key=lambda cand: predict_linear(cand, model["linear"], safety=safety),
    )
    margin = predict_linear(best, model["linear"], safety=safety) - predict_linear(direct, model["linear"], safety=safety)
    return margin, best


def choose_budget_candidate(case, model, safety=False, budget=30):
    selected = model["formal_budget_selected"]["safety" if safety else "linear"].get(budget, set())
    if case["key"] not in selected:
        return {cand["name"]: cand for cand in case["candidates"]}["direct_modular_d1930"]
    return candidate_margin(case, model, safety=safety)[1]


def route_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize_policy_split(cases, policy, model=None):
    selected_rows = []
    selected_candidates = []
    direct_rows = []
    helpful_keys = set()
    selected_non_direct = set()
    selected_helpful = set()
    for case in cases:
        direct = choose_candidate(case, "direct_only")
        chosen = choose_candidate(case, policy, model)
        direct_rows.append(direct["row"])
        selected_rows.append(chosen["row"])
        selected_candidates.append(chosen)
        if max(cand["metrics"]["score"] for cand in case["candidates"]) > direct["metrics"]["score"] + 1e-12:
            helpful_keys.add(case["key"])
        if chosen["name"] != "direct_modular_d1930":
            selected_non_direct.add(case["key"])
        if chosen["metrics"]["score"] > direct["metrics"]["score"] + 1e-12:
            selected_helpful.add(case["key"])

    summary = summarize_rows(selected_rows)
    direct_summary = summarize_rows(direct_rows)
    deltas = {
        metric: summary[metric] - direct_summary[metric]
        for metric in ["json_valid_rate", "trigger_f1", "argument_f1", "event_f1", "score"]
    }
    tp = len(selected_non_direct & helpful_keys)
    fp = len(selected_non_direct - helpful_keys)
    fn = len(helpful_keys - selected_non_direct)
    non_direct = [cand for cand in selected_candidates if cand["name"] != "direct_modular_d1930"]
    counts = Counter(cand["name"] for cand in selected_candidates)
    return {
        "policy": policy,
        "num_examples": len(cases),
        "candidate_counts": dict(counts),
        "non_direct_count": len(non_direct),
        "non_direct_rate": len(non_direct) / len(cases) if cases else 0.0,
        "summary": summary,
        "direct": direct_summary,
        "delta_vs_direct": deltas,
        "helpful_any_count": len(helpful_keys),
        "helpful_any_rate": len(helpful_keys) / len(cases) if cases else 0.0,
        "selected_helpful_count": len(selected_helpful),
        "selected_harm_count": sum(1 for cand in non_direct if cand["target_gain"] < -1e-12),
        "selected_harm_rate": (
            sum(1 for cand in non_direct if cand["target_gain"] < -1e-12) / len(non_direct) if non_direct else 0.0
        ),
        "selected_non_direct_gain_mean": mean(cand["target_gain"] for cand in non_direct) if non_direct else 0.0,
        "selection_vs_any_helpful": route_prf(tp, fp, fn),
    }


def aggregate_split(rows):
    total = sum(row["num_examples"] for row in rows)
    out = {
        "split": "test",
        "policy": rows[0]["policy"],
        "num_examples": total,
        "candidate_counts": dict(sum((Counter(row["candidate_counts"]) for row in rows), Counter())),
        "non_direct_count": sum(row["non_direct_count"] for row in rows),
        "helpful_any_count": sum(row["helpful_any_count"] for row in rows),
        "selected_helpful_count": sum(row["selected_helpful_count"] for row in rows),
        "selected_harm_count": sum(row["selected_harm_count"] for row in rows),
    }
    out["non_direct_rate"] = out["non_direct_count"] / total if total else 0.0
    out["helpful_any_rate"] = out["helpful_any_count"] / total if total else 0.0
    out["selected_harm_rate"] = out["selected_harm_count"] / out["non_direct_count"] if out["non_direct_count"] else 0.0
    out["selected_non_direct_gain_mean"] = (
        sum(row["selected_non_direct_gain_mean"] * row["non_direct_count"] for row in rows) / out["non_direct_count"]
        if out["non_direct_count"]
        else 0.0
    )
    for name in ["summary", "direct", "delta_vs_direct"]:
        out[name] = {}
        for metric in ["json_valid_rate", "trigger_f1", "argument_f1", "event_f1", "score"]:
            if name == "delta_vs_direct":
                continue
            out[name][metric] = sum(row[name][metric] * row["num_examples"] for row in rows) / total
        if name != "delta_vs_direct":
            out[name]["num_examples"] = total
    out["delta_vs_direct"] = {
        metric: out["summary"][metric] - out["direct"][metric]
        for metric in ["json_valid_rate", "trigger_f1", "argument_f1", "event_f1", "score"]
    }
    tp = sum(row["selection_vs_any_helpful"]["precision"] * row["non_direct_count"] for row in rows)
    fp = out["non_direct_count"] - tp
    fn = out["helpful_any_count"] - tp
    out["selection_vs_any_helpful"] = route_prf(tp, fp, fn)
    return out


POLICIES = [
    "direct_only",
    "reason_only",
    "free_type_plan_reason_best",
    "free_type_plan_direct_anchor",
    "oracle_best",
    "dev_linear_rerank",
    "dev_safety_linear_rerank",
    "dev_linear_threshold",
    "dev_safety_linear_threshold",
    "dev_linear_top15",
    "dev_linear_top30",
    "dev_linear_top50",
    "dev_safety_linear_top15",
    "dev_safety_linear_top30",
    "dev_safety_linear_top50",
]


def evaluate():
    dev_rows = build_dev_rows()
    linear_model = fit_linear(dev_rows)
    model = {
        "linear": linear_model,
        "thresholds": {
            "linear": choose_dev_threshold(dev_rows, linear_model, safety=False),
            "safety": choose_dev_threshold(dev_rows, linear_model, safety=True),
        },
        "formal_budget_selected": {"linear": {}, "safety": {}},
    }
    cases = build_formal_cases()
    by_split = defaultdict(list)
    for case in cases:
        by_split[case["split"]].append(case)

    for safety_key, safety in [("linear", False), ("safety", True)]:
        ranked = sorted(
            ((case["key"],) + candidate_margin(case, model, safety=safety) for case in cases),
            key=lambda item: item[1],
            reverse=True,
        )
        for budget in [15, 30, 50]:
            model["formal_budget_selected"][safety_key][budget] = {item[0] for item in ranked[:budget]}

    rows = []
    for policy in POLICIES:
        split_rows = []
        for split in SPLITS:
            row = summarize_policy_split(by_split[split], policy, model)
            row["split"] = split
            split_rows.append(row)
            rows.append(row)
        rows.insert(len(rows) - len(split_rows), aggregate_split(split_rows))

    return {
        "id": "2026-05-20_stage2_strong_system_v0_supervised_rerank",
        "output_root": OUTPUT_ROOT.as_posix(),
        "case_jsonl": CASE_JSONL.as_posix(),
        "report_md": REPORT_MD.as_posix(),
        "report_json": REPORT_JSON.as_posix(),
        "dev_training": {
            "num_rows": len(dev_rows),
            "num_cases": len({row["key"] for row in dev_rows}),
            "candidate_counts": dict(Counter(row["candidate"] for row in dev_rows)),
            "model": model,
        },
        "inputs": {
            "formal_candidates": [
                {"split": split, "name": spec["name"], "role": spec["role"], "path": spec["path"].as_posix()}
                for split in SPLITS
                for spec in candidate_specs(split)
            ],
            "dev_candidates": [
                {"name": spec["name"], "role": spec["role"], "path": spec["path"].as_posix()}
                for spec in dev_candidate_specs()
            ],
        },
        "rows": rows,
    }


def json_safe(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def metric_cell(summary):
    return "{}/{}/{}/{}".format(
        fmt(summary["argument_f1"]),
        fmt(summary["event_f1"]),
        fmt(summary["trigger_f1"]),
        fmt(summary["score"]),
    )


def delta_cell(delta):
    return "{}/{}/{}/{}".format(
        signed(delta["argument_f1"]),
        signed(delta["event_f1"]),
        signed(delta["trigger_f1"]),
        signed(delta["score"]),
    )


def prf_cell(prf):
    return "{}/{}/{}".format(fmt(prf["precision"]), fmt(prf["recall"]), fmt(prf["f1"]))


def render_main_table(rows):
    lines = [
        "| split | policy | non-direct | JSON | A/E/T/Score | delta vs direct A/E/T/Score | harm | P/R/F1 vs any helpful | candidates |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["split"] != "test":
            continue
        lines.append(
            "| `{split}` | `{policy}` | {rate} | {json} | {metrics} | {delta} | {harm} | {prf} | {counts} |".format(
                split=row["split"],
                policy=row["policy"],
                rate=pct(row["non_direct_rate"]),
                json=fmt(row["summary"]["json_valid_rate"]),
                metrics=metric_cell(row["summary"]),
                delta=delta_cell(row["delta_vs_direct"]),
                harm=pct(row["selected_harm_rate"]),
                prf=prf_cell(row["selection_vs_any_helpful"]),
                counts=", ".join(f"{k}:{v}" for k, v in sorted(row["candidate_counts"].items())),
            )
        )
    return "\n".join(lines)


def render_split_table(rows):
    lines = [
        "| split | policy | non-direct | score delta | A/E/T delta | harm | candidates |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["split"] == "test":
            continue
        lines.append(
            "| `{split}` | `{policy}` | {rate} | {score} | {delta} | {harm} | {counts} |".format(
                split=row["split"],
                policy=row["policy"],
                rate=pct(row["non_direct_rate"]),
                score=signed(row["delta_vs_direct"]["score"]),
                delta="{}/{}/{}".format(
                    signed(row["delta_vs_direct"]["argument_f1"]),
                    signed(row["delta_vs_direct"]["event_f1"]),
                    signed(row["delta_vs_direct"]["trigger_f1"]),
                ),
                harm=pct(row["selected_harm_rate"]),
                counts=", ".join(f"{k}:{v}" for k, v in sorted(row["candidate_counts"].items())),
            )
        )
    return "\n".join(lines)


def render_report(payload):
    rows = payload["rows"]
    best_non_oracle = max(
        [row for row in rows if row["split"] == "test" and not row["policy"].startswith("oracle")],
        key=lambda row: row["delta_vs_direct"]["score"],
    )
    oracle = next(row for row in rows if row["split"] == "test" and row["policy"] == "oracle_best")
    lines = [
        "# Strong System v0 Supervised Rerank",
        "",
        "This offline replay selects among existing direct, reason, and free-route candidate outputs. Dev supervision is used only to fit the `dev_*` rerankers; formal rows are frozen replay.",
        "",
        "## Main Test Table",
        "",
        render_main_table(rows),
        "",
        "## Split Details",
        "",
        render_split_table(rows),
        "",
        "## Reading",
        "",
        f"- Best non-oracle test policy: `{best_non_oracle['policy']}` with score delta `{signed(best_non_oracle['delta_vs_direct']['score'])}`.",
        f"- Oracle best-of-candidates score delta: `{signed(oracle['delta_vs_direct']['score'])}` with non-direct rate `{pct(oracle['non_direct_rate'])}`.",
        f"- Dev training rows: `{payload['dev_training']['num_rows']}` over `{payload['dev_training']['num_cases']}` dev_seen cases.",
        "",
        "## Artifacts",
        "",
        f"- case table: `{payload['case_jsonl']}`",
        f"- JSON: `{payload['report_json']}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    payload = evaluate()
    payload = json_safe(payload)
    write_text(REPORT_MD, render_report(payload))
    write_json(REPORT_JSON, payload)
    print(json.dumps({"report_md": REPORT_MD.as_posix(), "report_json": REPORT_JSON.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
