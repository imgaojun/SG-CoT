#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import prediction_key, score  # noqa: E402
from src.stage2_analysis.analyze_adaptive_outcome_router_execution import (  # noqa: E402
    load_prediction_map,
    row_metric,
    summarize_metrics,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


BRANCH = "multibudget_ternary_router_m08_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT = REPO / "outputs/stage2_multibudget/formal_route_likelihood_20260521" / BRANCH
OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_multibudget_ternary_router_m08_formal.json"
OUT_MD = REPO / "reports/2026-05-21_stage2_multibudget_ternary_router_m08_formal.md"

DIRECT_ROOT = REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_balrouteaux_20260516/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux/checkpoint-1930/forced_direct"
MID_ROOT = REPO / "outputs/stage2_adaptive_runs_user_formal_clean/richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30/frontier_seen_stable_best/forced_reason"
FULL_ROOT = REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux/checkpoint-2058/forced_reason"

POLICIES = [
    {
        "name": "m08_balanced_dev_locked",
        "checkpoint": "checkpoint-700",
        "start_pct": 0.300,
        "end_pct": 0.475,
        "dev_delta_aet": "+0.0119/+0.0115/+0.0127",
    },
    {
        "name": "m08_event_arg_dev_locked",
        "checkpoint": "checkpoint-400",
        "start_pct": 0.450,
        "end_pct": 0.500,
        "dev_delta_aet": "+0.0127/+0.0152/+0.0017",
    },
    {
        "name": "m08_stable_lowbudget_dev_locked",
        "checkpoint": "checkpoint-650",
        "start_pct": 0.450,
        "end_pct": 0.500,
        "dev_delta_aet": "+0.0100/+0.0102/+0.0068",
    },
]
SPLITS = ["test", "test_seen", "test_unseen"]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_score_rows(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def best_route_for_score_row(row):
    route = row.get("best_non_direct_route")
    if route in {"reason_mid", "reason_full"}:
        return route
    nll = row.get("nll_by_route") or {}
    non_direct = [r for r in ["reason_mid", "reason_full"] if r in nll]
    if not non_direct:
        return "direct"
    return min(non_direct, key=lambda r: nll[r])


def ordered_items(score_rows, common_keys):
    items = []
    for key in set(common_keys) & set(score_rows):
        row = score_rows[key]
        adv = row.get("best_non_direct_advantage_vs_direct")
        if adv is None:
            nll = row.get("nll_by_route") or {}
            route = best_route_for_score_row(row)
            adv = nll.get("direct", float("-inf")) - nll.get(route, float("inf"))
        items.append((float(adv), key, best_route_for_score_row(row)))
    items.sort(reverse=True)
    return items


def choose_row(route, key, direct_rows, mid_rows, full_rows):
    if route == "reason_mid":
        return mid_rows[key]
    if route == "reason_full":
        return full_rows[key]
    return direct_rows[key]


def evaluate(policy, split):
    direct_rows = load_prediction_map(DIRECT_ROOT / split / "predictions.jsonl")
    mid_rows = load_prediction_map(MID_ROOT / split / "predictions.jsonl")
    full_rows = load_prediction_map(FULL_ROOT / split / "predictions.jsonl")
    common_keys = sorted(set(direct_rows) & set(mid_rows) & set(full_rows))
    score_rows = load_score_rows(SCORE_ROOT / policy["checkpoint"] / split / "scores.jsonl")
    ranked = ordered_items(score_rows, common_keys)
    n = len(ranked)
    start = round(n * policy["start_pct"])
    end = round(n * policy["end_pct"])
    selected = {key: route for _, key, route in ranked[start:end]}

    routed_metrics = []
    direct_metrics = []
    mid_metrics = []
    full_metrics = []
    route_counts = {"direct": 0, "reason_mid": 0, "reason_full": 0}
    selected_gain = []
    for _, key, _ in ranked:
        route = selected.get(key, "direct")
        route_counts[route] += 1
        chosen = choose_row(route, key, direct_rows, mid_rows, full_rows)
        direct = direct_rows[key]
        routed_metrics.append(row_metric(chosen))
        direct_metrics.append(row_metric(direct))
        mid_metrics.append(row_metric(mid_rows[key]))
        full_metrics.append(row_metric(full_rows[key]))
        if route != "direct":
            selected_gain.append(score(chosen) - score(direct))

    direct_summary = summarize_metrics(direct_metrics)
    routed = summarize_metrics(routed_metrics)
    mid = summarize_metrics(mid_metrics)
    full = summarize_metrics(full_metrics)
    return {
        **policy,
        "split": split,
        "num_examples": len(ranked),
        "rank_window": {"start_rank": start + 1, "end_rank": end, "start_pct": policy["start_pct"], "end_pct": policy["end_pct"]},
        "route_counts": route_counts,
        "reason_rate": (route_counts["reason_mid"] + route_counts["reason_full"]) / len(ranked) if ranked else 0.0,
        "selected_mean_score_gain": sum(selected_gain) / len(selected_gain) if selected_gain else 0.0,
        "direct": direct_summary,
        "forced_mid_all": mid,
        "forced_full_all": full,
        "routed": routed,
        "routed_delta_vs_direct": {
            "argument_f1": routed["argument_f1"] - direct_summary["argument_f1"],
            "event_f1": routed["event_f1"] - direct_summary["event_f1"],
            "trigger_f1": routed["trigger_f1"] - direct_summary["trigger_f1"],
        },
    }


def fmt_delta(delta):
    return "{argument_f1:+.4f} / {event_f1:+.4f} / {trigger_f1:+.4f}".format(**delta)


def render(payload):
    lines = [
        "# Multibudget Ternary Router M08 Formal Replay",
        "",
        "Formal replay uses dev-locked policies only. No formal labels were used to choose checkpoint or rank window.",
        "",
        "| policy | split | reason rate | route counts | delta A/E/T | routed A/E/T |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        counts = row["route_counts"]
        routed = row["routed"]
        lines.append(
            "| {name} | {split} | {rate:.1%} | d/m/f={d}/{m}/{f} | {delta} | {a:.4f} / {e:.4f} / {t:.4f} |".format(
                name=row["name"],
                split=row["split"],
                rate=row["reason_rate"],
                d=counts["direct"],
                m=counts["reason_mid"],
                f=counts["reason_full"],
                delta=fmt_delta(row["routed_delta_vs_direct"]),
                a=routed["argument_f1"],
                e=routed["event_f1"],
                t=routed["trigger_f1"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    results = []
    for policy in POLICIES:
        for split in SPLITS:
            results.append(evaluate(policy, split))
    payload = {
        "branch": BRANCH,
        "score_root": SCORE_ROOT.as_posix(),
        "policies": POLICIES,
        "results": results,
    }
    write_json(OUT_JSON, payload)
    write_text(OUT_MD, render(payload))
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
