#!/usr/bin/env python3
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import prediction_key, score  # noqa: E402
from src.stage2_analysis.analyze_adaptive_outcome_router_execution import load_prediction_map, row_metric  # noqa: E402
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


BRANCH = "multibudget_ternary_router_m08_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT = REPO / "outputs/stage2_multibudget/formal_route_likelihood_20260521" / BRANCH
OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_multibudget_ternary_router_m08_selected_case_drift.json"
OUT_MD = REPO / "reports/2026-05-21_stage2_multibudget_ternary_router_m08_selected_case_drift.md"

DIRECT_ROOT = REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_balrouteaux_20260516/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux/checkpoint-1930/forced_direct"
MID_ROOT = REPO / "outputs/stage2_adaptive_runs_user_formal_clean/richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30/frontier_seen_stable_best/forced_reason"
FULL_ROOT = REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux/checkpoint-2058/forced_reason"

POLICIES = [
    {"name": "m08_balanced_dev_locked", "checkpoint": "checkpoint-700", "start_pct": 0.300, "end_pct": 0.475},
    {"name": "m08_event_arg_dev_locked", "checkpoint": "checkpoint-400", "start_pct": 0.450, "end_pct": 0.500},
    {"name": "m08_stable_lowbudget_dev_locked", "checkpoint": "checkpoint-650", "start_pct": 0.450, "end_pct": 0.500},
]
SPLITS = ["test", "test_seen", "test_unseen"]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str):
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
        route = best_route_for_score_row(row)
        nll = row.get("nll_by_route") or {}
        adv = row.get("best_non_direct_advantage_vs_direct")
        if adv is None:
            adv = nll.get("direct", float("-inf")) - nll.get(route, float("inf"))
        items.append((float(adv), key, route))
    items.sort(reverse=True)
    return items


def choose_row(route, key, direct_rows, mid_rows, full_rows):
    if route == "reason_mid":
        return mid_rows[key]
    if route == "reason_full":
        return full_rows[key]
    return direct_rows[key]


def event_families(row):
    meta = row.get("meta") or {}
    types = meta.get("gold_event_types") or []
    if not types:
        return ["none"]
    return sorted({t.split(":", 1)[0] for t in types if isinstance(t, str)})


def mean(rows, field):
    if not rows:
        return 0.0
    return sum(r[field] for r in rows) / len(rows)


def f1(metric, name):
    return metric[name]["f1"]


def summarize_group(rows):
    return {
        "count": len(rows),
        "helpful": sum(1 for r in rows if r["score_gain"] > 1e-9),
        "harmful": sum(1 for r in rows if r["score_gain"] < -1e-9),
        "neutral": sum(1 for r in rows if abs(r["score_gain"]) <= 1e-9),
        "mean_score_gain": mean(rows, "score_gain"),
        "mean_argument_gain": mean(rows, "argument_gain"),
        "mean_event_gain": mean(rows, "event_gain"),
        "mean_trigger_gain": mean(rows, "trigger_gain"),
    }


def selected_cases(policy, split):
    direct_rows = load_prediction_map(DIRECT_ROOT / split / "predictions.jsonl")
    mid_rows = load_prediction_map(MID_ROOT / split / "predictions.jsonl")
    full_rows = load_prediction_map(FULL_ROOT / split / "predictions.jsonl")
    common_keys = sorted(set(direct_rows) & set(mid_rows) & set(full_rows))
    score_rows = load_score_rows(SCORE_ROOT / policy["checkpoint"] / split / "scores.jsonl")
    ranked = ordered_items(score_rows, common_keys)
    start = round(len(ranked) * policy["start_pct"])
    end = round(len(ranked) * policy["end_pct"])

    cases = []
    for rank0, (adv, key, route) in enumerate(ranked[start:end], start=start):
        direct = direct_rows[key]
        chosen = choose_row(route, key, direct_rows, mid_rows, full_rows)
        direct_m = row_metric(direct)
        chosen_m = row_metric(chosen)
        score_gain = score(chosen) - score(direct)
        meta = direct.get("meta") or {}
        cases.append(
            {
                "rank": rank0 + 1,
                "key": key,
                "wnd_id": meta.get("wnd_id"),
                "route": route,
                "nll_advantage": adv,
                "score_gain": score_gain,
                "argument_gain": f1(chosen_m, "argument") - f1(direct_m, "argument"),
                "event_gain": f1(chosen_m, "event") - f1(direct_m, "event"),
                "trigger_gain": f1(chosen_m, "trigger") - f1(direct_m, "trigger"),
                "gold_event_types": meta.get("gold_event_types") or [],
                "event_families": event_families(direct),
                "candidate_types": meta.get("candidate_types") or [],
            }
        )
    return cases


def diagnose():
    results = []
    for policy in POLICIES:
        for split in SPLITS:
            cases = selected_cases(policy, split)
            by_route = defaultdict(list)
            by_family = defaultdict(list)
            for case in cases:
                by_route[case["route"]].append(case)
                for family in case["event_families"]:
                    by_family[family].append(case)
            results.append(
                {
                    "policy": policy["name"],
                    "checkpoint": policy["checkpoint"],
                    "split": split,
                    "rank_window": {"start_pct": policy["start_pct"], "end_pct": policy["end_pct"]},
                    "summary": summarize_group(cases),
                    "route_counts": dict(Counter(c["route"] for c in cases)),
                    "by_route": {k: summarize_group(v) for k, v in sorted(by_route.items())},
                    "by_family": {k: summarize_group(v) for k, v in sorted(by_family.items())},
                    "most_harmful": sorted(cases, key=lambda c: c["score_gain"])[:8],
                    "most_helpful": sorted(cases, key=lambda c: c["score_gain"], reverse=True)[:8],
                }
            )
    return {"branch": BRANCH, "score_root": SCORE_ROOT.as_posix(), "results": results}


def fmt_summary(s):
    return (
        f"{s['count']} | {s['helpful']}/{s['harmful']}/{s['neutral']} | "
        f"{s['mean_score_gain']:+.4f} | "
        f"{s['mean_argument_gain']:+.4f}/{s['mean_event_gain']:+.4f}/{s['mean_trigger_gain']:+.4f}"
    )


def render(payload):
    lines = [
        "# M08 Selected-Case Drift Diagnosis",
        "",
        "This is a formal diagnostic report. It explains failures after dev-locked policy selection; it is not a new formal-tuned selector.",
        "",
        "| policy | split | selected | helpful/harmful/neutral | mean score gain | mean A/E/T gain | route counts |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        s = row["summary"]
        counts = ",".join(f"{k}={v}" for k, v in sorted(row["route_counts"].items()))
        lines.append(
            f"| {row['policy']} | {row['split']} | {s['count']} | "
            f"{s['helpful']}/{s['harmful']}/{s['neutral']} | {s['mean_score_gain']:+.4f} | "
            f"{s['mean_argument_gain']:+.4f}/{s['mean_event_gain']:+.4f}/{s['mean_trigger_gain']:+.4f} | {counts} |"
        )

    lines.extend(["", "## Event-Family Diagnostics", ""])
    for row in payload["results"]:
        lines.append(f"### {row['policy']} / {row['split']}")
        for family, summary in row["by_family"].items():
            lines.append(f"- `{family}`: {fmt_summary(summary)}")
        lines.append("")

    lines.extend(["## Most Harmful Cases", ""])
    for row in payload["results"]:
        lines.append(f"### {row['policy']} / {row['split']}")
        for case in row["most_harmful"][:5]:
            lines.append(
                "- rank `{rank}` route `{route}` score `{score_gain:+.4f}` A/E/T `{a:+.4f}/{e:+.4f}/{t:+.4f}` gold `{gold}` wnd `{wnd}`".format(
                    rank=case["rank"],
                    route=case["route"],
                    score_gain=case["score_gain"],
                    a=case["argument_gain"],
                    e=case["event_gain"],
                    t=case["trigger_gain"],
                    gold=",".join(case["gold_event_types"]) or "none",
                    wnd=case["wnd_id"],
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    payload = diagnose()
    write_json(OUT_JSON, payload)
    write_text(OUT_MD, render(payload))
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
