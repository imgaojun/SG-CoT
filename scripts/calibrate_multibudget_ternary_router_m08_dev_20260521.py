#!/usr/bin/env python3
import json
import sys
from hashlib import md5
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import prediction_key, score  # noqa: E402
from src.stage2_analysis.analyze_adaptive_outcome_router_execution import (  # noqa: E402
    load_prediction_map,
    route_prf,
    row_metric,
    summarize_metrics,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


BRANCH = "multibudget_ternary_router_m08_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT = REPO / "outputs/stage2_multibudget/route_likelihood_20260521" / BRANCH
DIRECT_DEV = REPO / "outputs/stage2_adaptive_runs_user_devpick_frontier/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux_full_forced_direct_dev_seen_max512/checkpoint-1930/predictions.jsonl"
MID_DEV = REPO / "outputs/stage2_adaptive_outcome_mining/l15bal30_ckpt942/forced_reason/dev_seen/predictions.jsonl"
FULL_DEV = REPO / "outputs/stage2_adaptive_runs_user_devpick_frontier/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux_full_forced_reason_dev_seen_max512/checkpoint-2058/predictions.jsonl"
OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_multibudget_ternary_router_m08_dev.json"
OUT_MD = REPO / "reports/2026-05-21_stage2_multibudget_ternary_router_m08_dev.md"

WINDOW_ENDPOINTS = [i / 40 for i in range(0, 21)]  # 0.00..0.50 by 2.5%.
TARGET_MIN_RATE = 0.04
TARGET_MAX_RATE = 0.20


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ckpt_num(path: Path) -> int:
    return int(path.parent.name.split("-", 1)[1])


def load_score_rows(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def metric_delta(direct_row, chosen_row):
    direct_m = row_metric(direct_row)
    chosen_m = row_metric(chosen_row)
    return {
        "trigger_f1": chosen_m["trigger"]["f1"] - direct_m["trigger"]["f1"],
        "argument_f1": chosen_m["argument"]["f1"] - direct_m["argument"]["f1"],
        "event_f1": chosen_m["event"]["f1"] - direct_m["event"]["f1"],
        "score": score(chosen_row) - score(direct_row),
    }


def candidate_rows(route, direct_rows, mid_rows, full_rows):
    if route == "reason_mid":
        return mid_rows
    if route == "reason_full":
        return full_rows
    return direct_rows


def best_route_for_score_row(row):
    route = row.get("best_non_direct_route")
    if route in {"reason_mid", "reason_full"}:
        return route
    nll = row.get("nll_by_route") or {}
    non_direct = [r for r in ["reason_mid", "reason_full"] if r in nll]
    if not non_direct:
        return "direct"
    return min(non_direct, key=lambda r: nll[r])


def sorted_route_decisions(score_rows, common_keys):
    items = []
    for key in set(common_keys) & set(score_rows):
        row = score_rows[key]
        adv = row.get("best_non_direct_advantage_vs_direct")
        if adv is None:
            nll = row.get("nll_by_route") or {}
            if "direct" in nll:
                route = best_route_for_score_row(row)
                adv = nll["direct"] - nll.get(route, float("inf"))
            else:
                adv = float("-inf")
        items.append((float(adv), key, best_route_for_score_row(row)))
    items.sort(reverse=True)
    return items


def stable_fold(key):
    return int(md5(key.encode("utf-8")).hexdigest()[:8], 16) % 5


def evaluate_policy(name, ckpt, ordered_items, selected_items, score_rows, direct_rows, mid_rows, full_rows):
    selected = {key: route for _, key, route in selected_items}
    keys = [key for _, key, _ in ordered_items]
    routed_metrics = []
    direct_metrics = []
    mid_metrics = []
    full_metrics = []
    selected_deltas = []
    selected_examples = []
    route_counts = {"direct": 0, "reason_mid": 0, "reason_full": 0}
    label_tp = label_fp = label_fn = 0
    helpful_tp = helpful_fp = helpful_fn = helpful_count = 0
    correct = 0

    for rank, key in enumerate(keys, start=1):
        direct_row = direct_rows[key]
        mid_row = mid_rows[key]
        full_row = full_rows[key]
        score_row = score_rows[key]
        exec_route = selected.get(key, "direct")
        gold = score_row.get("gold_route") or "unknown"
        chosen_row = candidate_rows(exec_route, direct_rows, mid_rows, full_rows)[key]
        best_non_direct_row = candidate_rows(best_route_for_score_row(score_row), direct_rows, mid_rows, full_rows)[key]
        helpful = score(best_non_direct_row) - score(direct_row) > 0
        if helpful:
            helpful_count += 1
        if exec_route == gold:
            correct += 1
        if exec_route != "direct" and gold != "direct":
            label_tp += 1
        elif exec_route != "direct" and gold == "direct":
            label_fp += 1
        elif exec_route == "direct" and gold != "direct":
            label_fn += 1
        if exec_route != "direct" and helpful:
            helpful_tp += 1
        elif exec_route != "direct" and not helpful:
            helpful_fp += 1
        elif exec_route == "direct" and helpful:
            helpful_fn += 1

        route_counts[exec_route] += 1
        routed_metrics.append(row_metric(chosen_row))
        direct_metrics.append(row_metric(direct_row))
        mid_metrics.append(row_metric(mid_row))
        full_metrics.append(row_metric(full_row))
        if exec_route != "direct":
            delta = metric_delta(direct_row, chosen_row)
            selected_deltas.append(delta)
            selected_examples.append(
                {
                    "rank": rank,
                    "wnd_id": key,
                    "exec_route": exec_route,
                    "gold_route": gold,
                    "advantage": score_row.get("best_non_direct_advantage_vs_direct"),
                    **delta,
                }
            )

    routed = summarize_metrics(routed_metrics)
    direct = summarize_metrics(direct_metrics)
    mid = summarize_metrics(mid_metrics)
    full = summarize_metrics(full_metrics)
    return {
        "name": name,
        "checkpoint": ckpt,
        "num_examples": len(keys),
        "route_counts": route_counts,
        "reason_count": route_counts["reason_mid"] + route_counts["reason_full"],
        "reason_rate": (route_counts["reason_mid"] + route_counts["reason_full"]) / len(keys) if keys else 0.0,
        "route_accuracy_vs_label": correct / len(keys) if keys else 0.0,
        "route_vs_non_direct_label": route_prf(label_tp, label_fp, label_fn),
        "positive_non_direct_helpful_count": helpful_count,
        "route_vs_positive_non_direct_helpful": route_prf(helpful_tp, helpful_fp, helpful_fn),
        "direct": direct,
        "forced_mid_all": mid,
        "forced_full_all": full,
        "routed": routed,
        "routed_delta_vs_direct": {
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
        },
        "selected_delta_mean": mean_dict(selected_deltas),
        "selected_examples": sorted(
            selected_examples,
            key=lambda row: row["advantage"] if row["advantage"] is not None else float("-inf"),
            reverse=True,
        )[:20],
    }


def mean_dict(rows):
    keys = ["trigger_f1", "argument_f1", "event_f1", "score"]
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def add_fold_floor(row, ordered_items, selected_items, score_rows, direct_rows, mid_rows, full_rows):
    selected = {(key, route) for _, key, route in selected_items}
    folds = []
    for fold in range(5):
        fold_ordered = [item for item in ordered_items if stable_fold(item[1]) == fold]
        fold_selected = [item for item in fold_ordered if (item[1], item[2]) in selected]
        if not fold_ordered:
            continue
        fold_row = evaluate_policy(
            f"{row['name']}_fold{fold}",
            row["checkpoint"],
            fold_ordered,
            fold_selected,
            score_rows,
            direct_rows,
            mid_rows,
            full_rows,
        )
        d = fold_row["routed_delta_vs_direct"]
        folds.append(
            {
                "fold": fold,
                "reason_rate": fold_row["reason_rate"],
                "delta": d,
                "min_aet": min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
            }
        )
    row["folds"] = folds
    row["fold_min_aet"] = min((fold["min_aet"] for fold in folds), default=-99.0)
    return row


def sweep(direct_rows, mid_rows, full_rows, common_keys):
    rows = []
    for score_path in sorted(SCORE_ROOT.glob("checkpoint-*/dev_seen_scores.jsonl"), key=ckpt_num):
        ckpt = score_path.parent.name
        score_rows = load_score_rows(score_path)
        ordered = sorted_route_decisions(score_rows, common_keys)
        n = len(ordered)
        for lo in WINDOW_ENDPOINTS:
            for hi in WINDOW_ENDPOINTS:
                if hi <= lo:
                    continue
                rate = hi - lo
                if rate < TARGET_MIN_RATE or rate > TARGET_MAX_RATE:
                    continue
                start = round(n * lo)
                end = round(n * hi)
                selected = ordered[start:end]
                if not selected:
                    continue
                row = evaluate_policy(
                    f"{ckpt}_rank{int(lo * 1000):03d}_{int(hi * 1000):03d}",
                    ckpt,
                    ordered,
                    selected,
                    score_rows,
                    direct_rows,
                    mid_rows,
                    full_rows,
                )
                row["branch"] = BRANCH
                row["rank_window"] = {
                    "start_pct": lo,
                    "end_pct": hi,
                    "start_rank": start + 1,
                    "end_rank": end,
                }
                rows.append(add_fold_floor(row, ordered, selected, score_rows, direct_rows, mid_rows, full_rows))
    return rows


def is_all_nonnegative(row):
    d = row["routed_delta_vs_direct"]
    return d["argument_f1"] >= 0 and d["event_f1"] >= 0 and d["trigger_f1"] >= 0


def balanced_score(row):
    d = row["routed_delta_vs_direct"]
    ckpt = int(row["checkpoint"].split("-", 1)[1])
    return (
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        row["fold_min_aet"],
        d["event_f1"],
        d["argument_f1"],
        -row["reason_rate"],
        -ckpt,
    )


def event_arg_score(row):
    d = row["routed_delta_vs_direct"]
    ckpt = int(row["checkpoint"].split("-", 1)[1])
    return (
        d["event_f1"] + d["argument_f1"],
        min(d["argument_f1"], d["event_f1"]),
        d["trigger_f1"],
        row["fold_min_aet"],
        -row["reason_rate"],
        -ckpt,
    )


def fmt_delta(delta):
    return "{argument_f1:+.4f} / {event_f1:+.4f} / {trigger_f1:+.4f}".format(**delta)


def render_table(rows):
    lines = [
        "| policy | reason rate | route counts | delta A/E/T | fold floor | routed A/E/T |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        routed = row["routed"]
        counts = row["route_counts"]
        lines.append(
            "| {name} | {rate:.1%} | d/m/f={d}/{m}/{f} | {delta} | {floor:+.4f} | {a:.4f} / {e:.4f} / {t:.4f} |".format(
                name=row["name"],
                rate=row["reason_rate"],
                d=counts["direct"],
                m=counts["reason_mid"],
                f=counts["reason_full"],
                delta=fmt_delta(row["routed_delta_vs_direct"]),
                floor=row["fold_min_aet"],
                a=routed["argument_f1"],
                e=routed["event_f1"],
                t=routed["trigger_f1"],
            )
        )
    return "\n".join(lines)


def render_report(payload):
    lines = [
        "# Multibudget Ternary Router M08 Dev Sweep",
        "",
        "This sweep ranks examples by best non-direct route-choice NLL advantage and executes the per-example best non-direct budget inside each selected rank window.",
        "",
        "## Locked Candidates",
        "",
        render_table([payload["balanced_candidate"], payload["event_arg_candidate"]]),
        "",
        "## Top All-Nonnegative",
        "",
        render_table(payload["top_all_nonnegative"][:20]),
        "",
        "## Top Event/Argument",
        "",
        render_table(payload["top_event_arg"][:20]),
        "",
    ]
    return "\n".join(lines)


def main():
    direct_rows = load_prediction_map(DIRECT_DEV)
    mid_rows = load_prediction_map(MID_DEV)
    full_rows = load_prediction_map(FULL_DEV)
    common_keys = sorted(set(direct_rows) & set(mid_rows) & set(full_rows))
    rows = sweep(direct_rows, mid_rows, full_rows, common_keys)
    all_nonnegative = [row for row in rows if is_all_nonnegative(row)]
    if not all_nonnegative:
        raise RuntimeError("no all-nonnegative M08 dev candidate")
    balanced = max(all_nonnegative, key=balanced_score)
    event_arg = max(all_nonnegative, key=event_arg_score)
    payload = {
        "selection_metric": "dev-only multiroute NLL rank-window replay; formal not used",
        "branch": BRANCH,
        "score_root": SCORE_ROOT.as_posix(),
        "num_candidates": len(rows),
        "num_all_nonnegative": len(all_nonnegative),
        "balanced_candidate": balanced,
        "event_arg_candidate": event_arg,
        "top_all_nonnegative": sorted(all_nonnegative, key=balanced_score, reverse=True)[:50],
        "top_event_arg": sorted(all_nonnegative, key=event_arg_score, reverse=True)[:50],
        "all_candidates": rows,
    }
    write_json(OUT_JSON, payload)
    write_text(OUT_MD, render_report(payload))
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
