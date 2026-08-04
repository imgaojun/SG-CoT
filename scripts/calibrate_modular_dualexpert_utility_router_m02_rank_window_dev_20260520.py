import json
import sys
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


BRANCH = "modular_d1930_r2058_utility_m02_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/utility_router_m02_20260520/route_likelihood" / BRANCH
DIRECT_DEV = REPO / (
    "outputs/stage2_adaptive_runs_user_devpick_frontier/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux_"
    "full_forced_direct_dev_seen_max512/checkpoint-1930/predictions.jsonl"
)
REASON_DEV = REPO / (
    "outputs/stage2_adaptive_runs_user_devpick_frontier/"
    "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_"
    "outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux_"
    "full_forced_reason_dev_seen_max512/checkpoint-2058/predictions.jsonl"
)
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_utility_router_m02_rank_window_calibration_dev.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_utility_router_m02_rank_window_calibration_dev.md"


FIXED_WINDOWS = [
    (0.00, 0.10),
    (0.00, 0.15),
    (0.00, 0.20),
    (0.00, 0.30),
    (0.10, 0.15),
    (0.10, 0.20),
    (0.10, 0.30),
    (0.15, 0.30),
    (0.20, 0.30),
    (0.20, 0.40),
]


def ckpt_num(name: str) -> int:
    return int(name.split("-", 1)[1])


def load_score_rows(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def sorted_keys_by_delta(score_rows, common_keys):
    scored = []
    for key in set(common_keys) & set(score_rows):
        delta = score_rows[key].get("delta_direct_minus_reason_route_nll")
        if delta is None:
            delta = float("-inf")
        scored.append((float(delta), key))
    scored.sort(reverse=True)
    return [key for _, key in scored]


def metric_delta(direct_row, reason_row):
    direct_m = row_metric(direct_row)
    reason_m = row_metric(reason_row)
    return {
        "trigger_f1": reason_m["trigger"]["f1"] - direct_m["trigger"]["f1"],
        "argument_f1": reason_m["argument"]["f1"] - direct_m["argument"]["f1"],
        "event_f1": reason_m["event"]["f1"] - direct_m["event"]["f1"],
        "score": score(reason_row) - score(direct_row),
    }


def evaluate_policy(name, ckpt, keys, selected_keys, score_rows, direct_rows, reason_rows):
    selected = set(selected_keys)
    routed_metrics = []
    direct_metrics = []
    reason_metrics = []
    label_tp = label_fp = label_fn = correct = 0
    helpful_tp = helpful_fp = helpful_fn = helpful_count = 0
    selected_deltas = []
    selected_examples = []

    for rank, key in enumerate(keys, start=1):
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        score_row = score_rows[key]
        gold = score_row.get("gold_route") or "unknown"
        exec_route = "reason" if key in selected else "direct"
        gain = score(reason_row) - score(direct_row)
        helpful = gain > 0
        if helpful:
            helpful_count += 1
        if exec_route == gold:
            correct += 1
        if exec_route == "reason" and gold == "reason":
            label_tp += 1
        elif exec_route == "reason" and gold != "reason":
            label_fp += 1
        elif exec_route != "reason" and gold == "reason":
            label_fn += 1
        if exec_route == "reason" and helpful:
            helpful_tp += 1
        elif exec_route == "reason" and not helpful:
            helpful_fp += 1
        elif exec_route != "reason" and helpful:
            helpful_fn += 1
        chosen = reason_row if exec_route == "reason" else direct_row
        routed_metrics.append(row_metric(chosen))
        direct_metrics.append(row_metric(direct_row))
        reason_metrics.append(row_metric(reason_row))
        if exec_route == "reason":
            delta = metric_delta(direct_row, reason_row)
            selected_deltas.append(delta)
            selected_examples.append(
                {
                    "rank": rank,
                    "wnd_id": key,
                    "gold_route": gold,
                    "delta_direct_minus_reason_route_nll": score_row.get("delta_direct_minus_reason_route_nll"),
                    **delta,
                }
            )

    routed = summarize_metrics(routed_metrics)
    direct = summarize_metrics(direct_metrics)
    forced_reason = summarize_metrics(reason_metrics)
    label_prf = route_prf(label_tp, label_fp, label_fn)
    helpful_prf = route_prf(helpful_tp, helpful_fp, helpful_fn)
    total = len(keys)
    result = {
        "name": name,
        "checkpoint": ckpt,
        "num_examples": total,
        "reason_count": len(selected),
        "reason_rate": len(selected) / total if total else 0.0,
        "route_accuracy_vs_label": correct / total if total else 0.0,
        "route_vs_label": label_prf,
        "positive_reason_helpful_count": helpful_count,
        "route_vs_positive_reason_helpful": helpful_prf,
        "direct": direct,
        "forced_reason_all": forced_reason,
        "routed": routed,
        "routed_delta_vs_direct": {
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
        },
        "selected_delta_mean": mean_dict(selected_deltas),
        "selected_examples": sorted(
            selected_examples,
            key=lambda row: row["delta_direct_minus_reason_route_nll"]
            if row["delta_direct_minus_reason_route_nll"] is not None
            else float("-inf"),
            reverse=True,
        )[:20],
    }
    return result


def mean_dict(rows):
    keys = ["trigger_f1", "argument_f1", "event_f1", "score"]
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def fixed_window_results(ckpt, keys, score_rows, direct_rows, reason_rows):
    rows = []
    n = len(keys)
    for lo, hi in FIXED_WINDOWS:
        start = round(n * lo)
        end = round(n * hi)
        selected = keys[start:end]
        if not selected:
            continue
        name = f"{ckpt}_rank{int(lo * 100):02d}_{int(hi * 100):02d}"
        row = evaluate_policy(name, ckpt, keys, selected, score_rows, direct_rows, reason_rows)
        row["rank_window"] = {"start_pct": lo, "end_pct": hi, "start_rank": start + 1, "end_rank": end}
        row["policy_family"] = "fixed_rank_window"
        rows.append(row)
    return rows


def exhaustive_window_results(ckpt, keys, score_rows, direct_rows, reason_rows):
    rows = []
    n = len(keys)
    endpoints = sorted({0, round(n * 0.05), round(n * 0.10), round(n * 0.15), round(n * 0.20), round(n * 0.25), round(n * 0.30), round(n * 0.40), round(n * 0.50)})
    for start in endpoints:
        for end in endpoints:
            if end <= start:
                continue
            selected = keys[start:end]
            name = f"{ckpt}_diag_rank{start + 1:03d}_{end:03d}"
            row = evaluate_policy(name, ckpt, keys, selected, score_rows, direct_rows, reason_rows)
            row["rank_window"] = {"start_rank": start + 1, "end_rank": end}
            row["policy_family"] = "diagnostic_rank_window"
            rows.append(row)
    return rows


def fmt_delta(delta):
    return "{argument_f1:+.4f} / {event_f1:+.4f} / {trigger_f1:+.4f}".format(**delta)


def render_report(payload):
    lines = [
        "# M02 Rank-Window Calibration Dev Sweep",
        "",
        "This report tests whether the m02 route-NLL selector improves when we skip the harmful top-ranked head and select a later rank window.",
        "",
        "## Summary",
        "",
        "| group | policy | reason rate | label F1 | helpful F1 | delta A/E/T | routed A/E/T |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group, row in payload["best"].items():
        lines.append(
            "| {group} | {name} | {rate:.1%} | {lf:.3f} | {hf:.3f} | {delta} | {a:.4f} / {e:.4f} / {t:.4f} |".format(
                group=group,
                name=row["name"],
                rate=row["reason_rate"],
                lf=row["route_vs_label"]["f1"],
                hf=row["route_vs_positive_reason_helpful"]["f1"],
                delta=fmt_delta(row["routed_delta_vs_direct"]),
                a=row["routed"]["argument_f1"],
                e=row["routed"]["event_f1"],
                t=row["routed"]["trigger_f1"],
            )
        )
    lines.extend(
        [
            "",
            "## Fixed Windows",
            "",
            "| policy | reason rate | delta A/E/T | selected mean A/E/T | label F1 | helpful F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    fixed = sorted(
        payload["fixed_window_results"],
        key=lambda row: (
            row["routed_delta_vs_direct"]["event_f1"],
            row["routed_delta_vs_direct"]["argument_f1"],
            row["routed_delta_vs_direct"]["trigger_f1"],
        ),
        reverse=True,
    )[:20]
    for row in fixed:
        lines.append(
            "| {name} | {rate:.1%} | {delta} | {sel} | {lf:.3f} | {hf:.3f} |".format(
                name=row["name"],
                rate=row["reason_rate"],
                delta=fmt_delta(row["routed_delta_vs_direct"]),
                sel=fmt_delta(row["selected_delta_mean"]),
                lf=row["route_vs_label"]["f1"],
                hf=row["route_vs_positive_reason_helpful"]["f1"],
            )
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Rank-window calibration is dev-only and should be treated as a selector diagnostic until a rule is locked.",
            "- The key question is whether a small skipped-head window preserves event gain while reducing trigger harm relative to `checkpoint-150_nll_top30`.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    direct_rows = load_prediction_map(DIRECT_DEV)
    reason_rows = load_prediction_map(REASON_DEV)
    common_keys = sorted(set(direct_rows) & set(reason_rows))

    fixed_rows = []
    diagnostic_rows = []
    for score_path in sorted(SCORE_ROOT.glob("checkpoint-*/dev_seen_scores.jsonl"), key=lambda p: ckpt_num(p.parent.name)):
        ckpt = score_path.parent.name
        score_rows = load_score_rows(score_path)
        keys = sorted_keys_by_delta(score_rows, common_keys)
        fixed_rows.extend(fixed_window_results(ckpt, keys, score_rows, direct_rows, reason_rows))
        diagnostic_rows.extend(exhaustive_window_results(ckpt, keys, score_rows, direct_rows, reason_rows))

    def best_by_event(rows):
        return max(
            rows,
            key=lambda row: (
                row["routed_delta_vs_direct"]["event_f1"],
                row["routed_delta_vs_direct"]["argument_f1"],
                row["routed_delta_vs_direct"]["trigger_f1"],
            ),
        )

    fixed_nonneg_trigger_event = [
        row
        for row in fixed_rows
        if row["routed_delta_vs_direct"]["event_f1"] > 0 and row["routed_delta_vs_direct"]["trigger_f1"] >= 0
    ]
    fixed_all_nonneg = [
        row
        for row in fixed_rows
        if row["routed_delta_vs_direct"]["event_f1"] > 0
        and row["routed_delta_vs_direct"]["argument_f1"] >= 0
        and row["routed_delta_vs_direct"]["trigger_f1"] >= 0
    ]
    diag_all_nonneg = [
        row
        for row in diagnostic_rows
        if row["routed_delta_vs_direct"]["event_f1"] > 0
        and row["routed_delta_vs_direct"]["argument_f1"] >= 0
        and row["routed_delta_vs_direct"]["trigger_f1"] >= 0
    ]
    payload = {
        "branch": BRANCH,
        "score_root": SCORE_ROOT.as_posix(),
        "direct_predictions": DIRECT_DEV.as_posix(),
        "reason_predictions": REASON_DEV.as_posix(),
        "num_fixed_results": len(fixed_rows),
        "num_diagnostic_results": len(diagnostic_rows),
        "fixed_window_results": fixed_rows,
        "diagnostic_window_results": diagnostic_rows,
        "best": {
            "fixed_best_event": best_by_event(fixed_rows),
            "fixed_event_trigger_nonnegative": best_by_event(fixed_nonneg_trigger_event)
            if fixed_nonneg_trigger_event
            else None,
            "fixed_all_nonnegative": best_by_event(fixed_all_nonneg) if fixed_all_nonneg else None,
            "diagnostic_all_nonnegative": best_by_event(diag_all_nonneg) if diag_all_nonneg else None,
        },
    }
    payload["best"] = {key: value for key, value in payload["best"].items() if value is not None}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
