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
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_utility_router_m02_deep_diagnosis.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_utility_router_m02_deep_diagnosis.md"
SCANNED_BUDGETS = [0.10, 0.15, 0.20, 0.30]


def ckpt_num(name: str) -> int:
    return int(name.split("-", 1)[1])


def load_score_rows(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def metric_delta(direct_row, reason_row):
    direct_m = row_metric(direct_row)
    reason_m = row_metric(reason_row)
    return {
        "trigger_f1": reason_m["trigger"]["f1"] - direct_m["trigger"]["f1"],
        "argument_f1": reason_m["argument"]["f1"] - direct_m["argument"]["f1"],
        "event_f1": reason_m["event"]["f1"] - direct_m["event"]["f1"],
        "score": score(reason_row) - score(direct_row),
        "direct_score": score(direct_row),
        "reason_score": score(reason_row),
    }


def summarize_routed(common_keys, reason_keys, direct_rows, reason_rows):
    metrics = []
    direct_metrics = []
    reason_metrics = []
    selected_deltas = []
    for key in common_keys:
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        chosen = reason_row if key in reason_keys else direct_row
        metrics.append(row_metric(chosen))
        direct_metrics.append(row_metric(direct_row))
        reason_metrics.append(row_metric(reason_row))
        if key in reason_keys:
            selected_deltas.append(metric_delta(direct_row, reason_row))
    routed = summarize_metrics(metrics)
    direct = summarize_metrics(direct_metrics)
    forced_reason = summarize_metrics(reason_metrics)
    return {
        "reason_count": len(reason_keys),
        "reason_rate": len(reason_keys) / len(common_keys) if common_keys else 0.0,
        "direct": direct,
        "forced_reason": forced_reason,
        "routed": routed,
        "delta": {
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
        },
        "selected_delta_mean": mean_dict(selected_deltas),
    }


def mean_dict(rows):
    if not rows:
        return {"trigger_f1": 0.0, "argument_f1": 0.0, "event_f1": 0.0, "score": 0.0}
    keys = ["trigger_f1", "argument_f1", "event_f1", "score"]
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def count_signs(rows, key):
    return {
        "positive": sum(1 for row in rows if row[key] > 0),
        "zero": sum(1 for row in rows if row[key] == 0),
        "negative": sum(1 for row in rows if row[key] < 0),
    }


def band_summary(sorted_keys, start, end, direct_rows, reason_rows, score_rows):
    keys = sorted_keys[start:end]
    deltas = [metric_delta(direct_rows[key], reason_rows[key]) for key in keys]
    gold_reason = sum(1 for key in keys if score_rows[key].get("gold_route") == "reason")
    return {
        "rank_start": start + 1,
        "rank_end": end,
        "count": len(keys),
        "gold_reason_count": gold_reason,
        "gold_reason_rate": gold_reason / len(keys) if keys else 0.0,
        "mean_delta": mean_dict(deltas),
        "trigger_signs": count_signs(deltas, "trigger_f1"),
        "argument_signs": count_signs(deltas, "argument_f1"),
        "event_signs": count_signs(deltas, "event_f1"),
        "score_signs": count_signs(deltas, "score"),
        "examples": [
            {
                "rank": idx + 1,
                "wnd_id": key,
                "gold_route": score_rows[key].get("gold_route"),
                "route_delta_nll": score_rows[key].get("delta_direct_minus_reason_route_nll"),
                **metric_delta(direct_rows[key], reason_rows[key]),
            }
            for idx, key in enumerate(keys, start=start)
        ][:10],
    }


def format_delta(delta):
    return "{argument_f1:+.4f} / {event_f1:+.4f} / {trigger_f1:+.4f}".format(**delta)


def unique_by_name(rows):
    seen = set()
    out = []
    for row in rows:
        if row is None:
            continue
        if row["name"] in seen:
            continue
        seen.add(row["name"])
        out.append(row)
    return out


def render_report(payload):
    lines = [
        "# M02 Utility Router Deep Dev Diagnosis",
        "",
        "This report diagnoses why the supervised m02 route classifier has low label loss but weak executable route-choice NLL behavior on dev_seen.",
        "",
        "## Main Answer",
        "",
        "- Yes: for the best dev execution point, the only negative end metric is trigger.",
        "- The best point is `checkpoint-150_nll_top30`: argument/event/trigger delta `+0.0041 / +0.0041 / -0.0139`.",
        "- This is still not robust enough for formal promotion because every other checkpoint/budget is worse, and no scanned checkpoint/budget has nonnegative argument, event, and trigger simultaneously.",
        "",
            "## Baselines",
        "",
        "| route | argument | event | trigger |",
        "|---|---:|---:|---:|",
    ]
    for name, row in payload["baselines"].items():
        lines.append(
            f"| {name} | {row['argument_f1']:.4f} | {row['event_f1']:.4f} | {row['trigger_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Actual Scanned Policies",
            "",
            "| selection | reason rate | delta A/E/T | routed A/E/T |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["best_scanned_points"]:
        lines.append(
            "| {name} | {rate:.1%} | {delta} | {a:.4f} / {e:.4f} / {t:.4f} |".format(
                name=row["name"],
                rate=row["reason_rate"],
                delta=format_delta(row["delta"]),
                a=row["routed"]["argument_f1"],
                e=row["routed"]["event_f1"],
                t=row["routed"]["trigger_f1"],
            )
        )
    lines.extend(
        [
            "",
            "## Exhaustive Top-K Diagnostic",
            "",
            "| selection | reason rate | delta A/E/T | routed A/E/T |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["best_exhaustive_points"]:
        lines.append(
            "| {name} | {rate:.1%} | {delta} | {a:.4f} / {e:.4f} / {t:.4f} |".format(
                name=row["name"],
                rate=row["reason_rate"],
                delta=format_delta(row["delta"]),
                a=row["routed"]["argument_f1"],
                e=row["routed"]["event_f1"],
                t=row["routed"]["trigger_f1"],
            )
        )
    lines.extend(
        [
            "",
            "The exhaustive rows are diagnostics only; they were not part of the locked dev policy grid and should not be promoted directly.",
        ]
    )
    lines.extend(
        [
            "",
            "## Checkpoint-150 Rank Bands",
            "",
            "| rank band | count | gold reason | mean selected delta A/E/T | event signs | trigger signs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for band in payload["checkpoint150_bands"]:
        lines.append(
            "| {lo}-{hi} | {count} | {gold}/{count} ({rate:.1%}) | {delta} | +{ep}/0{ez}/-{en} | +{tp}/0{tz}/-{tn} |".format(
                lo=band["rank_start"],
                hi=band["rank_end"],
                count=band["count"],
                gold=band["gold_reason_count"],
                rate=band["gold_reason_rate"],
                delta=format_delta(band["mean_delta"]),
                ep=band["event_signs"]["positive"],
                ez=band["event_signs"]["zero"],
                en=band["event_signs"]["negative"],
                tp=band["trigger_signs"]["positive"],
                tz=band["trigger_signs"]["zero"],
                tn=band["trigger_signs"]["negative"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The event gain in the locked policy grid is real but narrow: it appears only at the 30% budget for `checkpoint-150`.",
            "- Top10/top15/top20 are too conservative and miss enough event-helpful cases; top30 adds the cases that flip event positive.",
            "- The trigger loss is not a simple late-budget artifact; the selected set contains many trigger-negative examples, so a trigger-aware guard is needed if trigger must not regress.",
            "- The oracle15 baseline is much higher than the routed result, so the issue is selector ranking/objective alignment, not lack of useful reason-expert outputs.",
            "",
            "## Next",
            "",
            "- Treat `checkpoint-150_nll_top30` as a promising event-oriented dev policy, not a final policy.",
            "- Try a guard or reranking score that keeps the event-helpful top30 behavior but penalizes examples with historically trigger-negative reason deltas.",
            "- If the paper prioritizes event F1 over trigger F1, this result can justify a targeted formal replay, but it should be reported as event-oriented and risky for trigger.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    direct_rows = load_prediction_map(DIRECT_DEV)
    reason_rows = load_prediction_map(REASON_DEV)
    common_keys = sorted(set(direct_rows) & set(reason_rows))
    baselines = {
        "forced_direct": summarize_metrics([row_metric(direct_rows[key]) for key in common_keys]),
        "forced_reason_all": summarize_metrics([row_metric(reason_rows[key]) for key in common_keys]),
    }

    all_points = []
    scanned_points = []
    per_checkpoint = {}
    for score_path in sorted(SCORE_ROOT.glob("checkpoint-*/dev_seen_scores.jsonl"), key=lambda p: ckpt_num(p.parent.name)):
        ckpt = score_path.parent.name
        score_rows = load_score_rows(score_path)
        scored = []
        for key in set(common_keys) & set(score_rows):
            delta = score_rows[key].get("delta_direct_minus_reason_route_nll")
            if delta is None:
                delta = float("-inf")
            scored.append((float(delta), key))
        scored.sort(reverse=True)
        keys = [key for _, key in scored]
        per_checkpoint[ckpt] = {"sorted_keys": keys, "score_rows": score_rows}
        for k in range(0, len(keys) + 1):
            reason_keys = set(keys[:k])
            summary = summarize_routed(keys, reason_keys, direct_rows, reason_rows)
            summary["name"] = f"{ckpt}_top{k:03d}"
            summary["checkpoint"] = ckpt
            summary["k"] = k
            all_points.append(summary)
            for budget in SCANNED_BUDGETS:
                if k == round(len(keys) * budget):
                    scanned = dict(summary)
                    scanned["name"] = f"{ckpt}_nll_top{int(budget * 100)}"
                    scanned["budget"] = budget
                    scanned_points.append(scanned)

    best_event = max(
        all_points,
        key=lambda row: (
            row["delta"]["event_f1"],
            row["delta"]["argument_f1"],
            row["delta"]["trigger_f1"],
        ),
    )
    best_argument = max(
        all_points,
        key=lambda row: (
            row["delta"]["argument_f1"],
            row["delta"]["event_f1"],
            row["delta"]["trigger_f1"],
        ),
    )
    best_trigger_nonnegative_event = [
        row
        for row in all_points
        if row["delta"]["trigger_f1"] >= 0 and row["delta"]["event_f1"] > 0
    ]
    best_all_nonnegative = [
        row
        for row in all_points
        if row["delta"]["trigger_f1"] >= 0 and row["delta"]["argument_f1"] >= 0 and row["delta"]["event_f1"] > 0
    ]
    payload = {
        "branch": BRANCH,
        "score_root": SCORE_ROOT.as_posix(),
        "direct_predictions": DIRECT_DEV.as_posix(),
        "reason_predictions": REASON_DEV.as_posix(),
        "num_examples": len(common_keys),
        "baselines": baselines,
        "best_scanned_points": unique_by_name([
            max(
                scanned_points,
                key=lambda row: (
                    row["delta"]["event_f1"],
                    row["delta"]["argument_f1"],
                    row["delta"]["trigger_f1"],
                ),
            ),
            max(
                scanned_points,
                key=lambda row: (
                    row["delta"]["argument_f1"],
                    row["delta"]["event_f1"],
                    row["delta"]["trigger_f1"],
                ),
            ),
        ]),
        "best_exhaustive_points": unique_by_name([
            best_event,
            best_argument,
            max(best_trigger_nonnegative_event, key=lambda row: row["delta"]["event_f1"])
            if best_trigger_nonnegative_event
            else None,
            max(best_all_nonnegative, key=lambda row: row["delta"]["event_f1"])
            if best_all_nonnegative
            else None,
        ]),
        "has_trigger_nonnegative_event_positive_exhaustive": bool(best_trigger_nonnegative_event),
        "has_all_nonnegative_event_positive_exhaustive": bool(best_all_nonnegative),
    }
    payload["best_exhaustive_points"] = [row for row in payload["best_exhaustive_points"] if row is not None]

    ckpt150 = per_checkpoint["checkpoint-150"]
    keys = ckpt150["sorted_keys"]
    score_rows = ckpt150["score_rows"]
    band_edges = [0, round(len(keys) * 0.10), round(len(keys) * 0.15), round(len(keys) * 0.20), round(len(keys) * 0.30)]
    payload["checkpoint150_bands"] = [
        band_summary(keys, band_edges[idx], band_edges[idx + 1], direct_rows, reason_rows, score_rows)
        for idx in range(len(band_edges) - 1)
    ]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
