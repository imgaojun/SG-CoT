import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.calibrate_modular_dualexpert_utility_router_m02_rank_window_dev_20260520 import (  # noqa: E402
    DIRECT_DEV,
    REASON_DEV,
)
from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    DIRECT_ROOT,
    REASON_ROOT,
    load_prediction_map,
    load_score_rows,
    row_metric,
    score,
    sorted_keys_by_delta,
    summarize_metrics,
)


M02_BRANCH = "aet_stable_router_m02_routecls_noauxwarm_lr2e6_save50"
M05_BRANCH = "aet_positive_retention_router_m05_routecls_noauxwarm_lr2e6_save50"
M02_DEV_SCORE = (
    REPO
    / "outputs/stage2_modular_dualexpert/aet_stable_router_m02_20260520/route_likelihood"
    / M02_BRANCH
    / "checkpoint-50/dev_seen_scores.jsonl"
)
M05_DEV_SCORE = (
    REPO
    / "outputs/stage2_modular_dualexpert/aet_positive_retention_router_m05_20260521/route_likelihood"
    / M05_BRANCH
    / "checkpoint-100/dev_seen_scores.jsonl"
)
M02_FORMAL_ROOT = (
    REPO / "outputs/stage2_modular_dualexpert/aet_stable_router_m02_20260520/formal_route_likelihood" / M02_BRANCH
)
M05_FORMAL_ROOT = (
    REPO
    / "outputs/stage2_modular_dualexpert/aet_positive_retention_router_m05_20260521/formal_route_likelihood"
    / M05_BRANCH
)
OUT_DEV_JSON = REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_m06_combo_selectors_dev.json"
OUT_DEV_MD = REPO / "reports/2026-05-21_stage2_modular_dualexpert_aet_m06_combo_selectors_dev.md"
OUT_FORMAL_JSON = REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_m06_combo_selectors_formal.json"
OUT_FORMAL_MD = REPO / "reports/2026-05-21_stage2_modular_dualexpert_aet_m06_combo_selectors_formal.md"

M02_WINDOW = (0.425, 0.500)
M05_WINDOW = (0.050, 0.100)
FORMAL_SPLITS = ["test_seen", "test_unseen"]


def selected_by_window(score_rows, keys, window):
    ranked = sorted_keys_by_delta(score_rows, keys)
    start = round(len(ranked) * window[0])
    end = round(len(ranked) * window[1])
    return set(ranked[start:end]), {
        "start_pct": window[0],
        "end_pct": window[1],
        "start_rank": start + 1,
        "end_rank": end,
    }


def metric_delta(direct_row, reason_row):
    direct_m = row_metric(direct_row)
    reason_m = row_metric(reason_row)
    return {
        "trigger_f1": reason_m["trigger"]["f1"] - direct_m["trigger"]["f1"],
        "argument_f1": reason_m["argument"]["f1"] - direct_m["argument"]["f1"],
        "event_f1": reason_m["event"]["f1"] - direct_m["event"]["f1"],
        "score": score(reason_row) - score(direct_row),
    }


def mean_dict(rows):
    keys = ["trigger_f1", "argument_f1", "event_f1", "score"]
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def evaluate_selected(name, split, keys, selected, direct_rows, reason_rows, m02_selected, m05_selected):
    routed_metrics = []
    direct_metrics = []
    reason_metrics = []
    selected_deltas = []
    selected_examples = []
    for key in keys:
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        chosen = reason_row if key in selected else direct_row
        routed_metrics.append(row_metric(chosen))
        direct_metrics.append(row_metric(direct_row))
        reason_metrics.append(row_metric(reason_row))
        if key in selected:
            delta = metric_delta(direct_row, reason_row)
            selected_deltas.append(delta)
            selected_examples.append(
                {
                    "wnd_id": key,
                    "in_m02": key in m02_selected,
                    "in_m05": key in m05_selected,
                    **delta,
                }
            )
    direct = summarize_metrics(direct_metrics)
    reason = summarize_metrics(reason_metrics)
    routed = summarize_metrics(routed_metrics)
    return {
        "policy": name,
        "split": split,
        "num_examples": len(keys),
        "pred_reason_count": len(selected),
        "pred_reason_rate": len(selected) / len(keys) if keys else 0.0,
        "overlap_with_m02": len(selected & m02_selected),
        "overlap_with_m05": len(selected & m05_selected),
        "direct": direct,
        "forced_reason_all": reason,
        "routed": routed,
        "routed_minus_direct": {
            metric: routed[metric] - direct[metric]
            for metric in ["trigger_f1", "argument_f1", "event_f1"]
        },
        "routed_minus_reason_all": {
            metric: routed[metric] - reason[metric]
            for metric in ["trigger_f1", "argument_f1", "event_f1"]
        },
        "selected_delta_mean": mean_dict(selected_deltas),
        "selected_examples": sorted(selected_examples, key=lambda row: row["score"], reverse=True)[:20],
    }


def evaluate_split(split, direct_path, reason_path, m02_score_path, m05_score_path):
    for path in [direct_path, reason_path, m02_score_path, m05_score_path]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    direct_rows = load_prediction_map(Path(direct_path))
    reason_rows = load_prediction_map(Path(reason_path))
    m02_scores = load_score_rows(Path(m02_score_path))
    m05_scores = load_score_rows(Path(m05_score_path))
    keys = sorted(set(direct_rows) & set(reason_rows) & set(m02_scores) & set(m05_scores))
    m02_selected, m02_rank_window = selected_by_window(m02_scores, keys, M02_WINDOW)
    m05_selected, m05_rank_window = selected_by_window(m05_scores, keys, M05_WINDOW)
    policy_sets = {
        "m02_positive_retention": m02_selected,
        "m05_lowbudget": m05_selected,
        "union_m02_m05": m02_selected | m05_selected,
        "intersection_m02_m05": m02_selected & m05_selected,
        "m02_only_minus_m05": m02_selected - m05_selected,
        "m05_only_minus_m02": m05_selected - m02_selected,
    }
    rows = [
        evaluate_selected(name, split, keys, selected, direct_rows, reason_rows, m02_selected, m05_selected)
        for name, selected in policy_sets.items()
    ]
    return {
        "split": split,
        "num_examples": len(keys),
        "component_selectors": {
            "m02_positive_retention": {
                "branch": M02_BRANCH,
                "checkpoint": "checkpoint-50",
                "rank_window": m02_rank_window,
                "reason_count": len(m02_selected),
                "reason_rate": len(m02_selected) / len(keys) if keys else 0.0,
            },
            "m05_lowbudget": {
                "branch": M05_BRANCH,
                "checkpoint": "checkpoint-100",
                "rank_window": m05_rank_window,
                "reason_count": len(m05_selected),
                "reason_rate": len(m05_selected) / len(keys) if keys else 0.0,
            },
        },
        "overlap": {
            "count": len(m02_selected & m05_selected),
            "m02_count": len(m02_selected),
            "m05_count": len(m05_selected),
            "jaccard": (
                len(m02_selected & m05_selected) / len(m02_selected | m05_selected)
                if (m02_selected | m05_selected)
                else 0.0
            ),
        },
        "results": rows,
    }


def aggregate_test(split_payloads):
    by_policy = {}
    for payload in split_payloads:
        for row in payload["results"]:
            by_policy.setdefault(row["policy"], []).append(row)
    out = []
    for policy, rows in by_policy.items():
        total = sum(row["num_examples"] for row in rows)
        pred_reason_count = sum(row["pred_reason_count"] for row in rows)
        agg = {
            "policy": policy,
            "split": "test",
            "num_examples": total,
            "pred_reason_count": pred_reason_count,
            "pred_reason_rate": pred_reason_count / total if total else 0.0,
            "overlap_with_m02": sum(row["overlap_with_m02"] for row in rows),
            "overlap_with_m05": sum(row["overlap_with_m05"] for row in rows),
        }
        for group in ["direct", "forced_reason_all", "routed"]:
            agg[group] = {}
            for metric in ["trigger_f1", "argument_f1", "event_f1"]:
                agg[group][metric] = (
                    sum(row[group][metric] * row["num_examples"] for row in rows) / total
                    if total
                    else 0.0
                )
        agg["routed_minus_direct"] = {
            metric: agg["routed"][metric] - agg["direct"][metric]
            for metric in ["trigger_f1", "argument_f1", "event_f1"]
        }
        agg["routed_minus_reason_all"] = {
            metric: agg["routed"][metric] - agg["forced_reason_all"][metric]
            for metric in ["trigger_f1", "argument_f1", "event_f1"]
        }
        denom = pred_reason_count
        agg["selected_delta_mean"] = {
            metric: (
                sum(row["selected_delta_mean"][metric] * row["pred_reason_count"] for row in rows) / denom
                if denom
                else 0.0
            )
            for metric in ["trigger_f1", "argument_f1", "event_f1", "score"]
        }
        out.append(agg)
    return out


def signed(value):
    return f"{value:+.4f}"


def fmt_delta(delta):
    return "{argument_f1:+.4f} / {event_f1:+.4f} / {trigger_f1:+.4f}".format(**delta)


def render_table(rows):
    lines = [
        "| policy | split | reason rate | overlap m02/m05 | delta A/E/T | selected mean A/E/T | routed A/E/T |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        routed = row["routed"]
        sel = row["selected_delta_mean"]
        lines.append(
            "| {policy} | {split} | {rate:.1%} | {om02}/{om05} | {delta} | {sel_a:+.4f} / {sel_e:+.4f} / {sel_t:+.4f} | {a:.4f} / {e:.4f} / {t:.4f} |".format(
                policy=row["policy"],
                split=row["split"],
                rate=row["pred_reason_rate"],
                om02=row["overlap_with_m02"],
                om05=row["overlap_with_m05"],
                delta=fmt_delta(row["routed_minus_direct"]),
                sel_a=sel["argument_f1"],
                sel_e=sel["event_f1"],
                sel_t=sel["trigger_f1"],
                a=routed["argument_f1"],
                e=routed["event_f1"],
                t=routed["trigger_f1"],
            )
        )
    return "\n".join(lines)


def render_dev_report(payload):
    rows = sorted(payload["results"], key=lambda row: (row["pred_reason_rate"], row["policy"]))
    lines = [
        "# A/E/T M06 Combo Selectors Dev Replay",
        "",
        "This report composes two dev-locked selectors without training or new scoring.",
        "",
        "## Components",
        "",
        f"- m02 positive-retention: checkpoint-50 rank {M02_WINDOW[0]:.1%}-{M02_WINDOW[1]:.1%}.",
        f"- m05 low-budget: checkpoint-100 rank {M05_WINDOW[0]:.1%}-{M05_WINDOW[1]:.1%}.",
        f"- dev overlap: {payload['overlap']['count']} / union {payload['overlap']['jaccard']:.3f} Jaccard.",
        "",
        "## Results",
        "",
        render_table(rows),
        "",
    ]
    return "\n".join(lines)


def render_formal_report(payload):
    rows = sorted(payload["results"], key=lambda row: (row["policy"], row["split"]))
    test_rows = [row for row in payload["results"] if row["split"] == "test"]
    lines = [
        "# A/E/T M06 Combo Selectors Formal Replay",
        "",
        "This report applies dev-defined selector combinations to formal route-NLL scores. Formal labels are not used for selection.",
        "",
        "## Test Summary",
        "",
        render_table(sorted(test_rows, key=lambda row: (row["pred_reason_rate"], row["policy"]))),
        "",
        "## Split Results",
        "",
        render_table(rows),
        "",
        "## Reading",
        "",
    ]
    for row in sorted(test_rows, key=lambda item: item["policy"]):
        lines.append(
            f"- `{row['policy']}`: reason rate `{row['pred_reason_rate']:.1%}`, "
            f"A/E/T `{fmt_delta(row['routed_minus_direct'])}`."
        )
    return "\n".join(lines) + "\n"


def main():
    dev_payload = evaluate_split("dev_seen", DIRECT_DEV, REASON_DEV, M02_DEV_SCORE, M05_DEV_SCORE)
    OUT_DEV_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_DEV_JSON.write_text(json.dumps(dev_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_DEV_MD.write_text(render_dev_report(dev_payload), encoding="utf-8")

    split_payloads = []
    for split in FORMAL_SPLITS:
        split_payloads.append(
            evaluate_split(
                split,
                DIRECT_ROOT / split / "predictions.jsonl",
                REASON_ROOT / split / "predictions.jsonl",
                M02_FORMAL_ROOT / "checkpoint-50" / split / "scores.jsonl",
                M05_FORMAL_ROOT / "checkpoint-100" / split / "scores.jsonl",
            )
        )
    formal_rows = []
    for payload in split_payloads:
        formal_rows.extend(payload["results"])
    formal_rows.extend(aggregate_test(split_payloads))
    formal_payload = {
        "component_selectors": {
            "m02_positive_retention": {
                "branch": M02_BRANCH,
                "checkpoint": "checkpoint-50",
                "rank_window": {"start_pct": M02_WINDOW[0], "end_pct": M02_WINDOW[1]},
            },
            "m05_lowbudget": {
                "branch": M05_BRANCH,
                "checkpoint": "checkpoint-100",
                "rank_window": {"start_pct": M05_WINDOW[0], "end_pct": M05_WINDOW[1]},
            },
        },
        "split_payloads": split_payloads,
        "results": formal_rows,
    }
    OUT_FORMAL_JSON.write_text(json.dumps(formal_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_FORMAL_MD.write_text(render_formal_report(formal_payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "dev_json": OUT_DEV_JSON.as_posix(),
                "dev_md": OUT_DEV_MD.as_posix(),
                "formal_json": OUT_FORMAL_JSON.as_posix(),
                "formal_md": OUT_FORMAL_MD.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
