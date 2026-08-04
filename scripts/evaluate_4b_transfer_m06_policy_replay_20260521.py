import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.evaluate_modular_dualexpert_aet_m06_combo_selectors_20260521 import (  # noqa: E402
    M02_BRANCH,
    M02_FORMAL_ROOT,
    M02_WINDOW,
    M05_BRANCH,
    M05_FORMAL_ROOT,
    M05_WINDOW,
    selected_by_window,
)
from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    load_prediction_map,
    load_score_rows,
    row_metric,
    score,
    summarize_metrics,
)


FORMAL_SPLITS = ["test_seen", "test_unseen"]
SYSTEMS = {
    "qwen3_4b_confrare10_typerolelite": REPO
    / "outputs/stage2_adaptive_runs_user_formal_clean"
    / "richere_split1_qwen3_4b_adaptive_confrare10_heur10_typerolelite",
    "qwen3_4b_confrare10_typeonlylite": REPO
    / "outputs/stage2_adaptive_runs_user_formal_clean"
    / "richere_split1_qwen3_4b_adaptive_confrare10_heur10_typeonlylite",
}
OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_4b_transfer_m06_policy_replay.json"
OUT_MD = REPO / "reports/2026-05-21_stage2_4b_transfer_m06_policy_replay.md"


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


def evaluate_selected(system, policy, split, keys, selected, direct_rows, reason_rows, m02_selected, m05_selected):
    routed_metrics = []
    direct_metrics = []
    reason_metrics = []
    selected_deltas = []
    for key in keys:
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        chosen = reason_row if key in selected else direct_row
        routed_metrics.append(row_metric(chosen))
        direct_metrics.append(row_metric(direct_row))
        reason_metrics.append(row_metric(reason_row))
        if key in selected:
            selected_deltas.append(metric_delta(direct_row, reason_row))
    direct = summarize_metrics(direct_metrics)
    reason = summarize_metrics(reason_metrics)
    routed = summarize_metrics(routed_metrics)
    return {
        "system": system,
        "policy": policy,
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
    }


def evaluate_split(system, root, split):
    paths = {
        "direct": root / "forced_direct" / split / "predictions.jsonl",
        "reason": root / "forced_reason" / split / "predictions.jsonl",
        "m02": M02_FORMAL_ROOT / "checkpoint-50" / split / "scores.jsonl",
        "m05": M05_FORMAL_ROOT / "checkpoint-100" / split / "scores.jsonl",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    direct_rows = load_prediction_map(paths["direct"])
    reason_rows = load_prediction_map(paths["reason"])
    m02_scores = load_score_rows(paths["m02"])
    m05_scores = load_score_rows(paths["m05"])
    keys = sorted(set(direct_rows) & set(reason_rows) & set(m02_scores) & set(m05_scores))
    m02_selected, m02_rank_window = selected_by_window(m02_scores, keys, M02_WINDOW)
    m05_selected, m05_rank_window = selected_by_window(m05_scores, keys, M05_WINDOW)
    oracle_selected = {
        key for key in keys if score(reason_rows[key]) > score(direct_rows[key])
    }
    policy_sets = {
        "direct_only": set(),
        "reason_all": set(keys),
        "m02_transfer": m02_selected,
        "m05_transfer": m05_selected,
        "m06_union_transfer": m02_selected | m05_selected,
        "oracle_4b_direct_reason": oracle_selected,
    }
    return {
        "system": system,
        "split": split,
        "num_examples": len(keys),
        "input_counts": {
            "direct": len(direct_rows),
            "reason": len(reason_rows),
            "m02_scores": len(m02_scores),
            "m05_scores": len(m05_scores),
            "common": len(keys),
        },
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
            "union_count": len(m02_selected | m05_selected),
            "jaccard": (
                len(m02_selected & m05_selected) / len(m02_selected | m05_selected)
                if (m02_selected | m05_selected)
                else 0.0
            ),
        },
        "results": [
            evaluate_selected(system, policy, split, keys, selected, direct_rows, reason_rows, m02_selected, m05_selected)
            for policy, selected in policy_sets.items()
        ],
    }


def aggregate_test(rows):
    out = []
    for system in SYSTEMS:
        policies = sorted({row["policy"] for row in rows if row["system"] == system})
        for policy in policies:
            items = [row for row in rows if row["system"] == system and row["policy"] == policy]
            total = sum(row["num_examples"] for row in items)
            if not total:
                continue
            pred_reason_count = sum(row["pred_reason_count"] for row in items)
            agg = {
                "system": system,
                "policy": policy,
                "split": "test",
                "num_examples": total,
                "pred_reason_count": pred_reason_count,
                "pred_reason_rate": pred_reason_count / total,
                "overlap_with_m02": sum(row["overlap_with_m02"] for row in items),
                "overlap_with_m05": sum(row["overlap_with_m05"] for row in items),
            }
            for group in ["direct", "forced_reason_all", "routed"]:
                agg[group] = {}
                for metric in ["trigger_f1", "argument_f1", "event_f1"]:
                    agg[group][metric] = (
                        sum(row[group][metric] * row["num_examples"] for row in items) / total
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
                    sum(row["selected_delta_mean"][metric] * row["pred_reason_count"] for row in items)
                    / denom
                    if denom
                    else 0.0
                )
                for metric in ["trigger_f1", "argument_f1", "event_f1", "score"]
            }
            out.append(agg)
    return out


def fmt(value):
    return f"{value:.4f}"


def signed(value):
    return f"{value:+.4f}"


def pct(value):
    return f"{100 * value:.1f}%"


def aet(row, group="routed"):
    metrics = row[group]
    return f"{fmt(metrics['argument_f1'])} / {fmt(metrics['event_f1'])} / {fmt(metrics['trigger_f1'])}"


def delta_aet(row):
    delta = row["routed_minus_direct"]
    return (
        f"{signed(delta['argument_f1'])} / "
        f"{signed(delta['event_f1'])} / "
        f"{signed(delta['trigger_f1'])}"
    )


def render_table(rows):
    lines = [
        "| system | policy | split | reason rate | routed A/E/T | delta vs direct A/E/T | selected mean A/E/T |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        selected = row["selected_delta_mean"]
        lines.append(
            f"| `{row['system']}` | `{row['policy']}` | `{row['split']}` | {pct(row['pred_reason_rate'])} | "
            f"{aet(row)} | {delta_aet(row)} | "
            f"{signed(selected['argument_f1'])} / {signed(selected['event_f1'])} / {signed(selected['trigger_f1'])} |"
        )
    return "\n".join(lines)


def render_headroom(rows):
    lines = [
        "| system | split | direct A/E/T | reason-all A/E/T | reason-all minus direct A/E/T | oracle A/E/T | oracle delta A/E/T | oracle rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        for split in ["test", "test_seen", "test_unseen"]:
            reason = next(row for row in rows if row["system"] == system and row["split"] == split and row["policy"] == "reason_all")
            oracle = next(
                row
                for row in rows
                if row["system"] == system and row["split"] == split and row["policy"] == "oracle_4b_direct_reason"
            )
            lines.append(
                f"| `{system}` | `{split}` | {aet(reason, 'direct')} | {aet(reason, 'forced_reason_all')} | "
                f"{delta_aet(reason)} | {aet(oracle)} | {delta_aet(oracle)} | {pct(oracle['pred_reason_rate'])} |"
            )
    return "\n".join(lines)


def render_report(payload):
    rows = payload["results"]
    transfer_rows = [
        row
        for row in rows
        if row["policy"] in {"m02_transfer", "m05_transfer", "m06_union_transfer"}
    ]
    test_transfer = [row for row in transfer_rows if row["split"] == "test"]
    lines = [
        "# 4B M06 Policy Transfer Replay",
        "",
        "This replay keeps the 1.7B M02/M05 selector scores fixed and applies their selected windows to existing Qwen3-4B forced-direct/forced-reason outputs. Selection does not use 4B formal labels except for the oracle diagnostic.",
        "",
        "## Selector Source",
        "",
        f"- M02: `{M02_BRANCH}` checkpoint-50 rank window `{M02_WINDOW[0]:.1%}-{M02_WINDOW[1]:.1%}`.",
        f"- M05: `{M05_BRANCH}` checkpoint-100 rank window `{M05_WINDOW[0]:.1%}-{M05_WINDOW[1]:.1%}`.",
        "- M06 transfer is the union of M02 and M05 selected samples.",
        "",
        "## Headroom",
        "",
        render_headroom(rows),
        "",
        "## Transfer Results",
        "",
        render_table(sorted(test_transfer, key=lambda row: (row["system"], row["policy"]))),
        "",
        "## Split Details",
        "",
        render_table(sorted(transfer_rows, key=lambda row: (row["system"], row["policy"], row["split"]))),
        "",
        "## Reading",
        "",
    ]
    for row in sorted(test_transfer, key=lambda item: (item["system"], item["policy"])):
        lines.append(
            f"- `{row['system']}` `{row['policy']}`: reason rate `{pct(row['pred_reason_rate'])}`, "
            f"test A/E/T delta `{delta_aet(row)}`."
        )
    lines.extend(
        [
            "- The oracle row is an upper-bound diagnostic: it uses 4B outcome labels to pick direct vs reason per sample and must not be reported as a deployable selector.",
            "- If oracle is strong but M06 transfer is weak, the bottleneck is selector/model transfer. If oracle is also weak, the 4B reason expert itself is not providing enough per-sample upside.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    split_payloads = []
    rows = []
    for system, root in SYSTEMS.items():
        for split in FORMAL_SPLITS:
            payload = evaluate_split(system, root, split)
            split_payloads.append(payload)
            rows.extend(payload["results"])
    rows.extend(aggregate_test(rows))
    payload = {
        "systems": {name: root.as_posix() for name, root in SYSTEMS.items()},
        "selector_source": {
            "m02": {
                "branch": M02_BRANCH,
                "checkpoint": "checkpoint-50",
                "window": M02_WINDOW,
            },
            "m05": {
                "branch": M05_BRANCH,
                "checkpoint": "checkpoint-100",
                "window": M05_WINDOW,
            },
        },
        "split_payloads": split_payloads,
        "results": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
