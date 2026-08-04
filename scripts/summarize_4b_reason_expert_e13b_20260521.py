import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.evaluate_4b_transfer_m06_policy_replay_20260521 import (  # noqa: E402
    M02_FORMAL_ROOT,
    M02_WINDOW,
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


NEW_ROOT = REPO / "outputs/stage2_4b_reason_expert/e13b_formal_20260521"
OLD_ROOT = (
    REPO
    / "outputs/stage2_adaptive_runs_user_formal_clean"
    / "richere_split1_qwen3_4b_adaptive_confrare10_heur10_typeonlylite"
)
OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_4b_reason_expert_e13b.json"
OUT_MD = REPO / "reports/2026-05-21_stage2_4b_reason_expert_e13b.md"
SPLITS = ["test_seen", "test_unseen"]


def metric_delta(direct_row, reason_row):
    direct_m = row_metric(direct_row)
    reason_m = row_metric(reason_row)
    return {
        "trigger_f1": reason_m["trigger"]["f1"] - direct_m["trigger"]["f1"],
        "argument_f1": reason_m["argument"]["f1"] - direct_m["argument"]["f1"],
        "event_f1": reason_m["event"]["f1"] - direct_m["event"]["f1"],
        "score": score(reason_row) - score(direct_row),
    }


def load_pair(root, split):
    return (
        load_prediction_map(root / "forced_direct" / split / "predictions.jsonl"),
        load_prediction_map(root / "forced_reason" / split / "predictions.jsonl"),
    )


def load_new_pair(split):
    return (
        load_prediction_map(NEW_ROOT / "forced_direct" / split / "predictions.jsonl"),
        load_prediction_map(NEW_ROOT / "forced_reason" / split / "predictions.jsonl"),
    )


def mean_selected(deltas):
    if not deltas:
        return {key: 0.0 for key in ["argument_f1", "event_f1", "trigger_f1", "score"]}
    return {
        key: sum(row[key] for row in deltas) / len(deltas)
        for key in ["argument_f1", "event_f1", "trigger_f1", "score"]
    }


def evaluate_policy(system, split, policy, direct_rows, reason_rows, selected):
    keys = sorted(set(direct_rows) & set(reason_rows))
    direct_metrics = []
    reason_metrics = []
    routed_metrics = []
    selected_deltas = []
    for key in keys:
        direct = direct_rows[key]
        reason = reason_rows[key]
        chosen = reason if key in selected else direct
        direct_metrics.append(row_metric(direct))
        reason_metrics.append(row_metric(reason))
        routed_metrics.append(row_metric(chosen))
        if key in selected:
            selected_deltas.append(metric_delta(direct, reason))
    direct = summarize_metrics(direct_metrics)
    reason = summarize_metrics(reason_metrics)
    routed = summarize_metrics(routed_metrics)
    return {
        "system": system,
        "split": split,
        "policy": policy,
        "num_examples": len(keys),
        "pred_reason_count": len(selected),
        "pred_reason_rate": len(selected) / len(keys) if keys else 0.0,
        "direct": direct,
        "forced_reason_all": reason,
        "routed": routed,
        "routed_minus_direct": {
            metric: routed[metric] - direct[metric]
            for metric in ["trigger_f1", "argument_f1", "event_f1"]
        },
        "selected_delta_mean": mean_selected(selected_deltas),
    }


def m06_selected(split, keys):
    m02_scores = load_score_rows(M02_FORMAL_ROOT / "checkpoint-50" / split / "scores.jsonl")
    m05_scores = load_score_rows(M05_FORMAL_ROOT / "checkpoint-100" / split / "scores.jsonl")
    common = sorted(set(keys) & set(m02_scores) & set(m05_scores))
    m02, _ = selected_by_window(m02_scores, common, M02_WINDOW)
    m05, _ = selected_by_window(m05_scores, common, M05_WINDOW)
    return m02 | m05


def split_rows(system, split, direct_rows, reason_rows):
    keys = sorted(set(direct_rows) & set(reason_rows))
    return [
        evaluate_policy(system, split, "direct_only", direct_rows, reason_rows, set()),
        evaluate_policy(system, split, "reason_all", direct_rows, reason_rows, set(keys)),
        evaluate_policy(
            system,
            split,
            "m06_transfer",
            direct_rows,
            reason_rows,
            m06_selected(split, keys),
        ),
        evaluate_policy(
            system,
            split,
            "oracle_4b_direct_reason",
            direct_rows,
            reason_rows,
            {key for key in keys if score(reason_rows[key]) > score(direct_rows[key])},
        ),
    ]


def aggregate(rows):
    out = []
    for system in sorted({row["system"] for row in rows}):
        for policy in sorted({row["policy"] for row in rows if row["system"] == system}):
            items = [row for row in rows if row["system"] == system and row["policy"] == policy]
            total = sum(row["num_examples"] for row in items)
            if not total:
                continue
            pred_reason_count = sum(row["pred_reason_count"] for row in items)
            agg = {
                "system": system,
                "split": "test",
                "policy": policy,
                "num_examples": total,
                "pred_reason_count": pred_reason_count,
                "pred_reason_rate": pred_reason_count / total,
            }
            for group in ["direct", "forced_reason_all", "routed"]:
                agg[group] = {}
                for metric in ["trigger_f1", "argument_f1", "event_f1"]:
                    agg[group][metric] = sum(row[group][metric] * row["num_examples"] for row in items) / total
            agg["routed_minus_direct"] = {
                metric: agg["routed"][metric] - agg["direct"][metric]
                for metric in ["trigger_f1", "argument_f1", "event_f1"]
            }
            agg["selected_delta_mean"] = mean_selected([])
            out.append(agg)
    return out


def fmt(value):
    return f"{value:.4f}"


def signed(value):
    return f"{value:+.4f}"


def pct(value):
    return f"{100 * value:.1f}%"


def aet(row, group="routed"):
    m = row[group]
    return f"{fmt(m['argument_f1'])} / {fmt(m['event_f1'])} / {fmt(m['trigger_f1'])}"


def delta(row):
    d = row["routed_minus_direct"]
    return f"{signed(d['argument_f1'])} / {signed(d['event_f1'])} / {signed(d['trigger_f1'])}"


def table(rows):
    lines = [
        "| system | policy | split | reason rate | routed A/E/T | delta vs direct A/E/T |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['system']}` | `{row['policy']}` | `{row['split']}` | "
            f"{pct(row['pred_reason_rate'])} | {aet(row)} | {delta(row)} |"
        )
    return "\n".join(lines)


def render(payload):
    test_rows = [row for row in payload["results"] if row["split"] == "test"]
    lines = [
        "# 4B Reason Expert E13B Formal Replay",
        "",
        "This compares the old 4B typeonlylite reason expert with the direct-warmup retention E13B expert.",
        "",
        "## Test Results",
        "",
        table(sorted(test_rows, key=lambda row: (row["system"], row["policy"]))),
        "",
        "## Reading",
        "",
    ]
    for row in sorted(test_rows, key=lambda item: (item["system"], item["policy"])):
        lines.append(
            f"- `{row['system']}` `{row['policy']}`: reason rate `{pct(row['pred_reason_rate'])}`, "
            f"A/E/T `{delta(row)}`."
        )
    return "\n".join(lines) + "\n"


def main():
    rows = []
    for split in SPLITS:
        rows.extend(split_rows("old_typeonlylite", split, *load_pair(OLD_ROOT, split)))
        rows.extend(split_rows("e13b_directwarm_retention", split, *load_new_pair(split)))
    rows.extend(aggregate(rows))
    payload = {"new_root": NEW_ROOT.as_posix(), "old_root": OLD_ROOT.as_posix(), "results": rows}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
