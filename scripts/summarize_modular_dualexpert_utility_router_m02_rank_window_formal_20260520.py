import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from src.stage2_analysis.analyze_adaptive_hardness_boundary import prediction_key, score  # noqa: E402
from src.stage2_analysis.analyze_adaptive_outcome_router_execution import row_metric, summarize_metrics  # noqa: E402
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


BRANCH = "modular_d1930_r2058_utility_m02_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/utility_router_m02_20260520/formal_route_likelihood" / BRANCH
DIRECT_ROOT = REPO / (
    "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_balrouteaux_20260516/"
    "richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux/"
    "checkpoint-1930/forced_direct"
)
REASON_ROOT = REPO / (
    "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/"
    "richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux/"
    "checkpoint-2058/forced_reason"
)
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_utility_router_m02_rank_window_formal.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_utility_router_m02_rank_window_formal.md"

POLICIES = [
    {
        "name": "aet_event_checkpoint50_rank225_400",
        "checkpoint": "checkpoint-50",
        "start_pct": 0.225,
        "end_pct": 0.400,
        "source": "dev_locked_aet_event_candidate",
    },
    {
        "name": "robust_checkpoint100_rank20_40",
        "checkpoint": "checkpoint-100",
        "start_pct": 0.20,
        "end_pct": 0.40,
        "source": "dev_locked_best_all_nonnegative",
    },
    {
        "name": "event_checkpoint150_rank10_30",
        "checkpoint": "checkpoint-150",
        "start_pct": 0.10,
        "end_pct": 0.30,
        "source": "dev_locked_best_event",
    },
    {
        "name": "aet_balanced_checkpoint200_rank175_325",
        "checkpoint": "checkpoint-200",
        "start_pct": 0.175,
        "end_pct": 0.325,
        "source": "dev_locked_aet_balanced_candidate",
    },
]
SPLITS = ["test_seen", "test_unseen"]


def load_prediction_map(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def load_score_rows(path: Path):
    return {prediction_key(row): row for row in load_jsonl(path)}


def sorted_keys_by_delta(score_rows, common_keys):
    scored = []
    for key in set(score_rows) & set(common_keys):
        delta = score_rows[key].get("delta_direct_minus_reason_route_nll")
        if delta is None:
            delta = float("-inf")
        scored.append((float(delta), key))
    scored.sort(reverse=True)
    return [key for _, key in scored]


def evaluate(policy, split):
    score_path = SCORE_ROOT / policy["checkpoint"] / split / "scores.jsonl"
    direct_path = DIRECT_ROOT / split / "predictions.jsonl"
    reason_path = REASON_ROOT / split / "predictions.jsonl"
    for path in [score_path, direct_path, reason_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    score_rows = load_score_rows(score_path)
    direct_rows = load_prediction_map(direct_path)
    reason_rows = load_prediction_map(reason_path)
    keys = sorted_keys_by_delta(score_rows, set(direct_rows) & set(reason_rows))
    start = round(len(keys) * policy["start_pct"])
    end = round(len(keys) * policy["end_pct"])
    reason_keys = set(keys[start:end])

    routed_metrics = []
    direct_metrics = []
    reason_metrics = []
    selected_gains = []
    selected_examples = []
    for rank, key in enumerate(keys, start=1):
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        chosen = reason_row if key in reason_keys else direct_row
        routed_metrics.append(row_metric(chosen))
        direct_metrics.append(row_metric(direct_row))
        reason_metrics.append(row_metric(reason_row))
        if key in reason_keys:
            gain = score(reason_row) - score(direct_row)
            selected_gains.append(gain)
            selected_examples.append(
                {
                    "rank": rank,
                    "wnd_id": key,
                    "delta_direct_minus_reason_route_nll": score_rows[key].get(
                        "delta_direct_minus_reason_route_nll"
                    ),
                    "direct_argument_f1": direct_row.get("argument_f1", 0.0),
                    "reason_argument_f1": reason_row.get("argument_f1", 0.0),
                    "direct_event_f1": direct_row.get("event_f1", 0.0),
                    "reason_event_f1": reason_row.get("event_f1", 0.0),
                    "direct_trigger_f1": direct_row.get("trigger_f1", 0.0),
                    "reason_trigger_f1": reason_row.get("trigger_f1", 0.0),
                    "score_gain": gain,
                }
            )

    direct = summarize_metrics(direct_metrics)
    reason = summarize_metrics(reason_metrics)
    routed = summarize_metrics(routed_metrics)
    return {
        "policy": policy["name"],
        "checkpoint": policy["checkpoint"],
        "source": policy["source"],
        "split": split,
        "num_examples": len(keys),
        "rank_window": {
            "start_pct": policy["start_pct"],
            "end_pct": policy["end_pct"],
            "start_rank": start + 1,
            "end_rank": end,
        },
        "pred_reason_count": len(reason_keys),
        "pred_reason_rate": len(reason_keys) / len(keys) if keys else 0.0,
        "selected_reason_avg_score_gain": (
            sum(selected_gains) / len(selected_gains) if selected_gains else 0.0
        ),
        "direct": direct,
        "forced_reason_all": reason,
        "routed": routed,
        "routed_minus_direct": {
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
        },
        "routed_minus_reason_all": {
            "trigger_f1": routed["trigger_f1"] - reason["trigger_f1"],
            "argument_f1": routed["argument_f1"] - reason["argument_f1"],
            "event_f1": routed["event_f1"] - reason["event_f1"],
        },
        "selected_examples": sorted(
            selected_examples,
            key=lambda row: row["delta_direct_minus_reason_route_nll"]
            if row["delta_direct_minus_reason_route_nll"] is not None
            else float("-inf"),
            reverse=True,
        )[:20],
    }


def aggregate_test(rows):
    out = []
    for policy in POLICIES:
        items = [row for row in rows if row["policy"] == policy["name"]]
        total = sum(row["num_examples"] for row in items)
        if not total:
            continue
        agg = {
            "policy": policy["name"],
            "checkpoint": policy["checkpoint"],
            "source": policy["source"],
            "split": "test",
            "num_examples": total,
            "pred_reason_count": sum(row["pred_reason_count"] for row in items),
        }
        agg["pred_reason_rate"] = agg["pred_reason_count"] / total
        denom = agg["pred_reason_count"]
        agg["selected_reason_avg_score_gain"] = (
            sum(row["selected_reason_avg_score_gain"] * row["pred_reason_count"] for row in items) / denom
            if denom
            else 0.0
        )
        for group in ["direct", "forced_reason_all", "routed"]:
            agg[group] = {}
            for metric in ["trigger_f1", "argument_f1", "event_f1"]:
                agg[group][metric] = sum(row[group][metric] * row["num_examples"] for row in items) / total
        agg["routed_minus_direct"] = {
            metric: agg["routed"][metric] - agg["direct"][metric]
            for metric in ["trigger_f1", "argument_f1", "event_f1"]
        }
        agg["routed_minus_reason_all"] = {
            metric: agg["routed"][metric] - agg["forced_reason_all"][metric]
            for metric in ["trigger_f1", "argument_f1", "event_f1"]
        }
        out.append(agg)
    return out


def signed(value):
    return f"{value:+.4f}"


def fmt(value):
    return f"{value:.4f}"


def pct(value):
    return f"{100 * value:.1f}%"


def render_table(rows):
    lines = [
        "| policy | split | reason rate | routed A/E/T | delta vs direct A/E/T | selected gain |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        routed = row["routed"]
        delta = row["routed_minus_direct"]
        lines.append(
            f"| `{row['policy']}` | `{row['split']}` | {pct(row['pred_reason_rate'])} | "
            f"{fmt(routed['argument_f1'])}/{fmt(routed['event_f1'])}/{fmt(routed['trigger_f1'])} | "
            f"{signed(delta['argument_f1'])}/{signed(delta['event_f1'])}/{signed(delta['trigger_f1'])} | "
            f"{signed(row['selected_reason_avg_score_gain'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    rows = sorted(payload["results"], key=lambda row: (row["policy"], row["split"]))
    lines = [
        "# M02 Rank-Window Formal Replay",
        "",
        "This report applies dev-locked rank-window policies to formal route-NLL scores. No formal labels are used for policy selection.",
        "",
        "## Policies",
        "",
        "- `aet_event_checkpoint50_rank225_400`: dev A/E/T event candidate, checkpoint-50, select ranks 22.5%-40%.",
        "- `robust_checkpoint100_rank20_40`: dev all-nonnegative policy, checkpoint-100, select ranks 20%-40%.",
        "- `event_checkpoint150_rank10_30`: dev best-event policy, checkpoint-150, select ranks 10%-30%.",
        "- `aet_balanced_checkpoint200_rank175_325`: dev A/E/T balanced candidate, checkpoint-200, select ranks 17.5%-32.5%.",
        "",
        "## Results",
        "",
        render_table(rows),
        "",
        "## Reading",
        "",
    ]
    for row in rows:
        if row["split"] != "test":
            continue
        delta = row["routed_minus_direct"]
        lines.append(
            f"- `{row['policy']}` on `test`: reason rate `{row['pred_reason_rate']:.1%}`, "
            f"A/E/T delta `{delta['argument_f1']:+.4f}/{delta['event_f1']:+.4f}/{delta['trigger_f1']:+.4f}`."
        )
    return "\n".join(lines) + "\n"


def main():
    split_rows = []
    for policy in POLICIES:
        for split in SPLITS:
            split_rows.append(evaluate(policy, split))
    rows = split_rows + aggregate_test(split_rows)
    payload = {
        "branch": BRANCH,
        "score_root": SCORE_ROOT.as_posix(),
        "direct_root": DIRECT_ROOT.as_posix(),
        "reason_root": REASON_ROOT.as_posix(),
        "policies": POLICIES,
        "results": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
