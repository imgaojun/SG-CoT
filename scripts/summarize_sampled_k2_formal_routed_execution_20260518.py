#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


EXEC_ROOT = REPO / "outputs/stage2_adaptive_route_formal_execution_20260518/sampledk2_ckpt50_margin025"
NLL_ROOT = (
    REPO
    / "outputs/stage2_adaptive_route_formal_nll_20260518"
    / "sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
    / "checkpoint-50"
)
REPORT_MD = REPO / "reports/2026-05-18_stage2_sampled_k2_formal_routed_execution.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-18_stage2_sampled_k2_formal_routed_execution.json"
SPLITS = ["test_seen", "test_unseen"]
THRESHOLD = 0.25


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def key_for(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("wnd_id")


def metric_score(row):
    return row.get("argument_f1", 0.0) + row.get("event_f1", 0.0) + 0.25 * row.get("trigger_f1", 0.0)


def load_predictions(mode: str, split: str):
    path = EXEC_ROOT / mode / split / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = {}
    for row in load_jsonl(path):
        key = key_for(row)
        if key:
            rows[key] = row
    return rows


def load_routes(split: str):
    path = NLL_ROOT / split / "scores.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    routes = {}
    margins = {}
    for row in load_jsonl(path):
        key = key_for(row)
        if not key:
            continue
        margin = row.get("delta_direct_minus_reason_route_nll")
        margins[key] = margin
        routes[key] = "reason" if margin is not None and margin >= THRESHOLD else "direct"
    return routes, margins


def summarize_rows(rows):
    if not rows:
        return {
            "num_examples": 0,
            "json_valid_rate": 0.0,
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
            "score": 0.0,
        }
    return {
        "num_examples": len(rows),
        "json_valid_rate": sum(1 for row in rows if row.get("valid_final_json", row.get("valid_json"))) / len(rows),
        "trigger_f1": sum(row.get("trigger_f1", 0.0) for row in rows) / len(rows),
        "argument_f1": sum(row.get("argument_f1", 0.0) for row in rows) / len(rows),
        "event_f1": sum(row.get("event_f1", 0.0) for row in rows) / len(rows),
        "score": sum(metric_score(row) for row in rows) / len(rows),
    }


def route_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize_split(split: str):
    direct = load_predictions("forced_direct", split)
    reason = load_predictions("forced_reason", split)
    routes, margins = load_routes(split)
    keys = sorted(set(direct) & set(reason) & set(routes))
    if not keys:
        raise ValueError(f"no common execution keys for {split}")

    direct_rows = []
    reason_rows = []
    routed_rows = []
    selected = []
    helpful = set()
    selected_set = set()
    gain_rows = []
    for key in keys:
        drow = direct[key]
        rrow = reason[key]
        direct_rows.append(drow)
        reason_rows.append(rrow)
        gain = metric_score(rrow) - metric_score(drow)
        if gain > 0:
            helpful.add(key)
        route = routes[key]
        if route == "reason":
            selected_set.add(key)
            selected.append(
                {
                    "wnd_id": key,
                    "margin": margins[key],
                    "score_gain": gain,
                    "argument_gain": rrow.get("argument_f1", 0.0) - drow.get("argument_f1", 0.0),
                    "event_gain": rrow.get("event_f1", 0.0) - drow.get("event_f1", 0.0),
                    "trigger_gain": rrow.get("trigger_f1", 0.0) - drow.get("trigger_f1", 0.0),
                }
            )
        routed_rows.append(rrow if route == "reason" else drow)
        gain_rows.append(gain)

    tp = len(selected_set & helpful)
    fp = len(selected_set - helpful)
    fn = len(helpful - selected_set)
    direct_summary = summarize_rows(direct_rows)
    reason_summary = summarize_rows(reason_rows)
    routed_summary = summarize_rows(routed_rows)
    return {
        "split": split,
        "num_examples": len(keys),
        "threshold": THRESHOLD,
        "pred_reason_count": len(selected_set),
        "pred_reason_rate": len(selected_set) / len(keys),
        "single_gen_reason_helpful_count": len(helpful),
        "single_gen_reason_helpful_rate": len(helpful) / len(keys),
        "route_vs_single_gen_helpful": route_prf(tp, fp, fn),
        "selected_reason_score_gain_mean": (
            sum(row["score_gain"] for row in selected) / len(selected) if selected else 0.0
        ),
        "selected_reason_harm_count": sum(1 for row in selected if row["score_gain"] < 0),
        "selected_reason_harm_rate": (
            sum(1 for row in selected if row["score_gain"] < 0) / len(selected) if selected else 0.0
        ),
        "direct": direct_summary,
        "reason_all": reason_summary,
        "routed": routed_summary,
        "routed_minus_direct": {
            metric: routed_summary[metric] - direct_summary[metric]
            for metric in ["json_valid_rate", "trigger_f1", "argument_f1", "event_f1", "score"]
        },
        "routed_minus_reason_all": {
            metric: routed_summary[metric] - reason_summary[metric]
            for metric in ["json_valid_rate", "trigger_f1", "argument_f1", "event_f1", "score"]
        },
        "selected_reason_best": sorted(selected, key=lambda row: row["score_gain"], reverse=True)[:10],
        "selected_reason_worst": sorted(selected, key=lambda row: row["score_gain"])[:10],
    }


def aggregate(rows):
    total = sum(row["num_examples"] for row in rows)
    out = {
        "split": "test",
        "num_examples": total,
        "threshold": THRESHOLD,
        "pred_reason_count": sum(row["pred_reason_count"] for row in rows),
        "single_gen_reason_helpful_count": sum(row["single_gen_reason_helpful_count"] for row in rows),
    }
    out["pred_reason_rate"] = out["pred_reason_count"] / total if total else 0.0
    out["single_gen_reason_helpful_rate"] = out["single_gen_reason_helpful_count"] / total if total else 0.0
    for name in ["direct", "reason_all", "routed"]:
        out[name] = {}
        for metric in ["json_valid_rate", "trigger_f1", "argument_f1", "event_f1", "score"]:
            out[name][metric] = sum(row[name][metric] * row["num_examples"] for row in rows) / total
        out[name]["num_examples"] = total
    out["routed_minus_direct"] = {
        metric: out["routed"][metric] - out["direct"][metric]
        for metric in ["json_valid_rate", "trigger_f1", "argument_f1", "event_f1", "score"]
    }
    out["routed_minus_reason_all"] = {
        metric: out["routed"][metric] - out["reason_all"][metric]
        for metric in ["json_valid_rate", "trigger_f1", "argument_f1", "event_f1", "score"]
    }
    tp = sum(
        row["route_vs_single_gen_helpful"]["precision"]
        * row["pred_reason_count"]
        for row in rows
    )
    fp = out["pred_reason_count"] - tp
    fn = out["single_gen_reason_helpful_count"] - tp
    out["route_vs_single_gen_helpful"] = route_prf(tp, fp, fn)
    selected_total = out["pred_reason_count"]
    out["selected_reason_score_gain_mean"] = (
        sum(row["selected_reason_score_gain_mean"] * row["pred_reason_count"] for row in rows) / selected_total
        if selected_total
        else 0.0
    )
    out["selected_reason_harm_count"] = sum(row["selected_reason_harm_count"] for row in rows)
    out["selected_reason_harm_rate"] = (
        out["selected_reason_harm_count"] / selected_total if selected_total else 0.0
    )
    out["selected_reason_best"] = []
    out["selected_reason_worst"] = []
    return out


def metric_cell(row):
    return f"{fmt(row['argument_f1'])}/{fmt(row['event_f1'])}/{fmt(row['trigger_f1'])}/{fmt(row['score'])}"


def delta_cell(row):
    return f"{signed(row['argument_f1'])}/{signed(row['event_f1'])}/{signed(row['trigger_f1'])}/{signed(row['score'])}"


def render_summary_table(rows):
    lines = [
        "| split | system | reason rate | JSON | A/E/T/Score | delta vs Direct A/E/T/Score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        for system, key, reason_rate in [
            ("Direct-all", "direct", 0.0),
            ("Reason-all", "reason_all", 1.0),
            ("Routed", "routed", row["pred_reason_rate"]),
        ]:
            delta = {metric: 0.0 for metric in ["argument_f1", "event_f1", "trigger_f1", "score"]}
            if system == "Reason-all":
                delta = {
                    metric: row["reason_all"][metric] - row["direct"][metric]
                    for metric in ["argument_f1", "event_f1", "trigger_f1", "score"]
                }
            elif system == "Routed":
                delta = row["routed_minus_direct"]
            lines.append(
                f"| `{row['split']}` | {system} | {pct(reason_rate)} | {fmt(row[key]['json_valid_rate'])} | "
                f"{metric_cell(row[key])} | {delta_cell(delta)} |"
            )
    return "\n".join(lines)


def render_policy_table(rows):
    lines = [
        "| split | selected Reason | single-gen helpful rate | P/R/F1 vs helpful | selected gain mean | harm rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        prf = row["route_vs_single_gen_helpful"]
        lines.append(
            f"| `{row['split']}` | {row['pred_reason_count']} ({pct(row['pred_reason_rate'])}) | "
            f"{pct(row['single_gen_reason_helpful_rate'])} | "
            f"{fmt(prf['precision'], 3)}/{fmt(prf['recall'], 3)}/{fmt(prf['f1'], 3)} | "
            f"{signed(row['selected_reason_score_gain_mean'])} | {pct(row['selected_reason_harm_rate'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    rows = payload["results"]
    test = next(row for row in rows if row["split"] == "test")
    lines = [
        "# Sampled K2 Formal Routed Execution",
        "",
        "This report freezes the route policy `checkpoint-50 / margin >= 0.25`, then executes routing by selecting between single deterministic forced-direct and forced-reason generations.",
        "",
        "## Extraction Metrics",
        "",
        render_summary_table(rows),
        "",
        "## Route Decision Quality",
        "",
        render_policy_table(rows),
        "",
        "## Reading",
        "",
        f"- Aggregated `test` routed execution score delta vs Direct is `{test['routed_minus_direct']['score']:+.4f}` with Reason rate `{test['pred_reason_rate']:.1%}`.",
        f"- Argument/event/trigger deltas are `{test['routed_minus_direct']['argument_f1']:+.4f}` / `{test['routed_minus_direct']['event_f1']:+.4f}` / `{test['routed_minus_direct']['trigger_f1']:+.4f}`.",
        f"- Single-generation selected Reason gain mean is `{test['selected_reason_score_gain_mean']:+.4f}` with harm rate `{test['selected_reason_harm_rate']:.1%}`.",
        "",
        "## Inputs",
        "",
        f"- execution root: `{EXEC_ROOT}`",
        f"- route-NLL root: `{NLL_ROOT}`",
        f"- threshold: `{THRESHOLD}`",
        f"- artifact JSON: `{REPORT_JSON}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    split_rows = [summarize_split(split) for split in SPLITS]
    rows = [aggregate(split_rows), *split_rows]
    payload = {
        "execution_root": EXEC_ROOT.as_posix(),
        "route_nll_root": NLL_ROOT.as_posix(),
        "policy": {"checkpoint": "checkpoint-50", "margin_threshold": THRESHOLD},
        "results": rows,
    }
    write_json(REPORT_JSON, payload)
    write_text(REPORT_MD, render_report(payload))
    print(json.dumps({"output_json": REPORT_JSON.as_posix(), "output_md": REPORT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
