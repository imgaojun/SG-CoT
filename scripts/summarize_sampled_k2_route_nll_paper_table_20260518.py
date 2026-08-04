#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


FORMAL_JSON = REPO / "reports/artifacts/2026-05-18_stage2_sampled_k2_formal_route_nll_probe.json"
DEV_JSON = REPO / "reports/artifacts/2026-05-18_stage2_sampled_k2_seedpair_route_nll_margin_calibration.json"
REPORT_MD = REPO / "reports/2026-05-18_stage2_sampled_k2_route_nll_paper_table.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-18_stage2_sampled_k2_route_nll_paper_table.json"

MAIN = ("checkpoint-50", "margin_ge_0p25")
COMPARATOR = ("checkpoint-75", "margin_ge_0p05")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def formal_row(data, checkpoint, policy, split):
    for row in data["results"]:
        if row["checkpoint"] == checkpoint and row["policy"] == policy and row["split"] == split:
            return row
    raise KeyError((checkpoint, policy, split))


def dev_aggregate(data, checkpoint, policy):
    for row in data["aggregates"]:
        if row["checkpoint"] == checkpoint and row["policy"] == policy:
            return row
    raise KeyError((checkpoint, policy))


def metric_cell(row, route):
    item = row[route]
    return f"{fmt(item['argument_f1'])}/{fmt(item['event_f1'])}/{fmt(item['trigger_f1'])}/{fmt(item['score'])}"


def delta_cell(delta):
    return (
        f"{signed(delta['argument_f1'])}/{signed(delta['event_f1'])}/"
        f"{signed(delta['trigger_f1'])}/{signed(delta['score'])}"
    )


def formal_system_rows(formal):
    rows = []
    main_by_split = {split: formal_row(formal, *MAIN, split) for split in ["test", "test_seen", "test_unseen"]}
    comp_by_split = {split: formal_row(formal, *COMPARATOR, split) for split in ["test", "test_seen", "test_unseen"]}
    for split in ["test", "test_seen", "test_unseen"]:
        base = main_by_split[split]
        direct_delta = {k: 0.0 for k in ["argument_f1", "event_f1", "trigger_f1", "score"]}
        reason_delta = {
            k: base["reason"][k] - base["direct"][k]
            for k in ["argument_f1", "event_f1", "trigger_f1", "score"]
        }
        rows.extend(
            [
                {
                    "split": split,
                    "system": "Direct-all",
                    "checkpoint": None,
                    "policy": None,
                    "reason_rate": 0.0,
                    "metrics": base["direct"],
                    "delta_vs_direct": direct_delta,
                },
                {
                    "split": split,
                    "system": "Reason-all",
                    "checkpoint": None,
                    "policy": None,
                    "reason_rate": 1.0,
                    "metrics": base["reason"],
                    "delta_vs_direct": reason_delta,
                },
                {
                    "split": split,
                    "system": "Routed main",
                    "checkpoint": MAIN[0],
                    "policy": MAIN[1],
                    "reason_rate": base["pred_reason_rate"],
                    "metrics": base["routed"],
                    "delta_vs_direct": base["routed_minus_direct"],
                },
                {
                    "split": split,
                    "system": "Routed comparator",
                    "checkpoint": COMPARATOR[0],
                    "policy": COMPARATOR[1],
                    "reason_rate": comp_by_split[split]["pred_reason_rate"],
                    "metrics": comp_by_split[split]["routed"],
                    "delta_vs_direct": comp_by_split[split]["routed_minus_direct"],
                },
            ]
        )
    return rows


def render_formal_table(rows):
    lines = [
        "| split | system | route policy | reason rate | A/E/T/Score | delta vs Direct A/E/T/Score |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        policy = "-"
        if row["checkpoint"]:
            policy = f"{row['checkpoint']} / {row['policy']}"
        lines.append(
            f"| `{row['split']}` | {row['system']} | `{policy}` | {pct(row['reason_rate'])} | "
            f"{metric_cell({'x': row['metrics']}, 'x')} | {delta_cell(row['delta_vs_direct'])} |"
        )
    return "\n".join(lines)


def render_dev_table(dev_rows):
    lines = [
        "| policy | reason rate | label P/R/F1 | score min/mean/max | trigger min/mean | mean delta A/E/T |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in dev_rows:
        lines.append(
            f"| {name} | {pct(row['pred_reason_mean'])} | "
            f"{fmt(row['precision_mean'], 3)}/{fmt(row['recall_mean'], 3)}/{fmt(row['reason_f1_mean'], 3)} | "
            f"{signed(row['score_min'])}/{signed(row['score_mean'])}/{signed(row['score_max'])} | "
            f"{signed(row['trigger_min'])}/{signed(row['trigger_mean'])} | "
            f"{signed(row['argument_mean'])}/{signed(row['event_mean'])}/{signed(row['trigger_mean'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    main_test = next(
        row for row in payload["formal_system_rows"]
        if row["split"] == "test" and row["system"] == "Routed main"
    )
    comp_test = next(
        row for row in payload["formal_system_rows"]
        if row["split"] == "test" and row["system"] == "Routed comparator"
    )
    lines = [
        "# Sampled K2 Route-NLL Paper Table",
        "",
        "This report consolidates the dev seed-pair robustness evidence and the formal K2 expected-utility result for the fixed-margin route-NLL selector.",
        "",
        "## Dev Robustness",
        "",
        render_dev_table(payload["dev_rows"]),
        "",
        "## Formal Expected Utility",
        "",
        render_formal_table(payload["formal_system_rows"]),
        "",
        "## Reading",
        "",
        f"- Main policy `checkpoint-50 / margin >= 0.25` improves aggregated formal `test` by "
        f"`{delta_cell(main_test['delta_vs_direct'])}` A/E/T/Score at `{main_test['reason_rate']:.1%}` Reason rate.",
        f"- Comparator `checkpoint-75 / margin >= 0.05` is not a good main result: aggregated formal `test` score delta is "
        f"`{comp_test['delta_vs_direct']['score']:+.4f}` and event delta is `{comp_test['delta_vs_direct']['event_f1']:+.4f}`.",
        "- The dev result should be read as seed-pair robustness under K8 labels; the formal result should be read as K2 sampled expected utility, not a single deterministic routed execution.",
        "",
        "## Inputs",
        "",
        f"- formal JSON: `{FORMAL_JSON}`",
        f"- dev JSON: `{DEV_JSON}`",
        f"- artifact JSON: `{REPORT_JSON}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    formal = load_json(FORMAL_JSON)
    dev = load_json(DEV_JSON)
    dev_rows = [
        ("Main: checkpoint-50 / margin >= 0.25", dev_aggregate(dev, *MAIN)),
        ("Comparator: checkpoint-75 / margin >= 0.05", dev_aggregate(dev, *COMPARATOR)),
    ]
    payload = {
        "formal_json": FORMAL_JSON.as_posix(),
        "dev_json": DEV_JSON.as_posix(),
        "main_policy": {"checkpoint": MAIN[0], "policy": MAIN[1]},
        "comparator_policy": {"checkpoint": COMPARATOR[0], "policy": COMPARATOR[1]},
        "dev_rows": dev_rows,
        "formal_system_rows": formal_system_rows(formal),
    }
    write_json(REPORT_JSON, payload)
    write_text(REPORT_MD, render_report(payload))
    print(json.dumps({"output_json": REPORT_JSON.as_posix(), "output_md": REPORT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
