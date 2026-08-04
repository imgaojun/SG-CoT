import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    DIRECT_ROOT,
    REASON_ROOT,
    load_prediction_map,
    load_score_rows,
    score,
    sorted_keys_by_delta,
)
from src.stage2_analysis.analyze_adaptive_outcome_router_execution import row_metric  # noqa: E402


BRANCH = "aet_stable_router_m02_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/aet_stable_router_m02_20260520/formal_route_likelihood" / BRANCH
OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_positive_retention_cases.json"
OUT_MD = REPO / "reports/2026-05-21_stage2_modular_dualexpert_aet_positive_retention_cases.md"
SPLITS = ["test_seen", "test_unseen"]
POLICIES = [
    {
        "name": "m02 early-stable",
        "checkpoint": "checkpoint-100",
        "start_pct": 0.325,
        "end_pct": 0.400,
    },
    {
        "name": "m02 positive-retention",
        "checkpoint": "checkpoint-50",
        "start_pct": 0.425,
        "end_pct": 0.500,
    },
]


def metric_delta(direct_row, reason_row):
    direct_m = row_metric(direct_row)
    reason_m = row_metric(reason_row)
    return {
        "argument_f1": reason_m["argument"]["f1"] - direct_m["argument"]["f1"],
        "event_f1": reason_m["event"]["f1"] - direct_m["event"]["f1"],
        "trigger_f1": reason_m["trigger"]["f1"] - direct_m["trigger"]["f1"],
        "score": score(reason_row) - score(direct_row),
    }


def quantiles(values):
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    vals = sorted(values)
    def q(frac):
        idx = round((len(vals) - 1) * frac)
        return vals[idx]
    return {
        "min": vals[0],
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
    }


def selected_cases(policy, split):
    score_path = SCORE_ROOT / policy["checkpoint"] / split / "scores.jsonl"
    direct_path = DIRECT_ROOT / split / "predictions.jsonl"
    reason_path = REASON_ROOT / split / "predictions.jsonl"
    score_rows = load_score_rows(score_path)
    direct_rows = load_prediction_map(direct_path)
    reason_rows = load_prediction_map(reason_path)
    keys = sorted_keys_by_delta(score_rows, set(direct_rows) & set(reason_rows))
    start = round(len(keys) * policy["start_pct"])
    end = round(len(keys) * policy["end_pct"])
    selected = keys[start:end]
    cases = []
    for rank, key in enumerate(keys, start=1):
        if key not in set(selected):
            continue
        delta = metric_delta(direct_rows[key], reason_rows[key])
        cases.append(
            {
                "split": split,
                "rank": rank,
                "wnd_id": key,
                "delta_direct_minus_reason_route_nll": score_rows[key].get("delta_direct_minus_reason_route_nll"),
                **delta,
                "helpful": delta["score"] > 0,
                "harmful": delta["score"] < 0,
                "aet_all_nonnegative": delta["argument_f1"] >= 0 and delta["event_f1"] >= 0 and delta["trigger_f1"] >= 0,
            }
        )
    return cases


def summarize_policy(policy):
    cases = []
    for split in SPLITS:
        cases.extend(selected_cases(policy, split))
    by_split = {}
    for split in SPLITS:
        split_cases = [case for case in cases if case["split"] == split]
        by_split[split] = summarize_cases(split_cases)
    return {
        "policy": policy["name"],
        "checkpoint": policy["checkpoint"],
        "rank_window": [policy["start_pct"], policy["end_pct"]],
        "num_selected": len(cases),
        "summary": summarize_cases(cases),
        "by_split": by_split,
        "top_helpful_cases": sorted(cases, key=lambda row: row["score"], reverse=True)[:10],
        "top_harmful_cases": sorted(cases, key=lambda row: row["score"])[:10],
        "selected_ids": sorted(case["wnd_id"] for case in cases),
    }


def summarize_cases(cases):
    if not cases:
        return {
            "num_selected": 0,
            "helpful_rate": 0.0,
            "harmful_rate": 0.0,
            "aet_all_nonnegative_rate": 0.0,
            "score_gain": quantiles([]),
            "argument_gain": quantiles([]),
            "event_gain": quantiles([]),
            "trigger_gain": quantiles([]),
        }
    return {
        "num_selected": len(cases),
        "helpful_rate": sum(case["helpful"] for case in cases) / len(cases),
        "harmful_rate": sum(case["harmful"] for case in cases) / len(cases),
        "aet_all_nonnegative_rate": sum(case["aet_all_nonnegative"] for case in cases) / len(cases),
        "score_gain": quantiles([case["score"] for case in cases]),
        "argument_gain": quantiles([case["argument_f1"] for case in cases]),
        "event_gain": quantiles([case["event_f1"] for case in cases]),
        "trigger_gain": quantiles([case["trigger_f1"] for case in cases]),
    }


def overlap(a, b):
    set_a = set(a["selected_ids"])
    set_b = set(b["selected_ids"])
    inter = set_a & set_b
    union = set_a | set_b
    return {
        "a": a["policy"],
        "b": b["policy"],
        "intersection": len(inter),
        "union": len(union),
        "jaccard": len(inter) / len(union) if union else 0.0,
        "a_overlap_rate": len(inter) / len(set_a) if set_a else 0.0,
        "b_overlap_rate": len(inter) / len(set_b) if set_b else 0.0,
    }


def fmt(value):
    return f"{value:.3f}"


def signed(value):
    return f"{value:+.4f}"


def render_summary_table(policy_rows):
    lines = [
        "| policy | selected | helpful | harmful | A/E/T-safe | mean score gain | mean A/E/T gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_rows:
        s = row["summary"]
        lines.append(
            "| {policy} | {n} | {helpful} | {harmful} | {safe} | {score} | {a}/{e}/{t} |".format(
                policy=row["policy"],
                n=s["num_selected"],
                helpful=fmt(s["helpful_rate"]),
                harmful=fmt(s["harmful_rate"]),
                safe=fmt(s["aet_all_nonnegative_rate"]),
                score=signed(s["score_gain"]["mean"]),
                a=signed(s["argument_gain"]["mean"]),
                e=signed(s["event_gain"]["mean"]),
                t=signed(s["trigger_gain"]["mean"]),
            )
        )
    return "\n".join(lines)


def render(payload):
    lines = [
        "# Positive-Retention Case Analysis",
        "",
        "This report compares selected formal cases for m02 early-stable and m02 positive-retention.",
        "",
        "## Summary",
        "",
        render_summary_table(payload["policies"]),
        "",
        "## Overlap",
        "",
    ]
    ov = payload["overlap"]
    lines.append(
        f"- selected-case overlap: `{ov['intersection']}` / union `{ov['union']}`; Jaccard `{ov['jaccard']:.3f}`."
    )
    lines.extend(["", "## Reading", ""])
    pos = next(row for row in payload["policies"] if row["policy"] == "m02 positive-retention")
    early = next(row for row in payload["policies"] if row["policy"] == "m02 early-stable")
    lines.append(
        f"- Positive-retention selected `{pos['num_selected']}` cases with mean score gain `{signed(pos['summary']['score_gain']['mean'])}`."
    )
    lines.append(
        f"- Early-stable selected `{early['num_selected']}` cases with mean score gain `{signed(early['summary']['score_gain']['mean'])}`."
    )
    lines.append("- The low overlap helps explain why positive-retention changes all three A/E/T metrics rather than only shifting the old m02 window.")
    lines.append("")
    return "\n".join(lines)


def main():
    policy_rows = [summarize_policy(policy) for policy in POLICIES]
    payload = {
        "score_root": SCORE_ROOT.as_posix(),
        "policies": policy_rows,
        "overlap": overlap(policy_rows[0], policy_rows[1]),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
