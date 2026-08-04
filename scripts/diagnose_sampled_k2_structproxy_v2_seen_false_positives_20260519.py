#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402
from scripts.summarize_sampled_k2_structural_proxy_locked_validation_20260519 import (  # noqa: E402
    DEFAULT_FRESH_NLL_ROOT,
    DEFAULT_NEW_NLL_ROOT,
    DEFAULT_OLD_NLL_ROOT,
    DEFAULT_SAMPLE_ROOT,
    build_cases,
    route_reason,
)
from scripts.summarize_sampled_k2_structural_proxy_router_formal_20260519 import load_scores, routes_from_scores  # noqa: E402


DEFAULT_V2_SCORE_ROOT = REPO / (
    "outputs/stage2_adaptive_route_formal_nll_structproxy_v2_seedpair23_24_20260519/"
    "sampled_k2_structproxy_strictv2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
)
DEFAULT_OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_structproxy_v2_seen_fp_diagnosis_20260519"
DEFAULT_REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_structproxy_v2_seen_false_positive_diagnosis.md"
DEFAULT_REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_structproxy_v2_seen_false_positive_diagnosis.json"
DEFAULT_EXPERIMENT_NOTE = REPO / (
    "experiments/2026-05-19_stage2_sampled_k2_structproxy_v2_seen_false_positive_diagnosis_"
    "richere_split1_oracle_mixed_noise_qwen3_1_7b.md"
)
SPLIT = "test_seen"
FEATURES = [
    "v2_route_nll_margin",
    "fresh_margin",
    "old17_18_margin",
    "new19_20_margin",
    "margin_range",
    "num_margins_ge_0p25",
    "sample_arg_text_jaccard_mean",
    "sample_event_count_delta_mean",
    "single_gen_score_gain",
    "k2_expected_score_gain",
    "single_gen_direct_score",
    "single_gen_reason_score",
]


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gain_bucket(gain):
    if gain > 0:
        return "helpful"
    if gain < 0:
        return "harmful"
    return "neutral"


def compact_case(case):
    row = {
        "case_id": case["case_id"],
        "key": case["key"],
        "v2_route": case["v2_route"],
        "locked_route": case["locked_route"],
        "membership": case["membership"],
        "gain_bucket": case["gain_bucket"],
    }
    for feature in FEATURES:
        row[feature] = case.get(feature)
    return row


def summarize_group(name, cases):
    buckets = Counter(case["gain_bucket"] for case in cases)
    return {
        "group": name,
        "count": len(cases),
        "helpful": buckets["helpful"],
        "harmful": buckets["harmful"],
        "neutral": buckets["neutral"],
        "harm_rate": buckets["harmful"] / len(cases) if cases else 0.0,
        "mean_single_gen_gain": mean(case["single_gen_score_gain"] for case in cases),
        "mean_k2_expected_gain": mean(case["k2_expected_score_gain"] for case in cases),
        "features": {
            feature: mean(case[feature] for case in cases if case.get(feature) is not None)
            for feature in FEATURES
        },
    }


def select_cases(args):
    structural_args = argparse.Namespace(
        seeds=args.seeds,
        sample_root=args.sample_root,
        fresh_nll_root=args.fresh_nll_root,
        old_nll_root=args.old_nll_root,
        new_nll_root=args.new_nll_root,
        checkpoint=args.locked_checkpoint,
    )
    base_cases = build_cases(structural_args)[SPLIT]
    score_rows = load_scores(args.v2_score_root, args.v2_checkpoint, SPLIT)
    routes, policy = routes_from_scores(score_rows, args.v2_budget)
    if policy != args.v2_policy_name:
        raise ValueError((policy, args.v2_policy_name))
    score_by_key = dict(score_rows)
    cases = []
    for case in base_cases:
        key = case["key"]
        v2_reason = routes.get(key) == "reason"
        locked_reason = route_reason(case, "locked_structural_proxy")
        if v2_reason and locked_reason:
            membership = "overlap"
        elif v2_reason:
            membership = "v2_only"
        elif locked_reason:
            membership = "locked_only"
        else:
            membership = "neither"
        direct = case["single_gen_execution_direct"]
        reason = case["single_gen_execution_reason"]
        gain = reason["score"] - direct["score"]
        cases.append(
            {
                **case,
                "case_id": f"{SPLIT}::{key}",
                "v2_route": "reason" if v2_reason else "direct",
                "locked_route": "reason" if locked_reason else "direct",
                "membership": membership,
                "gain_bucket": gain_bucket(gain),
                "v2_route_nll_margin": score_by_key.get(key),
                "single_gen_score_gain": gain,
                "k2_expected_score_gain": case["k2_expected_reason"]["score"] - case["k2_expected_direct"]["score"],
                "single_gen_direct_score": direct["score"],
                "single_gen_reason_score": reason["score"],
            }
        )
    return cases


def render_group_table(groups):
    lines = [
        "| group | count | helpful/harmful/neutral | harm | single gain | K2 gain | v2 margin | fresh margin | margin range | >=.25 | arg J | event delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        feat = group["features"]
        lines.append(
            f"| `{group['group']}` | {group['count']} | "
            f"{group['helpful']}/{group['harmful']}/{group['neutral']} | "
            f"{pct(group['harm_rate'])} | {signed(group['mean_single_gen_gain'])} | {signed(group['mean_k2_expected_gain'])} | "
            f"{fmt(feat['v2_route_nll_margin'])} | {fmt(feat['fresh_margin'])} | "
            f"{fmt(feat['margin_range'])} | {fmt(feat['num_margins_ge_0p25'])} | "
            f"{fmt(feat['sample_arg_text_jaccard_mean'])} | {fmt(feat['sample_event_count_delta_mean'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    groups = payload["groups"]
    by_name = {row["group"]: row for row in groups}
    v2 = by_name["v2_selected"]
    locked = by_name["locked_selected"]
    v2_only = by_name["v2_only"]
    lines = [
        "# Sampled K2 StructProxy v2 Seen False-Positive Diagnosis",
        "",
        "This compares the strict-v2 trained selector policy `checkpoint-50/top050` against the locked structural proxy on `test_seen` seedpair23/24 formal cases.",
        "",
        "## Summary",
        "",
        render_group_table(groups),
        "",
        "## Reading",
        "",
        f"- v2 selected `{v2['count']}` seen cases; locked structural proxy selected `{locked['count']}`.",
        f"- v2-only selected cases: `{v2_only['count']}`, with harm `{v2_only['harmful']}` and mean single-generation gain `{v2_only['mean_single_gen_gain']:+.4f}`.",
        "- The key diagnostic question is whether v2-only cases look structurally weaker than locked-only or overlap cases.",
        "- If v2-only harmful cases have low cross-seed margin agreement or weaker fresh margins, the next policy should require margin consensus before allowing learned-selector expansion.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['report_json']}`",
        f"- selected cases JSONL: `{payload['selected_cases_jsonl']}`",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    cases = select_cases(args)
    selected_cases = [case for case in cases if case["membership"] != "neither"]
    groups = [
        summarize_group("v2_selected", [case for case in cases if case["v2_route"] == "reason"]),
        summarize_group("locked_selected", [case for case in cases if case["locked_route"] == "reason"]),
        summarize_group("overlap", [case for case in cases if case["membership"] == "overlap"]),
        summarize_group("v2_only", [case for case in cases if case["membership"] == "v2_only"]),
        summarize_group("locked_only", [case for case in cases if case["membership"] == "locked_only"]),
        summarize_group("neither", [case for case in cases if case["membership"] == "neither"]),
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_cases_jsonl = args.output_root / "selected_cases.jsonl"
    write_jsonl(selected_cases_jsonl, [compact_case(case) for case in selected_cases])
    payload = {
        "split": SPLIT,
        "seeds": args.seeds,
        "v2_score_root": args.v2_score_root.as_posix(),
        "v2_checkpoint": args.v2_checkpoint,
        "v2_policy": args.v2_policy_name,
        "locked_rule": "fresh_margin >= 0.25 and margin_range <= 0.50 and num_margins_ge_0p25 >= 2 and sample_arg_text_jaccard_mean >= 0.40 and sample_event_count_delta_mean <= 0",
        "num_cases": len(cases),
        "groups": groups,
        "selected_cases_jsonl": selected_cases_jsonl.as_posix(),
        "report_md": args.report_md.as_posix(),
        "report_json": args.report_json.as_posix(),
    }
    write_json(args.report_json, payload)
    write_json(args.output_root / "summary.json", payload)
    write_text(args.report_md, render_report(payload))
    print(json.dumps({"report_md": args.report_md.as_posix(), "report_json": args.report_json.as_posix()}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 24])
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--fresh-nll-root", type=Path, default=DEFAULT_FRESH_NLL_ROOT)
    parser.add_argument("--old-nll-root", type=Path, default=DEFAULT_OLD_NLL_ROOT)
    parser.add_argument("--new-nll-root", type=Path, default=DEFAULT_NEW_NLL_ROOT)
    parser.add_argument("--locked-checkpoint", default="checkpoint-50")
    parser.add_argument("--v2-score-root", type=Path, default=DEFAULT_V2_SCORE_ROOT)
    parser.add_argument("--v2-checkpoint", default="checkpoint-50")
    parser.add_argument("--v2-budget", type=float, default=0.05)
    parser.add_argument("--v2-policy-name", default="top050")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--experiment-note", type=Path, default=DEFAULT_EXPERIMENT_NOTE)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
