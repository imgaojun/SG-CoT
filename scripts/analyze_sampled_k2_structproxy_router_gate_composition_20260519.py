#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
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
    SPLITS,
    avg_metrics,
    build_cases,
    route_reason,
)
from scripts.summarize_sampled_k2_structural_proxy_router_formal_20260519 import (  # noqa: E402
    CHECKPOINTS,
    DEFAULT_SCORE_ROOT,
    load_scores,
    routes_from_scores,
)


METRICS = ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json"]
DEFAULT_OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_route_formal_structproxy_router_gate_composition_20260519"
DEFAULT_REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_structproxy_router_gate_composition.md"
DEFAULT_REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_structproxy_router_gate_composition.json"
DEFAULT_EXPERIMENT_NOTE = REPO / (
    "experiments/2026-05-19_stage2_sampled_k2_structproxy_router_gate_composition_"
    "richere_split1_oracle_mixed_noise_qwen3_1_7b.md"
)
DEFAULT_FOCUS = [
    "checkpoint-50:top150",
    "checkpoint-50:top200",
    "checkpoint-75:argmin",
    "checkpoint-75:top100",
    "checkpoint-75:top150",
    "checkpoint-98:argmin",
    "checkpoint-98:top100",
    "checkpoint-98:top150",
]
FEATURES = [
    "fresh_margin",
    "margin_range",
    "num_margins_ge_0p25",
    "sample_arg_text_jaccard_mean",
    "sample_event_count_delta_mean",
]


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def gate_pass(case, gate):
    if gate == "none":
        return True
    if gate in {"fresh_margin", "base_margin_stability", "locked_structural_proxy"}:
        return route_reason(case, gate)
    if gate == "structural_only":
        return case["sample_arg_text_jaccard_mean"] >= 0.40 and case["sample_event_count_delta_mean"] <= 0.0
    if gate == "loose_structural_only":
        return case["sample_arg_text_jaccard_mean"] >= 0.25 and case["sample_event_count_delta_mean"] <= 0.5
    raise KeyError(gate)


def summarize_routing(cases, source, policy_name, routes, gate):
    direct_rows = []
    routed_rows = []
    selected_gains = []
    selected_cases = []
    for case in cases:
        key = case["key"]
        direct = case[f"{source}_direct"]
        reason = case[f"{source}_reason"]
        direct_rows.append(direct)
        selected = routes.get(key) == "reason" and gate_pass(case, gate)
        if selected:
            gain = reason["score"] - direct["score"]
            selected_gains.append(gain)
            selected_cases.append({**case, "gain": gain})
            routed_rows.append(reason)
        else:
            routed_rows.append(direct)
    direct_summary = avg_metrics(direct_rows)
    routed_summary = avg_metrics(routed_rows)
    selected_count = len(selected_cases)
    return {
        "source": source,
        "policy": policy_name,
        "gate": gate,
        "num_examples": len(cases),
        "pred_reason_count": selected_count,
        "pred_reason_rate": selected_count / len(cases) if cases else 0.0,
        "selected_reason_score_gain_mean": mean(selected_gains),
        "selected_reason_harm_rate": mean(1.0 if gain < 0 else 0.0 for gain in selected_gains),
        "selected_helpful_count": sum(1 for gain in selected_gains if gain > 0),
        "selected_harmful_count": sum(1 for gain in selected_gains if gain < 0),
        "selected_neutral_count": sum(1 for gain in selected_gains if gain == 0),
        "direct": direct_summary,
        "routed": routed_summary,
        "routed_minus_direct": {
            metric: routed_summary[metric] - direct_summary[metric]
            for metric in METRICS
        },
    }


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["policy"], row["gate"])].append(row)
    out = []
    for (source, policy, gate), items in grouped.items():
        total = sum(row["num_examples"] for row in items)
        selected = sum(row["pred_reason_count"] for row in items)
        agg = {
            "split": "test",
            "source": source,
            "policy": policy,
            "gate": gate,
            "num_examples": total,
            "pred_reason_count": selected,
            "pred_reason_rate": selected / total if total else 0.0,
            "selected_helpful_count": sum(row["selected_helpful_count"] for row in items),
            "selected_harmful_count": sum(row["selected_harmful_count"] for row in items),
            "selected_neutral_count": sum(row["selected_neutral_count"] for row in items),
        }
        agg["selected_reason_score_gain_mean"] = (
            sum(row["selected_reason_score_gain_mean"] * row["pred_reason_count"] for row in items) / selected
            if selected else 0.0
        )
        agg["selected_reason_harm_rate"] = (
            sum(row["selected_harmful_count"] for row in items) / selected
            if selected else 0.0
        )
        for route in ["direct", "routed"]:
            agg[route] = {
                metric: sum(row[route][metric] * row["num_examples"] for row in items) / total
                for metric in METRICS
            }
        agg["routed_minus_direct"] = {
            metric: agg["routed"][metric] - agg["direct"][metric]
            for metric in METRICS
        }
        out.append(agg)
    return out


def feature_profile(cases, routes, gate, split):
    groups = {"helpful": [], "harmful": [], "neutral": [], "blocked_by_gate": []}
    for case in cases:
        if routes.get(case["key"]) != "reason":
            continue
        if not gate_pass(case, gate):
            groups["blocked_by_gate"].append(case)
            continue
        gain = case["single_gen_execution_reason"]["score"] - case["single_gen_execution_direct"]["score"]
        if gain > 0:
            groups["helpful"].append(case)
        elif gain < 0:
            groups["harmful"].append(case)
        else:
            groups["neutral"].append(case)
    profiles = {}
    for name, rows in groups.items():
        profiles[name] = {
            "count": len(rows),
            "features": {feature: mean(row[feature] for row in rows) for feature in FEATURES},
        }
    return {"split": split, "gate": gate, "groups": profiles}


def parse_focus(items):
    parsed = []
    for item in items:
        checkpoint, policy = item.split(":", 1)
        if policy == "argmin":
            budget = None
        elif policy.startswith("top"):
            budget = int(policy[3:]) / 1000.0
        else:
            raise ValueError(f"unknown focus policy: {item}")
        parsed.append((checkpoint, budget, policy))
    return parsed


def delta_cell(row):
    delta = row["routed_minus_direct"]
    return f"{signed(delta['argument_f1'])}/{signed(delta['event_f1'])}/{signed(delta['trigger_f1'])}/{signed(delta['score'])}"


def render_results_table(rows, source):
    lines = [
        "| policy | gate | split | reason rate | delta A/E/T/Score | harm | selected gain | selected H/+/-/0 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    order = {"test": 0, "test_seen": 1, "test_unseen": 2}
    filtered = [row for row in rows if row["source"] == source]
    filtered.sort(key=lambda row: (row["policy"], row["gate"], order.get(row["split"], 9)))
    for row in filtered:
        lines.append(
            f"| `{row['policy']}` | `{row['gate']}` | `{row['split']}` | {pct(row['pred_reason_rate'])} | "
            f"{delta_cell(row)} | {pct(row['selected_reason_harm_rate'])} | "
            f"{signed(row['selected_reason_score_gain_mean'])} | "
            f"{row['pred_reason_count']}/{row['selected_helpful_count']}/{row['selected_harmful_count']}/{row['selected_neutral_count']} |"
        )
    return "\n".join(lines)


def render_profile_table(profiles):
    lines = [
        "| split | gate | group | count | fresh margin | margin range | margins >= .25 | arg text J | event delta |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in profiles:
        for group, data in profile["groups"].items():
            feat = data["features"]
            lines.append(
                f"| `{profile['split']}` | `{profile['gate']}` | `{group}` | {data['count']} | "
                f"{fmt(feat['fresh_margin'])} | {fmt(feat['margin_range'])} | "
                f"{fmt(feat['num_margins_ge_0p25'])} | {fmt(feat['sample_arg_text_jaccard_mean'])} | "
                f"{fmt(feat['sample_event_count_delta_mean'])} |"
            )
    return "\n".join(lines)


def render_report(payload):
    test_rows = [
        row for row in payload["results"]
        if row["split"] == "test" and row["source"] == "single_gen_execution"
    ]
    best = max(test_rows, key=lambda row: row["routed_minus_direct"]["score"])
    lines = [
        "# Sampled K2 StructProxy Router Gate Composition",
        "",
        "This is a no-training post-hoc diagnosis. It composes the trained supervised router ranking with gold-free structural gates and evaluates routed offline forced direct/reason execution.",
        "",
        f"- selector score root: `{payload['selector_score_root']}`",
        f"- structural evidence seedpair: `{payload['seeds']}`",
        f"- structural fresh NLL root: `{payload['fresh_nll_root']}`",
        "",
        "## Single-Generation Execution",
        "",
        render_results_table(payload["results"], "single_gen_execution"),
        "",
        "## K2 Expected",
        "",
        render_results_table(payload["results"], "k2_expected"),
        "",
        "## False-Positive Profile",
        "",
        render_profile_table(payload["false_positive_profiles"]),
        "",
        "## Reading",
        "",
        f"- Best single-generation composition: `{best['policy']} + {best['gate']}` with score delta `{best['routed_minus_direct']['score']:+.4f}`, reason rate `{best['pred_reason_rate']:.1%}`, harm `{best['selected_reason_harm_rate']:.1%}`.",
        "- Compare this against the locked structural proxy reference: score delta `+0.0085`, reason rate `3.4%`, harm `13.3%`.",
        "- If selector+gate cannot beat the locked rule, the next optimization should change supervision labels rather than only inference thresholds.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['report_json']}`",
        f"- output summary: `{Path(payload['output_root']) / 'summary.json'}`",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    structural_args = argparse.Namespace(
        seeds=args.seeds,
        sample_root=args.sample_root,
        fresh_nll_root=args.fresh_nll_root,
        old_nll_root=args.old_nll_root,
        new_nll_root=args.new_nll_root,
        checkpoint=args.structural_checkpoint,
    )
    cases_by_split = build_cases(structural_args)
    focus = parse_focus(args.focus)
    gates = args.gates
    results = []
    profiles = []
    for checkpoint, budget, policy in focus:
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"unsupported checkpoint: {checkpoint}")
        policy_name = f"{checkpoint}/{policy}"
        routes_by_split = {}
        for split in SPLITS:
            score_rows = load_scores(args.selector_score_root, checkpoint, split)
            routes, resolved_policy = routes_from_scores(score_rows, budget)
            if resolved_policy != policy:
                raise ValueError((resolved_policy, policy))
            routes_by_split[split] = routes
            for gate in gates:
                for source in ["single_gen_execution", "k2_expected"]:
                    row = summarize_routing(cases_by_split[split], source, policy_name, routes, gate)
                    row["split"] = split
                    results.append(row)
        for gate in ["none", "locked_structural_proxy"]:
            profiles.append(feature_profile(cases_by_split["test_seen"], routes_by_split["test_seen"], gate, "test_seen"))
    results.extend(aggregate_rows([row for row in results if row["split"] in SPLITS]))
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": args.seeds,
        "selector_score_root": args.selector_score_root.as_posix(),
        "sample_root": args.sample_root.as_posix(),
        "fresh_nll_root": args.fresh_nll_root.as_posix(),
        "old_nll_root": args.old_nll_root.as_posix(),
        "new_nll_root": args.new_nll_root.as_posix(),
        "structural_checkpoint": args.structural_checkpoint,
        "focus": args.focus,
        "gates": gates,
        "splits": {split: len(cases) for split, cases in cases_by_split.items()},
        "results": results,
        "false_positive_profiles": profiles,
        "output_root": args.output_root.as_posix(),
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
    parser.add_argument("--selector-score-root", type=Path, default=DEFAULT_SCORE_ROOT)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--fresh-nll-root", type=Path, default=DEFAULT_FRESH_NLL_ROOT)
    parser.add_argument("--old-nll-root", type=Path, default=DEFAULT_OLD_NLL_ROOT)
    parser.add_argument("--new-nll-root", type=Path, default=DEFAULT_NEW_NLL_ROOT)
    parser.add_argument("--structural-checkpoint", default="checkpoint-50")
    parser.add_argument("--focus", nargs="+", default=DEFAULT_FOCUS)
    parser.add_argument(
        "--gates",
        nargs="+",
        default=["none", "fresh_margin", "base_margin_stability", "loose_structural_only", "structural_only", "locked_structural_proxy"],
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--experiment-note", type=Path, default=DEFAULT_EXPERIMENT_NOTE)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
