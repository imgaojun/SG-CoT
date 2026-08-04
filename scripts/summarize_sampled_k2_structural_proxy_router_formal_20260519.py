#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.diagnose_sampled_k2_goldfree_harmful_cases_20260519 import (  # noqa: E402
    avg_metric_dict,
    load_exec_rows,
    load_sample_rows,
)
from scripts.diagnose_sampled_k2_formal_unseen_false_positives_20260519 import key_for, metric_dict  # noqa: E402
from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


SPLITS = ["test_seen", "test_unseen"]
METRICS = ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json"]
DEFAULT_BRANCH = "sampled_k2_structproxy_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
DEFAULT_SCORE_ROOT = REPO / (
    "outputs/stage2_adaptive_route_formal_nll_structproxy_router_seedpair23_24_20260519/"
    f"{DEFAULT_BRANCH}"
)
DEFAULT_SAMPLE_ROOT = REPO / (
    "outputs/stage2_modular_dualexpert/formal_k2_counterfactual_utility_20260518/"
    "sampled_reason_expert_forcedreason_from_noaux_20260517_checkpoint-258"
)
DEFAULT_REPORT_MD = REPO / "reports/2026-05-19_stage2_sampled_k2_structural_proxy_supervised_router_formal.md"
DEFAULT_REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_structural_proxy_supervised_router_formal.json"
CHECKPOINTS = ["checkpoint-25", "checkpoint-50", "checkpoint-75", "checkpoint-98"]
BUDGETS = [None, 0.03, 0.05, 0.076, 0.10, 0.15, 0.20]


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def avg_metrics(rows):
    if not rows:
        return {metric: 0.0 for metric in METRICS}
    return {metric: mean(row[metric] for row in rows) for metric in METRICS}


def prediction_key(row):
    return key_for(row)


def load_scores(score_root: Path, checkpoint: str, split: str):
    path = score_root / checkpoint / split / "scores.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = prediction_key(row)
            if key:
                rows.append((key, float(row.get("delta_direct_minus_reason_route_nll", float("-inf")))))
    return rows


def routes_from_scores(score_rows, budget):
    sorted_rows = sorted(score_rows, key=lambda item: (item[1], item[0]), reverse=True)
    if budget is None:
        reason_keys = {key for key, delta in sorted_rows if delta >= 0.0}
        policy = "argmin"
    else:
        cap = round(len(sorted_rows) * budget)
        reason_keys = {key for key, _delta in sorted_rows[:cap]}
        policy = f"top{int(budget * 1000):03d}"
    return {key: ("reason" if key in reason_keys else "direct") for key, _delta in score_rows}, policy


def load_cases(args):
    cases = {}
    for split in SPLITS:
        exec_rows = {route: load_exec_rows(split, route) for route in ["direct", "reason"]}
        sample_rows = {
            route: load_sample_rows(args.sample_root, split, route, args.seeds)
            for route in ["direct", "reason"]
        }
        keys = set(exec_rows["direct"]) & set(exec_rows["reason"]) & set(sample_rows["direct"]) & set(sample_rows["reason"])
        split_cases = {}
        for key in sorted(keys):
            split_cases[key] = {
                "single_gen_execution_direct": metric_dict(exec_rows["direct"][key]),
                "single_gen_execution_reason": metric_dict(exec_rows["reason"][key]),
                "k2_expected_direct": avg_metric_dict(sample_rows["direct"][key]),
                "k2_expected_reason": avg_metric_dict(sample_rows["reason"][key]),
            }
        cases[split] = split_cases
    return cases


def summarize_policy(cases, split, source, checkpoint, policy, routes):
    direct_rows = []
    routed_rows = []
    selected_gains = []
    selected_count = 0
    harmful_count = 0
    keys = sorted(set(cases) & set(routes))
    for key in keys:
        direct = cases[key][f"{source}_direct"]
        reason = cases[key][f"{source}_reason"]
        direct_rows.append(direct)
        gain = reason["score"] - direct["score"]
        if routes[key] == "reason":
            selected_count += 1
            selected_gains.append(gain)
            harmful_count += 1 if gain < 0 else 0
            routed_rows.append(reason)
        else:
            routed_rows.append(direct)
    direct_summary = avg_metrics(direct_rows)
    routed_summary = avg_metrics(routed_rows)
    return {
        "split": split,
        "source": source,
        "checkpoint": checkpoint,
        "policy": policy,
        "num_examples": len(keys),
        "pred_reason_count": selected_count,
        "pred_reason_rate": selected_count / len(keys) if keys else 0.0,
        "selected_reason_score_gain_mean": mean(selected_gains),
        "selected_reason_harm_rate": harmful_count / selected_count if selected_count else 0.0,
        "direct": direct_summary,
        "routed": routed_summary,
        "routed_minus_direct": {
            metric: routed_summary[metric] - direct_summary[metric]
            for metric in METRICS
        },
    }


def aggregate_test(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] in SPLITS:
            grouped[(row["source"], row["checkpoint"], row["policy"])].append(row)
    out = []
    for (source, checkpoint, policy), items in grouped.items():
        total = sum(row["num_examples"] for row in items)
        agg = {
            "split": "test",
            "source": source,
            "checkpoint": checkpoint,
            "policy": policy,
            "num_examples": total,
            "pred_reason_count": sum(row["pred_reason_count"] for row in items),
        }
        agg["pred_reason_rate"] = agg["pred_reason_count"] / total if total else 0.0
        denom = sum(row["pred_reason_count"] for row in items)
        agg["selected_reason_score_gain_mean"] = (
            sum(row["selected_reason_score_gain_mean"] * row["pred_reason_count"] for row in items) / denom
            if denom else 0.0
        )
        agg["selected_reason_harm_rate"] = (
            sum(row["selected_reason_harm_rate"] * row["pred_reason_count"] for row in items) / denom
            if denom else 0.0
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


def render_table(rows, source):
    lines = [
        "| checkpoint | policy | split | reason rate | delta A/E/T/Score | harm | selected gain |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    order = {"test": 0, "test_seen": 1, "test_unseen": 2}
    filtered = [row for row in rows if row["source"] == source]
    filtered.sort(key=lambda row: (row["checkpoint"], row["policy"], order.get(row["split"], 9)))
    for row in filtered:
        delta = row["routed_minus_direct"]
        lines.append(
            f"| `{row['checkpoint']}` | `{row['policy']}` | `{row['split']}` | {pct(row['pred_reason_rate'])} | "
            f"{signed(delta['argument_f1'])}/{signed(delta['event_f1'])}/{signed(delta['trigger_f1'])}/{signed(delta['score'])} | "
            f"{pct(row['selected_reason_harm_rate'])} | {signed(row['selected_reason_score_gain_mean'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    single = [row for row in payload["results"] if row["source"] == "single_gen_execution" and row["split"] == "test"]
    best = max(single, key=lambda row: (row["routed_minus_direct"]["score"], row["routed_minus_direct"]["event_f1"]))
    lines = [
        "# Sampled K2 Structural-Proxy Supervised Router Formal Summary",
        "",
        "This evaluates the trained structural-proxy route selector on formal seedpair23/24 compact-evidence prompts. It uses route-choice NLL ranking and offline forced direct/reason executions; no new extraction generation is required.",
        "",
        f"- branch: `{payload['branch']}`",
        f"- score root: `{payload['score_root']}`",
        f"- sample root: `{payload['sample_root']}`",
        f"- seeds: `{payload['seeds']}`",
        "",
        "## Single-Generation Execution",
        "",
        render_table([row for row in payload["results"] if row["policy"] in payload["focus_policies"]], "single_gen_execution"),
        "",
        "## K2 Expected",
        "",
        render_table([row for row in payload["results"] if row["policy"] in payload["focus_policies"]], "k2_expected"),
        "",
        "## Reading",
        "",
        f"- Best single-generation aggregate policy: `{best['checkpoint']}/{best['policy']}` with reason rate `{best['pred_reason_rate']:.1%}`, score delta `{best['routed_minus_direct']['score']:+.4f}`, harm `{best['selected_reason_harm_rate']:.1%}`.",
        f"- Report JSON: `{payload['report_json']}`",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    cases = load_cases(args)
    results = []
    for checkpoint in args.checkpoints:
        for split in SPLITS:
            score_rows = load_scores(args.score_root, checkpoint, split)
            for budget in args.budgets:
                routes, policy = routes_from_scores(score_rows, budget)
                for source in ["single_gen_execution", "k2_expected"]:
                    results.append(summarize_policy(cases[split], split, source, checkpoint, policy, routes))
    results.extend(aggregate_test(results))
    focus_policies = ["argmin", "top030", "top050", "top076", "top100", "top150", "top200"]
    payload = {
        "branch": args.branch,
        "score_root": args.score_root.as_posix(),
        "sample_root": args.sample_root.as_posix(),
        "seeds": args.seeds,
        "checkpoints": args.checkpoints,
        "focus_policies": focus_policies,
        "results": results,
        "report_json": args.report_json.as_posix(),
        "report_md": args.report_md.as_posix(),
    }
    write_json(args.report_json, payload)
    write_text(args.report_md, render_report(payload))
    print(json.dumps({"report_md": args.report_md.as_posix(), "report_json": args.report_json.as_posix()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--score_root", type=Path, default=DEFAULT_SCORE_ROOT)
    parser.add_argument("--sample_root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 24])
    parser.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    parser.add_argument("--budgets", type=float, nargs="*", default=BUDGETS)
    parser.add_argument("--report_md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report_json", type=Path, default=DEFAULT_REPORT_JSON)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
