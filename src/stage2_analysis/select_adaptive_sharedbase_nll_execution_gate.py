import argparse
import json
import sys
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from src.stage2_analysis.analyze_adaptive_outcome_router_execution import (  # noqa: E402
    analyze_score_router,
    load_prediction_map,
    write_json,
    write_text,
)


RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
BUDGETS = [None, 0.05, 0.10, 0.15, 0.20, 0.30]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_key(tag: str):
    if tag.startswith("checkpoint-"):
        return int(tag.split("-", 1)[1])
    return 10**12


def paths_for(branch: str):
    return {
        "free": REPO / f"outputs/stage2_adaptive_runs_user_devpick/{RUN_PREFIX}_{branch}_full_free_dev_seen_max512",
        "direct": REPO / f"outputs/stage2_adaptive_runs_user_devpick_frontier/{RUN_PREFIX}_{branch}_full_forced_direct_dev_seen_max512",
        "reason": REPO / f"outputs/stage2_adaptive_runs_user_devpick_frontier/{RUN_PREFIX}_{branch}_full_forced_reason_dev_seen_max512",
        "route": REPO / f"outputs/stage2_adaptive_runs_user_devpick_route/{RUN_PREFIX}_{branch}_full_route_dev_seen_max16",
        "score": REPO / f"outputs/stage2_adaptive_route_likelihood_probe/outcome_helpful_sharedbase_20260515/{branch}",
        "run_dir": REPO / f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full",
    }


def summary(root: Path, tag: str):
    return load_json(root / tag / "summary.json")


def metric(payload, key: str, default=0.0):
    return payload.get(key, payload.get(f"final_{key}", default))


def summary_score(payload):
    return (
        metric(payload, "argument_f1")
        + metric(payload, "event_f1")
        + 0.25 * metric(payload, "trigger_f1")
    )


def json_ok(row):
    return all(
        metric(row[f"{mode}_summary"], "json_valid_rate") >= 0.99
        for mode in ["free", "forced_direct", "forced_reason"]
    )


def budget_label(budget):
    if budget is None:
        return "argmin"
    return f"top{int(budget * 100):02d}"


def pass_gate(row, route_result):
    if not json_ok(row):
        return False
    if metric(row["forced_direct_summary"], "argument_f1") < 0.30:
        return False
    if summary_score(row["forced_reason_summary"]) <= summary_score(row["forced_direct_summary"]):
        return False
    delta = route_result["routed_delta_vs_direct"]
    return delta["argument_f1"] >= 0.010 or (
        delta["event_f1"] >= 0.005 and delta["argument_f1"] >= -0.003
    )


def route_result_score(row, route_result):
    delta = route_result["routed_delta_vs_direct"]
    routed = route_result["routed"]
    helpful = route_result["route_vs_positive_reason_helpful"]
    return (
        int(route_result["gate_pass"]),
        int(json_ok(row)),
        routed["argument_f1"] + routed["event_f1"] + 0.25 * routed["trigger_f1"],
        delta["argument_f1"],
        delta["event_f1"],
        helpful["f1"],
        metric(row["forced_direct_summary"], "argument_f1"),
        -abs(route_result["pred_reason_rate"] - 0.20),
    )


def discover_tags(paths):
    status_path = paths["direct"] / "status.json"
    if status_path.exists():
        payload = load_json(status_path)
        tags = [
            row.get("checkpoint_tag")
            for row in payload.get("completed", []) + payload.get("results", [])
            if row.get("checkpoint_tag")
        ]
        if tags:
            return sorted(set(tags), key=checkpoint_key)
    tags = [path.name for path in paths["direct"].glob("checkpoint-*") if path.is_dir()]
    return sorted(tags, key=checkpoint_key)


def analyze_checkpoint(branch: str, tag: str, paths):
    direct_pred = paths["direct"] / tag / "predictions.jsonl"
    reason_pred = paths["reason"] / tag / "predictions.jsonl"
    score_path = paths["score"] / tag / "dev_seen_scores.jsonl"
    if not direct_pred.exists() or not reason_pred.exists() or not score_path.exists():
        missing = [
            path.as_posix()
            for path in [direct_pred, reason_pred, score_path]
            if not path.exists()
        ]
        raise FileNotFoundError(f"missing inputs for {branch} {tag}: {missing}")

    direct_rows = load_prediction_map(direct_pred)
    reason_rows = load_prediction_map(reason_pred)
    row = {
        "branch": branch,
        "checkpoint_tag": tag,
        "free_summary": summary(paths["free"], tag),
        "forced_direct_summary": summary(paths["direct"], tag),
        "forced_reason_summary": summary(paths["reason"], tag),
        "route_summary": summary(paths["route"], tag),
        "nll_routes": [],
    }
    for budget in BUDGETS:
        result = analyze_score_router(
            f"{branch}_{tag}_{budget_label(budget)}",
            score_path,
            budget,
            direct_rows,
            reason_rows,
        )
        result["budget"] = budget
        result["budget_label"] = budget_label(budget)
        result["gate_pass"] = pass_gate(row, result)
        row["nll_routes"].append(result)
    row["best_nll_route"] = max(row["nll_routes"], key=lambda result: route_result_score(row, result))
    row["gate_pass"] = row["best_nll_route"]["gate_pass"]
    return row


def selection_score(row):
    return route_result_score(row, row["best_nll_route"])


def make_selection_summary(branch: str, rows, best, output_path: Path):
    selected_record = {
        "branch": branch,
        "checkpoint_tag": best["checkpoint_tag"],
        "checkpoint_path": (paths_for(branch)["run_dir"] / best["checkpoint_tag"]).as_posix(),
        "budget": best["best_nll_route"]["budget"],
        "budget_label": best["best_nll_route"]["budget_label"],
        "gate_pass": best["gate_pass"],
        "free_summary": best["free_summary"],
        "forced_direct_summary": best["forced_direct_summary"],
        "forced_reason_summary": best["forced_reason_summary"],
        "route_summary": best["route_summary"],
        "nll_route": best["best_nll_route"],
    }
    payload = {
        "run_dir": paths_for(branch)["run_dir"].as_posix(),
        "eval_jsonl": f"data/stage2_adaptive_datasets/{DATA_PREFIX}_{branch}_dev_seen_pos.jsonl",
        "metric_keys": ["sharedbase_nll_execution_gate_score"],
        "greater_is_better": True,
        "checkpoint_tags": [row["checkpoint_tag"] for row in rows],
        "best": {
            "checkpoint_tag": best["checkpoint_tag"],
            "checkpoint_path": (paths_for(branch)["run_dir"] / best["checkpoint_tag"]).as_posix(),
            "metric_keys": ["sharedbase_nll_execution_gate_score"],
            "score_tuple": selection_score(best),
            "summary": best["free_summary"],
        },
        "candidates": [
            {
                "checkpoint_tag": row["checkpoint_tag"],
                "checkpoint_path": (paths_for(branch)["run_dir"] / row["checkpoint_tag"]).as_posix(),
                "metric_keys": ["sharedbase_nll_execution_gate_score"],
                "score_tuple": selection_score(row),
                "summary": row["free_summary"],
                "best_nll_budget": row["best_nll_route"]["budget_label"],
                "gate_pass": row["gate_pass"],
            }
            for row in rows
        ],
        "selection_protocol": "adaptive_sharedbase_nll_execution_gate",
        "frontier_protocol": "sharedbase_nll_execution_gate",
        "frontier_branch": branch,
        "frontier_selected_checkpoint": selected_record,
    }
    write_json(output_path, payload)


def render_markdown(branch: str, rows, best):
    lines = [
        f"# Adaptive Shared-Base NLL Execution Gate: {branch}",
        "",
        "| checkpoint | best budget | gate | json free/direct/reason | direct arg/event | reason arg/event | routed arg/event | delta arg/event | pred reason | helpful P/R/F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        free = row["free_summary"]
        direct = row["forced_direct_summary"]
        reason = row["forced_reason_summary"]
        route = row["best_nll_route"]
        routed = route["routed"]
        delta = route["routed_delta_vs_direct"]
        helpful = route["route_vs_positive_reason_helpful"]
        lines.append(
            "| `{}` | {} | {} | {:.4f}/{:.4f}/{:.4f} | {:.4f}/{:.4f} | {:.4f}/{:.4f} | {:.4f}/{:.4f} | {:+.4f}/{:+.4f} | {:.1%} | {:.3f}/{:.3f}/{:.3f} |".format(
                row["checkpoint_tag"],
                route["budget_label"],
                "pass" if row["gate_pass"] else "fail",
                metric(free, "json_valid_rate"),
                metric(direct, "json_valid_rate"),
                metric(reason, "json_valid_rate"),
                metric(direct, "argument_f1"),
                metric(direct, "event_f1"),
                metric(reason, "argument_f1"),
                metric(reason, "event_f1"),
                routed["argument_f1"],
                routed["event_f1"],
                delta["argument_f1"],
                delta["event_f1"],
                route["pred_reason_rate"],
                helpful["precision"],
                helpful["recall"],
                helpful["f1"],
            )
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- selected checkpoint: `{best['checkpoint_tag']}`",
            f"- selected NLL budget: `{best['best_nll_route']['budget_label']}`",
            f"- gate pass: `{best['gate_pass']}`",
            f"- selection summary: `outputs/stage2_adaptive_runs_user_devpick_frontier/protocol_selections/{branch}__sharedbase_nll_execution_gate/selection_summary.json`",
            "",
            "## Gate",
            "",
            "- JSON validity must be at least `0.99` for free, forced-direct, and forced-reason dev outputs.",
            "- forced-direct dev argument F1 must be at least `0.30`.",
            "- forced-reason must have positive combined-score headroom over forced-direct.",
            "- some NLL budget must have argument delta `>= +0.010`, or event delta `>= +0.005` with argument delta `>= -0.003`.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--selection_summary", default=None)
    args = parser.parse_args()

    p = paths_for(args.branch)
    tags = discover_tags(p)
    if not tags:
        raise ValueError(f"No devpick checkpoints found for branch {args.branch}")

    rows = [analyze_checkpoint(args.branch, tag, p) for tag in tags]
    best = max(rows, key=selection_score)
    payload = {"branch": args.branch, "selected": best, "rows": rows}

    output_json = Path(args.output_json) if args.output_json else REPO / f"reports/artifacts/2026-05-15_stage2_adaptive_{args.branch}_sharedbase_nll_execution_gate.json"
    output_md = Path(args.output_md) if args.output_md else REPO / f"reports/2026-05-15_stage2_adaptive_{args.branch}_sharedbase_nll_execution_gate.md"
    selection_summary = (
        Path(args.selection_summary)
        if args.selection_summary
        else REPO / f"outputs/stage2_adaptive_runs_user_devpick_frontier/protocol_selections/{args.branch}__sharedbase_nll_execution_gate/selection_summary.json"
    )

    write_json(output_json, payload)
    write_text(output_md, render_markdown(args.branch, rows, best))
    make_selection_summary(args.branch, rows, best, selection_summary)
    print(
        json.dumps(
            {
                "branch": args.branch,
                "selected_checkpoint": best["checkpoint_tag"],
                "selected_budget": best["best_nll_route"]["budget_label"],
                "gate_pass": best["gate_pass"],
                "output_json": output_json.as_posix(),
                "output_md": output_md.as_posix(),
                "selection_summary": selection_summary.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
