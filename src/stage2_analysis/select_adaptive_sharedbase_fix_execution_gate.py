import argparse
import json
import sys
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from src.stage2_analysis.analyze_adaptive_outcome_router_execution import (  # noqa: E402
    analyze_router,
    load_prediction_map,
    write_json,
    write_text,
)


RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"


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
        "run_dir": REPO / f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full",
    }


def summary(root: Path, tag: str):
    return load_json(root / tag / "summary.json")


def metric(payload, key: str, default=0.0):
    return payload.get(key, payload.get(f"final_{key}", default))


def simulate(tag: str, branch: str, direct_root: Path, reason_root: Path, route_root: Path):
    direct_rows = load_prediction_map(direct_root / tag / "predictions.jsonl")
    reason_rows = load_prediction_map(reason_root / tag / "predictions.jsonl")
    return analyze_router(
        f"{branch}_{tag}",
        route_root / tag / "predictions.jsonl",
        direct_rows,
        reason_rows,
    )


def branch_kind(branch: str):
    if "routeauxclf" in branch:
        return "routeauxclf"
    if "noaux" in branch:
        return "noaux"
    return "unknown"


def json_ok(row):
    return all(
        metric(row[f"{mode}_summary"], "json_valid_rate") >= 0.99
        for mode in ["free", "forced_direct", "forced_reason"]
    )


def pass_gate(row):
    kind = branch_kind(row["branch"])
    free = row["free_summary"]
    direct = row["forced_direct_summary"]
    route = row["route_summary"]
    delta = row["execution"]["routed_delta_vs_direct"]
    if not json_ok(row):
        return False
    if kind == "noaux":
        return metric(direct, "argument_f1") >= 0.30 and metric(free, "route_reason_rate") > 0.0
    if kind == "routeauxclf":
        rr = metric(free, "route_reason_rate")
        route_f1 = route.get("reason_f1", 0.0)
        gain_ok = delta["argument_f1"] >= 0.005 or (
            delta["event_f1"] >= 0.005 and delta["argument_f1"] >= -0.003
        )
        return 0.05 <= rr <= 0.20 and route_f1 >= 0.45 and gain_ok
    return False


def selection_score(row):
    kind = branch_kind(row["branch"])
    free = row["free_summary"]
    direct = row["forced_direct_summary"]
    route = row["route_summary"]
    routed = row["execution"]["routed"]
    delta = row["execution"]["routed_delta_vs_direct"]
    if kind == "noaux":
        return (
            int(row["gate_pass"]),
            int(json_ok(row)),
            metric(direct, "argument_f1"),
            metric(free, "argument_f1") + metric(free, "event_f1"),
            metric(free, "route_reason_rate"),
        )
    return (
        int(row["gate_pass"]),
        int(json_ok(row)),
        routed["argument_f1"] + routed["event_f1"] + 0.25 * routed["trigger_f1"],
        route.get("reason_f1", 0.0),
        delta["argument_f1"],
        delta["event_f1"],
        -abs(metric(free, "route_reason_rate") - 0.15),
    )


def make_selection_summary(branch: str, best, output_path: Path):
    selected_record = {key: value for key, value in best.items() if key != "all_rows"}
    payload = {
        "run_dir": paths_for(branch)["run_dir"].as_posix(),
        "eval_jsonl": f"data/stage2_adaptive_datasets/{DATA_PREFIX}_{branch}_dev_seen_pos.jsonl",
        "metric_keys": ["sharedbase_fix_execution_gate_score"],
        "greater_is_better": True,
        "batch_size": 2,
        "checkpoint_tags": [row["checkpoint_tag"] for row in best["all_rows"]],
        "best": {
            "checkpoint_tag": best["checkpoint_tag"],
            "checkpoint_path": (paths_for(branch)["run_dir"] / best["checkpoint_tag"]).as_posix(),
            "eval_dir": (paths_for(branch)["free"] / best["checkpoint_tag"]).as_posix(),
            "metric_keys": ["sharedbase_fix_execution_gate_score"],
            "score_tuple": selection_score(best),
            "summary": best["free_summary"],
        },
        "candidates": [
            {
                "checkpoint_tag": row["checkpoint_tag"],
                "checkpoint_path": (paths_for(branch)["run_dir"] / row["checkpoint_tag"]).as_posix(),
                "eval_dir": (paths_for(branch)["free"] / row["checkpoint_tag"]).as_posix(),
                "metric_keys": ["sharedbase_fix_execution_gate_score"],
                "score_tuple": selection_score(row),
                "summary": row["free_summary"],
            }
            for row in best["all_rows"]
        ],
        "selection_protocol": "adaptive_sharedbase_fix_execution_gate",
        "frontier_protocol": "sharedbase_fix_execution_gate",
        "frontier_branch": branch,
        "frontier_selected_checkpoint": selected_record,
    }
    write_json(output_path, payload)


def render_markdown(branch: str, rows, best):
    lines = [
        f"# Adaptive Shared-Base Fix Execution Gate: {branch}",
        "",
        "| checkpoint | gate | json free/direct/reason | reason rate | route P/R/F1 | free arg/event | forced direct arg/event | routed arg/event | delta arg/event |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        free = row["free_summary"]
        direct = row["forced_direct_summary"]
        reason = row["forced_reason_summary"]
        route = row["route_summary"]
        routed = row["execution"]["routed"]
        delta = row["execution"]["routed_delta_vs_direct"]
        lines.append(
            "| `{}` | {} | {:.4f}/{:.4f}/{:.4f} | {:.4f} | {:.3f}/{:.3f}/{:.3f} | {:.4f}/{:.4f} | {:.4f}/{:.4f} | {:.4f}/{:.4f} | {:+.4f}/{:+.4f} |".format(
                row["checkpoint_tag"],
                "pass" if row["gate_pass"] else "fail",
                metric(free, "json_valid_rate"),
                metric(direct, "json_valid_rate"),
                metric(reason, "json_valid_rate"),
                metric(free, "route_reason_rate"),
                route.get("reason_precision", 0.0),
                route.get("reason_recall", 0.0),
                route.get("reason_f1", 0.0),
                metric(free, "argument_f1"),
                metric(free, "event_f1"),
                metric(direct, "argument_f1"),
                metric(direct, "event_f1"),
                routed["argument_f1"],
                routed["event_f1"],
                delta["argument_f1"],
                delta["event_f1"],
            )
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- selected checkpoint: `{best['checkpoint_tag']}`",
            f"- gate pass: `{best['gate_pass']}`",
            f"- selection summary: `outputs/stage2_adaptive_runs_user_devpick_frontier/protocol_selections/{branch}__sharedbase_fix_execution_gate/selection_summary.json`",
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
    route_selection = load_json(p["route"] / "selection_summary.json")
    tags = sorted([row["checkpoint_tag"] for row in route_selection.get("candidates", [])], key=checkpoint_key)
    if not tags:
        raise ValueError(f"No route candidates found for branch {args.branch}")

    rows = []
    for tag in tags:
        row = {
            "branch": args.branch,
            "checkpoint_tag": tag,
            "free_summary": summary(p["free"], tag),
            "forced_direct_summary": summary(p["direct"], tag),
            "forced_reason_summary": summary(p["reason"], tag),
            "route_summary": summary(p["route"], tag),
            "execution": simulate(tag, args.branch, p["direct"], p["reason"], p["route"]),
        }
        row["gate_pass"] = pass_gate(row)
        rows.append(row)

    best_row = max(rows, key=selection_score)
    best = {key: value for key, value in best_row.items()}
    best["all_rows"] = rows
    payload = {"branch": args.branch, "selected": {key: value for key, value in best.items() if key != "all_rows"}, "rows": rows}

    output_json = Path(args.output_json) if args.output_json else REPO / f"reports/artifacts/2026-05-14_stage2_adaptive_{args.branch}_sharedbase_fix_execution_gate.json"
    output_md = Path(args.output_md) if args.output_md else REPO / f"reports/2026-05-14_stage2_adaptive_{args.branch}_sharedbase_fix_execution_gate.md"
    selection_summary = (
        Path(args.selection_summary)
        if args.selection_summary
        else REPO / f"outputs/stage2_adaptive_runs_user_devpick_frontier/protocol_selections/{args.branch}__sharedbase_fix_execution_gate/selection_summary.json"
    )

    write_json(output_json, payload)
    write_text(output_md, render_markdown(args.branch, rows, best))
    make_selection_summary(args.branch, best, selection_summary)
    print(
        json.dumps(
            {
                "branch": args.branch,
                "selected_checkpoint": best["checkpoint_tag"],
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
