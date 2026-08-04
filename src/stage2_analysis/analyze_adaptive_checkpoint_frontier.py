import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())


BRANCHES = [
    {
        "name": "confrare10_heur10_plan_lite",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_plan_lite_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_plan_lite",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_plan_lite",
    },
    {
        "name": "confrare10_heur10_type_plan_lite",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_plan_lite_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_plan_lite",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_plan_lite",
    },
    {
        "name": "roleconf10_heur10_plan_lite",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_roleconf10_heur10_plan_lite_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_roleconf10_heur10_plan_lite",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_roleconf10_heur10_plan_lite",
    },
    {
        "name": "confrare10_heur10_plan_lite_pairdirect",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_plan_lite_pairdirect_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_plan_lite_pairdirect",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_plan_lite_pairdirect",
    },
    {
        "name": "confrare10_heur10_type_plan_v2",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_plan_v2_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_plan_v2",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_plan_v2",
    },
    {
        "name": "confrare10_heur10_type_role_hint_plan_lite",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_role_hint_plan_lite_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_role_hint_plan_lite",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_role_hint_plan_lite",
    },
    {
        "name": "confrare5_heur5_type_role_hint_plan_lite",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_confrare5_heur5_type_role_hint_plan_lite_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare5_heur5_type_role_hint_plan_lite",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare5_heur5_type_role_hint_plan_lite",
    },
    {
        "name": "confrare10_heur10_type_role_hint_plan_lite_pairdirect",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_role_hint_plan_lite_pairdirect_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_role_hint_plan_lite_pairdirect",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_role_hint_plan_lite_pairdirect",
    },
    {
        "name": "confrare10_heur10_type_role_hint_plan_lite_directanchor",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_role_hint_plan_lite_directanchor_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_role_hint_plan_lite_directanchor",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_role_hint_plan_lite_directanchor",
    },
    {
        "name": "hardconf10_heur10_type_role_hint_plan_lite",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_heur10_type_role_hint_plan_lite_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_heur10_type_role_hint_plan_lite",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_heur10_type_role_hint_plan_lite",
    },
    {
        "name": "hardconf15_heur15_type_role_hint_plan_lite",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_hardconf15_heur15_type_role_hint_plan_lite_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf15_heur15_type_role_hint_plan_lite",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_hardconf15_heur15_type_role_hint_plan_lite",
    },
    {
        "name": "hardconf10_calibrated_type_role_hint_plan_lite",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_calibrated_type_role_hint_plan_lite_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_calibrated_type_role_hint_plan_lite",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_calibrated_type_role_hint_plan_lite",
    },
    {
        "name": "hardconf10_directdup",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_directdup_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_hardconf10_directdup",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_directdup",
    },
    {
        "name": "qwen3_4b_hardconf10_heur10_type_role_hint_plan_lite",
        "run_slug": "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_heur10_type_role_hint_plan_lite_full",
        "formal_slug": "richere_split1_qwen3_4b_adaptive_hardconf10_heur10_type_role_hint_plan_lite",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_heur10_type_role_hint_plan_lite",
    },
    {
        "name": "qwen3_4b_hardconf10_directdup",
        "run_slug": "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_directdup_full",
        "formal_slug": "richere_split1_qwen3_4b_adaptive_hardconf10_directdup",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_directdup",
    },
    {
        "name": "llama3_2_3b_hardconf10_heur10_type_role_hint_plan_lite",
        "run_slug": "richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_heur10_type_role_hint_plan_lite_full",
        "formal_slug": "richere_split1_llama3_2_3b_adaptive_hardconf10_heur10_type_role_hint_plan_lite",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_heur10_type_role_hint_plan_lite",
    },
    {
        "name": "llama3_2_3b_hardconf10_directdup",
        "run_slug": "richere_split1_llama3_2_3b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_directdup_full",
        "formal_slug": "richere_split1_llama3_2_3b_adaptive_hardconf10_directdup",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_hardconf10_directdup",
    },
    {
        "name": "likelihood10_goldplan_type_role_hint_plan_lite_raw",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_raw_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_raw",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_raw",
    },
    {
        "name": "likelihood10_goldplan_type_role_hint_plan_lite_bal30",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_bal30_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_bal30",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_bal30",
    },
    {
        "name": "likelihood15_goldplan_type_role_hint_plan_lite_raw",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_raw_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_raw",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_raw",
    },
    {
        "name": "likelihood15_goldplan_type_role_hint_plan_lite_bal30",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30",
    },
    {
        "name": "likelihood10_goldplan_type_role_hint_plan_lite_pairdirect_bal30",
        "run_slug": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_pairdirect_bal30_full",
        "formal_slug": "richere_split1_qwen3_1_7b_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_pairdirect_bal30",
        "dataset_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_likelihood10_goldplan_type_role_hint_plan_lite_pairdirect_bal30",
    },
]

MODES = ["free_route", "forced_direct", "forced_reason"]
SPLITS = ["test", "test_seen", "test_unseen"]
PROTOCOLS = [
    "free_arg_best",
    "reason_expert_best",
    "adaptive_pareto",
    "direct_anchor_best",
    "adaptive_tradeoff_best",
    "seen_stable_best",
    "hard_reason_best",
    "balanced_hardroute_best",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def summary_path(devpick_root: Path, run_slug: str, mode: str):
    mode_slug = "free" if mode == "free_route" else mode
    return devpick_root / f"{run_slug}_{mode_slug}_dev_seen_max512" / "selection_summary.json"


def metric(row, mode, key, default=0.0):
    summary = row.get(mode, {}).get("summary") or {}
    return float(summary.get(key, default))


def checkpoint_path(row, mode):
    return row.get(mode, {}).get("checkpoint_path") or row.get(mode, {}).get("checkpoint_path")


def json_ok(row, mode, threshold=0.99):
    return metric(row, mode, "json_valid_rate", 0.0) >= threshold


def score_tuple(row, mode):
    return (
        metric(row, mode, "argument_f1"),
        metric(row, mode, "event_f1"),
        metric(row, mode, "trigger_f1"),
    )


def load_branch(devpick_root: Path, branch):
    mode_payloads = {}
    for mode in MODES:
        path = summary_path(devpick_root, branch["run_slug"], mode)
        if not path.exists():
            raise FileNotFoundError(f"missing {mode} summary for {branch['name']}: {path}")
        mode_payloads[mode] = load_json(path)

    rows = {}
    for mode, payload in mode_payloads.items():
        for candidate in payload.get("candidates", []):
            tag = candidate["checkpoint_tag"]
            rows.setdefault(tag, {"checkpoint_tag": tag})[mode] = candidate

    full_rows = []
    for tag, row in rows.items():
        if all(mode in row for mode in MODES):
            row["checkpoint_path"] = row["free_route"]["checkpoint_path"]
            row["run_dir"] = mode_payloads["free_route"]["run_dir"]
            full_rows.append(row)

    full_rows.sort(key=lambda x: int(x["checkpoint_tag"].split("-")[-1]) if "-" in x["checkpoint_tag"] else 10**9)
    return {
        "branch": branch,
        "mode_payloads": mode_payloads,
        "rows": full_rows,
    }


def choose_free_arg_best(rows):
    pool = [row for row in rows if json_ok(row, "free_route")]
    pool = pool or rows
    return max(pool, key=lambda row: score_tuple(row, "free_route"))


def choose_reason_expert_best(rows):
    pool = [row for row in rows if json_ok(row, "forced_reason") and json_ok(row, "forced_direct")]
    pool = pool or rows
    return max(
        pool,
        key=lambda row: (
            metric(row, "forced_reason", "argument_f1") - metric(row, "forced_direct", "argument_f1"),
            metric(row, "forced_reason", "event_f1") - metric(row, "forced_direct", "event_f1"),
            metric(row, "forced_reason", "argument_f1"),
            metric(row, "forced_reason", "event_f1"),
        ),
    )


def choose_adaptive_pareto(rows):
    pool = []
    for row in rows:
        rr = metric(row, "free_route", "route_reason_rate")
        free_arg = metric(row, "free_route", "argument_f1")
        direct_arg = metric(row, "forced_direct", "argument_f1")
        if json_ok(row, "free_route") and 0.05 <= rr <= 0.20 and free_arg >= direct_arg - 0.01:
            pool.append(row)
    if not pool:
        pool = [row for row in rows if json_ok(row, "free_route")]
    if not pool:
        pool = rows
    return max(
        pool,
        key=lambda row: (
            metric(row, "free_route", "argument_f1"),
            metric(row, "free_route", "event_f1"),
            metric(row, "forced_reason", "argument_f1") - metric(row, "forced_direct", "argument_f1"),
            -abs(metric(row, "free_route", "route_reason_rate") - 0.10),
        ),
    )


def choose_adaptive_tradeoff_best(rows):
    pool = [row for row in rows if json_ok(row, "free_route")]
    pool = pool or rows
    return max(
        pool,
        key=lambda row: (
            metric(row, "free_route", "argument_f1")
            + 0.50
            * max(
                metric(row, "forced_reason", "argument_f1")
                - metric(row, "forced_direct", "argument_f1"),
                0.0,
            )
            - 0.10 * abs(metric(row, "free_route", "route_reason_rate") - 0.15),
            metric(row, "free_route", "event_f1"),
            metric(row, "forced_reason", "event_f1") - metric(row, "forced_direct", "event_f1"),
        ),
    )


def choose_direct_anchor_best(rows):
    pool = [row for row in rows if json_ok(row, "forced_direct")]
    pool = pool or rows
    return max(pool, key=lambda row: score_tuple(row, "forced_direct"))


def choose_seen_stable_best(rows):
    pool = [
        row
        for row in rows
        if json_ok(row, "free_route") and metric(row, "free_route", "route_reason_rate") <= 0.20
    ]
    pool = pool or [row for row in rows if json_ok(row, "free_route")]
    pool = pool or rows
    return max(
        pool,
        key=lambda row: (
            metric(row, "free_route", "argument_f1"),
            metric(row, "free_route", "event_f1"),
            metric(row, "free_route", "trigger_f1"),
            -abs(metric(row, "free_route", "route_reason_rate") - 0.12),
        ),
    )


def choose_hard_reason_best(rows):
    pool = []
    for row in rows:
        reason_arg = metric(row, "forced_reason", "argument_f1")
        direct_arg = metric(row, "forced_direct", "argument_f1")
        if json_ok(row, "forced_reason") and reason_arg >= direct_arg - 0.01:
            pool.append(row)
    pool = pool or [row for row in rows if json_ok(row, "forced_reason") and json_ok(row, "forced_direct")]
    pool = pool or rows
    return max(
        pool,
        key=lambda row: (
            metric(row, "forced_reason", "argument_f1") - metric(row, "forced_direct", "argument_f1"),
            metric(row, "forced_reason", "event_f1") - metric(row, "forced_direct", "event_f1"),
            metric(row, "forced_reason", "argument_f1"),
            metric(row, "forced_reason", "event_f1"),
        ),
    )


def balanced_hardroute_score(row):
    reason_delta_arg = metric(row, "forced_reason", "argument_f1") - metric(row, "forced_direct", "argument_f1")
    reason_delta_event = metric(row, "forced_reason", "event_f1") - metric(row, "forced_direct", "event_f1")
    return (
        0.45 * metric(row, "free_route", "argument_f1")
        + 0.25 * metric(row, "free_route", "event_f1")
        + 0.15 * max(reason_delta_arg, 0.0)
        + 0.10 * max(reason_delta_event, 0.0)
        - 0.05 * abs(metric(row, "free_route", "route_reason_rate") - 0.12)
    )


def choose_balanced_hardroute_best(rows):
    pool = [row for row in rows if json_ok(row, "free_route")]
    pool = pool or rows
    return max(
        pool,
        key=lambda row: (
            balanced_hardroute_score(row),
            metric(row, "free_route", "argument_f1"),
            metric(row, "free_route", "event_f1"),
        ),
    )


def choose_protocol(rows, protocol):
    if protocol == "free_arg_best":
        return choose_free_arg_best(rows)
    if protocol == "reason_expert_best":
        return choose_reason_expert_best(rows)
    if protocol == "adaptive_pareto":
        return choose_adaptive_pareto(rows)
    if protocol == "adaptive_tradeoff_best":
        return choose_adaptive_tradeoff_best(rows)
    if protocol == "direct_anchor_best":
        return choose_direct_anchor_best(rows)
    if protocol == "seen_stable_best":
        return choose_seen_stable_best(rows)
    if protocol == "hard_reason_best":
        return choose_hard_reason_best(rows)
    if protocol == "balanced_hardroute_best":
        return choose_balanced_hardroute_best(rows)
    raise ValueError(protocol)


def row_record(row):
    return {
        "checkpoint_tag": row["checkpoint_tag"],
        "free_route": row["free_route"]["summary"],
        "forced_direct": row["forced_direct"]["summary"],
        "forced_reason": row["forced_reason"]["summary"],
        "reason_direct_delta": {
            "trigger_f1": metric(row, "forced_reason", "trigger_f1") - metric(row, "forced_direct", "trigger_f1"),
            "argument_f1": metric(row, "forced_reason", "argument_f1") - metric(row, "forced_direct", "argument_f1"),
            "event_f1": metric(row, "forced_reason", "event_f1") - metric(row, "forced_direct", "event_f1"),
        },
        "free_direct_delta": {
            "trigger_f1": metric(row, "free_route", "trigger_f1") - metric(row, "forced_direct", "trigger_f1"),
            "argument_f1": metric(row, "free_route", "argument_f1") - metric(row, "forced_direct", "argument_f1"),
            "event_f1": metric(row, "free_route", "event_f1") - metric(row, "forced_direct", "event_f1"),
        },
        "adaptive_tradeoff_score": (
            metric(row, "free_route", "argument_f1")
            + 0.50
            * max(
                metric(row, "forced_reason", "argument_f1")
                - metric(row, "forced_direct", "argument_f1"),
                0.0,
            )
            - 0.10 * abs(metric(row, "free_route", "route_reason_rate") - 0.15)
        ),
        "balanced_hardroute_score": balanced_hardroute_score(row),
    }


def protocol_selection_summary(branch_data, protocol, row, output_dir: Path):
    branch = branch_data["branch"]
    free_payload = branch_data["mode_payloads"]["free_route"]
    selected = row["free_route"]
    payload = {
        "run_dir": free_payload["run_dir"],
        "eval_jsonl": free_payload["eval_jsonl"],
        "metric_keys": ["argument_f1", "event_f1", "trigger_f1"],
        "greater_is_better": True,
        "batch_size": free_payload.get("batch_size", 1),
        "checkpoint_tags": free_payload["checkpoint_tags"],
        "best": selected,
        "candidates": free_payload["candidates"],
        "selection_protocol": f"adaptive_checkpoint_frontier/{protocol}",
        "frontier_protocol": protocol,
        "frontier_branch": branch["name"],
        "frontier_selected_checkpoint": row_record(row),
    }
    path = output_dir / f"{branch['name']}__{protocol}" / "selection_summary.json"
    write_json(path, payload)
    return path


def eval_spec(dataset_prefix: str, mode: str, split: str, output_base: Path):
    mode_prefix = "" if mode == "free_route" else f"_{mode}"
    return {
        "name": f"{mode}/{split}",
        "eval_jsonl": f"data/stage2_adaptive_datasets/{dataset_prefix}{mode_prefix}_{split}_pos.jsonl",
        "output_dir": (output_base / mode / split).as_posix(),
    }


def build_formal_manifest(base_model, selections, output_root: Path):
    runs = []
    seen = set()
    for item in selections:
        branch = item["branch"]
        protocol = item["protocol"]
        selection_path = item["selection_path"]
        key = (branch["name"], protocol)
        if key in seen:
            continue
        seen.add(key)
        output_base = output_root / branch["formal_slug"] / f"frontier_{protocol}"
        runs.append(
            {
                "tag": f"{branch['name']}__{protocol}",
                "selection_summary": selection_path.as_posix(),
                "evals": [
                    eval_spec(branch["dataset_prefix"], mode, split, output_base)
                    for mode in MODES
                    for split in SPLITS
                ],
            }
        )
    return {"base_model": base_model, "runs": runs}


def build_selected_formal_manifest(base_model, selections, output_root: Path, selected_protocols):
    selected = [item for item in selections if item["protocol"] in selected_protocols]
    return build_formal_manifest(base_model, selected, output_root)


def markdown_report(payload):
    lines = ["# Adaptive Checkpoint Frontier Analysis", ""]
    lines.append("## Protocol Selections")
    lines.append("")
    lines.append(
        "| branch | protocol | checkpoint | free json | free reason rate | free arg | direct arg | reason arg | reason-direct arg delta | free-direct arg delta | tradeoff score | hard-route score |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for branch in payload["branches"]:
        for protocol in PROTOCOLS:
            row = branch["protocols"][protocol]
            lines.append(
                "| `{branch}` | `{protocol}` | `{ckpt}` | {free_json:.4f} | {rr:.4f} | {free_arg:.4f} | {direct_arg:.4f} | {reason_arg:.4f} | {rd_delta:.4f} | {fd_delta:.4f} | {tradeoff:.4f} | {hardroute:.4f} |".format(
                    branch=branch["name"],
                    protocol=protocol,
                    ckpt=row["checkpoint_tag"],
                    free_json=row["free_route"]["json_valid_rate"],
                    rr=row["free_route"].get("route_reason_rate", 0.0),
                    free_arg=row["free_route"]["argument_f1"],
                    direct_arg=row["forced_direct"]["argument_f1"],
                    reason_arg=row["forced_reason"]["argument_f1"],
                    rd_delta=row["reason_direct_delta"]["argument_f1"],
                    fd_delta=row["free_direct_delta"]["argument_f1"],
                    tradeoff=row["adaptive_tradeoff_score"],
                    hardroute=row["balanced_hardroute_score"],
                )
            )
    lines.append("")
    lines.append("## Gate Reading")
    lines.append("")
    for branch in payload["branches"]:
        reason = branch["protocols"]["reason_expert_best"]
        tradeoff = branch["protocols"]["adaptive_tradeoff_best"]
        hardroute = branch["protocols"]["balanced_hardroute_best"]
        lines.append(
            "- `{}`: reason-expert checkpoint `{}` has forced reason-direct argument delta `{:.4f}`; adaptive-tradeoff checkpoint `{}` has free-direct argument delta `{:.4f}` with reason rate `{:.4f}`; balanced hard-route checkpoint `{}` has score `{:.4f}`.".format(
                branch["name"],
                reason["checkpoint_tag"],
                reason["reason_direct_delta"]["argument_f1"],
                tradeoff["checkpoint_tag"],
                tradeoff["free_direct_delta"]["argument_f1"],
                tradeoff["free_route"].get("route_reason_rate", 0.0),
                hardroute["checkpoint_tag"],
                hardroute["balanced_hardroute_score"],
            )
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- formal manifest: `{payload['formal_manifest']}`")
    lines.append(f"- protocol selections: `{payload['protocol_selection_dir']}`")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devpick_root", default="outputs/stage2_adaptive_runs_user_devpick_frontier")
    parser.add_argument("--existing_free_root", default="outputs/stage2_adaptive_runs_user_devpick")
    parser.add_argument("--protocol_selection_dir", default="outputs/stage2_adaptive_runs_user_devpick_frontier/protocol_selections")
    parser.add_argument("--formal_output_root", default="outputs/stage2_adaptive_runs_user_formal_clean")
    parser.add_argument("--formal_manifest", default="configs/generated/stage2_adaptive/richere_qwen3_1_7b_adaptive_checkpoint_frontier_formal_manifest.json")
    parser.add_argument("--selected_formal_manifest", default=None)
    parser.add_argument(
        "--selected_protocols",
        nargs="+",
        default=["direct_anchor_best", "reason_expert_best", "adaptive_tradeoff_best"],
    )
    parser.add_argument("--output_md", default="reports/2026-05-09_stage2_adaptive_checkpoint_frontier_analysis.md")
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-09_stage2_adaptive_checkpoint_frontier_analysis.json")
    parser.add_argument("--base_model", default="/workspace/models/LLM-Research/Qwen3-1.7B")
    parser.add_argument(
        "--branch_names",
        nargs="+",
        default=None,
        help="Optional subset of branch names to analyze. Defaults to the original plan-lite branches.",
    )
    args = parser.parse_args()

    devpick_root = Path(args.devpick_root)
    existing_free_root = Path(args.existing_free_root)
    protocol_selection_dir = Path(args.protocol_selection_dir)

    # The current free-route sweeps were launched before the frontier root existed.
    # Symlinks/copies are not required; read free summaries from the existing root
    # and forced-mode summaries from the frontier root.
    merged_root = devpick_root
    payload = {
        "branches": [],
        "formal_manifest": args.formal_manifest,
        "protocol_selection_dir": args.protocol_selection_dir,
    }
    selections = []
    default_branch_names = {
        "confrare10_heur10_plan_lite",
        "confrare10_heur10_type_plan_lite",
        "roleconf10_heur10_plan_lite",
        "confrare10_heur10_plan_lite_pairdirect",
    }
    requested = set(args.branch_names) if args.branch_names else default_branch_names
    branches = [branch for branch in BRANCHES if branch["name"] in requested]
    missing_branch_names = sorted(requested - {branch["name"] for branch in branches})
    if missing_branch_names:
        raise ValueError(f"unknown branch_names: {missing_branch_names}")

    for branch in branches:
        # Load free summary from existing root, forced summaries from frontier root.
        free_path = summary_path(existing_free_root, branch["run_slug"], "free_route")
        forced_direct_path = summary_path(merged_root, branch["run_slug"], "forced_direct")
        forced_reason_path = summary_path(merged_root, branch["run_slug"], "forced_reason")
        if not free_path.exists() or not forced_direct_path.exists() or not forced_reason_path.exists():
            missing = [p.as_posix() for p in [free_path, forced_direct_path, forced_reason_path] if not p.exists()]
            raise FileNotFoundError("missing frontier summaries:\n" + "\n".join(missing))

        temp_root = protocol_selection_dir / "_merged_inputs"
        temp_root.mkdir(parents=True, exist_ok=True)
        # Build branch data without mutating source summaries.
        mode_payloads = {
            "free_route": load_json(free_path),
            "forced_direct": load_json(forced_direct_path),
            "forced_reason": load_json(forced_reason_path),
        }
        rows = {}
        for mode, mode_payload in mode_payloads.items():
            for candidate in mode_payload.get("candidates", []):
                tag = candidate["checkpoint_tag"]
                rows.setdefault(tag, {"checkpoint_tag": tag})[mode] = candidate
        full_rows = []
        for tag, row in rows.items():
            if all(mode in row for mode in MODES):
                row["checkpoint_path"] = row["free_route"]["checkpoint_path"]
                row["run_dir"] = mode_payloads["free_route"]["run_dir"]
                full_rows.append(row)
        full_rows.sort(key=lambda x: int(x["checkpoint_tag"].split("-")[-1]))
        branch_data = {"branch": branch, "mode_payloads": mode_payloads, "rows": full_rows}

        branch_payload = {"name": branch["name"], "rows": [row_record(row) for row in full_rows], "protocols": {}}
        for protocol in PROTOCOLS:
            row = choose_protocol(full_rows, protocol)
            record = row_record(row)
            branch_payload["protocols"][protocol] = record
            selection_path = protocol_selection_summary(branch_data, protocol, row, protocol_selection_dir)
            selections.append({"branch": branch, "protocol": protocol, "selection_path": selection_path})
        payload["branches"].append(branch_payload)

    manifest = build_formal_manifest(args.base_model, selections, Path(args.formal_output_root))
    write_json(Path(args.formal_manifest), manifest)
    selected_formal_manifest = None
    if args.selected_formal_manifest:
        selected_manifest = build_selected_formal_manifest(
            args.base_model,
            selections,
            Path(args.formal_output_root),
            set(args.selected_protocols),
        )
        write_json(Path(args.selected_formal_manifest), selected_manifest)
        selected_formal_manifest = args.selected_formal_manifest
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), markdown_report(payload))
    print(
        json.dumps(
            {
                "output_md": args.output_md,
                "output_json": args.output_json,
                "formal_manifest": args.formal_manifest,
                "selected_formal_manifest": selected_formal_manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
