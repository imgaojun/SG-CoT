#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_sampled_confident_router_dev_20260518 import fmt, pct, signed, write_json, write_text  # noqa: E402


BRANCH = "sampled_k2_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
SPLITS = ["test_seen", "test_unseen"]
SEEDS = [19, 20]
SAMPLE_ROOT = (
    REPO
    / "outputs/stage2_modular_dualexpert/formal_k2_counterfactual_utility_20260518"
    / "sampled_reason_expert_forcedreason_from_noaux_20260517_checkpoint-258"
)
NEW_NLL_ROOT = REPO / f"outputs/stage2_adaptive_route_formal_nll_seedpair19_20_20260518/{BRANCH}"
OLD_NLL_ROOT = REPO / f"outputs/stage2_adaptive_route_formal_nll_20260518/{BRANCH}"
EXEC_ROOT = REPO / "outputs/stage2_adaptive_route_formal_execution_20260518/sampledk2_ckpt50_margin025"
REPORT_MD = REPO / "reports/2026-05-18_stage2_sampled_k2_formal_seedpair19_20_robustness.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-18_stage2_sampled_k2_formal_seedpair19_20_robustness.json"
METRICS = ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json_rate"]
POLICIES = {
    "main_ckpt50_margin025": {
        "label": "main: checkpoint-50 margin >= 0.25",
        "thresholds": {"checkpoint-50": 0.25},
    },
    "guard_ckpt50_margin025_ckpt75_margin005": {
        "label": "guard: checkpoint-50 margin >= 0.25 AND checkpoint-75 margin >= 0.05",
        "thresholds": {"checkpoint-50": 0.25, "checkpoint-75": 0.05},
    },
}


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def key_for(row):
    meta = row.get("meta") or {}
    return row.get("sample_key") or meta.get("wnd_id") or row.get("wnd_id")


def metric_score(row):
    return row.get("argument_f1", 0.0) + row.get("event_f1", 0.0) + 0.25 * row.get("trigger_f1", 0.0)


def metric_dict(row):
    return {
        "argument_f1": row.get("argument_f1", 0.0),
        "event_f1": row.get("event_f1", 0.0),
        "trigger_f1": row.get("trigger_f1", 0.0),
        "score": metric_score(row),
        "valid_json_rate": 1.0 if row.get("valid_final_json", row.get("valid_json")) else 0.0,
    }


def mean_metric_dict(rows):
    if not rows:
        return {metric: 0.0 for metric in METRICS}
    return {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}


def load_sample_metrics(split: str, route: str):
    grouped = defaultdict(list)
    for seed in SEEDS:
        path = SAMPLE_ROOT / split / route / f"seed-{seed}" / "predictions.jsonl"
        for row in load_jsonl(path):
            key = key_for(row)
            if key:
                grouped[key].append(metric_dict(row))
    out = {}
    for key, rows in grouped.items():
        if len(rows) != len(SEEDS):
            raise ValueError(f"{split}/{route}/{key} expected {len(SEEDS)} samples, got {len(rows)}")
        out[key] = mean_metric_dict(rows)
        out[key]["sample_count"] = len(rows)
    return out


def load_execution_metrics(split: str, route: str):
    path = EXEC_ROOT / route / split / "predictions.jsonl"
    out = {}
    for row in load_jsonl(path):
        key = key_for(row)
        if key:
            out[key] = metric_dict(row)
    return out


def load_margins(root: Path, checkpoint: str, split: str):
    path = root / checkpoint / split / "scores.jsonl"
    margins = {}
    for row in load_jsonl(path):
        key = key_for(row)
        if key:
            margins[key] = row.get("delta_direct_minus_reason_route_nll")
    return margins


def route_reason(case, policy_name: str, root_name: str):
    thresholds = POLICIES[policy_name]["thresholds"]
    for checkpoint, threshold in thresholds.items():
        value = case[f"{root_name}_{checkpoint}_margin"]
        if value is None or value < threshold:
            return False
    return True


def route_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def case_id(case):
    return f"{case['split']}::{case['key']}"


def build_split_cases(split: str):
    k2_direct = load_sample_metrics(split, "direct")
    k2_reason = load_sample_metrics(split, "reason")
    exec_direct = load_execution_metrics(split, "forced_direct")
    exec_reason = load_execution_metrics(split, "forced_reason")
    margins = {
        ("new", "checkpoint-50"): load_margins(NEW_NLL_ROOT, "checkpoint-50", split),
        ("new", "checkpoint-75"): load_margins(NEW_NLL_ROOT, "checkpoint-75", split),
        ("old", "checkpoint-50"): load_margins(OLD_NLL_ROOT, "checkpoint-50", split),
        ("old", "checkpoint-75"): load_margins(OLD_NLL_ROOT, "checkpoint-75", split),
    }
    keys = set(k2_direct) & set(k2_reason) & set(exec_direct) & set(exec_reason)
    for margin_map in margins.values():
        keys &= set(margin_map)
    cases = []
    for key in sorted(keys):
        case = {
            "split": split,
            "key": key,
            "k2_expected_direct": k2_direct[key],
            "k2_expected_reason": k2_reason[key],
            "single_gen_execution_direct": exec_direct[key],
            "single_gen_execution_reason": exec_reason[key],
        }
        for (root_name, checkpoint), margin_map in margins.items():
            case[f"{root_name}_{checkpoint}_margin"] = margin_map[key]
        cases.append(case)
    if not cases:
        raise ValueError(f"no common cases for {split}")
    return cases


def avg_metrics(items):
    if not items:
        return {metric: 0.0 for metric in METRICS}
    return {metric: sum(item[metric] for item in items) / len(items) for metric in METRICS}


def summarize_cases(cases, source: str, policy_name: str):
    selected = []
    helpful = set()
    selected_set = set()
    direct_rows = []
    reason_rows = []
    routed_rows = []
    for case in cases:
        key = case_id(case)
        direct = case[f"{source}_direct"]
        reason = case[f"{source}_reason"]
        direct_rows.append(direct)
        reason_rows.append(reason)
        gain = reason["score"] - direct["score"]
        if gain > 0:
            helpful.add(key)
        if route_reason(case, policy_name, "new"):
            selected_set.add(key)
            selected.append(
                {
                    "split": case["split"],
                    "key": key,
                    "score_gain": gain,
                    "argument_gain": reason["argument_f1"] - direct["argument_f1"],
                    "event_gain": reason["event_f1"] - direct["event_f1"],
                    "trigger_gain": reason["trigger_f1"] - direct["trigger_f1"],
                    "checkpoint_50_margin": case["new_checkpoint-50_margin"],
                    "checkpoint_75_margin": case["new_checkpoint-75_margin"],
                }
            )
            routed_rows.append(reason)
        else:
            routed_rows.append(direct)

    n = len(cases)
    direct_summary = avg_metrics(direct_rows)
    reason_summary = avg_metrics(reason_rows)
    routed_summary = avg_metrics(routed_rows)
    tp = len(selected_set & helpful)
    fp = len(selected_set - helpful)
    fn = len(helpful - selected_set)
    harm_count = sum(1 for row in selected if row["score_gain"] < 0)
    return {
        "source": source,
        "policy": policy_name,
        "policy_label": POLICIES[policy_name]["label"],
        "num_examples": n,
        "pred_reason_count": len(selected_set),
        "pred_reason_rate": len(selected_set) / n,
        "reason_helpful_count": len(helpful),
        "reason_helpful_rate": len(helpful) / n,
        "route_vs_helpful": route_prf(tp, fp, fn),
        "selected_reason_score_gain_mean": (
            sum(row["score_gain"] for row in selected) / len(selected) if selected else 0.0
        ),
        "selected_reason_harm_count": harm_count,
        "selected_reason_harm_rate": harm_count / len(selected) if selected else 0.0,
        "direct": direct_summary,
        "reason_all": reason_summary,
        "routed": routed_summary,
        "routed_minus_direct": {
            metric: routed_summary[metric] - direct_summary[metric] for metric in METRICS
        },
        "routed_minus_reason_all": {
            metric: routed_summary[metric] - reason_summary[metric] for metric in METRICS
        },
        "selected_reason_best": sorted(selected, key=lambda row: row["score_gain"], reverse=True)[:10],
        "selected_reason_worst": sorted(selected, key=lambda row: row["score_gain"])[:10],
    }


def selected_keys(cases, policy_name: str, root_name: str):
    return {case_id(case) for case in cases if route_reason(case, policy_name, root_name)}


def summarize_overlap(cases, split: str, policy_name: str):
    old = selected_keys(cases, policy_name, "old")
    new = selected_keys(cases, policy_name, "new")
    inter = old & new
    union = old | new
    return {
        "split": split,
        "policy": policy_name,
        "policy_label": POLICIES[policy_name]["label"],
        "num_examples": len(cases),
        "old_seedpair17_18_reason_count": len(old),
        "new_seedpair19_20_reason_count": len(new),
        "intersection_count": len(inter),
        "jaccard": len(inter) / len(union) if union else 1.0,
        "old_retention_rate": len(inter) / len(old) if old else 1.0,
        "new_overlap_rate": len(inter) / len(new) if new else 1.0,
    }


def metric_cell(row):
    return f"{fmt(row['argument_f1'])}/{fmt(row['event_f1'])}/{fmt(row['trigger_f1'])}/{fmt(row['score'])}"


def delta_cell(row):
    return f"{signed(row['argument_f1'])}/{signed(row['event_f1'])}/{signed(row['trigger_f1'])}/{signed(row['score'])}"


def split_order(name):
    return {"test": 0, "test_seen": 1, "test_unseen": 2}.get(name, 99)


def render_result_table(rows, source):
    lines = [
        "| split | policy | reason rate | routed A/E/T/Score | delta vs Direct A/E/T/Score | selected gain | harm rate | P/R/F1 vs helpful |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    source_rows = [row for row in rows if row["source"] == source]
    source_rows.sort(key=lambda row: (split_order(row["split"]), row["policy"]))
    for row in source_rows:
        prf = row["route_vs_helpful"]
        lines.append(
            f"| `{row['split']}` | `{row['policy']}` | {pct(row['pred_reason_rate'])} | "
            f"{metric_cell(row['routed'])} | {delta_cell(row['routed_minus_direct'])} | "
            f"{signed(row['selected_reason_score_gain_mean'])} | {pct(row['selected_reason_harm_rate'])} | "
            f"{fmt(prf['precision'], 3)}/{fmt(prf['recall'], 3)}/{fmt(prf['f1'], 3)} |"
        )
    return "\n".join(lines)


def render_baseline_table(rows):
    lines = [
        "| source | split | Direct A/E/T/Score | Reason-all A/E/T/Score | Reason-all delta Score |",
        "|---|---|---:|---:|---:|",
    ]
    seen = set()
    for row in sorted(rows, key=lambda item: (item["source"], split_order(item["split"]))):
        key = (row["source"], row["split"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| `{row['source']}` | `{row['split']}` | {metric_cell(row['direct'])} | "
            f"{metric_cell(row['reason_all'])} | {signed(row['reason_all']['score'] - row['direct']['score'])} |"
        )
    return "\n".join(lines)


def render_overlap_table(rows):
    lines = [
        "| split | policy | 17/18 Reason | 19/20 Reason | intersection | Jaccard | old retained | new overlap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (split_order(item["split"]), item["policy"])):
        lines.append(
            f"| `{row['split']}` | `{row['policy']}` | {row['old_seedpair17_18_reason_count']} | "
            f"{row['new_seedpair19_20_reason_count']} | {row['intersection_count']} | "
            f"{fmt(row['jaccard'], 3)} | {pct(row['old_retention_rate'])} | {pct(row['new_overlap_rate'])} |"
        )
    return "\n".join(lines)


def reading_bullets(rows):
    lines = []
    for source in ["k2_expected", "single_gen_execution"]:
        main = next(row for row in rows if row["source"] == source and row["split"] == "test" and row["policy"] == "main_ckpt50_margin025")
        guard = next(row for row in rows if row["source"] == source and row["split"] == "test" and row["policy"] == "guard_ckpt50_margin025_ckpt75_margin005")
        lines.append(
            f"- `{source}` main policy: reason rate `{main['pred_reason_rate']:.1%}`, "
            f"score delta `{main['routed_minus_direct']['score']:+.4f}`, selected gain "
            f"`{main['selected_reason_score_gain_mean']:+.4f}`, harm `{main['selected_reason_harm_rate']:.1%}`."
        )
        lines.append(
            f"- `{source}` guard policy: reason rate `{guard['pred_reason_rate']:.1%}`, "
            f"score delta `{guard['routed_minus_direct']['score']:+.4f}`, selected gain "
            f"`{guard['selected_reason_score_gain_mean']:+.4f}`, harm `{guard['selected_reason_harm_rate']:.1%}`."
        )
    return lines


def render_report(payload):
    lines = [
        "# Sampled K2 Formal Seedpair 19/20 Robustness",
        "",
        "This report repeats the formal K2 compact-evidence route-NLL evaluation with independent sampled evidence seeds `19,20`. Thresholds are fixed from the previous experiment; no formal retuning is performed.",
        "",
        f"- sample root: `{payload['sample_root']}`",
        f"- new route-NLL root: `{payload['new_nll_root']}`",
        f"- old route-NLL root: `{payload['old_nll_root']}`",
        f"- reused deterministic execution root: `{payload['execution_root']}`",
        "",
        "## Policies",
        "",
        "- `main_ckpt50_margin025`: route Reason when `checkpoint-50` margin `>= 0.25`.",
        "- `guard_ckpt50_margin025_ckpt75_margin005`: route Reason only when `checkpoint-50` margin `>= 0.25` and `checkpoint-75` margin `>= 0.05`.",
        "",
        "## Baselines",
        "",
        render_baseline_table(payload["results"]),
        "",
        "## K2 Expected Utility",
        "",
        render_result_table(payload["results"], "k2_expected"),
        "",
        "## Reused Single-Generation Execution",
        "",
        render_result_table(payload["results"], "single_gen_execution"),
        "",
        "## Route Stability vs Seeds 17/18",
        "",
        render_overlap_table(payload["overlap"]),
        "",
        "## Reading",
        "",
        *reading_bullets(payload["results"]),
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['output_json']}`",
    ]
    return "\n".join(lines) + "\n"


def run():
    split_cases = {split: build_split_cases(split) for split in SPLITS}
    all_cases = [case for split in SPLITS for case in split_cases[split]]
    case_sets = {"test": all_cases, **split_cases}

    results = []
    overlap = []
    for split, cases in case_sets.items():
        for policy_name in POLICIES:
            overlap.append(summarize_overlap(cases, split, policy_name))
            for source in ["k2_expected", "single_gen_execution"]:
                row = summarize_cases(cases, source, policy_name)
                row["split"] = split
                results.append(row)

    payload = {
        "branch": BRANCH,
        "seed_pair": SEEDS,
        "splits": {split: len(cases) for split, cases in split_cases.items()},
        "sample_root": SAMPLE_ROOT.as_posix(),
        "new_nll_root": NEW_NLL_ROOT.as_posix(),
        "old_nll_root": OLD_NLL_ROOT.as_posix(),
        "execution_root": EXEC_ROOT.as_posix(),
        "policies": POLICIES,
        "results": results,
        "overlap": overlap,
        "output_json": REPORT_JSON.as_posix(),
        "output_md": REPORT_MD.as_posix(),
    }
    write_json(REPORT_JSON, payload)
    write_text(REPORT_MD, render_report(payload))
    print(json.dumps({"output_json": REPORT_JSON.as_posix(), "output_md": REPORT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    run()
