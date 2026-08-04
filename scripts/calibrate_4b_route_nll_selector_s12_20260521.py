import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.evaluate_4b_transfer_m06_policy_replay_20260521 import (  # noqa: E402
    M02_FORMAL_ROOT,
    M02_WINDOW,
    M05_FORMAL_ROOT,
    M05_WINDOW,
    selected_by_window,
)
from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    load_prediction_map,
    load_score_rows,
    row_metric,
    score,
    sorted_keys_by_delta,
    summarize_metrics,
)


SYSTEMS = {
    "typeonlylite": {
        "branch": "confrare10_heur10_typeonlylite",
        "pred_root": REPO
        / "outputs/stage2_adaptive_runs_user_formal_clean"
        / "richere_split1_qwen3_4b_adaptive_confrare10_heur10_typeonlylite",
    },
    "typerolelite": {
        "branch": "confrare10_heur10_typerolelite",
        "pred_root": REPO
        / "outputs/stage2_adaptive_runs_user_formal_clean"
        / "richere_split1_qwen3_4b_adaptive_confrare10_heur10_typerolelite",
    },
}
SCORE_ROOT = REPO / "outputs/stage2_4b_selector/route_likelihood_s12_20260521"
DEV_EXEC_ROOT = REPO / "outputs/stage2_4b_selector/dev_forced_execution_s12_20260521"
OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_4b_route_nll_selector_s12.json"
OUT_MD = REPO / "reports/2026-05-21_stage2_4b_route_nll_selector_s12.md"
FORMAL_SPLITS = ["test_seen", "test_unseen"]


def metric_delta(direct_row, reason_row):
    direct_m = row_metric(direct_row)
    reason_m = row_metric(reason_row)
    return {
        "trigger_f1": reason_m["trigger"]["f1"] - direct_m["trigger"]["f1"],
        "argument_f1": reason_m["argument"]["f1"] - direct_m["argument"]["f1"],
        "event_f1": reason_m["event"]["f1"] - direct_m["event"]["f1"],
        "score": score(reason_row) - score(direct_row),
    }


def mean_dict(rows):
    keys = ["trigger_f1", "argument_f1", "event_f1", "score"]
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def load_predictions(system, split):
    if split == "dev_seen":
        direct = load_prediction_map(DEV_EXEC_ROOT / system / "forced_direct" / "predictions.jsonl")
        reason = load_prediction_map(DEV_EXEC_ROOT / system / "forced_reason" / "predictions.jsonl")
        return direct, reason
    root = SYSTEMS[system]["pred_root"]
    direct = load_prediction_map(root / "forced_direct" / split / "predictions.jsonl")
    reason = load_prediction_map(root / "forced_reason" / split / "predictions.jsonl")
    return direct, reason


def load_scores(system, split):
    return load_score_rows(SCORE_ROOT / system / split / "scores.jsonl")


def ranked_keys(scores, keys):
    scored = []
    for key in keys:
        delta = scores[key].get("delta_direct_minus_reason_route_nll")
        scored.append((float(delta) if delta is not None else float("-inf"), key))
    scored.sort(reverse=True)
    return [key for _, key in scored]


def m06_keys(split, keys):
    if split == "dev_seen":
        return set()
    m02_scores = load_score_rows(M02_FORMAL_ROOT / "checkpoint-50" / split / "scores.jsonl")
    m05_scores = load_score_rows(M05_FORMAL_ROOT / "checkpoint-100" / split / "scores.jsonl")
    common = sorted(set(keys) & set(m02_scores) & set(m05_scores))
    m02, _ = selected_by_window(m02_scores, common, M02_WINDOW)
    m05, _ = selected_by_window(m05_scores, common, M05_WINDOW)
    return m02 | m05


def evaluate_selected(system, split, policy, selected, direct_rows, reason_rows, keys):
    direct_metrics = []
    reason_metrics = []
    routed_metrics = []
    selected_deltas = []
    for key in keys:
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        chosen = reason_row if key in selected else direct_row
        direct_metrics.append(row_metric(direct_row))
        reason_metrics.append(row_metric(reason_row))
        routed_metrics.append(row_metric(chosen))
        if key in selected:
            selected_deltas.append(metric_delta(direct_row, reason_row))
    direct = summarize_metrics(direct_metrics)
    reason = summarize_metrics(reason_metrics)
    routed = summarize_metrics(routed_metrics)
    return {
        "system": system,
        "split": split,
        "policy": policy,
        "num_examples": len(keys),
        "pred_reason_count": len(selected),
        "pred_reason_rate": len(selected) / len(keys) if keys else 0.0,
        "direct": direct,
        "forced_reason_all": reason,
        "routed": routed,
        "routed_minus_direct": {
            metric: routed[metric] - direct[metric]
            for metric in ["trigger_f1", "argument_f1", "event_f1"]
        },
        "selected_delta_mean": mean_dict(selected_deltas),
    }


def candidate_sets(system, split, direct_rows, reason_rows, scores):
    keys = sorted(set(direct_rows) & set(reason_rows) & set(scores))
    ranked = ranked_keys(scores, keys)
    out = {
        "direct_only": set(),
        "reason_all": set(keys),
        "oracle_4b_direct_reason": {
            key for key in keys if score(reason_rows[key]) > score(direct_rows[key])
        },
    }
    for pct in [0.05, 0.10, 0.15]:
        out[f"s12_nll_topk_{int(pct * 100):02d}"] = set(ranked[: round(len(ranked) * pct)])
    for start in [0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]:
        for width in [0.025, 0.05, 0.075, 0.10]:
            end = start + width
            if end > 0.20:
                continue
            s = round(len(ranked) * start)
            e = round(len(ranked) * end)
            out[f"s12_nll_window_{int(start*1000):03d}_{int(end*1000):03d}"] = set(ranked[s:e])
    if split != "dev_seen":
        prior = m06_keys(split, keys)
        out["m06_transfer"] = prior
        out["s12_m06_prior_intersect"] = set(ranked[: round(len(ranked) * 0.20)]) & prior
        out["s12_m06_prior_union_safe"] = set(ranked[: round(len(ranked) * 0.05)]) | prior
    return keys, out


def select_dev_candidate(rows):
    eligible = [
        row
        for row in rows
        if row["policy"].startswith("s12_")
        and row["pred_reason_rate"] <= 0.15
        and row["routed_minus_direct"]["trigger_f1"] >= -0.001
    ]
    if not eligible:
        eligible = [
            row
            for row in rows
            if row["policy"].startswith("s12_") and row["pred_reason_rate"] <= 0.15
        ]
    def key(row):
        delta = row["routed_minus_direct"]
        return (
            (delta["argument_f1"] + delta["event_f1"]) / 2,
            delta["argument_f1"],
            delta["event_f1"],
            -row["pred_reason_rate"],
        )
    return max(eligible, key=key)


def aggregate_test(rows):
    out = []
    for system in SYSTEMS:
        policies = sorted({row["policy"] for row in rows if row["system"] == system})
        for policy in policies:
            items = [
                row
                for row in rows
                if row["system"] == system and row["policy"] == policy and row["split"] in FORMAL_SPLITS
            ]
            total = sum(row["num_examples"] for row in items)
            if not total:
                continue
            pred_reason_count = sum(row["pred_reason_count"] for row in items)
            agg = {
                "system": system,
                "split": "test",
                "policy": policy,
                "num_examples": total,
                "pred_reason_count": pred_reason_count,
                "pred_reason_rate": pred_reason_count / total,
            }
            for group in ["direct", "forced_reason_all", "routed"]:
                agg[group] = {}
                for metric in ["trigger_f1", "argument_f1", "event_f1"]:
                    agg[group][metric] = (
                        sum(row[group][metric] * row["num_examples"] for row in items) / total
                    )
            agg["routed_minus_direct"] = {
                metric: agg["routed"][metric] - agg["direct"][metric]
                for metric in ["trigger_f1", "argument_f1", "event_f1"]
            }
            denom = pred_reason_count
            agg["selected_delta_mean"] = {
                metric: (
                    sum(row["selected_delta_mean"][metric] * row["pred_reason_count"] for row in items)
                    / denom
                    if denom
                    else 0.0
                )
                for metric in ["trigger_f1", "argument_f1", "event_f1", "score"]
            }
            out.append(agg)
    return out


def fmt(value):
    return f"{value:.4f}"


def signed(value):
    return f"{value:+.4f}"


def pct(value):
    return f"{value * 100:.1f}%"


def aet(row, group="routed"):
    metrics = row[group]
    return f"{fmt(metrics['argument_f1'])} / {fmt(metrics['event_f1'])} / {fmt(metrics['trigger_f1'])}"


def delta_aet(row):
    delta = row["routed_minus_direct"]
    return f"{signed(delta['argument_f1'])} / {signed(delta['event_f1'])} / {signed(delta['trigger_f1'])}"


def render_table(rows):
    lines = [
        "| system | policy | split | reason rate | routed A/E/T | delta vs direct A/E/T |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['system']}` | `{row['policy']}` | `{row['split']}` | "
            f"{pct(row['pred_reason_rate'])} | {aet(row)} | {delta_aet(row)} |"
        )
    return "\n".join(lines)


def render_report(payload):
    rows = payload["results"]
    dev = payload["dev_candidates"]
    formal_policies = set()
    for item in payload["locked_candidates"].values():
        formal_policies.add(item["policy"])
    formal_policies |= {"direct_only", "reason_all", "m06_transfer", "oracle_4b_direct_reason"}
    formal_rows = [
        row for row in rows if row["split"] == "test" and row["policy"] in formal_policies
    ]
    lines = [
        "# 4B Route-NLL Selector S12",
        "",
        "This report calibrates 4B-specific route-choice NLL selectors on dev_seen and replays locked policies on formal splits.",
        "",
        "## Locked Dev Candidates",
        "",
        render_table(sorted(dev, key=lambda row: row["system"])),
        "",
        "## Formal Test Results",
        "",
        render_table(sorted(formal_rows, key=lambda row: (row["system"], row["policy"]))),
        "",
        "## Reading",
        "",
    ]
    for row in sorted(formal_rows, key=lambda item: (item["system"], item["policy"])):
        lines.append(
            f"- `{row['system']}` `{row['policy']}`: reason rate `{pct(row['pred_reason_rate'])}`, "
            f"A/E/T delta `{delta_aet(row)}`."
        )
    return "\n".join(lines) + "\n"


def main():
    dev_candidates = []
    locked = {}
    all_rows = []
    for system in SYSTEMS:
        dev_direct, dev_reason = load_predictions(system, "dev_seen")
        dev_scores = load_scores(system, "dev_seen")
        dev_keys = sorted(set(dev_direct) & set(dev_reason) & set(dev_scores))
        dev_candidate_sets = candidate_sets(system, "dev_seen", dev_direct, dev_reason, dev_scores)[1]
        dev_rows = [
            evaluate_selected(system, "dev_seen", policy, selected, dev_direct, dev_reason, dev_keys)
            for policy, selected in dev_candidate_sets.items()
        ]
        best = select_dev_candidate(dev_rows)
        dev_candidates.append(best)
        locked[system] = {
            "policy": best["policy"],
            "reason_rate": best["pred_reason_rate"],
            "dev_delta": best["routed_minus_direct"],
        }

    for system in SYSTEMS:
        locked_policy = locked[system]["policy"]
        for split in FORMAL_SPLITS:
            direct_rows, reason_rows = load_predictions(system, split)
            scores = load_scores(system, split)
            keys, sets = candidate_sets(system, split, direct_rows, reason_rows, scores)
            selected_sets = {
                "direct_only": sets["direct_only"],
                "reason_all": sets["reason_all"],
                "oracle_4b_direct_reason": sets["oracle_4b_direct_reason"],
                "m06_transfer": sets.get("m06_transfer", set()),
            }
            if locked_policy.startswith("s12_nll_topk_"):
                selected_sets[locked_policy] = sets[locked_policy]
            elif locked_policy.startswith("s12_nll_window_"):
                selected_sets[locked_policy] = sets[locked_policy]
            else:
                selected_sets[locked_policy] = sets[locked_policy]
            for policy, selected in selected_sets.items():
                all_rows.append(
                    evaluate_selected(system, split, policy, selected, direct_rows, reason_rows, keys)
                )
    all_rows.extend(aggregate_test(all_rows))
    payload = {
        "score_root": SCORE_ROOT.as_posix(),
        "systems": SYSTEMS,
        "locked_candidates": locked,
        "dev_candidates": dev_candidates,
        "results": all_rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
