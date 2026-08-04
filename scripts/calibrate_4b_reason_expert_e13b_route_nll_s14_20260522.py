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
from scripts.summarize_4b_reason_expert_e13b_20260521 import NEW_ROOT  # noqa: E402
from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    load_prediction_map,
    load_score_rows,
    row_metric,
    score,
    summarize_metrics,
)


SCORE_ROOT = REPO / "outputs/stage2_4b_reason_expert/e13b_route_nll_s14_20260522"
DEV_EXEC_ROOT = REPO / "outputs/stage2_4b_reason_expert/e13b_dev_forced_execution_s14_20260522"
OUT_JSON = REPO / "reports/artifacts/2026-05-22_stage2_4b_reason_expert_e13b_route_nll_selector_s14.json"
OUT_MD = REPO / "reports/2026-05-22_stage2_4b_reason_expert_e13b_route_nll_selector_s14.md"
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


def load_dev_predictions():
    return (
        load_prediction_map(DEV_EXEC_ROOT / "forced_direct" / "predictions.jsonl"),
        load_prediction_map(DEV_EXEC_ROOT / "forced_reason" / "predictions.jsonl"),
    )


def load_formal_predictions(split):
    return (
        load_prediction_map(NEW_ROOT / "forced_direct" / split / "predictions.jsonl"),
        load_prediction_map(NEW_ROOT / "forced_reason" / split / "predictions.jsonl"),
    )


def load_scores(split):
    return load_score_rows(SCORE_ROOT / split / "scores.jsonl")


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


def evaluate_selected(split, policy, selected, direct_rows, reason_rows, keys):
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


def parse_policy(policy, ranked):
    if policy == "direct_only":
        return set()
    if policy == "reason_all":
        return set(ranked)
    if policy.startswith("s14_top"):
        pct = int(policy.removeprefix("s14_top")) / 1000
        return set(ranked[: round(len(ranked) * pct)])
    if policy.startswith("s14_window_"):
        _, _, start_raw, end_raw = policy.split("_")
        start = int(start_raw) / 1000
        end = int(end_raw) / 1000
        s = round(len(ranked) * start)
        e = round(len(ranked) * end)
        return set(ranked[s:e])
    raise ValueError(policy)


def candidate_sets(split, direct_rows, reason_rows, scores):
    keys = sorted(set(direct_rows) & set(reason_rows) & set(scores))
    ranked = ranked_keys(scores, keys)
    out = {
        "direct_only": set(),
        "reason_all": set(keys),
        "oracle_4b_direct_reason": {
            key for key in keys if score(reason_rows[key]) > score(direct_rows[key])
        },
    }
    for pct in [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]:
        out[f"s14_top{int(pct * 1000):03d}"] = set(ranked[: round(len(ranked) * pct)])
    for start in [i / 40 for i in range(0, 36)]:
        for width in [0.025, 0.05, 0.075, 0.10, 0.125, 0.15]:
            end = start + width
            if end > 1.0:
                continue
            s = round(len(ranked) * start)
            e = round(len(ranked) * end)
            if e <= s:
                continue
            out[f"s14_window_{int(start * 1000):03d}_{int(end * 1000):03d}"] = set(ranked[s:e])
    if split != "dev_seen":
        out["m06_transfer"] = m06_keys(split, keys)
    return keys, ranked, out


def select_dev_candidate(dev_rows):
    candidates = [
        row
        for row in dev_rows
        if row["policy"].startswith("s14_")
        and 0 < row["pred_reason_rate"] <= 0.15
    ]
    nonnegative = [
        row
        for row in candidates
        if row["routed_minus_direct"]["argument_f1"] >= -0.001
        and row["routed_minus_direct"]["event_f1"] >= -0.001
        and row["routed_minus_direct"]["trigger_f1"] >= -0.001
    ]
    pool = nonnegative or candidates

    def key(row):
        delta = row["routed_minus_direct"]
        return (
            min(delta["argument_f1"], delta["event_f1"], delta["trigger_f1"]),
            (delta["argument_f1"] + delta["event_f1"]) / 2,
            delta["trigger_f1"],
            -row["pred_reason_rate"],
        )

    return max(pool, key=key)


def aggregate_test(rows):
    policies = sorted({row["policy"] for row in rows if row["split"] in FORMAL_SPLITS})
    out = []
    for policy in policies:
        items = [row for row in rows if row["policy"] == policy and row["split"] in FORMAL_SPLITS]
        total = sum(row["num_examples"] for row in items)
        if not total:
            continue
        pred_reason_count = sum(row["pred_reason_count"] for row in items)
        agg = {
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
    return f"{100 * value:.1f}%"


def aet(row, group="routed"):
    metrics = row[group]
    return f"{fmt(metrics['argument_f1'])} / {fmt(metrics['event_f1'])} / {fmt(metrics['trigger_f1'])}"


def delta_aet(row):
    delta = row["routed_minus_direct"]
    return f"{signed(delta['argument_f1'])} / {signed(delta['event_f1'])} / {signed(delta['trigger_f1'])}"


def render_table(rows):
    lines = [
        "| policy | split | reason rate | routed A/E/T | delta vs direct A/E/T | selected mean A/E/T |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        selected = row["selected_delta_mean"]
        lines.append(
            f"| `{row['policy']}` | `{row['split']}` | {pct(row['pred_reason_rate'])} | "
            f"{aet(row)} | {delta_aet(row)} | "
            f"{signed(selected['argument_f1'])} / {signed(selected['event_f1'])} / {signed(selected['trigger_f1'])} |"
        )
    return "\n".join(lines)


def render(payload):
    locked = payload["locked_candidate"]["policy"]
    dev_rows = payload["dev_candidates"]
    formal_policies = {"direct_only", "reason_all", "m06_transfer", "oracle_4b_direct_reason", locked}
    formal_rows = [
        row for row in payload["results"] if row["split"] == "test" and row["policy"] in formal_policies
    ]
    lines = [
        "# E13B Route-NLL Selector S14",
        "",
        "This report uses E13B-specific route-NLL scores and dev forced execution to lock one selector policy, then replays it on formal splits.",
        "",
        "## Locked Dev Candidate",
        "",
        render_table([payload["locked_candidate"]]),
        "",
        "## Formal Test Results",
        "",
        render_table(sorted(formal_rows, key=lambda row: row["policy"])),
        "",
        "## Top Dev Candidates",
        "",
        render_table(dev_rows[:10]),
        "",
        "## Reading",
        "",
    ]
    locked_row = next(row for row in formal_rows if row["policy"] == locked)
    lines.append(
        f"- Locked S14 policy `{locked}` gives formal `test` A/E/T delta `{delta_aet(locked_row)}` at reason rate `{pct(locked_row['pred_reason_rate'])}`."
    )
    lines.append("- The policy is selected only from dev execution; formal oracle rows are diagnostic comparators.")
    return "\n".join(lines) + "\n"


def main():
    dev_direct, dev_reason = load_dev_predictions()
    dev_scores = load_scores("dev_seen")
    dev_keys, _, dev_sets = candidate_sets("dev_seen", dev_direct, dev_reason, dev_scores)
    dev_rows = [
        evaluate_selected("dev_seen", policy, selected, dev_direct, dev_reason, dev_keys)
        for policy, selected in dev_sets.items()
        if policy.startswith("s14_")
    ]
    best = select_dev_candidate(dev_rows)

    rows = []
    for split in FORMAL_SPLITS:
        direct_rows, reason_rows = load_formal_predictions(split)
        scores = load_scores(split)
        keys, ranked, sets = candidate_sets(split, direct_rows, reason_rows, scores)
        selected_sets = {
            "direct_only": sets["direct_only"],
            "reason_all": sets["reason_all"],
            "m06_transfer": sets["m06_transfer"],
            "oracle_4b_direct_reason": sets["oracle_4b_direct_reason"],
            best["policy"]: parse_policy(best["policy"], ranked),
        }
        for policy, selected in selected_sets.items():
            rows.append(evaluate_selected(split, policy, selected, direct_rows, reason_rows, keys))
    rows.extend(aggregate_test(rows))

    top_dev = sorted(
        dev_rows,
        key=lambda row: (
            min(row["routed_minus_direct"].values()),
            (row["routed_minus_direct"]["argument_f1"] + row["routed_minus_direct"]["event_f1"]) / 2,
        ),
        reverse=True,
    )[:20]
    payload = {
        "score_root": SCORE_ROOT.as_posix(),
        "dev_exec_root": DEV_EXEC_ROOT.as_posix(),
        "locked_candidate": best,
        "dev_candidates": top_dev,
        "results": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
