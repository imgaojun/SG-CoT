import json
import sys
from hashlib import md5
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.calibrate_modular_dualexpert_utility_router_m02_rank_window_dev_20260520 import (  # noqa: E402
    DIRECT_DEV,
    REASON_DEV,
    evaluate_policy,
    load_prediction_map,
    load_score_rows,
    sorted_keys_by_delta,
)
from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    DIRECT_ROOT,
    REASON_ROOT,
    row_metric,
    score,
    summarize_metrics,
)


BRANCH = "aet_stable_router_m02_routecls_noauxwarm_lr2e6_save50"
DEV_SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/aet_stable_router_m02_20260520/route_likelihood" / BRANCH
FORMAL_SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/aet_stable_router_m02_20260520/formal_route_likelihood" / BRANCH
OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_positive_retention_robustness.json"
OUT_MD = REPO / "reports/2026-05-21_stage2_modular_dualexpert_aet_positive_retention_robustness.md"

CHECKPOINTS = ["checkpoint-50", "checkpoint-100"]
SPLITS = ["test_seen", "test_unseen"]
PRECISION_THRESHOLDS = [0.30, 0.35, 0.40]
RECALL_THRESHOLDS = [0.10, 0.12, 0.15]
TARGET_RATES = [(0.045, 0.065), (0.065, 0.085), (0.085, 0.110), (0.110, 0.135)]
ENDPOINTS = [i / 40 for i in range(0, 21)]
ANCHOR_CHECKPOINT = "checkpoint-50"
ANCHOR_WINDOW = (0.425, 0.500)


def stable_fold(key):
    return int(md5(key.encode("utf-8")).hexdigest()[:8], 16) % 5


def selected_fold_floor(row, keys, selected_keys, score_rows, direct_rows, reason_rows):
    folds = []
    selected = set(selected_keys)
    for fold in range(5):
        fold_keys = [key for key in keys if stable_fold(key) == fold]
        if not fold_keys:
            continue
        fold_selected = [key for key in fold_keys if key in selected]
        fold_row = evaluate_policy(
            f"{row['name']}_fold{fold}",
            row["checkpoint"],
            fold_keys,
            fold_selected,
            score_rows,
            direct_rows,
            reason_rows,
        )
        d = fold_row["routed_delta_vs_direct"]
        folds.append(
            {
                "fold": fold,
                "reason_rate": fold_row["reason_rate"],
                "delta": d,
                "min_aet": min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
            }
        )
    return {
        "folds": folds,
        "fold_min_aet": min((fold["min_aet"] for fold in folds), default=-99.0),
    }


def fmt_delta(delta):
    return "{argument_f1:+.4f} / {event_f1:+.4f} / {trigger_f1:+.4f}".format(**delta)


def retention_score(row):
    helpful = row["route_vs_positive_reason_helpful"]
    d = row["routed_delta_vs_direct"]
    return (
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        helpful["f1"],
        helpful["precision"],
        helpful["recall"],
        d["event_f1"],
        d["argument_f1"],
    )


def ckpt_path(checkpoint):
    return DEV_SCORE_ROOT / checkpoint / "dev_seen_scores.jsonl"


def make_dev_candidates():
    direct_rows = load_prediction_map(DIRECT_DEV)
    reason_rows = load_prediction_map(REASON_DEV)
    common_keys = sorted(set(direct_rows) & set(reason_rows))
    candidates = []
    for checkpoint in CHECKPOINTS:
        score_rows = load_score_rows(ckpt_path(checkpoint))
        keys = sorted_keys_by_delta(score_rows, common_keys)
        n = len(keys)
        for lo in ENDPOINTS:
            for hi in ENDPOINTS:
                if hi <= lo:
                    continue
                width = hi - lo
                if width < 0.05 or width > 0.20:
                    continue
                start = round(n * lo)
                end = round(n * hi)
                selected = keys[start:end]
                if not selected:
                    continue
                name = f"{checkpoint}_rank{int(lo * 1000):03d}_{int(hi * 1000):03d}"
                row = evaluate_policy(name, checkpoint, keys, selected, score_rows, direct_rows, reason_rows)
                row["branch"] = BRANCH
                row["rank_window"] = {"start_pct": lo, "end_pct": hi, "start_rank": start + 1, "end_rank": end}
                row.update(selected_fold_floor(row, keys, selected, score_rows, direct_rows, reason_rows))
                candidates.append(row)
    return candidates


def choose_policies(candidates):
    selected = []
    seen = set()
    for precision in PRECISION_THRESHOLDS:
        for recall in RECALL_THRESHOLDS:
            for lo_rate, hi_rate in TARGET_RATES:
                pool = []
                for row in candidates:
                    helpful = row["route_vs_positive_reason_helpful"]
                    d = row["routed_delta_vs_direct"]
                    if not (lo_rate <= row["reason_rate"] <= hi_rate):
                        continue
                    if helpful["precision"] < precision or helpful["recall"] < recall:
                        continue
                    if d["argument_f1"] < 0 or d["event_f1"] < 0:
                        continue
                    pool.append(row)
                if not pool:
                    continue
                row = max(pool, key=retention_score)
                ident = (row["checkpoint"], row["rank_window"]["start_pct"], row["rank_window"]["end_pct"])
                if ident in seen:
                    continue
                seen.add(ident)
                copy = dict(row)
                copy["selection_constraints"] = {
                    "min_helpful_precision": precision,
                    "min_helpful_recall": recall,
                    "target_reason_rate": [lo_rate, hi_rate],
                }
                selected.append(copy)
    anchor = [
        row
        for row in candidates
        if row["checkpoint"] == ANCHOR_CHECKPOINT
        and row["rank_window"]["start_pct"] == ANCHOR_WINDOW[0]
        and row["rank_window"]["end_pct"] == ANCHOR_WINDOW[1]
    ]
    if anchor:
        anchor_row = dict(anchor[0])
        anchor_row["selection_constraints"] = {"anchor_policy": True}
        ident = (
            anchor_row["checkpoint"],
            anchor_row["rank_window"]["start_pct"],
            anchor_row["rank_window"]["end_pct"],
        )
        if ident not in seen:
            selected.insert(0, anchor_row)
    return selected


def load_prediction_map_formal(path: Path):
    from src.stage2_analysis.analyze_adaptive_hardness_boundary import prediction_key  # noqa: E402
    from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402

    return {prediction_key(row): row for row in load_jsonl(path)}


def evaluate_formal(policy, split):
    from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import load_score_rows as load_scores  # noqa: E402

    score_path = FORMAL_SCORE_ROOT / policy["checkpoint"] / split / "scores.jsonl"
    direct_path = DIRECT_ROOT / split / "predictions.jsonl"
    reason_path = REASON_ROOT / split / "predictions.jsonl"
    score_rows = load_scores(score_path)
    direct_rows = load_prediction_map_formal(direct_path)
    reason_rows = load_prediction_map_formal(reason_path)
    keys = sorted_keys_by_delta(score_rows, set(direct_rows) & set(reason_rows))
    start = round(len(keys) * policy["rank_window"]["start_pct"])
    end = round(len(keys) * policy["rank_window"]["end_pct"])
    reason_keys = set(keys[start:end])
    routed_metrics = []
    direct_metrics = []
    reason_metrics = []
    selected_gains = []
    for key in keys:
        direct_row = direct_rows[key]
        reason_row = reason_rows[key]
        chosen = reason_row if key in reason_keys else direct_row
        routed_metrics.append(row_metric(chosen))
        direct_metrics.append(row_metric(direct_row))
        reason_metrics.append(row_metric(reason_row))
        if key in reason_keys:
            selected_gains.append(score(reason_row) - score(direct_row))
    direct = summarize_metrics(direct_metrics)
    reason = summarize_metrics(reason_metrics)
    routed = summarize_metrics(routed_metrics)
    return {
        "policy": policy["name"],
        "checkpoint": policy["checkpoint"],
        "split": split,
        "num_examples": len(keys),
        "pred_reason_count": len(reason_keys),
        "pred_reason_rate": len(reason_keys) / len(keys) if keys else 0.0,
        "selected_reason_avg_score_gain": sum(selected_gains) / len(selected_gains) if selected_gains else 0.0,
        "direct": direct,
        "forced_reason_all": reason,
        "routed": routed,
        "routed_minus_direct": {
            "trigger_f1": routed["trigger_f1"] - direct["trigger_f1"],
            "argument_f1": routed["argument_f1"] - direct["argument_f1"],
            "event_f1": routed["event_f1"] - direct["event_f1"],
        },
    }


def aggregate_test(rows, policies):
    out = []
    for policy in policies:
        items = [row for row in rows if row["policy"] == policy["name"]]
        total = sum(row["num_examples"] for row in items)
        if not total:
            continue
        pred_reason_count = sum(row["pred_reason_count"] for row in items)
        agg = {
            "policy": policy["name"],
            "checkpoint": policy["checkpoint"],
            "split": "test",
            "num_examples": total,
            "pred_reason_count": pred_reason_count,
            "pred_reason_rate": pred_reason_count / total,
            "selected_reason_avg_score_gain": (
                sum(row["selected_reason_avg_score_gain"] * row["pred_reason_count"] for row in items) / pred_reason_count
                if pred_reason_count
                else 0.0
            ),
        }
        for group in ["direct", "forced_reason_all", "routed"]:
            agg[group] = {}
            for metric in ["trigger_f1", "argument_f1", "event_f1"]:
                agg[group][metric] = sum(row[group][metric] * row["num_examples"] for row in items) / total
        agg["routed_minus_direct"] = {
            metric: agg["routed"][metric] - agg["direct"][metric]
            for metric in ["trigger_f1", "argument_f1", "event_f1"]
        }
        out.append(agg)
    return out


def render_table(rows):
    lines = [
        "| policy | split | reason rate | delta A/E/T | selected gain |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {split} | {rate:.1%} | {delta} | {gain:+.4f} |".format(
                policy=row["policy"],
                split=row["split"],
                rate=row["pred_reason_rate"],
                delta=fmt_delta(row["routed_minus_direct"]),
                gain=row["selected_reason_avg_score_gain"],
            )
        )
    return "\n".join(lines)


def render(payload):
    test_rows = [row for row in payload["formal_results"] if row["split"] == "test"]
    all_positive = [
        row
        for row in test_rows
        if min(
            row["routed_minus_direct"]["argument_f1"],
            row["routed_minus_direct"]["event_f1"],
            row["routed_minus_direct"]["trigger_f1"],
        )
        >= 0
    ]
    lines = [
        "# Positive-Retention Robustness Sweep",
        "",
        "This sweep perturbs positive-retention constraints and rank windows around the current m02 main selector.",
        "",
        f"- formal test policies evaluated: `{len(test_rows)}`",
        f"- all-positive formal test policies: `{len(all_positive)}`",
        "",
        "## Formal Test Results",
        "",
        render_table(sorted(test_rows, key=lambda row: min(row["routed_minus_direct"].values()), reverse=True)),
        "",
        "## Dev-Locked Policies",
        "",
        "| policy | reason rate | helpful P/R/F1 | dev delta A/E/T | fold floor |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["policies"]:
        helpful = row["route_vs_positive_reason_helpful"]
        lines.append(
            "| {name} | {rate:.1%} | {p:.3f}/{r:.3f}/{f:.3f} | {delta} | {floor:+.4f} |".format(
                name=row["name"],
                rate=row["reason_rate"],
                p=helpful["precision"],
                r=helpful["recall"],
                f=helpful["f1"],
                delta=fmt_delta(row["routed_delta_vs_direct"]),
                floor=row["fold_min_aet"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main():
    dev_candidates = make_dev_candidates()
    policies = choose_policies(dev_candidates)
    split_rows = []
    for policy in policies:
        for split in SPLITS:
            split_rows.append(evaluate_formal(policy, split))
    formal_rows = split_rows + aggregate_test(split_rows, policies)
    payload = {
        "branch": BRANCH,
        "dev_score_root": DEV_SCORE_ROOT.as_posix(),
        "formal_score_root": FORMAL_SCORE_ROOT.as_posix(),
        "num_dev_candidates": len(dev_candidates),
        "policies": policies,
        "formal_results": formal_rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
