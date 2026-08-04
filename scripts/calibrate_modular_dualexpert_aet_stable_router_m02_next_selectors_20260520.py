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
from src.stage2_data.build_formal_stage2_dataset import load_jsonl  # noqa: E402


BRANCH = "aet_stable_router_m02_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/aet_stable_router_m02_20260520/route_likelihood" / BRANCH
LABEL_DEV = REPO / (
    "data/stage2_adaptive_datasets/labels/"
    "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_"
    "modular_d1930_r2058_aet_stable_m02_dev_seen_labels.jsonl"
)
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_next_selectors_dev.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_next_selectors_dev.md"

FORMAL_SCORED_CHECKPOINTS = {"checkpoint-50", "checkpoint-100"}
TARGET_REASON_RATE = (0.05, 0.12)
RETENTION_MIN_PRECISION = 0.35
RETENTION_MIN_RECALL = 0.12


def ckpt_num(path: Path) -> int:
    return int(path.parent.name.split("-", 1)[1])


def label_key(row):
    return row.get("wnd_id") or row.get("id")


def load_label_map(path: Path):
    return {label_key(row): row for row in load_jsonl(path)}


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
        "fold_min_event": min((fold["delta"]["event_f1"] for fold in folds), default=-99.0),
    }


def annotate(row, keys, selected, score_rows, direct_rows, reason_rows, policy_family, extra=None):
    row["branch"] = BRANCH
    row["policy_family"] = policy_family
    if extra:
        row.update(extra)
    row.update(selected_fold_floor(row, keys, selected, score_rows, direct_rows, reason_rows))
    return row


def fmt_delta(delta):
    return "{argument_f1:+.4f} / {event_f1:+.4f} / {trigger_f1:+.4f}".format(**delta)


def is_target_rate(row):
    lo, hi = TARGET_REASON_RATE
    return lo <= row["reason_rate"] <= hi


def is_all_nonnegative(row):
    d = row["routed_delta_vs_direct"]
    return d["argument_f1"] >= 0 and d["event_f1"] >= 0 and d["trigger_f1"] >= 0


def balanced_score(row):
    d = row["routed_delta_vs_direct"]
    helpful = row["route_vs_positive_reason_helpful"]
    return (
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        row["fold_min_aet"],
        helpful["recall"],
        helpful["precision"],
        d["event_f1"],
        d["argument_f1"],
    )


def retention_score(row):
    d = row["routed_delta_vs_direct"]
    helpful = row["route_vs_positive_reason_helpful"]
    return (
        helpful["f1"],
        helpful["recall"],
        helpful["precision"],
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        d["event_f1"],
        d["argument_f1"],
    )


def source_score(row):
    d = row["routed_delta_vs_direct"]
    helpful = row["route_vs_positive_reason_helpful"]
    return (
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        row["fold_min_aet"],
        d["event_f1"],
        helpful["precision"],
        helpful["recall"],
    )


def sweep_global(direct_rows, reason_rows, label_rows, common_keys):
    rows = []
    endpoints = [i / 40 for i in range(0, 21)]
    score_paths = sorted(SCORE_ROOT.glob("checkpoint-*/dev_seen_scores.jsonl"), key=ckpt_num)
    for score_path in score_paths:
        ckpt = score_path.parent.name
        if ckpt not in FORMAL_SCORED_CHECKPOINTS:
            continue
        score_rows = load_score_rows(score_path)
        keys = sorted_keys_by_delta(score_rows, common_keys)
        n = len(keys)
        for lo in endpoints:
            for hi in endpoints:
                if hi <= lo:
                    continue
                rate = hi - lo
                if rate < 0.05 or rate > 0.20:
                    continue
                start = round(n * lo)
                end = round(n * hi)
                selected = keys[start:end]
                if not selected:
                    continue
                name = f"{ckpt}_global_rank{int(lo * 1000):03d}_{int(hi * 1000):03d}"
                row = evaluate_policy(name, ckpt, keys, selected, score_rows, direct_rows, reason_rows)
                row = annotate(
                    row,
                    keys,
                    selected,
                    score_rows,
                    direct_rows,
                    reason_rows,
                    "global_positive_retention",
                    {
                        "rank_window": {
                            "start_pct": lo,
                            "end_pct": hi,
                            "start_rank": start + 1,
                            "end_rank": end,
                        },
                        "group_rules": [
                            {
                                "group": "all",
                                "start_pct": lo,
                                "end_pct": hi,
                            }
                        ],
                    },
                )
                rows.append(row)
    return rows


def sweep_source_aware(direct_rows, reason_rows, label_rows, common_keys):
    rows = []
    endpoints = [i / 40 for i in range(0, 41)]
    score_paths = sorted(SCORE_ROOT.glob("checkpoint-*/dev_seen_scores.jsonl"), key=ckpt_num)
    for score_path in score_paths:
        ckpt = score_path.parent.name
        if ckpt not in FORMAL_SCORED_CHECKPOINTS:
            continue
        score_rows = load_score_rows(score_path)
        keys = sorted_keys_by_delta(score_rows, common_keys)
        stable_keys = [
            key
            for key in keys
            if label_rows.get(key, {}).get("stable_reason_bucket") is True
        ]
        group_sorted = [key for key in keys if key in set(stable_keys)]
        n_group = len(group_sorted)
        if not n_group:
            continue
        for lo in endpoints:
            for hi in endpoints:
                if hi <= lo:
                    continue
                start = round(n_group * lo)
                end = round(n_group * hi)
                selected = group_sorted[start:end]
                if not selected:
                    continue
                name = f"{ckpt}_stablebucket_rank{int(lo * 1000):03d}_{int(hi * 1000):03d}"
                row = evaluate_policy(name, ckpt, keys, selected, score_rows, direct_rows, reason_rows)
                row = annotate(
                    row,
                    keys,
                    selected,
                    score_rows,
                    direct_rows,
                    reason_rows,
                    "source_aware_stable_bucket",
                    {
                        "rank_window": {
                            "group": "stable_reason_bucket=true",
                            "start_pct": lo,
                            "end_pct": hi,
                            "start_rank_in_group": start + 1,
                            "end_rank_in_group": end,
                            "group_size": n_group,
                        },
                        "group_rules": [
                            {
                                "group": "stable_reason_bucket=true",
                                "start_pct": lo,
                                "end_pct": hi,
                            },
                            {
                                "group": "stable_reason_bucket=false",
                                "action": "direct",
                            },
                        ],
                    },
                )
                rows.append(row)
    return rows


def select_candidates(global_rows, source_rows):
    target_global = [row for row in global_rows if is_target_rate(row)]
    target_source = [row for row in source_rows if is_target_rate(row)]
    all_nonnegative = [row for row in target_global + target_source if is_all_nonnegative(row)]
    retention = [
        row
        for row in target_global + target_source
        if row["route_vs_positive_reason_helpful"]["precision"] >= RETENTION_MIN_PRECISION
        and row["route_vs_positive_reason_helpful"]["recall"] >= RETENTION_MIN_RECALL
        and row["routed_delta_vs_direct"]["argument_f1"] >= 0
        and row["routed_delta_vs_direct"]["event_f1"] >= 0
    ]
    if not all_nonnegative:
        raise RuntimeError("no all-nonnegative m02 next-selector candidate in target rate range")
    if not retention:
        raise RuntimeError("no positive-retention candidate met precision/recall constraints")
    source_pool = [row for row in target_source if is_all_nonnegative(row)] or target_source
    return {
        "balanced_candidate": max(all_nonnegative, key=balanced_score),
        "positive_retention_candidate": max(retention, key=retention_score),
        "source_aware_candidate": max(source_pool, key=source_score),
        "top_all_nonnegative": sorted(all_nonnegative, key=balanced_score, reverse=True)[:30],
        "top_positive_retention": sorted(retention, key=retention_score, reverse=True)[:30],
        "top_source_aware": sorted(target_source, key=source_score, reverse=True)[:30],
    }


def render_table(rows):
    lines = [
        "| policy | family | reason rate | helpful P/R/F1 | delta A/E/T | fold floor |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        helpful = row["route_vs_positive_reason_helpful"]
        lines.append(
            "| {name} | {family} | {rate:.1%} | {p:.3f}/{r:.3f}/{f:.3f} | {delta} | {floor:+.4f} |".format(
                name=row["name"],
                family=row["policy_family"],
                rate=row["reason_rate"],
                p=helpful["precision"],
                r=helpful["recall"],
                f=helpful["f1"],
                delta=fmt_delta(row["routed_delta_vs_direct"]),
                floor=row["fold_min_aet"],
            )
        )
    return "\n".join(lines)


def render_report(payload):
    lines = [
        "# A/E/T Stable Router M02 Next Selectors Dev",
        "",
        "This dev-only sweep tests two follow-up selector ideas without retraining: stable-bucket source-aware routing and positive-retention calibration.",
        "",
        "## Locked Candidates",
        "",
        render_table(
            [
                payload["balanced_candidate"],
                payload["source_aware_candidate"],
                payload["positive_retention_candidate"],
            ]
        ),
        "",
        "## Top All-Nonnegative",
        "",
        render_table(payload["top_all_nonnegative"][:15]),
        "",
        "## Top Positive-Retention",
        "",
        render_table(payload["top_positive_retention"][:15]),
        "",
        "## Top Source-Aware",
        "",
        render_table(payload["top_source_aware"][:15]),
        "",
    ]
    return "\n".join(lines)


def main():
    direct_rows = load_prediction_map(DIRECT_DEV)
    reason_rows = load_prediction_map(REASON_DEV)
    label_rows = load_label_map(LABEL_DEV)
    common_keys = sorted(set(direct_rows) & set(reason_rows) & set(label_rows))
    global_rows = sweep_global(direct_rows, reason_rows, label_rows, common_keys)
    source_rows = sweep_source_aware(direct_rows, reason_rows, label_rows, common_keys)
    selected = select_candidates(global_rows, source_rows)
    payload = {
        "selection_metric": "m02 next selectors; dev only; formal not used",
        "branch": BRANCH,
        "score_root": SCORE_ROOT.as_posix(),
        "label_dev": LABEL_DEV.as_posix(),
        "formal_scored_checkpoints": sorted(FORMAL_SCORED_CHECKPOINTS),
        "target_reason_rate": TARGET_REASON_RATE,
        "retention_constraints": {
            "min_helpful_precision": RETENTION_MIN_PRECISION,
            "min_helpful_recall": RETENTION_MIN_RECALL,
        },
        "num_global_candidates": len(global_rows),
        "num_source_aware_candidates": len(source_rows),
        **selected,
        "all_global_candidates": global_rows,
        "all_source_aware_candidates": source_rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
