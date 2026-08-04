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


BRANCH = "aet_stable_router_m02_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/aet_stable_router_m02_20260520/route_likelihood" / BRANCH
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_dev.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_dev.md"


def ckpt_num(path: Path) -> int:
    return int(path.parent.name.split("-", 1)[1])


def fmt_delta(delta):
    return "{argument_f1:+.4f} / {event_f1:+.4f} / {trigger_f1:+.4f}".format(**delta)


def is_all_nonnegative(row):
    d = row["routed_delta_vs_direct"]
    return d["argument_f1"] >= 0 and d["event_f1"] >= 0 and d["trigger_f1"] >= 0


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


def sweep(direct_rows, reason_rows, common_keys):
    rows = []
    endpoints = [i / 40 for i in range(0, 21)]
    score_paths = sorted(SCORE_ROOT.glob("checkpoint-*/dev_seen_scores.jsonl"), key=ckpt_num)
    if not score_paths:
        raise RuntimeError(f"no score files found under {SCORE_ROOT}")
    for score_path in score_paths:
        ckpt = score_path.parent.name
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
                name = f"{ckpt}_rank{int(lo * 1000):03d}_{int(hi * 1000):03d}"
                row = evaluate_policy(name, ckpt, keys, selected, score_rows, direct_rows, reason_rows)
                row["branch"] = BRANCH
                row["rank_window"] = {
                    "start_pct": lo,
                    "end_pct": hi,
                    "start_rank": start + 1,
                    "end_rank": end,
                }
                row.update(selected_fold_floor(row, keys, selected, score_rows, direct_rows, reason_rows))
                rows.append(row)
    return rows


def balanced_score(row):
    d = row["routed_delta_vs_direct"]
    return (
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        row["fold_min_aet"],
        d["event_f1"],
        -row["rank_window"]["end_pct"],
    )


def stable_score(row):
    d = row["routed_delta_vs_direct"]
    return (
        row["fold_min_aet"],
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        -row["rank_window"]["end_pct"],
        d["event_f1"],
    )


def render_table(rows):
    lines = [
        "| policy | reason rate | delta A/E/T | fold floor | routed A/E/T |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        routed = row["routed"]
        lines.append(
            "| {name} | {rate:.1%} | {delta} | {floor:+.4f} | {a:.4f} / {e:.4f} / {t:.4f} |".format(
                name=row["name"],
                rate=row["reason_rate"],
                delta=fmt_delta(row["routed_delta_vs_direct"]),
                floor=row["fold_min_aet"],
                a=routed["argument_f1"],
                e=routed["event_f1"],
                t=routed["trigger_f1"],
            )
        )
    return "\n".join(lines)


def render_report(payload):
    lines = [
        "# A/E/T Stable Router M02 Dev Sweep",
        "",
        "This sweep selects windows using raw A/E/T deltas plus dev-only fold stability.",
        "",
        "## Locked Candidates",
        "",
        render_table([payload["balanced_candidate"], payload["early_stable_candidate"]]),
        "",
        "## Top All-Nonnegative",
        "",
        render_table(payload["top_all_nonnegative"][:20]),
        "",
        "## Top Stable",
        "",
        render_table(payload["top_stable"][:20]),
        "",
    ]
    return "\n".join(lines)


def main():
    direct_rows = load_prediction_map(DIRECT_DEV)
    reason_rows = load_prediction_map(REASON_DEV)
    common_keys = sorted(set(direct_rows) & set(reason_rows))
    rows = sweep(direct_rows, reason_rows, common_keys)
    all_nonnegative = [
        row
        for row in rows
        if is_all_nonnegative(row) and 0.075 <= row["reason_rate"] <= 0.175
    ]
    if not all_nonnegative:
        raise RuntimeError("no all-nonnegative dev candidate in target reason-rate range")
    balanced = max(all_nonnegative, key=balanced_score)
    stable = max(all_nonnegative, key=stable_score)
    payload = {
        "selection_metric": "A/E/T plus dev-only fold stability; formal not used",
        "branch": BRANCH,
        "score_root": SCORE_ROOT.as_posix(),
        "num_candidates": len(rows),
        "num_all_nonnegative_target_rate": len(all_nonnegative),
        "balanced_candidate": balanced,
        "early_stable_candidate": stable,
        "top_all_nonnegative": sorted(all_nonnegative, key=balanced_score, reverse=True)[:50],
        "top_stable": sorted(all_nonnegative, key=stable_score, reverse=True)[:50],
        "all_candidates": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
