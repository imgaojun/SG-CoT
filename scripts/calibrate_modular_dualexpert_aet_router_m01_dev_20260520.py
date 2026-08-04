import json
import sys
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


BRANCHES = [
    "aet_safe_router_m01_routecls_noauxwarm_lr2e6_save50",
    "aet_event_router_m01_routecls_noauxwarm_lr2e6_save50",
]
SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/aet_router_m01_20260520/route_likelihood"
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_router_m01_dev.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_aet_router_m01_dev.md"


def ckpt_num(path: Path) -> int:
    return int(path.parent.name.split("-", 1)[1])


def fmt_delta(delta):
    return "{argument_f1:+.4f} / {event_f1:+.4f} / {trigger_f1:+.4f}".format(**delta)


def candidate_score_balanced(row):
    d = row["routed_delta_vs_direct"]
    return (
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        d["event_f1"],
        d["argument_f1"],
        d["trigger_f1"],
        -abs(row["reason_rate"] - 0.15),
    )


def candidate_score_event(row):
    d = row["routed_delta_vs_direct"]
    return (
        d["event_f1"],
        min(d["argument_f1"], d["trigger_f1"]),
        d["argument_f1"],
        d["trigger_f1"],
        -abs(row["reason_rate"] - 0.10),
    )


def is_all_nonnegative(row):
    d = row["routed_delta_vs_direct"]
    return d["argument_f1"] >= 0 and d["event_f1"] >= 0 and d["trigger_f1"] >= 0


def is_event_safe(row):
    d = row["routed_delta_vs_direct"]
    return d["event_f1"] > 0 and d["argument_f1"] >= -0.001 and d["trigger_f1"] >= -0.001


def sweep_branch(branch, direct_rows, reason_rows, common_keys):
    rows = []
    endpoints = [i / 40 for i in range(0, 21)]  # 0.000 ... 0.500 in 2.5% steps.
    branch_score_root = SCORE_ROOT / branch
    score_paths = sorted(branch_score_root.glob("checkpoint-*/dev_seen_scores.jsonl"), key=ckpt_num)
    if not score_paths:
        raise RuntimeError(f"no score files found under {branch_score_root}")

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
                if end <= start:
                    continue
                selected = keys[start:end]
                name = f"{branch}/{ckpt}_aet_rank{int(lo * 1000):03d}_{int(hi * 1000):03d}"
                row = evaluate_policy(name, ckpt, keys, selected, score_rows, direct_rows, reason_rows)
                row["branch"] = branch
                row["rank_window"] = {
                    "start_pct": lo,
                    "end_pct": hi,
                    "start_rank": start + 1,
                    "end_rank": end,
                }
                row["policy_family"] = "aet_constrained_rank_window"
                rows.append(row)
    return rows


def render_table(rows):
    lines = [
        "| policy | reason rate | delta A/E/T | routed A/E/T | label F1 | helpful F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        d = row["routed_delta_vs_direct"]
        routed = row["routed"]
        lines.append(
            "| {name} | {rate:.1%} | {delta} | {a:.4f} / {e:.4f} / {t:.4f} | {lf:.3f} | {hf:.3f} |".format(
                name=row["name"],
                rate=row["reason_rate"],
                delta=fmt_delta(d),
                a=routed["argument_f1"],
                e=routed["event_f1"],
                t=routed["trigger_f1"],
                lf=row["route_vs_label"]["f1"],
                hf=row["route_vs_positive_reason_helpful"]["f1"],
            )
        )
    return "\n".join(lines)


def render_report(payload):
    locked = [payload[key] for key in ("balanced_candidate", "event_candidate") if payload.get(key)]
    lines = [
        "# A/E/T Router M01 Dev Selector Sweep",
        "",
        "This sweep compares the newly trained A/E/T safe/event routers using raw Argument/Event/Trigger deltas only.",
        "",
        "## Locked Dev Candidates",
        "",
        render_table(locked),
        "",
        "## Best All-Nonnegative Windows",
        "",
        render_table(payload["top_all_nonnegative"][:20]),
        "",
        "## Best Event-Safe Windows",
        "",
        render_table(payload["top_event_safe"][:20]),
        "",
        "## Reading",
        "",
        "- `balanced_candidate` maximizes the worst A/E/T delta among all-nonnegative dev windows.",
        "- `event_candidate` maximizes Event F1 with Argument/Trigger constrained near nonnegative.",
        "- These are dev-selected candidates; formal test should replay the locked checkpoint/window without additional test tuning.",
        "",
    ]
    return "\n".join(lines)


def main():
    direct_rows = load_prediction_map(DIRECT_DEV)
    reason_rows = load_prediction_map(REASON_DEV)
    common_keys = sorted(set(direct_rows) & set(reason_rows))

    rows = []
    by_branch = {}
    for branch in BRANCHES:
        branch_rows = sweep_branch(branch, direct_rows, reason_rows, common_keys)
        by_branch[branch] = {
            "num_candidates": len(branch_rows),
            "num_all_nonnegative": sum(1 for row in branch_rows if is_all_nonnegative(row)),
            "num_event_safe": sum(1 for row in branch_rows if is_event_safe(row)),
        }
        rows.extend(branch_rows)

    all_nonnegative = [row for row in rows if is_all_nonnegative(row)]
    event_safe = [row for row in rows if is_event_safe(row)]
    payload = {
        "selection_metric": "A/E/T constrained; utility is not used",
        "branches": BRANCHES,
        "score_root": SCORE_ROOT.as_posix(),
        "direct_predictions": DIRECT_DEV.as_posix(),
        "reason_predictions": REASON_DEV.as_posix(),
        "by_branch": by_branch,
        "num_candidates": len(rows),
        "num_all_nonnegative": len(all_nonnegative),
        "num_event_safe": len(event_safe),
        "balanced_candidate": max(all_nonnegative, key=candidate_score_balanced) if all_nonnegative else None,
        "event_candidate": max(event_safe, key=candidate_score_event) if event_safe else None,
        "top_all_nonnegative": sorted(all_nonnegative, key=candidate_score_balanced, reverse=True)[:50],
        "top_event_safe": sorted(event_safe, key=candidate_score_event, reverse=True)[:50],
        "all_candidates": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
