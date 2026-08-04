import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.calibrate_modular_dualexpert_aet_stable_router_m02_dev_20260520 as cal


cal.BRANCH = "aet_rankstable_router_m04a_routecls_noauxwarm_lr2e6_save50"
cal.SCORE_ROOT = cal.REPO / "outputs/stage2_modular_dualexpert/aet_rankstable_router_m04a_20260520/route_likelihood" / cal.BRANCH
cal.OUT_JSON = cal.REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_rankstable_router_m04a_dev.json"
cal.OUT_MD = cal.REPO / "reports/2026-05-20_stage2_modular_dualexpert_aet_rankstable_router_m04a_dev.md"
BASE_RENDER_TABLE = cal.render_table


def low_budget_score(row):
    d = row["routed_delta_vs_direct"]
    ckpt = int(row["checkpoint"].split("-", 1)[1])
    return (
        row["fold_min_aet"],
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        d["event_f1"],
        -ckpt,
        -row["rank_window"]["end_pct"],
    )


def balanced_score(row):
    d = row["routed_delta_vs_direct"]
    ckpt = int(row["checkpoint"].split("-", 1)[1])
    return (
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        row["fold_min_aet"],
        d["event_f1"],
        -ckpt,
        -row["rank_window"]["end_pct"],
    )


def render_report(payload):
    lines = [
        "# A/E/T Rank-Stable Router M04A Dev Sweep",
        "",
        "This sweep selects low-budget windows using raw A/E/T deltas plus dev-only fold stability.",
        "",
        "## Locked Candidates",
        "",
        BASE_RENDER_TABLE([payload["balanced_candidate"], payload["early_stable_candidate"]]),
        "",
        "## Top Low-Budget Stable",
        "",
        BASE_RENDER_TABLE(payload["top_low_budget_stable"][:20]),
        "",
        "## Top All-Nonnegative",
        "",
        BASE_RENDER_TABLE(payload["top_all_nonnegative"][:20]),
        "",
    ]
    return "\n".join(lines)


def main():
    direct_rows = cal.load_prediction_map(cal.DIRECT_DEV)
    reason_rows = cal.load_prediction_map(cal.REASON_DEV)
    common_keys = sorted(set(direct_rows) & set(reason_rows))
    rows = cal.sweep(direct_rows, reason_rows, common_keys)
    all_nonnegative = [
        row
        for row in rows
        if cal.is_all_nonnegative(row) and 0.05 <= row["reason_rate"] <= 0.12
    ]
    if not all_nonnegative:
        all_nonnegative = [
            row
            for row in rows
            if cal.is_all_nonnegative(row) and 0.05 <= row["reason_rate"] <= 0.175
        ]
    if not all_nonnegative:
        raise RuntimeError("no all-nonnegative dev candidate in low/expanded target reason-rate range")
    balanced = max(all_nonnegative, key=balanced_score)
    stable = max(all_nonnegative, key=low_budget_score)
    payload = {
        "selection_metric": "low-budget A/E/T plus dev-only fold stability; formal not used",
        "branch": cal.BRANCH,
        "score_root": cal.SCORE_ROOT.as_posix(),
        "target_reason_rate": [0.05, 0.12],
        "fallback_reason_rate": [0.05, 0.175],
        "num_candidates": len(rows),
        "num_all_nonnegative_target_rate": len(all_nonnegative),
        "balanced_candidate": balanced,
        "early_stable_candidate": stable,
        "top_low_budget_stable": sorted(all_nonnegative, key=low_budget_score, reverse=True)[:50],
        "top_all_nonnegative": sorted(all_nonnegative, key=balanced_score, reverse=True)[:50],
        "all_candidates": rows,
    }
    cal.OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    cal.OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    cal.OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": cal.OUT_JSON.as_posix(), "output_md": cal.OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
