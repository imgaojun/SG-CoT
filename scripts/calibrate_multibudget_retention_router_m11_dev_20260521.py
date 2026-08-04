#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.calibrate_multibudget_fourclass_router_m09_dev_20260521 as base  # noqa: E402


base.BRANCH = "multibudget_retention_router_m11_routecls_noauxwarm_lr2e6_save50"
base.SCORE_ROOT = base.REPO / "outputs/stage2_multibudget/route_likelihood_20260521" / base.BRANCH
base.OUT_JSON = base.REPO / "reports/artifacts/2026-05-21_stage2_multibudget_retention_router_m11_dev.json"
base.OUT_MD = base.REPO / "reports/2026-05-21_stage2_multibudget_retention_router_m11_dev.md"


def render_report(payload):
    lines = [
        "# Multibudget Retention Router M11 Dev Sweep",
        "",
        "This sweep ranks examples by best non-direct route-choice NLL advantage and executes the per-example best non-direct budget inside each selected rank window.",
        "",
        "## Locked Candidates",
        "",
        base.render_table([payload["balanced_candidate"], payload["event_arg_candidate"]]),
        "",
        "## Top All-Nonnegative",
        "",
        base.render_table(payload["top_all_nonnegative"][:20]),
        "",
        "## Top Event/Argument",
        "",
        base.render_table(payload["top_event_arg"][:20]),
        "",
    ]
    return "\n".join(lines)


def main():
    direct_rows = base.load_prediction_map(base.DIRECT_DEV)
    light_rows = base.load_prediction_map(base.LIGHT_DEV)
    mid_rows = base.load_prediction_map(base.MID_DEV)
    full_rows = base.load_prediction_map(base.FULL_DEV)
    common_keys = sorted(set(direct_rows) & set(light_rows) & set(mid_rows) & set(full_rows))
    rows = base.sweep(direct_rows, light_rows, mid_rows, full_rows, common_keys)
    all_nonnegative = [row for row in rows if base.is_all_nonnegative(row)]
    if not all_nonnegative:
        raise RuntimeError("no all-nonnegative M11 dev candidate")
    robust = [row for row in all_nonnegative if row["fold_min_aet"] >= 0.0]
    selection_pool = robust or all_nonnegative
    balanced = max(selection_pool, key=base.balanced_score)
    event_arg = max(selection_pool, key=base.event_arg_score)
    payload = {
        "selection_metric": "dev-only multiroute NLL rank-window replay; formal not used",
        "branch": base.BRANCH,
        "score_root": base.SCORE_ROOT.as_posix(),
        "num_candidates": len(rows),
        "num_all_nonnegative": len(all_nonnegative),
        "num_fold_nonnegative": len(robust),
        "balanced_candidate": balanced,
        "event_arg_candidate": event_arg,
        "top_all_nonnegative": sorted(all_nonnegative, key=base.balanced_score, reverse=True)[:50],
        "top_event_arg": sorted(all_nonnegative, key=base.event_arg_score, reverse=True)[:50],
        "all_candidates": rows,
    }
    base.write_json(base.OUT_JSON, payload)
    base.write_text(base.OUT_MD, render_report(payload))
    print(json.dumps({"output_json": base.OUT_JSON.as_posix(), "output_md": base.OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
