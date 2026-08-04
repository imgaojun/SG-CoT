#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.summarize_multibudget_fourclass_router_m09_formal_20260521 as base  # noqa: E402


base.BRANCH = "multibudget_retention_router_m11_routecls_noauxwarm_lr2e6_save50"
base.SCORE_ROOT = REPO / "outputs/stage2_multibudget/formal_route_likelihood_20260521" / base.BRANCH
base.OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_multibudget_retention_router_m11_formal.json"
base.OUT_MD = REPO / "reports/2026-05-21_stage2_multibudget_retention_router_m11_formal.md"
base.POLICIES = [
    {
        "name": "m11_retention_dev_locked",
        "checkpoint": "checkpoint-150",
        "start_pct": 0.275,
        "end_pct": 0.425,
        "dev_delta_aet": "+0.0147/+0.0125/+0.0234",
    }
]


def render(payload):
    lines = [
        "# Multibudget Retention Router M11 Formal Replay",
        "",
        "Formal replay uses dev-locked policies only. No formal labels were used to choose checkpoint or rank window.",
        "",
        "| policy | split | reason rate | route counts | delta A/E/T | routed A/E/T | selected mean score gain |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        counts = row["route_counts"]
        routed = row["routed"]
        lines.append(
            "| {name} | {split} | {rate:.1%} | d/l/m/f={d}/{l}/{m}/{f} | {delta} | {a:.4f} / {e:.4f} / {t:.4f} | {gain:+.4f} |".format(
                name=row["name"],
                split=row["split"],
                rate=row["reason_rate"],
                d=counts["direct"],
                l=counts["reason_light"],
                m=counts["reason_mid"],
                f=counts["reason_full"],
                delta=base.fmt_delta(row["routed_delta_vs_direct"]),
                a=routed["argument_f1"],
                e=routed["event_f1"],
                t=routed["trigger_f1"],
                gain=row["selected_mean_score_gain"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    results = []
    for policy in base.POLICIES:
        for split in base.SPLITS:
            results.append(base.evaluate(policy, split))
    payload = {
        "branch": base.BRANCH,
        "score_root": base.SCORE_ROOT.as_posix(),
        "policies": base.POLICIES,
        "results": results,
    }
    base.write_json(base.OUT_JSON, payload)
    base.write_text(base.OUT_MD, render(payload))
    print(json.dumps({"output_json": base.OUT_JSON.as_posix(), "output_md": base.OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
