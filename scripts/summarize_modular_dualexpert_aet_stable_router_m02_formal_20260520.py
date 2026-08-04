import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 import (  # noqa: E402
    DIRECT_ROOT,
    REASON_ROOT,
    aggregate_test,
    evaluate,
    render_table,
)


import scripts.summarize_modular_dualexpert_aet_router_m01_formal_20260520 as base  # noqa: E402


BRANCH = "aet_stable_router_m02_routecls_noauxwarm_lr2e6_save50"
SCORE_ROOT = REPO / "outputs/stage2_modular_dualexpert/aet_stable_router_m02_20260520/formal_route_likelihood"
DEV_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_dev.json"
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_formal.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_formal.md"
SPLITS = ["test_seen", "test_unseen"]


def load_policies():
    dev = json.loads(DEV_JSON.read_text(encoding="utf-8"))
    policies = []
    seen = set()
    for key in ["balanced_candidate", "early_stable_candidate"]:
        row = dev[key]
        name = f"{key}_{row['checkpoint']}_rank{int(row['rank_window']['start_pct'] * 1000):03d}_{int(row['rank_window']['end_pct'] * 1000):03d}"
        ident = (row["checkpoint"], row["rank_window"]["start_pct"], row["rank_window"]["end_pct"])
        if ident in seen:
            continue
        seen.add(ident)
        policies.append(
            {
                "name": name,
                "branch": BRANCH,
                "checkpoint": row["checkpoint"],
                "start_pct": row["rank_window"]["start_pct"],
                "end_pct": row["rank_window"]["end_pct"],
                "source": key,
            }
        )
    return policies


def render_report(payload):
    rows = sorted(payload["results"], key=lambda row: (row["policy"], row["split"]))
    lines = [
        "# A/E/T Stable Router M02 Formal Replay",
        "",
        "This report applies dev-locked stable-window policies to formal route-NLL scores. No formal labels are used for policy selection.",
        "",
        "## Results",
        "",
        render_table(rows),
        "",
        "## Reading",
        "",
    ]
    for row in rows:
        if row["split"] == "test":
            d = row["routed_minus_direct"]
            lines.append(
                f"- `{row['policy']}` on `test`: reason rate `{row['pred_reason_rate']:.1%}`, "
                f"A/E/T delta `{d['argument_f1']:+.4f}/{d['event_f1']:+.4f}/{d['trigger_f1']:+.4f}`."
            )
    return "\n".join(lines) + "\n"


def main():
    policies = load_policies()
    base.SCORE_ROOT = SCORE_ROOT
    base.POLICIES = policies
    split_rows = []
    for policy in policies:
        for split in SPLITS:
            split_rows.append(evaluate(policy, split))
    rows = split_rows + aggregate_test(split_rows)
    payload = {
        "score_root": SCORE_ROOT.as_posix(),
        "direct_root": DIRECT_ROOT.as_posix(),
        "reason_root": REASON_ROOT.as_posix(),
        "policies": policies,
        "results": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
