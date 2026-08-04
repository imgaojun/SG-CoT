import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
M02_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_utility_router_m02_main_result_candidate.json"
OLD_JSON = REPO / "reports/artifacts/2026-05-16_stage2_adaptive_outcome_helpful_sharedbase_balrouteaux_nll_formal_checkpoint-1930_test.json"
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_utility_router_m02_main_result_comparison.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_utility_router_m02_main_result_comparison.md"


def utility(delta):
    return delta["argument_f1"] + delta["event_f1"] + 0.25 * delta["trigger_f1"]


def signed(value):
    return f"{value:+.4f}"


def pct(value):
    return f"{100 * value:.1f}%"


def render(rows):
    lines = [
        "# M02 Main Result Comparison",
        "",
        "This table compares within-report routed-minus-direct deltas. The old baseline uses its original formal `test` report; the m02 result uses the dev-locked rank-window formal replay aggregated from `test_seen` and `test_unseen`.",
        "",
        "| method | policy | reason rate | delta Utility | delta A/E/T | note |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        d = row["delta"]
        lines.append(
            f"| {row['method']} | `{row['policy']}` | {pct(row['reason_rate'])} | "
            f"{signed(row['utility_delta'])} | {signed(d['argument_f1'])}/{signed(d['event_f1'])}/{signed(d['trigger_f1'])} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- The current m02 rank-window candidate has the largest weighted utility gain among these comparable routed-minus-direct entries.",
            "- Its weakness is still event: event is slightly negative, but argument and trigger gains compensate under the project utility metric.",
            "- This is enough for a short-term main result table, with the caveat that unseen transfer remains weak.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    m02 = json.loads(M02_JSON.read_text(encoding="utf-8"))
    old = json.loads(OLD_JSON.read_text(encoding="utf-8"))
    m02_main = m02["main_result"]
    rows = [
        {
            "method": "m02 supervised rank-window",
            "policy": m02_main["policy"],
            "reason_rate": m02_main["pred_reason_rate"],
            "delta": m02_main["routed_minus_direct"],
            "utility_delta": m02_main["routed_minus_direct"]["utility"],
            "note": "dev-locked checkpoint-150 rank 10%-30%",
        }
    ]
    old_rows = []
    for router in old["routers"]:
        delta = router["routed_delta_vs_direct"]
        old_rows.append(
            {
                "method": "previous balrouteaux NLL",
                "policy": router["name"],
                "reason_rate": router["pred_reason_rate"],
                "delta": delta,
                "utility_delta": utility(delta),
                "note": "original 2026-05-16 formal test report",
            }
        )
    old_best = max(old_rows, key=lambda row: row["utility_delta"])
    rows.append(old_best)
    rows.extend(
        sorted(
            [row for row in old_rows if row["policy"] != old_best["policy"]],
            key=lambda row: row["utility_delta"],
            reverse=True,
        )[:2]
    )
    rows = sorted(rows, key=lambda row: row["utility_delta"], reverse=True)
    payload = {
        "m02_source": M02_JSON.as_posix(),
        "old_source": OLD_JSON.as_posix(),
        "utility_metric": "argument_f1 + event_f1 + 0.25 * trigger_f1",
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(rows), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
