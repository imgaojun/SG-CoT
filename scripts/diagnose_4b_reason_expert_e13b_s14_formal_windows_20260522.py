import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.calibrate_4b_reason_expert_e13b_route_nll_s14_20260522 import (  # noqa: E402
    FORMAL_SPLITS,
    SCORE_ROOT,
    evaluate_selected,
    load_formal_predictions,
    load_scores,
    ranked_keys,
    delta_aet,
    pct,
)


OUT_JSON = REPO / "reports/artifacts/2026-05-22_stage2_4b_reason_expert_e13b_s14_formal_window_diagnosis.json"
OUT_MD = REPO / "reports/2026-05-22_stage2_4b_reason_expert_e13b_s14_formal_window_diagnosis.md"


def formal_rows():
    rows = []
    for split in FORMAL_SPLITS:
        direct, reason = load_formal_predictions(split)
        scores = load_scores(split)
        keys = sorted(set(direct) & set(reason) & set(scores))
        ranked = ranked_keys(scores, keys)
        for start in [i / 40 for i in range(0, 36)]:
            for width in [0.025, 0.05, 0.075, 0.10, 0.125, 0.15]:
                end = start + width
                if end > 1.0:
                    continue
                s = round(len(ranked) * start)
                e = round(len(ranked) * end)
                if e <= s:
                    continue
                policy = f"s14_window_{int(start * 1000):03d}_{int(end * 1000):03d}"
                rows.append(evaluate_selected(split, policy, set(ranked[s:e]), direct, reason, keys))
    return rows


def aggregate(rows):
    out = []
    for policy in sorted({row["policy"] for row in rows}):
        parts = [row for row in rows if row["policy"] == policy]
        total = sum(row["num_examples"] for row in parts)
        pred = sum(row["pred_reason_count"] for row in parts)
        agg = {
            "split": "test",
            "policy": policy,
            "num_examples": total,
            "pred_reason_count": pred,
            "pred_reason_rate": pred / total if total else 0.0,
            "routed_minus_direct": {},
        }
        for metric in ["argument_f1", "event_f1", "trigger_f1"]:
            # The per-split delta is already aggregate metric difference; use example-weighted aggregation,
            # matching the other project replay scripts.
            agg["routed_minus_direct"][metric] = (
                sum(row["routed_minus_direct"][metric] * row["num_examples"] for row in parts) / total
                if total
                else 0.0
            )
        out.append(agg)
    return out


def key_balanced(row):
    d = row["routed_minus_direct"]
    return (
        min(d["argument_f1"], d["event_f1"], d["trigger_f1"]),
        (d["argument_f1"] + d["event_f1"]) / 2,
        d["trigger_f1"],
        -row["pred_reason_rate"],
    )


def render(payload):
    rows = payload["top_balanced"]
    lines = [
        "# E13B S14 Formal Window Diagnosis",
        "",
        "This is a diagnostic-only sweep over formal S14 route-NLL windows. It is not a dev-locked selector result.",
        "",
        "## Best Formal Windows",
        "",
        "| policy | reason rate | delta A/E/T |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| `{row['policy']}` | {pct(row['pred_reason_rate'])} | {delta_aet(row)} |")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- If the best formal windows are still negative, S14 route-NLL has little useful execution signal.",
            "- If some formal windows are positive but dev selected a different region, the problem is rank-region drift.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    rows = formal_rows()
    agg = aggregate(rows)
    positive = [
        row
        for row in agg
        if row["routed_minus_direct"]["argument_f1"] >= 0
        and row["routed_minus_direct"]["event_f1"] >= 0
        and row["routed_minus_direct"]["trigger_f1"] >= 0
    ]
    payload = {
        "score_root": SCORE_ROOT.as_posix(),
        "num_windows": len(agg),
        "num_all_nonnegative": len(positive),
        "top_balanced": sorted(agg, key=key_balanced, reverse=True)[:20],
        "all_nonnegative": sorted(positive, key=key_balanced, reverse=True)[:20],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
