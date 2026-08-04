import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_next_selector_main_result_comparison.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_aet_next_selector_main_result_comparison.md"

SOURCES = {
    "m02_stable": REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_formal.json",
    "m04a": REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_rankstable_router_m04a_formal.json",
    "m04b": REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_rankstable_router_m04b_formal.json",
    "m02_next": REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_next_selectors_formal.json",
}

POLICIES = [
    ("m02 early-stable", "m02_stable", "early_stable_candidate_checkpoint-100_rank325_400"),
    ("m04a balanced", "m04a", "balanced_candidate_checkpoint-250_rank425_500"),
    ("m04b balanced", "m04b", "balanced_candidate_checkpoint-237_rank425_500"),
    (
        "m02 positive-retention",
        "m02_next",
        "positive_retention_candidate_checkpoint-50_global_rank425_500",
    ),
]


def signed(value):
    return f"{value:+.4f}"


def fmt(value):
    return f"{value:.4f}"


def load_rows():
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in SOURCES.items()}
    rows = []
    for display, source, policy in POLICIES:
        candidates = [
            row
            for row in loaded[source]["results"]
            if row["policy"] == policy and row["split"] == "test"
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one test row for {source}:{policy}, got {len(candidates)}")
        row = dict(candidates[0])
        row["display_policy"] = display
        rows.append(row)
    return rows


def render(rows):
    lines = [
        "# A/E/T Next-Selector Main Result Comparison",
        "",
        "This table compares formal `test` results for recent low-budget adaptive-routing policies. Deltas are against direct-only.",
        "",
        "| policy | reason rate | routed A/E/T | delta A/E/T | selected gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        routed = row["routed"]
        delta = row["routed_minus_direct"]
        lines.append(
            f"| {row['display_policy']} | {row['pred_reason_rate']:.1%} | "
            f"{fmt(routed['argument_f1'])} / {fmt(routed['event_f1'])} / {fmt(routed['trigger_f1'])} | "
            f"{signed(delta['argument_f1'])} / {signed(delta['event_f1'])} / {signed(delta['trigger_f1'])} | "
            f"{signed(row['selected_reason_avg_score_gain'])} |"
        )
    best = next(row for row in rows if row["display_policy"] == "m02 positive-retention")
    d = best["routed_minus_direct"]
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `m02 positive-retention` is the current main adaptive-routing candidate.",
            f"- It uses a `7.4%` reason budget and improves formal `test` A/E/T by `{signed(d['argument_f1'])} / {signed(d['event_f1'])} / {signed(d['trigger_f1'])}`.",
            "- Compared with m02 early-stable, it turns Trigger positive while improving Argument/Event.",
            "- m04a/m04b remain useful negative/partial results for the hard-negative weighting line.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    rows = load_rows()
    payload = {
        "sources": {name: path.as_posix() for name, path in SOURCES.items()},
        "results": rows,
        "main_candidate": "m02 positive-retention",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(rows), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
