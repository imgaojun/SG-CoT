import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
M02_FORMAL = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_aet_stable_router_m02_next_selectors_formal.json"
M05_FORMAL = REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_positive_retention_router_m05_formal.json"
M06_FORMAL = REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_m06_combo_selectors_formal.json"
OUT_JSON = REPO / "reports/artifacts/2026-05-21_stage2_modular_dualexpert_aet_m06_main_result_comparison.json"
OUT_MD = REPO / "reports/2026-05-21_stage2_modular_dualexpert_aet_m06_main_result_comparison.md"


ROWS = [
    {
        "name": "m02_positive_retention",
        "source": M02_FORMAL,
        "policy": "positive_retention_candidate_checkpoint-50_global_rank425_500",
        "reading": "conservative all-positive aggregate, strongest Event-preserving baseline",
    },
    {
        "name": "m05_lowbudget",
        "source": M05_FORMAL,
        "policy": "early_stable_candidate_checkpoint-100_rank050_100",
        "reading": "lower budget, stronger Argument/Trigger, weaker Event",
    },
    {
        "name": "m06_union",
        "source": M06_FORMAL,
        "policy": "union_m02_m05",
        "reading": "strongest aggregate result by combining complementary m02/m05 selections",
    },
]


def load_test_row(path, policy):
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data["results"]:
        if row["split"] == "test" and row["policy"] == policy:
            return row
    raise KeyError(f"missing policy={policy} split=test in {path}")


def signed(value):
    return f"{value:+.4f}"


def render(payload):
    lines = [
        "# A/E/T M06 Main Result Comparison",
        "",
        "This comparison uses formal `test` rows only. Policies were selected without formal labels.",
        "",
        "| candidate | reason rate | delta Argument | delta Event | delta Trigger | reading |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in payload["rows"]:
        d = item["routed_minus_direct"]
        lines.append(
            "| {name} | {rate:.1%} | {a} | {e} | {t} | {reading} |".format(
                name=item["name"],
                rate=item["pred_reason_rate"],
                a=signed(d["argument_f1"]),
                e=signed(d["event_f1"]),
                t=signed(d["trigger_f1"]),
                reading=item["reading"],
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Use `m06_union` as the strongest aggregate main-result candidate: A/E/T `+0.0111 / +0.0059 / +0.0134` at `12.2%` reason rate.",
            "- Keep `m02_positive_retention` as the conservative low-budget all-positive candidate: A/E/T `+0.0051 / +0.0050 / +0.0026` at `7.4%` reason rate.",
            "- Treat `m05_lowbudget` as an ablation showing that supervised positive-retention distillation mostly contributes Argument/Trigger gains.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    rows = []
    for item in ROWS:
        row = load_test_row(item["source"], item["policy"])
        rows.append(
            {
                "name": item["name"],
                "policy": item["policy"],
                "source": item["source"].as_posix(),
                "reading": item["reading"],
                "pred_reason_rate": row["pred_reason_rate"],
                "routed": row["routed"],
                "direct": row["direct"],
                "routed_minus_direct": row["routed_minus_direct"],
            }
        )
    payload = {"rows": rows}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
