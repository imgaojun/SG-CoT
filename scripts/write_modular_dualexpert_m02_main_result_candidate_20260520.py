import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
INPUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_utility_router_m02_rank_window_formal.json"
OUT_JSON = REPO / "reports/artifacts/2026-05-20_stage2_modular_dualexpert_utility_router_m02_main_result_candidate.json"
OUT_MD = REPO / "reports/2026-05-20_stage2_modular_dualexpert_utility_router_m02_main_result_candidate.md"


def utility(row):
    return row["argument_f1"] + row["event_f1"] + 0.25 * row["trigger_f1"]


def fmt(value):
    return f"{value:.4f}"


def signed(value):
    return f"{value:+.4f}"


def pct(value):
    return f"{100 * value:.1f}%"


def enrich(row):
    out = dict(row)
    for key in ["direct", "forced_reason_all", "routed"]:
        out[key] = dict(row[key])
        out[key]["utility"] = utility(row[key])
    out["routed_minus_direct"] = dict(row["routed_minus_direct"])
    out["routed_minus_direct"]["utility"] = out["routed"]["utility"] - out["direct"]["utility"]
    out["routed_minus_reason_all"] = dict(row["routed_minus_reason_all"])
    out["routed_minus_reason_all"]["utility"] = out["routed"]["utility"] - out["forced_reason_all"]["utility"]
    return out


def render_table(rows):
    lines = [
        "| policy | split | reason rate | direct U/A/E/T | routed U/A/E/T | delta U/A/E/T |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        direct = row["direct"]
        routed = row["routed"]
        delta = row["routed_minus_direct"]
        lines.append(
            f"| `{row['policy']}` | `{row['split']}` | {pct(row['pred_reason_rate'])} | "
            f"{fmt(direct['utility'])}/{fmt(direct['argument_f1'])}/{fmt(direct['event_f1'])}/{fmt(direct['trigger_f1'])} | "
            f"{fmt(routed['utility'])}/{fmt(routed['argument_f1'])}/{fmt(routed['event_f1'])}/{fmt(routed['trigger_f1'])} | "
            f"{signed(delta['utility'])}/{signed(delta['argument_f1'])}/{signed(delta['event_f1'])}/{signed(delta['trigger_f1'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    main = payload["main_result"]
    rows = sorted(payload["rows"], key=lambda row: (row["policy"], row["split"]))
    lines = [
        "# M02 Main Result Candidate",
        "",
        "This report reframes the dev-locked rank-window formal replay as a paper-facing main result candidate using the project utility metric `Argument + Event + 0.25 * Trigger`.",
        "",
        "## Main Candidate",
        "",
        f"- policy: `{main['policy']}`",
        f"- formal `test` reason rate: `{main['pred_reason_rate']:.1%}`",
        f"- direct utility: `{main['direct']['utility']:.4f}`",
        f"- routed utility: `{main['routed']['utility']:.4f}`",
        f"- utility delta: `{main['routed_minus_direct']['utility']:+.4f}`",
        f"- A/E/T delta: `{main['routed_minus_direct']['argument_f1']:+.4f} / {main['routed_minus_direct']['event_f1']:+.4f} / {main['routed_minus_direct']['trigger_f1']:+.4f}`",
        "",
        "## Formal Table",
        "",
        render_table(rows),
        "",
        "## Reading",
        "",
        "- `event_checkpoint150_rank10_30` is the current main-result candidate: it improves formal aggregated utility, argument, and trigger, while event is slightly negative.",
        "- The gain is concentrated on `test_seen`; `test_unseen` remains the main weakness.",
        "- `robust_checkpoint100_rank20_40` is not a main-result candidate because its formal utility delta is negative despite trigger improvement.",
        "- This is a credible short-term main result if framed as calibrated adaptive routing improving weighted extraction utility, not as an event-only improvement.",
        "",
        "## Next",
        "",
        "- Improve unseen transfer or add an unseen-safe guard.",
        "- Compare this main candidate against the previous best formal NLL top-budget baseline in one consolidated table.",
        "",
    ]
    return "\n".join(lines)


def main():
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    rows = [enrich(row) for row in data["results"]]
    test_rows = [row for row in rows if row["split"] == "test"]
    main_result = max(test_rows, key=lambda row: row["routed_minus_direct"]["utility"])
    payload = {
        "source_json": INPUT_JSON.as_posix(),
        "utility_metric": "argument_f1 + event_f1 + 0.25 * trigger_f1",
        "main_result": main_result,
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"output_json": OUT_JSON.as_posix(), "output_md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
