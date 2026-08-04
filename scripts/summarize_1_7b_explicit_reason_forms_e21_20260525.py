import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FORMAL_ROOT = REPO / "outputs/stage2_1_7b_explicit_reason_forms/e21_formal_20260525"
DEVPICK_ROOT = REPO / "outputs/stage2_1_7b_explicit_reason_forms/e21_devpick_20260525"
OUT_JSON = REPO / "reports/artifacts/2026-05-25_stage2_1_7b_explicit_reason_forms_e21.json"
OUT_MD = REPO / "reports/2026-05-25_stage2_1_7b_explicit_reason_forms_e21.md"

VARIANTS = {
    "e21a_explicit_forms": {"variant": "e21a", "budgets": ["none", "light", "standard", "deep"]},
}
SPLITS = ["test_seen", "test_unseen"]
METRICS = ["argument_f1", "event_f1", "trigger_f1"]
EXPECTED_TAGS = {
    "none": [],
    "light": ["SCHEMA_CHECK"],
    "standard": ["ROLE_TABLE"],
    "deep": ["ARGUMENT_VERIFY"],
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(value):
    return f"{value:.4f}"


def signed(value):
    return f"{value:+.4f}"


def aet(row):
    return " / ".join(fmt(row[metric]) for metric in METRICS)


def delta(row, field="delta_vs_none"):
    return " / ".join(signed(row[field][metric]) for metric in METRICS)


def tag_present(text, tag):
    return f"<{tag}>" in text and f"</{tag}>" in text


def tag_rates(variant, budget, split):
    path = FORMAL_ROOT / variant / f"forced_{budget}" / split / "predictions.jsonl"
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    total = len(rows)
    if not total:
        return {}
    rates = {}
    for tag in ["EVENT_MENTIONS", "REASONING_BUDGET", "SCHEMA_CHECK", "ROLE_TABLE", "ARGUMENT_VERIFY", "FINAL"]:
        count = sum(1 for row in rows if tag_present(row.get("generated_payload", ""), tag))
        rates[f"{tag.lower()}_rate"] = count / total
    expected = EXPECTED_TAGS[budget] + ["EVENT_MENTIONS", "REASONING_BUDGET", "FINAL"]
    rates["expected_form_rate"] = sum(
        1 for row in rows if all(tag_present(row.get("generated_payload", ""), tag) for tag in expected)
    ) / total
    return rates


def load_summary(variant, budget, split):
    path = FORMAL_ROOT / variant / f"forced_{budget}" / split / "summary.json"
    if not path.exists():
        return None
    summary = load_json(path)
    row = {
        "variant": variant,
        "budget": budget,
        "split": split,
        "num_examples": summary.get("num_examples", 0),
        "argument_f1": summary.get("argument_f1", summary.get("final_argument_f1", 0.0)),
        "event_f1": summary.get("event_f1", summary.get("final_event_f1", 0.0)),
        "trigger_f1": summary.get("trigger_f1", summary.get("final_trigger_f1", 0.0)),
        "json_valid_rate": summary.get("json_valid_rate", summary.get("final_json_valid_rate", 0.0)),
        "route_reason_rate": summary.get("route_reason_rate", 0.0),
        "route_direct_rate": summary.get("route_direct_rate", 0.0),
        "route_unknown_rate": summary.get("route_unknown_rate", 0.0),
        "summary_path": path.as_posix(),
    }
    row.update(tag_rates(variant, budget, split))
    return row


def aggregate(rows):
    out = []
    variants = sorted({row["variant"] for row in rows})
    for variant in variants:
        budgets = sorted({row["budget"] for row in rows if row["variant"] == variant})
        for budget in budgets:
            items = [row for row in rows if row["variant"] == variant and row["budget"] == budget]
            total = sum(row["num_examples"] for row in items)
            if not total:
                continue
            agg = {"variant": variant, "budget": budget, "split": "test", "num_examples": total, "summary_path": ""}
            metric_keys = set(METRICS + ["json_valid_rate", "route_reason_rate", "route_direct_rate", "route_unknown_rate"])
            for row in items:
                metric_keys.update(key for key in row if key.endswith("_rate"))
            for metric in sorted(metric_keys):
                agg[metric] = sum(row.get(metric, 0.0) * row["num_examples"] for row in items) / total
            out.append(agg)
    return out


def add_deltas(rows):
    by_key = {(row["variant"], row["split"], row["budget"]): row for row in rows}
    for row in rows:
        baseline = by_key.get((row["variant"], row["split"], "none"))
        row["delta_vs_none"] = (
            {metric: row[metric] - baseline[metric] for metric in METRICS}
            if baseline
            else {metric: 0.0 for metric in METRICS}
        )


def load_dev_selection(variant):
    path = DEVPICK_ROOT / variant / "forced_standard_dev" / "selection_summary.json"
    if not path.exists():
        return None
    data = load_json(path)
    best = data.get("best", {})
    return {
        "variant": variant,
        "checkpoint": best.get("checkpoint_tag"),
        "checkpoint_path": best.get("checkpoint_path"),
        "metrics": best.get("summary"),
        "path": path.as_posix(),
    }


def budget_winners(rows):
    winners = []
    for variant in sorted({row["variant"] for row in rows}):
        for split in ["test", "test_seen", "test_unseen"]:
            items = [row for row in rows if row["variant"] == variant and row["split"] == split]
            if not items:
                continue
            best = max(items, key=lambda row: row["argument_f1"] + row["event_f1"] + row["trigger_f1"])
            winners.append(
                {
                    "variant": variant,
                    "split": split,
                    "budget": best["budget"],
                    "score_sum": best["argument_f1"] + best["event_f1"] + best["trigger_f1"],
                    "argument_f1": best["argument_f1"],
                    "event_f1": best["event_f1"],
                    "trigger_f1": best["trigger_f1"],
                }
            )
    return winners


def render(payload):
    lines = [
        "# 1.7B Explicit Reasoning Forms E21",
        "",
        "This report summarizes E21A, where each reasoning budget has a distinct visible intermediate form.",
        "",
    ]
    if payload["dev_selection"]:
        lines.extend(["## Dev Selection", ""])
        for item in payload["dev_selection"]:
            metrics = item.get("metrics") or {}
            metric_text = " / ".join(fmt(metrics.get(metric, 0.0)) for metric in METRICS)
            lines.append(f"- `{item['variant']}` selected `{item.get('checkpoint')}` with dev A/E/T `{metric_text}`.")
        lines.append("")
    test_rows = [row for row in payload["results"] if row["split"] == "test"]
    if test_rows:
        lines.extend(
            [
                "## Test Results",
                "",
                "| variant | budget | A/E/T | delta vs none A/E/T | json valid | expected form |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(test_rows, key=lambda item: (item["variant"], item["budget"])):
            lines.append(
                f"| `{row['variant']}` | `{row['budget']}` | {aet(row)} | {delta(row)} | "
                f"{fmt(row['json_valid_rate'])} | {fmt(row.get('expected_form_rate', 0.0))} |"
            )
        lines.append("")
    else:
        lines.extend(["## Test Results", "", "Formal summaries are not available yet.", ""])
    if payload["budget_winners"]:
        lines.extend(["## Budget Winners", "", "| variant | split | best budget | A/E/T sum |", "|---|---|---|---:|"])
        for row in payload["budget_winners"]:
            lines.append(f"| `{row['variant']}` | `{row['split']}` | `{row['budget']}` | {fmt(row['score_sum'])} |")
        lines.append("")
    lines.extend(
        [
            "## Reading",
            "",
            "- E21 tests whether differentiated forms, not just budget labels, create useful specialization.",
            "- A positive result requires both extraction gains and high expected-form adherence for the forced budget.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    rows = []
    for spec in VARIANTS.values():
        for budget in spec["budgets"]:
            for split in SPLITS:
                row = load_summary(spec["variant"], budget, split)
                if row is not None:
                    rows.append(row)
    rows.extend(aggregate(rows))
    add_deltas(rows)
    payload = {
        "formal_root": FORMAL_ROOT.as_posix(),
        "variants": VARIANTS,
        "dev_selection": [item for item in (load_dev_selection("e21a"),) if item],
        "results": rows,
        "budget_winners": budget_winners(rows),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix(), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
