import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FORMAL_ROOT = REPO / "outputs/stage2_1_7b_reason_lightweight/e24_e25_formal_20260526"
DEVPICK_ROOT = REPO / "outputs/stage2_1_7b_reason_lightweight/e24_e25_devpick_20260526"
OUT_JSON = REPO / "reports/artifacts/2026-05-26_stage2_1_7b_reason_lightweight_e24_e25.json"
OUT_MD = REPO / "reports/2026-05-26_stage2_1_7b_reason_lightweight_e24_e25.md"

VARIANTS = {
    "e24a": {"label": "lite_standard", "budgets": ["none", "light", "standard", "deep"]},
    "e25a": {"label": "final_first_weakreason", "budgets": ["none", "light", "standard", "deep"]},
}
SPLITS = ["test_seen", "test_unseen"]
METRICS = ["argument_f1", "event_f1", "trigger_f1"]
BASELINES = {
    "e21a_standard": {"argument_f1": 0.3341606725353904, "event_f1": 0.2515586369988176, "trigger_f1": 0.5146726862302483},
    "e21a_none": {"argument_f1": 0.3168514642748772, "event_f1": 0.24383555975481505, "trigger_f1": 0.5169285651673087},
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


def delta(row, field):
    return " / ".join(signed(row[field][metric]) for metric in METRICS)


def tag_present(text, tag):
    return f"<{tag}>" in text and f"</{tag}>" in text


def expected_tags(budget):
    tags = ["EVENT_MENTIONS", "REASONING_BUDGET", "FINAL"]
    if budget != "none":
        tags.append("REASON_HINT")
    return tags


def tag_rates(variant, budget, split):
    path = FORMAL_ROOT / variant / f"forced_{budget}" / split / "predictions.jsonl"
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    total = len(rows)
    if not total:
        return {}
    all_tags = ["EVENT_MENTIONS", "REASONING_BUDGET", "REASON_HINT", "FINAL"]
    rates = {}
    for tag in all_tags:
        rates[f"{tag.lower()}_rate"] = sum(1 for row in rows if tag_present(row.get("generated_payload", ""), tag)) / total
    expected = expected_tags(budget)
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
    for variant in sorted({row["variant"] for row in rows}):
        for budget in sorted({row["budget"] for row in rows if row["variant"] == variant}):
            items = [row for row in rows if row["variant"] == variant and row["budget"] == budget]
            total = sum(row["num_examples"] for row in items)
            if not total:
                continue
            agg = {"variant": variant, "budget": budget, "split": "test", "num_examples": total, "summary_path": ""}
            keys = set(METRICS + ["json_valid_rate", "route_reason_rate", "route_direct_rate", "route_unknown_rate"])
            for row in items:
                keys.update(key for key in row if key.endswith("_rate"))
            for key in keys:
                agg[key] = sum(row.get(key, 0.0) * row["num_examples"] for row in items) / total
            out.append(agg)
    return out


def add_deltas(rows):
    by_key = {(row["variant"], row["split"], row["budget"]): row for row in rows}
    for row in rows:
        none = by_key.get((row["variant"], row["split"], "none"))
        row["delta_vs_none"] = {metric: row[metric] - none[metric] for metric in METRICS} if none else {metric: 0.0 for metric in METRICS}
        row["delta_vs_e21a_standard"] = {metric: row[metric] - BASELINES["e21a_standard"][metric] for metric in METRICS}
        row["delta_vs_e21a_none"] = {metric: row[metric] - BASELINES["e21a_none"][metric] for metric in METRICS}


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
        "# 1.7B Lightweight Reasoning E24/E25",
        "",
        "This report summarizes lightweight standard and final-first weak-reason variants against E21A.",
        "",
    ]
    if payload["dev_selection"]:
        lines.extend(["## Dev Selection", ""])
        for item in payload["dev_selection"]:
            metrics = item.get("metrics") or {}
            metric_text = " / ".join(fmt(metrics.get(metric, 0.0)) for metric in METRICS)
            lines.append(f"- `{item['variant']}` selected `{item.get('checkpoint')}` with dev A/E/T `{metric_text}`.")
        lines.append("")
    test_rows = [row for row in payload["results"] if row["split"] == "test" and row["budget"] == "standard"]
    if test_rows:
        lines.extend(
            [
                "## Standard Budget Results",
                "",
                "| variant | label | A/E/T | delta vs none A/E/T | delta vs E21A standard A/E/T | expected form |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(test_rows, key=lambda item: item["variant"]):
            label = VARIANTS[row["variant"]]["label"]
            lines.append(
                f"| `{row['variant']}` | `{label}` | {aet(row)} | {delta(row, 'delta_vs_none')} | "
                f"{delta(row, 'delta_vs_e21a_standard')} | {fmt(row.get('expected_form_rate', 0.0))} |"
            )
        lines.append("")
    else:
        lines.extend(["## Standard Budget Results", "", "Formal summaries are not available yet.", ""])
    if payload["budget_winners"]:
        lines.extend(["## Budget Winners", "", "| variant | split | best budget | A/E/T sum |", "|---|---|---|---:|"])
        for row in payload["budget_winners"]:
            lines.append(f"| `{row['variant']}` | `{row['split']}` | `{row['budget']}` | {fmt(row['score_sum'])} |")
        lines.append("")
    lines.extend(
        [
            "## Reading",
            "",
            "- E24A is successful if standard keeps E21A's Argument/Event gains while reducing Trigger loss.",
            "- E25A is successful if final-first weak reasoning beats E21A standard or at least improves Trigger without losing Argument/Event.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    rows = []
    for variant, spec in VARIANTS.items():
        for budget in spec["budgets"]:
            for split in SPLITS:
                row = load_summary(variant, budget, split)
                if row is not None:
                    rows.append(row)
    rows.extend(aggregate(rows))
    add_deltas(rows)
    payload = {
        "formal_root": FORMAL_ROOT.as_posix(),
        "variants": VARIANTS,
        "baseline": BASELINES,
        "dev_selection": [item for item in (load_dev_selection("e24a"), load_dev_selection("e25a")) if item],
        "results": rows,
        "budget_winners": budget_winners(rows),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix(), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
