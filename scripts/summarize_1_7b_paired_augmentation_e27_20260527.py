import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FORMAL_ROOT = REPO / "outputs/stage2_1_7b_paired_augmentation/e27_formal_20260527"
DEVPICK_ROOT = REPO / "outputs/stage2_1_7b_paired_augmentation/e27_devpick_20260527"
OUT_JSON = REPO / "reports/artifacts/2026-05-27_stage2_1_7b_paired_augmentation_e27.json"
OUT_MD = REPO / "reports/2026-05-27_stage2_1_7b_paired_augmentation_e27.md"

VARIANTS = {
    "e27a": {"label": "none_aug", "dev_budget": "none", "budgets": ["none", "standard"]},
    "e27b": {"label": "span_reason_aug", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e27c": {"label": "paired_none_standard_aug", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e27d": {"label": "balanced_none_aug", "dev_budget": "none", "budgets": ["none", "standard"]},
    "e27e": {"label": "hardneg_none_aug", "dev_budget": "none", "budgets": ["none", "standard"]},
    "e28a": {"label": "balanced_natural_step_reason", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e29a": {"label": "hardneg_natural_step_reason", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e29b": {"label": "balanced_compact_step_reason", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e29c": {"label": "hardneg_compact_step_reason", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e30a": {"label": "tail_type_balanced_none_aug", "dev_budget": "none", "budgets": ["none", "standard"]},
    "e30b": {"label": "tail_type_balanced_natural_step", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e30c": {"label": "tail_type_balanced_minimal_type_step", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e31a": {"label": "type_complexity_none_aug", "dev_budget": "none", "budgets": ["none", "standard"]},
    "e31b": {"label": "type_complexity_natural_step", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e32a": {"label": "trigger_preserving_tail_natural_step", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e32b": {"label": "trigger_role_ground_natural_step", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e32c": {"label": "trigger_role_ground_direct_anchor", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e35a": {"label": "boundary_contrast_direct", "dev_budget": "none", "budgets": ["none", "standard"]},
    "e35b": {"label": "boundary_check_reason", "dev_budget": "standard", "budgets": ["none", "standard"]},
    "e35c": {"label": "boundary_check_direct_anchor", "dev_budget": "standard", "budgets": ["none", "standard"]},
}
SPLITS = ["test_seen", "test_unseen"]
METRICS = ["argument_f1", "event_f1", "trigger_f1"]
BASELINES = {
    "e21a_standard": {"argument_f1": 0.3341606725353904, "event_f1": 0.2515586369988176, "trigger_f1": 0.5146726862302483},
    "e21a_none": {"argument_f1": 0.3168514642748772, "event_f1": 0.24383555975481505, "trigger_f1": 0.5169285651673087},
    "e26a_none": {"argument_f1": 0.34144448478534256, "event_f1": 0.2520298111720234, "trigger_f1": 0.5491096062202157},
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
    return " / ".join(fmt(row[m]) for m in METRICS)


def delta(row, field):
    return " / ".join(signed(row[field][m]) for m in METRICS)


def tag_present(text, tag):
    return f"<{tag}>" in text and f"</{tag}>" in text


def expected_tags(variant, budget):
    tags = ["EVENT_MENTIONS", "REASONING_BUDGET", "FINAL"]
    if variant in {"e28a", "e29a", "e29b", "e29c", "e30b", "e30c", "e31b", "e32a", "e32b", "e32c", "e35b", "e35c"} and budget == "standard":
        tags.append("STEP_REASONING")
    elif budget == "standard":
        tags.append("SPAN_HINT")
    return tags


def tag_rates(variant, budget, split):
    path = FORMAL_ROOT / variant / f"forced_{budget}" / split / "predictions.jsonl"
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    total = len(rows)
    if not total:
        return {}
    all_tags = ["EVENT_MENTIONS", "REASONING_BUDGET", "SPAN_HINT", "STEP_REASONING", "FINAL"]
    rates = {}
    for tag in all_tags:
        rates[f"{tag.lower()}_rate"] = sum(1 for row in rows if tag_present(row.get("generated_payload", ""), tag)) / total
    expected = expected_tags(variant, budget)
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
            keys = set(METRICS + ["json_valid_rate"])
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
        row["delta_vs_none"] = {m: row[m] - none[m] for m in METRICS} if none else {m: 0.0 for m in METRICS}
        for name, baseline in BASELINES.items():
            row[f"delta_vs_{name}"] = {m: row[m] - baseline[m] for m in METRICS}


def load_dev_selection(variant):
    budget = VARIANTS[variant]["dev_budget"]
    path = DEVPICK_ROOT / variant / f"forced_{budget}_dev" / "selection_summary.json"
    if not path.exists():
        return None
    data = load_json(path)
    best = data.get("best", {})
    return {
        "variant": variant,
        "budget": budget,
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
        "# 1.7B Paired Data Augmentation E27",
        "",
        "This report compares augmentation-only and paired reasoning augmentation under the same test sets.",
        "",
    ]
    if payload["dev_selection"]:
        lines.extend(["## Dev Selection", ""])
        for item in payload["dev_selection"]:
            metrics = item.get("metrics") or {}
            metric_text = " / ".join(fmt(metrics.get(m, 0.0)) for m in METRICS)
            lines.append(f"- `{item['variant']}` forced `{item['budget']}` selected `{item.get('checkpoint')}` with dev A/E/T `{metric_text}`.")
        lines.append("")
    test_rows = [row for row in payload["results"] if row["split"] == "test"]
    if test_rows:
        lines.extend(
            [
                "## Aggregate Results",
                "",
                "| variant | label | budget | A/E/T | delta vs own none | delta vs E21A standard | delta vs E26A none | expected form |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(test_rows, key=lambda item: (item["variant"], item["budget"])):
            label = VARIANTS[row["variant"]]["label"]
            lines.append(
                f"| `{row['variant']}` | `{label}` | `{row['budget']}` | {aet(row)} | "
                f"{delta(row, 'delta_vs_none')} | {delta(row, 'delta_vs_e21a_standard')} | "
                f"{delta(row, 'delta_vs_e26a_none')} | {fmt(row.get('expected_form_rate', 0.0))} |"
            )
        lines.append("")
    else:
        lines.extend(["## Aggregate Results", "", "Formal summaries are not available yet.", ""])
    if payload["budget_winners"]:
        lines.extend(["## Budget Winners", "", "| variant | split | best budget | A/E/T sum |", "|---|---|---|---:|"])
        for row in payload["budget_winners"]:
            lines.append(f"| `{row['variant']}` | `{row['split']}` | `{row['budget']}` | {fmt(row['score_sum'])} |")
        lines.append("")
    lines.extend(
        [
            "## Reading",
            "",
            "- E27A measures whether targeted augmentation helps the direct/none path.",
            "- E27B uses the same augmentation input distribution but forces span-hint reasoning.",
            "- E27C tests whether paired none/standard targets create a useful budget split inside one model.",
            "- E27D controls total train-row count and tests whether a balanced high-risk augmentation subset helps direct extraction.",
            "- E27E keeps the E27D row budget but concentrates augmentation on hard negatives.",
            "- E28A keeps the E27D balanced input distribution and replaces direct targets with natural step-by-step reasoning.",
            "- E30B adds event-type frequency balancing to natural step reasoning.",
            "- E31A/B extend E30 by selecting augmentation with event-type rarity plus argument/role complexity.",
            "- E32A/B/C return to E30B-style tail balancing with trigger-preserving augmentation and more controlled step transmission.",
            "- E35A/B/C target E34's argument-boundary errors with nearby-span perturbations and compact exact-span boundary checks.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    rows = []
    for variant, spec in VARIANTS.items():
        for budget in spec["budgets"]:
            for split in SPLITS:
                row = load_summary(variant, budget, split)
                if row:
                    rows.append(row)
    rows.extend(aggregate(rows))
    add_deltas(rows)
    payload = {
        "formal_root": FORMAL_ROOT.as_posix(),
        "variants": VARIANTS,
        "baseline": BASELINES,
        "dev_selection": [item for item in (load_dev_selection(v) for v in VARIANTS) if item],
        "results": rows,
        "budget_winners": budget_winners(rows),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix(), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
