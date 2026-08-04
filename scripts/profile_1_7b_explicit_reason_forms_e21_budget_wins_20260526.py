import json
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/stage2_1_7b_explicit_reason_forms/e21_formal_20260525/e21a"
OUT_JSON = REPO / "reports/artifacts/2026-05-26_stage2_1_7b_explicit_reason_forms_e21a_budget_win_profile.json"
OUT_MD = REPO / "reports/2026-05-26_stage2_1_7b_explicit_reason_forms_e21a_budget_win_profile.md"
BUDGETS = ["none", "light", "standard", "deep"]
SPLITS = ["test_seen", "test_unseen"]
METRICS = ["argument_f1", "event_f1", "trigger_f1"]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prediction_key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("sample_key") or meta.get("doc_id") or row.get("input", "")[:200]


def score(row, metrics=METRICS):
    return sum(float(row.get(metric, 0.0)) for metric in metrics)


def metric_deltas(per_budget, budget):
    return {metric: float(per_budget[budget].get(metric, 0.0)) - float(per_budget["none"].get(metric, 0.0)) for metric in METRICS}


def tag_present(text, tag):
    return f"<{tag}>" in text and f"</{tag}>" in text


def expected_form_ok(row, budget):
    tags = {
        "none": ["EVENT_MENTIONS", "REASONING_BUDGET", "FINAL"],
        "light": ["EVENT_MENTIONS", "REASONING_BUDGET", "SCHEMA_CHECK", "FINAL"],
        "standard": ["EVENT_MENTIONS", "REASONING_BUDGET", "ROLE_TABLE", "FINAL"],
        "deep": ["EVENT_MENTIONS", "REASONING_BUDGET", "ARGUMENT_VERIFY", "FINAL"],
    }[budget]
    text = row.get("generated_payload", "")
    return all(tag_present(text, tag) for tag in tags)


def event_stats(row):
    gold = row.get("gold") or {}
    events = [event for event in gold.get("events", []) if isinstance(event, dict)]
    event_types = []
    arg_count = 0
    role_set = set()
    for event in events:
        event_type = event.get("event_type")
        if event_type:
            event_types.append(event_type)
        args = event.get("arguments") or []
        arg_count += len(args)
        for arg in args:
            if isinstance(arg, dict) and arg.get("role"):
                role_set.add((event_type, arg.get("role")))
    return {
        "event_count": len(events),
        "argument_count": arg_count,
        "role_count": len(role_set),
        "event_types": event_types,
        "has_rare_type": any(
            event_type
            in {
                "Business:Declare-Bankruptcy",
                "Business:End-Org",
                "Business:Merge-Org",
                "Justice:Acquit",
                "Justice:Appeal",
                "Justice:Execute",
                "Justice:Extradite",
                "Justice:Fine",
                "Justice:Release-Parole",
            }
            for event_type in event_types
        ),
    }


def complexity_bucket(stats):
    if stats["event_count"] >= 3 or stats["argument_count"] >= 6:
        return "high"
    if stats["event_count"] >= 2 or stats["argument_count"] >= 3:
        return "medium"
    return "low"


def load_split(split):
    by_budget = {}
    for budget in BUDGETS:
        path = ROOT / f"forced_{budget}" / split / "predictions.jsonl"
        by_budget[budget] = {prediction_key(row): row for row in load_jsonl(path)}
    common = sorted(set.intersection(*(set(rows) for rows in by_budget.values())))
    rows = []
    for key in common:
        per_budget = {budget: by_budget[budget][key] for budget in BUDGETS}
        stats = event_stats(per_budget["none"])
        scores = {budget: score(row) for budget, row in per_budget.items()}
        ae_scores = {budget: score(row, ["argument_f1", "event_f1"]) for budget, row in per_budget.items()}
        best_score = max(scores.values())
        winners = [budget for budget in BUDGETS if abs(scores[budget] - best_score) < 1e-12]
        ae_best_score = max(ae_scores.values())
        ae_winners = [budget for budget in BUDGETS if abs(ae_scores[budget] - ae_best_score) < 1e-12]
        rows.append(
            {
                "key": key,
                "split": split,
                "scores": scores,
                "ae_scores": ae_scores,
                "winner": winners[0] if len(winners) == 1 else "tie",
                "winners": winners,
                "ae_winner": ae_winners[0] if len(ae_winners) == 1 else "tie",
                "ae_winners": ae_winners,
                "standard_gain": scores["standard"] - scores["none"],
                "light_gain": scores["light"] - scores["none"],
                "deep_gain": scores["deep"] - scores["none"],
                "standard_metric_deltas": metric_deltas(per_budget, "standard"),
                "light_metric_deltas": metric_deltas(per_budget, "light"),
                "deep_metric_deltas": metric_deltas(per_budget, "deep"),
                "form_ok": {budget: expected_form_ok(per_budget[budget], budget) for budget in BUDGETS},
                "stats": stats,
                "complexity": complexity_bucket(stats),
                "metrics": {
                    budget: {metric: per_budget[budget].get(metric, 0.0) for metric in METRICS}
                    for budget in BUDGETS
                },
                "text_preview": (per_budget["none"].get("input") or "").split("\n\nTokens:")[0][:300],
            }
        )
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize_group(rows):
    out = {
        "count": len(rows),
        "winner_counts": dict(Counter(row["winner"] for row in rows)),
        "ae_winner_counts": dict(Counter(row["ae_winner"] for row in rows)),
    }
    for budget in ["light", "standard", "deep"]:
        gains = [row[f"{budget}_gain"] for row in rows]
        out[f"{budget}_gain_mean"] = mean(gains)
        out[f"{budget}_positive_rate"] = mean([1.0 if gain > 1e-12 else 0.0 for gain in gains])
        out[f"{budget}_negative_rate"] = mean([1.0 if gain < -1e-12 else 0.0 for gain in gains])
        for metric in METRICS:
            out[f"{budget}_{metric}_delta_mean"] = mean([row[f"{budget}_metric_deltas"][metric] for row in rows])
    for budget in BUDGETS:
        out[f"{budget}_score_mean"] = mean([row["scores"][budget] for row in rows])
        out[f"{budget}_form_ok_rate"] = mean([1.0 if row["form_ok"][budget] else 0.0 for row in rows])
    return out


def grouped(rows, key_fn):
    buckets = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)
    return {name: summarize_group(items) for name, items in sorted(buckets.items())}


def top_cases(rows, gain_key, n=8, reverse=True):
    ordered = sorted(rows, key=lambda row: row[gain_key], reverse=reverse)
    return [
        {
            "key": row["key"],
            "split": row["split"],
            "winner": row["winner"],
            "ae_winner": row["ae_winner"],
            "standard_gain": row["standard_gain"],
            "light_gain": row["light_gain"],
            "deep_gain": row["deep_gain"],
            "scores": row["scores"],
            "metric_deltas": {
                "light": row["light_metric_deltas"],
                "standard": row["standard_metric_deltas"],
                "deep": row["deep_metric_deltas"],
            },
            "stats": row["stats"],
            "text_preview": row["text_preview"],
        }
        for row in ordered[:n]
    ]


def fmt(value):
    return f"{value:.4f}"


def pct(value):
    return f"{100 * value:.1f}%"


def winner_text(row, field="winner_counts"):
    return ", ".join(f"{k}:{v}" for k, v in sorted(row[field].items()))


def render(payload):
    lines = [
        "# E21A Explicit-Form Budget-Win Profile",
        "",
        "This profiles forced `none/light/standard/deep` outputs per example for the E21A explicit-form model.",
        "",
        "## Overall",
        "",
        "| group | count | A/E/T winners | A+E winners | standard gain mean | standard + rate | standard - rate | deep gain mean |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    overall = payload["overall"]
    for name in ["all", "test_seen", "test_unseen"]:
        row = overall[name]
        lines.append(
            f"| `{name}` | {row['count']} | {winner_text(row)} | {winner_text(row, 'ae_winner_counts')} | "
            f"{fmt(row['standard_gain_mean'])} | {pct(row['standard_positive_rate'])} | "
            f"{pct(row['standard_negative_rate'])} | {fmt(row['deep_gain_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Complexity",
            "",
            "| complexity | count | winners | standard gain mean | deep gain mean |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for name, row in payload["by_complexity"].items():
        lines.append(
            f"| `{name}` | {row['count']} | {winner_text(row)} | "
            f"{fmt(row['standard_gain_mean'])} | {fmt(row['deep_gain_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Metric Deltas",
            "",
            "| group | standard delta A/E/T | light delta A/E/T | deep delta A/E/T |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ["all", "test_seen", "test_unseen"]:
        row = overall[name]
        standard = " / ".join(fmt(row[f"standard_{metric}_delta_mean"]) for metric in METRICS)
        light = " / ".join(fmt(row[f"light_{metric}_delta_mean"]) for metric in METRICS)
        deep = " / ".join(fmt(row[f"deep_{metric}_delta_mean"]) for metric in METRICS)
        lines.append(f"| `{name}` | {standard} | {light} | {deep} |")
    lines.extend(["", "## Reading", ""])
    lines.append("- `standard` is the best aggregate budget, but this profile checks whether it wins broadly or only through a few large repairs.")
    lines.append("- `deep` should be treated as an unseen A/E repair candidate only if the Trigger loss can be guarded or corrected.")
    lines.append("- Use the JSON artifact for qualitative inspection of standard-positive, standard-negative, deep-positive, and deep-negative cases.")
    return "\n".join(lines) + "\n"


def main():
    rows = []
    for split in SPLITS:
        rows.extend(load_split(split))
    payload = {
        "root": ROOT.as_posix(),
        "budgets": BUDGETS,
        "overall": {
            "all": summarize_group(rows),
            "test_seen": summarize_group([row for row in rows if row["split"] == "test_seen"]),
            "test_unseen": summarize_group([row for row in rows if row["split"] == "test_unseen"]),
        },
        "by_complexity": grouped(rows, lambda row: row["complexity"]),
        "by_split_complexity": grouped(rows, lambda row: f"{row['split']}:{row['complexity']}"),
        "by_event_count": grouped(rows, lambda row: str(row["stats"]["event_count"])),
        "by_argument_count": grouped(rows, lambda row: str(row["stats"]["argument_count"])),
        "by_rare_type": grouped(rows, lambda row: "rare" if row["stats"]["has_rare_type"] else "nonrare"),
        "top_standard_positive": top_cases(rows, "standard_gain", 10, True),
        "top_standard_negative": top_cases(rows, "standard_gain", 10, False),
        "top_deep_positive": top_cases(rows, "deep_gain", 10, True),
        "top_deep_negative": top_cases(rows, "deep_gain", 10, False),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix(), "num_examples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
