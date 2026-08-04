import json
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/stage2_1_7b_reason_budget/e20_formal_20260524/e20b"
OUT_JSON = REPO / "reports/artifacts/2026-05-25_stage2_1_7b_reason_budget_e20b_budget_win_profile.json"
OUT_MD = REPO / "reports/2026-05-25_stage2_1_7b_reason_budget_e20b_budget_win_profile.md"
BUDGETS = ["none", "light", "standard", "deep"]
SPLITS = ["test_seen", "test_unseen"]
METRICS = ["argument_f1", "event_f1", "trigger_f1"]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prediction_key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("sample_key") or meta.get("doc_id") or row.get("input", "")[:200]


def score(row):
    return sum(float(row.get(metric, 0.0)) for metric in METRICS)


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
        best_score = max(scores.values())
        winners = [budget for budget in BUDGETS if abs(scores[budget] - best_score) < 1e-12]
        deep_gain = scores["deep"] - scores["none"]
        standard_gain = scores["standard"] - scores["none"]
        rows.append(
            {
                "key": key,
                "split": split,
                "scores": scores,
                "winner": winners[0] if len(winners) == 1 else "tie",
                "winners": winners,
                "deep_gain": deep_gain,
                "standard_gain": standard_gain,
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
        "deep_positive_rate": mean([1.0 if row["deep_gain"] > 1e-12 else 0.0 for row in rows]),
        "deep_negative_rate": mean([1.0 if row["deep_gain"] < -1e-12 else 0.0 for row in rows]),
        "deep_gain_mean": mean([row["deep_gain"] for row in rows]),
        "standard_gain_mean": mean([row["standard_gain"] for row in rows]),
    }
    for budget in BUDGETS:
        out[f"{budget}_score_mean"] = mean([row["scores"][budget] for row in rows])
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
            "deep_gain": row["deep_gain"],
            "standard_gain": row["standard_gain"],
            "scores": row["scores"],
            "stats": row["stats"],
            "text_preview": row["text_preview"],
        }
        for row in ordered[:n]
    ]


def fmt(value):
    return f"{value:.4f}"


def pct(value):
    return f"{100 * value:.1f}%"


def render(payload):
    lines = [
        "# E20B Budget-Win Profile",
        "",
        "This profiles forced `none/light/standard/deep` outputs per example for the E20B multi-budget model.",
        "",
        "## Overall",
        "",
        "| group | count | winners | deep gain mean | deep + rate | deep - rate |",
        "|---|---:|---|---:|---:|---:|",
    ]
    overall = payload["overall"]
    for name in ["all", "test_seen", "test_unseen"]:
        row = overall[name]
        winners = ", ".join(f"{k}:{v}" for k, v in sorted(row["winner_counts"].items()))
        lines.append(
            f"| `{name}` | {row['count']} | {winners} | {fmt(row['deep_gain_mean'])} | "
            f"{pct(row['deep_positive_rate'])} | {pct(row['deep_negative_rate'])} |"
        )
    lines.extend(["", "## Complexity", "", "| complexity | count | winners | deep gain mean | deep + rate |", "|---|---:|---|---:|---:|"])
    for name, row in payload["by_complexity"].items():
        winners = ", ".join(f"{k}:{v}" for k, v in sorted(row["winner_counts"].items()))
        lines.append(f"| `{name}` | {row['count']} | {winners} | {fmt(row['deep_gain_mean'])} | {pct(row['deep_positive_rate'])} |")
    lines.extend(["", "## Event Count", "", "| event count | count | winners | deep gain mean |", "|---|---:|---|---:|"])
    for name, row in payload["by_event_count"].items():
        winners = ", ".join(f"{k}:{v}" for k, v in sorted(row["winner_counts"].items()))
        lines.append(f"| `{name}` | {row['count']} | {winners} | {fmt(row['deep_gain_mean'])} |")
    lines.extend(["", "## Reading", ""])
    lines.append("- `deep` is useful when its per-example score sum beats `none`; this is common on seen but much less stable on unseen.")
    lines.append("- `none` remains the robustness budget for unseen; a selector should learn when not to spend reasoning budget.")
    lines.append("- Use the top-case JSON artifact for qualitative inspection of deep-positive and deep-negative examples.")
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
        "top_deep_positive": top_cases(rows, "deep_gain", 10, True),
        "top_deep_negative": top_cases(rows, "deep_gain", 10, False),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix(), "num_examples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
