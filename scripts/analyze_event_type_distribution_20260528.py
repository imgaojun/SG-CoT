import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
SCRIPT_DIR = REPO / "scripts"
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

import prepare_1_7b_explicit_reason_forms_e21_20260525 as e21  # noqa: E402


DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
FORMAL_DATA_DIR = REPO / "data/stage2_formal_datasets"
OUT_JSON = REPO / "reports/artifacts/2026-05-28_event_type_distribution.json"
OUT_MD = REPO / "reports/2026-05-28_event_type_distribution.md"
SPLITS = ["train", "dev_seen", "test_seen", "test_unseen"]


def event_types(row):
    types = []
    for event in e21.gold_json(row).get("events", []) or []:
        if isinstance(event, dict) and event.get("event_type"):
            types.append(event["event_type"])
    return sorted(set(types))


def bucket(count):
    if count < 10:
        return "ultra_tail"
    if count < 30:
        return "tail"
    if count < 100:
        return "mid"
    return "head"


def load_split(split):
    path = FORMAL_DATA_DIR / f"{DATA_PREFIX}_{split}_pos.jsonl"
    return e21.e15.load_jsonl(path)


def main():
    rows_by_split = {split: load_split(split) for split in SPLITS}
    type_counts = {split: Counter() for split in SPLITS}
    sample_counts = {split: Counter() for split in SPLITS}
    stats_by_type = defaultdict(lambda: {"argument_count": 0, "event_count": 0, "sample_count": 0})

    for split, rows in rows_by_split.items():
        for row in rows:
            types = event_types(row)
            for event in e21.gold_json(row).get("events", []) or []:
                if isinstance(event, dict) and event.get("event_type"):
                    type_counts[split][event["event_type"]] += 1
            for typ in types:
                sample_counts[split][typ] += 1
            if split == "train":
                row_stats = e21.event_stats(row)
                for typ in types:
                    stats_by_type[typ]["argument_count"] += row_stats["argument_count"]
                    stats_by_type[typ]["event_count"] += row_stats["event_count"]
                    stats_by_type[typ]["sample_count"] += 1

    all_types = sorted(set().union(*(set(c) for c in type_counts.values()), *(set(c) for c in sample_counts.values())))
    table = []
    for typ in all_types:
        train_events = type_counts["train"][typ]
        train_samples = sample_counts["train"][typ]
        st = stats_by_type[typ]
        sample_count = max(st["sample_count"], 1)
        table.append(
            {
                "event_type": typ,
                "bucket": bucket(train_samples),
                "train_event_count": train_events,
                "train_sample_count": train_samples,
                "dev_seen_sample_count": sample_counts["dev_seen"][typ],
                "test_seen_sample_count": sample_counts["test_seen"][typ],
                "test_unseen_sample_count": sample_counts["test_unseen"][typ],
                "avg_train_arguments": st["argument_count"] / sample_count if st["sample_count"] else 0.0,
                "avg_train_events": st["event_count"] / sample_count if st["sample_count"] else 0.0,
            }
        )
    table.sort(key=lambda row: (row["train_sample_count"], row["event_type"]))

    bucket_counts = Counter(row["bucket"] for row in table)
    payload = {
        "splits": {split: {"num_rows": len(rows_by_split[split])} for split in SPLITS},
        "bucket_counts": dict(bucket_counts),
        "table": table,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Event Type Distribution",
        "",
        "## Summary",
        "",
        f"- splits: " + ", ".join(f"`{k}`={v['num_rows']}" for k, v in payload["splits"].items()),
        f"- bucket counts: " + ", ".join(f"`{k}`={v}" for k, v in sorted(bucket_counts.items())),
        "",
        "## Type Counts",
        "",
        "| event_type | bucket | train samples | train events | dev | test seen | test unseen | avg args |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| `{row['event_type']}` | `{row['bucket']}` | {row['train_sample_count']} | {row['train_event_count']} | "
            f"{row['dev_seen_sample_count']} | {row['test_seen_sample_count']} | {row['test_unseen_sample_count']} | "
            f"{row['avg_train_arguments']:.2f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix(), "num_types": len(table)}, indent=2))


if __name__ == "__main__":
    main()
