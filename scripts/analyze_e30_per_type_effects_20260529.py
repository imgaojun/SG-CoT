import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


FORMAL_ROOT = ROOT / "outputs/stage2_1_7b_paired_augmentation/e27_formal_20260527"
DIST_PATH = ROOT / "reports/artifacts/2026-05-28_event_type_distribution.json"
REPORT_PATH = ROOT / "reports/2026-05-29_e30_per_type_analysis.md"
ARTIFACT_PATH = ROOT / "reports/artifacts/2026-05-29_e30_per_type_analysis.json"

SYSTEMS = [
    ("e28a_standard", "E28A natural step standard", "e28a", "standard"),
    ("e30a_none", "E30A tail-type direct none", "e30a", "none"),
    ("e30a_standard", "E30A tail-type direct standard", "e30a", "standard"),
    ("e30b_none", "E30B tail-type natural step none", "e30b", "none"),
    ("e30b_standard", "E30B tail-type natural step standard", "e30b", "standard"),
    ("e30c_none", "E30C minimal type none", "e30c", "none"),
    ("e30c_standard", "E30C minimal type standard", "e30c", "standard"),
    ("e31a_none", "E31A type-complexity direct none", "e31a", "none"),
    ("e31a_standard", "E31A type-complexity direct standard", "e31a", "standard"),
    ("e31b_none", "E31B type-complexity natural step none", "e31b", "none"),
    ("e31b_standard", "E31B type-complexity natural step standard", "e31b", "standard"),
]

SPLITS = ["test_seen", "test_unseen"]
METRICS = ["trigger", "argument", "event"]
BUCKET_ORDER = ["head", "mid", "tail", "ultra_tail", "unknown"]


def normalize_events(events_payload):
    events = events_payload.get("events", []) if isinstance(events_payload, dict) else []
    trigger_set = set()
    argument_set = set()
    event_set = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        trigger = event.get("trigger", {})
        if not isinstance(trigger, dict):
            trigger = {}
        trig = (event_type, trigger.get("start"), trigger.get("end"))
        trigger_set.add(trig)
        args = []
        raw_arguments = event.get("arguments", [])
        if not isinstance(raw_arguments, list):
            raw_arguments = []
        for arg in raw_arguments:
            if not isinstance(arg, dict):
                continue
            arg_tuple = (
                event_type,
                trigger.get("start"),
                trigger.get("end"),
                arg.get("role"),
                arg.get("start"),
                arg.get("end"),
            )
            argument_set.add(arg_tuple)
            args.append((arg.get("role"), arg.get("start"), arg.get("end")))
        sorted_args = tuple(
            sorted(
                args,
                key=lambda item: (
                    item[0] or "",
                    -1 if item[1] is None else item[1],
                    -1 if item[2] is None else item[2],
                ),
            )
        )
        event_set.add((event_type, trigger.get("start"), trigger.get("end"), sorted_args))
    return trigger_set, argument_set, event_set


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def empty_counts():
    return {"tp": 0, "pred": 0, "gold": 0}


def add_counts(acc, pred_set, gold_set):
    acc["tp"] += len(pred_set & gold_set)
    acc["pred"] += len(pred_set)
    acc["gold"] += len(gold_set)


def prf_from_counts(c):
    p = c["tp"] / c["pred"] if c["pred"] else 0.0
    r = c["tp"] / c["gold"] if c["gold"] else 0.0
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {"p": p, "r": r, "f1": f1, **c}


def event_type_of(item):
    return item[0] if item else None


def grouped_by_type(items):
    grouped = defaultdict(set)
    for item in items:
        grouped[event_type_of(item)].add(item)
    return grouped


def split_sets(payload):
    trigger, argument, event = normalize_events(payload or {"events": []})
    return {"trigger": trigger, "argument": argument, "event": event}


def build_type_info():
    dist = json.loads(DIST_PATH.read_text(encoding="utf-8"))
    info = {}
    for row in dist["table"]:
        info[row["event_type"]] = row
    return dist, info


def load_system_rows(variant, budget, split):
    path = FORMAL_ROOT / variant / f"forced_{budget}" / split / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_jsonl(path)


def compute_system_metrics(type_info):
    all_types = set(type_info)
    systems = {}
    for sys_id, label, variant, budget in SYSTEMS:
        type_counts = {
            split: {metric: defaultdict(empty_counts) for metric in METRICS}
            for split in SPLITS
        }
        row_metrics = {split: [] for split in SPLITS}
        valid_json = {split: {"valid": 0, "total": 0} for split in SPLITS}

        for split in SPLITS:
            rows = load_system_rows(variant, budget, split)
            valid_json[split]["total"] += len(rows)
            for idx, row in enumerate(rows):
                if row.get("valid_json") or row.get("valid_final_json"):
                    valid_json[split]["valid"] += 1
                pred_payload = row.get("final_predicted", row.get("predicted"))
                gold_payload = row.get("gold")
                pred = split_sets(pred_payload)
                gold = split_sets(gold_payload)
                row_entry = {
                    "index": idx,
                    "split": split,
                    "gold_types": sorted({event_type_of(x) for x in gold["event"]}),
                    "trigger_f1": row.get("trigger_f1", 0.0),
                    "argument_f1": row.get("argument_f1", 0.0),
                    "event_f1": row.get("event_f1", 0.0),
                }
                row_metrics[split].append(row_entry)
                for metric in METRICS:
                    pred_by_type = grouped_by_type(pred[metric])
                    gold_by_type = grouped_by_type(gold[metric])
                    touched_types = set(pred_by_type) | set(gold_by_type)
                    all_types.update(touched_types)
                    for event_type in touched_types:
                        add_counts(
                            type_counts[split][metric][event_type],
                            pred_by_type.get(event_type, set()),
                            gold_by_type.get(event_type, set()),
                        )

        systems[sys_id] = {
            "label": label,
            "variant": variant,
            "budget": budget,
            "type_counts": type_counts,
            "row_metrics": row_metrics,
            "valid_json": valid_json,
        }
    return systems, sorted(all_types)


def merge_counts(counts_list):
    merged = empty_counts()
    for counts in counts_list:
        merged["tp"] += counts["tp"]
        merged["pred"] += counts["pred"]
        merged["gold"] += counts["gold"]
    return merged


def compute_views(systems, all_types, type_info):
    per_type = {}
    bucket_metrics = {}
    aggregate_metrics = {}

    for sys_id, sys_data in systems.items():
        per_type[sys_id] = {}
        bucket_metrics[sys_id] = {}
        aggregate_metrics[sys_id] = {}
        for split_group, splits in [("test_seen", ["test_seen"]), ("test_unseen", ["test_unseen"]), ("test", SPLITS)]:
            per_type[sys_id][split_group] = {}
            bucket_metrics[sys_id][split_group] = {}
            aggregate_metrics[sys_id][split_group] = {}
            for metric in METRICS:
                aggregate_counts = merge_counts(
                    sys_data["type_counts"][split][metric].get(t, empty_counts())
                    for split in splits
                    for t in all_types
                )
                aggregate_metrics[sys_id][split_group][metric] = prf_from_counts(aggregate_counts)

                for bucket in BUCKET_ORDER:
                    bucket_types = [
                        t
                        for t in all_types
                        if type_info.get(t, {}).get("bucket", "unknown") == bucket
                    ]
                    bucket_counts = merge_counts(
                        sys_data["type_counts"][split][metric].get(t, empty_counts())
                        for split in splits
                        for t in bucket_types
                    )
                    bucket_metrics[sys_id][split_group].setdefault(bucket, {})[metric] = prf_from_counts(bucket_counts)

            for event_type in all_types:
                per_type[sys_id][split_group][event_type] = {
                    "bucket": type_info.get(event_type, {}).get("bucket", "unknown"),
                    "train_sample_count": type_info.get(event_type, {}).get("train_sample_count", 0),
                }
                for metric in METRICS:
                    counts = merge_counts(
                        sys_data["type_counts"][split][metric].get(event_type, empty_counts())
                        for split in splits
                    )
                    per_type[sys_id][split_group][event_type][metric] = prf_from_counts(counts)

    return per_type, bucket_metrics, aggregate_metrics


def type_deltas(per_type, base_id, target_id, split_group, metric):
    rows = []
    for event_type, target in per_type[target_id][split_group].items():
        base = per_type[base_id][split_group][event_type]
        gold = target[metric]["gold"] or base[metric]["gold"]
        if gold <= 0:
            continue
        rows.append(
            {
                "event_type": event_type,
                "bucket": target["bucket"],
                "train_sample_count": target["train_sample_count"],
                "gold": gold,
                "base_f1": base[metric]["f1"],
                "target_f1": target[metric]["f1"],
                "delta": target[metric]["f1"] - base[metric]["f1"],
            }
        )
    return sorted(rows, key=lambda x: x["delta"], reverse=True)


def row_bucket(gold_types, type_info):
    if not gold_types:
        return "unknown"
    order = {bucket: idx for idx, bucket in enumerate(BUCKET_ORDER)}
    buckets = [type_info.get(t, {}).get("bucket", "unknown") for t in gold_types]
    return sorted(buckets, key=lambda b: order.get(b, 99), reverse=True)[0]


def compute_row_pair_deltas(systems, type_info, base_id="e28a_standard", target_id="e30b_standard"):
    result = {}
    for split in SPLITS:
        base_rows = systems[base_id]["row_metrics"][split]
        target_rows = systems[target_id]["row_metrics"][split]
        by_bucket = defaultdict(lambda: {"n": 0, "argument_delta": 0.0, "event_delta": 0.0, "trigger_delta": 0.0})
        for base, target in zip(base_rows, target_rows):
            bucket = row_bucket(target["gold_types"] or base["gold_types"], type_info)
            item = by_bucket[bucket]
            item["n"] += 1
            for metric in METRICS:
                item[f"{metric}_delta"] += target[f"{metric}_f1"] - base[f"{metric}_f1"]
        result[split] = {}
        for bucket, vals in by_bucket.items():
            n = vals["n"]
            result[split][bucket] = {
                "n": n,
                "argument_delta": vals["argument_delta"] / n if n else 0.0,
                "event_delta": vals["event_delta"] / n if n else 0.0,
                "trigger_delta": vals["trigger_delta"] / n if n else 0.0,
            }
    return result


def fmt_metric(m):
    return f"{m['argument']['f1']:.4f} / {m['event']['f1']:.4f} / {m['trigger']['f1']:.4f}"


def fmt_delta(v):
    return f"{v:+.4f}"


def make_markdown(dist, systems, bucket_metrics, aggregate_metrics, per_type, row_deltas):
    lines = []
    lines.append("# E30 Per-Type Effect Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- compares E28A natural-step standard against E30 tail-type balanced variants.")
    lines.append("- metrics are micro P/R/F1 recomputed from normalized predicted and gold event structures.")
    lines.append("- main format below is `Argument / Event / Trigger` F1.")
    lines.append("- event-type buckets come from `reports/artifacts/2026-05-28_event_type_distribution.json`.")
    lines.append("")
    lines.append("## Aggregate Recomputed Micro F1")
    lines.append("")
    lines.append("| system | test | test_seen | test_unseen |")
    lines.append("|---|---:|---:|---:|")
    for sys_id, label, _, _ in SYSTEMS:
        lines.append(
            f"| `{sys_id}` {label} | {fmt_metric(aggregate_metrics[sys_id]['test'])} | "
            f"{fmt_metric(aggregate_metrics[sys_id]['test_seen'])} | {fmt_metric(aggregate_metrics[sys_id]['test_unseen'])} |"
        )
    lines.append("")
    lines.append("## Bucket-Level Test F1")
    lines.append("")
    lines.append("| bucket | E28A std | E30B std | E31B std | E31A std | E30B-E28A Event | E31B-E30B Event |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for bucket in BUCKET_ORDER:
        e28 = bucket_metrics["e28a_standard"]["test"].get(bucket, {})
        e30b = bucket_metrics["e30b_standard"]["test"].get(bucket, {})
        e31b = bucket_metrics["e31b_standard"]["test"].get(bucket, {})
        e31a = bucket_metrics["e31a_standard"]["test"].get(bucket, {})
        if not e28 or (e28["event"]["gold"] == 0 and e30b["event"]["gold"] == 0):
            continue
        lines.append(
            f"| `{bucket}` | {fmt_metric(e28)} | {fmt_metric(e30b)} | {fmt_metric(e31b)} | {fmt_metric(e31a)} | "
            f"{fmt_delta(e30b['event']['f1'] - e28['event']['f1'])} | {fmt_delta(e31b['event']['f1'] - e30b['event']['f1'])} |"
        )
    lines.append("")
    lines.append("## Seen vs Unseen Bucket Event F1")
    lines.append("")
    lines.append("| split | bucket | E28A std | E30B std | E31B std | E31A std |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for split_group in ["test_seen", "test_unseen"]:
        for bucket in BUCKET_ORDER:
            vals = []
            has_gold = False
            for sys_id in ["e28a_standard", "e30b_standard", "e31b_standard", "e31a_standard"]:
                metric = bucket_metrics[sys_id][split_group].get(bucket, {}).get("event")
                vals.append(metric)
                has_gold = has_gold or (metric and metric["gold"] > 0)
            if not has_gold:
                continue
            lines.append(
                f"| `{split_group}` | `{bucket}` | "
                f"{vals[0]['f1']:.4f} | {vals[1]['f1']:.4f} | {vals[2]['f1']:.4f} | {vals[3]['f1']:.4f} |"
            )
    lines.append("")
    lines.append("## E30B Standard vs E28A Standard: Type Event F1 Deltas")
    lines.append("")
    for title, rows in [
        ("Largest Gains", type_deltas(per_type, "e28a_standard", "e30b_standard", "test", "event")[:12]),
        ("Largest Losses", list(reversed(type_deltas(per_type, "e28a_standard", "e30b_standard", "test", "event")[-12:]))),
    ]:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| event_type | bucket | train samples | test gold | E28A | E30B | delta |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| `{row['event_type']}` | `{row['bucket']}` | {row['train_sample_count']} | {row['gold']} | "
                f"{row['base_f1']:.4f} | {row['target_f1']:.4f} | {fmt_delta(row['delta'])} |"
            )
        lines.append("")
    lines.append("## Paired Row-Level Mean Delta")
    lines.append("")
    lines.append("| comparison | split | bucket | n | Argument delta | Event delta | Trigger delta |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for comparison, by_split in row_deltas.items():
        for split, buckets in by_split.items():
            for bucket in BUCKET_ORDER:
                if bucket not in buckets:
                    continue
                vals = buckets[bucket]
                lines.append(
                    f"| `{comparison}` | `{split}` | `{bucket}` | {vals['n']} | {fmt_delta(vals['argument_delta'])} | "
                    f"{fmt_delta(vals['event_delta'])} | {fmt_delta(vals['trigger_delta'])} |"
                )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    e28_event = aggregate_metrics["e28a_standard"]["test"]["event"]["f1"]
    e30b_event = aggregate_metrics["e30b_standard"]["test"]["event"]["f1"]
    e30b_arg = aggregate_metrics["e30b_standard"]["test"]["argument"]["f1"]
    e28_arg = aggregate_metrics["e28a_standard"]["test"]["argument"]["f1"]
    head_delta = bucket_metrics["e30b_standard"]["test"]["head"]["event"]["f1"] - bucket_metrics["e28a_standard"]["test"]["head"]["event"]["f1"]
    tail_delta = bucket_metrics["e30b_standard"]["test"]["tail"]["event"]["f1"] - bucket_metrics["e28a_standard"]["test"]["tail"]["event"]["f1"]
    ultra_delta = bucket_metrics["e30b_standard"]["test"]["ultra_tail"]["event"]["f1"] - bucket_metrics["e28a_standard"]["test"]["ultra_tail"]["event"]["f1"]
    lines.append(f"- E30B standard is close to E28A standard on aggregate: Event {e30b_event:.4f} vs {e28_event:.4f}, Argument {e30b_arg:.4f} vs {e28_arg:.4f}.")
    lines.append(f"- The tail-type balancing effect is not a clean tail win: Event delta is head {head_delta:+.4f}, tail {tail_delta:+.4f}, ultra-tail {ultra_delta:+.4f}.")
    lines.append("- Because many `test_unseen` event types have zero train examples, train-set type oversampling cannot directly teach those labels; improvements there mostly indicate transfer or reduced head-type overfitting.")
    lines.append("- Minimal type-step E30C remains weaker than natural-step E30B, so the useful signal is not just event-type listing; the natural-language decomposition appears to carry role/argument constraints.")
    lines.append("")
    lines.append("## Next")
    lines.append("")
    lines.append("- Use this report to choose the next augmentation policy: keep event-type balancing only if it improves specific low-frequency seen types without damaging head types.")
    lines.append("- Prefer a complexity-aware augmentation pass that balances event type plus argument/role complexity, instead of frequency-only oversampling.")
    lines.append("- Add per-type evaluation to future formal summaries so aggregate improvements cannot hide head/tail tradeoffs.")
    lines.append("")
    return "\n".join(lines)


def main():
    dist, type_info = build_type_info()
    systems, all_types = compute_system_metrics(type_info)
    per_type, bucket_metrics, aggregate_metrics = compute_views(systems, all_types, type_info)
    row_deltas = {
        "e30b_standard_minus_e28a_standard": compute_row_pair_deltas(systems, type_info, "e28a_standard", "e30b_standard"),
        "e31b_standard_minus_e30b_standard": compute_row_pair_deltas(systems, type_info, "e30b_standard", "e31b_standard"),
    }

    artifact = {
        "systems": {k: {kk: vv for kk, vv in v.items() if kk not in {"type_counts", "row_metrics"}} for k, v in systems.items()},
        "bucket_counts": dist["bucket_counts"],
        "aggregate_metrics": aggregate_metrics,
        "bucket_metrics": bucket_metrics,
        "per_type": per_type,
        "row_deltas": row_deltas,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(
        make_markdown(dist, systems, bucket_metrics, aggregate_metrics, per_type, row_deltas),
        encoding="utf-8",
    )
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
