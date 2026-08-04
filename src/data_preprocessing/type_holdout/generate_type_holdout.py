import argparse
import json
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl_and_json(path_stem, rows):
    jsonl_path = path_stem.with_suffix(".jsonl")
    json_path = path_stem.with_suffix(".json")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.copyfile(jsonl_path, json_path)


def collect_all_types(splits):
    all_types = set()
    for rows in splits.values():
        for row in rows:
            for ev in row["event_mentions"]:
                all_types.add(ev["event_type"])
    return sorted(all_types)


def row_types(row):
    return {ev["event_type"] for ev in row["event_mentions"]}


def filter_rows_excluding_unseen(rows, unseen_types):
    unseen_set = set(unseen_types)
    kept = []
    removed_rows = 0
    removed_mentions = 0
    removed_type_counter = Counter()
    for row in rows:
        present = row_types(row)
        hit = present & unseen_set
        if hit:
            removed_rows += 1
            for ev in row["event_mentions"]:
                if ev["event_type"] in unseen_set:
                    removed_mentions += 1
                    removed_type_counter[ev["event_type"]] += 1
        else:
            kept.append(row)
    return kept, {
        "removed_rows": removed_rows,
        "removed_mentions": removed_mentions,
        "removed_type_counter": dict(sorted(removed_type_counter.items())),
    }


def filter_event_mentions(rows, target_types):
    target = set(target_types)
    filtered_rows = []
    kept_mentions = 0
    kept_type_counter = Counter()
    nonempty_rows = 0
    for row in rows:
        new_row = deepcopy(row)
        new_events = [ev for ev in row["event_mentions"] if ev["event_type"] in target]
        new_row["event_mentions"] = new_events
        filtered_rows.append(new_row)
        if new_events:
            nonempty_rows += 1
            for ev in new_events:
                kept_mentions += 1
                kept_type_counter[ev["event_type"]] += 1
    return filtered_rows, {
        "rows": len(filtered_rows),
        "rows_with_target_mentions": nonempty_rows,
        "mentions": kept_mentions,
        "type_counter": dict(sorted(kept_type_counter.items())),
    }


def split_stats(rows):
    docs = set()
    event_types = Counter()
    role_types = Counter()
    total_events = 0
    total_args = 0
    nonempty = 0
    for row in rows:
        docs.add(row["doc_id"])
        if row["event_mentions"]:
            nonempty += 1
        for ev in row["event_mentions"]:
            total_events += 1
            event_types[ev["event_type"]] += 1
            for arg in ev["arguments"]:
                total_args += 1
                role_types[arg["role"]] += 1
    return {
        "instances": len(rows),
        "docs": len(docs),
        "instances_with_events": nonempty,
        "event_types": len(event_types),
        "event_mentions": total_events,
        "role_types": len(role_types),
        "arguments": total_args,
    }


def generate_dataset_protocol(input_root, output_root, dataset, protocol, unseen_types):
    dataset_in = input_root / dataset
    dataset_out = output_root / dataset / protocol
    splits = [p.name for p in dataset_in.iterdir() if p.is_dir() and p.name.startswith("split")]
    splits.sort()

    print(f"dataset={dataset} protocol={protocol}")
    print(f"unseen_types={unseen_types}")

    for split in splits:
        split_in = dataset_in / split
        split_out = dataset_out / split
        train_rows = load_jsonl(split_in / "train.jsonl")
        dev_rows = load_jsonl(split_in / "dev.jsonl")
        test_rows = load_jsonl(split_in / "test.jsonl")

        all_types = collect_all_types({"train": train_rows, "dev": dev_rows, "test": test_rows})
        seen_types = sorted(t for t in all_types if t not in set(unseen_types))

        train_seen, train_removed = filter_rows_excluding_unseen(train_rows, unseen_types)
        dev_seen, dev_removed = filter_rows_excluding_unseen(dev_rows, unseen_types)
        test_seen, test_seen_stats = filter_event_mentions(test_rows, seen_types)
        test_unseen, test_unseen_stats = filter_event_mentions(test_rows, unseen_types)

        split_out.mkdir(parents=True, exist_ok=True)
        write_jsonl_and_json(split_out / "train", train_seen)
        write_jsonl_and_json(split_out / "dev_seen", dev_seen)
        write_jsonl_and_json(split_out / "dev_mixed", dev_rows)
        write_jsonl_and_json(split_out / "test", test_rows)
        write_jsonl_and_json(split_out / "test_seen", test_seen)
        write_jsonl_and_json(split_out / "test_unseen", test_unseen)

        with open(split_out / "seen_types.json", "w", encoding="utf-8") as f:
            json.dump(seen_types, f, ensure_ascii=False, indent=2)
        with open(split_out / "unseen_types.json", "w", encoding="utf-8") as f:
            json.dump(sorted(unseen_types), f, ensure_ascii=False, indent=2)

        stats = {
            "dataset": dataset,
            "protocol": protocol,
            "split": split,
            "seen_types": seen_types,
            "unseen_types": sorted(unseen_types),
            "source": {
                "train": split_stats(train_rows),
                "dev": split_stats(dev_rows),
                "test": split_stats(test_rows),
            },
            "generated": {
                "train": split_stats(train_seen),
                "dev_seen": split_stats(dev_seen),
                "dev_mixed": split_stats(dev_rows),
                "test": split_stats(test_rows),
                "test_seen": split_stats(test_seen),
                "test_unseen": split_stats(test_unseen),
            },
            "train_filtering": train_removed,
            "dev_filtering": dev_removed,
            "test_seen_partition": test_seen_stats,
            "test_unseen_partition": test_unseen_stats,
        }
        with open(split_out / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        removed_ratio = 0.0
        if train_rows:
            removed_ratio = train_removed["removed_rows"] / len(train_rows)
        print(
            f"  {split}: train {len(train_seen)}/{len(train_rows)} "
            f"removed_rows={train_removed['removed_rows']} removed_ratio={removed_ratio:.4f} "
            f"dev_seen={len(dev_seen)}/{len(dev_rows)} "
            f"test_unseen_mentions={test_unseen_stats['mentions']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="data/processed/textee")
    parser.add_argument("--output_root", default="data/processed/type_holdout")
    parser.add_argument(
        "--protocol_config",
        default="configs/seen_unseen_type_holdout_protocols.json",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    with open(args.protocol_config, "r", encoding="utf-8") as f:
        protocol_map = json.load(f)

    for dataset, protocols in protocol_map.items():
        for protocol, unseen_types in protocols.items():
            generate_dataset_protocol(input_root, output_root, dataset, protocol, unseen_types)


if __name__ == "__main__":
    main()
