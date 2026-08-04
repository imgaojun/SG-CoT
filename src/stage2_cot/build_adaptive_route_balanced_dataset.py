import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_data.build_formal_stage2_dataset import load_jsonl, update_dataset_info, write_json  # noqa: E402


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def route_label(row):
    return (row.get("meta") or {}).get("adaptive_route_label", "direct")


def duplicate_reason_rows(reason_rows, extra_count, rng):
    extras = []
    if extra_count <= 0:
        return extras
    for dup_idx in range(extra_count):
        source_idx = dup_idx % len(reason_rows)
        row = copy.deepcopy(reason_rows[source_idx])
        meta = dict(row.get("meta") or {})
        meta.update(
            {
                "adaptive_route_balance": "reason_oversample",
                "adaptive_route_balance_duplicate": True,
                "adaptive_route_balance_duplicate_index": dup_idx,
                "adaptive_route_balance_source_wnd_id": meta.get("wnd_id"),
            }
        )
        row["meta"] = meta
        extras.append(row)
    rng.shuffle(extras)
    return extras


def build_balanced_rows(rows, target_reason_rate, seed):
    direct_rows = [row for row in rows if route_label(row) != "reason"]
    reason_rows = [row for row in rows if route_label(row) == "reason"]
    if not reason_rows:
        raise ValueError("No reason-labeled rows found; cannot route-balance dataset.")
    if not (0.0 < target_reason_rate < 1.0):
        raise ValueError("--target_reason_rate must be between 0 and 1.")

    desired_reason_count = math.ceil(target_reason_rate * len(direct_rows) / (1.0 - target_reason_rate))
    extra_count = max(0, desired_reason_count - len(reason_rows))
    rng = random.Random(seed)
    balanced = [copy.deepcopy(row) for row in rows]
    balanced.extend(duplicate_reason_rows(reason_rows, extra_count, rng))
    rng.shuffle(balanced)
    actual_reason = sum(1 for row in balanced if route_label(row) == "reason")
    return balanced, {
        "source_num_examples": len(rows),
        "source_direct_count": len(direct_rows),
        "source_reason_count": len(reason_rows),
        "target_reason_rate": target_reason_rate,
        "desired_reason_count": desired_reason_count,
        "extra_reason_duplicates": extra_count,
        "num_examples": len(balanced),
        "direct_count": len(direct_rows),
        "reason_count": actual_reason,
        "actual_reason_rate": actual_reason / len(balanced) if balanced else 0.0,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_train_jsonl", required=True)
    parser.add_argument("--source_train_meta_json", required=True)
    parser.add_argument("--dataset_dir", default="data/stage2_adaptive_datasets")
    parser.add_argument("--output_dataset_name", required=True)
    parser.add_argument("--target_reason_rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    source_rows = load_jsonl(Path(args.source_train_jsonl))
    source_meta = json.loads(Path(args.source_train_meta_json).read_text(encoding="utf-8"))
    rows, summary = build_balanced_rows(source_rows, args.target_reason_rate, args.seed)

    file_name = f"{args.output_dataset_name}.jsonl"
    write_jsonl(dataset_dir / file_name, rows)
    update_dataset_info(dataset_dir, args.output_dataset_name, file_name)
    write_json(
        dataset_dir / f"{args.output_dataset_name}.meta.json",
        {
            **source_meta,
            "dataset_name": args.output_dataset_name,
            "file_name": file_name,
            "route_balance_source_jsonl": args.source_train_jsonl,
            "route_balance_source_meta_json": args.source_train_meta_json,
            "route_balance_strategy": "reason_oversample",
            **summary,
        },
    )
    print(json.dumps({"dataset": args.output_dataset_name, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
