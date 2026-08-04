import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_data.build_formal_stage2_dataset import load_jsonl, write_json


def load_label_map(path: Path):
    rows = load_jsonl(path)
    mapping = {}
    for row in rows:
        row_hash = row["row_hash"]
        if row_hash in mapping:
            raise ValueError(f"Duplicate row_hash in labels: {path} -> {row_hash}")
        mapping[row_hash] = row
    return mapping


def load_score_map(path: Path):
    rows = load_jsonl(path)
    mapping = {}
    for row in rows:
        row_hash = row["row_hash"]
        if row_hash in mapping:
            raise ValueError(f"Duplicate row_hash in scores: {path} -> {row_hash}")
        mapping[row_hash] = row
    return mapping


def load_pair_stats(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["pair_stats"]


def canonical_pair_key(score_row):
    pair = score_row.get("canonical_top_pair")
    if not pair:
        return None
    return " || ".join(pair)


def build_relaxed_rows(
    label_map,
    score_map,
    pair_stats,
    *,
    base_label_key: str,
    target_cot_rate: float,
    min_pair_support: int,
    min_pair_cot_better_rate: float,
    max_backfill_per_pair: int,
):
    common_hashes = set(label_map) & set(score_map)
    if common_hashes != set(label_map) or common_hashes != set(score_map):
        raise ValueError(
            "Labels and scores do not align perfectly: "
            f"label_only={len(set(label_map) - common_hashes)} "
            f"score_only={len(set(score_map) - common_hashes)}"
        )

    target_cot_count = round(len(common_hashes) * target_cot_rate)
    base_cot_hashes = {
        row_hash
        for row_hash, row in label_map.items()
        if row.get(base_label_key) == "COT"
    }

    candidates = []
    for row_hash in common_hashes:
        if row_hash in base_cot_hashes:
            continue
        score_row = score_map[row_hash]
        pair_key = canonical_pair_key(score_row)
        stats = pair_stats.get(pair_key or "", {})
        support = stats.get("n", 0)
        cot_better_rate = stats.get("cot_better_rate", 0.0)
        if support < min_pair_support or cot_better_rate < min_pair_cot_better_rate:
            continue
        candidates.append(
            {
                "row_hash": row_hash,
                "pair_key": pair_key,
                "heuristic_v2_score": score_row.get("heuristic_v2_score", 0.0),
                "pair_support": support,
                "pair_cot_better_rate": cot_better_rate,
            }
        )

    candidates.sort(
        key=lambda row: (
            row["heuristic_v2_score"],
            row["pair_cot_better_rate"],
            row["pair_support"],
            row["row_hash"],
        ),
        reverse=True,
    )

    selected_backfill = set()
    pair_counts = Counter()
    need = max(0, target_cot_count - len(base_cot_hashes))
    for row in candidates:
        if len(selected_backfill) >= need:
            break
        pair_key = row["pair_key"]
        if pair_counts[pair_key] >= max_backfill_per_pair:
            continue
        selected_backfill.add(row["row_hash"])
        pair_counts[pair_key] += 1

    rows = []
    for row_hash in sorted(common_hashes):
        base_row = dict(label_map[row_hash])
        score_row = score_map[row_hash]
        pair_key = canonical_pair_key(score_row)
        stats = pair_stats.get(pair_key or "", {})
        if row_hash in base_cot_hashes:
            relaxed_label = "COT"
            relaxed_reason = "preserve_base_cot"
        elif row_hash in selected_backfill:
            relaxed_label = "COT"
            relaxed_reason = "ambiguity_backfill"
        else:
            relaxed_label = "DIRECT"
            relaxed_reason = "preserve_base_direct"
        base_row.update(
            {
                "relaxed_label": relaxed_label,
                "relaxed_reason": relaxed_reason,
                "relaxed_pair_key": pair_key,
                "relaxed_pair_support": stats.get("n", 0),
                "relaxed_pair_cot_better_rate": stats.get("cot_better_rate", 0.0),
                "relaxed_heuristic_v2_score": score_row.get("heuristic_v2_score", 0.0),
            }
        )
        rows.append(base_row)

    return rows, {
        "target_cot_rate": target_cot_rate,
        "target_cot_count": target_cot_count,
        "base_cot_count": len(base_cot_hashes),
        "eligible_backfill_pool": len(candidates),
        "selected_backfill_count": len(selected_backfill),
        "selected_backfill_pair_counts": dict(pair_counts),
    }


def summarize(rows, extra):
    relaxed_counts = Counter(row["relaxed_label"] for row in rows)
    relaxed_reason_counts = Counter(row["relaxed_reason"] for row in rows)
    return {
        **extra,
        "num_examples": len(rows),
        "relaxed_counts": dict(relaxed_counts),
        "relaxed_reason_counts": dict(relaxed_reason_counts),
        "achieved_cot_count": relaxed_counts.get("COT", 0),
        "achieved_cot_rate": relaxed_counts.get("COT", 0) / len(rows) if rows else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_labels_jsonl", required=True)
    parser.add_argument("--router_scores_jsonl", required=True)
    parser.add_argument("--router_prior_json", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary_json", default=None)
    parser.add_argument("--base_label_key", default="strict_label")
    parser.add_argument("--target_cot_rate", type=float, default=0.10)
    parser.add_argument("--min_pair_support", type=int, default=10)
    parser.add_argument("--min_pair_cot_better_rate", type=float, default=0.20)
    parser.add_argument("--max_backfill_per_pair", type=int, default=40)
    args = parser.parse_args()

    label_map = load_label_map(Path(args.base_labels_jsonl))
    score_map = load_score_map(Path(args.router_scores_jsonl))
    pair_stats = load_pair_stats(Path(args.router_prior_json))
    rows, extra = build_relaxed_rows(
        label_map,
        score_map,
        pair_stats,
        base_label_key=args.base_label_key,
        target_cot_rate=args.target_cot_rate,
        min_pair_support=args.min_pair_support,
        min_pair_cot_better_rate=args.min_pair_cot_better_rate,
        max_backfill_per_pair=args.max_backfill_per_pair,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = Path(args.output_summary_json) if args.output_summary_json else output_path.with_suffix(".summary.json")
    write_json(
        summary_path,
        {
            "base_labels_jsonl": args.base_labels_jsonl,
            "router_scores_jsonl": args.router_scores_jsonl,
            "router_prior_json": args.router_prior_json,
            "base_label_key": args.base_label_key,
            "target_cot_rate": args.target_cot_rate,
            "min_pair_support": args.min_pair_support,
            "min_pair_cot_better_rate": args.min_pair_cot_better_rate,
            "max_backfill_per_pair": args.max_backfill_per_pair,
            **summarize(rows, extra),
        },
    )
    print(json.dumps({"output_jsonl": args.output_jsonl, "output_summary_json": summary_path.as_posix(), **summarize(rows, extra)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
