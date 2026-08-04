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

from src.stage2_data.build_formal_stage2_dataset import load_jsonl, update_dataset_info, write_json


def row_wnd_id(row):
    meta = row.get("meta", {})
    wnd_id = meta.get("wnd_id")
    if not wnd_id:
        raise ValueError("Each source row must contain meta.wnd_id")
    return wnd_id


def row_candidate_types(row):
    meta = row.get("meta", {})
    return meta.get("candidate_types")


def load_and_index(path: Path):
    rows = load_jsonl(path)
    mapping = {}
    for row in rows:
        wnd_id = row_wnd_id(row)
        if wnd_id in mapping:
            raise ValueError(f"Duplicate wnd_id={wnd_id} in {path}")
        mapping[wnd_id] = row
    return rows, mapping


def validate_alignment(direct_rows, contrast_rows):
    direct_ids = [row_wnd_id(row) for row in direct_rows]
    contrast_ids = [row_wnd_id(row) for row in contrast_rows]
    if direct_ids != contrast_ids:
        missing_in_contrast = sorted(set(direct_ids) - set(contrast_ids))
        missing_in_direct = sorted(set(contrast_ids) - set(direct_ids))
        raise ValueError(
            "Direct/contrast rows are not aligned by wnd_id order. "
            f"missing_in_contrast={missing_in_contrast[:10]} "
            f"missing_in_direct={missing_in_direct[:10]}"
        )

    for direct_row, contrast_row in zip(direct_rows, contrast_rows):
        wnd_id = row_wnd_id(direct_row)
        if row_candidate_types(direct_row) != row_candidate_types(contrast_row):
            raise ValueError(f"Candidate types mismatch for wnd_id={wnd_id}")
        direct_meta = direct_row.get("meta", {})
        contrast_meta = contrast_row.get("meta", {})
        if direct_meta.get("doc_id") != contrast_meta.get("doc_id"):
            raise ValueError(f"doc_id mismatch for wnd_id={wnd_id}")
        if direct_meta.get("gold_event_types") != contrast_meta.get("gold_event_types"):
            raise ValueError(f"gold_event_types mismatch for wnd_id={wnd_id}")


def sample_contrastive_rows(contrast_rows, desired_count: int, seed: int):
    if desired_count <= 0:
        return []

    rng = random.Random(seed)
    if desired_count <= len(contrast_rows):
        indices = sorted(rng.sample(range(len(contrast_rows)), desired_count))
        return [contrast_rows[idx] for idx in indices]

    full_copies = desired_count // len(contrast_rows)
    remainder = desired_count % len(contrast_rows)
    sampled = []
    for _ in range(full_copies):
        sampled.extend(contrast_rows)
    if remainder:
        indices = sorted(rng.sample(range(len(contrast_rows)), remainder))
        sampled.extend(contrast_rows[idx] for idx in indices)
    return sampled


def annotate_row(row, *, curriculum_source: str, variant_tag: str, mixture_spec: str):
    item = copy.deepcopy(row)
    meta = dict(item.get("meta", {}))
    meta["curriculum_source"] = curriculum_source
    meta["curriculum_variant"] = variant_tag
    meta["mixture_spec"] = mixture_spec
    item["meta"] = meta
    return item


def build_mixed_train_rows(direct_rows, contrast_rows, direct_ratio: int, contrast_ratio: int, seed: int, variant_tag: str):
    direct_count = len(direct_rows)
    desired_contrast_count = int(round(direct_count * contrast_ratio / direct_ratio))
    sampled_contrast = sample_contrastive_rows(contrast_rows, desired_contrast_count, seed)
    mixture_spec = f"{direct_ratio}:{contrast_ratio}"

    mixed = [
        annotate_row(row, curriculum_source="direct", variant_tag=variant_tag, mixture_spec=mixture_spec)
        for row in direct_rows
    ]
    mixed.extend(
        annotate_row(row, curriculum_source="contrastive_v6", variant_tag=variant_tag, mixture_spec=mixture_spec)
        for row in sampled_contrast
    )

    rng = random.Random(seed)
    rng.shuffle(mixed)
    stats = {
        "direct_count": direct_count,
        "selected_contrastive_count": len(sampled_contrast),
        "total_count": len(mixed),
        "direct_ratio": direct_ratio,
        "contrast_ratio": contrast_ratio,
    }
    return mixed, stats


def materialize_eval_rows(rows, variant_tag: str, mixture_spec: str):
    return [
        annotate_row(row, curriculum_source="contrastive_v6_eval", variant_tag=variant_tag, mixture_spec=mixture_spec)
        for row in rows
    ]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def register_dataset(dataset_dir: Path, dataset_name: str, rows, meta: dict):
    file_name = f"{dataset_name}.jsonl"
    write_jsonl(dataset_dir / file_name, rows)
    update_dataset_info(dataset_dir, dataset_name, file_name)
    write_json(dataset_dir / f"{dataset_name}.meta.json", {"dataset_name": dataset_name, "file_name": file_name, **meta})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_train_jsonl", required=True)
    parser.add_argument("--contrast_train_jsonl", required=True)
    parser.add_argument("--contrast_dev_jsonl", required=True)
    parser.add_argument("--contrast_test_jsonl", required=True)
    parser.add_argument("--contrast_test_seen_jsonl", required=True)
    parser.add_argument("--contrast_test_unseen_jsonl", required=True)
    parser.add_argument("--dataset_dir", default="data/stage2_cot_datasets")
    parser.add_argument("--train_dataset_name", required=True)
    parser.add_argument("--dev_dataset_name", required=True)
    parser.add_argument("--test_dataset_name", required=True)
    parser.add_argument("--test_seen_dataset_name", required=True)
    parser.add_argument("--test_unseen_dataset_name", required=True)
    parser.add_argument("--variant_tag", required=True)
    parser.add_argument("--direct_ratio", type=int, required=True)
    parser.add_argument("--contrast_ratio", type=int, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    direct_rows, _ = load_and_index(Path(args.direct_train_jsonl))
    contrast_train_rows, _ = load_and_index(Path(args.contrast_train_jsonl))
    validate_alignment(direct_rows, contrast_train_rows)

    mixed_train_rows, train_stats = build_mixed_train_rows(
        direct_rows=direct_rows,
        contrast_rows=contrast_train_rows,
        direct_ratio=args.direct_ratio,
        contrast_ratio=args.contrast_ratio,
        seed=args.seed,
        variant_tag=args.variant_tag,
    )

    mixture_spec = f"{args.direct_ratio}:{args.contrast_ratio}"
    shared_meta = {
        "variant_tag": args.variant_tag,
        "mixture_spec": mixture_spec,
        "seed": args.seed,
        "direct_train_jsonl": args.direct_train_jsonl,
        "contrast_train_jsonl": args.contrast_train_jsonl,
        "contrast_dev_jsonl": args.contrast_dev_jsonl,
        "contrast_test_jsonl": args.contrast_test_jsonl,
        "contrast_test_seen_jsonl": args.contrast_test_seen_jsonl,
        "contrast_test_unseen_jsonl": args.contrast_test_unseen_jsonl,
    }

    register_dataset(
        dataset_dir=dataset_dir,
        dataset_name=args.train_dataset_name,
        rows=mixed_train_rows,
        meta={**shared_meta, **train_stats, "dataset_role": "train"},
    )

    eval_specs = [
        (args.contrast_dev_jsonl, args.dev_dataset_name, "dev_seen"),
        (args.contrast_test_jsonl, args.test_dataset_name, "test"),
        (args.contrast_test_seen_jsonl, args.test_seen_dataset_name, "test_seen"),
        (args.contrast_test_unseen_jsonl, args.test_unseen_dataset_name, "test_unseen"),
    ]
    for source_jsonl, dataset_name, role in eval_specs:
        rows = load_jsonl(Path(source_jsonl))
        eval_rows = materialize_eval_rows(rows, args.variant_tag, mixture_spec)
        register_dataset(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            rows=eval_rows,
            meta={**shared_meta, "dataset_role": role, "num_examples": len(eval_rows)},
        )

    print(
        json.dumps(
            {
                "train_dataset_name": args.train_dataset_name,
                "variant_tag": args.variant_tag,
                "mixture_spec": mixture_spec,
                **train_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
