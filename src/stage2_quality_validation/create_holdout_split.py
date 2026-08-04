import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_dataset_info(dataset_dir: Path, train_name: str, train_file: str, eval_name: str, eval_file: str):
    info_path = dataset_dir / "dataset_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}

    for name, file_name in [(train_name, train_file), (eval_name, eval_file)]:
        info[name] = {
            "file_name": file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--dataset_dir", default="data/stage2_quality_splits")
    parser.add_argument("--dataset_prefix", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    rng = random.Random(args.seed)
    rows = rows[:]
    rng.shuffle(rows)

    train_size = int(len(rows) * args.train_ratio)
    train_rows = rows[:train_size]
    eval_rows = rows[train_size:]

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_name = f"{args.dataset_prefix}_train"
    eval_name = f"{args.dataset_prefix}_eval"
    train_file = f"{train_name}.jsonl"
    eval_file = f"{eval_name}.jsonl"

    write_jsonl(dataset_dir / train_file, train_rows)
    write_jsonl(dataset_dir / eval_file, eval_rows)
    update_dataset_info(dataset_dir, train_name, train_file, eval_name, eval_file)

    meta = {
        "dataset_prefix": args.dataset_prefix,
        "input_jsonl": args.input_jsonl,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_dataset_name": train_name,
        "eval_dataset_name": eval_name,
    }
    meta_path = dataset_dir / f"{args.dataset_prefix}.split_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"wrote {dataset_dir / train_file}")
    print(f"wrote {dataset_dir / eval_file}")
    print(f"updated {dataset_dir / 'dataset_info.json'}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
