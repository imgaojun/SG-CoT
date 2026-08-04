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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    rng = random.Random(args.seed)
    rows = rows[:]
    rng.shuffle(rows)
    rows = rows[: args.num_samples]
    write_jsonl(Path(args.output_jsonl), rows)
    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
