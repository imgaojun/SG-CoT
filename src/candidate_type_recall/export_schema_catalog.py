import argparse
import json
from pathlib import Path

from schema_library import SCHEMA_LIBRARY, schema_document


def collect_types(dataset_root: Path):
    types = set()
    split1 = dataset_root / "split1"
    for part in ["train", "dev", "test"]:
        path = split1 / f"{part}.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                for ev in row["event_mentions"]:
                    types.add(ev["event_type"])
    return sorted(types)


def export_dataset_schema(dataset: str, dataset_root: Path, output_dir: Path):
    event_types = collect_types(dataset_root)
    missing = [t for t in event_types if t not in SCHEMA_LIBRARY]
    if missing:
        raise ValueError(f"Missing schema definitions for {dataset}: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for event_type in event_types:
        entry = dict(SCHEMA_LIBRARY[event_type])
        entry["event_type"] = event_type
        entry["document"] = schema_document(entry)
        payload.append(entry)

    output_path = output_dir / f"{dataset}.event_schema.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="data/processed/textee")
    parser.add_argument("--output_dir", default="data/schema")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    for dataset in ["ace05-en", "richere-en"]:
        export_dataset_schema(dataset, input_root / dataset, output_dir)


if __name__ == "__main__":
    main()
