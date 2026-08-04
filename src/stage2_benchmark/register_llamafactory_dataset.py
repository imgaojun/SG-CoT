import argparse
import json
from pathlib import Path

from common import dataset_info_path, load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/stage2_llamafactory_benchmark_spec.json")
    parser.add_argument("--slice_key", required=True)
    args = parser.parse_args()

    spec = load_json(Path(args.spec))
    slice_spec = spec["benchmark_slices"][args.slice_key]
    dataset_dir = Path(slice_spec["dataset_dir_host"])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    info_path = dataset_info_path(slice_spec)
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}

    info[slice_spec["dataset_name"]] = {
        "file_name": slice_spec["dataset_file"],
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"updated {info_path}")
    print(f"registered dataset {slice_spec['dataset_name']}")


if __name__ == "__main__":
    main()
