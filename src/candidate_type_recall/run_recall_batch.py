import argparse
import json
import subprocess
import sys
from pathlib import Path


def sanitize_slug(text: str) -> str:
    return text.replace("-", "_").replace("/", "_")


def load_batch_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def output_dir_for(output_root: Path, dataset: str, protocol: str, split: str, part: str, tag: str):
    name = f"{sanitize_slug(dataset)}_{sanitize_slug(protocol)}_{split}_{part}_{tag}"
    return output_root / name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--python_bin", default=sys.executable)
    args = parser.parse_args()

    cfg = load_batch_config(Path(args.config))
    data_root = Path(cfg["data_root"])
    schema_dir = Path(cfg["schema_dir"])
    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = cfg["datasets"]
    protocols = cfg["protocols"]
    splits = cfg["splits"]
    parts = cfg["parts"]
    models = cfg["models"]
    top_k = [str(v) for v in cfg["top_k"]]
    num_workers = str(cfg.get("num_workers", 4))
    tag = cfg.get("tag", "full")

    for dataset in datasets:
        schema_path = schema_dir / f"{dataset}.event_schema.json"
        for protocol in protocols:
            for split in splits:
                for part in parts:
                    out_dir = output_dir_for(output_root, dataset, protocol, split, part, tag)
                    summary_path = out_dir / "summary.json"
                    if summary_path.exists():
                        print(f"skip existing {summary_path}")
                        continue

                    cmd = [
                        args.python_bin,
                        "src/candidate_type_recall/run_siliconflow_rerank.py",
                        "--data_root",
                        str(data_root),
                        "--dataset",
                        dataset,
                        "--protocol",
                        protocol,
                        "--split",
                        split,
                        "--part",
                        part,
                        "--schema_path",
                        str(schema_path),
                        "--models",
                        *models,
                        "--top_k",
                        *top_k,
                        "--num_workers",
                        num_workers,
                        "--output_dir",
                        str(out_dir),
                    ]

                    print("running:", " ".join(cmd))
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
