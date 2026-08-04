import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_dirs(run_dir: Path):
    items = []
    for path in run_dir.glob("checkpoint-*"):
        if path.is_dir():
            try:
                step = int(path.name.split("-")[-1])
            except ValueError:
                continue
            items.append((step, path))
    items.sort(key=lambda x: x[0])
    return [path for _, path in items]


def contains_weights(path: Path):
    return any(
        (path / name).exists()
        for name in [
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        ]
    )


def maybe_final_dir(run_dir: Path):
    return run_dir if contains_weights(run_dir) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--metric_keys", nargs="+", default=["event_f1", "argument_f1", "trigger_f1"])
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reuse_existing", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = checkpoint_dirs(run_dir)
    final_dir = maybe_final_dir(run_dir)
    if final_dir is not None:
        candidates.append(final_dir)

    if not candidates:
        raise ValueError(f"No checkpoint candidates found under {run_dir}")

    best = None
    rows = []
    for candidate in candidates:
        tag = "final" if candidate == run_dir else candidate.name
        eval_dir = output_root / tag
        summary_path = eval_dir / "summary.json"
        if not (args.reuse_existing and summary_path.exists()):
            cmd = [
                sys.executable,
                "src/stage2_quality_validation/eval_adapter_generation.py",
                "--base_model",
                args.base_model,
                "--adapter_path",
                candidate.as_posix(),
                "--eval_jsonl",
                args.eval_jsonl,
                "--output_dir",
                eval_dir.as_posix(),
                "--batch_size",
                str(args.batch_size),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
            ]
            subprocess.run(cmd, check=True)
        summary = load_json(summary_path)
        score_tuple = tuple(summary[key] for key in args.metric_keys)
        row = {
            "checkpoint_tag": tag,
            "checkpoint_path": candidate.as_posix(),
            "eval_dir": eval_dir.as_posix(),
            "metric_keys": args.metric_keys,
            "score_tuple": score_tuple,
            "summary": summary,
        }
        rows.append(row)
        if best is None:
            best = row
        else:
            if args.greater_is_better:
                if score_tuple > tuple(best["score_tuple"]):
                    best = row
            else:
                if score_tuple < tuple(best["score_tuple"]):
                    best = row

    payload = {
        "run_dir": run_dir.as_posix(),
        "eval_jsonl": args.eval_jsonl,
        "metric_keys": args.metric_keys,
        "greater_is_better": args.greater_is_better,
        "batch_size": args.batch_size,
        "best": best,
        "candidates": rows,
    }
    with open(output_root / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
