import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_yaml(config: dict):
    lines = []
    for key, value in config.items():
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (int, float)):
            text = str(value)
        else:
            text = f'"{value}"'
        lines.append(f"{key}: {text}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_spec", default="configs/stage2_llamafactory_benchmark_spec.json")
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--preset_key", required=True)
    parser.add_argument("--dataset_prefix", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_out", required=True)
    parser.add_argument("--max_steps_override", type=int, default=None)
    parser.add_argument("--lora_rank_override", type=int, default=None)
    parser.add_argument("--save_only_model_override", choices=["true", "false"], default=None)
    args = parser.parse_args()

    spec = load_json(Path(args.benchmark_spec))
    model = spec["models"][args.model_key]
    preset = spec["throughput_presets"][args.preset_key]
    defaults = spec["default_logging"]

    train_name = f"{args.dataset_prefix}_train"
    eval_name = f"{args.dataset_prefix}_eval"

    cfg = {
        "model_name_or_path": model["model_path_container"],
        "template": model["template"],
        "dataset_dir": "/workspace/project/data/stage2_quality_splits",
        "dataset": train_name,
        "eval_dataset": eval_name,
        "output_dir": f"/workspace/project/{args.output_dir}",
        **defaults,
        **preset,
        "val_size": 0.0,
        "eval_strategy": "steps",
        "eval_steps": 10,
        "save_strategy": "no",
        "do_eval": True,
        "save_only_model": True,
    }

    if args.max_steps_override is not None:
        cfg["max_steps"] = args.max_steps_override
    if args.lora_rank_override is not None:
        cfg["lora_rank"] = args.lora_rank_override
    if args.save_only_model_override is not None:
        cfg["save_only_model"] = args.save_only_model_override == "true"

    config_path = Path(args.config_out)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_yaml(cfg), encoding="utf-8")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
