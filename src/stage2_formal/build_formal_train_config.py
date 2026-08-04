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
        elif isinstance(value, float):
            # Avoid scientific notation such as 1e-05, which some local YAML
            # parsing paths can load back as strings instead of numeric scalars.
            text = format(value, ".10f").rstrip("0").rstrip(".")
            if "." not in text:
                text += ".0"
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
    parser.add_argument("--dataset_dir", default="/workspace/project/data/stage2_formal_datasets")
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--eval_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_out", required=True)
    parser.add_argument("--max_steps_override", type=int, default=None)
    parser.add_argument("--num_train_epochs_override", type=float, default=None)
    parser.add_argument("--disable_max_steps", action="store_true")
    parser.add_argument("--max_samples_override", type=int, default=None)
    parser.add_argument("--disable_max_samples", action="store_true")
    parser.add_argument("--finetuning_type_override", choices=["lora", "full"], default=None)
    parser.add_argument("--lora_rank_override", type=int, default=None)
    parser.add_argument("--cutoff_len_override", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size_override", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps_override", type=int, default=None)
    parser.add_argument("--learning_rate_override", type=float, default=None)
    parser.add_argument("--warmup_ratio_override", type=float, default=None)
    parser.add_argument("--packing_override", choices=["true", "false"], default=None)
    parser.add_argument("--best_checkpoint", action="store_true")
    parser.add_argument("--save_steps_override", type=int, default=None)
    parser.add_argument("--save_total_limit_override", type=int, default=None)
    parser.add_argument("--keep_all_checkpoints", action="store_true")
    parser.add_argument("--save_strategy_override", choices=["no", "steps", "epoch"], default=None)
    parser.add_argument("--eval_strategy_override", choices=["no", "steps", "epoch"], default=None)
    parser.add_argument("--load_best_model_at_end_override", choices=["true", "false"], default=None)
    parser.add_argument("--save_only_model_override", choices=["true", "false"], default=None)
    parser.add_argument("--metric_for_best_model_override", default=None)
    parser.add_argument("--greater_is_better_override", choices=["true", "false"], default=None)
    parser.add_argument("--deepspeed_override", default=None)
    args = parser.parse_args()

    spec = load_json(Path(args.benchmark_spec))
    model = spec["models"][args.model_key]
    preset = spec["throughput_presets"][args.preset_key]
    defaults = spec["default_logging"]

    cfg = {
        "model_name_or_path": model["model_path_container"],
        "template": model["template"],
        "dataset_dir": args.dataset_dir,
        "dataset": args.train_dataset,
        "eval_dataset": args.eval_dataset,
        "output_dir": f"/workspace/project/{args.output_dir}",
        **defaults,
        **preset,
        "val_size": 0.0,
        "eval_strategy": "steps",
        "eval_steps": 10,
        "save_strategy": "no",
        "do_eval": True,
        # Default to model-only checkpoints to avoid saving optimizer states,
        # which dominate disk usage in full-SFT runs.
        "save_only_model": True,
    }

    if args.max_steps_override is not None:
        cfg["max_steps"] = args.max_steps_override
    if args.disable_max_steps:
        cfg.pop("max_steps", None)
    if args.num_train_epochs_override is not None:
        cfg["num_train_epochs"] = args.num_train_epochs_override
        cfg.pop("max_steps", None)
    if args.finetuning_type_override is not None:
        cfg["finetuning_type"] = args.finetuning_type_override
        if args.finetuning_type_override == "full":
            cfg.pop("lora_rank", None)
            cfg.pop("lora_target", None)
    if args.lora_rank_override is not None:
        cfg["lora_rank"] = args.lora_rank_override
    if args.cutoff_len_override is not None:
        cfg["cutoff_len"] = args.cutoff_len_override
    if args.per_device_train_batch_size_override is not None:
        cfg["per_device_train_batch_size"] = args.per_device_train_batch_size_override
    if args.gradient_accumulation_steps_override is not None:
        cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps_override
    if args.learning_rate_override is not None:
        cfg["learning_rate"] = args.learning_rate_override
    if args.warmup_ratio_override is not None:
        cfg["warmup_ratio"] = args.warmup_ratio_override
    if args.packing_override is not None:
        cfg["packing"] = args.packing_override == "true"
    if args.save_steps_override is not None:
        cfg["save_steps"] = args.save_steps_override
    if args.save_total_limit_override is not None:
        cfg["save_total_limit"] = args.save_total_limit_override
    if args.save_strategy_override is not None:
        cfg["save_strategy"] = args.save_strategy_override
    if args.eval_strategy_override is not None:
        cfg["eval_strategy"] = args.eval_strategy_override
    if args.load_best_model_at_end_override is not None:
        cfg["load_best_model_at_end"] = args.load_best_model_at_end_override == "true"
    if args.save_only_model_override is not None:
        cfg["save_only_model"] = args.save_only_model_override == "true"
    if args.metric_for_best_model_override is not None:
        cfg["metric_for_best_model"] = args.metric_for_best_model_override
    if args.greater_is_better_override is not None:
        cfg["greater_is_better"] = args.greater_is_better_override == "true"
    if args.deepspeed_override is not None:
        cfg["deepspeed"] = args.deepspeed_override
    if args.best_checkpoint:
        save_steps = args.save_steps_override if args.save_steps_override is not None else cfg["eval_steps"]
        cfg["save_strategy"] = "steps"
        cfg["save_steps"] = save_steps
        cfg["load_best_model_at_end"] = True
        cfg["metric_for_best_model"] = "eval_loss"
        cfg["greater_is_better"] = False
        cfg["save_total_limit"] = args.save_total_limit_override if args.save_total_limit_override is not None else 2
    if args.keep_all_checkpoints:
        cfg.pop("save_total_limit", None)
    if args.disable_max_samples:
        cfg.pop("max_samples", None)
    if args.max_samples_override is not None:
        cfg["max_samples"] = args.max_samples_override

    config_path = Path(args.config_out)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_yaml(cfg), encoding="utf-8")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
