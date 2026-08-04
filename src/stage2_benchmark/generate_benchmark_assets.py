import argparse
from pathlib import Path

from common import benchmark_output_dir, generated_config_path, lab_image_name, load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/stage2_llamafactory_benchmark_spec.json")
    parser.add_argument(
        "--model_keys",
        nargs="+",
        default=["qwen3_0_6b_base", "qwen3_1_7b_base", "qwen3_4b_base", "llama3_2_1b", "llama3_2_3b"],
    )
    parser.add_argument("--preset_keys", nargs="+", default=["stageA_default", "throughput_short_ctx", "throughput_mid_ctx"])
    parser.add_argument(
        "--slice_keys",
        nargs="+",
        default=["richere_balanced_split1_predicted_top5", "ace05_balanced_split1_predicted_top5"],
    )
    parser.add_argument("--output_markdown", default="reports/tables/2026-03-26_stage2_llamafactory_benchmark_matrix.md")
    args = parser.parse_args()

    spec = load_json(Path(args.spec))
    image_name = lab_image_name(spec)

    lines = [
        "# Stage2 LLaMA-Factory Benchmark Matrix",
        "",
        f"- image: `{image_name}`",
        "- benchmark goal: select stage-2 model and throughput preset before large-scale direct/CoT runs",
        "",
        "## Planned Runs",
        "",
        "| slice | model_key | model_path | preset | generated_yaml | output_dir |",
        "|---|---|---|---|---|---|",
    ]

    for slice_key in args.slice_keys:
        slice_spec = spec["benchmark_slices"][slice_key]
        for model_key in args.model_keys:
            model = spec["models"][model_key]
            for preset_key in args.preset_keys:
                cfg_path = generated_config_path(model_key, preset_key, slice_key)
                out_dir = benchmark_output_dir(model_key, preset_key, slice_key)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            slice_key,
                            model_key,
                            model["model_path_host"],
                            preset_key,
                            cfg_path.as_posix(),
                            out_dir.as_posix(),
                        ]
                    )
                    + " |"
                )

    output_path = Path(args.output_markdown)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
