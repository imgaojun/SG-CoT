import argparse
from pathlib import Path

from common import (
    benchmark_output_dir,
    dataset_info_path,
    generated_config_path,
    lab_image_name,
    load_json,
    render_yaml,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/stage2_llamafactory_benchmark_spec.json")
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--preset_key", required=True)
    parser.add_argument("--slice_key", required=True)
    parser.add_argument("--print_docker_cmd", action="store_true")
    args = parser.parse_args()

    spec = load_json(Path(args.spec))
    model = spec["models"][args.model_key]
    preset = spec["throughput_presets"][args.preset_key]
    slice_spec = spec["benchmark_slices"][args.slice_key]
    defaults = spec["default_logging"]
    runtime = spec.get("default_runtime", {})

    output_dir = benchmark_output_dir(args.model_key, args.preset_key, args.slice_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = generated_config_path(args.model_key, args.preset_key, args.slice_key)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model_name_or_path": model["model_path_container"],
        "template": model["template"],
        "dataset_dir": slice_spec["dataset_dir_container"],
        "dataset": slice_spec["dataset_name"],
        "output_dir": f"/workspace/project/{output_dir.as_posix()}",
        **defaults,
        **preset,
    }

    yaml_text = render_yaml(cfg)
    config_path.write_text(yaml_text, encoding="utf-8")
    print(f"wrote {config_path}")

    info_path = dataset_info_path(slice_spec)
    print(f"dataset_info expected at {info_path}")

    image_name = lab_image_name(spec)
    gpu_devices = runtime.get("gpu_devices_host", "all")
    shm_size = runtime.get("shm_size", "16g")
    docker_user = runtime.get("docker_user", "root")
    host_uid = runtime.get("host_uid", 1000)
    host_gid = runtime.get("host_gid", 1000)
    output_dir_container = f"/workspace/project/{output_dir.as_posix()}"
    docker_cmd = (
        f'docker run --rm --user {docker_user} --gpus "device={gpu_devices}" --ipc host --shm-size {shm_size} '
        f"-v {Path.cwd()}:/workspace/project "
        "-v /mnt/disk/gaojun/research/llamafactory-lab/cache/huggingface:/workspace/.cache/huggingface "
        "-v /mnt/disk/gaojun/research/llamafactory-lab/cache/torch_extensions:/workspace/.cache/torch_extensions "
        "-v /mnt/disk/gaojun/research/llamafactory-lab/logs:/workspace/logs "
        "-v /mnt/disk/gaojun/models:/workspace/models:ro "
        "-e CUDA_VISIBLE_DEVICES=0 "
        f"-w /workspace/project {image_name} "
        f"bash -lc 'llamafactory-cli train {config_path.as_posix()} && chown -R {host_uid}:{host_gid} {output_dir_container}'"
    )

    if args.print_docker_cmd:
        print(docker_cmd)


if __name__ == "__main__":
    main()
