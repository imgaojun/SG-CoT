import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lab_env(env_path: Path):
    env = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'").strip('"')
    return env


def lab_image_name(spec: dict):
    env = load_lab_env(Path(spec["lab_env_file"]))
    template = spec["lab_image_template"]
    return template.format(**env)


def benchmark_output_tag(model_key: str, preset_key: str, slice_key: str):
    return f"{slice_key}__{model_key}__{preset_key}"


def generated_config_path(model_key: str, preset_key: str, slice_key: str):
    return Path("configs/generated/stage2_benchmark") / f"{benchmark_output_tag(model_key, preset_key, slice_key)}.yaml"


def benchmark_output_dir(model_key: str, preset_key: str, slice_key: str):
    return Path("outputs/stage2_benchmarks") / benchmark_output_tag(model_key, preset_key, slice_key)


def dataset_info_path(slice_spec: dict):
    return Path(slice_spec["dataset_dir_host"]) / "dataset_info.json"


def render_yaml(config: dict):
    lines = []
    for key, value in config.items():
        if isinstance(value, bool):
            value_text = "true" if value else "false"
        elif isinstance(value, (int, float)):
            value_text = str(value)
        else:
            value_text = f'"{value}"'
        lines.append(f"{key}: {value_text}")
    return "\n".join(lines) + "\n"
