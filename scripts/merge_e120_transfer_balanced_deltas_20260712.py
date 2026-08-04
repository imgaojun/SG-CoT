#!/usr/bin/env python3
"""Compose category-isolated E120 checkpoints as scaled task vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

from src.stage2_preference.transfer_balanced_composition import (
    CATEGORIES,
    combine_tensor,
)


MODEL_AUXILIARY_FILES = (
    "README.md",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_expert(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expert must be CATEGORY=MODEL_DIR")
    category, path = value.split("=", 1)
    if category not in CATEGORIES:
        raise argparse.ArgumentTypeError(f"unknown category: {category}")
    return category, Path(path)


def load_index(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "model.safetensors.index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("weight_map"):
        raise ValueError(f"invalid model index: {path}")
    return payload


def tensor_locations(model_dir: Path, index: dict[str, Any]) -> dict[str, Path]:
    return {
        key: model_dir / shard for key, shard in index["weight_map"].items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=Path, required=True)
    parser.add_argument("--expert", action="append", type=parse_expert, required=True)
    parser.add_argument("--weights_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--composition_scale", type=float, default=5.0)
    args = parser.parse_args()

    experts = dict(args.expert)
    if set(experts) != set(CATEGORIES) or len(args.expert) != len(CATEGORIES):
        raise ValueError("exactly one checkpoint per canonical category is required")
    weights_payload = json.loads(args.weights_json.read_text(encoding="utf-8"))
    if weights_payload.get("composition_authorized") is not True:
        raise ValueError("weight solution did not authorize checkpoint composition")
    if weights_payload.get("frozen") is not True:
        raise ValueError("weight solution is not frozen")
    if weights_payload.get("test_data_access") is not False:
        raise ValueError("weight solution does not certify training-only selection")
    weights = {
        category: float(value) for category, value in weights_payload["weights"].items()
    }
    if set(weights) != set(CATEGORIES):
        raise ValueError("weight categories differ from the canonical categories")
    if abs(float(weights_payload["composition_scale"]) - args.composition_scale) > 1e-12:
        raise ValueError("composition scale differs from the frozen weight solution")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing output reuse: {args.output_dir}")
    temporary = args.output_dir.with_name(args.output_dir.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"stale incomplete output exists: {temporary}")

    base_index = load_index(args.base_model)
    base_locations = tensor_locations(args.base_model, base_index)
    expert_indices = {category: load_index(path) for category, path in experts.items()}
    for category, index in expert_indices.items():
        if index["weight_map"] != base_index["weight_map"]:
            raise ValueError(f"expert shard map differs from base: {category}")
    expert_locations = {
        category: tensor_locations(experts[category], expert_indices[category])
        for category in CATEGORIES
    }

    temporary.mkdir(parents=True)
    for name in MODEL_AUXILIARY_FILES:
        source = args.base_model / name
        if source.exists():
            shutil.copy2(source, temporary / name)
    shutil.copy2(
        args.base_model / "model.safetensors.index.json",
        temporary / "model.safetensors.index.json",
    )

    shard_names = sorted(set(base_index["weight_map"].values()))
    output_hashes: dict[str, str] = {}
    ordered_weights = [weights[category] for category in CATEGORIES]
    try:
        for shard_number, shard_name in enumerate(shard_names, start=1):
            keys = sorted(
                key
                for key, location in base_locations.items()
                if location.name == shard_name
            )
            tensors = {}
            with ExitStack() as stack:
                base_handle = stack.enter_context(
                    safe_open(args.base_model / shard_name, framework="pt", device="cpu")
                )
                expert_handles = [
                    stack.enter_context(
                        safe_open(
                            expert_locations[category][keys[0]],
                            framework="pt",
                            device="cpu",
                        )
                    )
                    for category in CATEGORIES
                ]
                for key_index, key in enumerate(keys, start=1):
                    base_tensor = base_handle.get_tensor(key)
                    expert_tensors = [handle.get_tensor(key) for handle in expert_handles]
                    tensors[key] = combine_tensor(
                        base_tensor,
                        expert_tensors,
                        ordered_weights,
                        args.composition_scale,
                    ).contiguous()
                    if key_index % 50 == 0 or key_index == len(keys):
                        print(
                            f"composed shard {shard_number}/{len(shard_names)}: "
                            f"{key_index}/{len(keys)} tensors",
                            flush=True,
                        )
            output_path = temporary / shard_name
            save_file(tensors, output_path, metadata={"format": "pt"})
            output_hashes[shard_name] = sha256(output_path)
            del tensors
    except Exception:
        raise

    manifest = {
        "protocol": "E120 atomic transfer-balanced delta composition",
        "base_model": str(args.base_model),
        "base_index_sha256": sha256(args.base_model / "model.safetensors.index.json"),
        "experts": {
            category: {
                "path": str(experts[category]),
                "index_sha256": sha256(
                    experts[category] / "model.safetensors.index.json"
                ),
            }
            for category in CATEGORIES
        },
        "weights_json": str(args.weights_json),
        "weights_sha256": sha256(args.weights_json),
        "weights": weights,
        "composition_scale": args.composition_scale,
        "formula": "theta_base + scale * sum(weight_c * (theta_c - theta_base))",
        "output_shard_sha256": output_hashes,
        "test_data_access": False,
    }
    (temporary / "composition_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary.rename(args.output_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
