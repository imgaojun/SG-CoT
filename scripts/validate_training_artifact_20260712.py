#!/usr/bin/env python3
"""Validate that a saved training artifact is complete enough for a gated run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


class ArtifactValidationError(ValueError):
    pass


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"expected a JSON object: {path}")
    return value


def validate_model_weights(model_dir: Path) -> list[str]:
    config = model_dir / "config.json"
    if not config.is_file():
        raise ArtifactValidationError(f"missing model config: {config}")
    load_object(config)

    for single_name in ("model.safetensors", "pytorch_model.bin"):
        single = model_dir / single_name
        if single.is_file():
            if single.stat().st_size <= 0:
                raise ArtifactValidationError(f"empty model weight file: {single}")
            return [single.name]

    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = model_dir / index_name
        if not index_path.is_file():
            continue
        index = load_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ArtifactValidationError(f"missing weight_map: {index_path}")
        shard_names = sorted(set(weight_map.values()))
        if not shard_names or not all(isinstance(name, str) and name for name in shard_names):
            raise ArtifactValidationError(f"invalid shard names: {index_path}")
        for name in shard_names:
            shard = model_dir / name
            if not shard.is_file():
                raise ArtifactValidationError(f"missing model shard: {shard}")
            if shard.stat().st_size <= 0:
                raise ArtifactValidationError(f"empty model shard: {shard}")
        return shard_names

    raise ArtifactValidationError(f"no supported model weights found in {model_dir}")


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_trainer_state(
    trainer_state_path: Path,
    *,
    min_global_step: int,
    require_finite_step_log: bool,
) -> dict[str, Any]:
    if not trainer_state_path.is_file():
        raise ArtifactValidationError(f"missing trainer state: {trainer_state_path}")
    state = load_object(trainer_state_path)
    try:
        global_step = int(state.get("global_step", -1))
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("invalid trainer global_step") from exc
    if global_step < min_global_step:
        raise ArtifactValidationError(
            f"global_step {global_step} is below required {min_global_step}"
        )

    finite_records = []
    history = state.get("log_history", [])
    if not isinstance(history, list):
        raise ArtifactValidationError("trainer log_history is not a list")
    for record in history:
        if not isinstance(record, dict) or "loss" not in record or "grad_norm" not in record:
            continue
        if finite_number(record["loss"]) and finite_number(record["grad_norm"]):
            finite_records.append(record)
    if require_finite_step_log and not finite_records:
        raise ArtifactValidationError("no training step has finite loss and gradient norm")
    return {
        "global_step": global_step,
        "finite_step_log_count": len(finite_records),
    }


def validate_artifact(
    model_dir: Path,
    *,
    trainer_state_path: Path | None = None,
    min_global_step: int = 0,
    require_finite_step_log: bool = False,
) -> dict[str, Any]:
    shards = validate_model_weights(model_dir)
    report: dict[str, Any] = {
        "model_dir": str(model_dir.resolve()),
        "weight_files": shards,
        "weight_file_count": len(shards),
    }
    if trainer_state_path is not None:
        report["trainer_state"] = validate_trainer_state(
            trainer_state_path,
            min_global_step=min_global_step,
            require_finite_step_log=require_finite_step_log,
        )
    elif min_global_step > 0 or require_finite_step_log:
        raise ArtifactValidationError(
            "trainer_state is required for step or finite-log validation"
        )
    report["passed"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--trainer_state", type=Path)
    parser.add_argument("--min_global_step", type=int, default=0)
    parser.add_argument("--require_finite_step_log", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_artifact(
            args.model_dir,
            trainer_state_path=args.trainer_state,
            min_global_step=args.min_global_step,
            require_finite_step_log=args.require_finite_step_log,
        )
    except (ArtifactValidationError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 6
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
