#!/usr/bin/env python3
"""Solve E120 composition weights from frozen training-only transfer reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from src.stage2_preference.transfer_balanced_composition import (
    CATEGORIES,
    solve_maximin_weights,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("report must be CATEGORY=PATH")
    category, path = value.split("=", 1)
    if category not in CATEGORIES:
        raise argparse.ArgumentTypeError(f"unknown category: {category}")
    return category, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=parse_report, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--composition_scale", type=float, default=5.0)
    parser.add_argument("--full_floor", type=float, default=0.0)
    args = parser.parse_args()

    reports = dict(args.report)
    if set(reports) != set(CATEGORIES) or len(args.report) != len(CATEGORIES):
        raise ValueError("exactly one transfer report per canonical category is required")
    masked: dict[str, dict[str, float]] = {}
    full: dict[str, dict[str, float]] = {}
    inputs: dict[str, Any] = {}
    for category in CATEGORIES:
        path = reports[category]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("target_category") != category:
            raise ValueError(f"target category mismatch: {path}")
        if payload.get("target_learnable") is not True:
            raise ValueError(f"target gate did not pass: {category}")
        masked[category] = {
            key: float(value)
            for key, value in payload["masked_category_mean_margin_deltas"].items()
        }
        full[category] = {
            key: float(value)
            for key, value in payload[
                "full_response_category_mean_margin_deltas"
            ].items()
        }
        inputs[category] = {"path": str(path), "sha256": sha256(path)}

    solution = solve_maximin_weights(
        masked, full, full_floor=args.full_floor
    )
    scale = float(args.composition_scale)
    result = {
        "protocol": "E120 atomic transfer-balanced delta composition",
        "frozen": True,
        "test_data_access": False,
        "categories": list(CATEGORIES),
        "composition_scale": scale,
        "constraint": "maximize worst masked category delta; full-response category deltas >= 0",
        "input_reports": inputs,
        "masked_transfer_matrix": masked,
        "full_response_transfer_matrix": full,
        **solution,
        "scaled_predicted_masked_margin_deltas": {
            key: scale * value
            for key, value in solution["predicted_masked_margin_deltas"].items()
        },
        "scaled_predicted_full_response_margin_deltas": {
            key: scale * value
            for key, value in solution[
                "predicted_full_response_margin_deltas"
            ].items()
        },
    }
    minimum_masked = min(result["predicted_masked_margin_deltas"].values())
    minimum_full = min(
        result["predicted_full_response_margin_deltas"].values()
    )
    composition_authorized = minimum_masked > 0.0 and minimum_full >= args.full_floor
    result["composition_authorized"] = composition_authorized
    result["gate"] = {
        "strictly_positive_masked_maximin": minimum_masked > 0.0,
        "full_response_floor_satisfied": minimum_full >= args.full_floor,
        "minimum_masked_margin_delta": minimum_masked,
        "minimum_full_response_margin_delta": minimum_full,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=False)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not composition_authorized:
        print(
            "E120 composition gate failed; diagnostic saved and checkpoint merge is not authorized",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
