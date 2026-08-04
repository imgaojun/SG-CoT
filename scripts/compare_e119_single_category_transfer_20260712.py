#!/usr/bin/env python3
"""Measure target and cross-category transfer from one E119 optimizer step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.compare_e115_margin_gate_20260712 import load_json, mean, trainer_is_complete
from scripts.compare_e118_difference_masked_gate_20260712 import (
    paired_deltas,
    runtime_mask_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_category", required=True)
    parser.add_argument("--pre_score_json", type=Path, required=True)
    parser.add_argument("--post_score_json", type=Path, required=True)
    parser.add_argument("--trainer_state", type=Path, required=True)
    parser.add_argument("--training_log", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    before = load_json(args.pre_score_json)
    after = load_json(args.post_score_json)
    masked_rows, masked_categories = paired_deltas(before, after, "masked_margin")
    full_rows, full_categories = paired_deltas(before, after, "full_margin")
    training_complete, training = trainer_is_complete(
        load_json(args.trainer_state), args.model_dir, 1
    )
    runtime_active, runtime = runtime_mask_evidence(args.training_log)
    if args.target_category not in masked_categories:
        raise ValueError(f"target category absent: {args.target_category}")
    criteria = {
        "training_complete_and_finite": training_complete,
        "runtime_difference_mask_active": runtime_active,
        "target_masked_margin_positive": masked_categories[args.target_category] > 0.0,
        "target_full_response_margin_positive": full_categories[args.target_category]
        > 0.0,
    }
    result = {
        "diagnostic": "E119 single-category masked-SimPO transfer",
        "target_category": args.target_category,
        "target_learnable": all(criteria.values()),
        "criteria": criteria,
        "masked_mean_margin_delta": mean(
            [float(row["margin_delta"]) for row in masked_rows]
        ),
        "masked_category_mean_margin_deltas": masked_categories,
        "full_response_mean_margin_delta": mean(
            [float(row["margin_delta"]) for row in full_rows]
        ),
        "full_response_category_mean_margin_deltas": full_categories,
        "training": training,
        "runtime_mask": runtime,
        "masked_rows": masked_rows,
        "full_response_rows": full_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"masked_rows", "full_response_rows"}
            },
            indent=2,
        )
    )
    return 0 if result["target_learnable"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
