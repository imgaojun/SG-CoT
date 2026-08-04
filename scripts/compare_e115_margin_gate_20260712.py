#!/usr/bin/env python3
"""Apply the preregistered E115 corrected-smoke training gate."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare_margin_rows(
    before: dict[str, Any], after: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    before_rows = {str(row["wnd_id"]): row for row in before.get("rows", [])}
    after_rows = {str(row["wnd_id"]): row for row in after.get("rows", [])}
    if set(before_rows) != set(after_rows):
        raise ValueError("pre/post margin scores do not contain identical windows")
    if not before_rows:
        raise ValueError("pre/post margin scores are empty")
    deltas = []
    by_category: dict[str, list[float]] = defaultdict(list)
    for wnd_id in sorted(before_rows):
        pre = before_rows[wnd_id]
        post = after_rows[wnd_id]
        if pre["error_category"] != post["error_category"]:
            raise ValueError(f"category changed for {wnd_id}")
        delta = float(post["margin"]) - float(pre["margin"])
        if not math.isfinite(delta):
            raise ValueError(f"non-finite margin delta for {wnd_id}")
        category = str(pre["error_category"])
        by_category[category].append(delta)
        deltas.append(
            {
                "wnd_id": wnd_id,
                "document_id": pre.get("document_id"),
                "error_category": category,
                "pre_margin": float(pre["margin"]),
                "post_margin": float(post["margin"]),
                "margin_delta": delta,
            }
        )
    return deltas, {
        category: mean(values) for category, values in sorted(by_category.items())
    }


def trainer_is_complete(
    trainer_state: dict[str, Any], model_dir: Path, expected_steps: int
) -> tuple[bool, dict[str, Any]]:
    global_step = int(trainer_state.get("global_step", -1))
    log_history = trainer_state.get("log_history", [])
    numeric_values = []
    for record in log_history:
        if not isinstance(record, dict):
            continue
        for key in ("loss", "grad_norm", "train_loss", "sft_loss", "odds_ratio_loss"):
            value = record.get(key)
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
    finite = bool(numeric_values) and all(math.isfinite(value) for value in numeric_values)
    weights = sorted(model_dir.glob("*.safetensors"))
    complete = global_step == expected_steps and finite and bool(weights)
    return complete, {
        "global_step": global_step,
        "expected_steps": expected_steps,
        "finite_training_values": finite,
        "checked_training_values": len(numeric_values),
        "saved_weight_files": [path.name for path in weights],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_margin_json", type=Path, required=True)
    parser.add_argument("--post_margin_json", type=Path, required=True)
    parser.add_argument("--style_json", type=Path, required=True)
    parser.add_argument("--trainer_state", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--minimum_improved_categories", type=int, default=4)
    parser.add_argument(
        "--required_improved_categories",
        nargs="+",
        default=["extra_frame", "trigger_drift"],
    )
    parser.add_argument("--maximum_nll_gap", type=float, default=0.3)
    parser.add_argument("--expected_steps", type=int, default=5)
    args = parser.parse_args()

    before = load_json(args.pre_margin_json)
    after = load_json(args.post_margin_json)
    style = load_json(args.style_json)
    trainer_state = load_json(args.trainer_state)
    rows, category_deltas = compare_margin_rows(before, after)
    overall_delta = mean([row["margin_delta"] for row in rows])
    improved_categories = sorted(
        category for category, delta in category_deltas.items() if delta > 0.0
    )
    nll_gap = float(style["mean_canonical_minus_native_nll"])
    training_complete, training = trainer_is_complete(
        trainer_state, args.model_dir, args.expected_steps
    )

    criteria = {
        "training_complete_and_finite": training_complete,
        "overall_mean_margin_delta_positive": overall_delta > 0.0,
        "minimum_four_categories_improved": len(improved_categories)
        >= args.minimum_improved_categories,
        "required_categories_improved": all(
            category_deltas.get(category, float("-inf")) > 0.0
            for category in args.required_improved_categories
        ),
        "canonical_native_nll_gap_within_limit": math.isfinite(nll_gap)
        and nll_gap <= args.maximum_nll_gap,
    }
    result = {
        "gate": "E115 corrected training-only smoke",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "pairs": len(rows),
        "overall_mean_margin_delta": overall_delta,
        "pre_mean_margin": float(before["mean_margin"]),
        "post_mean_margin": float(after["mean_margin"]),
        "category_mean_margin_deltas": category_deltas,
        "improved_categories": improved_categories,
        "required_improved_categories": args.required_improved_categories,
        "canonical_minus_native_nll": nll_gap,
        "maximum_canonical_minus_native_nll": args.maximum_nll_gap,
        "thinking_nll_gap": float(style["mean_thinking_nll_gap"]),
        "final_nll_gap": float(style["mean_final_nll_gap"]),
        "training": training,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
