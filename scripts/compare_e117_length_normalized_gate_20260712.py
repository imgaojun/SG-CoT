#!/usr/bin/env python3
"""Apply the preregistered E117 length-normalized atomic SimPO gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_e115_margin_gate_20260712 import (
    compare_margin_rows,
    load_json,
    mean,
    trainer_is_complete,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_margin_json", type=Path, required=True)
    parser.add_argument("--post_margin_json", type=Path, required=True)
    parser.add_argument("--trainer_state", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--minimum_overall_delta", type=float, default=0.002)
    parser.add_argument("--maximum_chosen_logp_drop", type=float, default=0.02)
    parser.add_argument("--expected_steps", type=int, default=5)
    args = parser.parse_args()

    before = load_json(args.pre_margin_json)
    after = load_json(args.post_margin_json)
    trainer_state = load_json(args.trainer_state)
    rows, category_deltas = compare_margin_rows(before, after)
    before_rows = {str(row["wnd_id"]): row for row in before["rows"]}
    after_rows = {str(row["wnd_id"]): row for row in after["rows"]}
    chosen_logp_delta = mean(
        [
            float(after_rows[wnd_id]["chosen_logp"])
            - float(before_rows[wnd_id]["chosen_logp"])
            for wnd_id in sorted(before_rows)
        ]
    )
    rejected_logp_delta = mean(
        [
            float(after_rows[wnd_id]["rejected_logp"])
            - float(before_rows[wnd_id]["rejected_logp"])
            for wnd_id in sorted(before_rows)
        ]
    )
    overall_delta = mean([row["margin_delta"] for row in rows])
    training_complete, training = trainer_is_complete(
        trainer_state, args.model_dir, args.expected_steps
    )
    improved_categories = sorted(
        category for category, delta in category_deltas.items() if delta > 0.0
    )
    criteria = {
        "training_complete_and_finite": training_complete,
        "overall_margin_delta_at_least_0_002": overall_delta
        >= args.minimum_overall_delta,
        "all_five_categories_improved": len(improved_categories) == 5,
        "extra_frame_improved": category_deltas.get("extra_frame", 0.0) > 0.0,
        "trigger_drift_improved": category_deltas.get("trigger_drift", 0.0) > 0.0,
        "chosen_logp_preserved": chosen_logp_delta >= -args.maximum_chosen_logp_drop,
    }
    result = {
        "gate": "E117 length-normalized atomic SimPO smoke",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "pairs": len(rows),
        "overall_mean_margin_delta": overall_delta,
        "minimum_overall_mean_margin_delta": args.minimum_overall_delta,
        "pre_mean_margin": float(before["mean_margin"]),
        "post_mean_margin": float(after["mean_margin"]),
        "category_mean_margin_deltas": category_deltas,
        "improved_categories": improved_categories,
        "mean_chosen_logp_delta": chosen_logp_delta,
        "mean_rejected_logp_delta": rejected_logp_delta,
        "maximum_chosen_logp_drop": args.maximum_chosen_logp_drop,
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
