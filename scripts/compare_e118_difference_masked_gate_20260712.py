#!/usr/bin/env python3
"""Apply the preregistered E118 difference-masked SimPO smoke gate."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.compare_e115_margin_gate_20260712 import (
    load_json,
    mean,
    trainer_is_complete,
)


def paired_deltas(
    before: dict[str, Any], after: dict[str, Any], margin_key: str
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    before_rows = {str(row["wnd_id"]): row for row in before["rows"]}
    after_rows = {str(row["wnd_id"]): row for row in after["rows"]}
    if set(before_rows) != set(after_rows) or not before_rows:
        raise ValueError("pre/post scores must contain identical nonempty windows")
    rows = []
    by_category: dict[str, list[float]] = defaultdict(list)
    for wnd_id in sorted(before_rows):
        pre = before_rows[wnd_id]
        post = after_rows[wnd_id]
        category = str(pre["error_category"])
        if category != str(post["error_category"]):
            raise ValueError(f"category changed for {wnd_id}")
        delta = float(post[margin_key]) - float(pre[margin_key])
        if not math.isfinite(delta):
            raise ValueError(f"non-finite {margin_key} delta for {wnd_id}")
        by_category[category].append(delta)
        rows.append(
            {
                "wnd_id": wnd_id,
                "document_id": pre.get("document_id"),
                "error_category": category,
                "pre_margin": float(pre[margin_key]),
                "post_margin": float(post[margin_key]),
                "margin_delta": delta,
            }
        )
    return rows, {
        category: mean(values) for category, values in sorted(by_category.items())
    }


def runtime_mask_evidence(path: Path) -> tuple[bool, dict[str, Any]]:
    matches = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"event": "e118_difference_mask_active"' not in line:
                continue
            start = line.find("{")
            try:
                payload = json.loads(line[start:])
            except (json.JSONDecodeError, ValueError):
                continue
            matches.append(payload)
    if len(matches) != 1:
        return False, {"marker_count": len(matches), "markers": matches}
    marker = matches[0]
    valid = (
        int(marker.get("batch_pairs", 0)) > 0
        and 0 < int(marker.get("chosen_kept_tokens", 0))
        < int(marker.get("chosen_response_tokens", 0))
        and 0 < int(marker.get("rejected_kept_tokens", 0))
        < int(marker.get("rejected_response_tokens", 0))
        and int(marker.get("context_tokens", -1)) == 1
    )
    return valid, {"marker_count": 1, "marker": marker}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_score_json", type=Path, required=True)
    parser.add_argument("--post_score_json", type=Path, required=True)
    parser.add_argument("--trainer_state", type=Path, required=True)
    parser.add_argument("--training_log", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--minimum_masked_delta", type=float, default=0.005)
    parser.add_argument("--maximum_chosen_full_logp_drop", type=float, default=0.02)
    parser.add_argument("--expected_steps", type=int, default=5)
    args = parser.parse_args()

    before = load_json(args.pre_score_json)
    after = load_json(args.post_score_json)
    trainer_state = load_json(args.trainer_state)
    masked_rows, masked_categories = paired_deltas(
        before, after, "masked_margin"
    )
    full_rows, full_categories = paired_deltas(before, after, "full_margin")
    masked_delta = mean([row["margin_delta"] for row in masked_rows])
    full_delta = mean([row["margin_delta"] for row in full_rows])
    before_rows = {str(row["wnd_id"]): row for row in before["rows"]}
    after_rows = {str(row["wnd_id"]): row for row in after["rows"]}
    chosen_full_delta = mean(
        [
            float(after_rows[wnd_id]["chosen_full_logp"])
            - float(before_rows[wnd_id]["chosen_full_logp"])
            for wnd_id in sorted(before_rows)
        ]
    )
    rejected_full_delta = mean(
        [
            float(after_rows[wnd_id]["rejected_full_logp"])
            - float(before_rows[wnd_id]["rejected_full_logp"])
            for wnd_id in sorted(before_rows)
        ]
    )
    training_complete, training = trainer_is_complete(
        trainer_state, args.model_dir, args.expected_steps
    )
    runtime_mask_active, runtime_mask = runtime_mask_evidence(args.training_log)
    criteria = {
        "training_complete_and_finite": training_complete,
        "runtime_difference_mask_active": runtime_mask_active,
        "masked_margin_delta_at_least_0_005": masked_delta
        >= args.minimum_masked_delta,
        "all_five_masked_categories_positive": len(masked_categories) == 5
        and all(value > 0.0 for value in masked_categories.values()),
        "full_response_overall_delta_positive": full_delta > 0.0,
        "full_response_extra_frame_positive": full_categories.get(
            "extra_frame", 0.0
        )
        > 0.0,
        "full_response_trigger_drift_positive": full_categories.get(
            "trigger_drift", 0.0
        )
        > 0.0,
        "chosen_full_logp_preserved": chosen_full_delta
        >= -args.maximum_chosen_full_logp_drop,
    }
    result = {
        "gate": "E118 difference-masked atomic SimPO smoke",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "pairs": len(masked_rows),
        "masked_mean_margin_delta": masked_delta,
        "minimum_masked_mean_margin_delta": args.minimum_masked_delta,
        "masked_category_mean_margin_deltas": masked_categories,
        "full_response_mean_margin_delta": full_delta,
        "full_response_category_mean_margin_deltas": full_categories,
        "mean_chosen_full_logp_delta": chosen_full_delta,
        "mean_rejected_full_logp_delta": rejected_full_delta,
        "maximum_chosen_full_logp_drop": args.maximum_chosen_full_logp_drop,
        "training": training,
        "runtime_mask": runtime_mask,
        "masked_rows": masked_rows,
        "full_response_rows": full_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
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
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
