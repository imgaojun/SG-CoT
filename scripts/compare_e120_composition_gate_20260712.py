#!/usr/bin/env python3
"""Apply the frozen E120 training-only gate to a composed checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.compare_e115_margin_gate_20260712 import load_json, mean
from scripts.compare_e118_difference_masked_gate_20260712 import paired_deltas
from src.stage2_preference.transfer_balanced_composition import CATEGORIES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_score_json", type=Path, required=True)
    parser.add_argument("--post_score_json", type=Path, required=True)
    parser.add_argument("--weights_json", type=Path, required=True)
    parser.add_argument("--composition_manifest", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--min_masked_mean_delta", type=float, default=0.005)
    parser.add_argument("--min_chosen_full_delta", type=float, default=-0.02)
    args = parser.parse_args()

    before = load_json(args.pre_score_json)
    after = load_json(args.post_score_json)
    weights = load_json(args.weights_json)
    manifest = load_json(args.composition_manifest)
    if manifest["weights_sha256"] != sha256(args.weights_json):
        raise ValueError("composition did not use the frozen weights file")
    if manifest.get("test_data_access") is not False:
        raise ValueError("composition manifest lacks the no-test boundary")

    masked_rows, masked_categories = paired_deltas(before, after, "masked_margin")
    full_rows, full_categories = paired_deltas(before, after, "full_margin")
    if set(masked_categories) != set(CATEGORIES) or set(full_categories) != set(
        CATEGORIES
    ):
        raise ValueError("scored categories differ from the frozen category set")
    masked_mean = mean([float(row["margin_delta"]) for row in masked_rows])
    full_mean = mean([float(row["margin_delta"]) for row in full_rows])
    before_rows = {str(row["wnd_id"]): row for row in before["rows"]}
    after_rows = {str(row["wnd_id"]): row for row in after["rows"]}
    chosen_full_delta = mean(
        [
            float(after_rows[key]["chosen_full_logp"])
            - float(before_rows[key]["chosen_full_logp"])
            for key in sorted(before_rows)
        ]
    )
    criteria = {
        "predicted_masked_all_categories_positive": min(
            weights["predicted_masked_margin_deltas"].values()
        )
        > 0.0,
        "predicted_full_all_categories_nonnegative": min(
            weights["predicted_full_response_margin_deltas"].values()
        )
        >= -1e-10,
        "actual_masked_mean_at_least_threshold": masked_mean
        >= args.min_masked_mean_delta,
        "actual_masked_all_categories_positive": min(masked_categories.values())
        > 0.0,
        "actual_full_mean_positive": full_mean > 0.0,
        "actual_full_all_categories_positive": min(full_categories.values()) > 0.0,
        "chosen_full_logp_preserved": chosen_full_delta
        >= args.min_chosen_full_delta,
    }
    result = {
        "protocol": "E120 atomic transfer-balanced delta composition gate",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "thresholds": {
            "min_masked_mean_delta": args.min_masked_mean_delta,
            "min_chosen_full_delta": args.min_chosen_full_delta,
        },
        "weights": weights["weights"],
        "composition_scale": weights["composition_scale"],
        "predicted_masked_category_deltas": weights[
            "scaled_predicted_masked_margin_deltas"
        ],
        "predicted_full_category_deltas": weights[
            "scaled_predicted_full_response_margin_deltas"
        ],
        "actual_masked_mean_margin_delta": masked_mean,
        "actual_masked_category_mean_margin_deltas": masked_categories,
        "actual_full_response_mean_margin_delta": full_mean,
        "actual_full_response_category_mean_margin_deltas": full_categories,
        "chosen_full_logp_delta": chosen_full_delta,
        "post_score_sha256": sha256(args.post_score_json),
        "composition_manifest_sha256": sha256(args.composition_manifest),
        "test_data_access": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
