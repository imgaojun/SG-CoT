#!/usr/bin/env python3
"""Compare E129 Direct, pure SG-CoT, and mixed direct-decode development runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.stage2_quality_validation.event_metrics import normalize_events, prf


METRICS = ("argument", "event", "trigger")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("meta", {}).get("wnd_id", ""))


def assert_aligned(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], label: str) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"{label} row count mismatch: {len(reference)} != {len(candidate)}")
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if row_id(left) != row_id(right) or left.get("input") != right.get("input") or left.get("gold") != right.get("gold"):
            raise ValueError(f"{label} alignment mismatch at row {index}")


def union_sets(rows: list[dict[str, Any]], field: str) -> tuple[set, set, set]:
    trigger, argument, event = set(), set(), set()
    for index, row in enumerate(rows):
        payload = row.get(field) or {"events": []}
        current = normalize_events(payload)
        trigger.update((index, *item) for item in current[0])
        argument.update((index, *item) for item in current[1])
        event.update((index, *item) for item in current[2])
    return trigger, argument, event


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    pred_sets = union_sets(rows, "predicted")
    gold_sets = union_sets(rows, "gold")
    micro = {
        metric: prf(pred_sets[index], gold_sets[index])
        for index, metric in enumerate(("trigger", "argument", "event"))
    }
    predicted_events = sum(len((row.get("predicted") or {}).get("events", [])) for row in rows)
    gold_events = sum(len((row.get("gold") or {}).get("events", [])) for row in rows)
    return {
        "rows": total,
        "macro": {
            metric: sum(float(row.get(f"{metric}_f1", 0.0) or 0.0) for row in rows) / total
            for metric in METRICS
        },
        "micro": micro,
        "json_valid_rate": sum(bool(row.get("valid_json")) for row in rows) / total,
        "candidate_type_valid_rate": sum(bool(row.get("candidate_types_valid")) for row in rows) / total,
        "predicted_events": predicted_events,
        "gold_events": gold_events,
        "predicted_to_gold_event_ratio": predicted_events / gold_events if gold_events else 0.0,
    }


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_delta": {
            metric: candidate["macro"][metric] - reference["macro"][metric] for metric in METRICS
        },
        "micro_f1_delta": {
            metric: candidate["micro"][metric]["f1"] - reference["micro"][metric]["f1"]
            for metric in METRICS
        },
        "trigger_precision_delta": candidate["micro"]["trigger"]["p"]
        - reference["micro"]["trigger"]["p"],
        "trigger_recall_delta": candidate["micro"]["trigger"]["r"]
        - reference["micro"]["trigger"]["r"],
        "json_valid_rate_delta": candidate["json_valid_rate"] - reference["json_valid_rate"],
        "event_ratio_multiplier": (
            candidate["predicted_to_gold_event_ratio"] / reference["predicted_to_gold_event_ratio"]
            if reference["predicted_to_gold_event_ratio"]
            else 0.0
        ),
    }


def evaluate_gate(splits: dict[str, Any]) -> dict[str, Any]:
    seen_delta = splits["seen"]["mixed_vs_direct"]["macro_delta"]
    unseen_delta = splits["unseen"]["mixed_vs_direct"]["macro_delta"]
    nonnegative_cells = sum(value >= 0.0 for value in [*seen_delta.values(), *unseen_delta.values()])
    checks = {
        "unseen_all_three_macro_positive": all(value > 0.0 for value in unseen_delta.values()),
        "seen_each_macro_delta_at_least_minus_0_01": all(value >= -0.01 for value in seen_delta.values()),
        "at_least_four_of_six_macro_cells_nonnegative": nonnegative_cells >= 4,
        "unseen_trigger_precision_drop_at_most_0_01": splits["unseen"]["mixed_vs_direct"][
            "trigger_precision_delta"
        ]
        >= -0.01,
        "unseen_event_ratio_inflation_at_most_1_05x": splits["unseen"]["mixed_vs_direct"][
            "event_ratio_multiplier"
        ]
        <= 1.05,
        "seen_json_drop_at_most_0_01": splits["seen"]["mixed_vs_direct"][
            "json_valid_rate_delta"
        ]
        >= -0.01,
        "unseen_json_drop_at_most_0_01": splits["unseen"]["mixed_vs_direct"][
            "json_valid_rate_delta"
        ]
        >= -0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "nonnegative_macro_cells": nonnegative_cells,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# E129 Nested Strict Development Comparison",
        "",
        f"Gate passed: **{str(payload['gate']['passed']).lower()}**",
        "",
        "| split | method | Argument | Event | Trigger | Trigger P/R | pred/gold events | JSON |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("seen", "unseen"):
        for method in ("direct", "pure", "mixed"):
            row = payload["splits"][split][method]
            lines.append(
                f"| {split} | {method} | {row['macro']['argument']:.4f} | "
                f"{row['macro']['event']:.4f} | {row['macro']['trigger']:.4f} | "
                f"{row['micro']['trigger']['p']:.4f}/{row['micro']['trigger']['r']:.4f} | "
                f"{row['predicted_to_gold_event_ratio']:.3f} | {row['json_valid_rate']:.4f} |"
            )
    lines.extend(["", "## Gate Checks", ""])
    for name, passed in payload["gate"]["checks"].items():
        lines.append(f"- `{name}`: `{str(passed).lower()}`")
    lines.extend(["", "## Decision", "", payload["decision"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    for method in ("direct", "pure", "mixed"):
        for split in ("seen", "unseen"):
            parser.add_argument(f"--{method}_{split}", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    raw = {
        split: {
            method: load_jsonl(getattr(args, f"{method}_{split}") / "predictions.jsonl")
            for method in ("direct", "pure", "mixed")
        }
        for split in ("seen", "unseen")
    }
    for split in ("seen", "unseen"):
        assert_aligned(raw[split]["direct"], raw[split]["pure"], f"{split}/pure")
        assert_aligned(raw[split]["direct"], raw[split]["mixed"], f"{split}/mixed")

    splits: dict[str, Any] = {}
    for split in ("seen", "unseen"):
        splits[split] = {method: summarize(raw[split][method]) for method in ("direct", "pure", "mixed")}
        splits[split]["pure_vs_direct"] = compare(splits[split]["direct"], splits[split]["pure"])
        splits[split]["mixed_vs_direct"] = compare(splits[split]["direct"], splits[split]["mixed"])

    gate = evaluate_gate(splits)
    decision = (
        "Advance mixed co-training to two additional strict nested folds and three seeds."
        if gate["passed"]
        else "Do not scale this mixed recipe. Use the failed checks to design the Direct-anchored arbitration branch."
    )
    payload = {
        "id": "e129_nested_strict_development_comparison_v1",
        "splits": splits,
        "gate": gate,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "comparison.json", payload)
    (args.output_dir / "comparison.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if gate["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
