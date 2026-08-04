#!/usr/bin/env python3
"""Select an E129 continuation checkpoint using seen-development loss only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def latest_trainer_state(run_dir: Path) -> Path:
    root_state = run_dir / "trainer_state.json"
    if root_state.is_file():
        return root_state
    states = sorted(
        run_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.removeprefix("checkpoint-")),
    )
    if not states:
        raise ValueError(f"no trainer_state.json found under {run_dir}")
    return states[-1]


def select_checkpoint(
    run_dir: Path,
    *,
    metric: str = "eval_loss",
    expected_candidates: int | None = None,
) -> dict[str, Any]:
    state_path = latest_trainer_state(run_dir)
    state = load_json(state_path)
    candidates = []
    seen_steps: set[int] = set()
    for record in state.get("log_history", []):
        if not isinstance(record, dict) or metric not in record or "step" not in record:
            continue
        step = int(record["step"])
        value = float(record[metric])
        if step in seen_steps or not math.isfinite(value):
            raise ValueError(f"duplicate step or non-finite {metric}: {record}")
        seen_steps.add(step)
        checkpoint = run_dir / f"checkpoint-{step}"
        if checkpoint.is_dir():
            candidates.append(
                {
                    "checkpoint": checkpoint.name,
                    "step": step,
                    "epoch": record.get("epoch"),
                    metric: value,
                }
            )
    if expected_candidates is not None and len(candidates) != expected_candidates:
        raise ValueError(
            f"expected {expected_candidates} retained candidates, found {len(candidates)}"
        )
    if not candidates:
        raise ValueError(f"no retained checkpoints have a finite {metric}")
    selected = min(candidates, key=lambda row: (row[metric], row["step"]))
    return {
        "id": "e129_seen_loss_checkpoint_selection_v1",
        "run_dir": str(run_dir.resolve()),
        "trainer_state": str(state_path.resolve()),
        "selection_split": "dev_seen",
        "metric": metric,
        "greater_is_better": False,
        "candidates": candidates,
        "selected_checkpoint": selected["checkpoint"],
        "selected_step": selected["step"],
        "selected_metric": selected[metric],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--metric", default="eval_loss")
    parser.add_argument("--expected_candidates", type=int)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    payload = select_checkpoint(
        args.run_dir,
        metric=args.metric,
        expected_candidates=args.expected_candidates,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
