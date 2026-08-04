#!/usr/bin/env python3
"""Rescore saved N-path samples with the shared surface recovery and fixed voting."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.rescore_surface_predictions_20260712 import (  # noqa: E402
    f1_from_counts,
    micro_counts,
)
from src.stage2_preference.reasoning_preference import (  # noqa: E402
    metric_f1s,
    recover_offsets_from_evidence,
)


EventKey = tuple[int | None, int | None, str | None]
ArgumentKey = tuple[str | None, int | None, int | None]
NormalizedEvent = tuple[EventKey, frozenset[ArgumentKey]]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(payload: dict[str, Any]) -> list[NormalizedEvent]:
    output = []
    for event in payload.get("events", []) if isinstance(payload, dict) else []:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        event_key: EventKey = (
            trigger.get("start"),
            trigger.get("end"),
            event.get("event_type"),
        )
        arguments = frozenset(
            (
                argument.get("role"),
                argument.get("start"),
                argument.get("end"),
            )
            for argument in event.get("arguments", [])
            if isinstance(argument, dict)
        )
        output.append((event_key, arguments))
    return output


def vote(sample_events: list[list[NormalizedEvent]], threshold: int) -> list[NormalizedEvent]:
    event_counts: collections.Counter[EventKey] = collections.Counter()
    argument_counts: dict[EventKey, collections.Counter[ArgumentKey]] = collections.defaultdict(
        collections.Counter
    )
    for events in sample_events:
        seen_events: set[EventKey] = set()
        for event_key, arguments in events:
            if event_key in seen_events:
                continue
            seen_events.add(event_key)
            event_counts[event_key] += 1
            for argument in arguments:
                argument_counts[event_key][argument] += 1
    return [
        (
            event_key,
            frozenset(
                argument
                for argument, count in argument_counts[event_key].items()
                if count >= threshold
            ),
        )
        for event_key, count in event_counts.items()
        if count >= threshold
    ]


def as_payload(events: list[NormalizedEvent]) -> dict[str, Any]:
    return {
        "events": [
            {
                "event_type": event_type,
                "trigger": {"start": start, "end": end, "text": None},
                "arguments": [
                    {"role": role, "start": arg_start, "end": arg_end, "text": None}
                    for role, arg_start, arg_end in sorted(
                        arguments,
                        key=lambda item: (
                            item[0] or "",
                            -1 if item[1] is None else int(item[1]),
                            -1 if item[2] is None else int(item[2]),
                        ),
                    )
                ],
            }
            for (start, end, event_type), arguments in events
        ]
    }


def aggregate(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    metric_names = ("argument", "event", "trigger")
    macro_values = {metric: [] for metric in metric_names}
    totals = {metric: [0, 0, 0] for metric in metric_names}
    for row in rows:
        predicted = row[prediction_key]
        gold = row["gold"]
        values = metric_f1s(predicted, gold)
        counts = micro_counts(predicted, gold)
        for metric in metric_names:
            macro_values[metric].append(values[metric])
            totals[metric] = [
                left + right for left, right in zip(totals[metric], counts[metric])
            ]
    return {
        "macro": {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in macro_values.items()
        },
        "corpus_micro": {
            metric: f1_from_counts(totals[metric]) for metric in metric_names
        },
        "micro_counts": totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_jsonl", type=Path, required=True)
    parser.add_argument("--eval_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--vote_k", type=int, default=3)
    args = parser.parse_args()

    saved = load_jsonl(args.samples_jsonl)
    eval_rows = load_jsonl(args.eval_jsonl)
    if len(saved) != len(eval_rows):
        raise ValueError(f"row count mismatch: {len(saved)} != {len(eval_rows)}")

    rescored = []
    sample_count = None
    greedy_missing_offsets = 0
    sampled_missing_offsets = 0
    for index, (saved_row, eval_row) in enumerate(zip(saved, eval_rows)):
        if int(saved_row.get("idx", index)) != index:
            raise ValueError(f"sample index mismatch at row {index}")
        gold_raw = eval_row.get("gold_output", eval_row["output"])
        gold = json.loads(gold_raw) if isinstance(gold_raw, str) else gold_raw
        greedy, greedy_diag = recover_offsets_from_evidence(
            saved_row.get("greedy_surface") or {"events": []}, eval_row["input"]
        )
        greedy_missing_offsets += int(greedy_diag.get("missing_offsets", 0))
        sample_payloads = []
        for surface in saved_row.get("sample_surfaces", []):
            recovered, diagnostics = recover_offsets_from_evidence(
                surface or {"events": []}, eval_row["input"]
            )
            sampled_missing_offsets += int(diagnostics.get("missing_offsets", 0))
            sample_payloads.append(recovered)
        if sample_count is None:
            sample_count = len(sample_payloads)
        elif sample_count != len(sample_payloads):
            raise ValueError(f"sample count mismatch at row {index}")
        voted = as_payload(vote([normalize(payload) for payload in sample_payloads], args.vote_k))
        rescored.append(
            {
                "idx": index,
                "meta": eval_row.get("meta", {}),
                "input": eval_row["input"],
                "gold": gold,
                "greedy_predicted": greedy,
                "predicted": voted,
            }
        )

    greedy_summary = aggregate(rescored, "greedy_predicted")
    vote_summary = aggregate(rescored, "predicted")
    summary = {
        "rows": len(rescored),
        "n_samples": sample_count or 0,
        "vote_k": args.vote_k,
        "greedy": greedy_summary,
        "vote": vote_summary,
        "macro_delta": {
            metric: vote_summary["macro"][metric] - greedy_summary["macro"][metric]
            for metric in ("argument", "event", "trigger")
        },
        "corpus_micro_delta": {
            metric: vote_summary["corpus_micro"][metric]
            - greedy_summary["corpus_micro"][metric]
            for metric in ("argument", "event", "trigger")
        },
        "greedy_missing_offsets": greedy_missing_offsets,
        "sampled_missing_offsets": sampled_missing_offsets,
        "recovery_implementation": "src.stage2_preference.reasoning_preference",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rescored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
