#!/usr/bin/env python3
"""Validate the preregistered E121 type rule and materialize its five splits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.data_preprocessing.type_holdout.generate_type_holdout import (  # noqa: E402
    generate_dataset_protocol,
    load_jsonl,
)


def family(event_type: str) -> str:
    return event_type.split(":", 1)[0]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def derive_types(
    train_rows: list[dict[str, Any]],
    schema_types: list[str],
    prior_heldout: set[str],
    minimum: int,
    maximum: int,
) -> tuple[list[str], dict[str, Any]]:
    counts = Counter(
        event["event_type"]
        for row in train_rows
        for event in row.get("event_mentions", [])
    )
    by_family: dict[str, list[str]] = defaultdict(list)
    for event_type in schema_types:
        by_family[family(event_type)].append(event_type)

    eligible: dict[str, list[str]] = {}
    selected = []
    for coarse_family in sorted(by_family):
        members = sorted(by_family[coarse_family])
        candidates = [
            event_type
            for event_type in members
            if event_type not in prior_heldout
            and minimum <= counts[event_type] <= maximum
            and len(members) >= 2
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda event_type: (-counts[event_type], event_type))
        eligible[coarse_family] = candidates
        selected.append(candidates[0])
    return sorted(selected), {
        "train_type_counts": dict(sorted(counts.items())),
        "eligible_by_family": eligible,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol_config",
        type=Path,
        default=Path("configs/seen_unseen_type_holdout_protocols.json"),
    )
    parser.add_argument("--input_root", type=Path, default=Path("data/processed/textee"))
    parser.add_argument(
        "--output_root", type=Path, default=Path("data/processed/type_holdout")
    )
    parser.add_argument("--dataset", default="richere-en")
    parser.add_argument("--protocol", default="balanced-subtype-v2-confirmation")
    parser.add_argument(
        "--schema_path", type=Path, default=Path("data/schema/richere-en.event_schema.json")
    )
    parser.add_argument("--selection_split", default="split1")
    parser.add_argument("--min_train_mentions", type=int, default=20)
    parser.add_argument("--max_train_mentions", type=int, default=250)
    parser.add_argument("--audit_output", type=Path, required=True)
    args = parser.parse_args()

    protocol_map = load_json(args.protocol_config)
    dataset_protocols = protocol_map.get(args.dataset, {})
    configured = sorted(dataset_protocols.get(args.protocol, []))
    if not configured:
        raise ValueError(f"missing configured protocol {args.dataset}/{args.protocol}")

    prior_protocols = {
        name: sorted(types)
        for name, types in dataset_protocols.items()
        if name != args.protocol
    }
    prior_heldout = {event_type for values in prior_protocols.values() for event_type in values}
    schema = load_json(args.schema_path)
    schema_types = sorted(entry["event_type"] for entry in schema)
    train_path = args.input_root / args.dataset / args.selection_split / "train.jsonl"
    train_rows = load_jsonl(train_path)
    derived, diagnostics = derive_types(
        train_rows,
        schema_types,
        prior_heldout,
        args.min_train_mentions,
        args.max_train_mentions,
    )
    if configured != derived:
        raise ValueError(
            "configured E121 types do not match the preregistered rule: "
            f"configured={configured}, derived={derived}"
        )

    for event_type in configured:
        siblings = [
            candidate
            for candidate in schema_types
            if family(candidate) == family(event_type) and candidate not in configured
        ]
        if not siblings:
            raise ValueError(f"held-out type has no remaining seen sibling: {event_type}")

    generate_dataset_protocol(
        input_root=args.input_root,
        output_root=args.output_root,
        dataset=args.dataset,
        protocol=args.protocol,
        unseen_types=configured,
    )
    report = {
        "dataset": args.dataset,
        "protocol": args.protocol,
        "selection_split": args.selection_split,
        "selection_rule": {
            "exclude_types_used_by_prior_richere_holdouts": True,
            "require_remaining_seen_sibling": True,
            "min_train_mentions": args.min_train_mentions,
            "max_train_mentions": args.max_train_mentions,
            "per_family_choice": "highest_split1_train_frequency_then_lexicographic",
            "uses_model_predictions": False,
            "uses_test_metrics": False,
        },
        "prior_protocols": prior_protocols,
        "selected_types": configured,
        **diagnostics,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
