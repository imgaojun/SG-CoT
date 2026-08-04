#!/usr/bin/env python3
"""Apply E132 input enrichment to frozen E95 train/dev traces without changing targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_PATH = Path(__file__).resolve().parents[1]
if REPO_PATH.as_posix() not in sys.path:
    sys.path.insert(0, REPO_PATH.as_posix())

from scripts.build_e132_enriched_train_manifest_20260713 import (
    REPO_ROOT,
    SCHEMA_MARKER,
    load_jsonl,
    render_cards,
    render_cards_compact,
    sha256_file,
)


TRAIN_NAME = "e132_enriched_e95_frozen_train1320"
DEV_NAME = "e132_enriched_e95_frozen_dev_seen197"


def normalized_digest(rows: list[dict[str, Any]], key: str) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row[key], ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def transform_rows(
    rows: list[dict[str, Any]],
    schema_by_type: dict[str, dict[str, Any]],
    lexicon: dict[str, dict[str, Any]],
    unseen_cards: dict[str, dict[str, Any]],
    learned_max: int,
    examples_max: int,
    render_mode: str,
    unseen_max: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    transformed = []
    counters = {"candidate_cards": 0, "unseen_candidate_cards": 0, "changed_inputs": 0}
    for source in rows:
        if SCHEMA_MARKER not in source["input"]:
            raise ValueError("frozen E95 row is missing schema cards")
        prefix, _old_cards = source["input"].split(SCHEMA_MARKER, 1)
        candidates = list(source["meta"]["candidate_types"])
        if render_mode == "additive_v1":
            cards = render_cards(
                candidates,
                schema_by_type,
                lexicon,
                unseen_cards,
                learned_max,
                examples_max,
            )
            enrichment_id = "empirical_seen_synthetic_unseen_v1"
        elif render_mode in {"compact_v2", "compact_v3"}:
            cards = render_cards_compact(
                candidates,
                schema_by_type,
                lexicon,
                unseen_cards,
                learned_max,
                unseen_max,
                examples_max,
            )
            enrichment_id = (
                "compact_empirical_seen_synthetic_unseen_v2"
                if render_mode == "compact_v2"
                else "compact_empirical_seen_synthetic_unseen_v3"
            )
        else:
            raise ValueError(f"unknown render mode: {render_mode}")
        record = json.loads(json.dumps(source, ensure_ascii=False))
        record["input"] = prefix + SCHEMA_MARKER + cards
        record["meta"]["e132_schema_enrichment"] = enrichment_id
        record["meta"]["e132_unseen_candidate_count"] = sum(
            candidate in unseen_cards for candidate in candidates
        )
        counters["candidate_cards"] += len(candidates)
        counters["unseen_candidate_cards"] += record["meta"]["e132_unseen_candidate_count"]
        counters["changed_inputs"] += int(record["input"] != source["input"])
        transformed.append(record)
    return transformed, counters


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--unseen_cards", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--render_mode",
        choices=("additive_v1", "compact_v2", "compact_v3"),
        default="additive_v1",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("id") != "e132_trigger_cue_enrichment_v1":
        raise ValueError("unexpected protocol id")
    train_source = REPO_ROOT / protocol["frozen_e95_train"]
    dev_source = REPO_ROOT / protocol["frozen_e95_dev_seen"]
    if sha256_file(train_source) != protocol["frozen_e95_train_sha256"]:
        raise ValueError("frozen E95 train hash mismatch")
    if sha256_file(dev_source) != protocol["frozen_e95_dev_seen_sha256"]:
        raise ValueError("frozen E95 dev-seen hash mismatch")
    output_paths = {
        "train": args.output_dir / f"{TRAIN_NAME}.jsonl",
        "dev_seen": args.output_dir / f"{DEV_NAME}.jsonl",
        "registry": args.output_dir / "dataset_info.json",
        "audit": args.output_dir / "frozen_e95_enrichment_audit.json",
    }
    if any(path.exists() for path in output_paths.values()):
        raise SystemExit("refusing to overwrite frozen E132 E95 artifacts")

    schema = json.loads((REPO_ROOT / protocol["schema"]).read_text(encoding="utf-8"))
    schema_by_type = {entry["event_type"]: entry for entry in schema}
    lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"))["types"]
    unseen_cards = {
        entry["event_type"]: entry for entry in load_jsonl(args.unseen_cards)
    }
    sources = {"train": load_jsonl(train_source), "dev_seen": load_jsonl(dev_source)}
    transformed = {}
    counters = {}
    for split, rows in sources.items():
        learned_max = int(protocol["render_learned_cues_max"])
        unseen_max = int(protocol["render_unseen_cues_max"])
        examples_max = int(protocol["render_synthetic_examples"])
        if args.render_mode == "compact_v3":
            learned_max = int(protocol["compact_v3_learned_cues_max"])
            unseen_max = int(protocol["compact_v3_unseen_cues_max"])
            examples_max = int(protocol["compact_v3_examples"])
        transformed[split], counters[split] = transform_rows(
            rows,
            schema_by_type,
            lexicon,
            unseen_cards,
            learned_max,
            examples_max,
            args.render_mode,
            unseen_max,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_paths["train"], transformed["train"])
    write_jsonl(output_paths["dev_seen"], transformed["dev_seen"])
    registry = {
        TRAIN_NAME: {
            "file_name": output_paths["train"].name,
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        DEV_NAME: {
            "file_name": output_paths["dev_seen"].name,
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    output_paths["registry"].write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "id": "e132_frozen_e95_input_only_enrichment_audit_v1",
        "render_mode": args.render_mode,
        "train_rows": len(sources["train"]),
        "dev_seen_rows": len(sources["dev_seen"]),
        "train_counters": counters["train"],
        "dev_seen_counters": counters["dev_seen"],
        "instruction_exact": all(
            source["instruction"] == target["instruction"]
            for split in sources
            for source, target in zip(sources[split], transformed[split])
        ),
        "output_exact": all(
            source["output"] == target["output"]
            for split in sources
            for source, target in zip(sources[split], transformed[split])
        ),
        "gold_output_exact": all(
            source.get("gold_output") == target.get("gold_output")
            for split in sources
            for source, target in zip(sources[split], transformed[split])
        ),
        "row_order_exact": all(
            source["meta"]["wnd_id"] == target["meta"]["wnd_id"]
            for split in sources
            for source, target in zip(sources[split], transformed[split])
        ),
        "candidate_order_exact": all(
            source["meta"]["candidate_types"] == target["meta"]["candidate_types"]
            for split in sources
            for source, target in zip(sources[split], transformed[split])
        ),
        "response_normalized_sha256": {
            split: normalized_digest(rows, "output") for split, rows in transformed.items()
        },
        "test_rows_read": 0,
        "output_sha256": {
            key: sha256_file(path) for key, path in output_paths.items() if key != "audit"
        },
    }
    audit["passed"] = all(
        (
            audit["train_rows"] == 1320,
            audit["dev_seen_rows"] == 197,
            audit["train_counters"]["changed_inputs"] == 1320,
            audit["dev_seen_counters"]["changed_inputs"] == 197,
            audit["instruction_exact"],
            audit["output_exact"],
            audit["gold_output_exact"],
            audit["row_order_exact"],
            audit["candidate_order_exact"],
            audit["test_rows_read"] == 0,
        )
    )
    output_paths["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
