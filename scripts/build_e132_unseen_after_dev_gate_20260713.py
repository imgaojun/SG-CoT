#!/usr/bin/env python3
"""Materialize E132 unseen inputs only after its frozen seen-development gate passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_e132_enriched_frozen_e95_20260713 import (
    load_jsonl,
    sha256_file,
    transform_rows,
    write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def require_dev_gate(path: Path) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("id") != "e132_dev_seen_effectiveness_gate_v1" or not gate.get("passed"):
        raise ValueError("a passing frozen E132 dev-seen gate is required")
    if int(gate.get("test_rows_read", -1)) != 0:
        raise ValueError("dev gate reports unexpected test access")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dev_gate_json", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--unseen_cards", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("id") != "e132_trigger_cue_enrichment_v1":
        raise ValueError("unexpected protocol id")
    require_dev_gate(args.dev_gate_json)
    section = protocol["effectiveness_evaluation"]["test_unseen_after_dev_pass"]
    source_path = REPO_ROOT / section["source_rows"]
    output_path = REPO_ROOT / section["candidate_rows"]
    if args.output_dir.resolve() != output_path.parent.resolve():
        raise ValueError("output directory differs from frozen protocol")
    if sha256_file(source_path) != section["source_rows_sha256"]:
        raise ValueError("unseen source rows hash mismatch")
    sources = load_jsonl(source_path)
    if len(sources) != int(section["rows"]):
        raise ValueError("unseen source row count mismatch")

    schema = json.loads((REPO_ROOT / protocol["schema"]).read_text(encoding="utf-8"))
    schema_by_type = {entry["event_type"]: entry for entry in schema}
    lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"))["types"]
    unseen_cards = {
        entry["event_type"]: entry for entry in load_jsonl(args.unseen_cards)
    }
    transformed, counters = transform_rows(
        sources,
        schema_by_type,
        lexicon,
        unseen_cards,
        int(protocol["compact_v3_learned_cues_max"]),
        int(protocol["compact_v3_examples"]),
        "compact_v3",
        int(protocol["compact_v3_unseen_cues_max"]),
    )
    checks = {
        "row_count_exact": len(transformed) == int(section["rows"]),
        "all_inputs_changed": counters["changed_inputs"] == len(sources),
        "instruction_exact": all(a["instruction"] == b["instruction"] for a, b in zip(sources, transformed)),
        "output_exact": all(a["output"] == b["output"] for a, b in zip(sources, transformed)),
        "gold_exact": all(a.get("gold_output") == b.get("gold_output") for a, b in zip(sources, transformed)),
        "row_order_exact": all(a["meta"]["wnd_id"] == b["meta"]["wnd_id"] for a, b in zip(sources, transformed)),
        "candidate_order_exact": all(a["meta"]["candidate_types"] == b["meta"]["candidate_types"] for a, b in zip(sources, transformed)),
    }
    args.output_dir.mkdir(parents=True)
    write_jsonl(output_path, transformed)
    audit = {
        "id": "e132_unseen_after_dev_gate_build_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "rows": len(transformed),
        "counters": counters,
        "test_rows_read": len(sources),
        "input_sha256": {
            "source": section["source_rows_sha256"],
            "dev_gate": sha256_file(args.dev_gate_json),
        },
        "output_sha256": sha256_file(output_path),
    }
    (args.output_dir / "build_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
