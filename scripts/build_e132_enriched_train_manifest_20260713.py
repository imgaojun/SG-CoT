#!/usr/bin/env python3
"""Render E132 cue enrichment into the frozen E81 train manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_MARKER = "\n\nSchema cards:\n"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(row: dict[str, Any], seed: int) -> str:
    meta = row.get("meta") or {}
    identity = f"{seed}|{meta.get('doc_id', '')}|{meta.get('wnd_id', '')}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def learned_cues(entry: dict[str, Any], maximum: int) -> list[str]:
    values = []
    seen = set()
    for item in entry.get("surface_cues", []) + entry.get("lemma_cues", []):
        cue = str(item.get("cue") or "").strip()
        key = cue.casefold()
        if cue and key not in seen:
            values.append(cue)
            seen.add(key)
        if len(values) >= maximum:
            break
    if not values:
        values = [str(cue) for cue in entry.get("fallback_schema_cues", [])[:maximum]]
    return values


def render_cards(
    candidate_types: list[str],
    schema_by_type: dict[str, dict[str, Any]],
    seen_lexicon: dict[str, dict[str, Any]],
    unseen_cards: dict[str, dict[str, Any]],
    learned_max: int,
    examples_max: int,
) -> str:
    blocks = []
    for index, event_type in enumerate(candidate_types, start=1):
        schema = schema_by_type[event_type]
        lines = [
            f"[{index}] Event type: {event_type}",
            f"Definition: {schema.get('definition', '')}",
            "Trigger cues: " + ", ".join(schema.get("trigger_cues") or []),
        ]
        if event_type in unseen_cards:
            card = unseen_cards[event_type]
            lines.append("Induced trigger cues: " + ", ".join(card["trigger_cues"]))
            examples = card["examples"][:examples_max]
            lines.append(
                "Synthetic trigger examples: "
                + " | ".join(
                    f"{example['trigger']} -> {example['sentence']}" for example in examples
                )
            )
        else:
            lines.append(
                "Learned train trigger forms: "
                + ", ".join(learned_cues(seen_lexicon[event_type], learned_max))
            )
        lines.append("Core roles: " + ", ".join(schema.get("core_roles") or []))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def merge_cues(original: list[str], additions: list[str], maximum: int) -> list[str]:
    merged = []
    seen = set()
    for cue in original + additions:
        value = str(cue).strip()
        key = value.casefold()
        if value and key not in seen:
            merged.append(value)
            seen.add(key)
        if len(merged) >= maximum:
            break
    return merged


def render_cards_compact(
    candidate_types: list[str],
    schema_by_type: dict[str, dict[str, Any]],
    seen_lexicon: dict[str, dict[str, Any]],
    unseen_cards: dict[str, dict[str, Any]],
    learned_max: int,
    unseen_max: int,
    examples_max: int,
) -> str:
    blocks = []
    for index, event_type in enumerate(candidate_types, start=1):
        schema = schema_by_type[event_type]
        original = list(schema.get("trigger_cues") or [])
        if event_type in unseen_cards:
            card = unseen_cards[event_type]
            cues = merge_cues(original, card["trigger_cues"], unseen_max)
        else:
            cues = merge_cues(
                original,
                learned_cues(seen_lexicon[event_type], learned_max),
                learned_max,
            )
        lines = [
            f"[{index}] Event type: {event_type}",
            f"Definition: {schema.get('definition', '')}",
            "Trigger cues: " + ", ".join(cues),
        ]
        if event_type in unseen_cards:
            examples = unseen_cards[event_type]["examples"][:examples_max]
            lines.append(
                "Examples: "
                + " | ".join(
                    f"{example['trigger']} -> {example['sentence']}" for example in examples
                )
            )
        lines.append("Core roles: " + ", ".join(schema.get("core_roles") or []))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--unseen_cards", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    full_path = args.output_dir / "e132_enriched_e81_train1500.jsonl"
    smoke_path = args.output_dir / "e132_enriched_e81_train_smoke40.jsonl"
    audit_path = args.output_dir / "enriched_manifest_audit.json"
    if any(path.exists() for path in (full_path, smoke_path, audit_path)):
        raise SystemExit("refusing to overwrite an existing E132 enriched manifest artifact")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_path = REPO_ROOT / protocol["source_manifest"]
    schema_path = REPO_ROOT / protocol["schema"]
    rows = load_jsonl(source_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"))["types"]
    unseen_payloads = load_jsonl(args.unseen_cards)
    schema_by_type = {entry["event_type"]: entry for entry in schema}
    unseen_cards = {entry["event_type"]: entry for entry in unseen_payloads}
    expected_unseen = set(
        json.loads((REPO_ROOT / protocol["heldout_types"]).read_text(encoding="utf-8"))
    )
    if set(unseen_cards) != expected_unseen:
        raise ValueError("accepted unseen cards do not exactly cover the frozen heldout types")
    if set(lexicon) != set(schema_by_type) - expected_unseen:
        raise ValueError("seen lexicon does not exactly cover the non-heldout schema")

    enriched = []
    changed_inputs = 0
    candidate_card_count = 0
    unseen_card_count = 0
    for source in rows:
        if SCHEMA_MARKER not in source["input"]:
            raise ValueError("source input is missing the schema-card marker")
        prefix, _old_cards = source["input"].split(SCHEMA_MARKER, 1)
        candidates = list(source["meta"]["candidate_types"])
        if len(candidates) != len(set(candidates)):
            raise ValueError("duplicate candidate type in source row")
        if not set(candidates) <= set(schema_by_type):
            raise ValueError("candidate type missing from schema")
        cards = render_cards(
            candidates,
            schema_by_type,
            lexicon,
            unseen_cards,
            int(protocol["render_learned_cues_max"]),
            int(protocol["render_synthetic_examples"]),
        )
        record = json.loads(json.dumps(source, ensure_ascii=False))
        record["input"] = prefix + SCHEMA_MARKER + cards
        record["meta"]["e132_schema_enrichment"] = "empirical_seen_synthetic_unseen_v1"
        record["meta"]["e132_unseen_candidate_count"] = sum(
            candidate in expected_unseen for candidate in candidates
        )
        changed_inputs += int(record["input"] != source["input"])
        candidate_card_count += len(candidates)
        unseen_card_count += record["meta"]["e132_unseen_candidate_count"]
        enriched.append(record)

    smoke_count = int(protocol["trace_generation_smoke_rows"])
    smoke_seed = int(protocol["trace_generation_smoke_seed"])
    smoke = sorted(
        enriched,
        key=lambda row: (
            -row["meta"]["e132_unseen_candidate_count"],
            stable_key(row, smoke_seed),
        ),
    )[:smoke_count]
    if len({row["meta"]["wnd_id"] for row in smoke}) != smoke_count:
        raise ValueError("smoke selection contains duplicate window ids")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in ((full_path, enriched), (smoke_path, smoke)):
        with path.open("w", encoding="utf-8") as handle:
            for row in payload:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "id": "e132_enriched_train_manifest_audit_v1",
        "source_rows": len(rows),
        "enriched_rows": len(enriched),
        "changed_inputs": changed_inputs,
        "outputs_exact": all(
            left["output"] == right["output"] for left, right in zip(rows, enriched)
        ),
        "row_order_exact": all(
            left["meta"]["wnd_id"] == right["meta"]["wnd_id"]
            for left, right in zip(rows, enriched)
        ),
        "candidate_order_exact": all(
            left["meta"]["candidate_types"] == right["meta"]["candidate_types"]
            for left, right in zip(rows, enriched)
        ),
        "candidate_cards": candidate_card_count,
        "unseen_candidate_cards": unseen_card_count,
        "smoke_rows": len(smoke),
        "smoke_min_unseen_candidates": min(
            row["meta"]["e132_unseen_candidate_count"] for row in smoke
        ),
        "test_rows_read": 0,
        "output_sha256": {
            "full": sha256_file(full_path),
            "smoke": sha256_file(smoke_path),
        },
    }
    audit["passed"] = all(
        (
            audit["source_rows"] == 1500,
            audit["enriched_rows"] == 1500,
            audit["changed_inputs"] == 1500,
            audit["outputs_exact"],
            audit["row_order_exact"],
            audit["candidate_order_exact"],
            audit["smoke_rows"] == smoke_count,
            audit["test_rows_read"] == 0,
        )
    )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
