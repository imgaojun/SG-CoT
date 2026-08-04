#!/usr/bin/env python3
"""Build E132 train-only empirical trigger lexicons and unseen synthesis requests."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

IRREGULAR = {
    "arose": "arise",
    "began": "begin",
    "broke": "break",
    "brought": "bring",
    "built": "build",
    "bought": "buy",
    "came": "come",
    "caught": "catch",
    "chose": "choose",
    "did": "do",
    "died": "die",
    "drew": "draw",
    "driven": "drive",
    "drove": "drive",
    "fell": "fall",
    "fought": "fight",
    "found": "find",
    "gave": "give",
    "gone": "go",
    "grew": "grow",
    "held": "hold",
    "kept": "keep",
    "knew": "know",
    "laid": "lay",
    "led": "lead",
    "left": "leave",
    "lost": "lose",
    "made": "make",
    "met": "meet",
    "paid": "pay",
    "ran": "run",
    "rose": "rise",
    "said": "say",
    "saw": "see",
    "sent": "send",
    "shot": "shoot",
    "spoke": "speak",
    "stood": "stand",
    "taught": "teach",
    "thought": "think",
    "told": "tell",
    "took": "take",
    "went": "go",
    "wore": "wear",
    "wrote": "write",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_surface(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().strip())


def lemma_token(token: str) -> str:
    if token in IRREGULAR:
        return IRREGULAR[token]
    if len(token) > 4 and token.endswith("ied"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        base = token[:-3]
        if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        if base.endswith(("at", "bl", "iz")):
            base += "e"
        return base
    if len(token) > 4 and token.endswith("eed"):
        return token[:-1]
    if len(token) > 4 and token.endswith("ed"):
        base = token[:-2]
        if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        if base.endswith(("at", "bl", "iz")):
            base += "e"
        return base
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "shes", "ches", "xes", "zes", "oes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def simple_lemma(value: str) -> str:
    surface = normalize_surface(value)
    pieces = re.findall(r"[a-z0-9]+|[^a-z0-9\s]+", surface)
    return " ".join(lemma_token(piece) if piece.isalnum() else piece for piece in pieces)


def rank_counts(counts: collections.Counter[str], minimum: int, top_k: int) -> list[dict[str, Any]]:
    return [
        {"cue": cue, "count": count}
        for cue, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= minimum
    ][:top_k]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("id") != "e132_trigger_cue_enrichment_v1":
        raise ValueError("unexpected protocol id")
    source_path = REPO_ROOT / protocol["source_manifest"]
    schema_path = REPO_ROOT / protocol["schema"]
    heldout_path = REPO_ROOT / protocol["heldout_types"]
    rows = load_jsonl(source_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    heldout = set(json.loads(heldout_path.read_text(encoding="utf-8")))
    if len(rows) != 1500:
        raise ValueError(f"expected 1500 frozen E81 rows, got {len(rows)}")
    if len(heldout) != int(protocol["unseen_types_expected"]):
        raise ValueError(f"unexpected heldout count: {len(heldout)}")

    surfaces: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    lemmas: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    source_windows: dict[str, set[str]] = collections.defaultdict(set)
    event_mentions = 0
    for row in rows:
        output = json.loads(row["output"]) if isinstance(row["output"], str) else row["output"]
        window_id = str((row.get("meta") or {}).get("wnd_id", ""))
        for event in output.get("events", []):
            event_type = event.get("event_type")
            trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
            cue = normalize_surface(str(trigger.get("text") or ""))
            if not event_type or not cue:
                continue
            if event_type in heldout:
                raise ValueError(f"heldout type leaked into empirical trigger supervision: {event_type}")
            surfaces[event_type][cue] += 1
            lemmas[event_type][simple_lemma(cue)] += 1
            source_windows[event_type].add(window_id)
            event_mentions += 1

    schema_types = {entry["event_type"] for entry in schema}
    if not heldout <= schema_types:
        raise ValueError("heldout type missing from schema")
    lexicon = {
        "id": "e132_seen_trigger_lexicon_v1",
        "source_manifest": protocol["source_manifest"],
        "lemma_method": protocol["lemma_method"],
        "types": {},
    }
    requests = []
    for entry in sorted(schema, key=lambda item: item["event_type"]):
        event_type = entry["event_type"]
        if event_type in heldout:
            requests.append(
                {
                    "event_type": event_type,
                    "definition": entry.get("definition", ""),
                    "core_roles": entry.get("core_roles") or [],
                    "original_trigger_cues": entry.get("trigger_cues") or [],
                    "requested_trigger_cues_min": int(protocol["synthetic_cues_min"]),
                    "requested_trigger_cues_max": int(protocol["synthetic_cues_max"]),
                    "requested_examples": int(protocol["synthetic_examples_per_unseen_type"]),
                }
            )
            continue
        surface_ranked = rank_counts(
            surfaces[event_type],
            int(protocol["empirical_surface_min_count"]),
            int(protocol["empirical_surface_top_k"]),
        )
        lemma_ranked = rank_counts(
            lemmas[event_type],
            int(protocol["empirical_surface_min_count"]),
            int(protocol["empirical_lemma_top_k"]),
        )
        lexicon["types"][event_type] = {
            "source": "e81_train_gold_triggers",
            "surface_cues": surface_ranked,
            "lemma_cues": lemma_ranked,
            "fallback_schema_cues": entry.get("trigger_cues") or [],
            "observed_mentions": sum(surfaces[event_type].values()),
            "observed_windows": len(source_windows[event_type]),
            "used_schema_fallback": not bool(surface_ranked),
        }

    args.output_dir.mkdir(parents=True)
    lexicon_path = args.output_dir / "seen_trigger_lexicon.json"
    requests_path = args.output_dir / "unseen_schema_synthesis_requests.jsonl"
    lexicon_path.write_text(json.dumps(lexicon, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with requests_path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    audit = {
        "id": "e132_trigger_lexicon_build_audit_v1",
        "source_rows": len(rows),
        "event_mentions": event_mentions,
        "schema_types": len(schema_types),
        "seen_types": len(schema_types - heldout),
        "unseen_types": len(heldout),
        "unseen_types_list": sorted(heldout),
        "empirical_heldout_mentions": 0,
        "synthesis_requests": len(requests),
        "seen_types_with_surface_cues": sum(
            bool(value["surface_cues"]) for value in lexicon["types"].values()
        ),
        "seen_types_using_schema_fallback": sum(
            bool(value["used_schema_fallback"]) for value in lexicon["types"].values()
        ),
        "test_rows_read": 0,
        "input_sha256": {
            "source_manifest": sha256_file(source_path),
            "schema": sha256_file(schema_path),
            "heldout_types": sha256_file(heldout_path),
        },
        "output_sha256": {
            "seen_trigger_lexicon": sha256_file(lexicon_path),
            "unseen_schema_synthesis_requests": sha256_file(requests_path),
        },
        "passed": len(requests) == int(protocol["unseen_types_expected"]),
    }
    (args.output_dir / "build_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
