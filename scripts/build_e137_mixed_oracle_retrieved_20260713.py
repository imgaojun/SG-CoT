#!/usr/bin/env python3
"""Build and audit E137's train-only mixed oracle/retrieved SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


FINAL_RE = re.compile(r"<final>(.*?)</final>", re.DOTALL)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def final_event_types(row: dict[str, Any]) -> set[str]:
    output = str(row.get("output", ""))
    match = FINAL_RE.search(output)
    if not match or not output.lstrip().startswith("<thinking>"):
        raise ValueError("training output does not contain the required thinking/final contract")
    payload = json.loads(match.group(1))
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("final events is not a list")
    return {str(event["event_type"]) for event in events}


def validate_train_rows(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    seen: set[str] = set()
    mode_counts: Counter[str] = Counter()
    indices: list[int] = []
    for row in rows:
        if not {"instruction", "input", "output", "meta"}.issubset(row):
            raise ValueError(f"{family} row misses Alpaca fields")
        meta = row["meta"]
        if meta.get("source_part") != "train":
            raise ValueError(f"{family} contains non-train row")
        wnd_id = str(meta.get("wnd_id", ""))
        if not wnd_id or wnd_id in seen:
            raise ValueError(f"{family} has missing/duplicate wnd_id: {wnd_id}")
        seen.add(wnd_id)
        source_index = meta.get("e40_source_index")
        if not isinstance(source_index, int):
            raise ValueError(f"{family} row lacks integer e40_source_index")
        indices.append(source_index)
        candidates = {str(value) for value in meta.get("candidate_types", [])}
        if not final_event_types(row).issubset(candidates):
            raise ValueError(f"{family} final emits a type outside candidates: {wnd_id}")
        if family == "oracle":
            if not set(meta.get("gold_event_types", [])).issubset(candidates):
                raise ValueError(f"oracle row omits a gold type: {wnd_id}")
        else:
            mode = str(meta.get("e130_target_mode", ""))
            if mode not in {"gold_present", "partial_supported", "abstain"}:
                raise ValueError(f"invalid retrieved target mode: {mode}")
            mode_counts[mode] += 1
            if meta.get("e130_candidate_consistent") is not True:
                raise ValueError(f"retrieved row is not candidate-consistent: {wnd_id}")
    return {
        "rows": len(rows),
        "unique_wnd_ids": len(seen),
        "source_index_min": min(indices),
        "source_index_max": max(indices),
        "mode_counts": dict(sorted(mode_counts.items())),
    }


def tagged_rows(rows: list[dict[str, Any]], family: str, source_hash: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copied = dict(row)
        meta = dict(copied["meta"])
        meta["e137_candidate_regime"] = family
        meta["e137_source_sha256"] = source_hash
        copied["meta"] = meta
        output.append(copied)
    return output


def build_predicted_dev(
    direct_rows: list[dict[str, Any]], oracle_reference: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not oracle_reference:
        raise ValueError("oracle development reference is empty")
    instruction = oracle_reference[0]["instruction"]
    placeholder = oracle_reference[0]["output"]
    modes: Counter[str] = Counter()
    output = []
    seen: set[str] = set()
    for row in direct_rows:
        meta = dict(row.get("meta", {}))
        if meta.get("source_part") != "dev_seen":
            raise ValueError("predicted development source contains non-dev_seen row")
        wnd_id = str(meta.get("wnd_id", ""))
        if not wnd_id or wnd_id in seen:
            raise ValueError(f"predicted development has missing/duplicate wnd_id: {wnd_id}")
        seen.add(wnd_id)
        gold_payload = json.loads(row["output"])
        gold_types = [str(event["event_type"]) for event in gold_payload.get("events", [])]
        candidates = set(meta.get("candidate_types", []))
        supported = [value for value in gold_types if value in candidates]
        if len(supported) == len(gold_types):
            mode = "gold_present"
        elif supported:
            mode = "partial_supported"
        else:
            mode = "abstain"
        modes[mode] += 1
        meta["e137_candidate_regime"] = "retrieved"
        meta["e137_target_mode"] = mode
        output.append(
            {
                "instruction": instruction,
                "input": row["input"],
                "output": placeholder,
                "gold_output": row["output"],
                "meta": meta,
            }
        )
    return output, dict(sorted(modes.items()))


def build(protocol_path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inputs = {name: Path(path) for name, path in protocol["inputs"].items()}
    for name, expected_hash in protocol["input_sha256"].items():
        actual = sha256_file(inputs[name])
        if actual != expected_hash:
            raise ValueError(f"{name} SHA256 mismatch: {actual} != {expected_hash}")

    retrieved_gate = json.loads(inputs["retrieved_gate"].read_text(encoding="utf-8"))
    if retrieved_gate.get("id") != protocol["parent_data_gate"] or retrieved_gate.get("passed") is not True:
        raise ValueError("E136 parent data gate is not an exact pass")

    oracle = load_jsonl(inputs["oracle_train"])
    retrieved = load_jsonl(inputs["retrieved_train"])
    oracle_dev = load_jsonl(inputs["oracle_dev_seen"])
    predicted_direct_dev = load_jsonl(inputs["predicted_dev_seen"])
    oracle_stats = validate_train_rows(oracle, "oracle")
    retrieved_stats = validate_train_rows(retrieved, "retrieved")

    oracle_tagged = tagged_rows(oracle, "oracle", protocol["input_sha256"]["oracle_train"])
    retrieved_tagged = tagged_rows(
        retrieved, "retrieved", protocol["input_sha256"]["retrieved_train"]
    )
    family_order = {"oracle": 0, "retrieved": 1}
    mixed = sorted(
        oracle_tagged + retrieved_tagged,
        key=lambda row: (
            int(row["meta"]["e40_source_index"]),
            family_order[row["meta"]["e137_candidate_regime"]],
            str(row["meta"]["wnd_id"]),
        ),
    )
    predicted_dev, predicted_dev_modes = build_predicted_dev(predicted_direct_dev, oracle_dev)

    expected = protocol["expected"]
    checks = {
        "oracle_train_rows_exact": len(oracle) == expected["oracle_train_rows"],
        "retrieved_train_rows_exact": len(retrieved) == expected["retrieved_train_rows"],
        "mixed_train_rows_exact": len(mixed) == expected["mixed_train_rows"],
        "retrieved_mode_counts_exact": retrieved_stats["mode_counts"]
        == expected["retrieved_mode_counts"],
        "oracle_dev_seen_rows_exact": len(oracle_dev) == expected["oracle_dev_seen_rows"],
        "predicted_dev_seen_rows_exact": len(predicted_dev)
        == expected["predicted_dev_seen_rows"],
        "test_rows_read_zero": expected["test_rows_read"] == 0,
        "all_training_outputs_candidate_consistent": True,
        "parent_gate_passed": True,
    }
    if not all(checks.values()):
        raise ValueError(f"E137 data checks failed: {checks}")

    dataset_info = {
        "e137_mixed_oracle_retrieved_train": {
            "file_name": "e137_mixed_oracle_retrieved_train.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "e137_oracle_dev_seen": {
            "file_name": "e137_oracle_dev_seen.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "e137_predicted_dev_seen": {
            "file_name": "e137_predicted_dev_seen.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    files = {
        "e137_mixed_oracle_retrieved_train.jsonl": jsonl_bytes(mixed),
        "e137_oracle_dev_seen.jsonl": inputs["oracle_dev_seen"].read_bytes(),
        "e137_predicted_dev_seen.jsonl": jsonl_bytes(predicted_dev),
        "dataset_info.json": json_bytes(dataset_info),
    }
    audit = {
        "id": "e137_mixed_oracle_retrieved_data_gate_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "source_sha256": protocol["input_sha256"],
        "output_sha256": {name: sha256_bytes(value) for name, value in files.items()},
        "oracle": oracle_stats,
        "retrieved": retrieved_stats,
        "mixed_rows": len(mixed),
        "cross_regime_window_overlap": len(
            {row["meta"]["wnd_id"] for row in oracle}
            & {row["meta"]["wnd_id"] for row in retrieved}
        ),
        "predicted_dev_seen_target_modes": predicted_dev_modes,
        "test_rows_read": 0,
        "checks": checks,
        "passed": all(checks.values()),
    }
    files["data_audit.json"] = json_bytes(audit)
    return files, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--verify_existing", action="store_true")
    args = parser.parse_args()
    files, audit = build(args.protocol)
    if args.verify_existing:
        for name, expected in files.items():
            path = args.output_dir / name
            if not path.exists() or path.read_bytes() != expected:
                raise SystemExit(f"existing E137 artifact mismatch: {path}")
        print(json.dumps({**audit, "existing_bytes_verified": True}, ensure_ascii=False, indent=2))
        return 0
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse E137 data directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    for name, value in files.items():
        (args.output_dir / name).write_bytes(value)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
