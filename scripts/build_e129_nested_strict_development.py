#!/usr/bin/env python3
"""Build and audit the train-only E129 strict nested-holdout datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.build_surface_evidence_dataset_20260712 import register_dataset  # noqa: E402
from src.stage2_preference.reasoning_preference import find_heldout_leaks  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_jsonl_as_json(jsonl_path: Path) -> None:
    shutil.copyfile(jsonl_path, jsonl_path.with_suffix(".json"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_types(row: dict[str, Any]) -> set[str]:
    return {event["event_type"] for event in row.get("event_mentions", [])}


def filtered_events(row: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    copied = deepcopy(row)
    copied["event_mentions"] = [
        event for event in copied.get("event_mentions", []) if event.get("event_type") in allowed
    ]
    return copied


def assert_unique_wnd_ids(rows: list[dict[str, Any]], label: str) -> None:
    ids = [row.get("wnd_id") or row.get("meta", {}).get("wnd_id") for row in rows]
    if any(not item for item in ids):
        raise ValueError(f"{label} contains rows without wnd_id")
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"{label} contains duplicate wnd_id values: {duplicates[:10]}")


def assert_no_leaks(rows: list[dict[str, Any]], heldout: list[str], label: str) -> None:
    leaks = []
    for index, row in enumerate(rows):
        for leak in find_heldout_leaks(row, heldout):
            leaks.append({"index": index, **leak})
            if len(leaks) >= 20:
                break
        if len(leaks) >= 20:
            break
    if leaks:
        raise ValueError(f"{label} contains held-out leakage: {leaks}")


def build_protocol(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    base_dir = (
        REPO_ROOT
        / "data/processed/type_holdout/richere-en"
        / config["base_protocol"]
        / config["base_split"]
    )
    target_dir = output_root / "richere-en" / config["output_protocol"] / config["base_split"]
    heldout = set(config["heldout_types"])
    prior_strict = set(config["prior_strict_heldout_types"])
    base_seen = set(load_json(base_dir / "seen_types.json"))
    if not heldout <= base_seen:
        raise ValueError(f"nested heldout is not contained in the base seen ontology: {heldout - base_seen}")
    if heldout & prior_strict:
        raise ValueError("nested heldout overlaps the E128 strict confirmation types")

    source_train = load_jsonl(base_dir / "train.jsonl")
    source_dev_seen = load_jsonl(base_dir / "dev_seen.jsonl")
    for label, rows in (("source_train", source_train), ("source_dev_seen", source_dev_seen)):
        bad = sorted({item for row in rows for item in event_types(row) if item in prior_strict})
        if bad:
            raise ValueError(f"{label} unexpectedly contains prior strict heldout targets: {bad}")

    nested_train = [row for row in source_train if not (event_types(row) & heldout)]
    pseudo_unseen = [filtered_events(row, heldout) for row in source_train if event_types(row) & heldout]
    nested_dev_seen = [row for row in source_dev_seen if not (event_types(row) & heldout)]
    assert_unique_wnd_ids(nested_train, "nested_train")
    assert_unique_wnd_ids(pseudo_unseen, "pseudo_unseen")
    if len(nested_train) < int(config["minimum_nested_train_rows"]):
        raise ValueError(f"nested train is too small: {len(nested_train)}")
    if len(pseudo_unseen) < int(config["minimum_pseudo_unseen_rows"]):
        raise ValueError(f"pseudo unseen set is too small: {len(pseudo_unseen)}")
    if any(not row.get("event_mentions") for row in pseudo_unseen):
        raise ValueError("pseudo unseen contains an empty target row")

    target_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("train", nested_train),
        ("dev_seen", nested_dev_seen),
        ("dev_unseen", pseudo_unseen),
    ):
        path = target_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        copy_jsonl_as_json(path)
    seen_types = sorted(base_seen - heldout)
    write_json(target_dir / "seen_types.json", seen_types)
    write_json(target_dir / "unseen_types.json", sorted(heldout))

    pseudo_counts = Counter(
        event["event_type"] for row in pseudo_unseen for event in row.get("event_mentions", [])
    )
    report = {
        "id": config["id"],
        "base_protocol": config["base_protocol"],
        "output_protocol": config["output_protocol"],
        "heldout_types": sorted(heldout),
        "prior_strict_heldout_types": sorted(prior_strict),
        "source_train_rows": len(source_train),
        "nested_train_rows": len(nested_train),
        "nested_dev_seen_rows": len(nested_dev_seen),
        "pseudo_unseen_rows": len(pseudo_unseen),
        "pseudo_unseen_mentions": sum(pseudo_counts.values()),
        "pseudo_unseen_type_counts": dict(sorted(pseudo_counts.items())),
        "train_dev_unseen_overlap": len(
            {row["wnd_id"] for row in nested_train} & {row["wnd_id"] for row in pseudo_unseen}
        ),
        "output_dir": str(target_dir.resolve()),
    }
    write_json(target_dir / "stats.json", report)
    return report


def filter_traces(config: dict[str, Any], output_path: Path, dataset_info: Path, dataset_name: str) -> dict[str, Any]:
    source_path = REPO_ROOT / config["source_trace_jsonl"]
    source_rows = load_jsonl(source_path)
    heldout = list(config["heldout_types"] + config["prior_strict_heldout_types"])
    retained = []
    excluded = []
    for row in source_rows:
        leaks = find_heldout_leaks(row, heldout)
        if leaks:
            excluded.append(
                {
                    "wnd_id": row.get("meta", {}).get("wnd_id"),
                    "leak_paths": sorted({leak.get("path", "unknown") for leak in leaks}),
                }
            )
        else:
            retained.append(row)
    assert_unique_wnd_ids(retained, "filtered traces")
    assert_no_leaks(retained, heldout, "filtered traces")
    if len(retained) < int(config["minimum_filtered_trace_rows"]):
        raise ValueError(f"filtered trace set is too small: {len(retained)}")
    write_jsonl(output_path, retained)
    register_dataset(dataset_info, dataset_name, output_path)
    excluded_path = output_path.with_name(output_path.stem + ".excluded.jsonl")
    write_jsonl(excluded_path, excluded)
    report = {
        "source": str(source_path.resolve()),
        "source_rows": len(source_rows),
        "retained_rows": len(retained),
        "excluded_rows": len(excluded),
        "excluded_training_types": heldout,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
    }
    write_json(output_path.with_suffix(".summary.json"), report)
    return report


def build_mix(
    config: dict[str, Any],
    direct_path: Path,
    trace_path: Path,
    output_path: Path,
    dataset_info: Path,
    dataset_name: str,
) -> dict[str, Any]:
    direct_rows = load_jsonl(direct_path)
    trace_rows = load_jsonl(trace_path)
    heldout = list(config["heldout_types"] + config["prior_strict_heldout_types"])
    assert_no_leaks(direct_rows, heldout, "direct replay")
    assert_no_leaks(trace_rows, heldout, "reasoning traces")
    rows = []
    for mode, source_rows in (("direct", direct_rows), ("sgcot", trace_rows)):
        for row in source_rows:
            copied = deepcopy(row)
            copied.setdefault("meta", {})["e129_training_mode"] = mode
            rows.append(copied)
    rng = random.Random(int(config["mix_seed"]))
    rng.shuffle(rows)
    direct_share = len(direct_rows) / len(rows)
    if direct_share > float(config["maximum_direct_share"]):
        raise ValueError(f"direct replay share is too high: {direct_share:.6f}")
    assert_no_leaks(rows, heldout, "mixed training set")
    write_jsonl(output_path, rows)
    register_dataset(dataset_info, dataset_name, output_path)
    report = {
        "direct_source": str(direct_path.resolve()),
        "trace_source": str(trace_path.resolve()),
        "direct_rows": len(direct_rows),
        "trace_rows": len(trace_rows),
        "total_rows": len(rows),
        "direct_share": direct_share,
        "mix_seed": config["mix_seed"],
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
    }
    write_json(output_path.with_suffix(".summary.json"), report)
    return report


def audit(config: dict[str, Any], data_dir: Path, output_path: Path) -> dict[str, Any]:
    names = {
        "direct_train": data_dir / "e129_jtrial_direct_train.jsonl",
        "direct_dev_seen": data_dir / "e129_jtrial_direct_dev_seen.jsonl",
        "direct_dev_unseen": data_dir / "e129_jtrial_direct_dev_unseen.jsonl",
        "sgcot_dev_seen": data_dir / "e129_jtrial_sgcot_dev_seen.jsonl",
        "sgcot_dev_unseen": data_dir / "e129_jtrial_sgcot_dev_unseen.jsonl",
        "filtered_traces": data_dir / "e129_jtrial_sgcot_filtered_train.jsonl",
        "mixed_train": data_dir / "e129_jtrial_dualmode_mixed_train.jsonl",
    }
    pseudo_heldout = list(config["heldout_types"])
    training_excluded = pseudo_heldout + list(config["prior_strict_heldout_types"])
    report: dict[str, Any] = {
        "id": config["id"],
        "heldout_types": pseudo_heldout,
        "training_excluded_types": training_excluded,
        "datasets": {},
    }
    for label, path in names.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = load_jsonl(path)
        if label in {"direct_train", "filtered_traces", "mixed_train"}:
            assert_no_leaks(rows, training_excluded, label)
        report["datasets"][label] = {"rows": len(rows), "sha256": sha256(path)}

    direct_unseen = load_jsonl(names["direct_dev_unseen"])
    sgcot_unseen = load_jsonl(names["sgcot_dev_unseen"])
    direct_ids = [row["meta"]["wnd_id"] for row in direct_unseen]
    sgcot_ids = [row["meta"]["wnd_id"] for row in sgcot_unseen]
    if direct_ids != sgcot_ids:
        raise ValueError("direct and SG-CoT pseudo-unseen evaluation rows are not aligned")
    for row in direct_unseen + sgcot_unseen:
        gold_types = set(row.get("meta", {}).get("gold_event_types", []))
        if gold_types != set(pseudo_heldout):
            raise ValueError(f"unexpected pseudo-unseen gold types: {sorted(gold_types)}")

    mix_summary = load_json(names["mixed_train"].with_suffix(".summary.json"))
    report["direct_share"] = mix_summary["direct_share"]
    report["pseudo_unseen_pairing_exact"] = True
    report["all_training_leak_checks_passed"] = True
    report["gate_ready"] = True
    write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["protocol", "filter-traces", "mix", "audit"])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generated/stage2_development/e129a_nested_jtrial_protocol.json"),
    )
    parser.add_argument(
        "--processed_root", type=Path, default=Path("data/processed/type_holdout")
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/stage2_development_e129"))
    args = parser.parse_args()
    config = load_json(args.config)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = args.data_dir / "dataset_info.json"

    if args.action == "protocol":
        payload = build_protocol(config, args.processed_root)
    elif args.action == "filter-traces":
        payload = filter_traces(
            config,
            args.data_dir / "e129_jtrial_sgcot_filtered_train.jsonl",
            dataset_info,
            "e129_jtrial_sgcot_filtered_train",
        )
    elif args.action == "mix":
        payload = build_mix(
            config,
            args.data_dir / "e129_jtrial_direct_train.jsonl",
            args.data_dir / "e129_jtrial_sgcot_filtered_train.jsonl",
            args.data_dir / "e129_jtrial_dualmode_mixed_train.jsonl",
            dataset_info,
            "e129_jtrial_dualmode_mixed_train",
        )
    else:
        payload = audit(config, args.data_dir, args.data_dir / "e129a_dataset_audit.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
