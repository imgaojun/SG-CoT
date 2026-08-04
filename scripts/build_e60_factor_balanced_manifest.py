#!/usr/bin/env python3
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


FACTOR_TARGETS = {
    "label_granularity_arbitration": 120,
    "ontology_boundary_arbitration": 100,
    "process_stage_arbitration": 100,
    "argument_minimality_abstention": 140,
    "variable_cardinality_event_separation": 100,
    "evidence_grounding_alignment": 40,
}

FACTOR_DESCRIPTIONS = {
    "label_granularity_arbitration": (
        "Decide whether a broad ontology label or a finer subtype is appropriate, "
        "and reject the wrong granularity using schema and local textual cues."
    ),
    "ontology_boundary_arbitration": (
        "Choose between neighboring event families whose lexical cues can overlap, "
        "using schema definitions rather than broad world knowledge."
    ),
    "process_stage_arbitration": (
        "Choose the correct stage inside a procedural event family, rather than "
        "collapsing related stages into one generic outcome."
    ),
    "argument_minimality_abstention": (
        "Fill only locally licensed role arguments and abstain from plausible but "
        "unsupported participants."
    ),
    "variable_cardinality_event_separation": (
        "Decide how many event frames are supported, separate repeated true events, "
        "and suppress duplicates or extra plausible events."
    ),
    "evidence_grounding_alignment": (
        "Align trigger text, argument text, local evidence, reasoning, and final JSON "
        "so surface strings can be recovered without numeric offsets."
    ),
}

CONTACT_TYPES = {"Contact:Contact", "Contact:Meet", "Contact:Broadcast", "Contact:Correspondence"}
LIFE_CONFLICT_TYPES = {"Life:Injure", "Life:Die", "Conflict:Attack"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_events(row: dict) -> list[dict]:
    payload = row.get("gold_output") or row.get("output")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, dict):
        return []
    events = payload.get("events") or []
    return events if isinstance(events, list) else []


def event_type(event: dict) -> str:
    return event.get("event_type") or event.get("type") or ""


def arguments(event: dict) -> list[dict]:
    args = event.get("arguments") or []
    return args if isinstance(args, list) else []


def candidate_types(row: dict) -> set[str]:
    meta = row.get("meta") or {}
    vals = meta.get("candidate_types") or []
    vals = set(vals)
    if not vals:
        vals = set(re.findall(r"[A-Z][A-Za-z]+:[A-Za-z][A-Za-z-]+", row.get("input") or ""))
    return vals


def event_text_key(row: dict) -> str:
    meta = row.get("meta") or {}
    return f"{meta.get('doc_id','')}::{meta.get('wnd_id','')}"


def row_factors(row: dict) -> list[tuple[str, str]]:
    events = parse_events(row)
    types = [event_type(e) for e in events]
    type_set = set(types)
    cands = candidate_types(row)
    factors = []

    if "Contact:Contact" in cands and (cands & (CONTACT_TYPES - {"Contact:Contact"})):
        factors.append(("label_granularity_arbitration", "RichERE instance: Contact generic label versus fine Contact subtypes."))
    elif type_set & CONTACT_TYPES:
        factors.append(("label_granularity_arbitration", "RichERE instance: Contact event family granularity."))

    if cands & {"Conflict:Attack"} and cands & {"Life:Injure", "Life:Die"}:
        factors.append(("ontology_boundary_arbitration", "RichERE instance: Conflict/Life boundary."))
    elif type_set & LIFE_CONFLICT_TYPES:
        factors.append(("ontology_boundary_arbitration", "RichERE instance: Life/Conflict neighboring family cues."))
    elif any(t.startswith("Movement:") for t in cands) and any(t.startswith("Transaction:") for t in cands):
        factors.append(("ontology_boundary_arbitration", "RichERE instance: Movement/Transaction neighboring schema cues."))

    if sum(1 for t in cands if t.startswith("Justice:")) >= 2:
        factors.append(("process_stage_arbitration", "RichERE instance: Justice procedural stages."))
    elif any(t.startswith("Justice:") for t in type_set):
        factors.append(("process_stage_arbitration", "RichERE instance: Justice stage selection."))

    arg_counts = [len(arguments(e)) for e in events]
    if any(c <= 1 for c in arg_counts) or any(c >= 3 for c in arg_counts):
        factors.append(("argument_minimality_abstention", "RichERE instance: sparse or dense role frame requiring local role gating."))

    if len(events) >= 2 or len(set(types)) < len(types):
        factors.append(("variable_cardinality_event_separation", "RichERE instance: multi-event or repeated same-type event frame decision."))

    factors.append(("evidence_grounding_alignment", "RichERE instance: local evidence must contain trigger and argument surface strings."))
    return factors


def priority(row: dict, factor: str) -> tuple[int, int, str]:
    events = parse_events(row)
    types = [event_type(e) for e in events]
    cands = candidate_types(row)
    score = 0
    if factor == "label_granularity_arbitration":
        score += 4 if "Contact:Contact" in cands and (cands & (CONTACT_TYPES - {"Contact:Contact"})) else 0
        score += 2 if set(types) & CONTACT_TYPES else 0
    elif factor == "ontology_boundary_arbitration":
        score += 4 if cands & {"Conflict:Attack"} and cands & {"Life:Injure", "Life:Die"} else 0
        score += 2 if set(types) & LIFE_CONFLICT_TYPES else 0
    elif factor == "process_stage_arbitration":
        score += min(5, sum(1 for t in cands if t.startswith("Justice:")))
        score += 2 if any(t.startswith("Justice:") for t in types) else 0
    elif factor == "argument_minimality_abstention":
        arg_counts = [len(arguments(e)) for e in events]
        score += sum(2 for c in arg_counts if c == 0)
        score += sum(1 for c in arg_counts if c == 1)
        score += sum(2 for c in arg_counts if c >= 3)
    elif factor == "variable_cardinality_event_separation":
        score += len(events)
        score += 3 if len(set(types)) < len(types) else 0
    elif factor == "evidence_grounding_alignment":
        score += len(events)
        score += sum(len(arguments(e)) for e in events)
    return (-score, -len(events), event_text_key(row))


def build_manifest(rows: list[dict], targets: dict[str, int], seed: int, probe_per_factor: int | None) -> list[dict]:
    rng = random.Random(seed)
    by_factor = defaultdict(list)
    for source_index, row in enumerate(rows):
        if not parse_events(row):
            continue
        for factor, instantiation in row_factors(row):
            by_factor[factor].append((source_index, row, instantiation))

    manifest = []
    for factor, target in targets.items():
        need = probe_per_factor if probe_per_factor is not None else target
        pool = list(by_factor.get(factor) or [])
        rng.shuffle(pool)
        pool.sort(key=lambda item: priority(item[1], factor))
        if len(pool) < need:
            repeats = []
            while len(pool) + len(repeats) < need and pool:
                repeats.extend(pool)
            pool = pool + repeats
        for local_i, (source_index, row, instantiation) in enumerate(pool[:need]):
            rec = json.loads(json.dumps(row, ensure_ascii=False))
            meta = rec.setdefault("meta", {})
            meta["e40_source_index"] = source_index
            meta["e40_sample_id"] = f"e60_{factor}_{local_i:04d}"
            meta["e60_factor"] = factor
            meta["e60_factor_description"] = FACTOR_DESCRIPTIONS[factor]
            meta["e60_dataset_instantiation"] = instantiation
            meta["e60_strict_unseen_safe"] = True
            meta["e60_schema_synthetic"] = False
            meta["e60_manifest_role"] = "probe" if probe_per_factor is not None else "pilot"
            manifest.append(rec)
    manifest.sort(key=lambda r: r["meta"]["e40_sample_id"])
    return manifest


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=REPO / "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_train_pos.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--seed", type=int, default=6060)
    ap.add_argument("--probe-per-factor", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    targets = dict(FACTOR_TARGETS)
    if args.size != sum(targets.values()):
        total = sum(targets.values())
        targets = {k: max(1, round(v * args.size / total)) for k, v in targets.items()}
        delta = args.size - sum(targets.values())
        keys = list(targets)
        for i in range(abs(delta)):
            targets[keys[i % len(keys)]] += 1 if delta > 0 else -1
    rows = load_jsonl(args.train)
    manifest = build_manifest(rows, targets, args.seed, args.probe_per_factor)
    write_jsonl(args.output, manifest)
    counts = Counter(r["meta"]["e60_factor"] for r in manifest)
    print(json.dumps({"output": args.output.as_posix(), "rows": len(manifest), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
