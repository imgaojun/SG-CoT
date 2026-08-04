#!/usr/bin/env python3
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data/stage2_formal_datasets"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
OUT_DIR = REPO / "outputs/stage2_strategy_cot_e76/e76_contrastive_exactness_glm51_full1500_20260614"

QUOTAS = {
    "contact_subtype_arbitration": 450,
    "argument_minimality": 350,
    "trigger_anchor_exactness": 300,
    "justice_life_conflict_boundary": 250,
    "extra_frame_abstention": 150,
}

CONTACT_CUES = {
    "say", "said", "says", "tell", "told", "email", "e-mail", "message", "letter", "call",
    "called", "write", "wrote", "post", "posted", "publish", "published", "report", "reported",
    "announce", "announced", "meet", "met", "meeting", "talk", "talked", "reply", "replied",
}
TRIGGER_ANCHOR_CUES = {
    "confirmed", "reported", "said", "went", "took", "made", "had", "found", "showed",
    "according", "claimed", "announced", "revealed", "became", "got",
}
BOUNDARY_TYPES = {
    "Justice:Arrest-Jail", "Justice:Sentence", "Justice:Convict", "Justice:Conviction",
    "Justice:Indict", "Justice:Indictment", "Justice:Release-Parole", "Life:Injure",
    "Life:Die", "Conflict:Attack", "Conflict:Demonstrate",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def event_payload(row: dict) -> dict:
    raw = row.get("gold_output") or row.get("output") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def events(row: dict) -> list[dict]:
    payload = event_payload(row)
    evs = payload.get("events", []) if isinstance(payload, dict) else []
    return evs if isinstance(evs, list) else []


def text(row: dict) -> str:
    inp = row.get("input", "")
    m = re.search(r"Text:\n(.*?)(?:\n\nTokens:|\Z)", inp, re.S)
    return (m.group(1) if m else inp).strip()


def tokens(row: dict) -> list[str]:
    inp = row.get("input", "")
    m = re.search(r"Tokens:\n(.*?)(?:\n\nCandidate event types:|\Z)", inp, re.S)
    if not m:
        return re.findall(r"\w+|[^\w\s]", text(row))
    return m.group(1).split()


def candidate_types(row: dict) -> list[str]:
    meta_types = row.get("meta", {}).get("candidate_types")
    if isinstance(meta_types, list):
        return [str(x) for x in meta_types]
    inp = row.get("input", "")
    m = re.search(r"Candidate event types:\n(.*?)(?:\n\nSchema cards:|\Z)", inp, re.S)
    if not m:
        return []
    return [x.strip() for x in m.group(1).replace("\n", " ").split(",") if x.strip()]


def gold_types(row: dict) -> set[str]:
    out = set()
    for ev in events(row):
        if isinstance(ev, dict) and ev.get("event_type"):
            out.add(str(ev["event_type"]))
    return out


def arg_count(row: dict) -> int:
    total = 0
    for ev in events(row):
        args = ev.get("arguments", []) if isinstance(ev, dict) else []
        total += len(args) if isinstance(args, list) else 0
    return total


def has_contact_signal(row: dict) -> bool:
    cands = set(candidate_types(row))
    gtypes = gold_types(row)
    if any(t.startswith("Contact:") for t in cands | gtypes):
        return True
    low = {t.lower() for t in tokens(row)}
    return bool(low & CONTACT_CUES)


def has_trigger_anchor_signal(row: dict) -> bool:
    low_tokens = [t.lower() for t in tokens(row)]
    if any(t in TRIGGER_ANCHOR_CUES for t in low_tokens):
        return True
    for ev in events(row):
        trig = ev.get("trigger", {}) if isinstance(ev, dict) else {}
        trigger_text = str(trig.get("text") or "").lower()
        if trigger_text and trigger_text in TRIGGER_ANCHOR_CUES:
            return True
    return False


def bucket_candidates(rows: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    buckets = defaultdict(list)
    for idx, row in enumerate(rows):
        if not events(row):
            continue
        cands = set(candidate_types(row))
        gtypes = gold_types(row)
        if has_contact_signal(row):
            buckets["contact_subtype_arbitration"].append((idx, row))
        if arg_count(row) >= 2 or len(cands) >= 8:
            buckets["argument_minimality"].append((idx, row))
        if has_trigger_anchor_signal(row):
            buckets["trigger_anchor_exactness"].append((idx, row))
        if (cands | gtypes) & BOUNDARY_TYPES or any(t.startswith(("Justice:", "Life:", "Conflict:")) for t in cands | gtypes):
            buckets["justice_life_conflict_boundary"].append((idx, row))
        if len(cands) >= 10 or len(events(row)) <= 1:
            buckets["extra_frame_abstention"].append((idx, row))
    return buckets


def clone_with_meta(row: dict, source_index: int, bucket: str, rank: int) -> dict:
    rec = json.loads(json.dumps(row, ensure_ascii=False))
    meta = rec.setdefault("meta", {})
    meta["e76_source_index"] = source_index
    meta["e76_bucket"] = bucket
    meta["e76_bucket_rank"] = rank
    return rec


def build(rows: list[dict], seed: int, quotas: dict[str, int]) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    candidates = bucket_candidates(rows)
    selected = []
    selected_indices = set()
    counts = Counter()
    availability = {bucket: len(vals) for bucket, vals in candidates.items()}

    for bucket, quota in quotas.items():
        vals = list(candidates.get(bucket, []))
        rng.shuffle(vals)
        rank = 0
        for idx, row in vals:
            if counts[bucket] >= quota:
                break
            if idx in selected_indices:
                continue
            selected_indices.add(idx)
            selected.append(clone_with_meta(row, idx, bucket, rank))
            counts[bucket] += 1
            rank += 1

    target = sum(quotas.values())
    fallback = [(idx, row) for idx, row in enumerate(rows) if idx not in selected_indices and events(row)]
    fallback.sort(key=lambda item: (-len(events(item[1])), -arg_count(item[1]), item[0]))
    for idx, row in fallback:
        if len(selected) >= target:
            break
        selected_indices.add(idx)
        selected.append(clone_with_meta(row, idx, "fallback_priority_fill", counts["fallback_priority_fill"]))
        counts["fallback_priority_fill"] += 1

    summary = {
        "seed": seed,
        "target_total": target,
        "selected_total": len(selected),
        "quotas": quotas,
        "selected_counts": dict(counts),
        "available_counts": availability,
        "candidate_type_counts": dict(Counter(t for row in selected for t in candidate_types(row)).most_common(30)),
        "gold_type_counts": dict(Counter(t for row in selected for t in gold_types(row)).most_common(30)),
    }
    return selected, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    ap.add_argument("--output_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=7601)
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    selected, summary = build(rows, args.seed, QUOTAS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "e76_manifest_rows.jsonl", selected)
    (args.output_dir / "e76_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
