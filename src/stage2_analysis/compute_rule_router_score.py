import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_data.build_formal_stage2_dataset import (
    event_family,
    jaccard,
    load_jsonl,
    load_schema_map,
    normalize_cue_tokens,
    write_json,
)


PAIR_FAMILY_WEIGHT = 0.5
PAIR_ROLE_WEIGHT = 0.3
PAIR_CUE_WEIGHT = 0.2

FINAL_SCHEMA_WEIGHT = 0.6
FINAL_TEXT_WEIGHT = 0.4

V2_MARGIN_SCALE = 0.03
V2_LEVEL_SCALE = 0.03
V2_FAMILY_COMP_SCALE = 4.0


def canonical_gold_json(row):
    payload = row.get("gold_output", row["output"])
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict) and "events" in payload:
        payload = {"events": payload.get("events", [])}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def row_hash_for_eval_row(row):
    digest = hashlib.sha256()
    text_section = extract_section(row["input"], "Text:\n", "\n\nTokens:\n")
    candidate_section = extract_section(row["input"], "Candidate event types:\n", "\n\nSchema cards:\n")
    digest.update(text_section.encode("utf-8"))
    digest.update(b"\n")
    digest.update(candidate_section.encode("utf-8"))
    digest.update(b"\n")
    digest.update(canonical_gold_json(row).encode("utf-8"))
    return digest.hexdigest()


def extract_section(text: str, start_marker: str, end_marker: str):
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def normalize_text_tokens(text: str):
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


def canonical_pair(left_type: str, right_type: str):
    return tuple(sorted([left_type, right_type]))


def pair_confusion(left_type: str, right_type: str, schema_by_type):
    left = schema_by_type[left_type]
    right = schema_by_type[right_type]
    same_family = 1.0 if event_family(left_type) == event_family(right_type) else 0.0
    role_overlap = jaccard(left.get("core_roles", []), right.get("core_roles", []))
    cue_overlap = jaccard(
        normalize_cue_tokens(left.get("trigger_cues", [])),
        normalize_cue_tokens(right.get("trigger_cues", [])),
    )
    score = (
        PAIR_FAMILY_WEIGHT * same_family
        + PAIR_ROLE_WEIGHT * role_overlap
        + PAIR_CUE_WEIGHT * cue_overlap
    )
    return {
        "left_type": left_type,
        "right_type": right_type,
        "same_family": bool(same_family),
        "role_overlap": role_overlap,
        "cue_overlap": cue_overlap,
        "score": score,
    }


def schema_ambiguity(candidate_types, schema_by_type):
    if len(candidate_types) < 2:
        return 0.0, []
    pairs = []
    for idx, left_type in enumerate(candidate_types):
        for right_type in candidate_types[idx + 1 :]:
            pairs.append(pair_confusion(left_type, right_type, schema_by_type))
    ranked = sorted(pairs, key=lambda item: (item["score"], item["left_type"], item["right_type"]), reverse=True)
    best = ranked[0]["score"]
    top2 = ranked[:2]
    top2_avg = sum(item["score"] for item in top2) / len(top2)
    score = 0.7 * best + 0.3 * top2_avg
    return score, ranked


def cue_matches(text_tokens, candidate_types, schema_by_type):
    rows = []
    for event_type in candidate_types:
        cue_tokens = normalize_cue_tokens(schema_by_type[event_type].get("trigger_cues", []))
        rows.append(
            {
                "event_type": event_type,
                "cue_tokens": sorted(cue_tokens),
                "score": jaccard(text_tokens, cue_tokens),
            }
        )
    rows.sort(key=lambda item: (item["score"], item["event_type"]), reverse=True)
    return rows


def cue_match_by_type(text_tokens, candidate_types, schema_by_type):
    mapping = {}
    for row in cue_matches(text_tokens, candidate_types, schema_by_type):
        mapping[row["event_type"]] = row
    return mapping


def text_ambiguity(text_tokens, candidate_types, schema_by_type):
    matches = cue_matches(text_tokens, candidate_types, schema_by_type)
    if not matches:
        return 0.0, matches
    top1 = matches[0]["score"]
    top2 = matches[1]["score"] if len(matches) > 1 else 0.0
    margin = max(0.0, top1 - top2)
    score = 1.0 - margin
    return score, matches


def family_size_map(candidate_types):
    counts = {}
    for event_type in candidate_types:
        family = event_family(event_type)
        counts[family] = counts.get(family, 0) + 1
    return counts


def pair_local_features(candidate_types, pair_rows, cue_match_map):
    if not pair_rows:
        return {
            "canonical_top_pair": None,
            "top_pair_family_count": 0,
            "pair_family_high_confusion_count": 0,
            "pair_doc_match_left": 0.0,
            "pair_doc_match_right": 0.0,
            "pair_doc_match_margin": 0.0,
            "pair_doc_match_level": 0.0,
        }

    top_pair = pair_rows[0]
    left_type = top_pair["left_type"]
    right_type = top_pair["right_type"]
    canonical_top_pair = canonical_pair(left_type, right_type)
    family_counts = family_size_map(candidate_types)
    left_family = event_family(left_type)
    right_family = event_family(right_type)
    if left_family == right_family:
        top_pair_family_count = family_counts.get(left_family, 0)
        pair_family_high_confusion_count = sum(
            1
            for row in pair_rows
            if row["score"] >= 0.80
            and event_family(row["left_type"]) == left_family
            and event_family(row["right_type"]) == left_family
        )
    else:
        top_pair_family_count = max(family_counts.get(left_family, 0), family_counts.get(right_family, 0))
        pair_family_high_confusion_count = sum(
            1
            for row in pair_rows
            if row["score"] >= 0.80
            and (
                event_family(row["left_type"]) in {left_family, right_family}
                or event_family(row["right_type"]) in {left_family, right_family}
            )
        )

    left_match = cue_match_map.get(left_type, {}).get("score", 0.0)
    right_match = cue_match_map.get(right_type, {}).get("score", 0.0)
    return {
        "canonical_top_pair": list(canonical_top_pair),
        "top_pair_family_count": top_pair_family_count,
        "pair_family_high_confusion_count": pair_family_high_confusion_count,
        "pair_doc_match_left": left_match,
        "pair_doc_match_right": right_match,
        "pair_doc_match_margin": abs(left_match - right_match),
        "pair_doc_match_level": max(left_match, right_match),
    }


def load_pair_prior(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clip01(value: float):
    return max(0.0, min(1.0, value))


def normalize_small_margin(margin: float):
    return clip01(1.0 - margin / V2_MARGIN_SCALE)


def normalize_level(level: float):
    return clip01(level / V2_LEVEL_SCALE)


def normalize_family_comp(count: int):
    return clip01(count / V2_FAMILY_COMP_SCALE)


def heuristic_v2_score(record, pair_prior_payload):
    canonical_top_pair = tuple(record["canonical_top_pair"]) if record["canonical_top_pair"] else None
    prior_rows = pair_prior_payload.get("pair_stats", {})
    pair_prior = 0.0
    pair_support = 0
    pair_cot_better_rate = 0.0
    if canonical_top_pair:
        key = " || ".join(canonical_top_pair)
        prior = prior_rows.get(key)
        if prior is not None:
            pair_prior = prior.get("pair_prior_score", 0.0)
            pair_support = prior.get("n", 0)
            pair_cot_better_rate = prior.get("cot_better_rate", 0.0)

    margin_norm = normalize_small_margin(record["pair_doc_match_margin"])
    level_norm = normalize_level(record["pair_doc_match_level"])
    family_comp_norm = normalize_family_comp(record["pair_family_high_confusion_count"])
    score = 0.5 * pair_prior + 0.2 * margin_norm + 0.15 * level_norm + 0.15 * family_comp_norm
    return {
        "pair_prior_score": pair_prior,
        "pair_prior_support": pair_support,
        "pair_prior_cot_better_rate": pair_cot_better_rate,
        "pair_doc_small_margin_norm": margin_norm,
        "pair_doc_match_level_norm": level_norm,
        "pair_family_comp_norm": family_comp_norm,
        "heuristic_v2_score": score,
    }


def summarize(records):
    if not records:
        return {
            "num_examples": 0,
            "avg_schema_score": 0.0,
            "avg_text_score": 0.0,
            "avg_ambiguity_score": 0.0,
            "min_ambiguity_score": 0.0,
            "max_ambiguity_score": 0.0,
        }
    ambiguity_scores = [row["ambiguity_score"] for row in records]
    schema_scores = [row["schema_score"] for row in records]
    text_scores = [row["text_score"] for row in records]
    payload = {
        "num_examples": len(records),
        "avg_schema_score": sum(schema_scores) / len(schema_scores),
        "avg_text_score": sum(text_scores) / len(text_scores),
        "avg_ambiguity_score": sum(ambiguity_scores) / len(ambiguity_scores),
        "min_ambiguity_score": min(ambiguity_scores),
        "max_ambiguity_score": max(ambiguity_scores),
        "top5_row_hashes": [
            row["row_hash"]
            for row in sorted(records, key=lambda item: (item["ambiguity_score"], item["row_hash"]), reverse=True)[:5]
        ],
    }
    if records and "heuristic_v2_score" in records[0]:
        v2_scores = [row["heuristic_v2_score"] for row in records]
        payload["avg_heuristic_v2_score"] = sum(v2_scores) / len(v2_scores)
        payload["min_heuristic_v2_score"] = min(v2_scores)
        payload["max_heuristic_v2_score"] = max(v2_scores)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_path", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary_json", default=None)
    parser.add_argument("--router_prior_json", default=None)
    args = parser.parse_args()

    schema_by_type = load_schema_map(Path(args.schema_path))
    pair_prior_payload = load_pair_prior(Path(args.router_prior_json)) if args.router_prior_json else None
    rows = load_jsonl(Path(args.eval_jsonl))
    records = []

    for row in rows:
        candidate_types = row.get("meta", {}).get("candidate_types")
        if not candidate_types:
            candidate_text = extract_section(row["input"], "Candidate event types:\n", "\n\nSchema cards:\n")
            candidate_types = [item.strip() for item in candidate_text.split(",") if item.strip()]
        token_text = extract_section(row["input"], "Tokens:\n", "\n\nCandidate event types:\n")
        text_tokens = normalize_text_tokens(token_text)

        schema_score, pair_rows = schema_ambiguity(candidate_types, schema_by_type)
        text_score, cue_rows = text_ambiguity(text_tokens, candidate_types, schema_by_type)
        ambiguity_score = FINAL_SCHEMA_WEIGHT * schema_score + FINAL_TEXT_WEIGHT * text_score

        top_pair = pair_rows[0] if pair_rows else None
        top1_match = cue_rows[0] if cue_rows else None
        top2_match = cue_rows[1] if len(cue_rows) > 1 else None
        cue_match_map = cue_match_by_type(text_tokens, candidate_types, schema_by_type)
        local_features = pair_local_features(candidate_types, pair_rows, cue_match_map)

        record = {
            "row_hash": row_hash_for_eval_row(row),
            "wnd_id": row.get("meta", {}).get("wnd_id"),
            "doc_id": row.get("meta", {}).get("doc_id"),
            "noise_mode": row.get("meta", {}).get("noise_mode"),
            "candidate_types": candidate_types,
            "gold_event_types": row.get("meta", {}).get("gold_event_types"),
            "schema_score": schema_score,
            "text_score": text_score,
            "ambiguity_score": ambiguity_score,
            "top_confusion_pair": top_pair,
            "top1_cue_match": top1_match,
            "top2_cue_match": top2_match,
            "pair_rows": pair_rows[:5],
            **local_features,
        }
        if pair_prior_payload is not None:
            record.update(heuristic_v2_score(record, pair_prior_payload))
        records.append(record)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = Path(args.output_summary_json) if args.output_summary_json else output_path.with_suffix(".summary.json")
    write_json(
        summary_path,
        {
            "schema_path": args.schema_path,
            "eval_jsonl": args.eval_jsonl,
            "router_prior_json": args.router_prior_json,
            **summarize(records),
        },
    )

    print(json.dumps({"output_jsonl": args.output_jsonl, "output_summary_json": summary_path.as_posix(), **summarize(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
