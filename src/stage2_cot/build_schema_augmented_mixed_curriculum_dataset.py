import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_data.build_formal_stage2_dataset import (
    event_family,
    jaccard,
    load_json,
    load_jsonl,
    normalize_cue_tokens,
    update_dataset_info,
    write_json,
)


def load_schema_map(schema_path: Path):
    items = load_json(schema_path)
    return {item["event_type"]: item for item in items}


def sort_key_for_confusion(candidate_type, other_type, schema_by_type):
    left = schema_by_type[candidate_type]
    right = schema_by_type[other_type]
    same_family = 1.0 if event_family(candidate_type) == event_family(other_type) else 0.0
    role_overlap = jaccard(left.get("core_roles", []), right.get("core_roles", []))
    cue_overlap = jaccard(
        normalize_cue_tokens(left.get("trigger_cues", [])),
        normalize_cue_tokens(right.get("trigger_cues", [])),
    )
    return (same_family, role_overlap, cue_overlap, other_type)


def top_confusions(event_type, candidate_types, schema_by_type, max_items=2):
    others = [item for item in candidate_types if item != event_type]
    ranked = sorted(
        others,
        key=lambda other_type: sort_key_for_confusion(event_type, other_type, schema_by_type),
        reverse=True,
    )
    return ranked[:max_items]


def normalize_sentence(text: str):
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    return text


def paraphrase_definition(definition: str):
    base = normalize_sentence(definition)
    if not base:
        return ""
    lowered = base[0].lower() + base[1:] if len(base) > 1 else base.lower()
    return f"This event covers cases where {lowered}."


def join_or_none(items):
    return ", ".join(items) if items else "none"


def render_card(event_type, candidate_types, schema_by_type, mode: str):
    schema = schema_by_type[event_type]
    definition = schema.get("definition", "").strip()
    cue_text = join_or_none(schema.get("trigger_cues", []))
    role_text = join_or_none(schema.get("core_roles", []))

    if mode == "static":
        return schema["document"]

    if mode == "paraphrase":
        return "\n".join(
            [
                f"Event type: {event_type}",
                f"Description: {paraphrase_definition(definition)}",
                f"Typical trigger signs: {cue_text}",
                f"Key roles to extract: {role_text}",
            ]
        )

    if mode == "confusion_aware":
        confusions = top_confusions(event_type, candidate_types, schema_by_type)
        confusion_text = join_or_none(confusions)
        return "\n".join(
            [
                f"Event type: {event_type}",
                f"Description: {paraphrase_definition(definition)}",
                f"Typical trigger signs: {cue_text}",
                f"Key roles to extract: {role_text}",
                f"Distinguish from: {confusion_text}",
                f"Confusion note: Prefer another candidate when its trigger meaning or role pattern fits the text better.",
            ]
        )

    if mode == "confusion_lite":
        confusions = top_confusions(event_type, candidate_types, schema_by_type, max_items=1)
        confusion_tag = confusions[0] if confusions else "none"
        return "\n".join(
            [
                f"Event type: {event_type}",
                f"Description: {paraphrase_definition(definition)}",
                f"Typical trigger signs: {cue_text}",
                f"Key roles to extract: {role_text}",
                f"Confusable: {confusion_tag}",
            ]
        )

    if mode == "role_lite":
        return "\n".join(
            [
                f"Event type: {event_type}",
                f"Trigger cues: {cue_text}",
                f"Argument roles: {role_text}",
                "Ground only explicit role spans.",
            ]
        )

    raise ValueError(f"Unsupported mode: {mode}")


def render_schema_cards(candidate_types, schema_by_type, mode: str):
    cards = []
    for idx, event_type in enumerate(candidate_types, start=1):
        cards.append(f"[{idx}] {render_card(event_type, candidate_types, schema_by_type, mode)}")
    return "\n\n".join(cards)


def replace_schema_cards(input_text: str, new_cards: str):
    marker = "Schema cards:\n"
    if marker not in input_text:
        raise ValueError("Missing `Schema cards:` marker in input")

    start = input_text.index(marker) + len(marker)
    suffix_markers = ["\n\nOutput requirements:\n", "\n\nReturn JSON only."]
    suffix_start = -1
    suffix_marker = None
    for candidate in suffix_markers:
        pos = input_text.find(candidate, start)
        if pos != -1 and (suffix_start == -1 or pos < suffix_start):
            suffix_start = pos
            suffix_marker = candidate

    if suffix_start == -1:
        raise ValueError("Could not find schema-card suffix marker in input")

    prefix = input_text[:start]
    suffix = input_text[suffix_start:]
    return prefix + new_cards + suffix


def transform_row(row, schema_by_type, mode: str, variant_tag: str):
    item = copy.deepcopy(row)
    candidate_types = item["meta"]["candidate_types"]
    new_cards = render_schema_cards(candidate_types, schema_by_type, mode)
    item["input"] = replace_schema_cards(item["input"], new_cards)
    meta = dict(item.get("meta", {}))
    meta["schema_variant_mode"] = mode
    meta["schema_variant_tag"] = variant_tag
    item["meta"] = meta
    return item


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def register_dataset(dataset_dir: Path, dataset_name: str, rows, meta: dict):
    file_name = f"{dataset_name}.jsonl"
    write_jsonl(dataset_dir / file_name, rows)
    update_dataset_info(dataset_dir, dataset_name, file_name)
    write_json(dataset_dir / f"{dataset_name}.meta.json", {"dataset_name": dataset_name, "file_name": file_name, **meta})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_path", required=True)
    parser.add_argument("--source_train_jsonl", required=True)
    parser.add_argument("--source_dev_jsonl", required=True)
    parser.add_argument("--source_test_jsonl", required=True)
    parser.add_argument("--source_test_seen_jsonl", required=True)
    parser.add_argument("--source_test_unseen_jsonl", required=True)
    parser.add_argument("--dataset_dir", default="data/stage2_cot_datasets")
    parser.add_argument("--train_dataset_name", required=True)
    parser.add_argument("--dev_dataset_name", required=True)
    parser.add_argument("--test_dataset_name", required=True)
    parser.add_argument("--test_seen_dataset_name", required=True)
    parser.add_argument("--test_unseen_dataset_name", required=True)
    parser.add_argument(
        "--schema_variant_mode",
        choices=["paraphrase", "confusion_aware", "confusion_lite", "role_lite"],
        required=True,
    )
    parser.add_argument("--variant_tag", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    schema_by_type = load_schema_map(Path(args.schema_path))

    source_specs = [
        (args.source_train_jsonl, args.train_dataset_name, "train"),
        (args.source_dev_jsonl, args.dev_dataset_name, "dev_seen"),
        (args.source_test_jsonl, args.test_dataset_name, "test"),
        (args.source_test_seen_jsonl, args.test_seen_dataset_name, "test_seen"),
        (args.source_test_unseen_jsonl, args.test_unseen_dataset_name, "test_unseen"),
    ]

    for source_jsonl, dataset_name, role in source_specs:
        rows = load_jsonl(Path(source_jsonl))
        transformed = [transform_row(row, schema_by_type, args.schema_variant_mode, args.variant_tag) for row in rows]
        register_dataset(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            rows=transformed,
            meta={
                "schema_path": args.schema_path,
                "source_jsonl": source_jsonl,
                "dataset_role": role,
                "schema_variant_mode": args.schema_variant_mode,
                "schema_variant_tag": args.variant_tag,
                "num_examples": len(transformed),
            },
        )

    print(
        json.dumps(
            {
                "variant_tag": args.variant_tag,
                "schema_variant_mode": args.schema_variant_mode,
                "train_dataset_name": args.train_dataset_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
