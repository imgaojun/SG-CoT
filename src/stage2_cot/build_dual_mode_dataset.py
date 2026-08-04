import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_data.build_formal_stage2_dataset import load_jsonl, update_dataset_info, write_json


DIRECT_MODE = "<DIRECT>"
COT_MODE = "<COT>"


def extract_section(text: str, start_marker: str, end_marker: str):
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def canonical_gold_json(payload):
    if isinstance(payload, dict) and "events" in payload:
        payload = {"events": payload.get("events", [])}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def row_hash_from_dataset_row(row):
    payload = row.get("gold_output", row["output"])
    if isinstance(payload, str):
        payload = json.loads(payload)
    digest = hashlib.sha256()
    text_section = extract_section(row["input"], "Text:\n", "\n\nTokens:\n")
    candidate_section = extract_section(row["input"], "Candidate event types:\n", "\n\nSchema cards:\n")
    digest.update(text_section.encode("utf-8"))
    digest.update(b"\n")
    digest.update(candidate_section.encode("utf-8"))
    digest.update(b"\n")
    digest.update(canonical_gold_json(payload).encode("utf-8"))
    return digest.hexdigest()


def load_dataset_map(path: Path):
    rows = load_jsonl(path)
    mapping = {}
    for row in rows:
        row_hash = row_hash_from_dataset_row(row)
        if row_hash in mapping:
            raise ValueError(f"Duplicate row_hash in dataset: {path} -> {row_hash}")
        mapping[row_hash] = row
    return mapping


def load_label_map(path: Path):
    rows = load_jsonl(path)
    mapping = {}
    for row in rows:
        row_hash = row["row_hash"]
        if row_hash in mapping:
            raise ValueError(f"Duplicate row_hash in labels: {path} -> {row_hash}")
        mapping[row_hash] = row
    return mapping


def render_dual_mode_instruction():
    return (
        "Decide whether the current sample should use <DIRECT> mode or <COT> mode. "
        "Start the response with exactly one mode token: <DIRECT> or <COT>. "
        "If you choose <DIRECT>, immediately output JSON with top-level key `events`. "
        "If you choose <COT>, immediately output JSON with top-level keys `decisions` and `events`. "
        "Return the mode token followed by JSON only."
    )


def render_dual_mode_instruction_shortcotv2():
    return (
        "Decide whether the current sample should use <DIRECT> mode or <COT> mode. "
        "Start the response with exactly one mode token: <DIRECT> or <COT>. "
        "If you choose <DIRECT>, immediately output JSON with top-level key `events`. "
        "If you choose <COT>, immediately output compact JSON with top-level keys `decisions` and `events`. "
        "In compact <COT> mode, each item in `decisions` should contain only `event_type` and `contrast_type`. "
        "Return the mode token followed by JSON only."
    )


def render_forced_direct_instruction():
    return (
        "Use <DIRECT> mode. Start the response with exactly <DIRECT>. "
        "Then immediately output JSON with top-level key `events`. "
        "Return the mode token followed by JSON only."
    )


def render_forced_cot_instruction():
    return (
        "Use <COT> mode. Start the response with exactly <COT>. "
        "Then immediately output JSON with top-level keys `decisions` and `events`. "
        "Return the mode token followed by JSON only."
    )


def render_forced_cot_instruction_shortcotv2():
    return (
        "Use <COT> mode. Start the response with exactly <COT>. "
        "Then immediately output compact JSON with top-level keys `decisions` and `events`. "
        "Each item in `decisions` should contain only `event_type` and `contrast_type`. "
        "Return the mode token followed by JSON only."
    )


def wrap_target(mode_token: str, payload_text: str):
    return f"{mode_token}\n{payload_text}"


def compact_cot_payload(payload_text: str, cot_style: str):
    if cot_style == "full":
        return payload_text
    if cot_style != "shortcotv2":
        raise ValueError(f"Unsupported cot_style: {cot_style}")
    payload = json.loads(payload_text)
    compact_decisions = []
    raw_decisions = payload.get("decisions", [])
    if not isinstance(raw_decisions, list):
        raw_decisions = []
    for item in raw_decisions:
        if not isinstance(item, dict):
            continue
        compact = {}
        if item.get("event_type") is not None:
            compact["event_type"] = item["event_type"]
        if item.get("contrast_type") is not None:
            compact["contrast_type"] = item["contrast_type"]
        if compact:
            compact_decisions.append(compact)
    compact_payload = {
        "decisions": compact_decisions,
        "events": payload.get("events", []),
    }
    return json.dumps(compact_payload, ensure_ascii=False)


def select_mode_label(label_row, mode_label_key: str):
    if label_row is None:
        return None
    value = label_row.get(mode_label_key)
    if value not in {"DIRECT", "COT"}:
        raise ValueError(f"Unsupported mode label in key={mode_label_key}: {value}")
    return value


def build_record(direct_row, cot_row, *, output_mode: str, label_row, mode_label_key: str, cot_style: str):
    mode_label = select_mode_label(label_row, mode_label_key) if label_row else None
    direct_target = direct_row["output"]
    cot_target = compact_cot_payload(cot_row["output"], cot_style)
    gold_output = direct_row.get("gold_output", direct_row["output"])

    if output_mode == "labeled":
        if mode_label is None:
            raise ValueError("labeled output mode requires labels")
        instruction = render_dual_mode_instruction_shortcotv2() if cot_style == "shortcotv2" else render_dual_mode_instruction()
        if mode_label == "DIRECT":
            output = wrap_target(DIRECT_MODE, direct_target)
        else:
            output = wrap_target(COT_MODE, cot_target)
    elif output_mode == "free":
        instruction = render_dual_mode_instruction_shortcotv2() if cot_style == "shortcotv2" else render_dual_mode_instruction()
        if mode_label == "COT":
            output = wrap_target(COT_MODE, cot_target)
        else:
            output = wrap_target(DIRECT_MODE, direct_target)
    elif output_mode == "forced_direct":
        instruction = render_forced_direct_instruction()
        output = wrap_target(DIRECT_MODE, direct_target)
    elif output_mode == "forced_cot":
        instruction = render_forced_cot_instruction_shortcotv2() if cot_style == "shortcotv2" else render_forced_cot_instruction()
        output = wrap_target(COT_MODE, cot_target)
    else:
        raise ValueError(f"Unsupported output_mode: {output_mode}")

    item = copy.deepcopy(direct_row)
    meta = dict(item.get("meta", {}))
    meta["row_hash"] = row_hash_from_dataset_row(direct_row)
    meta["oracle_mode_label"] = mode_label
    meta["dual_mode_output_mode"] = output_mode
    meta["dual_mode_label_key"] = mode_label_key if label_row else None
    meta["dual_mode_cot_style"] = cot_style
    item["meta"] = meta
    item["instruction"] = instruction
    item["output"] = output
    item["gold_output"] = gold_output
    item["response_prefix"] = ""
    return item


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_jsonl", required=True)
    parser.add_argument("--cot_jsonl", required=True)
    parser.add_argument("--label_jsonl", default=None)
    parser.add_argument("--mode_label_key", default="strict_label")
    parser.add_argument("--output_mode", choices=["labeled", "free", "forced_direct", "forced_cot"], required=True)
    parser.add_argument("--cot_style", choices=["full", "shortcotv2"], default="full")
    parser.add_argument("--dataset_dir", default="data/stage2_cot_datasets")
    parser.add_argument("--dataset_name", required=True)
    args = parser.parse_args()

    direct_map = load_dataset_map(Path(args.direct_jsonl))
    cot_map = load_dataset_map(Path(args.cot_jsonl))
    common_hashes = set(direct_map) & set(cot_map)
    if common_hashes != set(direct_map) or common_hashes != set(cot_map):
        raise ValueError(
            "Direct and CoT datasets do not align perfectly: "
            f"direct_only={len(set(direct_map) - common_hashes)} "
            f"cot_only={len(set(cot_map) - common_hashes)}"
        )

    label_map = load_label_map(Path(args.label_jsonl)) if args.label_jsonl else {}
    if args.output_mode == "labeled" and set(label_map) != common_hashes:
        raise ValueError(
            "Labeled mode requires full label coverage: "
            f"missing={len(common_hashes - set(label_map))} extra={len(set(label_map) - common_hashes)}"
        )

    rows = []
    for row_hash in sorted(common_hashes):
        label_row = label_map.get(row_hash)
        rows.append(
            build_record(
                direct_map[row_hash],
                cot_map[row_hash],
                output_mode=args.output_mode,
                label_row=label_row,
                mode_label_key=args.mode_label_key,
                cot_style=args.cot_style,
            )
        )

    dataset_dir = Path(args.dataset_dir)
    file_name = f"{args.dataset_name}.jsonl"
    write_jsonl(dataset_dir / file_name, rows)
    update_dataset_info(dataset_dir, args.dataset_name, file_name)
    write_json(
        dataset_dir / f"{args.dataset_name}.meta.json",
        {
            "dataset_name": args.dataset_name,
            "file_name": file_name,
            "direct_jsonl": args.direct_jsonl,
            "cot_jsonl": args.cot_jsonl,
            "label_jsonl": args.label_jsonl,
            "mode_label_key": args.mode_label_key if args.label_jsonl else None,
            "output_mode": args.output_mode,
            "cot_style": args.cot_style,
            "num_examples": len(rows),
        },
    )
    print(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "file_name": file_name,
                "output_mode": args.output_mode,
                "num_examples": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
