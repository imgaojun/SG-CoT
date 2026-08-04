#!/usr/bin/env python3
"""E2: GoLLIE-style external-method baseline (adapted to our candidate-conditioned protocol).

GoLLIE's key idea: represent the schema as CODE (Python class definitions) whose docstrings carry
annotation GUIDELINES, and instruction-tune the model to follow them. We adapt that schema
representation to our protocol: same candidate conditioning (top-10), same text, same strict-JSON
offset output (so training/eval machinery is unchanged); only the schema representation and the
instruction change from "schema cards" to GoLLIE-style code+guidelines.
"""
import argparse
import json
import re
from pathlib import Path

INSTRUCTION = (
    "You are doing event extraction. The annotation guidelines are given below as Python class "
    "definitions: each class is a candidate event type, its docstring is the guideline, and its "
    "fields are the argument roles. Use only these candidate classes. Extract all event mentions "
    "supported by the text and output strict JSON with token offsets (same schema as the classes). "
    "If no valid event is expressed by the candidate set, output {\"events\": []}."
)


def class_name(event_type):
    return re.sub(r"[^0-9a-zA-Z]", "", event_type.replace(":", " ").title())


def to_code(schema_by_type, candidate_types):
    lines = ["```python"]
    for t in candidate_types:
        e = schema_by_type[t]
        cues = ", ".join(e.get("trigger_cues") or [])
        lines.append("@dataclass")
        lines.append(f"class {class_name(t)}(Event):")
        lines.append(f'    """event_type = "{t}". {e.get("definition", "").strip()}')
        lines.append(f'    The trigger is the minimal word or phrase that most directly evokes the event')
        lines.append(f'    (typical cues: {cues})."""')
        lines.append(f'    mention: str  # the exact trigger span copied from the text')
        for r in e.get("core_roles") or []:
            lines.append(f"    {r}: List[str]")
        lines.append("")
    lines.append("```")
    return "\n".join(lines)


def convert_input(inp, schema_by_type):
    # keep Text/Tokens/Candidate blocks; replace "Schema cards:" block with code guidelines
    head, _, _ = inp.partition("Schema cards:")
    m = re.search(r"Candidate event types:\n(.+?)\n", inp, re.S)
    cands = [c.strip() for c in m.group(1).split(",")] if m else []
    return head + "Annotation guidelines (Python classes):\n" + to_code(schema_by_type, cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_path", default="data/schema/richere-en.event_schema.json")
    ap.add_argument("--src_prefix", default="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle")
    ap.add_argument("--dst_prefix", default="data/stage2_formal_datasets/richere_balanced_split1_gollie_style_top10_shuffle")
    ap.add_argument("--dataset_dir", default="data/stage2_formal_datasets")
    args = ap.parse_args()

    schema = json.loads(Path(args.schema_path).read_text())
    sbt = {e["event_type"]: e for e in schema}

    info_path = Path(args.dataset_dir) / "dataset_info.json"
    info = json.loads(info_path.read_text())

    for part in ["train", "dev_seen", "test_seen", "test_unseen"]:
        src = Path(f"{args.src_prefix}_{part}_pos.jsonl")
        dst = Path(f"{args.dst_prefix}_{part}_pos.jsonl")
        n = 0
        with open(dst, "w") as f:
            for ln in open(src):
                d = json.loads(ln)
                d["instruction"] = INSTRUCTION
                d["input"] = convert_input(d["input"], sbt)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1
        name = dst.stem
        info[name] = {"file_name": dst.name,
                      "columns": {"prompt": "instruction", "query": "input", "response": "output"}}
        print(f"{part}: {n} rows -> {dst.name}")

    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print("dataset_info.json updated")


if __name__ == "__main__":
    main()
