#!/usr/bin/env python3
"""L1: DEGREE-style definition+template-conditioned generation baseline (adapted to our protocol).

DEGREE (Hsu et al. 2022) represents each event type by a natural-language DEFINITION and a TEMPLATE
with typed role placeholders, and trains the model to fill the template. We adapt that to our
candidate-conditioned setting: the input lists each candidate type's definition and template; the
output fills a template line per present event (pipe-delimited, verbatim spans) that a deterministic
parser maps back to (trigger, type, arguments) with token offsets recovered by matching.
"""
import argparse
import json
import re
from pathlib import Path

INSTRUCTION = (
    "You are doing event extraction. Each candidate event type is given with its definition and a "
    "fill-in template listing its roles. For every event expressed in the text, output one line: "
    "`<event_type> || trigger: <text> || <Role>: <text> || ...`, copying trigger and argument spans "
    "verbatim from the text and including only locally supported roles. If no candidate event is "
    "expressed, output `NONE`."
)


def templates_block(candidate_types, schema_by_type):
    lines = []
    for t in candidate_types:
        e = schema_by_type[t]
        roles = ", ".join(f"<{r}>" for r in (e.get("core_roles") or []))
        lines.append(f"- {t}: {e.get('definition','').strip()} Template: trigger <trigger>; {roles}")
    return "\n".join(lines)


def convert_input(inp, schema_by_type):
    head, _, _ = inp.partition("Schema cards:")
    m = re.search(r"Candidate event types:\n(.+?)\n", inp, re.S)
    cands = [c.strip() for c in m.group(1).split(",")] if m else []
    return head + "Event templates:\n" + templates_block(cands, schema_by_type)


def build_target(gold_json):
    evs = gold_json.get("events") or []
    if not evs:
        return "NONE"
    lines = []
    for e in evs:
        t = e.get("event_type")
        tr = (e.get("trigger") or {}).get("text", "")
        parts = [f"{t} || trigger: {tr}"]
        for a in (e.get("arguments") or []):
            parts.append(f"{a.get('role')}: {a.get('text')}")
        lines.append(" || ".join(parts))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_path", default="data/schema/richere-en.event_schema.json")
    ap.add_argument("--src_prefix", default="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle")
    ap.add_argument("--dst_prefix", default="data/stage2_formal_datasets/richere_balanced_split1_degree_style")
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
                gold = json.loads(d["output"])
                nd = {"instruction": INSTRUCTION,
                      "input": convert_input(d["input"], sbt),
                      "output": build_target(gold),
                      "gold_output": d["output"],
                      "meta": d.get("meta", {})}
                f.write(json.dumps(nd, ensure_ascii=False) + "\n")
                n += 1
        name = dst.stem
        info[name] = {"file_name": dst.name,
                      "columns": {"prompt": "instruction", "query": "input", "response": "output"}}
        print(f"{part}: {n} -> {dst.name}")
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print("dataset_info.json updated")


if __name__ == "__main__":
    main()
