#!/usr/bin/env python3
"""Hybrid: GoLLIE-style code+guidelines INPUT representation x SG-CoT reasoning OUTPUT supervision.

Joins the existing accepted E81 CoT rows (thinking + surface-evidence final, input-format-agnostic)
with the GoLLIE-style formal rows by window id, replacing instruction+input only. No new teacher
generation. Also builds eval files (dev_seen/test_seen/test_unseen) the same way, keeping gold_output.
"""
import argparse
import json
from pathlib import Path

INSTR_SUFFIX = (
    " First write your reasoning inside <thinking>...</thinking>, then output the final surface-only "
    "event JSON with evidence fields inside <final>...</final>."
)


def load_by_wnd(path):
    m = {}
    for ln in open(path):
        d = json.loads(ln)
        m[d["meta"]["wnd_id"]] = d
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e81_prefix", default="data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot")
    ap.add_argument("--gollie_prefix", default="data/stage2_formal_datasets/richere_balanced_split1_gollie_style_top10_shuffle")
    ap.add_argument("--dst_prefix", default="data/stage2_adaptive_datasets/richere_balanced_split1_gollie_style_top10_shuffle_adaptive_hybrid_sgcot_thinking_evidence_cot")
    ap.add_argument("--dataset_dir", default="data/stage2_adaptive_datasets")
    args = ap.parse_args()

    info_path = Path(args.dataset_dir) / "dataset_info.json"
    info = json.loads(info_path.read_text())

    for part in ["train", "dev_seen", "test_seen", "test_unseen"]:
        e81 = [json.loads(l) for l in open(f"{args.e81_prefix}_{part}_pos.jsonl")]
        gol = load_by_wnd(f"{args.gollie_prefix}_{part}_pos.jsonl")
        dst = Path(f"{args.dst_prefix}_{part}_pos.jsonl")
        n = miss = 0
        with open(dst, "w") as f:
            for r in e81:
                w = r["meta"]["wnd_id"]
                g = gol.get(w)
                if g is None:
                    miss += 1
                    continue
                # adapt gollie instruction: replace strict-offset-JSON demand with thinking+evidence protocol
                gi = g["instruction"].replace(
                    "Extract all event mentions supported by the text and output strict JSON with token offsets (same schema as the classes).",
                    "Extract all event mentions supported by the text.")
                out = dict(r)
                out["instruction"] = gi + INSTR_SUFFIX
                out["input"] = g["input"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                n += 1
        name = dst.stem
        info[name] = {"file_name": dst.name,
                      "columns": {"prompt": "instruction", "query": "input", "response": "output"}}
        print(f"{part}: {n} rows (missed {miss}) -> {dst.name}")

    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print("dataset_info.json updated")


if __name__ == "__main__":
    main()
