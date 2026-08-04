#!/usr/bin/env python3
"""Prepare E77b: row-matched same-data direct control for E81 (reviewer fix).

E77b uses the exact E81 rows (same order), but strips the `<thinking>...</thinking>`
block from the output and keeps only the `<final>{...}</final>` surface-evidence
JSON part, exactly as it appears in the E81 trace. This mirrors the E77/E76
control (scripts/prepare_e77_e76_direct_control_20260615.py) but for E81 and
keeping the E81 final surface format (evidence JSON, `<final>` tags) instead of
gold offset JSON, so the eval uses the evidence-format script.

Instruction: e77-style (no thinking request) = the E81 instruction with the
`First output <thinking>...` sentence removed and `Then output` -> `Output`.

Configs: copy of the E77 config fields, warm-start = restored Direct root
(the s8322/s8333 E77 variants' warm-start; the original checkpoint-2064 no
longer exists), 3 seeds (default/none, 8322, 8333) per
scripts/gen_strengthen_seed_configs_20260709.py rules.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data/stage2_adaptive_datasets"
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"

DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_PREFIX = "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
SOURCE_BRANCH = "e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot"
TARGET_BRANCH = "e77b_e81_rowmatched_control"
WARM_START = (
    "/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user/"
    "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full"
)
SEEDS = [None, 8322, 8333]

FINAL_RE = re.compile(r"^<thinking>.*</thinking>\s*(<final>(.*)</final>)$", re.S)

THINKING_SENTENCE_RE = re.compile(r"First output `<thinking>[^`]*`[^.]*\. Then output", re.S)


def strip_instruction(instruction: str) -> str:
    stripped, n = THINKING_SENTENCE_RE.subn("Output", instruction)
    if n != 1 or "<thinking>" in stripped:
        raise ValueError("unexpected E81 instruction shape")
    return stripped


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def convert_rows(rows: list[dict], split: str) -> list[dict]:
    converted = []
    for idx, row in enumerate(rows):
        m = FINAL_RE.match(row["output"])
        if not m:
            raise ValueError(f"unexpected output shape at split={split} idx={idx}")
        final_part = m.group(1)
        json.loads(m.group(2))  # must parse
        meta = dict(row.get("meta") or {})
        meta.update(
            {
                "adaptive_source": TARGET_BRANCH,
                "adaptive_target_style": "final_evidence_json_no_cot",
                "adaptive_dataset_role": split,
                "control_source_branch": SOURCE_BRANCH,
                "control_changed_variable": "remove_thinking_keep_final",
            }
        )
        converted.append(
            {
                "instruction": strip_instruction(row["instruction"]),
                "input": row["input"],
                "output": final_part,
                "meta": meta,
                "gold_output": row["gold_output"],
            }
        )
    return converted


def update_dataset_info(names: list[str]) -> None:
    info_path = DATA_DIR / "dataset_info.json"
    data = json.loads(info_path.read_text(encoding="utf-8"))
    for name in names:
        data[name] = {
            "file_name": f"{name}.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    info_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_configs(train_name: str, dev_name: str) -> list[Path]:
    src_cfg = CONFIG_DIR / f"{RUN_PREFIX}_e77_e76_direct_control_full_stepmatch.yaml"
    lines = src_cfg.read_text().splitlines()
    made = []
    for seed in SEEDS:
        seed_tag = "" if seed is None else f"_s{seed}"
        out = []
        for ln in lines:
            if ln.startswith("model_name_or_path:"):
                out.append(f"model_name_or_path: {WARM_START}")
                if seed is not None:
                    out.append(f"seed: {seed}")
                continue
            if ln.startswith("dataset:"):
                out.append(f"dataset: {train_name}")
                continue
            if ln.startswith("eval_dataset:"):
                out.append(f"eval_dataset: {dev_name}")
                continue
            if ln.startswith("output_dir:"):
                out.append(
                    "output_dir: /workspace/project/outputs/stage2_adaptive_runs_user/"
                    f"{RUN_PREFIX}_{TARGET_BRANCH}{seed_tag}_full"
                )
                continue
            out.append(ln)
        dst = CONFIG_DIR / f"{RUN_PREFIX}_{TARGET_BRANCH}{seed_tag}_full_stepmatch.yaml"
        dst.write_text("\n".join(out) + "\n")
        made.append(dst)
    return made


def main() -> None:
    counts = {}
    names = []
    for split in ["train", "dev_seen", "test_seen", "test_unseen"]:
        src = DATA_DIR / f"{DATA_PREFIX}_{SOURCE_BRANCH}_{split}_pos.jsonl"
        target_name = f"{DATA_PREFIX}_{TARGET_BRANCH}_{split}_pos"
        rows = convert_rows(load_jsonl(src), split)
        write_jsonl(DATA_DIR / f"{target_name}.jsonl", rows)
        counts[split] = len(rows)
        names.append(target_name)
    update_dataset_info(names)
    cfgs = write_configs(names[0], names[1])
    print(json.dumps({"counts": counts, "configs": [str(c) for c in cfgs]}, indent=2))


if __name__ == "__main__":
    main()
