#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.generate_strategy_natural_cot_e37_20260604 import register_dataset  # noqa: E402
from scripts.generate_strategy_variants_cot_e47_20260606 import QWEN4_RUN_PREFIX, QWEN4_WARM_START  # noqa: E402


DATA_DIR = REPO / "data/stage2_adaptive_datasets"
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"


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


def compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def extract_final(output: str) -> dict:
    match = re.search(r"<final>\s*(\{.*\})\s*</final>", output or "", flags=re.S | re.I)
    if not match:
        raise ValueError("missing <final> block")
    return json.loads(match.group(1))


def direct_row(row: dict, run_name: str, dataset_role: str) -> dict:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    final_obj = extract_final(row["output"])
    factor = (row.get("meta") or {}).get("e60_factor", "unknown")
    out["instruction"] = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "Output exactly `<final>{...}</final>` with a surface-only JSON event list: each trigger and argument must include "
        "`text` and a short contiguous local `evidence` quote from the input text. "
        "Do not output `<thinking>`, numeric offsets, token indices, or text outside the lowercase `<final>` tag."
    )
    out["output"] = f"<final>{compact_json(final_obj)}</final>"
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "factor_balanced_direct_e60",
            "adaptive_target_style": "factor_balanced_surface_evidence_direct",
            "adaptive_dataset_role": dataset_role,
            "e60_direct_run_name": run_name,
            "e60_paired_factor": factor,
            "e60_paired_from_cot": True,
        }
    )
    return out


def write_dataset(name: str, rows: list[dict]) -> Path:
    path = DATA_DIR / f"{name}.jsonl"
    write_jsonl(path, rows)
    register_dataset(name, path.name)
    return path


def write_train_config(branch: str, train_name: str, dev_name: str, epochs: float) -> Path:
    path = CONFIG_DIR / f"{QWEN4_RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    config = {
        "model_name_or_path": QWEN4_WARM_START,
        "template": "qwen",
        "dataset_dir": "/workspace/project/data/stage2_adaptive_datasets",
        "dataset": train_name,
        "eval_dataset": dev_name,
        "output_dir": f"/workspace/project/outputs/stage2_adaptive_runs_user/{QWEN4_RUN_PREFIX}_{branch}_full",
        "stage": "sft",
        "do_train": True,
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "logging_steps": 1,
        "report_to": "none",
        "finetuning_type": "full",
        "cutoff_len": 1536,
        "max_samples": 20000,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "packing": False,
        "learning_rate": 2.0e-6,
        "warmup_ratio": 0.05,
        "bf16": True,
        "val_size": 0.0,
        "eval_steps": 10,
        "do_eval": True,
        "save_only_model": True,
        "num_train_epochs": epochs,
        "load_best_model_at_end": False,
        "deepspeed": "/workspace/project/configs/deepspeed/zero2_optimizer_offload_cpu.json",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-train", type=Path, required=True)
    ap.add_argument("--cot-dev", type=Path, required=True)
    ap.add_argument("--cot-test-seen", type=Path, required=True)
    ap.add_argument("--cot-test-unseen", type=Path, required=True)
    ap.add_argument("--run-name", default="e60b_glm51_factor_balanced_600_w16_direct")
    ap.add_argument("--epochs", type=float, default=3.0)
    return ap.parse_args()


def main():
    args = parse_args()
    branch = f"{args.run_name}_surface_evidence_direct"
    train_name = f"richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_{branch}_train_pos"
    dev_name = f"richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_{branch}_dev_seen_pos"
    test_seen_name = f"richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_{branch}_test_seen_pos"
    test_unseen_name = f"richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_{branch}_test_unseen_pos"
    train_rows = [direct_row(r, args.run_name, "train") for r in load_jsonl(args.cot_train)]
    dev_rows = [direct_row(r, args.run_name, "dev_seen") for r in load_jsonl(args.cot_dev)]
    test_seen_rows = [direct_row(r, args.run_name, "test_seen") for r in load_jsonl(args.cot_test_seen)]
    test_unseen_rows = [direct_row(r, args.run_name, "test_unseen") for r in load_jsonl(args.cot_test_unseen)]
    paths = {
        "train": write_dataset(train_name, train_rows).as_posix(),
        "dev_seen": write_dataset(dev_name, dev_rows).as_posix(),
        "test_seen": write_dataset(test_seen_name, test_seen_rows).as_posix(),
        "test_unseen": write_dataset(test_unseen_name, test_unseen_rows).as_posix(),
    }
    config = write_train_config(branch, train_name, dev_name, args.epochs)
    summary = {
        "run_name": args.run_name,
        "branch": branch,
        "rows": {
            "train": len(train_rows),
            "dev_seen": len(dev_rows),
            "test_seen": len(test_seen_rows),
            "test_unseen": len(test_unseen_rows),
        },
        "datasets": {
            "train": train_name,
            "dev_seen": dev_name,
            "test_seen": test_seen_name,
            "test_unseen": test_unseen_name,
        },
        "paths": paths,
        "train_config": config.as_posix(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
