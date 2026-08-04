import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
SCRIPT_DIR = REPO / "scripts"
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

import prepare_4b_reason_format_ablation_e15_20260522 as e15  # noqa: E402


DATA_DIR = REPO / "data/stage2_adaptive_datasets"
FORMAL_DATA_DIR = REPO / "data/stage2_formal_datasets"
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPTIVE_PREFIX = f"{DATA_PREFIX}_adaptive"
SOURCE_BRANCH = "confrare10_heur10_typeonlylite"
BRANCH = "confrare10_typeonlylite_reasonfmt_e18a_latentreason_outcomepos"
RUN_PREFIX = "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
WARM_START = (
    "/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user/"
    "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full/checkpoint-2064"
)
OUTCOME_ROOT = REPO / "outputs/stage2_adaptive_outcome_mining/l15bal30_ckpt942"
TZ = timezone(timedelta(hours=8))

REASON_REPEAT = 7
LEARNING_RATE = 2.0e-6
NUM_EPOCHS = 4.0


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def key(row):
    return (row.get("meta") or {}).get("wnd_id")


def pred_key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("sample_key") or meta.get("doc_id")


def load_prediction_map(path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[pred_key(row)] = row
    return rows


def outcome_positive_keys():
    direct = load_prediction_map(OUTCOME_ROOT / "forced_direct/train/predictions.jsonl")
    reason = load_prediction_map(OUTCOME_ROOT / "forced_reason/train/predictions.jsonl")
    selected = {}
    for k in sorted(set(direct) & set(reason)):
        d = direct[k]
        r = reason[k]
        gains = {
            "argument_f1": r.get("argument_f1", 0.0) - d.get("argument_f1", 0.0),
            "event_f1": r.get("event_f1", 0.0) - d.get("event_f1", 0.0),
            "trigger_f1": r.get("trigger_f1", 0.0) - d.get("trigger_f1", 0.0),
        }
        gains["sum"] = gains["argument_f1"] + gains["event_f1"] + gains["trigger_f1"]
        if gains["argument_f1"] >= 0 and gains["event_f1"] >= 0 and gains["trigger_f1"] >= 0 and gains["sum"] > 0:
            selected[k] = gains
    return selected


def latent_reason_instruction():
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "Use the reasoning route internally: check which candidate event types are supported by the text, "
        "verify trigger spans against the token list, and ground each argument role in the schema. "
        "Do not output the reasoning. The final extraction must be strict JSON with top-level key `events` and token offsets. "
        "First output `<ROUTE>reason</ROUTE>`, then output `<FINAL>{...}</FINAL>`. "
        "Do not add text outside the requested tags. If no valid event is expressed by the candidate set, the final JSON is {\"events\": []}."
    )


def direct_instruction():
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "The final extraction must be strict JSON with top-level key `events` and token offsets. "
        "Wrap the final extraction as `<FINAL>{...}</FINAL>`. Do not add text outside the requested tags. "
        "Use the direct route. First output `<ROUTE>direct</ROUTE>`, then output `<FINAL>{...}</FINAL>`. "
        "If no valid event is expressed by the candidate set, the final JSON is {\"events\": []}."
    )


def free_route_instruction():
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "The final extraction must be strict JSON with top-level key `events` and token offsets. "
        "Wrap the final extraction as `<FINAL>{...}</FINAL>`. Do not add text outside the requested tags. "
        "First decide whether this case should be solved directly or with internal schema-grounded reasoning. "
        "If it is simple, output `<ROUTE>direct</ROUTE>` followed by `<FINAL>{...}</FINAL>`. "
        "If reasoning is useful, output `<ROUTE>reason</ROUTE>` followed by `<FINAL>{...}</FINAL>`; do not output the reasoning."
    )


def clone(row, source, duplicate_idx=0):
    out = json.loads(json.dumps(row, ensure_ascii=False))
    meta = out.setdefault("meta", {})
    meta["e18_source"] = source
    meta["e18_duplicate_index"] = duplicate_idx
    meta["e18_branch"] = BRANCH
    meta["adaptive_source"] = "strong_reason_supervision_e18"
    return out


def final_json_text(row):
    if "gold_output" in row:
        return row["gold_output"]
    output = row.get("output", "")
    start = output.find("<FINAL>")
    end = output.find("</FINAL>")
    if start != -1 and end != -1:
        return output[start + len("<FINAL>") : end].strip()
    return output


def direct_row(row, source, duplicate_idx=0):
    out = clone(row, source, duplicate_idx)
    out["instruction"] = direct_instruction()
    out["output"] = f"<ROUTE>direct</ROUTE>\n<FINAL>{row['output']}</FINAL>"
    out["gold_output"] = row["output"]
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_dataset_role": "train",
            "adaptive_route_mode": "forced_direct_anchor",
            "adaptive_route_label": "direct",
            "adaptive_target_style": "direct_retention",
        }
    )
    return out


def reason_row(source_row, gold_row, gains, duplicate_idx=0, role="train", route_mode="free_route"):
    out = clone(source_row, "outcome_positive_reason", duplicate_idx)
    out["instruction"] = latent_reason_instruction() if route_mode == "forced_reason" else free_route_instruction()
    final_text = final_json_text(gold_row)
    out["output"] = f"<ROUTE>reason</ROUTE>\n<FINAL>{final_text}</FINAL>"
    out["gold_output"] = final_text
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_dataset_role": role,
            "adaptive_route_mode": route_mode,
            "adaptive_route_label": "reason",
            "adaptive_target_style": "latent_no_reason_block",
            "e18_outcome_gain_argument_f1": gains.get("argument_f1", 0.0),
            "e18_outcome_gain_event_f1": gains.get("event_f1", 0.0),
            "e18_outcome_gain_trigger_f1": gains.get("trigger_f1", 0.0),
            "e18_outcome_gain_sum": gains.get("sum", 0.0),
        }
    )
    return out


def build_train_rows():
    rng = random.Random(20260523)
    adaptive = e15.load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_train_pos.jsonl")
    formal_direct = e15.load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    formal_by_key = {key(row): row for row in formal_direct}
    source_by_key = {key(row): row for row in adaptive}
    positives = outcome_positive_keys()

    direct_part = [direct_row(formal_by_key[key(row)], "all_train_direct_anchor", 0) for row in adaptive]
    reason_part = []
    for k, gains in positives.items():
        if k not in source_by_key or k not in formal_by_key:
            continue
        for dup in range(REASON_REPEAT):
            reason_part.append(reason_row(source_by_key[k], formal_by_key[k], gains, dup, "train", "forced_reason"))

    rows = direct_part + reason_part
    rng.shuffle(rows)
    audit = {
        "source_branch": SOURCE_BRANCH,
        "outcome_root": OUTCOME_ROOT.as_posix(),
        "selection_rule": "reason_gain argument/event/trigger all >= 0 and sum > 0 on train outcome mining",
        "adaptive_train_count": len(adaptive),
        "outcome_positive_count": len(positives),
        "reason_repeat": REASON_REPEAT,
        "reason_rows_after_repeat": len(reason_part),
        "direct_anchor_rows": len(direct_part),
        "total_count": len(rows),
        "route_label_counts": {"reason": len(reason_part), "direct": len(direct_part)},
        "training_recipe": {"learning_rate": LEARNING_RATE, "num_train_epochs": NUM_EPOCHS},
    }
    return rows, audit


def write_dataset(name, rows):
    file_name = f"{name}.jsonl"
    e15.write_jsonl(DATA_DIR / file_name, rows)
    e15.update_dataset_info(name, file_name)
    return file_name


def write_config(train_name, dev_name):
    out_config = CONFIG_DIR / f"{RUN_PREFIX}_{BRANCH}_full_stepmatch.yaml"
    config = {
        "model_name_or_path": WARM_START,
        "template": "qwen",
        "dataset_dir": "/workspace/project/data/stage2_adaptive_datasets",
        "dataset": train_name,
        "eval_dataset": dev_name,
        "output_dir": f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{BRANCH}_full",
        "stage": "sft",
        "do_train": True,
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "logging_steps": 5,
        "report_to": "none",
        "finetuning_type": "full",
        "cutoff_len": 1024,
        "max_samples": 10000,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "packing": False,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": 0.05,
        "bf16": True,
        "val_size": 0.0,
        "eval_steps": 10,
        "do_eval": True,
        "save_only_model": True,
        "num_train_epochs": NUM_EPOCHS,
        "load_best_model_at_end": False,
        "deepspeed": "/workspace/project/configs/deepspeed/zero2_optimizer_offload_cpu.json",
    }
    write_yaml(out_config, config)
    return out_config.as_posix()


def write_note(train_name, dev_name, audit):
    timestamp = now_iso()
    exp_id = "2026-05-23_stage2_4b_strong_reason_supervision_e18a_latentreason_outcomepos_richere_split1_oracle_mixed_noise_qwen3_4b"
    out_dir = REPO / f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{BRANCH}_full"
    config_path = CONFIG_DIR / f"{RUN_PREFIX}_{BRANCH}_full_stepmatch.yaml"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    body = f"""---
id: {exp_id}
title: Stage2 4B Strong Reason Supervision E18A Latent Reason Outcome Positives
kind: experiment
status: planned
created_at: {timestamp}
updated_at: {timestamp}
owners:
  - codex
tags:
  - stage2
  - qwen3-4b
  - reason-expert
  - outcome-supervision
objective: Train a stronger latent reason expert using outcome-positive train examples while keeping E15A's no-visible-reason format.
artifacts:
  configs:
    - {config_path}
  outputs:
    - {out_dir}
related:
  plans:
    - {REPO / 'PLANS.md'}
context:
  dataset: RichERE split1 oracle_mixed_noise_top10_shuffle
  base_model: Qwen3-4B-Instruct
  warm_start: {WARM_START}
  branch: {BRANCH}
  outcome_root: {OUTCOME_ROOT}
---

# Stage2 4B Strong Reason Supervision E18A

## Goal

Use train-set outcome mining to select stronger reason-positive examples, while keeping the low-interference E15A output format: `<ROUTE>reason</ROUTE><FINAL>...</FINAL>`.

## Setup

- train dataset: `{train_name}`
- dev dataset: `{dev_name}`
- audit: `{json.dumps(audit, ensure_ascii=False, sort_keys=True)}`

## Commands

```bash
cd {REPO}
python3 scripts/prepare_4b_strong_reason_supervision_e18_20260523.py
bash scripts/launch_4b_strong_reason_supervision_e18_20260523.sh train 0
bash scripts/launch_4b_strong_reason_supervision_e18_20260523.sh devpick 0
bash scripts/launch_4b_strong_reason_supervision_e18_20260523.sh formal 0 1 2 3
python3 scripts/summarize_4b_strong_reason_supervision_e18_20260523.py
```

## Run Log

### {datetime.now(TZ).strftime('%Y-%m-%d %H:%M %z')[:-2]}:{datetime.now(TZ).strftime('%z')[-2:]}

- prepared dataset/config/note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- run checkpoint selection and formal forced-direct/forced-reason evaluation.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    train_rows, audit = build_train_rows()
    train_name = f"{ADAPTIVE_PREFIX}_{BRANCH}_train_pos"
    dev_name = f"{ADAPTIVE_PREFIX}_{BRANCH}_dev_seen_pos"
    write_dataset(train_name, train_rows)

    dev_rows = e15.load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_confrare10_typeonlylite_reasonfmt_e15a_noreasonblock_dev_seen_pos.jsonl")
    for row in dev_rows:
        row["instruction"] = free_route_instruction()
        row.setdefault("meta", {})["adaptive_source"] = "strong_reason_supervision_e18"
        row.setdefault("meta", {})["adaptive_target_style"] = "latent_no_reason_block"
    write_dataset(dev_name, dev_rows)

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        for mode in ["forced_direct", "forced_reason"]:
            source_name = f"{ADAPTIVE_PREFIX}_confrare10_typeonlylite_reasonfmt_e15a_noreasonblock_{mode}_{split}_pos"
            target_name = f"{ADAPTIVE_PREFIX}_{BRANCH}_{mode}_{split}_pos"
            rows = e15.load_jsonl(DATA_DIR / f"{source_name}.jsonl")
            for row in rows:
                row.setdefault("meta", {})["adaptive_source"] = "strong_reason_supervision_e18"
                row.setdefault("meta", {})["adaptive_target_style"] = "latent_no_reason_block"
                if mode == "forced_reason":
                    row["instruction"] = latent_reason_instruction()
            write_dataset(target_name, rows)
            eval_names.append(target_name)

    config = write_config(train_name, dev_name)
    note = write_note(train_name, dev_name, audit)
    e15.write_json(DATA_DIR / f"{train_name}.meta.json", {"audit": audit, "created_at": now_iso()})
    print(json.dumps({"branch": BRANCH, "train_dataset": train_name, "dev_dataset": dev_name, "eval_datasets": eval_names, "config": config, "note": note.as_posix(), "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
