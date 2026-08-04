import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
DATA_DIR = REPO / "data/stage2_adaptive_datasets"
FORMAL_DATA_DIR = REPO / "data/stage2_formal_datasets"
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
ADAPTIVE_PREFIX = f"{DATA_PREFIX}_adaptive"
SOURCE_BRANCH = "confrare10_heur10_typeonlylite"
RUN_PREFIX = "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
WARM_START = (
    "/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user/"
    "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full/checkpoint-2064"
)
TZ = timezone(timedelta(hours=8))

VARIANTS = {
    "e15a_noreasonblock": {
        "title": "E15A No Reason Block",
        "branch": "confrare10_typeonlylite_reasonfmt_e15a_noreasonblock",
        "style": "no_reason_block",
    },
    "e15b_minimaltype": {
        "title": "E15B Minimal Type Reason",
        "branch": "confrare10_typeonlylite_reasonfmt_e15b_minimaltype",
        "style": "minimal_type_reason",
    },
    "e15c_finalfirst": {
        "title": "E15C Final First Reason",
        "branch": "confrare10_typeonlylite_reasonfmt_e15c_finalfirst",
        "style": "final_first_reason",
    },
}


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def update_dataset_info(dataset_name, file_name):
    info_path = DATA_DIR / "dataset_info.json"
    info = load_json(info_path)
    info[dataset_name] = {
        "file_name": file_name,
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
    write_json(info_path, info)


def key(row):
    return (row.get("meta") or {}).get("wnd_id")


def route_label(row):
    return (row.get("meta") or {}).get("adaptive_route_label", "direct")


def final_json_text(row):
    if "gold_output" in row:
        return row["gold_output"]
    output = row.get("output", "")
    start = output.find("<FINAL>")
    end = output.find("</FINAL>")
    if start != -1 and end != -1:
        return output[start + len("<FINAL>") : end].strip()
    return output


def reason_types(row):
    events = []
    try:
        payload = json.loads(final_json_text(row))
        events = payload.get("events") or []
    except Exception:
        events = []
    seen = []
    for event in events:
        event_type = event.get("event_type")
        if event_type and event_type not in seen:
            seen.append(event_type)
    return seen


def reason_output(row, style):
    final_text = final_json_text(row)
    if style == "no_reason_block":
        return f"<ROUTE>reason</ROUTE>\n<FINAL>{final_text}</FINAL>"
    types = reason_types(row)
    reason = json.dumps({"types": types}, ensure_ascii=False, separators=(",", ":"))
    if style == "minimal_type_reason":
        return f"<ROUTE>reason</ROUTE>\n<REASON>{reason}</REASON>\n<FINAL>{final_text}</FINAL>"
    if style == "final_first_reason":
        return f"<ROUTE>reason</ROUTE>\n<FINAL>{final_text}</FINAL>\n<REASON>{reason}</REASON>"
    raise ValueError(style)


def forced_reason_instruction(style):
    base = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "The final extraction must be strict JSON with top-level key `events` and token offsets. "
        "Wrap the final extraction as `<FINAL>{...}</FINAL>`. Do not add text outside the requested tags. "
        "Use the reasoning route. First output `<ROUTE>reason</ROUTE>`, "
    )
    if style == "no_reason_block":
        return base + "then output `<FINAL>{...}</FINAL>`. If no valid event is expressed by the candidate set, the final JSON is {\"events\": []}."
    if style == "minimal_type_reason":
        return (
            base
            + "then output minimal JSON reasoning inside `<REASON>...</REASON>` with only a `types` array, "
            + "then output `<FINAL>{...}</FINAL>`. If no valid event is expressed by the candidate set, use `{\"types\":[]}` and final JSON {\"events\": []}."
        )
    if style == "final_first_reason":
        return (
            base
            + "then output `<FINAL>{...}</FINAL>`, then output minimal JSON reasoning inside `<REASON>...</REASON>` with only a `types` array. "
            + "If no valid event is expressed by the candidate set, use final JSON {\"events\": []} and `{\"types\":[]}`."
        )
    raise ValueError(style)


def free_route_instruction(style):
    base = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "The final extraction must be strict JSON with top-level key `events` and token offsets. "
        "Wrap the final extraction as `<FINAL>{...}</FINAL>`. Do not add text outside the requested tags. "
        "First decide whether this case should be solved directly or with compact reasoning. "
        "If it is simple, output `<ROUTE>direct</ROUTE>` followed by `<FINAL>{...}</FINAL>`. "
    )
    if style == "no_reason_block":
        return base + "If reasoning is useful, output `<ROUTE>reason</ROUTE>` followed by `<FINAL>{...}</FINAL>`."
    if style == "minimal_type_reason":
        return (
            base
            + "If reasoning is useful, output `<ROUTE>reason</ROUTE>`, minimal JSON reasoning inside `<REASON>...</REASON>` with only a `types` array, then `<FINAL>{...}</FINAL>`."
        )
    if style == "final_first_reason":
        return (
            base
            + "If reasoning is useful, output `<ROUTE>reason</ROUTE>`, then `<FINAL>{...}</FINAL>`, then minimal JSON reasoning inside `<REASON>...</REASON>` with only a `types` array."
        )
    raise ValueError(style)


def clone(row, branch, source, duplicate_idx=0):
    out = json.loads(json.dumps(row, ensure_ascii=False))
    meta = out.setdefault("meta", {})
    meta["e15_source"] = source
    meta["e15_duplicate_index"] = duplicate_idx
    meta["e15_branch"] = branch
    return out


def direct_anchor_row(direct_row, branch, source, duplicate_idx=0):
    out = clone(direct_row, branch, source, duplicate_idx)
    final_text = direct_row["output"]
    out["output"] = f"<ROUTE>direct</ROUTE>\n<FINAL>{final_text}</FINAL>"
    out["gold_output"] = final_text
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "reason_format_ablation_e15",
            "adaptive_dataset_role": "train",
            "adaptive_route_mode": "forced_direct_anchor",
            "adaptive_route_label": "direct",
            "adaptive_target_style": "direct_retention",
        }
    )
    return out


def reason_row(row, branch, style, source, duplicate_idx=0, dataset_role="train", route_mode="free_route"):
    out = clone(row, branch, source, duplicate_idx)
    out["instruction"] = free_route_instruction(style) if route_mode == "free_route" else forced_reason_instruction(style)
    out["output"] = reason_output(row, style)
    out["gold_output"] = final_json_text(row)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "reason_format_ablation_e15",
            "adaptive_dataset_role": dataset_role,
            "adaptive_route_mode": route_mode,
            "adaptive_route_label": "reason",
            "adaptive_target_style": style,
        }
    )
    return out


def direct_eval_row(row, branch, source, dataset_role):
    out = clone(row, branch, source, 0)
    meta = out.setdefault("meta", {})
    meta["adaptive_source"] = "reason_format_ablation_e15"
    meta["adaptive_dataset_role"] = dataset_role
    meta["adaptive_route_mode"] = "forced_direct"
    meta["adaptive_route_label"] = "direct"
    return out


def build_train_rows(branch, style):
    rng = random.Random(20260522)
    adaptive = load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_train_pos.jsonl")
    formal_direct = load_jsonl(FORMAL_DATA_DIR / f"{DATA_PREFIX}_train_pos.jsonl")
    direct_by_key = {key(row): row for row in formal_direct}
    reason_rows = [row for row in adaptive if route_label(row) == "reason"]
    direct_rows = [row for row in adaptive if route_label(row) != "reason"]
    reason_repeat = 18
    reason_part = [
        reason_row(row, branch, style, "reason_oversample", dup)
        for row in reason_rows
        for dup in range(reason_repeat)
    ]
    anchor_part = [
        direct_anchor_row(direct_by_key[key(row)], branch, "direct_anchor", 0)
        for row in direct_rows
    ]
    retention_part = [
        direct_anchor_row(direct_by_key[key(row)], branch, "reason_window_direct_retention", dup)
        for dup in range(3)
        for row in reason_rows
    ]
    rows = reason_part + anchor_part + retention_part
    rng.shuffle(rows)
    audit = {
        "source_branch": SOURCE_BRANCH,
        "style": style,
        "adaptive_train_count": len(adaptive),
        "source_reason_count": len(reason_rows),
        "source_direct_count": len(direct_rows),
        "reason_repeat": reason_repeat,
        "reason_rows_after_repeat": len(reason_part),
        "direct_anchor_rows": len(anchor_part),
        "retention_rows": len(retention_part),
        "total_count": len(rows),
        "route_label_counts": {"reason": len(reason_part), "direct": len(anchor_part) + len(retention_part)},
    }
    return rows, audit


def write_dataset(name, rows):
    file_name = f"{name}.jsonl"
    write_jsonl(DATA_DIR / file_name, rows)
    update_dataset_info(name, file_name)
    return file_name


def build_variant(variant_id, spec):
    branch = spec["branch"]
    style = spec["style"]
    train_rows, audit = build_train_rows(branch, style)
    dev_source = load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_dev_seen_pos.jsonl")
    dev_rows = [
        reason_row(row, branch, style, "dev_seen_source", 0, "dev_seen", "free_route")
        if route_label(row) == "reason"
        else clone(row, branch, "dev_seen_source", 0)
        for row in dev_source
    ]
    for row in dev_rows:
        row["instruction"] = free_route_instruction(style)
        row.setdefault("meta", {})["adaptive_source"] = "reason_format_ablation_e15"
        row.setdefault("meta", {})["adaptive_dataset_role"] = "dev_seen"
        row.setdefault("meta", {})["adaptive_target_style"] = style

    train_name = f"{ADAPTIVE_PREFIX}_{branch}_train_pos"
    dev_name = f"{ADAPTIVE_PREFIX}_{branch}_dev_seen_pos"
    write_dataset(train_name, train_rows)
    write_dataset(dev_name, dev_rows)
    write_json(DATA_DIR / f"{train_name}.meta.json", {"audit": audit, "created_at": now_iso()})
    write_json(DATA_DIR / f"{dev_name}.meta.json", {"num_examples": len(dev_rows), "created_at": now_iso()})

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        direct_source = load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_forced_direct_{split}_pos.jsonl")
        reason_source = load_jsonl(DATA_DIR / f"{ADAPTIVE_PREFIX}_{SOURCE_BRANCH}_forced_reason_{split}_pos.jsonl")
        direct_name = f"{ADAPTIVE_PREFIX}_{branch}_forced_direct_{split}_pos"
        reason_name = f"{ADAPTIVE_PREFIX}_{branch}_forced_reason_{split}_pos"
        direct_rows = [direct_eval_row(row, branch, f"forced_direct_{split}_source", split) for row in direct_source]
        reason_rows = [
            reason_row(row, branch, style, f"forced_reason_{split}_source", 0, split, "forced_reason")
            for row in reason_source
        ]
        write_dataset(direct_name, direct_rows)
        write_dataset(reason_name, reason_rows)
        eval_names.extend([direct_name, reason_name])

    config = write_config(branch, train_name, dev_name)
    note = write_note(variant_id, spec, train_name, dev_name, audit)
    return {
        "variant": variant_id,
        "branch": branch,
        "train_dataset": train_name,
        "dev_dataset": dev_name,
        "eval_datasets": eval_names,
        "config": config,
        "note": note.as_posix(),
        "audit": audit,
    }


def write_config(branch, train_name, dev_name):
    out_config = CONFIG_DIR / f"{RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    config = {
        "model_name_or_path": WARM_START,
        "template": "qwen",
        "dataset_dir": "/workspace/project/data/stage2_adaptive_datasets",
        "dataset": train_name,
        "eval_dataset": dev_name,
        "output_dir": f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{branch}_full",
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
        "learning_rate": 2.0e-6,
        "warmup_ratio": 0.05,
        "bf16": True,
        "val_size": 0.0,
        "eval_steps": 10,
        "do_eval": True,
        "save_only_model": True,
        "num_train_epochs": 4.0,
        "load_best_model_at_end": False,
        "deepspeed": "/workspace/project/configs/deepspeed/zero2_optimizer_offload_cpu.json",
    }
    write_yaml(out_config, config)
    return out_config.as_posix()


def write_note(variant_id, spec, train_name, dev_name, audit):
    timestamp = now_iso()
    exp_id = f"2026-05-22_stage2_4b_reason_format_ablation_{variant_id}_richere_split1_oracle_mixed_noise_qwen3_4b"
    out_dir = REPO / f"outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{spec['branch']}_full"
    config_path = CONFIG_DIR / f"{RUN_PREFIX}_{spec['branch']}_full_stepmatch.yaml"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    body = f"""---
id: {exp_id}
title: Stage2 4B Reason Format Ablation {spec['title']}
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
  - format-ablation
objective: Test whether the reason output format itself harms final extraction quality.
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
  branch: {spec['branch']}
  style: {spec['style']}
---

# Stage2 4B Reason Format Ablation {spec['title']}

## Goal

Evaluate whether `{spec['style']}` improves forced-reason extraction by reducing reasoning-format interference.

## Setup

- train dataset: `{train_name}`
- dev dataset: `{dev_name}`
- audit: `{json.dumps(audit, ensure_ascii=False, sort_keys=True)}`

## Commands

```bash
cd {REPO}
python3 scripts/prepare_4b_reason_format_ablation_e15_20260522.py
bash scripts/launch_4b_reason_format_ablation_e15_20260522.sh train {variant_id} <gpu>
bash scripts/launch_4b_reason_format_ablation_e15_20260522.sh devpick {variant_id} <gpu>
bash scripts/launch_4b_reason_format_ablation_e15_20260522.sh formal {variant_id} <gpu0> <gpu1> <gpu2> <gpu3>
python3 scripts/summarize_4b_reason_format_ablation_e15_20260522.py
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
- run devpick and formal forced-direct/forced-reason evaluation.
"""
    note.write_text(body, encoding="utf-8")
    return note


def main():
    results = {variant_id: build_variant(variant_id, spec) for variant_id, spec in VARIANTS.items()}
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
