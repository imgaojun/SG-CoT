import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
SCRIPT_DIR = REPO / "scripts"
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import prepare_1_7b_paired_augmentation_e27_20260527 as e27  # noqa: E402
from src.candidate_type_recall.schema_library import SCHEMA_LIBRARY  # noqa: E402


RNG = random.Random(20260531)
SELECTED_ORIGINAL = 1456
AUGMENTED_COUNTS = {
    "argument_prefix_boundary": 220,
    "argument_suffix_boundary": 220,
    "role_contrast_tail": 100,
    "hard_negative_tail": 100,
}

SPECS = {
    "e35a": {
        "branch": "eventmentions_budget_e35a_boundary_contrast_direct",
        "title": "E35A Boundary Contrast Direct",
        "objective": "Test whether boundary-focused input augmentation alone improves direct event extraction.",
        "description": "Selected originals plus argument-boundary contrast inputs, trained with forced none targets only.",
        "target_style": "boundary_contrast_direct",
        "train_mode": "none_only",
        "devpick_budget": "none",
    },
    "e35b": {
        "branch": "eventmentions_budget_e35b_boundary_check_reason",
        "title": "E35B Boundary Check Reasoning",
        "objective": "Test whether compact exact-span boundary checks reduce the E34 argument-boundary failure mode.",
        "description": "Selected originals plus boundary contrast inputs, trained with forced standard STEP_REASONING.",
        "target_style": "boundary_check_reason",
        "train_mode": "standard_only",
        "devpick_budget": "standard",
    },
    "e35c": {
        "branch": "eventmentions_budget_e35c_boundary_check_direct_anchor",
        "title": "E35C Boundary Check With Direct Anchor",
        "objective": "Test whether direct anchors preserve trigger/direct behavior while STEP_REASONING improves arguments.",
        "description": "E35B standard rows plus forced none anchors for selected original inputs.",
        "target_style": "boundary_check_direct_anchor",
        "train_mode": "standard_with_none_anchor",
        "devpick_budget": "standard",
    },
}


def install_specs():
    for variant, spec in SPECS.items():
        e27.SPECS[variant] = {
            "branch": spec["branch"],
            "title": spec["title"],
            "objective": spec["objective"],
            "description": spec["description"],
            "train_budgets": ["standard"],
            "devpick_budget": spec["devpick_budget"],
            "target_style": spec["target_style"],
        }


def deep_clone(payload):
    return json.loads(json.dumps(payload, ensure_ascii=False))


def row_types(row):
    return sorted(
        {
            event.get("event_type")
            for event in e27.e21.gold_json(row).get("events", []) or []
            if isinstance(event, dict) and event.get("event_type")
        }
    )


def row_priority(row):
    stats = e27.e21.event_stats(row)
    rare_bonus = sum(1 for typ in row_types(row) if typ.startswith(("Justice:", "Life:", "Transaction:", "Movement:")))
    return stats["argument_count"] * 4 + stats["event_count"] * 2 + stats["role_count"] + rare_bonus


def select_original_rows(train_rows):
    ranked = [(row_priority(row), idx, row) for idx, row in enumerate(train_rows) if row_types(row)]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [row for _, _, row in ranked[:SELECTED_ORIGINAL]]
    RNG.shuffle(selected)
    return selected


def iter_arguments(row):
    for event_index, event in enumerate(e27.e21.gold_json(row).get("events", []) or []):
        if not isinstance(event, dict):
            continue
        for arg_index, arg in enumerate(event.get("arguments", []) or []):
            if not isinstance(arg, dict):
                continue
            if arg.get("start") is None or arg.get("end") is None or not arg.get("text"):
                continue
            yield event_index, arg_index, event, arg


def choose_argument(row):
    args = list(iter_arguments(row))
    if not args:
        return None
    args.sort(key=lambda item: (-(item[3].get("end", 0) - item[3].get("start", 0)), item[0], item[1]))
    return args[0]


def make_boundary_row(row, kind, aug_id):
    picked = choose_argument(row)
    if picked is None:
        return None
    _, _, event, arg = picked
    tokens, rest = e27.parse_input(row["input"])
    start = arg.get("start")
    end = arg.get("end")
    if start is None or end is None or start < 0 or end > len(tokens) or start >= end:
        return None

    if kind == "argument_prefix_boundary":
        inserted = ["nearby"]
        insertions = [(start, inserted)]
        wrong_span = " ".join(inserted + tokens[start:end])
    elif kind == "argument_suffix_boundary":
        inserted = ["nearby"]
        insertions = [(end, inserted)]
        wrong_span = " ".join(tokens[start:end] + inserted)
    else:
        return None

    shifted = e27.shifted_gold(row, insertions)
    out = deep_clone(row)
    out["input"] = e27.rebuild_input(e27.apply_insertions(tokens, insertions), rest)
    out["output"] = json.dumps(shifted, ensure_ascii=False)
    shifted_arg = list(iter_arguments({"input": out["input"], "output": out["output"], "meta": out.get("meta", {})}))[0][3]
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "e35_augmented": True,
            "e35_augmentation_kind": kind,
            "e35_aug_id": aug_id,
            "e35_boundary_focus": {
                "event_type": event.get("event_type"),
                "role": arg.get("role"),
                "correct_text": shifted_arg.get("text") or arg.get("text"),
                "wrong_nearby_span": wrong_span,
                "insertions": [{"pos": pos, "tokens": toks} for pos, toks in insertions],
            },
        }
    )
    return out


def make_tail_row(row, kind, aug_id):
    source_kind = "role_contrast" if kind == "role_contrast_tail" else "hard_negative"
    aug = e27.make_augmented_row(row, source_kind, aug_id)
    if aug is None:
        return None
    meta = aug.setdefault("meta", {})
    meta.update(
        {
            "e35_augmented": True,
            "e35_augmentation_kind": kind,
            "e35_aug_id": aug_id,
            "e35_boundary_focus": {
                "event_type": ",".join(row_types(row)),
                "role": "schema",
                "correct_text": "gold event arguments only",
                "wrong_nearby_span": "added tail context",
                "insertions": meta.get("e27_insertions", []),
            },
        }
    )
    return aug


def row_supports_kind(row, kind):
    stats = e27.e21.event_stats(row)
    if kind in {"argument_prefix_boundary", "argument_suffix_boundary"}:
        return stats["argument_count"] > 0 and choose_argument(row) is not None
    if kind == "role_contrast_tail":
        return stats["argument_count"] >= 2
    if kind == "hard_negative_tail":
        return len((row.get("meta") or {}).get("candidate_types") or []) >= 5
    return False


def make_augmented_row(row, kind, aug_id):
    if kind in {"argument_prefix_boundary", "argument_suffix_boundary"}:
        return make_boundary_row(row, kind, aug_id)
    return make_tail_row(row, kind, aug_id)


def build_augmented_pool(train_rows):
    ranked = [(row_priority(row), idx, row) for idx, row in enumerate(train_rows) if row_types(row)]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    base = [row for _, _, row in ranked]
    pool = []
    by_kind = {}
    for kind, limit in AUGMENTED_COUNTS.items():
        candidates = [row for row in base if row_supports_kind(row, kind)]
        if not candidates:
            raise ValueError(f"no candidates for {kind}")
        made = []
        cursor = 0
        attempts = 0
        while len(made) < limit and attempts < limit * 50:
            attempts += 1
            row = candidates[cursor % len(candidates)]
            cursor += 1
            aug = make_augmented_row(row, kind, f"e35_{kind}_{len(made):04d}")
            if aug is None:
                continue
            made.append(aug)
            pool.append(aug)
        if len(made) < limit:
            raise ValueError(f"only built {len(made)} {kind} rows; need {limit}")
        by_kind[kind] = made
    RNG.shuffle(pool)
    return pool


def allowed_roles(event_type):
    return SCHEMA_LIBRARY.get(event_type, {}).get("core_roles", []) or []


def boundary_instruction():
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans. "
        "Then output `<REASONING_BUDGET>standard</REASONING_BUDGET>`. "
        "After the budget tag, output `<STEP_REASONING>...</STEP_REASONING>` with compact event-by-event checks: "
        "trigger/type, allowed roles, exact argument text spans, and any nearby span that must be excluded. "
        "Do not include offsets in the reasoning block. "
        "Finally output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
        "Do not output text outside the requested tags."
    )


def boundary_reason(row):
    payload = e27.e21.gold_json(row)
    focus = (row.get("meta") or {}).get("e35_boundary_focus") or {}
    lines = []
    for event in payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        roles = ", ".join(allowed_roles(event_type)) or "schema roles only"
        args = []
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict) and arg.get("role") and arg.get("text"):
                args.append(f'{arg.get("role")}="{arg.get("text")}"')
        arg_text = "; ".join(args) if args else "none"
        lines.append(f'Trigger="{trigger.get("text") or ""}" -> Type={event_type} | Allowed roles={roles} | Exact arguments={arg_text}.')
    if focus:
        lines.append(
            f'Boundary check: keep exact {focus.get("role")}="{focus.get("correct_text")}"; '
            f'exclude nearby span "{focus.get("wrong_nearby_span")}".'
        )
    if not lines:
        lines.append("No event: no candidate event is supported by an explicit trigger.")
    return "\n".join(lines)


def standard_output(row):
    final_payload = e27.e21.gold_json(row)
    mentions = json.dumps(e27.event_mentions_from_payload(final_payload), ensure_ascii=False, separators=(",", ":"))
    final = json.dumps(final_payload, ensure_ascii=False, separators=(",", ":"))
    return "\n".join(
        [
            f"<EVENT_MENTIONS>{mentions}</EVENT_MENTIONS>",
            "<REASONING_BUDGET>standard</REASONING_BUDGET>",
            f"<STEP_REASONING>{boundary_reason(row)}</STEP_REASONING>",
            f"<FINAL>{final}</FINAL>",
        ]
    )


def clone_standard(row, role, variant, source_kind):
    spec = SPECS[variant]
    out = e27.clone(row, "standard", role, variant, source_kind)
    out["instruction"] = boundary_instruction()
    out["output"] = standard_output(row)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "boundary_stable_e35",
            "adaptive_target_style": spec["target_style"],
            "adaptive_reasoning_budget": "standard",
            "adaptive_budget_label": "standard",
            "e35_variant": variant,
            "e35_branch": spec["branch"],
            "e35_source_kind": source_kind,
        }
    )
    return out


def clone_none(row, role, variant, source_kind):
    spec = SPECS[variant]
    out = e27.clone(row, "none", role, variant, source_kind)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "boundary_stable_e35",
            "adaptive_target_style": spec["target_style"],
            "adaptive_reasoning_budget": "none",
            "adaptive_budget_label": "none",
            "e35_variant": variant,
            "e35_branch": spec["branch"],
            "e35_source_kind": source_kind,
        }
    )
    return out


def clone_for_eval(row, budget, role, variant, source_kind):
    if budget == "standard":
        return clone_standard(row, role, variant, source_kind)
    return clone_none(row, role, variant, source_kind)


def write_note(variant, train_name, dev_name, audit):
    timestamp = e27.now_iso()
    log_stamp = datetime.now(e27.TZ).strftime("%Y-%m-%d %H:%M %z")
    spec = SPECS[variant]
    branch = spec["branch"]
    exp_id = f"2026-05-31_stage2_1_7b_{branch}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    out_dir = REPO / f"outputs/stage2_adaptive_runs_user/{e27.RUN_PREFIX}_{branch}_full"
    config_path = e27.CONFIG_DIR / f"{e27.RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    note = e27.EXPERIMENT_DIR / f"{exp_id}.md"
    body = f"""---
id: {exp_id}
title: Stage2 1.7B {spec['title']}
kind: experiment
status: planned
created_at: {timestamp}
updated_at: {timestamp}
owners:
  - codex
tags:
  - stage2
  - qwen3-1.7b
  - boundary-stable
  - forced-reasoning
objective: {spec['objective']}
artifacts:
  configs:
    - {config_path}
  outputs:
    - {out_dir}
related:
  plans:
    - {REPO / 'PLANS.md'}
  reports:
    - {REPO / 'reports/2026-05-31_e34_gold_conditioned_error_diagnosis.md'}
context:
  dataset: RichERE split1 oracle_mixed_noise_top10_shuffle
  base_model: Qwen3-1.7B
  warm_start: {e27.WARM_START}
  branch: {branch}
  variant: {variant}
---

# Stage2 1.7B {spec['title']}

## Goal

{spec['description']}

## Setup

- train dataset: `{train_name}`
- dev dataset: `{dev_name}`
- audit: `{json.dumps(audit, ensure_ascii=False, sort_keys=True)}`

## Commands

```bash
cd {REPO}
python3 scripts/prepare_1_7b_boundary_stable_e35_20260531.py
bash scripts/launch_1_7b_paired_augmentation_e27_20260527.sh train {variant} 1
bash scripts/launch_1_7b_paired_augmentation_e27_20260527.sh devpick {variant} 1
bash scripts/launch_1_7b_paired_augmentation_e27_20260527.sh formal {variant} 1 2 3 4 7
python3 scripts/summarize_1_7b_paired_augmentation_e27_20260527.py
```

## Run Log

### {log_stamp}

- prepared E35 boundary-stable dataset/config/note.

## Result

Pending.

## Conclusion

Pending.

## Next

- launch training.
- run dev checkpoint selection and formal forced-budget evaluation.
"""
    note.write_text(body, encoding="utf-8")
    return note


def write_variant(variant, selected_original, selected_augmented, audit_base):
    spec = SPECS[variant]
    branch = spec["branch"]
    train = []
    if spec["train_mode"] == "none_only":
        for row in selected_original:
            train.append(clone_none(row, "train", variant, "selected_original"))
        for row in selected_augmented:
            train.append(clone_none(row, "train", variant, "selected_augmented"))
    else:
        for row in selected_original:
            train.append(clone_standard(row, "train", variant, "selected_original"))
        for row in selected_augmented:
            train.append(clone_standard(row, "train", variant, "selected_augmented"))
        if spec["train_mode"] == "standard_with_none_anchor":
            for row in selected_original:
                train.append(clone_none(row, "train", variant, "selected_original_anchor"))
    RNG.shuffle(train)

    train_name = f"{e27.ADAPTIVE_PREFIX}_{branch}_train_pos"
    e27.write_dataset(train_name, train)

    dev_rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_dev_seen_pos.jsonl")
    dev_budget = spec["devpick_budget"]
    dev_name = f"{e27.ADAPTIVE_PREFIX}_{branch}_forced_{dev_budget}_dev_seen_pos"
    e27.write_dataset(dev_name, [clone_for_eval(row, dev_budget, "dev_seen", variant, "original") for row in dev_rows])

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_{split}_pos.jsonl")
        for budget in e27.FORMAL_BUDGETS:
            name = f"{e27.ADAPTIVE_PREFIX}_{branch}_forced_{budget}_{split}_pos"
            e27.write_dataset(name, [clone_for_eval(row, budget, split, variant, "original") for row in rows])
            eval_names.append(name)

    source_counts = Counter(row["meta"].get("e35_source_kind", "unknown") for row in train)
    budget_counts = Counter(row["meta"]["adaptive_reasoning_budget"] for row in train)
    aug_kind_counts = Counter((row.get("meta") or {}).get("e35_augmentation_kind") for row in selected_augmented)
    audit = {
        **audit_base,
        "recipe": spec["description"],
        "variant": variant,
        "branch": branch,
        "target_style": spec["target_style"],
        "train_mode": spec["train_mode"],
        "selected_original_count": len(selected_original),
        "selected_augmented_count": len(selected_augmented),
        "total_train_rows": len(train),
        "train_source_counts": dict(source_counts),
        "train_budget_counts": dict(budget_counts),
        "augmentation": {
            "selected_kinds": dict(aug_kind_counts),
            "planned_counts": AUGMENTED_COUNTS,
        },
        "formal_budgets": e27.FORMAL_BUDGETS,
        "training_recipe": {"learning_rate": 3.0e-6, "num_train_epochs": 3.0},
    }
    config = e27.write_config(variant, train_name, dev_name)
    note = write_note(variant, train_name, dev_name, audit)
    e27.e21.e15.write_json(e27.DATA_DIR / f"{train_name}.meta.json", {"audit": audit, "created_at": e27.now_iso()})
    return {
        "name": variant,
        "branch": branch,
        "train_dataset": train_name,
        "dev_dataset": dev_name,
        "eval_datasets": eval_names,
        "config": config.as_posix(),
        "note": note.as_posix(),
        "audit": audit,
    }


def main():
    install_specs()
    train_rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_train_pos.jsonl")
    selected_original = select_original_rows(train_rows)
    selected_augmented = build_augmented_pool(train_rows)
    audit_base = {
        "source_analysis": "E34 found Reason introduced argument_boundary errors most often and recommended add_argument_specificity most often.",
        "selection_policy": "argument-heavy originals plus boundary-nearby input perturbations; no changes to formal test sets.",
        "row_budget": {
            "selected_original": SELECTED_ORIGINAL,
            "selected_augmented": sum(AUGMENTED_COUNTS.values()),
            "e35a_e35b_train_rows": SELECTED_ORIGINAL + sum(AUGMENTED_COUNTS.values()),
            "e35c_train_rows": SELECTED_ORIGINAL + sum(AUGMENTED_COUNTS.values()) + SELECTED_ORIGINAL,
        },
    }
    payload = [
        write_variant("e35a", selected_original, selected_augmented, audit_base),
        write_variant("e35b", selected_original, selected_augmented, audit_base),
        write_variant("e35c", selected_original, selected_augmented, audit_base),
    ]
    e27.e21.e15.write_json(
        e27.DATA_DIR / f"{e27.ADAPTIVE_PREFIX}_eventmentions_budget_e35_boundary_stable_pool.meta.json",
        {
            "created_at": e27.now_iso(),
            "original_train_count": len(train_rows),
            "selected_original_count": len(selected_original),
            "selected_augmented_count": len(selected_augmented),
            "augmentation_kind_counts": dict(Counter((row.get("meta") or {}).get("e35_augmentation_kind") for row in selected_augmented)),
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
