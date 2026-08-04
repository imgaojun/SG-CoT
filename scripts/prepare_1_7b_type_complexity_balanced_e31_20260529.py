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

import prepare_1_7b_aug_reason_parallel_e27e_e28a_20260527 as e28  # noqa: E402
import prepare_1_7b_paired_augmentation_e27_20260527 as e27  # noqa: E402


RNG = random.Random(20260529)
SELECTED_ORIGINAL = 1456
AUGMENTED_COUNT = 600
TYPE_CAP = 80

SPECS = {
    "e31a": {
        "branch": "eventmentions_budget_e31a_type_complexity_none_aug",
        "title": "E31A Type-Complexity Balanced Direct Augmentation",
        "objective": "Control whether type plus argument/role complexity balancing improves direct extraction.",
        "description": "Same row budget as E30, but augmented rows are selected by event-type rarity plus argument/role complexity.",
        "train_budget": "none",
        "reason_style": "none",
        "target_style": "type_complexity_none_aug",
    },
    "e31b": {
        "branch": "eventmentions_budget_e31b_type_complexity_natural_step",
        "title": "E31B Type-Complexity Balanced Natural Step Reasoning",
        "objective": "Test whether natural step reasoning benefits from augmentation that balances type rarity and argument/role complexity.",
        "description": "Same selected inputs as E31A, trained with natural step-by-step reasoning targets.",
        "train_budget": "standard",
        "reason_style": "full_natural",
        "target_style": "type_complexity_natural_step",
    },
}


def install_specs():
    for variant, spec in SPECS.items():
        e27.SPECS[variant] = {
            "branch": spec["branch"],
            "title": spec["title"],
            "objective": spec["objective"],
            "description": spec["description"],
            "train_budgets": [spec["train_budget"]],
            "devpick_budget": spec["train_budget"],
            "target_style": spec["target_style"],
        }
        e28.SPECS[variant] = {
            "branch": spec["branch"],
            "title": spec["title"],
            "objective": spec["objective"],
            "description": spec["description"],
            "train_budget": spec["train_budget"],
            "target_style": spec["target_style"],
        }


def row_types(row):
    return sorted(
        {
            event.get("event_type")
            for event in e27.e21.gold_json(row).get("events", []) or []
            if isinstance(event, dict) and event.get("event_type")
        }
    )


def row_roles(row):
    roles = []
    for event in e27.e21.gold_json(row).get("events", []) or []:
        if not isinstance(event, dict):
            continue
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict) and arg.get("role"):
                roles.append(arg["role"])
    return sorted(set(roles))


def type_sample_counts(rows):
    counts = Counter()
    for row in rows:
        for typ in row_types(row):
            counts[typ] += 1
    return counts


def row_complexity(row):
    stats = e27.e21.event_stats(row)
    return stats["argument_count"] * 3 + stats["role_count"] * 2 + stats["event_count"]


def row_priority(row, type_counts, current_counts=None):
    current_counts = current_counts or type_counts
    types = row_types(row)
    rarity = max((TYPE_CAP - type_counts.get(typ, 0) for typ in types), default=0)
    live_deficit = max((TYPE_CAP - current_counts.get(typ, 0) for typ in types), default=0)
    return max(live_deficit, 0) * 10 + max(rarity, 0) * 4 + row_complexity(row)


def select_original_rows(train_rows, type_counts):
    ranked = sorted(
        enumerate(train_rows),
        key=lambda item: (-row_priority(item[1], type_counts), -row_complexity(item[1]), item[0]),
    )
    selected = [row for _, row in ranked[:SELECTED_ORIGINAL]]
    RNG.shuffle(selected)
    return selected


def make_complexity_guard_row(row, aug_id):
    tokens, rest = e27.parse_input(row["input"])
    roles = row_roles(row)
    role_words = []
    for role in roles[:4]:
        role_words.extend(role.replace("-", " ").split())
    suffix = ["Only", "explicit", "schema", "roles", "should", "be", "attached"]
    if role_words:
        suffix += ["including"] + role_words[:8]
    suffix += ["."]
    insertions = [(len(tokens), suffix)]
    new_tokens = e27.apply_insertions(tokens, insertions)
    new_gold = e27.shifted_gold(row, insertions)
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out["input"] = e27.rebuild_input(new_tokens, rest)
    out["output"] = json.dumps(new_gold, ensure_ascii=False)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "e31_augmented": True,
            "e31_augmentation_kind": "complexity_guard",
            "e31_aug_id": aug_id,
            "e31_insertions": [{"pos": pos, "tokens": toks} for pos, toks in insertions],
        }
    )
    return out


def make_augmented_row(row, kind, aug_id):
    if kind == "complexity_guard":
        return make_complexity_guard_row(row, aug_id)
    aug = e27.make_augmented_row(row, kind, aug_id)
    if aug is None:
        return None
    meta = aug.setdefault("meta", {})
    meta["e31_augmented"] = True
    meta["e31_augmentation_kind"] = kind
    meta["e31_aug_id"] = aug_id
    return aug


def choose_aug_kind(row, pool_size):
    stats = e27.e21.event_stats(row)
    candidate_types = (row.get("meta") or {}).get("candidate_types") or []
    kinds = ["boundary", "complexity_guard"]
    if stats["argument_count"] >= 2:
        kinds.append("role_contrast")
    if len(candidate_types) >= 5:
        kinds.append("hard_negative")
    if stats["argument_count"] >= 4:
        kinds = ["complexity_guard", "role_contrast"] + [kind for kind in kinds if kind not in {"complexity_guard", "role_contrast"}]
    return kinds[pool_size % len(kinds)]


def build_augmented_pool(train_rows, type_counts):
    current_counts = Counter(type_counts)
    candidates = [row for row in train_rows if row_types(row)]
    pool = []
    attempts = 0
    while len(pool) < AUGMENTED_COUNT and attempts < AUGMENTED_COUNT * 80:
        attempts += 1
        candidates.sort(
            key=lambda row: (
                -row_priority(row, type_counts, current_counts),
                -row_complexity(row),
                row.get("meta", {}).get("doc_id", ""),
            )
        )
        row = candidates[(attempts - 1) % min(len(candidates), 200)]
        kind = choose_aug_kind(row, len(pool))
        aug = make_augmented_row(row, kind, f"type_complexity_aug{len(pool):04d}")
        if aug is None:
            continue
        meta = aug.setdefault("meta", {})
        meta["e31_tail_types"] = row_types(row)
        meta["e31_complexity_score"] = row_complexity(row)
        pool.append(aug)
        for typ in row_types(row):
            current_counts[typ] += 1
    if len(pool) < AUGMENTED_COUNT:
        raise ValueError(f"only built {len(pool)} augmented rows")
    return pool, current_counts


def clone_row(row, budget, role, variant, source_kind):
    spec = SPECS[variant]
    if spec["reason_style"] == "none":
        out = e27.clone(row, budget, role, variant, source_kind)
    elif spec["reason_style"] == "full_natural":
        out = e28.clone_natural(row, role, variant, source_kind)
    else:
        raise ValueError(f"unknown reason style: {spec['reason_style']}")
    meta = out.setdefault("meta", {})
    meta["adaptive_target_style"] = spec["target_style"]
    meta["adaptive_reasoning_budget"] = budget
    meta["adaptive_budget_label"] = budget
    meta["e31_variant"] = variant
    meta["e31_branch"] = spec["branch"]
    meta["e31_reason_style"] = spec["reason_style"]
    return out


def write_note(variant, train_name, dev_name, audit):
    timestamp = e27.now_iso()
    log_stamp = datetime.now(e27.TZ).strftime("%Y-%m-%d %H:%M %z")
    spec = SPECS[variant]
    branch = spec["branch"]
    exp_id = f"2026-05-29_stage2_1_7b_{branch}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
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
  - type-complexity-augmentation
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
    - {REPO / 'reports/2026-05-29_e30_per_type_analysis.md'}
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
python3 scripts/prepare_1_7b_type_complexity_balanced_e31_20260529.py
bash scripts/launch_1_7b_paired_augmentation_e27_20260527.sh train {variant} 1
bash scripts/launch_1_7b_paired_augmentation_e27_20260527.sh devpick {variant} 1
bash scripts/launch_1_7b_paired_augmentation_e27_20260527.sh formal {variant} 1 2 3 4 7
```

## Run Log

### {log_stamp}

- prepared type-complexity balanced dataset/config/note.

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
    budget = spec["train_budget"]
    branch = spec["branch"]
    train = [clone_row(row, budget, "train", variant, "selected_original") for row in selected_original]
    train.extend(clone_row(row, budget, "train", variant, "selected_augmented") for row in selected_augmented)
    RNG.shuffle(train)

    train_name = f"{e27.ADAPTIVE_PREFIX}_{branch}_train_pos"
    e27.write_dataset(train_name, train)

    dev_rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_dev_seen_pos.jsonl")
    dev_name = f"{e27.ADAPTIVE_PREFIX}_{branch}_forced_{budget}_dev_seen_pos"
    e27.write_dataset(dev_name, [clone_row(row, budget, "dev_seen", variant, "original") for row in dev_rows])

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_{split}_pos.jsonl")
        for eval_budget in e27.FORMAL_BUDGETS:
            name = f"{e27.ADAPTIVE_PREFIX}_{branch}_forced_{eval_budget}_{split}_pos"
            if eval_budget == budget:
                eval_rows = [clone_row(row, budget, split, variant, "original") for row in rows]
            else:
                eval_rows = [e27.clone(row, eval_budget, split, variant, "original") for row in rows]
            e27.write_dataset(name, eval_rows)
            eval_names.append(name)

    source_counts = Counter(row["meta"].get("e27_source_kind", "unknown") for row in train)
    budget_counts = Counter(row["meta"]["adaptive_reasoning_budget"] for row in train)
    target_tokens = Counter()
    for row in train:
        target_tokens[row["meta"]["adaptive_reasoning_budget"]] += len(row["output"].split())
    aug_kind_counts = Counter((row.get("meta") or {}).get("e31_augmentation_kind") for row in selected_augmented)
    aug_type_counts = Counter()
    complexity_counts = Counter()
    for row in selected_augmented:
        for typ in row_types(row):
            aug_type_counts[typ] += 1
        score = row.get("meta", {}).get("e31_complexity_score", row_complexity(row))
        if score >= 16:
            complexity_counts["high"] += 1
        elif score >= 9:
            complexity_counts["mid"] += 1
        else:
            complexity_counts["low"] += 1

    audit = {
        **audit_base,
        "recipe": spec["description"],
        "variant": variant,
        "branch": branch,
        "train_budget": budget,
        "reason_style": spec["reason_style"],
        "selected_original_count": len(selected_original),
        "selected_augmented_count": len(selected_augmented),
        "total_train_rows": len(train),
        "train_source_counts": dict(source_counts),
        "train_budget_counts": dict(budget_counts),
        "approx_target_token_counts": dict(target_tokens),
        "augmentation": {
            "selected_kinds": dict(aug_kind_counts),
            "complexity_buckets": dict(complexity_counts),
            "augmented_type_counts_top20": dict(aug_type_counts.most_common(20)),
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
    e28.install_specs()
    train_rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_train_pos.jsonl")
    type_counts = type_sample_counts(train_rows)
    selected_original = select_original_rows(train_rows, type_counts)
    selected_augmented, final_counts = build_augmented_pool(train_rows, type_counts)
    eligible = {typ: max(0, TYPE_CAP - count) for typ, count in type_counts.items() if 0 < count < TYPE_CAP}
    audit_base = {
        "selection_policy": "event-type rarity plus argument/role complexity",
        "type_balance": {
            "cap": TYPE_CAP,
            "eligible_type_count": len(eligible),
            "eligible_types": dict(sorted(eligible.items(), key=lambda item: (-item[1], item[0]))),
            "initial_counts": {typ: type_counts[typ] for typ in sorted(eligible)},
            "post_aug_counts": {typ: final_counts[typ] for typ in sorted(eligible)},
        },
        "row_budget": {
            "selected_original": SELECTED_ORIGINAL,
            "selected_augmented": AUGMENTED_COUNT,
            "total_train_rows_per_variant": SELECTED_ORIGINAL + AUGMENTED_COUNT,
        },
    }
    payload = [
        write_variant("e31a", selected_original, selected_augmented, audit_base),
        write_variant("e31b", selected_original, selected_augmented, audit_base),
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
