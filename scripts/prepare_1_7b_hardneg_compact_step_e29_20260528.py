import json
import random
import sys
from collections import Counter
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
SCRIPT_DIR = REPO / "scripts"
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

import prepare_1_7b_aug_reason_parallel_e27e_e28a_20260527 as e28  # noqa: E402
import prepare_1_7b_balanced_augmentation_e27d_20260527 as e27d  # noqa: E402
import prepare_1_7b_paired_augmentation_e27_20260527 as e27  # noqa: E402


RNG = random.Random(20260528)
AUGMENTED_COUNT = 600

SPECS = {
    "e29a": {
        "branch": "eventmentions_budget_e29a_hardneg_natural_step_reason",
        "title": "E29A Hard-Negative Natural Step Reasoning",
        "objective": "Test whether hard-negative augmentation stacks with full natural-language step reasoning.",
        "description": "Use E27E hard-negative-heavy inputs with E28A full natural step-reasoning targets.",
        "source_mix": "hard_negative",
        "reason_style": "full_natural",
        "target_style": "hardneg_natural_step_reason",
    },
    "e29b": {
        "branch": "eventmentions_budget_e29b_balanced_compact_step_reason",
        "title": "E29B Balanced Compact Step Reasoning",
        "objective": "Isolate whether compact semi-structured natural steps improve over E28A full natural steps.",
        "description": "Use E27D/E28A balanced augmented inputs with compact semi-structured step-reasoning targets.",
        "source_mix": "balanced",
        "reason_style": "compact_step",
        "target_style": "balanced_compact_step_reason",
    },
    "e29c": {
        "branch": "eventmentions_budget_e29c_hardneg_compact_step_reason",
        "title": "E29C Hard-Negative Compact Step Reasoning",
        "objective": "Main candidate combining the strongest augmentation signal with compact step reasoning.",
        "description": "Use E27E hard-negative-heavy inputs with compact semi-structured step-reasoning targets.",
        "source_mix": "hard_negative",
        "reason_style": "compact_step",
        "target_style": "hardneg_compact_step_reason",
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
            "devpick_budget": "standard",
            "target_style": spec["target_style"],
        }
        e28.SPECS[variant] = {
            "branch": spec["branch"],
            "title": spec["title"],
            "objective": spec["objective"],
            "description": spec["description"],
            "train_budget": "standard",
            "target_style": spec["target_style"],
        }


def compact_instruction():
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans. "
        "Then output `<REASONING_BUDGET>standard</REASONING_BUDGET>`. "
        "After the budget tag, output `<STEP_REASONING>...</STEP_REASONING>` using compact lines: "
        "Event k | Trigger=\"...\" | Type=... | Arguments=Role:\"text\"; ... | Check=explicit trigger, schema roles. "
        "Finally output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
        "Do not output text outside the requested tags."
    )


def compact_step_reason(row):
    events = e27.e21.gold_json(row).get("events", [])
    if not events:
        return 'No event | Trigger="" | Type=None | Arguments=None | Check=no explicit supported trigger.'
    lines = []
    for i, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        seen = set()
        for arg in event.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            text = arg.get("text")
            if not role or not text:
                continue
            key = (role, text)
            if key in seen:
                continue
            seen.add(key)
            args.append(f'{role}:"{text}"')
        arg_text = "; ".join(args) if args else "None"
        checks = "explicit trigger"
        if args:
            checks += ", schema roles"
        lines.append(
            f'Event {i} | Trigger="{trigger.get("text") or ""}" | Type={event.get("event_type")} | '
            f"Arguments={arg_text} | Check={checks}."
        )
    return "\n".join(lines)


def compact_output(row):
    final_payload = e27.e21.gold_json(row)
    mentions = json.dumps(e27.event_mentions_from_payload(final_payload), ensure_ascii=False, separators=(",", ":"))
    final = json.dumps(final_payload, ensure_ascii=False, separators=(",", ":"))
    return "\n".join(
        [
            f"<EVENT_MENTIONS>{mentions}</EVENT_MENTIONS>",
            "<REASONING_BUDGET>standard</REASONING_BUDGET>",
            f"<STEP_REASONING>{compact_step_reason(row)}</STEP_REASONING>",
            f"<FINAL>{final}</FINAL>",
        ]
    )


def clone_reason(row, role, variant, source_kind):
    style = SPECS[variant]["reason_style"]
    if style == "full_natural":
        out = e28.clone_natural(row, role, variant, source_kind)
    elif style == "compact_step":
        out = e27.clone(row, "standard", role, variant, source_kind)
        out["instruction"] = compact_instruction()
        out["output"] = compact_output(row)
    else:
        raise ValueError(f"unknown reason style: {style}")
    meta = out.setdefault("meta", {})
    meta["adaptive_target_style"] = SPECS[variant]["target_style"]
    meta["adaptive_reasoning_budget"] = "standard"
    meta["adaptive_budget_label"] = "standard"
    meta["e29_variant"] = variant
    meta["e29_branch"] = SPECS[variant]["branch"]
    meta["e29_reason_style"] = style
    return out


def write_variant(variant, selected_original, selected_augmented):
    spec = SPECS[variant]
    branch = spec["branch"]
    train = []
    for row in selected_original:
        train.append(clone_reason(row, "train", variant, "selected_original"))
    for row in selected_augmented:
        train.append(clone_reason(row, "train", variant, "selected_augmented"))
    RNG.shuffle(train)

    train_name = f"{e27.ADAPTIVE_PREFIX}_{branch}_train_pos"
    e27.write_dataset(train_name, train)

    dev_rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_dev_seen_pos.jsonl")
    dev_name = f"{e27.ADAPTIVE_PREFIX}_{branch}_forced_standard_dev_seen_pos"
    e27.write_dataset(dev_name, [clone_reason(row, "dev_seen", variant, "original") for row in dev_rows])

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_{split}_pos.jsonl")
        for budget in e27.FORMAL_BUDGETS:
            name = f"{e27.ADAPTIVE_PREFIX}_{branch}_forced_{budget}_{split}_pos"
            if budget == "standard":
                eval_rows = [clone_reason(row, split, variant, "original") for row in rows]
            else:
                eval_rows = [e27.clone(row, budget, split, variant, "original") for row in rows]
            e27.write_dataset(name, eval_rows)
            eval_names.append(name)

    source_counts = Counter(row["meta"].get("e27_source_kind", "unknown") for row in train)
    aug_kind_counts = Counter((row.get("meta") or {}).get("e27_augmentation_kind") for row in selected_augmented)
    budget_counts = Counter(row["meta"]["adaptive_reasoning_budget"] for row in train)
    target_tokens = Counter()
    for row in train:
        target_tokens[row["meta"]["adaptive_reasoning_budget"]] += len(row["output"].split())
    audit = {
        "recipe": spec["description"],
        "variant": variant,
        "branch": branch,
        "source_mix": spec["source_mix"],
        "reason_style": spec["reason_style"],
        "selected_original_count": len(selected_original),
        "selected_augmented_count": len(selected_augmented),
        "total_train_rows": len(train),
        "train_budget_counts": dict(budget_counts),
        "train_source_counts": dict(source_counts),
        "approx_target_token_counts": dict(target_tokens),
        "augmentation": {"selected_kinds": dict(aug_kind_counts)},
        "formal_budgets": e27.FORMAL_BUDGETS,
        "training_recipe": {"learning_rate": 3.0e-6, "num_train_epochs": 3.0},
    }
    config = e27.write_config(variant, train_name, dev_name)
    note = e27.write_note(variant, train_name, dev_name, audit)
    text = note.read_text(encoding="utf-8")
    text = text.replace(
        "python3 scripts/prepare_1_7b_paired_augmentation_e27_20260527.py",
        "python3 scripts/prepare_1_7b_hardneg_compact_step_e29_20260528.py",
    )
    note.write_text(text, encoding="utf-8")
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
    balanced_aug = e27d.select_augmented_rows(e27.build_augmented_pool(train_rows))
    hardneg_aug = e28.generate_kind_pool(train_rows, "hard_negative", AUGMENTED_COUNT)

    payload = [
        write_variant("e29a", e27d.select_original_rows(train_rows), hardneg_aug),
        write_variant("e29b", e27d.select_original_rows(train_rows), balanced_aug),
        write_variant("e29c", e27d.select_original_rows(train_rows), hardneg_aug),
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
