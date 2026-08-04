import json
import random
import sys
from collections import Counter
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
SCRIPT_DIR = REPO / "scripts"
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

import prepare_1_7b_balanced_augmentation_e27d_20260527 as e27d  # noqa: E402
import prepare_1_7b_paired_augmentation_e27_20260527 as e27  # noqa: E402


RNG = random.Random(20260528)
SELECTED_ORIGINAL = 1456
HIGH_RISK_ORIGINAL = 456
AUGMENTED_COUNT = 600

SPECS = {
    "e27e": {
        "branch": "eventmentions_budget_e27e_hardneg_none_aug",
        "title": "E27E Hard-Negative Direct Augmentation",
        "objective": "Ablate whether hard-negative augmentation is the strongest component under the same train-row budget.",
        "description": "Same total train rows as E27D, but all augmented inputs are hard-negative focused and training remains forced none.",
        "train_budget": "none",
        "target_style": "hardneg_none_aug",
    },
    "e28a": {
        "branch": "eventmentions_budget_e28a_balanced_natural_step_reason",
        "title": "E28A Balanced Natural Step Reasoning",
        "objective": "Test whether natural-language step-by-step reasoning adds value on top of the E27D balanced augmentation distribution.",
        "description": "Use the same 1456 original + 600 balanced augmented inputs as E27D, but train forced standard with a concise natural step-reasoning block.",
        "train_budget": "standard",
        "target_style": "balanced_natural_step_reason",
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


def generate_kind_pool(train_rows, kind, limit):
    pool = []
    for row in e27.augmentation_candidates(train_rows):
        if len(pool) >= limit:
            break
        if kind == "hard_negative" and len((row.get("meta") or {}).get("candidate_types") or []) < 5:
            continue
        if kind == "role_contrast" and e27.e21.event_stats(row)["argument_count"] < 2:
            continue
        aug = e27.make_augmented_row(row, kind, f"{kind}_aug{len(pool):04d}")
        if aug is not None:
            pool.append(aug)
    if len(pool) < limit:
        raise ValueError(f"could only build {len(pool)} {kind} rows; need {limit}")
    return pool


def natural_step_reason(row):
    events = e27.e21.gold_json(row).get("events", [])
    lines = []
    if not events:
        return "No candidate event is supported by an explicit trigger, so the final event list is empty."
    for i, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict) and arg.get("role") and arg.get("text"):
                args.append(f"{arg.get('role')}={arg.get('text')}")
        arg_text = "; ".join(args) if args else "no explicit arguments"
        lines.append(
            f"Step {i}: choose {event.get('event_type')} because trigger '{trigger.get('text')}' is explicit; "
            f"then attach arguments: {arg_text}."
        )
    return " ".join(lines)


def natural_instruction():
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans. "
        "Then output `<REASONING_BUDGET>standard</REASONING_BUDGET>`. "
        "After the budget tag, output `<STEP_REASONING>...</STEP_REASONING>` as concise natural-language steps: "
        "identify each trigger, choose its event type, and attach only explicitly supported arguments. "
        "Finally output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
        "Do not output text outside the requested tags."
    )


def natural_output(row):
    final_payload = e27.e21.gold_json(row)
    mentions = json.dumps(e27.event_mentions_from_payload(final_payload), ensure_ascii=False, separators=(",", ":"))
    final = json.dumps(final_payload, ensure_ascii=False, separators=(",", ":"))
    reason = natural_step_reason(row)
    return "\n".join(
        [
            f"<EVENT_MENTIONS>{mentions}</EVENT_MENTIONS>",
            "<REASONING_BUDGET>standard</REASONING_BUDGET>",
            f"<STEP_REASONING>{reason}</STEP_REASONING>",
            f"<FINAL>{final}</FINAL>",
        ]
    )


def clone_natural(row, role, variant, source_kind):
    out = e27.clone(row, "standard", role, variant, source_kind)
    out["instruction"] = natural_instruction()
    out["output"] = natural_output(row)
    meta = out.setdefault("meta", {})
    meta["adaptive_target_style"] = SPECS[variant]["target_style"]
    meta["adaptive_reasoning_budget"] = "standard"
    meta["adaptive_budget_label"] = "standard"
    return out


def write_variant(variant, selected_original, selected_augmented):
    spec = SPECS[variant]
    budget = spec["train_budget"]
    train = []
    for row in selected_original:
        if variant == "e28a":
            train.append(clone_natural(row, "train", variant, "selected_original"))
        else:
            train.append(e27.clone(row, budget, "train", variant, "selected_original"))
    for row in selected_augmented:
        if variant == "e28a":
            train.append(clone_natural(row, "train", variant, "selected_augmented"))
        else:
            train.append(e27.clone(row, budget, "train", variant, "selected_augmented"))
    RNG.shuffle(train)

    branch = spec["branch"]
    train_name = f"{e27.ADAPTIVE_PREFIX}_{branch}_train_pos"
    e27.write_dataset(train_name, train)

    dev_rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_dev_seen_pos.jsonl")
    dev_name = f"{e27.ADAPTIVE_PREFIX}_{branch}_forced_{budget}_dev_seen_pos"
    if variant == "e28a":
        dev = [clone_natural(row, "dev_seen", variant, "original") for row in dev_rows]
    else:
        dev = [e27.clone(row, budget, "dev_seen", variant, "original") for row in dev_rows]
    e27.write_dataset(dev_name, dev)

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_{split}_pos.jsonl")
        for eval_budget in e27.FORMAL_BUDGETS:
            name = f"{e27.ADAPTIVE_PREFIX}_{branch}_forced_{eval_budget}_{split}_pos"
            if variant == "e28a" and eval_budget == "standard":
                eval_rows = [clone_natural(row, split, variant, "original") for row in rows]
            else:
                eval_rows = [e27.clone(row, eval_budget, split, variant, "original") for row in rows]
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
    balanced_aug = e27d.select_augmented_rows(e27.build_augmented_pool(train_rows))
    hardneg_aug = generate_kind_pool(train_rows, "hard_negative", AUGMENTED_COUNT)

    selected_original_e28a = e27d.select_original_rows(train_rows)
    selected_original_e27e = e27d.select_original_rows(train_rows)
    payload = [
        write_variant("e28a", selected_original_e28a, balanced_aug),
        write_variant("e27e", selected_original_e27e, hardneg_aug),
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
