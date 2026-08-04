import json
import random
import sys
from collections import Counter
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
SCRIPT_DIR = REPO / "scripts"
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

import prepare_1_7b_paired_augmentation_e27_20260527 as e27  # noqa: E402


VARIANT = "e27d"
BRANCH = "eventmentions_budget_e27d_balanced_none_aug"
SELECTED_ORIGINAL = 1456
HIGH_RISK_ORIGINAL = 456
AUGMENTED_PER_KIND = 200
RNG = random.Random(20260527)


e27.SPECS[VARIANT] = {
    "branch": BRANCH,
    "title": "E27D Balanced Direct Augmentation",
    "objective": "Test whether targeted augmentation helps direct extraction under the same total train-row budget as the original direct baseline.",
    "description": (
        "Same total train rows as the original direct baseline: high-risk original inputs plus a balanced targeted "
        "augmentation subset, trained only with forced none outputs."
    ),
    "train_budgets": ["none"],
    "devpick_budget": "none",
    "target_style": "balanced_none_aug",
}


def score_original(row):
    stats = e27.e21.event_stats(row)
    score = stats["argument_count"] * 3 + stats["event_count"] * 2 + stats["role_count"]
    types = (row.get("meta") or {}).get("gold_event_types") or []
    if any(t.startswith(("Justice:", "Movement:", "Transaction:", "Life:")) for t in types):
        score += 2
    return score


def select_original_rows(train_rows):
    scored = [(score_original(row), idx, row) for idx, row in enumerate(train_rows)]
    scored.sort(key=lambda item: (-item[0], item[1]))
    high_risk_items = scored[:HIGH_RISK_ORIGINAL]
    high_risk_indices = {idx for _, idx, _ in high_risk_items}
    remaining = [(idx, row) for _, idx, row in scored if idx not in high_risk_indices]
    RNG.shuffle(remaining)
    random_items = remaining[: SELECTED_ORIGINAL - HIGH_RISK_ORIGINAL]
    selected = [row for _, _, row in high_risk_items] + [row for _, row in random_items]
    RNG.shuffle(selected)
    return selected


def select_augmented_rows(aug_rows):
    by_kind = {}
    for row in aug_rows:
        kind = (row.get("meta") or {}).get("e27_augmentation_kind")
        by_kind.setdefault(kind, []).append(row)
    selected = []
    for kind in ["boundary", "role_contrast", "hard_negative"]:
        rows = by_kind.get(kind, [])
        if len(rows) < AUGMENTED_PER_KIND:
            raise ValueError(f"not enough augmented rows for {kind}: {len(rows)}")
        selected.extend(rows[:AUGMENTED_PER_KIND])
    RNG.shuffle(selected)
    return selected


def write_balanced_variant(train_rows, aug_rows):
    selected_original = select_original_rows(train_rows)
    selected_augmented = select_augmented_rows(aug_rows)
    train = []
    for row in selected_original:
        train.append(e27.clone(row, "none", "train", VARIANT, "selected_original"))
    for row in selected_augmented:
        train.append(e27.clone(row, "none", "train", VARIANT, "selected_augmented"))
    RNG.shuffle(train)

    train_name = f"{e27.ADAPTIVE_PREFIX}_{BRANCH}_train_pos"
    e27.write_dataset(train_name, train)

    dev_rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_dev_seen_pos.jsonl")
    dev_name = f"{e27.ADAPTIVE_PREFIX}_{BRANCH}_forced_none_dev_seen_pos"
    e27.write_dataset(dev_name, [e27.clone(row, "none", "dev_seen", VARIANT, "original") for row in dev_rows])

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_{split}_pos.jsonl")
        for budget in e27.FORMAL_BUDGETS:
            name = f"{e27.ADAPTIVE_PREFIX}_{BRANCH}_forced_{budget}_{split}_pos"
            e27.write_dataset(name, [e27.clone(row, budget, split, VARIANT, "original") for row in rows])
            eval_names.append(name)

    source_counts = Counter(row["meta"].get("e27_source_kind", "unknown") for row in train)
    budget_counts = Counter(row["meta"]["adaptive_reasoning_budget"] for row in train)
    target_tokens = Counter()
    for row in train:
        target_tokens[row["meta"]["adaptive_reasoning_budget"]] += len(row["output"].split())
    audit = {
        "recipe": e27.SPECS[VARIANT]["description"],
        "variant": VARIANT,
        "branch": BRANCH,
        "original_train_count_available": len(train_rows),
        "selected_original_count": len(selected_original),
        "selected_augmented_count": len(selected_augmented),
        "selected_total_input_count": len(selected_original) + len(selected_augmented),
        "train_budgets": ["none"],
        "formal_budgets": e27.FORMAL_BUDGETS,
        "total_train_rows": len(train),
        "train_source_counts": dict(source_counts),
        "train_budget_counts": dict(budget_counts),
        "approx_target_token_counts": dict(target_tokens),
        "selection": {
            "high_risk_original": HIGH_RISK_ORIGINAL,
            "random_original": SELECTED_ORIGINAL - HIGH_RISK_ORIGINAL,
            "augmented_per_kind": AUGMENTED_PER_KIND,
            "same_train_rows_as_original_direct": True,
        },
        "augmentation": {
            "available_kinds": dict(Counter((row.get("meta") or {}).get("e27_augmentation_kind") for row in aug_rows)),
            "selected_kinds": dict(Counter((row.get("meta") or {}).get("e27_augmentation_kind") for row in selected_augmented)),
        },
        "training_recipe": {"learning_rate": 3.0e-6, "num_train_epochs": 3.0},
    }
    config = e27.write_config(VARIANT, train_name, dev_name)
    note = e27.write_note(VARIANT, train_name, dev_name, audit)
    e27.e21.e15.write_json(e27.DATA_DIR / f"{train_name}.meta.json", {"audit": audit, "created_at": e27.now_iso()})
    return {
        "name": VARIANT,
        "branch": BRANCH,
        "train_dataset": train_name,
        "dev_dataset": dev_name,
        "eval_datasets": eval_names,
        "config": config.as_posix(),
        "note": note.as_posix(),
        "audit": audit,
    }


def main():
    train_rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_train_pos.jsonl")
    aug_rows = e27.build_augmented_pool(train_rows)
    payload = write_balanced_variant(train_rows, aug_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
