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
SELECTED_ORIGINAL = 1456
AUGMENTED_COUNT = 600
TYPE_CAP = 80

SPECS = {
    "e30a": {
        "branch": "eventmentions_budget_e30a_tail_type_balanced_none_aug",
        "title": "E30A Tail-Type Balanced Direct Augmentation",
        "objective": "Test whether frequency-balancing low-resource event types helps direct extraction.",
        "description": "Same total train rows as prior controlled runs, but augmented rows are allocated to train event types below the cap.",
        "train_budget": "none",
        "reason_style": "none",
        "target_style": "tail_type_balanced_none_aug",
    },
    "e30b": {
        "branch": "eventmentions_budget_e30b_tail_type_balanced_natural_step",
        "title": "E30B Tail-Type Balanced Natural Step Reasoning",
        "objective": "Test whether tail-type balanced augmentation improves full natural step reasoning.",
        "description": "Tail-type balanced augmented inputs with full natural step-reasoning targets.",
        "train_budget": "standard",
        "reason_style": "full_natural",
        "target_style": "tail_type_balanced_natural_step",
    },
    "e30c": {
        "branch": "eventmentions_budget_e30c_tail_type_balanced_minimal_type_step",
        "title": "E30C Tail-Type Balanced Minimal Type Step Reasoning",
        "objective": "Main candidate: balance low-resource event types while keeping reasoning focused on trigger/type only.",
        "description": "Tail-type balanced augmented inputs with minimal trigger/type step reasoning and no argument list in the reasoning block.",
        "train_budget": "standard",
        "reason_style": "minimal_type_step",
        "target_style": "tail_type_balanced_minimal_type_step",
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
    types = []
    for event in e27.e21.gold_json(row).get("events", []) or []:
        if isinstance(event, dict) and event.get("event_type"):
            types.append(event["event_type"])
    return sorted(set(types))


def type_sample_counts(rows):
    counts = Counter()
    for row in rows:
        for typ in row_types(row):
            counts[typ] += 1
    return counts


def deficit_map(rows):
    counts = type_sample_counts(rows)
    return {typ: max(0, TYPE_CAP - count) for typ, count in counts.items() if 0 < count < TYPE_CAP}


def row_deficit(row, deficits):
    types = row_types(row)
    return max((deficits.get(typ, 0) for typ in types), default=0)


def select_original_rows(train_rows, deficits):
    prioritized = [(row_deficit(row, deficits), idx, row) for idx, row in enumerate(train_rows)]
    prioritized.sort(key=lambda item: (-item[0], item[1]))
    selected = [row for score, _, row in prioritized if score > 0]
    selected = selected[:SELECTED_ORIGINAL]
    selected_ids = {id(row) for row in selected}
    if len(selected) < SELECTED_ORIGINAL:
        fillers = [row for _, _, row in prioritized if id(row) not in selected_ids]
        selected.extend(fillers[: SELECTED_ORIGINAL - len(selected)])
    RNG.shuffle(selected)
    return selected


def make_context_augmented_row(row, aug_id):
    tokens, rest = e27.parse_input(row["input"])
    insertions = [(len(tokens), ["This", "additional", "background", "does", "not", "change", "the", "event", "type", "."])]
    new_tokens = e27.apply_insertions(tokens, insertions)
    new_gold = e27.shifted_gold(row, insertions)
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out["input"] = e27.rebuild_input(new_tokens, rest)
    out["output"] = json.dumps(new_gold, ensure_ascii=False)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "e30_augmented": True,
            "e30_augmentation_kind": "context_guard",
            "e30_aug_id": aug_id,
            "e30_insertions": [{"pos": pos, "tokens": toks} for pos, toks in insertions],
        }
    )
    return out


def make_tail_augmented_row(row, kind, aug_id):
    if kind == "context_guard":
        return make_context_augmented_row(row, aug_id)
    aug = e27.make_augmented_row(row, kind, aug_id)
    if aug is not None:
        meta = aug.setdefault("meta", {})
        meta["e30_augmented"] = True
        meta["e30_augmentation_kind"] = kind
        meta["e30_aug_id"] = aug_id
    return aug


def build_tail_augmented_pool(train_rows, deficits):
    initial_counts = type_sample_counts(train_rows)
    current = type_sample_counts(train_rows)
    candidates = [row for row in train_rows if row_deficit(row, deficits) > 0]
    kinds = ["boundary", "context_guard", "hard_negative", "role_contrast"]
    pool = []
    attempts = 0
    while len(pool) < AUGMENTED_COUNT and attempts < AUGMENTED_COUNT * 50:
        attempts += 1
        def priority(row):
            types = row_types(row)
            live_deficit = max((TYPE_CAP - current.get(typ, 0) for typ in types), default=0)
            rarity = max((TYPE_CAP - initial_counts.get(typ, 0) for typ in types), default=0)
            return max(live_deficit, 0), rarity

        candidates.sort(key=lambda row: (-priority(row)[0], -priority(row)[1], row.get("meta", {}).get("doc_id", "")))
        row = candidates[(attempts - 1) % len(candidates)]
        stats = e27.e21.event_stats(row)
        candidate_types = (row.get("meta") or {}).get("candidate_types") or []
        preferred = []
        for kind in kinds:
            if kind == "role_contrast" and stats["argument_count"] < 2:
                continue
            if kind == "hard_negative" and len(candidate_types) < 5:
                continue
            preferred.append(kind)
        kind = preferred[len(pool) % len(preferred)]
        aug = make_tail_augmented_row(row, kind, f"tail_type_aug{len(pool):04d}")
        if aug is None:
            continue
        meta = aug.setdefault("meta", {})
        meta["e30_tail_types"] = row_types(row)
        pool.append(aug)
        for typ in row_types(row):
            if typ in current:
                current[typ] += 1
    if len(pool) < AUGMENTED_COUNT:
        raise ValueError(f"only built {len(pool)} tail-type augmented rows")
    return pool, current


def minimal_type_instruction():
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans. "
        "Then output `<REASONING_BUDGET>standard</REASONING_BUDGET>`. "
        "After the budget tag, output `<STEP_REASONING>...</STEP_REASONING>` with one compact line per event: "
        "Event k | Trigger=\"...\" | Type=... . Do not list arguments in the reasoning block. "
        "Finally output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
        "Do not output text outside the requested tags."
    )


def minimal_type_reason(row):
    events = e27.e21.gold_json(row).get("events", []) or []
    if not events:
        return 'No event | Trigger="" | Type=None.'
    lines = []
    for i, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        lines.append(f'Event {i} | Trigger="{trigger.get("text") or ""}" | Type={event.get("event_type")}.')
    return "\n".join(lines)


def minimal_type_output(row):
    final_payload = e27.e21.gold_json(row)
    mentions = json.dumps(e27.event_mentions_from_payload(final_payload), ensure_ascii=False, separators=(",", ":"))
    final = json.dumps(final_payload, ensure_ascii=False, separators=(",", ":"))
    return "\n".join(
        [
            f"<EVENT_MENTIONS>{mentions}</EVENT_MENTIONS>",
            "<REASONING_BUDGET>standard</REASONING_BUDGET>",
            f"<STEP_REASONING>{minimal_type_reason(row)}</STEP_REASONING>",
            f"<FINAL>{final}</FINAL>",
        ]
    )


def clone_row(row, budget, role, variant, source_kind):
    spec = SPECS[variant]
    if spec["reason_style"] == "none":
        out = e27.clone(row, budget, role, variant, source_kind)
    elif spec["reason_style"] == "full_natural":
        out = e28.clone_natural(row, role, variant, source_kind)
    elif spec["reason_style"] == "minimal_type_step":
        out = e27.clone(row, "standard", role, variant, source_kind)
        out["instruction"] = minimal_type_instruction()
        out["output"] = minimal_type_output(row)
    else:
        raise ValueError(f"unknown reason style: {spec['reason_style']}")
    meta = out.setdefault("meta", {})
    meta["adaptive_target_style"] = spec["target_style"]
    meta["adaptive_reasoning_budget"] = budget
    meta["adaptive_budget_label"] = budget
    meta["e30_variant"] = variant
    meta["e30_branch"] = spec["branch"]
    meta["e30_reason_style"] = spec["reason_style"]
    return out


def write_variant(variant, selected_original, selected_augmented, audit_base):
    spec = SPECS[variant]
    budget = spec["train_budget"]
    branch = spec["branch"]
    train = []
    for row in selected_original:
        train.append(clone_row(row, budget, "train", variant, "selected_original"))
    for row in selected_augmented:
        train.append(clone_row(row, budget, "train", variant, "selected_augmented"))
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
    aug_kind_counts = Counter((row.get("meta") or {}).get("e30_augmentation_kind") for row in selected_augmented)
    aug_type_counts = Counter()
    for row in selected_augmented:
        for typ in row_types(row):
            aug_type_counts[typ] += 1
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
            "augmented_type_counts_top20": dict(aug_type_counts.most_common(20)),
        },
        "formal_budgets": e27.FORMAL_BUDGETS,
        "training_recipe": {"learning_rate": 3.0e-6, "num_train_epochs": 3.0},
    }
    config = e27.write_config(variant, train_name, dev_name)
    note = e27.write_note(variant, train_name, dev_name, audit)
    text = note.read_text(encoding="utf-8").replace(
        "python3 scripts/prepare_1_7b_paired_augmentation_e27_20260527.py",
        "python3 scripts/prepare_1_7b_tail_type_balanced_e30_20260528.py",
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
    deficits = deficit_map(train_rows)
    selected_original = select_original_rows(train_rows, deficits)
    selected_augmented, final_counts = build_tail_augmented_pool(train_rows, deficits)
    initial_counts = type_sample_counts(train_rows)
    audit_base = {
        "type_balance": {
            "cap": TYPE_CAP,
            "eligible_type_count": len(deficits),
            "eligible_types": dict(sorted(deficits.items(), key=lambda item: (-item[1], item[0]))),
            "initial_counts": {typ: initial_counts[typ] for typ in sorted(deficits)},
            "post_aug_counts": {typ: final_counts[typ] for typ in sorted(deficits)},
        }
    }
    payload = [
        write_variant("e30a", selected_original, selected_augmented, audit_base),
        write_variant("e30b", selected_original, selected_augmented, audit_base),
        write_variant("e30c", selected_original, selected_augmented, audit_base),
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
