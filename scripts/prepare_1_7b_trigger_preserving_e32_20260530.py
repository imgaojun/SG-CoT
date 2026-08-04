import json
import random
import sys
from collections import Counter
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
SCRIPT_DIR = REPO / "scripts"
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import prepare_1_7b_aug_reason_parallel_e27e_e28a_20260527 as e28  # noqa: E402
import prepare_1_7b_paired_augmentation_e27_20260527 as e27  # noqa: E402
from src.candidate_type_recall.schema_library import SCHEMA_LIBRARY  # noqa: E402


RNG = random.Random(20260530)
SELECTED_ORIGINAL = 1456
AUGMENTED_COUNTS = {"context_guard": 240, "hard_negative": 180, "role_contrast": 180}
TYPE_CAP = 80

SPECS = {
    "e32a": {
        "branch": "eventmentions_budget_e32a_trigger_preserving_tail_natural_step",
        "title": "E32A Trigger-Preserving Tail Natural Step",
        "objective": "Test whether E30B-style natural step reasoning improves when augmentation preserves trigger/argument spans.",
        "description": "Tail-type balanced, trigger-preserving augmented inputs with the original natural step target.",
        "target_style": "trigger_preserving_tail_natural_step",
        "reason_style": "full_natural",
        "train_budgets": ["standard"],
    },
    "e32b": {
        "branch": "eventmentions_budget_e32b_trigger_role_ground_natural_step",
        "title": "E32B Trigger-Role-Ground Natural Step",
        "objective": "Test whether explicit trigger-role-ground constraints improve natural step information transfer.",
        "description": "Same inputs as E32A, but STEP_REASONING transmits trigger/type, allowed roles, explicit argument texts, and a no-unsupported-role constraint.",
        "target_style": "trigger_role_ground_natural_step",
        "reason_style": "trigger_role_ground",
        "train_budgets": ["standard"],
    },
    "e32c": {
        "branch": "eventmentions_budget_e32c_trigger_role_ground_direct_anchor",
        "title": "E32C Trigger-Role-Ground With Direct Anchor",
        "objective": "Test whether direct final anchors protect Trigger/Argument while preserving E32B reasoning gains.",
        "description": "E32B standard rows plus direct none anchors for selected original rows.",
        "target_style": "trigger_role_ground_direct_anchor",
        "reason_style": "trigger_role_ground",
        "train_budgets": ["standard", "none_anchor"],
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


def row_types(row):
    return sorted(
        {
            event.get("event_type")
            for event in e27.e21.gold_json(row).get("events", []) or []
            if isinstance(event, dict) and event.get("event_type")
        }
    )


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
    return max((deficits.get(typ, 0) for typ in row_types(row)), default=0)


def row_priority(row, deficits, initial_counts, current_counts):
    types = row_types(row)
    live_deficit = max((TYPE_CAP - current_counts.get(typ, 0) for typ in types), default=0)
    rarity = max((TYPE_CAP - initial_counts.get(typ, 0) for typ in types), default=0)
    stats = e27.e21.event_stats(row)
    return max(live_deficit, 0) * 10 + max(rarity, 0) * 3 + stats["argument_count"] + row_deficit(row, deficits)


def select_original_rows(train_rows, deficits):
    prioritized = [(row_deficit(row, deficits), idx, row) for idx, row in enumerate(train_rows)]
    prioritized.sort(key=lambda item: (-item[0], item[1]))
    selected = [row for score, _, row in prioritized if score > 0][:SELECTED_ORIGINAL]
    selected_ids = {id(row) for row in selected}
    if len(selected) < SELECTED_ORIGINAL:
        fillers = [row for _, _, row in prioritized if id(row) not in selected_ids]
        selected.extend(fillers[: SELECTED_ORIGINAL - len(selected)])
    RNG.shuffle(selected)
    return selected


def make_context_guard_row(row, aug_id):
    tokens, rest = e27.parse_input(row["input"])
    insertions = [(len(tokens), ["This", "additional", "background", "does", "not", "change", "the", "event", "trigger", "."])]
    new_tokens = e27.apply_insertions(tokens, insertions)
    new_gold = e27.shifted_gold(row, insertions)
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out["input"] = e27.rebuild_input(new_tokens, rest)
    out["output"] = json.dumps(new_gold, ensure_ascii=False)
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "e32_augmented": True,
            "e32_augmentation_kind": "context_guard",
            "e32_aug_id": aug_id,
            "e32_trigger_preserving": True,
            "e32_insertions": [{"pos": pos, "tokens": toks} for pos, toks in insertions],
        }
    )
    return out


def make_augmented_row(row, kind, aug_id):
    if kind == "context_guard":
        aug = make_context_guard_row(row, aug_id)
    else:
        aug = e27.make_augmented_row(row, "hard_negative" if kind == "hard_negative" else "role_contrast", aug_id)
        if aug is None:
            return None
        meta = aug.setdefault("meta", {})
        meta.update(
            {
                "e32_augmented": True,
                "e32_augmentation_kind": kind,
                "e32_aug_id": aug_id,
                "e32_trigger_preserving": True,
            }
        )
    meta = aug.setdefault("meta", {})
    meta["e32_tail_types"] = row_types(row)
    return aug


def row_supports_kind(row, kind):
    stats = e27.e21.event_stats(row)
    candidate_types = (row.get("meta") or {}).get("candidate_types") or []
    if kind == "role_contrast":
        return stats["argument_count"] >= 2
    if kind == "hard_negative":
        return len(candidate_types) >= 5
    return True


def build_augmented_pool(train_rows, deficits):
    initial_counts = type_sample_counts(train_rows)
    current_counts = Counter(initial_counts)
    base_candidates = [row for row in train_rows if row_deficit(row, deficits) > 0]
    if not base_candidates:
        base_candidates = [row for row in train_rows if row_types(row)]
    pool = []
    by_kind = {}
    for kind, limit in AUGMENTED_COUNTS.items():
        made = []
        attempts = 0
        while len(made) < limit and attempts < limit * 100:
            attempts += 1
            candidates = [row for row in base_candidates if row_supports_kind(row, kind)]
            candidates.sort(
                key=lambda row: (
                    -row_priority(row, deficits, initial_counts, current_counts),
                    row.get("meta", {}).get("doc_id", ""),
                )
            )
            row = candidates[(attempts - 1) % min(len(candidates), 200)]
            aug = make_augmented_row(row, kind, f"e32_{kind}_{len(made):04d}")
            if aug is None:
                continue
            made.append(aug)
            pool.append(aug)
            for typ in row_types(row):
                current_counts[typ] += 1
        if len(made) < limit:
            raise ValueError(f"only built {len(made)} {kind} rows; need {limit}")
        by_kind[kind] = made
    RNG.shuffle(pool)
    return pool, current_counts


def trigger_role_ground_instruction():
    return (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<EVENT_MENTIONS>{...}</EVENT_MENTIONS>` containing only event_type and trigger spans. "
        "Then output `<REASONING_BUDGET>standard</REASONING_BUDGET>`. "
        "After the budget tag, output `<STEP_REASONING>...</STEP_REASONING>` with compact constraints for each event: "
        "Trigger=\"...\" -> Type=... | Allowed roles=... | Explicit arguments=Role=Text; ... | Do not attach unsupported roles. "
        "Do not include offsets in the reasoning block. "
        "Finally output `<FINAL>{...}</FINAL>` with the complete strict JSON event list including triggers, arguments, and roles. "
        "Do not output text outside the requested tags."
    )


def allowed_roles(row, event_type):
    return SCHEMA_LIBRARY.get(event_type, {}).get("core_roles", []) or []


def trigger_role_ground_reason(row):
    events = e27.e21.gold_json(row).get("events", []) or []
    if not events:
        return "No event: no candidate event is supported by an explicit trigger."
    lines = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        roles = ", ".join(allowed_roles(row, event_type)) or "schema roles only"
        args = []
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict) and arg.get("role") and arg.get("text"):
                args.append(f"{arg.get('role')}={arg.get('text')}")
        arg_text = "; ".join(args) if args else "none"
        lines.append(
            f"Trigger=\"{trigger.get('text') or ''}\" -> Type={event_type} | "
            f"Allowed roles={roles} | Explicit arguments={arg_text} | Do not attach unsupported roles."
        )
    return "\n".join(lines)


def trigger_role_ground_output(row):
    final_payload = e27.e21.gold_json(row)
    mentions = json.dumps(e27.event_mentions_from_payload(final_payload), ensure_ascii=False, separators=(",", ":"))
    final = json.dumps(final_payload, ensure_ascii=False, separators=(",", ":"))
    return "\n".join(
        [
            f"<EVENT_MENTIONS>{mentions}</EVENT_MENTIONS>",
            "<REASONING_BUDGET>standard</REASONING_BUDGET>",
            f"<STEP_REASONING>{trigger_role_ground_reason(row)}</STEP_REASONING>",
            f"<FINAL>{final}</FINAL>",
        ]
    )


def clone_standard(row, role, variant, source_kind):
    spec = SPECS[variant]
    if spec["reason_style"] == "full_natural":
        out = e28.clone_natural(row, role, variant, source_kind)
    elif spec["reason_style"] == "trigger_role_ground":
        out = e27.clone(row, "standard", role, variant, source_kind)
        out["instruction"] = trigger_role_ground_instruction()
        out["output"] = trigger_role_ground_output(row)
    else:
        raise ValueError(f"unknown reason style: {spec['reason_style']}")
    meta = out.setdefault("meta", {})
    meta["adaptive_target_style"] = spec["target_style"]
    meta["adaptive_reasoning_budget"] = "standard"
    meta["adaptive_budget_label"] = "standard"
    meta["e32_variant"] = variant
    meta["e32_branch"] = spec["branch"]
    meta["e32_reason_style"] = spec["reason_style"]
    return out


def clone_none(row, role, variant, source_kind):
    out = e27.clone(row, "none", role, variant, source_kind)
    meta = out.setdefault("meta", {})
    meta["adaptive_target_style"] = SPECS[variant]["target_style"]
    meta["adaptive_reasoning_budget"] = "none"
    meta["adaptive_budget_label"] = "none"
    meta["e32_variant"] = variant
    meta["e32_branch"] = SPECS[variant]["branch"]
    meta["e32_reason_style"] = "none_anchor"
    return out


def clone_for_eval(row, budget, role, variant, source_kind):
    if budget == "standard":
        return clone_standard(row, role, variant, source_kind)
    return clone_none(row, role, variant, source_kind)


def write_variant(variant, selected_original, selected_augmented, audit_base):
    spec = SPECS[variant]
    branch = spec["branch"]
    train = []
    for row in selected_original:
        train.append(clone_standard(row, "train", variant, "selected_original"))
    for row in selected_augmented:
        train.append(clone_standard(row, "train", variant, "selected_augmented"))
    if "none_anchor" in spec["train_budgets"]:
        for row in selected_original:
            train.append(clone_none(row, "train", variant, "selected_original_anchor"))
    RNG.shuffle(train)

    train_name = f"{e27.ADAPTIVE_PREFIX}_{branch}_train_pos"
    e27.write_dataset(train_name, train)

    dev_rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_dev_seen_pos.jsonl")
    dev_name = f"{e27.ADAPTIVE_PREFIX}_{branch}_forced_standard_dev_seen_pos"
    e27.write_dataset(dev_name, [clone_standard(row, "dev_seen", variant, "original") for row in dev_rows])

    eval_names = []
    for split in ["test_seen", "test_unseen"]:
        rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_{split}_pos.jsonl")
        for budget in e27.FORMAL_BUDGETS:
            name = f"{e27.ADAPTIVE_PREFIX}_{branch}_forced_{budget}_{split}_pos"
            e27.write_dataset(name, [clone_for_eval(row, budget, split, variant, "original") for row in rows])
            eval_names.append(name)

    source_counts = Counter(row["meta"].get("e27_source_kind", "unknown") for row in train)
    budget_counts = Counter(row["meta"]["adaptive_reasoning_budget"] for row in train)
    target_tokens = Counter()
    for row in train:
        target_tokens[row["meta"]["adaptive_reasoning_budget"]] += len(row["output"].split())
    aug_kind_counts = Counter((row.get("meta") or {}).get("e32_augmentation_kind") for row in selected_augmented)
    aug_type_counts = Counter()
    for row in selected_augmented:
        for typ in row_types(row):
            aug_type_counts[typ] += 1

    audit = {
        **audit_base,
        "recipe": spec["description"],
        "variant": variant,
        "branch": branch,
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
            "trigger_preserving": True,
        },
        "formal_budgets": e27.FORMAL_BUDGETS,
        "training_recipe": {"learning_rate": 3.0e-6, "num_train_epochs": 3.0},
    }
    config = e27.write_config(variant, train_name, dev_name)
    note = e27.write_note(variant, train_name, dev_name, audit)
    text = note.read_text(encoding="utf-8").replace(
        "python3 scripts/prepare_1_7b_paired_augmentation_e27_20260527.py",
        "python3 scripts/prepare_1_7b_trigger_preserving_e32_20260530.py",
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
    train_rows = e27.e21.e15.load_jsonl(e27.FORMAL_DATA_DIR / f"{e27.DATA_PREFIX}_train_pos.jsonl")
    deficits = deficit_map(train_rows)
    selected_original = select_original_rows(train_rows, deficits)
    selected_augmented, final_counts = build_augmented_pool(train_rows, deficits)
    initial_counts = type_sample_counts(train_rows)
    audit_base = {
        "selection_policy": "tail-type balancing with trigger-preserving end-of-text augmentations",
        "row_budget": {
            "selected_original": SELECTED_ORIGINAL,
            "selected_augmented": sum(AUGMENTED_COUNTS.values()),
            "e32a_e32b_train_rows": SELECTED_ORIGINAL + sum(AUGMENTED_COUNTS.values()),
            "e32c_train_rows": SELECTED_ORIGINAL + sum(AUGMENTED_COUNTS.values()) + SELECTED_ORIGINAL,
        },
        "type_balance": {
            "cap": TYPE_CAP,
            "eligible_type_count": len(deficits),
            "eligible_types": dict(sorted(deficits.items(), key=lambda item: (-item[1], item[0]))),
            "initial_counts": {typ: initial_counts[typ] for typ in sorted(deficits)},
            "post_aug_counts": {typ: final_counts[typ] for typ in sorted(deficits)},
        },
        "planned_augmentation_counts": AUGMENTED_COUNTS,
    }
    payload = [
        write_variant("e32a", selected_original, selected_augmented, audit_base),
        write_variant("e32b", selected_original, selected_augmented, audit_base),
        write_variant("e32c", selected_original, selected_augmented, audit_base),
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
