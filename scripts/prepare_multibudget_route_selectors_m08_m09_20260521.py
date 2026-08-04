#!/usr/bin/env python3
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from src.stage2_cot.build_adaptive_route_reasoning_dataset import audit_rows  # noqa: E402
from src.stage2_data.build_formal_stage2_dataset import update_dataset_info  # noqa: E402


TZ = timezone(timedelta(hours=8))
DATA_DIR = REPO / "data/stage2_adaptive_datasets"
LABEL_DIR = DATA_DIR / "labels"
CONFIG_DIR = REPO / "configs/generated/stage2_adaptive"
EXPERIMENT_DIR = REPO / "experiments"
REPORT_DIR = REPO / "reports"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
RUN_PREFIX = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
ROUTE_ROW_BRANCH = "aet_union_distill_router_m07_routecls_noauxwarm_lr2e6_save50"
TEMPLATE_CONFIG = CONFIG_DIR / f"{RUN_PREFIX}_{ROUTE_ROW_BRANCH}_full_stepmatch.yaml"
WARM_START = (
    "/workspace/project/outputs/stage2_adaptive_runs_user/"
    f"{RUN_PREFIX}_outcome15cal_nlltop15_type_role_hint_plan_lite_noaux_reasonos2_full/checkpoint-1184"
)

DIRECT = {
    "train": REPO / "outputs/stage2_modular_dualexpert/train_teacher_outputs_d1930_r2058_20260517/direct_expert_forced_direct_train/predictions.jsonl",
    "dev_seen": REPO / "outputs/stage2_adaptive_runs_user_devpick_frontier/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux_full_forced_direct_dev_seen_max512/checkpoint-1930/predictions.jsonl",
    "test": REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_balrouteaux_20260516/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux/checkpoint-1930/forced_direct/test/predictions.jsonl",
    "test_seen": REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_balrouteaux_20260516/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux/checkpoint-1930/forced_direct/test_seen/predictions.jsonl",
    "test_unseen": REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_balrouteaux_20260516/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_balrouteaux_reasonos2_from_noaux/checkpoint-1930/forced_direct/test_unseen/predictions.jsonl",
}
LIGHT = {
    "train": REPO / "outputs/stage2_multibudget/light_type_plan_lite_20260521/forced_reason/train/predictions.jsonl",
    "dev_seen": REPO / "outputs/stage2_adaptive_runs_user_devpick_frontier/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10_type_plan_lite_full_forced_reason_dev_seen_max512/checkpoint-1290/predictions.jsonl",
    "test": REPO / "outputs/stage2_adaptive_runs_user_formal_clean/richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_plan_lite/frontier_reason_expert_best/forced_reason/test/predictions.jsonl",
    "test_seen": REPO / "outputs/stage2_adaptive_runs_user_formal_clean/richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_plan_lite/frontier_reason_expert_best/forced_reason/test_seen/predictions.jsonl",
    "test_unseen": REPO / "outputs/stage2_adaptive_runs_user_formal_clean/richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_plan_lite/frontier_reason_expert_best/forced_reason/test_unseen/predictions.jsonl",
}
MID = {
    "train": REPO / "outputs/stage2_adaptive_outcome_mining/l15bal30_ckpt942/forced_reason/train/predictions.jsonl",
    "dev_seen": REPO / "outputs/stage2_adaptive_outcome_mining/l15bal30_ckpt942/forced_reason/dev_seen/predictions.jsonl",
    "test": REPO / "outputs/stage2_adaptive_runs_user_formal_clean/richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30/frontier_seen_stable_best/forced_reason/test/predictions.jsonl",
    "test_seen": REPO / "outputs/stage2_adaptive_runs_user_formal_clean/richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30/frontier_seen_stable_best/forced_reason/test_seen/predictions.jsonl",
    "test_unseen": REPO / "outputs/stage2_adaptive_runs_user_formal_clean/richere_split1_qwen3_1_7b_adaptive_likelihood15_goldplan_type_role_hint_plan_lite_bal30/frontier_seen_stable_best/forced_reason/test_unseen/predictions.jsonl",
}
FULL = {
    "train": REPO / "outputs/stage2_modular_dualexpert/train_teacher_outputs_d1930_r2058_20260517/reason_expert_forced_reason_train/predictions.jsonl",
    "dev_seen": REPO / "outputs/stage2_adaptive_runs_user_devpick_frontier/richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux_full_forced_reason_dev_seen_max512/checkpoint-2058/predictions.jsonl",
    "test": REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux/checkpoint-2058/forced_reason/test/predictions.jsonl",
    "test_seen": REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux/checkpoint-2058/forced_reason/test_seen/predictions.jsonl",
    "test_unseen": REPO / "outputs/stage2_adaptive_runs_user_formal_nll_gated/outcome_helpful_sharedbase_20260515/richere_split1_qwen3_1_7b_adaptive_outcome_l15bal30_15_type_role_hint_plan_lite_routeauxclf1x_pairdirect_reasonos2_from_noaux/checkpoint-2058/forced_reason/test_unseen/predictions.jsonl",
}

EXPERIMENTS = {
    "ternary": {
        "branch": "multibudget_ternary_router_m08_routecls_noauxwarm_lr2e6_save50",
        "label_source": "multibudget_ternary_m08_direct_mid_full",
        "routes": ["direct", "reason_mid", "reason_full"],
        "candidate_paths": {"direct": DIRECT, "reason_mid": MID, "reason_full": FULL},
        "title": "Stage2 Multibudget Ternary Router M08",
        "objective": "Train a route-only selector over direct, medium-budget reason, and full-budget reason actions.",
    },
    "fourclass": {
        "branch": "multibudget_fourclass_router_m09_routecls_noauxwarm_lr2e6_save50",
        "label_source": "multibudget_fourclass_m09_direct_light_mid_full",
        "routes": ["direct", "reason_light", "reason_mid", "reason_full"],
        "candidate_paths": {"direct": DIRECT, "reason_light": LIGHT, "reason_mid": MID, "reason_full": FULL},
        "title": "Stage2 Multibudget Four-Class Router M09",
        "objective": "Train a route-only selector over direct plus light, medium, and full reason-budget actions.",
    },
}

WEIGHTS = {"direct": 1, "reason_light": 5, "reason_mid": 5, "reason_full": 6}
BUDGET_ORDER = {"direct": 0, "reason_light": 1, "reason_mid": 2, "reason_full": 3}


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("wnd_id") or row.get("id")


def prediction_map(path):
    return {key(row): row for row in read_jsonl(path)}


def metric(row, name):
    return float(row.get(name, 0.0) or 0.0)


def valid(row):
    return bool(row.get("valid_final_json", row.get("valid_json", False)))


def utility(row):
    return metric(row, "argument_f1") + metric(row, "event_f1") + 0.25 * metric(row, "trigger_f1")


def route_instruction(routes):
    tags = ", ".join(f"`<ROUTE>{route}</ROUTE>`" for route in routes)
    return (
        "You are doing budgeted route selection for schema-conditioned event extraction. "
        "Use only the provided text, candidate event types, and schema cards. "
        f"Choose exactly one route tag from: {tags}. "
        "Use direct when extraction should be sufficient. Use a higher reason budget only when it is expected "
        "to improve event type, trigger, or argument grounding enough to justify the extra reasoning. "
        "Output exactly one tag and nothing else."
    )


def source_rows(split):
    return read_jsonl(DATA_DIR / f"{DATA_PREFIX}_{ROUTE_ROW_BRANCH}_{split}_pos.jsonl")


def choose_label(candidates, routes, wnd_id):
    direct = candidates["direct"][wnd_id]
    direct_u = utility(direct)
    safe = []
    diagnostics = {}
    for route in routes:
        row = candidates[route][wnd_id]
        gains = {
            "argument_gain": metric(row, "argument_f1") - metric(direct, "argument_f1"),
            "event_gain": metric(row, "event_f1") - metric(direct, "event_f1"),
            "trigger_gain": metric(row, "trigger_f1") - metric(direct, "trigger_f1"),
            "utility_gain": utility(row) - direct_u,
            "valid_json": valid(row),
        }
        diagnostics[route] = gains
        if route == "direct":
            continue
        if (
            gains["valid_json"]
            and gains["argument_gain"] >= 0.0
            and gains["event_gain"] >= 0.0
            and gains["trigger_gain"] >= -0.002
            and max(gains["argument_gain"], gains["event_gain"], gains["trigger_gain"]) >= 0.005
            and gains["utility_gain"] >= 0.005
        ):
            safe.append((gains["utility_gain"], -BUDGET_ORDER[route], route))
    if not safe:
        return "direct", diagnostics
    return max(safe)[2], diagnostics


def build_split(spec, split):
    routes = spec["routes"]
    missing = [p for route in routes for p in [spec["candidate_paths"][route][split]] if not p.exists()]
    if missing:
        raise FileNotFoundError("missing candidate predictions:\n" + "\n".join(p.as_posix() for p in missing))
    candidates = {route: prediction_map(spec["candidate_paths"][route][split]) for route in routes}
    common = set.intersection(*(set(rows) for rows in candidates.values()))
    rows = []
    labels = []
    label_counts = Counter()
    for source in source_rows(split):
        wnd_id = key(source)
        if wnd_id not in common:
            continue
        route, diagnostics = choose_label(candidates, routes, wnd_id)
        label_counts[route] += 1
        label = {
            "wnd_id": wnd_id,
            "route_label": route,
            "source_split": split,
            "label_source": spec["label_source"],
            "label_rule": "choose highest-utility A/E/T-safe budget; otherwise direct",
            "route_diagnostics": diagnostics,
        }
        labels.append(label)
        item = copy.deepcopy(source)
        item["instruction"] = route_instruction(routes)
        item["output"] = f"<ROUTE>{route}</ROUTE>"
        meta = dict(item.get("meta") or {})
        meta["adaptive_route_label"] = route
        meta["multibudget_label_source"] = spec["label_source"]
        item["meta"] = meta
        repeat = WEIGHTS.get(route, 1) if split == "train" else 1
        for dup_idx in range(repeat):
            dup = copy.deepcopy(item)
            dup_meta = dict(dup.get("meta") or {})
            dup_meta["multibudget_weight_repeat"] = repeat
            dup_meta["multibudget_duplicate_index"] = dup_idx
            dup["meta"] = dup_meta
            rows.append(dup)
    dataset_name = f"{DATA_PREFIX}_{spec['branch']}_{split}_pos"
    dataset_path = DATA_DIR / f"{dataset_name}.jsonl"
    meta_path = DATA_DIR / f"{dataset_name}.meta.json"
    label_path = LABEL_DIR / f"{DATA_PREFIX}_{spec['label_source']}_{split}_labels.jsonl"
    summary_path = LABEL_DIR / f"{DATA_PREFIX}_{spec['label_source']}_{split}_labels.summary.json"
    write_jsonl(dataset_path, rows)
    write_jsonl(label_path, labels)
    write_json(
        meta_path,
        {
            "dataset_name": dataset_name,
            "num_examples": len(rows),
            "num_unique_examples": len(labels),
            "routes": routes,
            "candidate_paths": {
                route: spec["candidate_paths"][route][split].as_posix() for route in routes
            },
            "audit": audit_rows(rows),
            "label_counts": dict(label_counts),
            "weights": WEIGHTS,
        },
    )
    write_json(
        summary_path,
        {
            "label_source": spec["label_source"],
            "source_split": split,
            "num_examples": len(labels),
            "weighted_num_examples": len(rows),
            "label_counts": dict(label_counts),
            "label_rates": {k: v / len(labels) for k, v in label_counts.items()} if labels else {},
            "dataset_jsonl": dataset_path.as_posix(),
            "labels_jsonl": label_path.as_posix(),
        },
    )
    update_dataset_info(DATA_DIR, dataset_name, dataset_path.name)
    return read_json(summary_path)


def make_config(spec):
    config = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    config["model_name_or_path"] = WARM_START
    config["dataset"] = f"{DATA_PREFIX}_{spec['branch']}_train_pos"
    config["eval_dataset"] = f"{DATA_PREFIX}_{spec['branch']}_dev_seen_pos"
    config["output_dir"] = f"/workspace/project/outputs/stage2_adaptive_runs_user/{RUN_PREFIX}_{spec['branch']}_full"
    config["learning_rate"] = 2.0e-6
    config["num_train_epochs"] = 1.5
    config["save_steps"] = 50
    config["eval_steps"] = 50
    config["logging_steps"] = 5
    config["save_strategy"] = "steps"
    config["eval_strategy"] = "steps"
    config["load_best_model_at_end"] = False
    out = CONFIG_DIR / f"{RUN_PREFIX}_{spec['branch']}_full_stepmatch.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return out, REPO / config["output_dir"].replace("/workspace/project/", "")


def make_note(kind, spec, summaries, cfg_path, output_dir, status, timestamp, blocked_reason=None):
    exp_id = f"2026-05-21_stage2_{spec['branch']}_richere_split1_oracle_mixed_noise_qwen3_1_7b"
    note = EXPERIMENT_DIR / f"{exp_id}.md"
    note.write_text(
        f"""---
id: {exp_id}
title: {spec['title']}
kind: experiment
status: {status}
created_at: {timestamp}
updated_at: {timestamp}
owners:
  - codex
tags:
  - stage2
  - adaptive-routing
  - multibudget-router
  - route-classification
objective: {spec['objective']}
artifacts:
  configs:
    - {cfg_path.as_posix() if cfg_path else "pending"}
  outputs:
    - {output_dir.as_posix() if output_dir else "pending"}
context:
  dataset: RichERE
  split: split1
  label_source: {spec['label_source']}
  routes: {json.dumps(spec['routes'])}
  warm_start_from_noaux: true
  learning_rate: 2.0e-6
  num_train_epochs: 1.5
---

# {spec['title']}

## Goal

Train a route-only selector that chooses among budgeted extraction actions instead of only direct/reason.

## Setup

- branch: `{spec['branch']}`
- routes: `{', '.join(spec['routes'])}`
- label rule: choose the highest-utility A/E/T-safe non-direct budget; otherwise direct.
- blocked reason: `{blocked_reason or 'none'}`

Label summaries:

```json
{json.dumps(summaries, ensure_ascii=False, indent=2)}
```

## Commands

```bash
cd {REPO}
python3 scripts/prepare_multibudget_route_selectors_m08_m09_20260521.py --only {kind}
bash scripts/launch_modular_dualexpert_utility_router_train_20260517.sh {spec['branch']}=<gpu>
```

## Run Log

### {timestamp.replace('T', ' ')[:16]} +08:00

- prepared multibudget route labels/datasets where all required candidate outputs were available.
- created training config when preparation was complete.

## Result

Pending.

## Conclusion

Pending.

## Next

- train selector.
- score route-choice NLL on dev checkpoints with multiroute labels.
- calibrate dev budget windows and replay formal.
""",
        encoding="utf-8",
    )
    return note


def update_plan(timestamp):
    path = REPO / "PLANS.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Plans\n"
    entry = (
        f"\n\n## {timestamp[:10]} Multibudget Route Selectors\n\n"
        "- Running M08 ternary selector: `direct / reason_mid / reason_full`.\n"
        "- Running M09 four-class selector after light-budget train forced output is available: "
        "`direct / reason_light / reason_mid / reason_full`.\n"
        "- Next: train both selectors, run multiroute NLL scoring, then dev-calibrate budgeted formal replay.\n"
    )
    if "Multibudget Route Selectors" not in text:
        path.write_text(text.rstrip() + entry + "\n", encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["ternary", "fourclass", "all"], default="all")
    args = parser.parse_args()
    timestamp = now_iso()
    selected = ["ternary", "fourclass"] if args.only == "all" else [args.only]
    payload = {}
    for kind in selected:
        spec = EXPERIMENTS[kind]
        summaries = {}
        cfg_path = output_dir = None
        status = "planned"
        blocked_reason = None
        try:
            for split in ["train", "dev_seen", "test", "test_seen", "test_unseen"]:
                summaries[split] = build_split(spec, split)
            cfg_path, output_dir = make_config(spec)
            status = "running"
        except FileNotFoundError as exc:
            status = "blocked"
            blocked_reason = str(exc)
        note = make_note(kind, spec, summaries, cfg_path, output_dir, status, timestamp, blocked_reason)
        payload[kind] = {
            "status": status,
            "blocked_reason": blocked_reason,
            "config": cfg_path.as_posix() if cfg_path else None,
            "output_dir": output_dir.as_posix() if output_dir else None,
            "note": note.as_posix(),
            "summaries": summaries,
        }
    update_plan(timestamp)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
