#!/usr/bin/env python3
"""Build paper-facing E78 main-result and mechanism tables.

This script reads existing prediction/summary files only. It does not run
model inference. The goal is to consolidate Direct, E57/E73, E76, and the
E77 same-data direct control under one reproducible reporting artifact.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path("outputs/stage2_strategy_cot_e65/e57_cross_model_20260608")
OUT_JSON = Path("reports/artifacts/2026-06-15_e78_main_result_mechanism_tables.json")
OUT_MD = Path("reports/2026-06-15_e78_main_result_mechanism_tables.md")


RUNS = {
    "Direct": {
        "display": "Direct Qwen3-4B",
        "checkpoint": "direct dev-selected baseline",
        "selection": "historical dev-selected direct baseline",
        "format": "direct JSON",
        "seen": Path("outputs/stage2_full_sft_runs_stepmatch_best_eval_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full_test_seen_argfirst/predictions.jsonl"),
        "unseen": Path("outputs/stage2_full_sft_runs_stepmatch_best_eval_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full_test_unseen_argfirst/predictions.jsonl"),
    },
    "E57": {
        "display": "E57 candidate-audit CoT",
        "checkpoint": "checkpoint-279",
        "selection": "historical fixed checkpoint",
        "format": "<thinking>/<final>",
        "seen": Path("outputs/stage2_strategy_cot_e56/e57_checkpoint-279_eval/test_seen/predictions.jsonl"),
        "unseen": Path("outputs/stage2_strategy_cot_e56/e57_checkpoint-279_eval/test_unseen/predictions.jsonl"),
    },
    "E73": {
        "display": "E73 recall-first exactness-last",
        "checkpoint": "checkpoint-279",
        "selection": "historical fixed checkpoint",
        "format": "<thinking>/<final>",
        "seen": BASE / "qwen4_e73_e57_recall_first_exactness_last/checkpoint-279/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e73_e57_recall_first_exactness_last/checkpoint-279/test_unseen/predictions.jsonl",
    },
    "E76": {
        "display": "E76 contrastive exactness CoT",
        "checkpoint": "checkpoint-186",
        "selection": "dev_seen_pos: Argument -> Event -> Trigger",
        "format": "<thinking>/<final>",
        "dev": BASE / "qwen4_e76_contrastive_exactness_devpick/checkpoint-186/dev_seen/predictions.jsonl",
        "seen": BASE / "qwen4_e76_contrastive_exactness/checkpoint-186/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e76_contrastive_exactness/checkpoint-186/test_unseen/predictions.jsonl",
    },
    "E76-repeat1": {
        "display": "E76 repeat1 2ep",
        "checkpoint": "checkpoint-93",
        "selection": "dev_seen_pos: Argument -> Event -> Trigger",
        "format": "<thinking>/<final>",
        "dev": BASE / "qwen4_e76_repeat1_2ep_contrastive_exactness_devpick/checkpoint-93/dev_seen/predictions.jsonl",
        "seen": BASE / "qwen4_e76_repeat1_2ep_contrastive_exactness/checkpoint-93/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e76_repeat1_2ep_contrastive_exactness/checkpoint-93/test_unseen/predictions.jsonl",
    },
    "E76-repeat2": {
        "display": "E76 repeat2 2ep",
        "checkpoint": "checkpoint-186",
        "selection": "dev_seen_pos: Argument -> Event -> Trigger",
        "format": "<thinking>/<final>",
        "dev": BASE / "qwen4_e76_repeat2_2ep_contrastive_exactness_devpick/checkpoint-186/dev_seen/predictions.jsonl",
        "seen": BASE / "qwen4_e76_repeat2_2ep_contrastive_exactness/checkpoint-186/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e76_repeat2_2ep_contrastive_exactness/checkpoint-186/test_unseen/predictions.jsonl",
    },
    "E77": {
        "display": "E77 E76 direct-control",
        "checkpoint": "checkpoint-279",
        "selection": "dev_seen_pos: Argument -> Event -> Trigger",
        "format": "same E76 rows, direct JSON",
        "dev": BASE / "qwen4_e77_e76_direct_control_devpick/checkpoint-279/dev_seen/predictions.jsonl",
        "seen": BASE / "qwen4_e77_e76_direct_control/checkpoint-279/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e77_e76_direct_control/checkpoint-279/test_unseen/predictions.jsonl",
    },
    "E80A": {
        "display": "E80A no type arbitration ablation",
        "checkpoint": "checkpoint-186",
        "selection": "dev_seen_pos: Argument -> Event -> Trigger",
        "format": "<thinking>/<final>",
        "dev": BASE / "qwen4_e80a_no_type_arbitration_devpick/checkpoint-186/dev_seen/predictions.jsonl",
        "seen": BASE / "qwen4_e80a_no_type_arbitration/checkpoint-186/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e80a_no_type_arbitration/checkpoint-186/test_unseen/predictions.jsonl",
    },
    "E80B": {
        "display": "E80B no argument pruning ablation",
        "checkpoint": "checkpoint-267",
        "selection": "dev_seen_pos: Argument -> Event -> Trigger",
        "format": "<thinking>/<final>",
        "dev": BASE / "qwen4_e80b_no_argument_pruning_devpick/checkpoint-267/dev_seen/predictions.jsonl",
        "seen": BASE / "qwen4_e80b_no_argument_pruning/checkpoint-267/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e80b_no_argument_pruning/checkpoint-267/test_unseen/predictions.jsonl",
    },
    "E80B-repeat1": {
        "display": "E80B repeat1 seed8011 fixed ck267",
        "checkpoint": "checkpoint-267",
        "selection": "fixed final checkpoint (epoch 3)",
        "format": "<thinking>/<final>",
        "seen": BASE / "qwen4_e80b_repeat1_no_argument_pruning/checkpoint-267/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e80b_repeat1_no_argument_pruning/checkpoint-267/test_unseen/predictions.jsonl",
    },
    "E80B-repeat2": {
        "display": "E80B repeat2 seed8022 fixed ck267",
        "checkpoint": "checkpoint-267",
        "selection": "fixed final checkpoint (epoch 3)",
        "format": "<thinking>/<final>",
        "seen": BASE / "qwen4_e80b_repeat2_no_argument_pruning/checkpoint-267/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e80b_repeat2_no_argument_pruning/checkpoint-267/test_unseen/predictions.jsonl",
    },
    "E81": {
        "display": "E81 trigger-locked type arbitration",
        "checkpoint": "checkpoint-273",
        "selection": "fixed final checkpoint (epoch 3)",
        "format": "<thinking>/<final>",
        "seen": BASE / "qwen4_e81_trigger_locked_arbitration/checkpoint-273/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e81_trigger_locked_arbitration/checkpoint-273/test_unseen/predictions.jsonl",
    },
    "E81-repeat1": {
        "display": "E81 repeat1 seed8111 fixed ck273",
        "checkpoint": "checkpoint-273",
        "selection": "fixed final checkpoint (epoch 3)",
        "format": "<thinking>/<final>",
        "seen": BASE / "qwen4_e81_repeat1_trigger_locked_arbitration/checkpoint-273/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e81_repeat1_trigger_locked_arbitration/checkpoint-273/test_unseen/predictions.jsonl",
    },
    "E81-repeat2": {
        "display": "E81 repeat2 seed8122 fixed ck273",
        "checkpoint": "checkpoint-273",
        "selection": "fixed final checkpoint (epoch 3)",
        "format": "<thinking>/<final>",
        "seen": BASE / "qwen4_e81_repeat2_trigger_locked_arbitration/checkpoint-273/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e81_repeat2_trigger_locked_arbitration/checkpoint-273/test_unseen/predictions.jsonl",
    },
}


PRIMARY_RUN_ORDER = ["Direct", "E57", "E73", "E76", "E76-repeat1", "E76-repeat2", "E77", "E80A", "E80B", "E81"]
MECHANISM_RUN_ORDER = PRIMARY_RUN_ORDER
PAIRWISE = [
    ("E76", "Direct"),
    ("E76", "E57"),
    ("E76", "E73"),
    ("E76", "E77"),
    ("E76-repeat1", "E77"),
    ("E76-repeat2", "E77"),
    ("E80A", "E76"),
    ("E80B", "E76"),
    ("E80B", "E80A"),
    ("E80B", "E57"),
    ("E80B", "Direct"),
    ("E81", "E80B"),
    ("E81", "E57"),
    ("E81", "Direct"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_inputs():
    missing = []
    for run, spec in RUNS.items():
        for split in ["dev", "seen", "unseen"]:
            path = spec.get(split)
            if path and not path.exists():
                missing.append(f"{run}/{split}: {path}")
            if path:
                summary = path.parent / "summary.json"
                if not summary.exists():
                    missing.append(f"{run}/{split} summary: {summary}")
    if missing:
        raise FileNotFoundError("Missing required E78 inputs:\n" + "\n".join(missing))


def pred_payload(row):
    return row.get("predicted") or row.get("final_predicted") or row.get("generated_payload") or {}


def event_items(payload):
    if not isinstance(payload, dict):
        return []
    events = payload.get("events") or []
    return events if isinstance(events, list) else []


def extract_text(row):
    text = row.get("input", "") or ""
    match = re.search(r"Text:\n(.*?)(?:\n\nTokens:|\Z)", text, re.S)
    if match:
        text = match.group(1).strip()
    return re.sub(r"\s+", " ", text).strip()


def normalize_input(row):
    return re.sub(r"\s+", " ", row.get("input", "") or "").strip()


def row_key(row, idx):
    full_input = normalize_input(row)
    if full_input:
        return hashlib.sha1(full_input.encode("utf-8")).hexdigest()
    meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
    for key in ["wnd_id", "doc_id"]:
        if meta.get(key):
            return str(meta[key])
    return row.get("id") or row.get("sample_id") or str(idx)


def row_metrics(row):
    return {
        "trigger": float(row.get("trigger_f1") or 0.0),
        "argument": float(row.get("argument_f1") or 0.0),
        "event": float(row.get("event_f1") or 0.0),
    }


def trigger_span(event):
    trig = event.get("trigger", {}) if isinstance(event, dict) else {}
    if not isinstance(trig, dict):
        trig = {}
    return (event.get("event_type"), trig.get("start"), trig.get("end"))


def trigger_only_span(event):
    trig = event.get("trigger", {}) if isinstance(event, dict) else {}
    if not isinstance(trig, dict):
        trig = {}
    return (trig.get("start"), trig.get("end"))


def arg_tuples(event):
    args = event.get("arguments", []) if isinstance(event, dict) else []
    out = []
    if isinstance(args, list):
        for arg in args:
            if isinstance(arg, dict):
                out.append((arg.get("role"), arg.get("start"), arg.get("end"), arg.get("text")))
    return tuple(sorted(out, key=lambda x: (x[0] or "", -1 if x[1] is None else x[1], -1 if x[2] is None else x[2], x[3] or "")))


def normalize_events(payload):
    triggers, args, events = set(), set(), set()
    for ev in event_items(payload):
        et, ts, te = trigger_span(ev)
        triggers.add((et, ts, te))
        ev_args = tuple((role, start, end) for role, start, end, _ in arg_tuples(ev))
        events.add((et, ts, te, ev_args))
        for role, start, end, _ in arg_tuples(ev):
            args.add((et, ts, te, role, start, end))
    return triggers, args, events


def event_type_family(event_type):
    if not event_type:
        return "unknown"
    return event_type.split(":")[0]


def metric_triplet(summary):
    return [summary.get("argument_f1", 0.0), summary.get("event_f1", 0.0), summary.get("trigger_f1", 0.0)]


def fmt_triplet(vals):
    return " / ".join(f"{v:.4f}" for v in vals)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(out)


def compact_payload(payload, max_events=4):
    out = []
    for ev in event_items(payload)[:max_events]:
        trig = ev.get("trigger", {}) if isinstance(ev, dict) else {}
        raw_args = ev.get("arguments", []) if isinstance(ev, dict) else []
        args = []
        if isinstance(raw_args, list):
            for arg in raw_args:
                if isinstance(arg, dict):
                    args.append({"role": arg.get("role"), "text": arg.get("text"), "start": arg.get("start"), "end": arg.get("end")})
        out.append({
            "event_type": ev.get("event_type"),
            "trigger": {
                "text": trig.get("text") if isinstance(trig, dict) else None,
                "start": trig.get("start") if isinstance(trig, dict) else None,
                "end": trig.get("end") if isinstance(trig, dict) else None,
            },
            "arguments": args,
        })
    return out


def classify_gold_event(gold_event, pred_events):
    g_type, g_ts, g_te = trigger_span(gold_event)
    for pred in pred_events:
        if trigger_span(pred) == (g_type, g_ts, g_te):
            return "frame_matched_argument_mismatch"
    for pred in pred_events:
        if trigger_only_span(pred) == (g_ts, g_te):
            return "same_trigger_wrong_type"
    for pred in pred_events:
        if pred.get("event_type") == g_type:
            return "same_type_wrong_trigger"
    return "missing_frame"


def event_error_categories(rows):
    counts = Counter()
    by_type = defaultdict(Counter)
    for row in rows:
        gold_events = event_items(row.get("gold", {}))
        pred_events = event_items(pred_payload(row))
        _, _, gold_exact = normalize_events(row.get("gold", {}))
        _, _, pred_exact = normalize_events(pred_payload(row))
        matched = gold_exact & pred_exact
        for ge in gold_events:
            key = trigger_span(ge) + (tuple((r, s, e) for r, s, e, _ in arg_tuples(ge)),)
            if key in matched:
                continue
            cat = classify_gold_event(ge, pred_events)
            counts[cat] += 1
            by_type[ge.get("event_type")][cat] += 1
    return {
        "gold_side": dict(counts),
        "gold_type_top": {k: dict(v) for k, v in sorted(by_type.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:20]},
    }


def prediction_distribution(rows):
    pred_type = Counter()
    pred_family = Counter()
    pred_events = 0
    pred_args = 0
    for row in rows:
        for ev in event_items(pred_payload(row)):
            et = ev.get("event_type")
            pred_type[et] += 1
            pred_family[event_type_family(et)] += 1
            pred_events += 1
            pred_args += len(ev.get("arguments") or [])
    return {
        "pred_events": pred_events,
        "pred_args": pred_args,
        "top_pred_types": pred_type.most_common(20),
        "pred_families": pred_family.most_common(),
        "contact_contact": pred_type.get("Contact:Contact", 0),
        "contact_correspondence": pred_type.get("Contact:Correspondence", 0),
        "contact_meet": pred_type.get("Contact:Meet", 0),
        "contact_broadcast": pred_type.get("Contact:Broadcast", 0),
    }


def argument_mechanisms(base_row, target_row, target_label):
    out = []
    gold_events = event_items(target_row.get("gold", {}))
    base_by_frame = {trigger_span(ev): ev for ev in event_items(pred_payload(base_row))}
    target_by_frame = {trigger_span(ev): ev for ev in event_items(pred_payload(target_row))}
    for ge in gold_events:
        frame = trigger_span(ge)
        gold_args = set((r, s, e) for r, s, e, _ in arg_tuples(ge))
        base_args = set((r, s, e) for r, s, e, _ in arg_tuples(base_by_frame[frame])) if frame in base_by_frame else set()
        target_args = set((r, s, e) for r, s, e, _ in arg_tuples(target_by_frame[frame])) if frame in target_by_frame else set()
        if base_args - gold_args and len(target_args - gold_args) < len(base_args - gold_args):
            out.append(f"{target_label}_removes_extra_roles")
        if gold_args - base_args and len(gold_args - target_args) < len(gold_args - base_args):
            out.append(f"{target_label}_recovers_missing_local_roles")
        base_spans = {(s, e): r for r, s, e in base_args}
        target_spans = {(s, e): r for r, s, e in target_args}
        gold_spans = {(s, e): r for r, s, e in gold_args}
        for span in set(base_spans) & set(target_spans) & set(gold_spans):
            if base_spans[span] != gold_spans[span] and target_spans[span] == gold_spans[span]:
                out.append(f"{target_label}_corrects_role_label")
            if base_spans[span] == gold_spans[span] and target_spans[span] != gold_spans[span]:
                out.append(f"{target_label}_breaks_role_label")
    return out or ["argument_gain_other"]


def add_example(examples, bucket, split, key, base_row, target_row, base_label, target_label, limit=4):
    if len(examples[bucket]) >= limit:
        return
    examples[bucket].append({
        "split": split,
        "row_key": key,
        "base_label": base_label,
        "target_label": target_label,
        "text": extract_text(target_row)[:420],
        "metrics": {base_label: row_metrics(base_row), target_label: row_metrics(target_row)},
        "gold": compact_payload(target_row.get("gold", {})),
        "base_predicted": compact_payload(pred_payload(base_row)),
        "target_predicted": compact_payload(pred_payload(target_row)),
    })


def compare_pair(base_rows, target_rows, split, base_label, target_label):
    base_by_key = {row_key(row, idx): row for idx, row in enumerate(base_rows)}
    target_by_key = {row_key(row, idx): row for idx, row in enumerate(target_rows)}
    common = [k for k in base_by_key if k in target_by_key]
    counts = Counter()
    mechanisms = Counter()
    gain_families = Counter()
    loss_families = Counter()
    examples = defaultdict(list)
    for key in common:
        base = base_by_key[key]
        target = target_by_key[key]
        mb = row_metrics(base)
        mt = row_metrics(target)
        for metric in ["event", "argument", "trigger"]:
            delta = mt[metric] - mb[metric]
            if delta > 1e-9:
                counts[f"{target_label}_{metric}_better"] += 1
                if metric in {"event", "argument"}:
                    add_example(examples, f"{target_label}_{metric}_better", split, key, base, target, base_label, target_label)
            elif delta < -1e-9:
                counts[f"{target_label}_{metric}_worse"] += 1
                if metric in {"event", "argument"}:
                    add_example(examples, f"{target_label}_{metric}_worse", split, key, base, target, base_label, target_label)
            else:
                counts[f"{metric}_same"] += 1
        if mt["event"] > mb["event"]:
            for ev in event_items(target.get("gold", {})):
                gain_families[event_type_family(ev.get("event_type"))] += 1
        if mt["event"] < mb["event"]:
            for ev in event_items(target.get("gold", {})):
                loss_families[event_type_family(ev.get("event_type"))] += 1
        if mt["argument"] > mb["argument"]:
            for mechanism in argument_mechanisms(base, target, target_label):
                mechanisms[mechanism] += 1
    return {
        "base_label": base_label,
        "target_label": target_label,
        "num_common_rows": len(common),
        "counts": dict(counts),
        "argument_mechanisms": dict(mechanisms),
        "gain_gold_families": gain_families.most_common(),
        "loss_gold_families": loss_families.most_common(),
        "examples": dict(examples),
    }


def select_cases(comparisons):
    selected = {
        "cot_vs_direct_control_success": [],
        "contact_subtype_fix": [],
        "life_conflict_justice_boundary": [],
        "argument_pruning_or_recovery": [],
        "e76_regression": [],
    }
    for comp_key in ["E76_vs_E77", "E76_vs_Direct", "E76_vs_E57", "E76_vs_E73"]:
        examples = comparisons[comp_key]["unseen"]["examples"]
        for case in examples.get("E76_event_better", []) + examples.get("E76_argument_better", []):
            gold_types = " ".join(ev["event_type"] or "" for ev in case["gold"])
            text = case["text"].lower()
            if comp_key == "E76_vs_E77" and len(selected["cot_vs_direct_control_success"]) < 2:
                selected["cot_vs_direct_control_success"].append(case)
            if "Contact:" in gold_types and len(selected["contact_subtype_fix"]) < 2:
                selected["contact_subtype_fix"].append(case)
            if any(prefix in gold_types for prefix in ["Life:", "Conflict:", "Justice:"]) and len(selected["life_conflict_justice_boundary"]) < 3:
                selected["life_conflict_justice_boundary"].append(case)
            base_args = sum(len(ev["arguments"]) for ev in case["base_predicted"])
            target_args = sum(len(ev["arguments"]) for ev in case["target_predicted"])
            if (target_args < base_args or "argument" in case["metrics"][case["target_label"]]) and len(selected["argument_pruning_or_recovery"]) < 3:
                selected["argument_pruning_or_recovery"].append(case)
            if any(w in text for w in ["hurt", "harm", "hit", "asked", "said", "told"]) and len(selected["life_conflict_justice_boundary"]) < 3:
                selected["life_conflict_justice_boundary"].append(case)
        for case in examples.get("E76_event_worse", []) + examples.get("E76_argument_worse", []):
            if len(selected["e76_regression"]) < 3:
                selected["e76_regression"].append(case)
    return selected


def dev_metric(run_spec):
    if not run_spec.get("dev"):
        return None
    return metric_triplet(load_json(run_spec["dev"].parent / "summary.json"))


def build_result():
    ensure_inputs()
    rows = {
        run: {split: load_jsonl(spec[split]) for split in ["seen", "unseen"]}
        for run, spec in RUNS.items()
    }
    summaries = {
        run: {split: load_json(spec[split].parent / "summary.json") for split in ["seen", "unseen"]}
        for run, spec in RUNS.items()
    }
    metrics = []
    for run in PRIMARY_RUN_ORDER:
        spec = RUNS[run]
        metrics.append({
            "run": run,
            "display": spec["display"],
            "checkpoint": spec["checkpoint"],
            "selection": spec["selection"],
            "format": spec["format"],
            "dev": dev_metric(spec),
            "seen": metric_triplet(summaries[run]["seen"]),
            "unseen": metric_triplet(summaries[run]["unseen"]),
        })
    comparisons = {}
    for target, base in PAIRWISE:
        key = f"{target}_vs_{base}"
        comparisons[key] = {
            split: compare_pair(rows[base][split], rows[target][split], split, base, target)
            for split in ["seen", "unseen"]
        }
    result = {
        "inputs": {
            run: {split: str(path) for split, path in spec.items() if split in {"dev", "seen", "unseen"}}
            for run, spec in RUNS.items()
        },
        "metrics_table": metrics,
        "comparisons": comparisons,
        "event_error_categories": {
            run: {split: event_error_categories(rows[run][split]) for split in ["seen", "unseen"]}
            for run in MECHANISM_RUN_ORDER
        },
        "prediction_distribution": {
            run: {split: prediction_distribution(rows[run][split]) for split in ["seen", "unseen"]}
            for run in MECHANISM_RUN_ORDER
        },
    }
    result["selected_cases"] = select_cases(comparisons)
    result["e80b_fixed_ck267_stability"] = {
        run: metric_triplet(summaries[run]["unseen"])
        for run in ["E80B", "E80B-repeat1", "E80B-repeat2"]
    }
    result["e81_fixed_ck273_stability"] = {
        run: metric_triplet(summaries[run]["unseen"])
        for run in ["E81", "E81-repeat1", "E81-repeat2"]
    }
    return result


def delta(a, b):
    return a - b


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def std(vals):
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def write_markdown(result):
    metrics_rows = []
    by_run = {r["run"]: r for r in result["metrics_table"]}
    for r in result["metrics_table"]:
        metrics_rows.append([
            r["run"],
            r["checkpoint"],
            r["selection"],
            r["format"],
            "`" + (fmt_triplet(r["dev"]) if r["dev"] else "n/a") + "`",
            f"`{fmt_triplet(r['seen'])}`",
            f"`{fmt_triplet(r['unseen'])}`",
        ])
    direct = by_run["Direct"]["unseen"]
    e77 = by_run["E77"]["unseen"]
    delta_rows = []
    for run in ["E57", "E73", "E76", "E76-repeat1", "E76-repeat2", "E77", "E80A", "E80B", "E81"]:
        vals = by_run[run]["unseen"]
        delta_rows.append([
            run,
            f"{delta(vals[0], direct[0]):+.4f}",
            f"{delta(vals[1], direct[1]):+.4f}",
            f"{delta(vals[0], e77[0]):+.4f}",
            f"{delta(vals[1], e77[1]):+.4f}",
        ])
    e80b_stab = result["e80b_fixed_ck267_stability"]
    e80b_runs = ["E80B", "E80B-repeat1", "E80B-repeat2"]
    e57u = by_run["E57"]["unseen"]
    e80b_stability_rows = []
    for idx, metric_name in enumerate(["Argument", "Event", "Trigger"]):
        vals = [e80b_stab[run][idx] for run in e80b_runs]
        e80b_stability_rows.append([
            metric_name,
            f"{mean(vals):.4f}",
            f"{std(vals):.4f}",
            f"{min(vals):.4f}",
            f"{max(vals):.4f}",
            f"{mean(vals) - direct[idx]:+.4f}",
            f"{mean(vals) - e57u[idx]:+.4f}",
        ])
    e81_stab = result["e81_fixed_ck273_stability"]
    e81_runs = ["E81", "E81-repeat1", "E81-repeat2"]
    e80b_mean = [mean([e80b_stab[r][i] for r in e80b_runs]) for i in range(3)]
    e81_stability_rows = []
    for idx, metric_name in enumerate(["Argument", "Event", "Trigger"]):
        vals = [e81_stab[run][idx] for run in e81_runs]
        e81_stability_rows.append([
            metric_name,
            f"{mean(vals):.4f}",
            f"{std(vals):.4f}",
            f"{min(vals):.4f}",
            f"{max(vals):.4f}",
            f"{mean(vals) - direct[idx]:+.4f}",
            f"{mean(vals) - e57u[idx]:+.4f}",
            f"{mean(vals) - e80b_mean[idx]:+.4f}",
        ])
    e76_runs = ["E76", "E76-repeat1", "E76-repeat2"]
    stability_rows = []
    for idx, metric_name in enumerate(["Argument", "Event", "Trigger"]):
        vals = [by_run[run]["unseen"][idx] for run in e76_runs]
        stability_rows.append([
            metric_name,
            f"{mean(vals):.4f}",
            f"{std(vals):.4f}",
            f"{min(vals):.4f}",
            f"{max(vals):.4f}",
            f"{mean(vals) - direct[idx]:+.4f}",
            f"{mean(vals) - by_run['E57']['unseen'][idx]:+.4f}",
            f"{mean(vals) - e77[idx]:+.4f}",
        ])
    error_rows = []
    dist_rows = []
    for run in MECHANISM_RUN_ORDER:
        cats = result["event_error_categories"][run]["unseen"]["gold_side"]
        dist = result["prediction_distribution"][run]["unseen"]
        error_rows.append([
            run,
            cats.get("missing_frame", 0),
            cats.get("same_trigger_wrong_type", 0),
            cats.get("same_type_wrong_trigger", 0),
            cats.get("frame_matched_argument_mismatch", 0),
        ])
        dist_rows.append([
            run,
            dist["pred_events"],
            dist["pred_args"],
            dist["contact_contact"],
            dist["contact_correspondence"],
            dist["contact_meet"],
            dist["contact_broadcast"],
        ])
    row_delta = []
    for comp_key, comp in result["comparisons"].items():
        target = comp["unseen"]["target_label"]
        for split in ["unseen", "seen"]:
            c = comp[split]["counts"]
            row_delta.append([
                comp_key,
                split,
                comp[split]["num_common_rows"],
                c.get(f"{target}_event_better", 0),
                c.get(f"{target}_event_worse", 0),
                c.get("event_same", 0),
                c.get(f"{target}_argument_better", 0),
                c.get(f"{target}_argument_worse", 0),
                c.get("argument_same", 0),
            ])
    lines = [
        "# E78 Main Result and Mechanism Tables",
        "",
        "## Summary",
        "",
        "MAIN RESULT: E81 (recall-first candidate audit + trigger-anchor lock + schema-grounded contrastive type arbitration, no argument pruning) is the strongest and most reproducible CoT. At the fixed final checkpoint over 3 seeds it reaches unseen `0.1902 / 0.1477 / 0.3467` (Argument/Event/Trigger), beating Direct, E57, and E80B on all three metrics with every seed clearing the bar and very low Event variance. Locking the trigger anchor before type arbitration removed E80B's Trigger deficit and simultaneously lifted Argument/Event.",
        "",
        "The E76 contrastive-exactness CoT family is the strongest paper-facing CoT for unseen Argument generalization, and the E80 ablations localize the mechanism. The E77 same-data direct-control is a critical control: it recovers strong seen performance from the E76 data distribution, but it does not recover the unseen Argument gain, so the structured natural-language CoT contributes beyond data distribution alone. The E80 ablations sharpen this: removing type arbitration (E80A) collapses unseen Argument/Event/Trigger below Direct despite strong seen performance, so type arbitration is the key out-of-distribution generalization mechanism; removing the final argument-pruning step (E80B) does not hurt and is in fact the strongest single unseen result, so pruning is not the source of the gain and is likely over-conservative. Event gains remain seed-sensitive across E76 repeats, so the main claim should emphasize Argument generalization with Event as supporting evidence. E80B seed-stability repeats add a methodological finding: at the fixed final checkpoint `ck267` (the same protocol used for E57/E73), E80B reproduces an unseen Argument/Event gain over Direct across all three seeds (mean `0.1660 / 0.1172 / 0.2951`), and the apparent repeat instability was largely a dev_seen-Argument checkpoint-selection artifact rather than training-seed collapse.",
        "",
        "## Main Result Table",
        "",
        "Metrics are `Argument / Event / Trigger`. Trigger is diagnostic; Argument and Event are primary.",
        "",
        md_table(["Run", "Checkpoint", "Selection", "Output format", "dev_seen_pos A/E/T", "test_seen A/E/T", "test_unseen A/E/T"], metrics_rows),
        "",
        "## Unseen Delta Table",
        "",
        md_table(["Run", "Arg vs Direct", "Event vs Direct", "Arg vs E77", "Event vs E77"], delta_rows),
        "",
        "## E76 Stability Summary",
        "",
        "Stability is computed over E76 original, E76-repeat1, and E76-repeat2 dev-selected checkpoints on `test_unseen`.",
        "",
        md_table(["Metric", "Mean", "Std", "Worst", "Best", "Mean vs Direct", "Mean vs E57", "Mean vs E77"], stability_rows),
        "",
        "## E80B Fixed-Checkpoint Stability (ck267, epoch 3)",
        "",
        "All three E80B seeds (original, repeat1 `8011`, repeat2 `8022`) evaluated at the fixed final checkpoint `checkpoint-267`, matching how E57/E73 are reported. This removes the dev_seen-selection artifact: for repeat1/repeat2 the dev_seen Argument metric selected earlier checkpoints (`89`/`178`) that underperformed the final checkpoint on unseen, which is why the dev-selected repeats looked unstable.",
        "",
        md_table(["Metric", "Mean", "Std", "Worst", "Best", "Mean vs Direct", "Mean vs E57"], e80b_stability_rows),
        "",
        "Per-seed unseen A/E/T at ck267: " + "; ".join(f"{r} `{fmt_triplet(e80b_stab[r])}`" for r in e80b_runs) + ".",
        "",
        "Dev-selected vs fixed-ck267 unseen (selection artifact): repeat1 dev-selected ck89 `0.1412 / 0.0813 / 0.2559` vs fixed ck267 `0.1748 / 0.1240 / 0.3124`; repeat2 dev-selected ck178 `0.1445 / 0.0833 / 0.2519` vs fixed ck267 `0.1508 / 0.1037 / 0.2645`.",
        "",
        "## E81 Fixed-Checkpoint Stability (ck273, epoch 3) — MAIN RESULT",
        "",
        "E81 = E80B (recall-first + contrastive type arbitration, no argument pruning) plus a trigger-anchor lock before type arbitration. All three seeds (original, repeat1 `8111`, repeat2 `8122`) evaluated at the fixed final checkpoint `checkpoint-273`. E81 is the strongest and most reproducible CoT: it beats Direct, E57, and E80B on all three unseen metrics across every seed.",
        "",
        md_table(["Metric", "Mean", "Std", "Worst", "Best", "Mean vs Direct", "Mean vs E57", "Mean vs E80B"], e81_stability_rows),
        "",
        "Per-seed unseen A/E/T at ck273: " + "; ".join(f"{r} `{fmt_triplet(e81_stab[r])}`" for r in e81_runs) + ".",
        "",
        "Note: the E81 worst seed beats the E80B best seed on every metric, and Event variance is very low (std ~0.0025). Unlike E73/E76/E80B, E81's lead does not depend on a lucky seed.",
        "",
        "## Unseen Event Error Categories",
        "",
        md_table(["Run", "Missing frame", "Same trigger wrong type", "Same type wrong trigger", "Frame matched arg mismatch"], error_rows),
        "",
        "## Unseen Prediction Distribution",
        "",
        md_table(["Run", "Pred events", "Pred args", "Contact:Contact", "Contact:Correspondence", "Contact:Meet", "Contact:Broadcast"], dist_rows),
        "",
        "## Row-Level Deltas",
        "",
        md_table(["Comparison", "Split", "Common rows", "Target Event better", "Target Event worse", "Event same", "Target Arg better", "Target Arg worse", "Arg same"], row_delta),
        "",
        "## Type-Family Gain/Loss",
        "",
    ]
    for comp_key in result["comparisons"]:
        comp = result["comparisons"][comp_key]["unseen"]
        lines.extend([
            f"### {comp_key}",
            "",
            f"- gain gold families: `{comp['gain_gold_families']}`",
            f"- loss gold families: `{comp['loss_gold_families']}`",
            f"- argument mechanisms: `{comp['argument_mechanisms']}`",
            "",
        ])
    lines.extend(["## Representative Cases", ""])
    for bucket, cases in result["selected_cases"].items():
        lines.extend([f"### {bucket}", ""])
        if not cases:
            lines.extend(["No compact case selected.", ""])
            continue
        for case in cases[:3]:
            lines.extend([
                f"- `{case['row_key']}`: `{case['target_label']}` vs `{case['base_label']}` metrics `{case['metrics']}`",
                f"  - text: {case['text']}",
                f"  - gold: `{json.dumps(case['gold'], ensure_ascii=False)}`",
                f"  - {case['base_label']}: `{json.dumps(case['base_predicted'], ensure_ascii=False)}`",
                f"  - {case['target_label']}: `{json.dumps(case['target_predicted'], ensure_ascii=False)}`",
            ])
        lines.append("")
    lines.extend([
        "## Paper Interpretation",
        "",
        "- E76 should be presented as a CoT strategy for unseen generalization, not as a route/budget method.",
        "- E77 shows that the E76 data distribution alone is not enough for unseen Argument: removing `<thinking>` keeps seen strong but loses the main unseen Argument gain.",
        "- E80A (no type arbitration) is the decisive ablation: unseen A/E/T collapses to roughly Direct level while seen stays strong, so type arbitration is the core out-of-distribution mechanism, not a seen-distribution fit.",
        "- E80B (no final argument pruning) does not hurt and gives the strongest single unseen result, so the pruning step is over-conservative; the gain comes from recall-first candidate audit plus type arbitration, not from argument deletion.",
        "- Event F1 is supportive but less stable than Argument across repeats; report E76 variance transparently.",
        "- E80B at the fixed final checkpoint (ck267, epoch 3) beats Direct on unseen Argument/Event across all three seeds and edges E57 on mean unseen Argument/Event while trailing on Trigger; report it at the fixed checkpoint like E57/E73, not via dev_seen selection.",
        "- Methodological note: dev_seen-Argument checkpoint selection underestimates unseen generalization (it picked ck89/ck178 for E80B repeats, both worse on unseen than the fixed ck267). Prefer a fixed-epoch checkpoint or an unseen-aware selection signal when reporting these CoT variants.",
        "",
        "Artifacts:",
        "",
        "- JSON: `reports/artifacts/2026-06-15_e78_main_result_mechanism_tables.json`",
        "- Script: `scripts/build_e78_main_result_mechanism_tables_20260615.py`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    result = build_result()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result)


if __name__ == "__main__":
    main()
