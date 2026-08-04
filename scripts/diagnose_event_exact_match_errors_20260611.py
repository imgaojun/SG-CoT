#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def event_items(payload):
    if not isinstance(payload, dict):
        return []
    events = payload.get("events", [])
    return events if isinstance(events, list) else []


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


def argument_tuples(event):
    trig = event.get("trigger", {}) if isinstance(event, dict) else {}
    if not isinstance(trig, dict):
        trig = {}
    out = []
    args = event.get("arguments", [])
    if not isinstance(args, list):
        args = []
    for arg in args:
        if not isinstance(arg, dict):
            continue
        out.append((arg.get("role"), arg.get("start"), arg.get("end")))
    return tuple(sorted(out, key=lambda x: (x[0] or "", -1 if x[1] is None else x[1], -1 if x[2] is None else x[2])))


def normalize(payload):
    triggers = set()
    arguments = set()
    events = set()
    frames = set()
    for ev in event_items(payload):
        if not isinstance(ev, dict):
            continue
        et, ts, te = trigger_span(ev)
        triggers.add((et, ts, te))
        frames.add((et, ts, te))
        args = argument_tuples(ev)
        events.add((et, ts, te, args))
        for role, start, end in args:
            arguments.add((et, ts, te, role, start, end))
    return triggers, arguments, events, frames


def prf_counts(pred, gold):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1}


def aggregate(rows):
    pred_trig = set()
    gold_trig = set()
    pred_arg = set()
    gold_arg = set()
    pred_evt = set()
    gold_evt = set()
    for idx, row in enumerate(rows):
        gt, ga, ge, _ = normalize(row.get("gold", {}))
        pt, pa, pe, _ = normalize(row.get("predicted", row.get("final_predicted", {})))
        gold_trig |= {(idx,) + x for x in gt}
        pred_trig |= {(idx,) + x for x in pt}
        gold_arg |= {(idx,) + x for x in ga}
        pred_arg |= {(idx,) + x for x in pa}
        gold_evt |= {(idx,) + x for x in ge}
        pred_evt |= {(idx,) + x for x in pe}
    return {
        "trigger": prf_counts(pred_trig, gold_trig),
        "argument": prf_counts(pred_arg, gold_arg),
        "event": prf_counts(pred_evt, gold_evt),
    }


def classify_gold_event(gold_event, pred_events):
    g_type, g_ts, g_te = trigger_span(gold_event)
    g_args = set(argument_tuples(gold_event))
    for pred in pred_events:
        if trigger_span(pred) == (g_type, g_ts, g_te):
            p_args = set(argument_tuples(pred))
            missing = sorted(g_args - p_args)
            extra = sorted(p_args - g_args)
            return "frame_match_arg_mismatch", missing, extra
    for pred in pred_events:
        if trigger_only_span(pred) == (g_ts, g_te):
            return "same_trigger_wrong_type", [], []
    for pred in pred_events:
        if pred.get("event_type") == g_type:
            return "same_type_wrong_trigger", [], []
    return "missing_frame", [], []


def classify_pred_event(pred_event, gold_events):
    p_type, p_ts, p_te = trigger_span(pred_event)
    p_args = set(argument_tuples(pred_event))
    for gold in gold_events:
        if trigger_span(gold) == (p_type, p_ts, p_te):
            g_args = set(argument_tuples(gold))
            missing = sorted(g_args - p_args)
            extra = sorted(p_args - g_args)
            return "frame_match_arg_mismatch", missing, extra
    for gold in gold_events:
        if trigger_only_span(gold) == (p_ts, p_te):
            return "same_trigger_wrong_type", [], []
    for gold in gold_events:
        if gold.get("event_type") == p_type:
            return "same_type_wrong_trigger", [], []
    return "extra_frame", [], []


def diagnose(rows):
    gold_error = Counter()
    pred_error = Counter()
    missing_arg_roles = Counter()
    extra_arg_roles = Counter()
    gold_type_errors = defaultdict(Counter)
    pred_type_errors = defaultdict(Counter)
    row_buckets = Counter()
    examples = defaultdict(list)

    for idx, row in enumerate(rows):
        gold_events = event_items(row.get("gold", {}))
        pred_events = event_items(row.get("predicted", row.get("final_predicted", {})))
        _, _, gold_event_set, _ = normalize(row.get("gold", {}))
        _, _, pred_event_set, _ = normalize(row.get("predicted", row.get("final_predicted", {})))

        matched = gold_event_set & pred_event_set
        if matched:
            row_buckets["has_exact_event_match"] += 1
        elif gold_events and pred_events:
            row_buckets["gold_and_pred_but_no_exact_event"] += 1
        elif gold_events:
            row_buckets["gold_only"] += 1
        elif pred_events:
            row_buckets["pred_only"] += 1
        else:
            row_buckets["empty_both"] += 1

        for ge in gold_events:
            key = (ge.get("event_type"),) + trigger_span(ge)[1:] + (argument_tuples(ge),)
            if key in gold_event_set & pred_event_set:
                continue
            cat, missing, extra = classify_gold_event(ge, pred_events)
            gold_error[cat] += 1
            gold_type_errors[ge.get("event_type")][cat] += 1
            for role, _, _ in missing:
                missing_arg_roles[role] += 1
            for role, _, _ in extra:
                extra_arg_roles[role] += 1
            if len(examples[cat]) < 3:
                examples[cat].append(
                    {
                        "row": idx,
                        "gold": ge,
                        "predicted": pred_events[:3],
                        "trigger_f1": row.get("trigger_f1"),
                        "argument_f1": row.get("argument_f1"),
                        "event_f1": row.get("event_f1"),
                    }
                )

        for pe in pred_events:
            key = (pe.get("event_type"),) + trigger_span(pe)[1:] + (argument_tuples(pe),)
            if key in gold_event_set & pred_event_set:
                continue
            cat, missing, extra = classify_pred_event(pe, gold_events)
            pred_error[cat] += 1
            pred_type_errors[pe.get("event_type")][cat] += 1
            for role, _, _ in extra:
                extra_arg_roles[role] += 1

    return {
        "row_buckets": dict(row_buckets),
        "gold_event_error_categories": dict(gold_error),
        "pred_event_error_categories": dict(pred_error),
        "missing_arg_roles": dict(missing_arg_roles.most_common()),
        "extra_arg_roles": dict(extra_arg_roles.most_common()),
        "gold_type_errors_top": {
            k: dict(v) for k, v in sorted(gold_type_errors.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:12]
        },
        "pred_type_errors_top": {
            k: dict(v) for k, v in sorted(pred_type_errors.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:12]
        },
        "examples": examples,
    }


def compare_rows(rows_a, rows_b, name_a, name_b):
    out = Counter()
    examples = []
    for idx, (a, b) in enumerate(zip(rows_a, rows_b)):
        af = a.get("event_f1", 0) or 0
        bf = b.get("event_f1", 0) or 0
        aa = a.get("argument_f1", 0) or 0
        ba = b.get("argument_f1", 0) or 0
        if bf > af:
            out[f"{name_b}_event_better"] += 1
        elif bf < af:
            out[f"{name_b}_event_worse"] += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "row": idx,
                        f"{name_a}_event_f1": af,
                        f"{name_b}_event_f1": bf,
                        f"{name_a}_argument_f1": aa,
                        f"{name_b}_argument_f1": ba,
                        "gold": a.get("gold", {}),
                        f"{name_a}_predicted": a.get("predicted", {}),
                        f"{name_b}_predicted": b.get("predicted", {}),
                    }
                )
        else:
            out[f"{name_b}_event_same"] += 1
        if ba > aa and bf <= af:
            out[f"{name_b}_arg_better_event_not_better"] += 1
    return {"counts": dict(out), "worse_examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_json", required=True)
    ap.add_argument("inputs", nargs="+", help="name=path entries")
    args = ap.parse_args()

    rows_by_name = {}
    for item in args.inputs:
        name, path = item.split("=", 1)
        rows_by_name[name] = load_jsonl(Path(path))

    result = {"runs": {}, "comparisons": {}}
    for name, rows in rows_by_name.items():
        result["runs"][name] = {
            "path": None,
            "rows": len(rows),
            "metrics": aggregate(rows),
            "diagnosis": diagnose(rows),
        }

    for base in ["direct", "e57", "e70"]:
        if base in rows_by_name:
            for target in ["e71a", "e71b"]:
                if target in rows_by_name:
                    result["comparisons"][f"{base}_vs_{target}"] = compare_rows(rows_by_name[base], rows_by_name[target], base, target)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(out)
    for name, payload in result["runs"].items():
        m = payload["metrics"]
        print(name, "A/E/T", f"{m['argument']['f1']:.4f}", f"{m['event']['f1']:.4f}", f"{m['trigger']['f1']:.4f}")
        print("  gold errors", payload["diagnosis"]["gold_event_error_categories"])


if __name__ == "__main__":
    main()
