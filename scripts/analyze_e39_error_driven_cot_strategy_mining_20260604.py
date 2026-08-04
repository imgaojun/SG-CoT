#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def event_key(event: dict, include_offsets: bool = True) -> tuple:
    trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
    args = []
    for arg in event.get("arguments", []) or []:
        if not isinstance(arg, dict):
            continue
        if include_offsets:
            args.append((arg.get("role"), arg.get("text"), arg.get("start"), arg.get("end")))
        else:
            args.append((arg.get("role"), arg.get("text")))
    args = tuple(sorted(args))
    if include_offsets:
        trig = (trigger.get("text"), trigger.get("start"), trigger.get("end"))
    else:
        trig = trigger.get("text")
    return (event.get("event_type"), trig, args)


def trigger_keys(payload: dict, include_offsets: bool = True) -> set[tuple]:
    out = set()
    for event in payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        if include_offsets:
            out.add((event.get("event_type"), trigger.get("text"), trigger.get("start"), trigger.get("end")))
        else:
            out.add((event.get("event_type"), trigger.get("text")))
    return out


def arg_keys(payload: dict, include_offsets: bool = True) -> set[tuple]:
    out = set()
    for event in payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        trig = (event.get("event_type"), trigger.get("text"))
        if include_offsets:
            trig = (event.get("event_type"), trigger.get("text"), trigger.get("start"), trigger.get("end"))
        for arg in event.get("arguments", []) or []:
            if not isinstance(arg, dict):
                continue
            if include_offsets:
                out.add((*trig, arg.get("role"), arg.get("text"), arg.get("start"), arg.get("end")))
            else:
                out.add((*trig, arg.get("role"), arg.get("text")))
    return out


def f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    return 2 * tp / (len(pred) + len(gold))


def score_payload(pred: dict, gold: dict, include_offsets: bool = True) -> dict:
    pred_events = {event_key(e, include_offsets) for e in pred.get("events", []) or [] if isinstance(e, dict)}
    gold_events = {event_key(e, include_offsets) for e in gold.get("events", []) or [] if isinstance(e, dict)}
    return {
        "trigger_f1": f1(trigger_keys(pred, include_offsets), trigger_keys(gold, include_offsets)),
        "argument_f1": f1(arg_keys(pred, include_offsets), arg_keys(gold, include_offsets)),
        "event_f1": f1(pred_events, gold_events),
    }


def get_id(row: dict) -> str:
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or f"{meta.get('doc_id')}::{row.get('input', '')[:40]}"


def classify(row: dict) -> list[str]:
    gold = row.get("gold") or {}
    pred = row.get("final_predicted") or row.get("predicted") or {}
    labels = []
    strict = score_payload(pred, gold, True)
    loose = score_payload(pred, gold, False)
    if row.get("valid_final_json") is not True:
        labels.append("invalid_final_json")
    if loose["trigger_f1"] > strict["trigger_f1"]:
        labels.append("offset_or_span_index_mismatch_trigger")
    if loose["argument_f1"] > strict["argument_f1"]:
        labels.append("offset_or_span_index_mismatch_argument")
    if loose["event_f1"] > strict["event_f1"]:
        labels.append("offset_or_span_index_mismatch_event")
    gold_types = Counter(e.get("event_type") for e in gold.get("events", []) or [] if isinstance(e, dict))
    pred_types = Counter(e.get("event_type") for e in pred.get("events", []) or [] if isinstance(e, dict))
    if pred_types - gold_types:
        labels.append("spurious_event_type")
    if gold_types - pred_types:
        labels.append("missing_event_type")
    gold_arg_roles = Counter()
    pred_arg_roles = Counter()
    for payload, counter in [(gold, gold_arg_roles), (pred, pred_arg_roles)]:
        for e in payload.get("events", []) or []:
            if not isinstance(e, dict):
                continue
            for a in e.get("arguments", []) or []:
                if isinstance(a, dict):
                    counter[(e.get("event_type"), a.get("role"))] += 1
    if pred_arg_roles - gold_arg_roles:
        labels.append("spurious_role")
    if gold_arg_roles - pred_arg_roles:
        labels.append("missing_role")
    if not labels and (strict["argument_f1"] < 1 or strict["event_f1"] < 1 or strict["trigger_f1"] < 1):
        labels.append("other_structural_mismatch")
    return labels


def summarize_split(name: str, direct_rows: list[dict], e37_rows: list[dict], e38_rows: list[dict]) -> dict:
    direct = {get_id(r): r for r in direct_rows}
    e37 = {get_id(r): r for r in e37_rows}
    e38 = {get_id(r): r for r in e38_rows}
    ids = sorted(set(direct) & set(e37) & set(e38))
    counters = Counter()
    deltas = []
    loose_gain = Counter()
    examples = defaultdict(list)
    for wid in ids:
        d, r37, r38 = direct[wid], e37[wid], e38[wid]
        d_score = score_payload(d.get("final_predicted") or d.get("predicted") or {}, d.get("gold") or {}, True)
        r37_score = score_payload(r37.get("final_predicted") or r37.get("predicted") or {}, r37.get("gold") or {}, True)
        r38_score = score_payload(r38.get("final_predicted") or r38.get("predicted") or {}, r38.get("gold") or {}, True)
        r38_loose = score_payload(r38.get("final_predicted") or r38.get("predicted") or {}, r38.get("gold") or {}, False)
        labels = classify(r38)
        counters.update(labels)
        for metric in ("trigger_f1", "argument_f1", "event_f1"):
            if r38_score[metric] > d_score[metric]:
                counters[f"e38_beats_direct_{metric}"] += 1
            elif r38_score[metric] < d_score[metric]:
                counters[f"e38_loses_direct_{metric}"] += 1
            else:
                counters[f"e38_ties_direct_{metric}"] += 1
            if r38_score[metric] > r37_score[metric]:
                counters[f"e38_beats_e37_{metric}"] += 1
            elif r38_score[metric] < r37_score[metric]:
                counters[f"e38_loses_e37_{metric}"] += 1
            else:
                counters[f"e38_ties_e37_{metric}"] += 1
            if r38_loose[metric] > r38_score[metric]:
                loose_gain[metric] += 1
        item = {
            "wnd_id": wid,
            "direct": d_score,
            "e37": r37_score,
            "e38": r38_score,
            "e38_loose_no_offsets": r38_loose,
            "labels": labels,
            "gold_event_types": (r38.get("meta") or {}).get("gold_event_types"),
            "text": ((r38.get("input") or "").split("Tokens:")[0]).strip(),
            "e38_generated": (r38.get("generated_text") or "")[:1200],
        }
        deltas.append(item)
        for label in labels[:3]:
            if len(examples[label]) < 5:
                examples[label].append(item)
    return {
        "split": name,
        "num_examples": len(ids),
        "counts": dict(counters.most_common()),
        "loose_no_offset_gain_counts": dict(loose_gain),
        "examples": {k: v for k, v in examples.items()},
        "worst_e38_argument": sorted(deltas, key=lambda x: (x["e38"]["argument_f1"], -x["direct"]["argument_f1"]))[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/stage2_strategy_cot_e39/error_mining_20260604"))
    args = ap.parse_args()
    paths = {
        "direct": Path("outputs/stage2_llm_reasoning_e36/formal_20260604/e36_s0_seed500_direct"),
        "e37": Path("outputs/stage2_strategy_cot_e37/formal_e37_seed500_20260604"),
        "e38": Path("outputs/stage2_strategy_cot_e37/formal_e38_seed1500_20260604"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"splits": {}}
    for split in ("test_seen", "test_unseen"):
        report["splits"][split] = summarize_split(
            split,
            load_jsonl(paths["direct"] / split / "predictions.jsonl"),
            load_jsonl(paths["e37"] / split / "predictions.jsonl"),
            load_jsonl(paths["e38"] / split / "predictions.jsonl"),
        )
    out = args.output_dir / "e39_error_mining_summary.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({s: report["splits"][s]["counts"] for s in report["splits"]}, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
