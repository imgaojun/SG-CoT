#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def event_items(payload):
    if not isinstance(payload, dict):
        return []
    events = payload.get("events", [])
    return events if isinstance(events, list) else []


def pred_payload(row):
    return row.get("predicted", row.get("final_predicted", {}))


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
    out = []
    args = event.get("arguments", []) if isinstance(event, dict) else []
    if not isinstance(args, list):
        return tuple()
    for arg in args:
        if isinstance(arg, dict):
            out.append((arg.get("role"), arg.get("start"), arg.get("end"), arg.get("text")))
    return tuple(sorted(out, key=lambda x: (x[0] or "", -1 if x[1] is None else x[1], -1 if x[2] is None else x[2], x[3] or "")))


def normalize(payload):
    triggers = set()
    args = set()
    events = set()
    frames = set()
    for ev in event_items(payload):
        et, ts, te = trigger_span(ev)
        frame = (et, ts, te)
        triggers.add(frame)
        frames.add(frame)
        ev_args = tuple((role, start, end) for role, start, end, _ in arg_tuples(ev))
        events.add((et, ts, te, ev_args))
        for role, start, end, _ in arg_tuples(ev):
            args.add((et, ts, te, role, start, end))
    return triggers, args, events, frames


def row_key(row, idx):
    meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
    return meta.get("wnd_id") or meta.get("doc_id") or str(idx)


def row_metrics(row):
    return {
        "trigger": float(row.get("trigger_f1", 0.0) or 0.0),
        "argument": float(row.get("argument_f1", 0.0) or 0.0),
        "event": float(row.get("event_f1", 0.0) or 0.0),
    }


def text_excerpt(row, limit=420):
    text = row.get("input", "")
    match = re.search(r"Text:\n(.*?)(?:\n\nTokens:|\Z)", text, re.S)
    if match:
        text = match.group(1).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit] + ("..." if len(text) > limit else "")


def compact_payload(payload, max_events=4):
    out = []
    for ev in event_items(payload)[:max_events]:
        trig = ev.get("trigger", {}) if isinstance(ev, dict) else {}
        args = []
        for arg in ev.get("arguments", []) if isinstance(ev.get("arguments", []), list) else []:
            if isinstance(arg, dict):
                args.append({
                    "role": arg.get("role"),
                    "text": arg.get("text"),
                    "start": arg.get("start"),
                    "end": arg.get("end"),
                })
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
    g_args = set((role, start, end) for role, start, end, _ in arg_tuples(gold_event))
    for pred in pred_events:
        if trigger_span(pred) == (g_type, g_ts, g_te):
            p_args = set((role, start, end) for role, start, end, _ in arg_tuples(pred))
            return "frame_matched_argument_mismatch", g_args - p_args, p_args - g_args
    for pred in pred_events:
        if trigger_only_span(pred) == (g_ts, g_te):
            return "same_trigger_wrong_type", set(), set()
    for pred in pred_events:
        if pred.get("event_type") == g_type:
            return "same_type_wrong_trigger", set(), set()
    return "missing_frame", set(), set()


def classify_pred_event(pred_event, gold_events):
    p_type, p_ts, p_te = trigger_span(pred_event)
    p_args = set((role, start, end) for role, start, end, _ in arg_tuples(pred_event))
    for gold in gold_events:
        if trigger_span(gold) == (p_type, p_ts, p_te):
            g_args = set((role, start, end) for role, start, end, _ in arg_tuples(gold))
            return "frame_matched_argument_mismatch", g_args - p_args, p_args - g_args
    for gold in gold_events:
        if trigger_only_span(gold) == (p_ts, p_te):
            return "same_trigger_wrong_type", set(), set()
    for gold in gold_events:
        if gold.get("event_type") == p_type:
            return "same_type_wrong_trigger", set(), set()
    return "extra_frame", set(), set()


def event_error_categories(rows):
    gold_counts = Counter()
    pred_counts = Counter()
    by_type = defaultdict(Counter)
    for row in rows:
        gold_events = event_items(row.get("gold", {}))
        pred_events = event_items(pred_payload(row))
        _, _, gold_exact, _ = normalize(row.get("gold", {}))
        _, _, pred_exact, _ = normalize(pred_payload(row))
        matched = gold_exact & pred_exact
        for ge in gold_events:
            key = trigger_span(ge) + (tuple((r, s, e) for r, s, e, _ in arg_tuples(ge)),)
            if key in matched:
                continue
            cat, _, _ = classify_gold_event(ge, pred_events)
            gold_counts[cat] += 1
            by_type[ge.get("event_type")][cat] += 1
        for pe in pred_events:
            key = trigger_span(pe) + (tuple((r, s, e) for r, s, e, _ in arg_tuples(pe)),)
            if key in matched:
                continue
            cat, _, _ = classify_pred_event(pe, gold_events)
            pred_counts[cat] += 1
    return {
        "gold_side": dict(gold_counts),
        "prediction_side": dict(pred_counts),
        "gold_type_top": {
            k: dict(v) for k, v in sorted(by_type.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:15]
        },
    }


def compare_rows(e57_rows, e73_rows, split):
    e57_by_key = {row_key(row, idx): row for idx, row in enumerate(e57_rows)}
    e73_by_key = {row_key(row, idx): row for idx, row in enumerate(e73_rows)}
    common = [k for k in e57_by_key if k in e73_by_key]
    counts = Counter()
    examples = defaultdict(list)
    arg_mechanisms = Counter()
    for key in common:
        r57 = e57_by_key[key]
        r73 = e73_by_key[key]
        m57 = row_metrics(r57)
        m73 = row_metrics(r73)
        if m73["event"] > m57["event"]:
            counts["e73_event_better"] += 1
            add_example(examples, "e73_event_better", split, key, r57, r73)
        elif m73["event"] < m57["event"]:
            counts["e73_event_worse"] += 1
            add_example(examples, "e73_event_worse", split, key, r57, r73)
        else:
            counts["event_same"] += 1
        if m73["argument"] > m57["argument"]:
            counts["e73_argument_better"] += 1
            add_example(examples, "e73_argument_better", split, key, r57, r73)
        elif m73["argument"] < m57["argument"]:
            counts["e73_argument_worse"] += 1
            add_example(examples, "e73_argument_worse", split, key, r57, r73)
        else:
            counts["argument_same"] += 1
        if m57["trigger"] > m73["trigger"] and m57["event"] <= m73["event"]:
            counts["e57_trigger_better_but_event_not_better"] += 1
            add_example(examples, "e57_trigger_better_but_event_not_better", split, key, r57, r73)
        if m57["trigger"] > m73["trigger"] and m57["event"] > m73["event"]:
            counts["e57_trigger_better_and_event_better"] += 1

        if m73["argument"] > m57["argument"]:
            for mechanism in argument_mechanisms(r57, r73):
                arg_mechanisms[mechanism] += 1

    return {"num_common_rows": len(common), "counts": dict(counts), "argument_mechanisms": dict(arg_mechanisms), "examples": examples}


def add_example(examples, bucket, split, key, r57, r73, limit=3):
    if len(examples[bucket]) >= limit:
        return
    examples[bucket].append({
        "split": split,
        "row_key": key,
        "text": text_excerpt(r73),
        "metrics": {"e57": row_metrics(r57), "e73": row_metrics(r73)},
        "gold": compact_payload(r73.get("gold", {})),
        "e57_predicted": compact_payload(pred_payload(r57)),
        "e73_predicted": compact_payload(pred_payload(r73)),
    })


def argument_mechanisms(r57, r73):
    out = []
    gold_events = event_items(r73.get("gold", {}))
    p57_events = event_items(pred_payload(r57))
    p73_events = event_items(pred_payload(r73))
    p57_by_frame = {trigger_span(ev): ev for ev in p57_events}
    p73_by_frame = {trigger_span(ev): ev for ev in p73_events}
    for ge in gold_events:
        frame = trigger_span(ge)
        gold_args = set((r, s, e) for r, s, e, _ in arg_tuples(ge))
        a57 = set()
        a73 = set()
        if frame in p57_by_frame:
            a57 = set((r, s, e) for r, s, e, _ in arg_tuples(p57_by_frame[frame]))
        if frame in p73_by_frame:
            a73 = set((r, s, e) for r, s, e, _ in arg_tuples(p73_by_frame[frame]))
        if a57 - gold_args and len(a73 - gold_args) < len(a57 - gold_args):
            out.append("e73_removes_e57_extra_roles")
        if gold_args - a57 and len(gold_args - a73) < len(gold_args - a57):
            out.append("e73_recovers_missing_local_roles")
        spans57 = {(s, e): r for r, s, e in a57}
        spans73 = {(s, e): r for r, s, e in a73}
        gold_spans = {(s, e): r for r, s, e in gold_args}
        for span in set(spans57) & set(spans73) & set(gold_spans):
            if spans57[span] != gold_spans[span] and spans73[span] == gold_spans[span]:
                out.append("e73_corrects_role_label")
            if spans57[span] == gold_spans[span] and spans73[span] != gold_spans[span]:
                out.append("e73_breaks_role_label")
    if row_metrics(r73)["argument"] > row_metrics(r57)["argument"] and row_metrics(r73)["event"] < row_metrics(r57)["event"]:
        out.append("e73_argument_better_despite_frame_loss")
    return out or ["argument_gain_other"]


def type_family(event_type):
    if not event_type:
        return "unknown"
    if event_type.startswith("Contact:"):
        return "Contact subtypes"
    if event_type.startswith("Justice:"):
        return "Justice events"
    if event_type.startswith("Movement:") or event_type.startswith("Transaction:"):
        return "Movement/Transaction"
    return "Other"


def type_level(rows_by_run):
    gold_type_counts = Counter()
    family = defaultdict(lambda: defaultdict(Counter))
    for row in rows_by_run["e73_unseen"]:
        for ev in event_items(row.get("gold", {})):
            gold_type_counts[ev.get("event_type")] += 1
    rare = {k for k, v in gold_type_counts.items() if v <= 2}
    for run, rows in rows_by_run.items():
        if not run.endswith("_unseen"):
            continue
        short = run.replace("_unseen", "")
        for row in rows:
            gold_types = {ev.get("event_type") for ev in event_items(row.get("gold", {}))}
            pred_types = {ev.get("event_type") for ev in event_items(pred_payload(row))}
            for et in gold_types:
                fam = "rare unseen event types" if et in rare else type_family(et)
                family[fam][short]["gold_rows"] += 1
                if et in pred_types:
                    family[fam][short]["type_recalled_row"] += 1
                else:
                    family[fam][short]["type_missing_row"] += 1
    return {fam: {run: dict(cnt) for run, cnt in runs.items()} for fam, runs in family.items()}


def metric_row(label, seen, unseen):
    def triplet(summary):
        return [summary["argument_f1"], summary["event_f1"], summary["trigger_f1"]]
    return {"run": label, "test_seen_AET": triplet(seen), "test_unseen_AET": triplet(unseen)}


def fmt_triplet(vals):
    return " / ".join(f"{v:.4f}" for v in vals)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(out)


def write_markdown(path, result):
    metrics_rows = [
        [r["run"], f"`{fmt_triplet(r['test_seen_AET'])}`", f"`{fmt_triplet(r['test_unseen_AET'])}`"]
        for r in result["metrics_table"]
    ]
    unseen = result["comparisons"]["test_unseen"]
    seen = result["comparisons"]["test_seen"]
    event_rows = []
    for split in ["test_unseen", "test_seen"]:
        for run in ["e57", "e73"]:
            cats = result["event_error_categories"][split][run]["gold_side"]
            event_rows.append([
                split,
                run.upper(),
                cats.get("missing_frame", 0),
                cats.get("same_trigger_wrong_type", 0),
                cats.get("same_type_wrong_trigger", 0),
                cats.get("frame_matched_argument_mismatch", 0),
            ])
    mech = unseen["argument_mechanisms"]
    lines = [
        "# E73 vs E57 Case Analysis",
        "",
        "## Summary",
        "",
        "Official metrics are `Argument / Event / Trigger`. E73 checkpoint-279 improves the priority `test_unseen` Argument and Event metrics over E57, while Trigger drops. The case analysis supports the current interpretation: E73 is not simply better at finding more triggers; it shifts the output toward more useful final event/argument tuples under the evidence-based evaluator.",
        "",
        "## Metrics",
        "",
        md_table(["Run", "test_seen A/E/T", "test_unseen A/E/T"], metrics_rows),
        "",
        "## Row-Level Delta",
        "",
        md_table(
            ["Split", "Common rows", "E73 Event better", "E73 Event worse", "Event same", "E73 Arg better", "E73 Arg worse", "Arg same", "E57 Trigger better but Event not better"],
            [
                ["test_unseen", unseen["num_common_rows"], unseen["counts"].get("e73_event_better", 0), unseen["counts"].get("e73_event_worse", 0), unseen["counts"].get("event_same", 0), unseen["counts"].get("e73_argument_better", 0), unseen["counts"].get("e73_argument_worse", 0), unseen["counts"].get("argument_same", 0), unseen["counts"].get("e57_trigger_better_but_event_not_better", 0)],
                ["test_seen", seen["num_common_rows"], seen["counts"].get("e73_event_better", 0), seen["counts"].get("e73_event_worse", 0), seen["counts"].get("event_same", 0), seen["counts"].get("e73_argument_better", 0), seen["counts"].get("e73_argument_worse", 0), seen["counts"].get("argument_same", 0), seen["counts"].get("e57_trigger_better_but_event_not_better", 0)],
            ],
        ),
        "",
        "## Event Error Categories",
        "",
        md_table(["Split", "Run", "Missing frame", "Same trigger wrong type", "Same type wrong trigger", "Frame matched arg mismatch"], event_rows),
        "",
        "## Argument Mechanisms On Unseen",
        "",
        md_table(["Mechanism", "Rows"], [[k, v] for k, v in sorted(mech.items(), key=lambda kv: kv[1], reverse=True)]),
        "",
        "## Type-Level Signals",
        "",
        "The JSON artifact contains the full type-family counters. Contact, Justice, Movement/Transaction, and rare unseen types are separated so paper writing can distinguish subtype arbitration from general frame recall.",
        "",
        "## Representative Cases",
        "",
    ]
    for bucket in ["e73_event_better", "e73_argument_better", "e57_trigger_better_but_event_not_better", "e73_event_worse"]:
        cases = unseen["examples"].get(bucket, [])
        lines.extend([f"### {bucket}", ""])
        if not cases:
            lines.extend(["No compact example selected.", ""])
            continue
        for case in cases[:2]:
            lines.extend([
                f"- `{case['row_key']}` metrics E57 `{case['metrics']['e57']}` E73 `{case['metrics']['e73']}`",
                f"  - text: {case['text']}",
                f"  - gold: `{json.dumps(case['gold'], ensure_ascii=False)}`",
                f"  - E57: `{json.dumps(case['e57_predicted'], ensure_ascii=False)}`",
                f"  - E73: `{json.dumps(case['e73_predicted'], ensure_ascii=False)}`",
            ])
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "E73 should be treated as the stronger current Qwen3-4B main-result candidate if repeat training preserves the unseen Argument/Event gain. The lower Trigger score is a real diagnostic concern, but it does not directly overturn the main claim because the paper's priority metrics are Argument and Event. The likely mechanism is that E73's recall-first/exactness-last CoT reduces some harmful argument/frame noise while still keeping enough locally supported frames to improve exact event tuples on unseen data.",
        "",
        "The next decision depends on repeat stability: if repeat1 again beats E57 on unseen Argument/Event, use E73 as the main result and report Trigger as a tradeoff; if repeat1 is mixed, report E57/E73 as a seed-sensitive family and continue prompt/data optimization from the E73 idea.",
        "",
        "Artifacts:",
        "",
        "- JSON: `reports/artifacts/2026-06-13_e73_vs_e57_case_analysis.json`",
        "- Script: `scripts/diagnose_e73_vs_e57_cases_20260613.py`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e57_seen", required=True)
    ap.add_argument("--e57_unseen", required=True)
    ap.add_argument("--e73_seen", required=True)
    ap.add_argument("--e73_unseen", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    paths = {
        "e57_seen": Path(args.e57_seen),
        "e57_unseen": Path(args.e57_unseen),
        "e73_seen": Path(args.e73_seen),
        "e73_unseen": Path(args.e73_unseen),
    }
    rows = {name: load_jsonl(path) for name, path in paths.items()}
    summary = {
        "direct_seen": {"argument_f1": 0.4821, "event_f1": 0.3593, "trigger_f1": 0.7128},
        "direct_unseen": {"argument_f1": 0.1324, "event_f1": 0.0996, "trigger_f1": 0.2053},
        "e57_seen": load_json(paths["e57_seen"].parent / "summary.json"),
        "e57_unseen": load_json(paths["e57_unseen"].parent / "summary.json"),
        "e70_seen": load_json(Path("outputs/stage2_strategy_cot_e65/e57_cross_model_20260608/qwen4_e70_candidate_audit_v2/checkpoint-282/test_seen/summary.json")),
        "e70_unseen": load_json(Path("outputs/stage2_strategy_cot_e65/e57_cross_model_20260608/qwen4_e70_candidate_audit_v2/checkpoint-282/test_unseen/summary.json")),
        "e72_seen": load_json(Path("outputs/stage2_strategy_cot_e65/e57_cross_model_20260608/qwen4_e72_e57_backbone_subtype_minarg/checkpoint-180/test_seen/summary.json")),
        "e72_unseen": load_json(Path("outputs/stage2_strategy_cot_e65/e57_cross_model_20260608/qwen4_e72_e57_backbone_subtype_minarg/checkpoint-180/test_unseen/summary.json")),
        "e73_seen": load_json(paths["e73_seen"].parent / "summary.json"),
        "e73_unseen": load_json(paths["e73_unseen"].parent / "summary.json"),
    }
    result = {
        "inputs": {name: str(path) for name, path in paths.items()},
        "metrics_table": [
            metric_row("Direct", summary["direct_seen"], summary["direct_unseen"]),
            metric_row("E57 checkpoint-279", summary["e57_seen"], summary["e57_unseen"]),
            metric_row("E70 checkpoint-282", summary["e70_seen"], summary["e70_unseen"]),
            metric_row("E72 checkpoint-180", summary["e72_seen"], summary["e72_unseen"]),
            metric_row("E73 checkpoint-279", summary["e73_seen"], summary["e73_unseen"]),
        ],
        "comparisons": {
            "test_seen": compare_rows(rows["e57_seen"], rows["e73_seen"], "test_seen"),
            "test_unseen": compare_rows(rows["e57_unseen"], rows["e73_unseen"], "test_unseen"),
        },
        "event_error_categories": {
            "test_seen": {
                "e57": event_error_categories(rows["e57_seen"]),
                "e73": event_error_categories(rows["e73_seen"]),
            },
            "test_unseen": {
                "e57": event_error_categories(rows["e57_unseen"]),
                "e73": event_error_categories(rows["e73_unseen"]),
            },
        },
        "type_level_analysis": type_level(rows),
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, result)


if __name__ == "__main__":
    main()
