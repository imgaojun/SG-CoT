#!/usr/bin/env python3
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path("outputs/stage2_strategy_cot_e65/e57_cross_model_20260608")
OUT_JSON = Path("reports/artifacts/2026-06-15_e76_contrastive_exactness_case_analysis.json")
OUT_MD = Path("reports/2026-06-15_e76_contrastive_exactness_case_analysis.md")


RUNS = {
    "E57": {
        "seen": Path("outputs/stage2_strategy_cot_e56/e57_checkpoint-279_eval/test_seen/predictions.jsonl"),
        "unseen": Path("outputs/stage2_strategy_cot_e56/e57_checkpoint-279_eval/test_unseen/predictions.jsonl"),
    },
    "E73": {
        "seen": BASE / "qwen4_e73_e57_recall_first_exactness_last/checkpoint-279/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e73_e57_recall_first_exactness_last/checkpoint-279/test_unseen/predictions.jsonl",
    },
    "E76-ck186": {
        "seen": BASE / "qwen4_e76_contrastive_exactness/checkpoint-186/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e76_contrastive_exactness/checkpoint-186/test_unseen/predictions.jsonl",
    },
    "E76-ck279": {
        "seen": BASE / "qwen4_e76_contrastive_exactness/checkpoint-279/test_seen/predictions.jsonl",
        "unseen": BASE / "qwen4_e76_contrastive_exactness/checkpoint-279/test_unseen/predictions.jsonl",
    },
}


DIRECT = {
    "seen": {"argument_f1": 0.4821, "event_f1": 0.3593, "trigger_f1": 0.7128},
    "unseen": {"argument_f1": 0.1324, "event_f1": 0.0996, "trigger_f1": 0.2053},
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pred_payload(row):
    return row.get("predicted") or row.get("final_predicted") or {}


def event_items(payload):
    if not isinstance(payload, dict):
        return []
    events = payload.get("events") or []
    return events if isinstance(events, list) else []


def row_key(row, idx):
    meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
    return meta.get("wnd_id") or meta.get("doc_id") or row.get("id") or row.get("sample_id") or row.get("input", "")[:120] or str(idx)


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


def text_excerpt(row, limit=360):
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
        raw_args = ev.get("arguments", []) if isinstance(ev, dict) else []
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


def metric_triplet(summary):
    return [summary["argument_f1"], summary["event_f1"], summary["trigger_f1"]]


def fmt_triplet(vals):
    return " / ".join(f"{v:.4f}" for v in vals)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(out)


def event_type_family(event_type):
    if not event_type:
        return "unknown"
    return event_type.split(":")[0]


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
        "gold_type_top": {k: dict(v) for k, v in sorted(by_type.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:15]},
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


def argument_mechanisms(base_row, e76_row, e76_label="e76"):
    out = []
    gold_events = event_items(e76_row.get("gold", {}))
    base_by_frame = {trigger_span(ev): ev for ev in event_items(pred_payload(base_row))}
    e76_by_frame = {trigger_span(ev): ev for ev in event_items(pred_payload(e76_row))}
    for ge in gold_events:
        frame = trigger_span(ge)
        gold_args = set((r, s, e) for r, s, e, _ in arg_tuples(ge))
        base_args = set()
        e76_args = set()
        if frame in base_by_frame:
            base_args = set((r, s, e) for r, s, e, _ in arg_tuples(base_by_frame[frame]))
        if frame in e76_by_frame:
            e76_args = set((r, s, e) for r, s, e, _ in arg_tuples(e76_by_frame[frame]))
        if base_args - gold_args and len(e76_args - gold_args) < len(base_args - gold_args):
            out.append(f"{e76_label}_removes_extra_roles")
        if gold_args - base_args and len(gold_args - e76_args) < len(gold_args - base_args):
            out.append(f"{e76_label}_recovers_missing_local_roles")
        base_spans = {(s, e): r for r, s, e in base_args}
        e76_spans = {(s, e): r for r, s, e in e76_args}
        gold_spans = {(s, e): r for r, s, e in gold_args}
        for span in set(base_spans) & set(e76_spans) & set(gold_spans):
            if base_spans[span] != gold_spans[span] and e76_spans[span] == gold_spans[span]:
                out.append(f"{e76_label}_corrects_role_label")
            if base_spans[span] == gold_spans[span] and e76_spans[span] != gold_spans[span]:
                out.append(f"{e76_label}_breaks_role_label")
    return out or ["argument_gain_other"]


def compare_pair(base_rows, e76_rows, split, base_label):
    base_by_key = {row_key(row, idx): row for idx, row in enumerate(base_rows)}
    e76_by_key = {row_key(row, idx): row for idx, row in enumerate(e76_rows)}
    common = [k for k in base_by_key if k in e76_by_key]
    counts = Counter()
    mechanisms = Counter()
    gain_families = Counter()
    loss_families = Counter()
    examples = defaultdict(list)
    for key in common:
        base = base_by_key[key]
        e76 = e76_by_key[key]
        mb = row_metrics(base)
        me = row_metrics(e76)
        for metric in ["event", "argument", "trigger"]:
            delta = me[metric] - mb[metric]
            if delta > 1e-9:
                counts[f"e76_{metric}_better"] += 1
                if metric in {"event", "argument"}:
                    add_example(examples, f"e76_{metric}_better", split, key, base, e76, base_label)
            elif delta < -1e-9:
                counts[f"e76_{metric}_worse"] += 1
                if metric in {"event", "argument"}:
                    add_example(examples, f"e76_{metric}_worse", split, key, base, e76, base_label)
            else:
                counts[f"{metric}_same"] += 1
        if me["event"] > mb["event"]:
            for ev in event_items(e76.get("gold", {})):
                gain_families[event_type_family(ev.get("event_type"))] += 1
        if me["event"] < mb["event"]:
            for ev in event_items(e76.get("gold", {})):
                loss_families[event_type_family(ev.get("event_type"))] += 1
        if me["argument"] > mb["argument"]:
            for mechanism in argument_mechanisms(base, e76):
                mechanisms[mechanism] += 1
    return {
        "base_label": base_label,
        "num_common_rows": len(common),
        "counts": dict(counts),
        "argument_mechanisms": dict(mechanisms),
        "gain_gold_families": gain_families.most_common(),
        "loss_gold_families": loss_families.most_common(),
        "examples": examples,
    }


def add_example(examples, bucket, split, key, base_row, e76_row, base_label, limit=4):
    if len(examples[bucket]) >= limit:
        return
    examples[bucket].append({
        "split": split,
        "row_key": key,
        "base_label": base_label,
        "text": text_excerpt(e76_row),
        "metrics": {base_label: row_metrics(base_row), "E76-ck186": row_metrics(e76_row)},
        "gold": compact_payload(e76_row.get("gold", {})),
        "base_predicted": compact_payload(pred_payload(base_row)),
        "e76_predicted": compact_payload(pred_payload(e76_row)),
    })


def selected_cases(comparisons):
    unseen_vs_e57 = comparisons["E76-ck186_vs_E57"]["unseen"]["examples"]
    unseen_vs_e73 = comparisons["E76-ck186_vs_E73"]["unseen"]["examples"]
    unseen_vs_279 = comparisons["E76-ck186_vs_E76-ck279"]["unseen"]["examples"]
    buckets = {
        "contact_generic_fix": [],
        "life_boundary_fix": [],
        "argument_pruning": [],
        "e76_worse": [],
        "ck279_overfit": [],
    }
    for case in unseen_vs_e57.get("e76_event_better", []) + unseen_vs_e73.get("e76_event_better", []):
        gold_types = " ".join(ev["event_type"] or "" for ev in case["gold"])
        text = case["text"].lower()
        if "Contact:" in gold_types and len(buckets["contact_generic_fix"]) < 2:
            buckets["contact_generic_fix"].append(case)
        if "Life:" in gold_types and len(buckets["life_boundary_fix"]) < 3:
            buckets["life_boundary_fix"].append(case)
        if len(buckets["argument_pruning"]) < 3:
            base_args = sum(len(ev["arguments"]) for ev in case["base_predicted"])
            e76_args = sum(len(ev["arguments"]) for ev in case["e76_predicted"])
            if e76_args < base_args or any(w in text for w in ["hurt", "harm", "hit", "made"]):
                buckets["argument_pruning"].append(case)
    for case in unseen_vs_e57.get("e76_event_worse", []) + unseen_vs_e73.get("e76_event_worse", []):
        if len(buckets["e76_worse"]) < 2:
            buckets["e76_worse"].append(case)
    for case in unseen_vs_279.get("e76_event_better", []) + unseen_vs_279.get("e76_argument_better", []):
        if len(buckets["ck279_overfit"]) < 2:
            buckets["ck279_overfit"].append(case)
    return buckets


def build_result():
    rows = {run: {split: load_jsonl(path) for split, path in splits.items()} for run, splits in RUNS.items()}
    summaries = {
        run: {split: load_json(path.parent / "summary.json") for split, path in splits.items()}
        for run, splits in RUNS.items()
    }
    metrics = [
        {"run": "Direct", "seen": metric_triplet(DIRECT["seen"]), "unseen": metric_triplet(DIRECT["unseen"])},
    ]
    for run in ["E57", "E73", "E76-ck186", "E76-ck279"]:
        metrics.append({"run": run, "seen": metric_triplet(summaries[run]["seen"]), "unseen": metric_triplet(summaries[run]["unseen"])})
    comparisons = {}
    for base in ["E57", "E73", "E76-ck279"]:
        key = f"E76-ck186_vs_{base}"
        comparisons[key] = {
            split: compare_pair(rows[base][split], rows["E76-ck186"][split], split, base)
            for split in ["seen", "unseen"]
        }
    result = {
        "inputs": {run: {split: str(path) for split, path in splits.items()} for run, splits in RUNS.items()},
        "metrics_table": metrics,
        "comparisons": comparisons,
        "event_error_categories": {
            run: {split: event_error_categories(rows[run][split]) for split in ["seen", "unseen"]}
            for run in ["E57", "E73", "E76-ck186", "E76-ck279"]
        },
        "prediction_distribution": {
            run: {split: prediction_distribution(rows[run][split]) for split in ["seen", "unseen"]}
            for run in ["E57", "E73", "E76-ck186", "E76-ck279"]
        },
    }
    result["selected_cases"] = selected_cases(comparisons)
    return result


def write_markdown(result):
    metrics_rows = [[r["run"], f"`{fmt_triplet(r['seen'])}`", f"`{fmt_triplet(r['unseen'])}`"] for r in result["metrics_table"]]
    row_delta = []
    for comp_key, comp in result["comparisons"].items():
        for split in ["unseen", "seen"]:
            c = comp[split]["counts"]
            row_delta.append([
                comp_key,
                split,
                comp[split]["num_common_rows"],
                c.get("e76_event_better", 0),
                c.get("e76_event_worse", 0),
                c.get("event_same", 0),
                c.get("e76_argument_better", 0),
                c.get("e76_argument_worse", 0),
                c.get("argument_same", 0),
            ])
    dist_rows = []
    for run in ["E57", "E73", "E76-ck186", "E76-ck279"]:
        d = result["prediction_distribution"][run]["unseen"]
        dist_rows.append([run, d["pred_events"], d["pred_args"], d["contact_contact"], d["contact_correspondence"], d["contact_meet"], d["contact_broadcast"]])
    error_rows = []
    for run in ["E57", "E73", "E76-ck186", "E76-ck279"]:
        cats = result["event_error_categories"][run]["unseen"]["gold_side"]
        error_rows.append([run, cats.get("missing_frame", 0), cats.get("same_trigger_wrong_type", 0), cats.get("same_type_wrong_trigger", 0), cats.get("frame_matched_argument_mismatch", 0)])
    lines = [
        "# E76 Contrastive Exactness Case Analysis",
        "",
        "## Summary",
        "",
        "E76 checkpoint-186 is the best E76 checkpoint on the priority unseen Argument/Event metrics. It improves E57 on unseen Argument and Event while lowering Trigger, and it improves E73 on unseen Event and Trigger while trailing E73 original on Argument. The mechanism is clearest on Contact subtype arbitration, Life/Conflict boundary control, and exactness-last argument pruning.",
        "",
        "## Metrics",
        "",
        md_table(["Run", "test_seen A/E/T", "test_unseen A/E/T"], metrics_rows),
        "",
        "## Row-Level Delta",
        "",
        md_table(["Comparison", "Split", "Common rows", "E76 Event better", "E76 Event worse", "Event same", "E76 Arg better", "E76 Arg worse", "Arg same"], row_delta),
        "",
        "## Unseen Error Categories",
        "",
        md_table(["Run", "Missing frame", "Same trigger wrong type", "Same type wrong trigger", "Frame matched arg mismatch"], error_rows),
        "",
        "## Unseen Prediction Distribution",
        "",
        md_table(["Run", "Pred events", "Pred args", "Contact:Contact", "Contact:Correspondence", "Contact:Meet", "Contact:Broadcast"], dist_rows),
        "",
        "## Type-Family Gain/Loss",
        "",
    ]
    for comp_key in ["E76-ck186_vs_E57", "E76-ck186_vs_E73", "E76-ck186_vs_E76-ck279"]:
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
                f"- `{case['row_key']}` vs `{case['base_label']}` metrics `{case['metrics']}`",
                f"  - text: {case['text']}",
                f"  - gold: `{json.dumps(case['gold'], ensure_ascii=False)}`",
                f"  - baseline: `{json.dumps(case['base_predicted'], ensure_ascii=False)}`",
                f"  - E76 ck186: `{json.dumps(case['e76_predicted'], ensure_ascii=False)}`",
            ])
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "E76 is not just a data-size effect. Compared with E75 scaling, E76 changes the distribution and content of the supervision: it reduces generic Contact overprediction, corrects several Life/Conflict boundary cases, and prunes unsupported arguments. The ck279 regression suggests that longer training increases plausible-but-not-gold output, especially extra arguments, so 2-epoch/ck186-style early stopping should be tested as the stability setting.",
        "",
        "Artifacts:",
        "",
        "- JSON: `reports/artifacts/2026-06-15_e76_contrastive_exactness_case_analysis.json`",
        "- Script: `scripts/diagnose_e76_contrastive_exactness_cases_20260615.py`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    result = build_result()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result)


if __name__ == "__main__":
    main()
