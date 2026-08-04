import json
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/stage2_1_7b_step_decomposed_reasoning/e26_formal_20260527"
OUT_JSON = REPO / "reports/artifacts/2026-05-27_stage2_1_7b_e26_error_analysis.json"
OUT_MD = REPO / "reports/2026-05-27_stage2_1_7b_e26_error_analysis.md"

VARIANTS = ["e26a", "e26b", "e26c"]
BUDGETS = ["none", "light", "standard", "deep"]
SPLITS = ["test_seen", "test_unseen"]
METRICS = ["argument_f1", "event_f1", "trigger_f1"]


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_tag(text, tag):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    if start == -1:
        return None
    start += len(start_tag)
    end = text.find(end_tag, start)
    return text[start:end if end != -1 else None].strip()


def extract_json_tag(text, tag):
    value = extract_tag(text, tag)
    if not value:
        return None
    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(value[start : end + 1])
    except Exception:
        return None


def events(payload):
    if not isinstance(payload, dict):
        return []
    raw = payload.get("events", [])
    return raw if isinstance(raw, list) else []


def event_key(event):
    trigger = event.get("trigger") if isinstance(event, dict) else {}
    trigger = trigger if isinstance(trigger, dict) else {}
    return (event.get("event_type"), trigger.get("start"), trigger.get("end"))


def trigger_text_key(event):
    trigger = event.get("trigger") if isinstance(event, dict) else {}
    trigger = trigger if isinstance(trigger, dict) else {}
    return (event.get("event_type"), (trigger.get("text") or "").lower())


def arg_key(event, arg):
    trigger = event.get("trigger") if isinstance(event, dict) else {}
    trigger = trigger if isinstance(trigger, dict) else {}
    return (
        event.get("event_type"),
        trigger.get("start"),
        trigger.get("end"),
        arg.get("role"),
        arg.get("start"),
        arg.get("end"),
    )


def arg_role_text_key(event, arg):
    trigger = event.get("trigger") if isinstance(event, dict) else {}
    trigger = trigger if isinstance(trigger, dict) else {}
    return (
        event.get("event_type"),
        trigger.get("start"),
        trigger.get("end"),
        arg.get("role"),
        (arg.get("text") or "").lower(),
    )


def event_args(event):
    raw = event.get("arguments", []) if isinstance(event, dict) else []
    return raw if isinstance(raw, list) else []


def sets(payload):
    trig = set()
    trig_text = set()
    args = set()
    arg_role_text = set()
    types = Counter()
    for event in events(payload):
        if not isinstance(event, dict):
            continue
        trig.add(event_key(event))
        trig_text.add(trigger_text_key(event))
        types[event.get("event_type")] += 1
        for arg in event_args(event):
            if isinstance(arg, dict):
                args.add(arg_key(event, arg))
                arg_role_text.add(arg_role_text_key(event, arg))
    return {
        "trig": trig,
        "trig_text": trig_text,
        "args": args,
        "arg_role_text": arg_role_text,
        "types": types,
    }


def case_id(row, fallback_index):
    meta = row.get("meta") or {}
    return f"{meta.get('doc_id','?')}::{meta.get('wnd_id','?')}::{fallback_index}"


def brief_events(payload, limit=3):
    out = []
    for event in events(payload)[:limit]:
        trigger = event.get("trigger") if isinstance(event, dict) else {}
        trigger = trigger if isinstance(trigger, dict) else {}
        args = []
        for arg in event_args(event)[:4]:
            args.append(f"{arg.get('role')}={arg.get('text')}[{arg.get('start')},{arg.get('end')}]")
        out.append(
            {
                "type": event.get("event_type"),
                "trigger": f"{trigger.get('text')}[{trigger.get('start')},{trigger.get('end')}]",
                "args": args,
            }
        )
    return out


def diagnose(row):
    gold = row.get("gold") or {}
    pred = row.get("final_predicted") or {}
    gold_sets = sets(gold)
    pred_sets = sets(pred)
    mention_sets = sets(extract_json_tag(row.get("generated_payload", ""), "EVENT_MENTIONS") or {})
    labels = []

    if not row.get("valid_final_json"):
        labels.append("invalid_final_json")
    if len(events(pred)) > len(events(gold)):
        labels.append("extra_event")
    if len(events(pred)) < len(events(gold)):
        labels.append("missing_event")
    if pred_sets["trig"] - gold_sets["trig"]:
        labels.append("trigger_extra_or_boundary")
    if gold_sets["trig"] - pred_sets["trig"]:
        labels.append("trigger_missing_or_boundary")
    if (pred_sets["trig_text"] & gold_sets["trig_text"]) and not (pred_sets["trig"] & gold_sets["trig"]):
        labels.append("trigger_boundary_shift")
    if set(pred_sets["types"]) - set(gold_sets["types"]):
        labels.append("spurious_event_type")
    if set(gold_sets["types"]) - set(pred_sets["types"]):
        labels.append("missing_event_type")
    if pred_sets["args"] - gold_sets["args"]:
        labels.append("argument_extra_or_boundary")
    if gold_sets["args"] - pred_sets["args"]:
        labels.append("argument_missing_or_boundary")
    if (pred_sets["arg_role_text"] & gold_sets["arg_role_text"]) and not (pred_sets["args"] & gold_sets["args"]):
        labels.append("argument_boundary_shift")
    if mention_sets["trig"] - gold_sets["trig"]:
        labels.append("event_mentions_has_extra_or_boundary")
    if gold_sets["trig"] - mention_sets["trig"]:
        labels.append("event_mentions_misses_gold")
    if mention_sets["trig"] and mention_sets["trig"] != pred_sets["trig"]:
        labels.append("final_drift_from_event_mentions")

    if not labels and row.get("event_f1", 0.0) < 1.0:
        labels.append("event_frame_partial_mismatch")
    if not labels:
        labels.append("correct_or_near_correct")
    return labels


def metric_bucket(row):
    if row.get("event_f1", 0.0) == 1.0:
        return "event_perfect"
    if row.get("trigger_f1", 0.0) == 0.0:
        return "trigger_zero"
    if row.get("argument_f1", 0.0) == 0.0:
        return "argument_zero"
    if row.get("event_f1", 0.0) == 0.0:
        return "event_zero"
    return "partial"


def analyze_file(path):
    rows = load_jsonl(path)
    label_counts = Counter()
    bucket_counts = Counter()
    event_count_delta = Counter()
    type_counts = Counter()
    for idx, row in enumerate(rows):
        label_counts.update(diagnose(row))
        bucket_counts[metric_bucket(row)] += 1
        event_count_delta[len(events(row.get("final_predicted") or {})) - len(events(row.get("gold") or {}))] += 1
        type_counts.update((row.get("meta") or {}).get("gold_event_types") or [])
    return {
        "num_examples": len(rows),
        "label_counts": dict(label_counts.most_common()),
        "bucket_counts": dict(bucket_counts.most_common()),
        "event_count_delta": dict(sorted(event_count_delta.items())),
        "gold_event_type_counts": dict(type_counts.most_common(20)),
    }


def load_variant_budget(variant, budget, split):
    path = ROOT / variant / f"forced_{budget}" / split / "predictions.jsonl"
    if not path.exists():
        return []
    return load_jsonl(path)


def paired_delta(variant, split, left_budget="none", right_budget="standard"):
    none_rows = load_variant_budget(variant, left_budget, split)
    std_rows = load_variant_budget(variant, right_budget, split)
    out = []
    for idx, (none, std) in enumerate(zip(none_rows, std_rows)):
        delta = {m: std.get(m, 0.0) - none.get(m, 0.0) for m in METRICS}
        labels = diagnose(std)
        out.append(
            {
                "id": case_id(std, idx),
                "index": idx,
                "variant": variant,
                "split": split,
                "delta": delta,
                "none": {m: none.get(m, 0.0) for m in METRICS},
                "standard": {m: std.get(m, 0.0) for m in METRICS},
                "labels": labels,
                "gold_event_types": (std.get("meta") or {}).get("gold_event_types") or [],
                "text": (std.get("input") or "").split("Tokens:")[0].replace("Text:\n", "").strip()[:500],
                "gold": brief_events(std.get("gold") or {}),
                "none_pred": brief_events(none.get("final_predicted") or {}),
                "standard_pred": brief_events(std.get("final_predicted") or {}),
                "event_mentions": brief_events(extract_json_tag(std.get("generated_payload", ""), "EVENT_MENTIONS") or {}),
            }
        )
    return out


def fmt(value):
    return f"{value:.4f}"


def render(payload):
    lines = [
        "# E26 Error Case Analysis",
        "",
        "Scope: E26 forced step-decomposed reasoning formal predictions. The key comparison is `standard` vs the same variant's `none` budget.",
        "",
        "## High-Level Finding",
        "",
        "- The dominant failure is not malformed output: expected form and final JSON are essentially valid.",
        "- The reasoning budgets usually hurt by changing the final event content: extra/wrong triggers, event type overprediction, and argument span boundary shifts.",
        "- `EVENT_MENTIONS` is often already wrong or over-complete; later reasoning blocks then preserve or amplify that error instead of correcting it.",
        "- On seen examples, reasoning often keeps high Trigger but damages full Event because exact argument spans/roles must match. On unseen, both trigger and argument errors are more severe.",
        "",
        "## Error Label Counts",
        "",
    ]
    for key, stats in payload["file_stats"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"- examples: `{stats['num_examples']}`")
        lines.append(f"- metric buckets: `{stats['bucket_counts']}`")
        top = list(stats["label_counts"].items())[:10]
        lines.append("- top labels:")
        for label, count in top:
            lines.append(f"  - `{label}`: {count}")
        lines.append("")

    lines.extend(["## Standard vs None Paired Deltas", ""])
    for key, stats in payload["paired_stats"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(
            f"- improved/neutral/worse by A/E/T: "
            f"A `{stats['argument']}`, E `{stats['event']}`, T `{stats['trigger']}`"
        )
        lines.append(f"- top worse-case labels: `{stats['worse_label_counts']}`")
        lines.append("")

    lines.extend(["## Representative Worse Cases", ""])
    for case in payload["representative_cases"]:
        delta = case["delta"]
        lines.append(f"### {case['variant']} {case['split']} idx={case['index']}")
        lines.append("")
        lines.append(f"- delta A/E/T: `{fmt(delta['argument_f1'])} / {fmt(delta['event_f1'])} / {fmt(delta['trigger_f1'])}`")
        lines.append(f"- labels: `{case['labels']}`")
        lines.append(f"- gold types: `{case['gold_event_types']}`")
        lines.append(f"- text: {case['text']}")
        lines.append(f"- gold: `{case['gold']}`")
        lines.append(f"- none pred: `{case['none_pred']}`")
        lines.append(f"- standard event_mentions: `{case['event_mentions']}`")
        lines.append(f"- standard final: `{case['standard_pred']}`")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "1. Trigger detection is a brittle bottleneck. A one-token boundary shift makes Trigger F1 zero for that event and also invalidates all attached arguments and full-event matches.",
            "2. The intermediate `EVENT_MENTIONS` block is not a reliable oracle. When it overpredicts a plausible trigger, the final block tends to copy the false event rather than prune it.",
            "3. Schema/verify text does not add missing evidence. It mostly restates event count, type, or role expectations, so it adds decoding burden without improving exact span grounding.",
            "4. Argument errors are mostly exact-boundary and role attachment errors, not JSON-format errors. This explains why Trigger can improve slightly while Event drops.",
            "5. The seen/unseen gap suggests the model learns dataset/style priors for seen-like texts; the explicit reasoning format does not transfer robustly to unseen event distributions.",
            "",
            "## Next",
            "",
            "- Stop optimizing tag shape alone for this line.",
            "- Analyze whether training should supervise final spans more directly, for example final-only weighted loss, argument-boundary curriculum, or correction-style training.",
            "- If using intermediate steps, make them verifiable and tied to final constraints: e.g. train a correction pass from predicted event mentions to gold final, rather than asking the model to freely generate both.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    file_stats = {}
    paired_stats = {}
    representative = []
    paired_all = []

    for variant in VARIANTS:
        for budget in BUDGETS:
            for split in SPLITS:
                path = ROOT / variant / f"forced_{budget}" / split / "predictions.jsonl"
                if path.exists():
                    file_stats[f"{variant}/{budget}/{split}"] = analyze_file(path)

        for split in SPLITS:
            pairs = paired_delta(variant, split)
            if not pairs:
                continue
            paired_all.extend(pairs)
            stats = {}
            for metric, short in [("argument_f1", "argument"), ("event_f1", "event"), ("trigger_f1", "trigger")]:
                stats[short] = {
                    "improved": sum(1 for row in pairs if row["delta"][metric] > 1e-9),
                    "neutral": sum(1 for row in pairs if abs(row["delta"][metric]) <= 1e-9),
                    "worse": sum(1 for row in pairs if row["delta"][metric] < -1e-9),
                }
            worse_labels = Counter()
            for row in pairs:
                if row["delta"]["event_f1"] < -1e-9 or row["delta"]["argument_f1"] < -1e-9:
                    worse_labels.update(row["labels"])
            stats["worse_label_counts"] = dict(worse_labels.most_common(12))
            paired_stats[f"{variant}/{split}/standard-vs-none"] = stats

    worse = [
        row
        for row in paired_all
        if row["delta"]["event_f1"] < -1e-9 or row["delta"]["argument_f1"] < -1e-9 or row["delta"]["trigger_f1"] < -1e-9
    ]
    worse.sort(key=lambda row: (row["delta"]["event_f1"] + row["delta"]["argument_f1"] + row["delta"]["trigger_f1"]))
    representative = worse[:12]

    payload = {
        "root": ROOT.as_posix(),
        "file_stats": file_stats,
        "paired_stats": paired_stats,
        "representative_cases": representative,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.as_posix(), "md": OUT_MD.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
