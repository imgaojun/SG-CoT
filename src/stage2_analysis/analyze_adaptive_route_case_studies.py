import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROLE_RE = re.compile(r"<ROUTE>\s*(direct|reason)\s*</ROUTE>", re.IGNORECASE)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def event_sets(payload):
    events = payload.get("events", []) if isinstance(payload, dict) else []
    triggers = set()
    args = set()
    events_full = set()
    type_triggers = defaultdict(set)
    roles = Counter()
    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("event_type")
        trigger = event.get("trigger") or {}
        if not isinstance(trigger, dict):
            trigger = {}
        t_start = trigger.get("start")
        t_end = trigger.get("end")
        trig = (etype, t_start, t_end)
        triggers.add(trig)
        type_triggers[etype].add((t_start, t_end))
        arg_items = []
        raw_args = event.get("arguments") or []
        if not isinstance(raw_args, list):
            raw_args = []
        for arg in raw_args:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            item = (etype, t_start, t_end, role, arg.get("start"), arg.get("end"))
            args.add(item)
            roles[role] += 1
            arg_items.append((role, arg.get("start"), arg.get("end")))
        events_full.add(
            (
                etype,
                t_start,
                t_end,
                tuple(sorted(arg_items, key=lambda x: (x[0] or "", -1 if x[1] is None else x[1], -1 if x[2] is None else x[2]))),
            )
        )
    return {
        "triggers": triggers,
        "arguments": args,
        "events": events_full,
        "type_triggers": type_triggers,
        "roles": roles,
    }


def iter_events(payload):
    events = payload.get("events", []) if isinstance(payload, dict) else []
    if not isinstance(events, list):
        return
    for event in events:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") or {}
        if not isinstance(trigger, dict):
            trigger = {}
        raw_args = event.get("arguments") or []
        if not isinstance(raw_args, list):
            raw_args = []
        yield event, trigger, raw_args


def trigger_tuple(event, trigger):
    return (event.get("event_type"), trigger.get("start"), trigger.get("end"))


def argument_items(payload):
    items = []
    for event, trigger, raw_args in iter_events(payload):
        etype, t_start, t_end = trigger_tuple(event, trigger)
        for arg in raw_args:
            if not isinstance(arg, dict):
                continue
            items.append(
                {
                    "event_type": etype,
                    "trigger": trigger.get("text"),
                    "trigger_span": (t_start, t_end),
                    "role": arg.get("role"),
                    "text": arg.get("text"),
                    "span": (arg.get("start"), arg.get("end")),
                    "key": (etype, t_start, t_end, arg.get("role"), arg.get("start"), arg.get("end")),
                }
            )
    return items


def span_distance(a, b):
    if a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return None
    if a[1] <= b[0]:
        return b[0] - a[1]
    if b[1] <= a[0]:
        return a[0] - b[1]
    return 0


def is_near_span(a, b):
    dist = span_distance(a, b)
    return dist is not None and dist <= 1


def categorize_fn_arg(gold_arg, pred_args):
    same_trigger = [p for p in pred_args if p["event_type"] == gold_arg["event_type"] and p["trigger_span"] == gold_arg["trigger_span"]]
    if any(p["role"] == gold_arg["role"] and is_near_span(p["span"], gold_arg["span"]) for p in same_trigger):
        return "span_near_miss"
    if any(p["span"] == gold_arg["span"] and p["role"] != gold_arg["role"] for p in same_trigger):
        return "role_label_mismatch"
    if any(p["role"] == gold_arg["role"] for p in same_trigger):
        return "same_trigger_role_wrong_span"
    if same_trigger:
        return "trigger_correct_role_or_arg_missing"
    if any(p["event_type"] == gold_arg["event_type"] and p["role"] == gold_arg["role"] and p["span"] == gold_arg["span"] for p in pred_args):
        return "trigger_span_mismatch"
    if any(p["role"] == gold_arg["role"] and p["span"] == gold_arg["span"] for p in pred_args):
        return "event_type_or_trigger_mismatch"
    return "missing_argument"


def categorize_fp_arg(pred_arg, gold_args):
    same_trigger = [g for g in gold_args if g["event_type"] == pred_arg["event_type"] and g["trigger_span"] == pred_arg["trigger_span"]]
    if any(g["role"] == pred_arg["role"] and is_near_span(g["span"], pred_arg["span"]) for g in same_trigger):
        return "span_near_miss"
    if any(g["span"] == pred_arg["span"] and g["role"] != pred_arg["role"] for g in same_trigger):
        return "role_label_mismatch"
    if any(g["role"] == pred_arg["role"] for g in same_trigger):
        return "same_trigger_role_wrong_span"
    if same_trigger:
        return "trigger_correct_extra_or_wrong_arg"
    if any(g["event_type"] == pred_arg["event_type"] and g["role"] == pred_arg["role"] and g["span"] == pred_arg["span"] for g in gold_args):
        return "trigger_span_mismatch"
    if any(g["role"] == pred_arg["role"] and g["span"] == pred_arg["span"] for g in gold_args):
        return "event_type_or_trigger_mismatch"
    return "spurious_argument"


def argument_error_breakdown(rows):
    fn_categories = Counter()
    fp_categories = Counter()
    fn_roles = Counter()
    fp_roles = Counter()
    examples = defaultdict(list)
    for row in rows:
        key = row_key(row, len(examples))
        gold_args = argument_items(row.get("gold") or {})
        pred_args = argument_items(row.get("predicted") or {})
        pred_keys = {item["key"] for item in pred_args}
        gold_keys = {item["key"] for item in gold_args}
        for gold_arg in gold_args:
            if gold_arg["key"] in pred_keys:
                continue
            category = categorize_fn_arg(gold_arg, pred_args)
            fn_categories[category] += 1
            fn_roles[gold_arg["role"]] += 1
            if len(examples[category]) < 3:
                examples[category].append(
                    {
                        "id": key,
                        "role": gold_arg["role"],
                        "text": gold_arg["text"],
                        "span": list(gold_arg["span"]),
                        "event_type": gold_arg["event_type"],
                    }
                )
        for pred_arg in pred_args:
            if pred_arg["key"] in gold_keys:
                continue
            category = categorize_fp_arg(pred_arg, gold_args)
            fp_categories[category] += 1
            fp_roles[pred_arg["role"]] += 1
    return {
        "fn_categories": fn_categories.most_common(),
        "fp_categories": fp_categories.most_common(),
        "fn_roles": fn_roles.most_common(12),
        "fp_roles": fp_roles.most_common(12),
        "examples": dict(examples),
    }


def row_argument_diagnosis(row):
    payload = argument_error_breakdown([row])
    return {
        "fn_categories": payload["fn_categories"][:5],
        "fp_categories": payload["fp_categories"][:5],
        "fn_roles": payload["fn_roles"][:5],
        "fp_roles": payload["fp_roles"][:5],
    }


def event_type_confusions(rows):
    wrong_type_pairs = Counter()
    same_type_wrong_trigger = Counter()
    missed_types = Counter()
    spurious_types = Counter()
    for row in rows:
        gold = event_sets(row.get("gold") or {})
        pred = event_sets(row.get("predicted") or {})
        gold_triggers = gold["triggers"]
        pred_triggers = pred["triggers"]
        gold_types = {item[0] for item in gold_triggers}
        pred_types = {item[0] for item in pred_triggers}
        for gtype in gold_types - pred_types:
            missed_types[gtype] += 1
        for ptype in pred_types - gold_types:
            spurious_types[ptype] += 1
        if gold_types and pred_types and not (gold_types & pred_types):
            for gtype in gold_types:
                for ptype in pred_types:
                    wrong_type_pairs[(gtype, ptype)] += 1
        for gtype in gold_types & pred_types:
            if not (gold["type_triggers"][gtype] & pred["type_triggers"][gtype]):
                same_type_wrong_trigger[gtype] += 1
    return {
        "wrong_type_pairs": [(f"{g} -> {p}", c) for (g, p), c in wrong_type_pairs.most_common(12)],
        "same_type_wrong_trigger": same_type_wrong_trigger.most_common(12),
        "missed_types": missed_types.most_common(12),
        "spurious_types": spurious_types.most_common(12),
    }


def prf(pred, gold):
    if not pred and not gold:
        return {"p": 1.0, "r": 1.0, "f1": 1.0}
    if not pred or not gold:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {"p": p, "r": r, "f1": f1}


def row_key(row, idx):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or meta.get("doc_id") or str(idx)


def extract_text_block(input_text):
    marker = "Text:\n"
    if marker not in input_text:
        return input_text[:500]
    rest = input_text.split(marker, 1)[1]
    if "\n\nTokens:" in rest:
        return rest.split("\n\nTokens:", 1)[0]
    return rest[:500]


def compact_events(payload):
    events = payload.get("events", []) if isinstance(payload, dict) else []
    out = []
    for event in events:
        trig = event.get("trigger") or {}
        args = []
        for arg in event.get("arguments") or []:
            args.append(
                {
                    "role": arg.get("role"),
                    "text": arg.get("text"),
                    "span": [arg.get("start"), arg.get("end")],
                }
            )
        out.append(
            {
                "type": event.get("event_type"),
                "trigger": trig.get("text"),
                "trigger_span": [trig.get("start"), trig.get("end")],
                "args": args,
            }
        )
    return out


def extract_reason(generated_payload):
    start = generated_payload.find("<REASON>")
    end = generated_payload.find("</REASON>")
    if start == -1:
        return ""
    if end == -1:
        return generated_payload[start : start + 700]
    return generated_payload[start + len("<REASON>") : end][:700]


def prompt_contamination(row):
    payload = row.get("generated_payload") or ""
    assistant_idx = payload.rfind("assistant")
    first_route = payload.find("<ROUTE>")
    first_final = payload.find("<FINAL>")
    route_before_assistant = assistant_idx != -1 and first_route != -1 and first_route < assistant_idx
    final_before_assistant = assistant_idx != -1 and first_final != -1 and first_final < assistant_idx
    prefix = payload[: first_route if first_route != -1 else min(len(payload), 300)]
    return {
        "has_assistant_marker": assistant_idx != -1,
        "prefix_before_route_chars": len(prefix.strip()),
        "route_tags": len(ROLE_RE.findall(payload)),
        "route_before_assistant": route_before_assistant,
        "final_before_assistant": final_before_assistant,
        "prefix_excerpt": prefix.strip()[:240],
    }


def classify_row(row):
    gold_sets = event_sets(row.get("gold") or {})
    pred_sets = event_sets(row.get("predicted") or {})
    trig = prf(pred_sets["triggers"], gold_sets["triggers"])["f1"]
    arg = prf(pred_sets["arguments"], gold_sets["arguments"])["f1"]
    event = prf(pred_sets["events"], gold_sets["events"])["f1"]
    if event >= 0.999:
        return "event_exact"
    if trig > 0 and arg > 0:
        return "trigger_and_partial_args"
    if trig > 0 and arg == 0:
        return "trigger_only_no_args"
    if pred_sets["triggers"] and gold_sets["triggers"]:
        gold_types = {x[0] for x in gold_sets["triggers"]}
        pred_types = {x[0] for x in pred_sets["triggers"]}
        if gold_types & pred_types:
            return "same_type_wrong_trigger"
        return "wrong_event_type"
    if not pred_sets["triggers"] and gold_sets["triggers"]:
        return "missed_event"
    if pred_sets["triggers"] and not gold_sets["triggers"]:
        return "spurious_event"
    return "other"


def role_delta(rows_by_mode, split):
    out = {}
    for mode, rows in rows_by_mode.items():
        fn = Counter()
        fp = Counter()
        for row in rows:
            gold = event_sets(row.get("gold") or {})
            pred = event_sets(row.get("predicted") or {})
            for item in gold["arguments"] - pred["arguments"]:
                fn[item[3]] += 1
            for item in pred["arguments"] - gold["arguments"]:
                fp[item[3]] += 1
        out[mode] = {
            "top_fn_roles": fn.most_common(12),
            "top_fp_roles": fp.most_common(12),
        }
    return out


def better(a, b, key, eps=1e-9):
    return a.get(key, 0.0) > b.get(key, 0.0) + eps


def pairwise(rows_a, rows_b, key):
    wins = ties = losses = 0
    deltas = []
    for ra, rb in zip(rows_a, rows_b):
        da = ra.get(key, 0.0)
        db = rb.get(key, 0.0)
        deltas.append(da - db)
        if da > db + 1e-9:
            wins += 1
        elif db > da + 1e-9:
            losses += 1
        else:
            ties += 1
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "avg_delta": sum(deltas) / len(deltas) if deltas else 0.0,
    }


def build_case(label, split, key, rows):
    free = rows["free_route"][key]
    direct = rows["forced_direct"][key]
    reason = rows["forced_reason"][key]
    meta = free.get("meta") or {}
    return {
        "label": label,
        "split": split,
        "id": key,
        "text": extract_text_block(free.get("input", "")),
        "candidate_types": meta.get("candidate_types"),
        "gold_event_types": meta.get("gold_event_types"),
        "heuristic_label": meta.get("adaptive_route_label"),
        "free_route_pred": free.get("route_pred"),
        "free_metrics": {k: free.get(k) for k in ["trigger_f1", "argument_f1", "event_f1"]},
        "direct_metrics": {k: direct.get(k) for k in ["trigger_f1", "argument_f1", "event_f1"]},
        "reason_metrics": {k: reason.get(k) for k in ["trigger_f1", "argument_f1", "event_f1"]},
        "gold": compact_events(free.get("gold") or {}),
        "free_pred": compact_events(free.get("predicted") or {}),
        "direct_pred": compact_events(direct.get("predicted") or {}),
        "reason_pred": compact_events(reason.get("predicted") or {}),
        "free_diagnosis": row_argument_diagnosis(free),
        "direct_diagnosis": row_argument_diagnosis(direct),
        "reason_diagnosis": row_argument_diagnosis(reason),
        "free_reason": extract_reason(free.get("generated_payload") or ""),
        "forced_reason": extract_reason(reason.get("generated_payload") or ""),
        "contamination": prompt_contamination(free),
    }


def choose_cases(aligned):
    cases = []
    for split, rows in aligned.items():
        keys = list(rows["free_route"].keys())
        candidates = []
        for key in keys:
            f = rows["free_route"][key]
            d = rows["forced_direct"][key]
            r = rows["forced_reason"][key]
            candidates.append((key, f, d, r))

        # Free chooses reason, but direct would have been better.
        bad_reason = [
            (key, f, d, r)
            for key, f, d, r in candidates
            if f.get("route_pred") == "reason" and d.get("event_f1", 0.0) > f.get("event_f1", 0.0)
        ]
        bad_reason.sort(key=lambda x: (x[2].get("event_f1", 0) - x[1].get("event_f1", 0), x[2].get("argument_f1", 0) - x[1].get("argument_f1", 0)), reverse=True)
        if bad_reason:
            cases.append(build_case(f"{split}: reason route hurts vs direct", split, bad_reason[0][0], rows))

        # Free/reason helps over direct.
        helps = [
            (key, f, d, r)
            for key, f, d, r in candidates
            if f.get("route_pred") == "reason" and f.get("argument_f1", 0.0) > d.get("argument_f1", 0.0)
        ]
        helps.sort(key=lambda x: (x[1].get("argument_f1", 0) - x[2].get("argument_f1", 0), x[1].get("event_f1", 0) - x[2].get("event_f1", 0)), reverse=True)
        if helps:
            cases.append(build_case(f"{split}: reason route helps arguments", split, helps[0][0], rows))

        # Forced reason drops roles while trigger is still right.
        role_drop = [
            (key, f, d, r)
            for key, f, d, r in candidates
            if d.get("trigger_f1", 0) > 0 and r.get("trigger_f1", 0) > 0 and d.get("argument_f1", 0) > r.get("argument_f1", 0)
        ]
        role_drop.sort(key=lambda x: x[2].get("argument_f1", 0) - x[3].get("argument_f1", 0), reverse=True)
        if role_drop:
            cases.append(build_case(f"{split}: forced reason loses roles", split, role_drop[0][0], rows))

        # Free route is direct, but forced reason would have been better.
        under_reason = [
            (key, f, d, r)
            for key, f, d, r in candidates
            if f.get("route_pred") == "direct" and r.get("argument_f1", 0.0) > f.get("argument_f1", 0.0)
        ]
        under_reason.sort(key=lambda x: x[3].get("argument_f1", 0) - x[1].get("argument_f1", 0), reverse=True)
        if under_reason:
            cases.append(build_case(f"{split}: missed opportunity to reason", split, under_reason[0][0], rows))
    return cases


def format_float(x):
    if x is None:
        return "-"
    return f"{x:.4f}"


def markdown_report(result):
    lines = []
    lines.append("# Adaptive Route Case Study Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- run: `{result['run_name']}`")
    lines.append("- compared modes: `free_route`, `forced_direct`, `forced_reason`")
    lines.append("- primary question: why adaptive free-route underperforms forced direct despite strong dev selection")
    lines.append("")
    lines.append("## Formal Metrics")
    lines.append("")
    lines.append("| mode | split | json | reason_rate | trigger_f1 | argument_f1 | event_f1 | latency |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in result["summary_rows"]:
        lines.append(
            f"| `{row['mode']}` | `{row['split']}` | {format_float(row['json_valid_rate'])} | {format_float(row['route_reason_rate'])} | {format_float(row['trigger_f1'])} | {format_float(row['argument_f1'])} | {format_float(row['event_f1'])} | {format_float(row['avg_latency_sec'])} |"
        )
    lines.append("")
    lines.append("## Pairwise Attribution")
    lines.append("")
    lines.append("| split | comparison | metric | wins | ties | losses | avg_delta |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for split, split_rows in result["pairwise"].items():
        for comparison, metrics in split_rows.items():
            for metric, payload in metrics.items():
                lines.append(
                    f"| `{split}` | `{comparison}` | `{metric}` | {payload['wins']} | {payload['ties']} | {payload['losses']} | {format_float(payload['avg_delta'])} |"
                )
    lines.append("")
    lines.append("## Route Behavior")
    lines.append("")
    lines.append("| split | pred_route | count | avg_arg | avg_event | heuristic_direct | heuristic_reason |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for split, route_rows in result["route_behavior"].items():
        for route, payload in route_rows.items():
            lines.append(
                f"| `{split}` | `{route}` | {payload['count']} | {format_float(payload['avg_argument_f1'])} | {format_float(payload['avg_event_f1'])} | {payload['heuristic_direct']} | {payload['heuristic_reason']} |"
            )
    lines.append("")
    lines.append("## Route Oracle Gap")
    lines.append("")
    lines.append("| split | free_reason | oracle_reason | route_match | harmful_reason | missed_reason | direct_better | reason_better | tie |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split, payload in result["route_oracle_gap"].items():
        lines.append(
            f"| `{split}` | {payload['free_reason']} | {payload['oracle_reason']} | {payload['route_match']} | {payload['harmful_reason']} | {payload['missed_reason']} | {payload['direct_better']} | {payload['reason_better']} | {payload['tie']} |"
        )
    lines.append("")
    lines.append("## Error Class Distribution")
    lines.append("")
    lines.append("| split | mode | top classes |")
    lines.append("|---|---|---|")
    for split, split_rows in result["error_classes"].items():
        for mode, classes in split_rows.items():
            top = ", ".join(f"{name}:{count}" for name, count in classes[:6])
            lines.append(f"| `{split}` | `{mode}` | {top} |")
    lines.append("")
    lines.append("## Role FN/FP")
    lines.append("")
    for split, split_rows in result["role_delta"].items():
        lines.append(f"### {split}")
        lines.append("")
        for mode, payload in split_rows.items():
            fn = ", ".join(f"{r}:{c}" for r, c in payload["top_fn_roles"][:8])
            fp = ", ".join(f"{r}:{c}" for r, c in payload["top_fp_roles"][:8])
            lines.append(f"- `{mode}` FN: {fn}")
            lines.append(f"- `{mode}` FP: {fp}")
        lines.append("")
    lines.append("## Argument Error Breakdown")
    lines.append("")
    lines.append("| split | mode | FN categories | FP categories |")
    lines.append("|---|---|---|---|")
    for split, split_rows in result["argument_error_breakdown"].items():
        for mode, payload in split_rows.items():
            fn = ", ".join(f"{name}:{count}" for name, count in payload["fn_categories"][:6])
            fp = ", ".join(f"{name}:{count}" for name, count in payload["fp_categories"][:6])
            lines.append(f"| `{split}` | `{mode}` | {fn} | {fp} |")
    lines.append("")
    lines.append("## Event-Type Confusions")
    lines.append("")
    lines.append("| split | mode | wrong type pairs | missed types | spurious types |")
    lines.append("|---|---|---|---|---|")
    for split, split_rows in result["event_type_confusions"].items():
        for mode, payload in split_rows.items():
            pairs = ", ".join(f"{name}:{count}" for name, count in payload["wrong_type_pairs"][:6])
            missed = ", ".join(f"{name}:{count}" for name, count in payload["missed_types"][:6])
            spurious = ", ".join(f"{name}:{count}" for name, count in payload["spurious_types"][:6])
            lines.append(f"| `{split}` | `{mode}` | {pairs} | {missed} | {spurious} |")
    lines.append("")
    lines.append("## Prompt Tail Contamination Check")
    lines.append("")
    lines.append("| split | mode | rows | assistant_marker | prefix_before_route | route_before_assistant | final_before_assistant | multi_route_tags |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for split, split_rows in result["contamination"].items():
        for mode, payload in split_rows.items():
            lines.append(
                f"| `{split}` | `{mode}` | {payload['rows']} | {payload['assistant_marker']} | {payload['prefix_before_route']} | {payload['route_before_assistant']} | {payload['final_before_assistant']} | {payload['multi_route_tags']} |"
            )
    lines.append("")
    if result.get("clean_generation"):
        lines.append("Interpretation: decoded payloads are clean after slicing generation by padded input width. Route/case-study fields can be trusted for this run.")
    else:
        lines.append("Interpretation: decoded outputs contain prompt tail before the first route tag. Route/case-study fields should be regenerated after fixing left-padding slicing.")
    lines.append("")
    lines.append("## Case Studies")
    for idx, case in enumerate(result["cases"], 1):
        lines.append("")
        lines.append(f"### Case {idx}: {case['label']}")
        lines.append("")
        lines.append(f"- split: `{case['split']}`")
        lines.append(f"- id: `{case['id']}`")
        lines.append(f"- heuristic label: `{case.get('heuristic_label')}`; free predicted route: `{case.get('free_route_pred')}`")
        lines.append(f"- gold event types: `{case.get('gold_event_types')}`")
        lines.append(f"- text: {case['text'][:700]}")
        lines.append(f"- free metrics: `{case['free_metrics']}`")
        lines.append(f"- direct metrics: `{case['direct_metrics']}`")
        lines.append(f"- reason metrics: `{case['reason_metrics']}`")
        lines.append(f"- free diagnosis: `{case['free_diagnosis']}`")
        lines.append(f"- direct diagnosis: `{case['direct_diagnosis']}`")
        lines.append(f"- reason diagnosis: `{case['reason_diagnosis']}`")
        lines.append("")
        lines.append("Gold:")
        lines.append("```json")
        lines.append(json.dumps(case["gold"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("Free prediction:")
        lines.append("```json")
        lines.append(json.dumps(case["free_pred"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("Forced direct prediction:")
        lines.append("```json")
        lines.append(json.dumps(case["direct_pred"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("Forced reason prediction:")
        lines.append("```json")
        lines.append(json.dumps(case["reason_pred"], ensure_ascii=False, indent=2))
        lines.append("```")
        if case.get("free_reason"):
            lines.append("Free reason:")
            lines.append("```json")
            lines.append(case["free_reason"])
            lines.append("```")
        if case.get("forced_reason") and case.get("forced_reason") != case.get("free_reason"):
            lines.append("Forced reason:")
            lines.append("```json")
            lines.append(case["forced_reason"])
            lines.append("```")
    lines.append("")
    lines.append("## Takeaways")
    lines.append("")
    for item in result["takeaways"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run_name", default="adaptive_confrare10_heur10_typeonlylite")
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    modes = ["free_route", "forced_direct", "forced_reason"]
    splits = ["test", "test_seen", "test_unseen"]

    summaries = {}
    raw_rows = defaultdict(dict)
    aligned = defaultdict(lambda: defaultdict(dict))
    summary_rows = []
    for mode in modes:
        for split in splits:
            summary_path = root / mode / split / "summary.json"
            pred_path = root / mode / split / "predictions.jsonl"
            summary = load_json(summary_path)
            rows = load_jsonl(pred_path)
            summaries[(mode, split)] = summary
            raw_rows[split][mode] = rows
            summary_rows.append(
                {
                    "mode": mode,
                    "split": split,
                    "json_valid_rate": summary.get("json_valid_rate"),
                    "route_reason_rate": summary.get("route_reason_rate"),
                    "trigger_f1": summary.get("trigger_f1"),
                    "argument_f1": summary.get("argument_f1"),
                    "event_f1": summary.get("event_f1"),
                    "avg_latency_sec": summary.get("avg_latency_sec"),
                }
            )
            for idx, row in enumerate(rows):
                aligned[split][mode][row_key(row, idx)] = row

    # Keep common sample order per split.
    for split in splits:
        common = [set(aligned[split][m].keys()) for m in modes]
        common_keys = set.intersection(*common)
        for mode in modes:
            aligned[split][mode] = {k: aligned[split][mode][k] for k in common_keys}

    pairwise_result = {}
    for split in splits:
        pairwise_result[split] = {}
        for comparison, left, right in [
            ("free_minus_direct", "free_route", "forced_direct"),
            ("free_minus_reason", "free_route", "forced_reason"),
            ("reason_minus_direct", "forced_reason", "forced_direct"),
        ]:
            keys = sorted(aligned[split][left])
            rows_left = [aligned[split][left][k] for k in keys]
            rows_right = [aligned[split][right][k] for k in keys]
            pairwise_result[split][comparison] = {
                metric: pairwise(rows_left, rows_right, metric)
                for metric in ["trigger_f1", "argument_f1", "event_f1"]
            }

    route_behavior = {}
    for split in splits:
        route_behavior[split] = {}
        grouped = defaultdict(list)
        for row in aligned[split]["free_route"].values():
            grouped[row.get("route_pred", "unknown")].append(row)
        for route, rows in grouped.items():
            labels = Counter((row.get("meta") or {}).get("adaptive_route_label", "unknown") for row in rows)
            route_behavior[split][route] = {
                "count": len(rows),
                "avg_argument_f1": sum(row.get("argument_f1", 0.0) for row in rows) / len(rows),
                "avg_event_f1": sum(row.get("event_f1", 0.0) for row in rows) / len(rows),
                "heuristic_direct": labels.get("direct", 0),
                "heuristic_reason": labels.get("reason", 0),
                "heuristic_unknown": labels.get("unknown", 0),
            }

    error_classes = {}
    for split in splits:
        error_classes[split] = {}
        for mode in modes:
            counts = Counter(classify_row(row) for row in aligned[split][mode].values())
            error_classes[split][mode] = counts.most_common()

    role_delta_result = {split: role_delta(raw_rows[split], split) for split in splits}

    route_oracle_gap = {}
    for split in splits:
        stats = Counter()
        for key in sorted(aligned[split]["free_route"]):
            free = aligned[split]["free_route"][key]
            direct = aligned[split]["forced_direct"][key]
            reason = aligned[split]["forced_reason"][key]
            direct_score = (direct.get("event_f1", 0.0), direct.get("argument_f1", 0.0), direct.get("trigger_f1", 0.0))
            reason_score = (reason.get("event_f1", 0.0), reason.get("argument_f1", 0.0), reason.get("trigger_f1", 0.0))
            free_route = free.get("route_pred", "unknown")
            if reason_score > direct_score:
                oracle_route = "reason"
                stats["reason_better"] += 1
            elif direct_score > reason_score:
                oracle_route = "direct"
                stats["direct_better"] += 1
            else:
                oracle_route = "direct"
                stats["tie"] += 1
            stats["free_reason"] += int(free_route == "reason")
            stats["oracle_reason"] += int(oracle_route == "reason")
            stats["route_match"] += int(free_route == oracle_route)
            stats["harmful_reason"] += int(free_route == "reason" and oracle_route == "direct" and direct_score > reason_score)
            stats["missed_reason"] += int(free_route != "reason" and oracle_route == "reason")
        route_oracle_gap[split] = dict(stats)

    argument_error_result = {}
    event_type_confusion_result = {}
    for split in splits:
        argument_error_result[split] = {}
        event_type_confusion_result[split] = {}
        for mode in modes:
            rows = list(aligned[split][mode].values())
            argument_error_result[split][mode] = argument_error_breakdown(rows)
            event_type_confusion_result[split][mode] = event_type_confusions(rows)

    contamination = {}
    for split in splits:
        contamination[split] = {}
        for mode in modes:
            rows = raw_rows[split][mode]
            stats = Counter()
            for row in rows:
                c = prompt_contamination(row)
                stats["rows"] += 1
                stats["assistant_marker"] += int(c["has_assistant_marker"])
                stats["prefix_before_route"] += int(c["prefix_before_route_chars"] > 0)
                stats["route_before_assistant"] += int(c["route_before_assistant"])
                stats["final_before_assistant"] += int(c["final_before_assistant"])
                stats["multi_route_tags"] += int(c["route_tags"] > 1)
            contamination[split][mode] = dict(stats)

    clean_generation = all(
        payload.get("assistant_marker", 0) == 0 and payload.get("prefix_before_route", 0) == 0 and payload.get("multi_route_tags", 0) == 0
        for split_rows in contamination.values()
        for payload in split_rows.values()
    )

    cases = choose_cases(aligned)

    takeaways = [
        "`forced_reason` is weaker than `forced_direct` on every formal split, so the current test-time reasoning content is not a useful expert yet.",
        "`free_route` routes more samples to reason on `test_unseen`, but those routed samples do not recover enough arguments/events; this explains the unseen drop.",
        "The common failure pattern is not JSON validity; it is semantic extraction after a valid final JSON, especially missing roles after the correct trigger.",
        "The type-only reason target can pick plausible event-type contrasts, but it does not supervise argument grounding, so final outputs often omit Place/Agent/role spans.",
        "The useful next diagnostic is to make the reason path itself stronger before spending more effort on the router, because per-sample oracle routing cannot help if forced reasoning is usually weaker than direct.",
    ]

    result = {
        "run_name": args.run_name,
        "root": root.as_posix(),
        "summary_rows": summary_rows,
        "pairwise": pairwise_result,
        "route_behavior": route_behavior,
        "route_oracle_gap": route_oracle_gap,
        "error_classes": error_classes,
        "role_delta": role_delta_result,
        "argument_error_breakdown": argument_error_result,
        "event_type_confusions": event_type_confusion_result,
        "contamination": contamination,
        "clean_generation": clean_generation,
        "cases": cases,
        "takeaways": takeaways,
    }
    write_json(Path(args.output_json), result)
    write_text(Path(args.output_md), markdown_report(result))


if __name__ == "__main__":
    main()
