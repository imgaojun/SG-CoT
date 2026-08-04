import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_route_case_studies import (  # noqa: E402
    argument_error_breakdown,
    compact_events,
    event_type_confusions,
    extract_text_block,
    row_argument_diagnosis,
    row_key,
)


MODES = ["free_route", "forced_direct", "forced_reason"]
SPLITS = ["test", "test_seen", "test_unseen"]

BRANCHES = [
    {
        "name": "confrare10_heur10_type_role_verify_lite",
        "run_dir": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_role_verify_lite",
        "label_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10",
    },
    {
        "name": "roleconf10_heur10_type_role_verify_lite",
        "run_dir": "richere_split1_qwen3_1_7b_adaptive_roleconf10_heur10_type_role_verify_lite",
        "label_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_roleconf10_heur10",
    },
    {
        "name": "confrare5_heur5_type_role_verify_lite",
        "run_dir": "richere_split1_qwen3_1_7b_adaptive_confrare5_heur5_type_role_verify_lite",
        "label_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare5_heur5",
    },
    {
        "name": "roleconf5_heur5_type_role_verify_lite",
        "run_dir": "richere_split1_qwen3_1_7b_adaptive_roleconf5_heur5_type_role_verify_lite",
        "label_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_roleconf5_heur5",
    },
]

REASON_RE = re.compile(r"<REASON>\s*(.*?)\s*</REASON>", re.DOTALL | re.IGNORECASE)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_float(value):
    if value is None:
        return 0.0
    return float(value)


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def reason_text(payload):
    match = REASON_RE.search(payload or "")
    return match.group(1).strip() if match else ""


def parse_reason_schema(row):
    text = reason_text(row.get("generated_payload") or row.get("generated_text") or "")
    out = {
        "has_reason_block": bool(text),
        "reason_parse_ok": False,
        "reason_empty": not bool(text),
        "has_events_key": False,
        "has_decisions_key": False,
        "has_type_decisions_key": False,
        "has_role_checks_key": False,
        "role_checks_count": 0,
        "role_checks_with_status": 0,
        "role_checks_present": 0,
        "role_checks_absent": 0,
        "reason_keys": [],
    }
    if not text:
        return out
    try:
        payload = json.loads(text)
    except Exception:
        return out
    if not isinstance(payload, dict):
        out["reason_parse_ok"] = True
        return out
    out["reason_parse_ok"] = True
    keys = sorted(str(k) for k in payload.keys())
    out["reason_keys"] = keys
    out["has_events_key"] = "events" in payload
    out["has_decisions_key"] = "decisions" in payload
    out["has_type_decisions_key"] = "type_decisions" in payload
    out["has_role_checks_key"] = "role_checks" in payload
    role_checks = payload.get("role_checks")
    if isinstance(role_checks, list):
        out["role_checks_count"] = len(role_checks)
        for item in role_checks:
            if not isinstance(item, dict):
                continue
            status = item.get("status")
            if status is not None:
                out["role_checks_with_status"] += 1
            if status == "present":
                out["role_checks_present"] += 1
            elif status == "absent":
                out["role_checks_absent"] += 1
    return out


def load_labels(label_dir, label_prefix, split):
    path = Path(label_dir) / f"{label_prefix}_{split}_labels.jsonl"
    rows = load_jsonl(path)
    return {row["wnd_id"]: row for row in rows}


def load_branch(root_base, branch, label_dir):
    root = Path(root_base) / branch["run_dir"]
    result = {
        "name": branch["name"],
        "run_dir": root.as_posix(),
        "summaries": {},
        "rows": defaultdict(dict),
        "aligned": defaultdict(lambda: defaultdict(dict)),
        "labels": {},
    }
    for split in SPLITS:
        result["labels"][split] = load_labels(label_dir, branch["label_prefix"], split)
        for mode in MODES:
            summary_path = root / mode / split / "summary.json"
            pred_path = root / mode / split / "predictions.jsonl"
            summary = load_json(summary_path)
            rows = load_jsonl(pred_path)
            result["summaries"][(mode, split)] = summary
            result["rows"][split][mode] = rows
            for idx, row in enumerate(rows):
                result["aligned"][split][mode][row_key(row, idx)] = row
    for split in SPLITS:
        common = set.intersection(*(set(result["aligned"][split][mode]) for mode in MODES))
        for mode in MODES:
            result["aligned"][split][mode] = {key: result["aligned"][split][mode][key] for key in common}
    return result


def pairwise(rows_a, rows_b, metric):
    wins = ties = losses = 0
    deltas = []
    for left, right in zip(rows_a, rows_b):
        delta = safe_float(left.get(metric)) - safe_float(right.get(metric))
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "row_avg_delta": sum(deltas) / len(deltas) if deltas else 0.0,
    }


def aggregate_rows(rows):
    if not rows:
        return {"n": 0, "trigger_row_avg": 0.0, "argument_row_avg": 0.0, "event_row_avg": 0.0}
    return {
        "n": len(rows),
        "trigger_row_avg": sum(safe_float(row.get("trigger_f1")) for row in rows) / len(rows),
        "argument_row_avg": sum(safe_float(row.get("argument_f1")) for row in rows) / len(rows),
        "event_row_avg": sum(safe_float(row.get("event_f1")) for row in rows) / len(rows),
    }


def group_name(label):
    components = (label or {}).get("score_components") or {}
    route_label = (label or {}).get("route_label", "unknown")
    confusion = components.get("confusion_norm")
    role_rarity = components.get("role_signature_rarity")
    names = [f"label_{route_label}"]
    if confusion is not None:
        names.append("high_confusion" if confusion >= 0.75 else "low_mid_confusion")
    if role_rarity is not None:
        names.append("high_role_rarity" if role_rarity >= 0.75 else "low_mid_role_rarity")
    return names


def bucket_analysis(branch):
    output = {}
    for split in SPLITS:
        labels = branch["labels"][split]
        split_out = {}
        for bucket in [
            "label_reason",
            "label_direct",
            "high_confusion",
            "low_mid_confusion",
            "high_role_rarity",
            "low_mid_role_rarity",
        ]:
            keys = []
            for key in branch["aligned"][split]["forced_direct"]:
                label = labels.get(key, {})
                if bucket in group_name(label):
                    keys.append(key)
            direct_rows = [branch["aligned"][split]["forced_direct"][key] for key in keys]
            reason_rows = [branch["aligned"][split]["forced_reason"][key] for key in keys]
            free_rows = [branch["aligned"][split]["free_route"][key] for key in keys]
            split_out[bucket] = {
                "direct": aggregate_rows(direct_rows),
                "reason": aggregate_rows(reason_rows),
                "free": aggregate_rows(free_rows),
                "reason_minus_direct_argument_row_avg": (
                    aggregate_rows(reason_rows)["argument_row_avg"] - aggregate_rows(direct_rows)["argument_row_avg"]
                    if keys
                    else 0.0
                ),
                "reason_minus_direct_event_row_avg": (
                    aggregate_rows(reason_rows)["event_row_avg"] - aggregate_rows(direct_rows)["event_row_avg"]
                    if keys
                    else 0.0
                ),
            }
        output[split] = split_out
    return output


def route_behavior(branch):
    out = {}
    for split in SPLITS:
        rows = list(branch["aligned"][split]["free_route"].values())
        labels = branch["labels"][split]
        pred_counts = Counter(row.get("route_pred", "unknown") for row in rows)
        label_counts = Counter((labels.get(row_key(row, idx)) or (row.get("meta") or {})).get("route_label") or (row.get("meta") or {}).get("adaptive_route_label", "unknown") for idx, row in enumerate(rows))
        reason_labeled = 0
        reason_labeled_routed_reason = 0
        direct_labeled_routed_reason = 0
        for idx, row in enumerate(rows):
            key = row_key(row, idx)
            label = (labels.get(key) or {}).get("route_label") or (row.get("meta") or {}).get("adaptive_route_label", "unknown")
            if label == "reason":
                reason_labeled += 1
                reason_labeled_routed_reason += int(row.get("route_pred") == "reason")
            elif label == "direct":
                direct_labeled_routed_reason += int(row.get("route_pred") == "reason")
        out[split] = {
            "num_examples": len(rows),
            "pred_route_counts": dict(pred_counts),
            "label_counts": dict(label_counts),
            "reason_labeled": reason_labeled,
            "reason_labeled_routed_reason": reason_labeled_routed_reason,
            "reason_labeled_routed_reason_rate": reason_labeled_routed_reason / reason_labeled if reason_labeled else 0.0,
            "direct_labeled_routed_reason": direct_labeled_routed_reason,
        }
    return out


def reason_schema_stats(branch):
    out = {}
    for split in SPLITS:
        rows = list(branch["aligned"][split]["forced_reason"].values())
        stats = Counter()
        key_counter = Counter()
        examples = []
        for row in rows:
            parsed = parse_reason_schema(row)
            for key, value in parsed.items():
                if key == "reason_keys":
                    key_counter[tuple(value)] += 1
                elif isinstance(value, bool):
                    stats[key] += int(value)
                elif isinstance(value, int):
                    stats[key] += value
            if (parsed["has_events_key"] or not parsed["reason_parse_ok"] or not row.get("valid_json", False)) and len(examples) < 4:
                examples.append(
                    {
                        "id": (row.get("meta") or {}).get("wnd_id"),
                        "valid_json": row.get("valid_json"),
                        "reason_keys": parsed["reason_keys"],
                        "reason_excerpt": reason_text(row.get("generated_payload") or "")[:500],
                    }
                )
        total = len(rows)
        out[split] = {
            "num_examples": total,
            "reason_parse_ok_rate": stats["reason_parse_ok"] / total if total else 0.0,
            "has_events_key_rate": stats["has_events_key"] / total if total else 0.0,
            "has_type_decisions_key_rate": stats["has_type_decisions_key"] / total if total else 0.0,
            "has_role_checks_key_rate": stats["has_role_checks_key"] / total if total else 0.0,
            "avg_role_checks_count": stats["role_checks_count"] / total if total else 0.0,
            "avg_role_checks_with_status": stats["role_checks_with_status"] / total if total else 0.0,
            "avg_absent_checks": stats["role_checks_absent"] / total if total else 0.0,
            "top_reason_key_sets": [(list(keys), count) for keys, count in key_counter.most_common(8)],
            "examples": examples,
        }
    return out


def trigger_correct_arg_wrong(rows):
    count = 0
    for row in rows:
        if safe_float(row.get("trigger_f1")) > 0.0 and safe_float(row.get("argument_f1")) < safe_float(row.get("trigger_f1")):
            count += 1
    return count


def pairwise_and_errors(branch):
    out = {}
    for split in SPLITS:
        keys = sorted(branch["aligned"][split]["forced_direct"])
        direct_rows = [branch["aligned"][split]["forced_direct"][key] for key in keys]
        reason_rows = [branch["aligned"][split]["forced_reason"][key] for key in keys]
        free_rows = [branch["aligned"][split]["free_route"][key] for key in keys]
        out[split] = {
            "reason_minus_direct": {
                metric: pairwise(reason_rows, direct_rows, metric)
                for metric in ["trigger_f1", "argument_f1", "event_f1"]
            },
            "free_minus_direct": {
                metric: pairwise(free_rows, direct_rows, metric)
                for metric in ["trigger_f1", "argument_f1", "event_f1"]
            },
            "trigger_correct_arg_wrong": {
                "forced_direct": trigger_correct_arg_wrong(direct_rows),
                "forced_reason": trigger_correct_arg_wrong(reason_rows),
                "free_route": trigger_correct_arg_wrong(free_rows),
            },
            "argument_error_breakdown": {
                "forced_direct": argument_error_breakdown(direct_rows),
                "forced_reason": argument_error_breakdown(reason_rows),
                "free_route": argument_error_breakdown(free_rows),
            },
            "event_type_confusions": {
                "forced_direct": event_type_confusions(direct_rows),
                "forced_reason": event_type_confusions(reason_rows),
                "free_route": event_type_confusions(free_rows),
            },
        }
    return out


def row_case(branch, split, key, label):
    free = branch["aligned"][split]["free_route"][key]
    direct = branch["aligned"][split]["forced_direct"][key]
    reason = branch["aligned"][split]["forced_reason"][key]
    meta = free.get("meta") or {}
    return {
        "branch": branch["name"],
        "split": split,
        "label": label,
        "id": key,
        "text": extract_text_block(free.get("input", ""))[:700],
        "adaptive_route_label": meta.get("adaptive_route_label"),
        "free_route_pred": free.get("route_pred"),
        "valid_reason_final_json": reason.get("valid_json"),
        "direct_metrics": {m: direct.get(m) for m in ["trigger_f1", "argument_f1", "event_f1"]},
        "reason_metrics": {m: reason.get(m) for m in ["trigger_f1", "argument_f1", "event_f1"]},
        "free_metrics": {m: free.get(m) for m in ["trigger_f1", "argument_f1", "event_f1"]},
        "gold": compact_events(free.get("gold") or {}),
        "direct_pred": compact_events(direct.get("predicted") or {}),
        "reason_pred": compact_events(reason.get("predicted") or {}),
        "free_pred": compact_events(free.get("predicted") or {}),
        "direct_diagnosis": row_argument_diagnosis(direct),
        "reason_diagnosis": row_argument_diagnosis(reason),
        "reason_schema": parse_reason_schema(reason),
        "reason_excerpt": reason_text(reason.get("generated_payload") or "")[:700],
    }


def choose_cases(branches):
    cases = []
    for branch in branches:
        for split in ["test", "test_unseen"]:
            keys = sorted(branch["aligned"][split]["forced_direct"])
            helps = []
            hurts = []
            invalid = []
            collapse = []
            schema_events = []
            for key in keys:
                direct = branch["aligned"][split]["forced_direct"][key]
                reason = branch["aligned"][split]["forced_reason"][key]
                free = branch["aligned"][split]["free_route"][key]
                direct_score = (
                    safe_float(direct.get("event_f1")),
                    safe_float(direct.get("argument_f1")),
                    safe_float(direct.get("trigger_f1")),
                )
                reason_score = (
                    safe_float(reason.get("event_f1")),
                    safe_float(reason.get("argument_f1")),
                    safe_float(reason.get("trigger_f1")),
                )
                reason_better = reason.get("valid_json", False) and reason_score > direct_score
                direct_better = direct_score > reason_score
                if reason_better:
                    helps.append((key, tuple(a - b for a, b in zip(reason_score, direct_score))))
                if direct_better:
                    hurts.append((key, tuple(a - b for a, b in zip(direct_score, reason_score))))
                if not reason.get("valid_json", False):
                    invalid.append(key)
                if (free.get("meta") or {}).get("adaptive_route_label") == "reason" and free.get("route_pred") != "reason":
                    collapse.append(key)
                if parse_reason_schema(reason)["has_events_key"]:
                    schema_events.append(key)
            if helps and len([c for c in cases if c["label"] == "forced_reason helps"]) < 3:
                helps.sort(key=lambda item: item[1], reverse=True)
                cases.append(row_case(branch, split, helps[0][0], "forced_reason helps"))
            if hurts and len([c for c in cases if c["label"] == "forced_reason hurts"]) < 3:
                hurts.sort(key=lambda item: item[1], reverse=True)
                cases.append(row_case(branch, split, hurts[0][0], "forced_reason hurts"))
            if invalid and len([c for c in cases if c["label"] == "forced_reason invalid final"]) < 3:
                cases.append(row_case(branch, split, invalid[0], "forced_reason invalid final"))
            if collapse and len([c for c in cases if c["label"] == "reason-label routed direct"]) < 3:
                cases.append(row_case(branch, split, collapse[0], "reason-label routed direct"))
            if schema_events and len([c for c in cases if c["label"] == "reason leaks events key"]) < 3:
                cases.append(row_case(branch, split, schema_events[0], "reason leaks events key"))
    return cases[:14]


def build_result(root_base, label_dir):
    branches = [load_branch(root_base, branch, label_dir) for branch in BRANCHES]
    result = {
        "root_base": root_base,
        "label_dir": label_dir,
        "branches": {},
        "cases": choose_cases(branches),
    }
    for branch in branches:
        name = branch["name"]
        result["branches"][name] = {
            "run_dir": branch["run_dir"],
            "summary_rows": [
                {
                    "mode": mode,
                    "split": split,
                    "json_valid_rate": branch["summaries"][(mode, split)].get("json_valid_rate"),
                    "route_reason_rate": branch["summaries"][(mode, split)].get("route_reason_rate"),
                    "trigger_f1": branch["summaries"][(mode, split)].get("trigger_f1"),
                    "argument_f1": branch["summaries"][(mode, split)].get("argument_f1"),
                    "event_f1": branch["summaries"][(mode, split)].get("event_f1"),
                    "avg_latency_sec": branch["summaries"][(mode, split)].get("avg_latency_sec"),
                }
                for mode in MODES
                for split in SPLITS
            ],
            "route_behavior": route_behavior(branch),
            "reason_schema": reason_schema_stats(branch),
            "pairwise_and_errors": pairwise_and_errors(branch),
            "buckets": bucket_analysis(branch),
        }
    result["takeaways"] = make_takeaways(result)
    return result


def make_takeaways(result):
    lines = []
    for name, branch in result["branches"].items():
        unseen = {row["mode"]: row for row in branch["summary_rows"] if row["split"] == "test_unseen"}
        reason_delta_arg = unseen["forced_reason"]["argument_f1"] - unseen["forced_direct"]["argument_f1"]
        reason_delta_event = unseen["forced_reason"]["event_f1"] - unseen["forced_direct"]["event_f1"]
        free_reason_rate = unseen["free_route"]["route_reason_rate"]
        lines.append(
            f"{name}: test_unseen forced_reason-direct delta arg={reason_delta_arg:.4f}, event={reason_delta_event:.4f}; free_route reason_rate={free_reason_rate:.4f}."
        )
    lines.append("The dominant failure is not routing alone: forced_reason is usually weaker than forced_direct, so the reason expert does not yet provide a strong action for the router.")
    lines.append("The free router collapses to direct on almost all formal examples, including many heuristic reason-labeled examples.")
    lines.append("The verify-lite reason schema is unstable: several branches often generate an `events` key inside <REASON>, directly violating the intended reasoning-only schema.")
    lines.append("Next experiments should optimize reason-path supervision and route balance jointly; pure router calibration is premature until forced_reason beats forced_direct on hard buckets.")
    return lines


def markdown_report(result):
    lines = []
    lines.append("# Adaptive Verify-Lite Mechanism Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Dataset: RichERE split1 oracle_mixed_noise_top10_shuffle.")
    lines.append("- Model: Qwen3-1.7B.")
    lines.append("- Target: adaptive `type_role_verify_lite` with `<ROUTE>`, optional `<REASON>`, and `<FINAL>`.")
    lines.append("- Inputs: completed formal predictions only; no new training.")
    lines.append("")
    lines.append("## Main Takeaways")
    lines.append("")
    for item in result["takeaways"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Formal Metrics")
    lines.append("")
    lines.append("| branch | mode | split | json | reason_rate | trigger | argument | event | latency |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for name, branch in result["branches"].items():
        for row in branch["summary_rows"]:
            lines.append(
                f"| `{name}` | `{row['mode']}` | `{row['split']}` | {fmt(row['json_valid_rate'])} | {fmt(row['route_reason_rate'])} | {fmt(row['trigger_f1'])} | {fmt(row['argument_f1'])} | {fmt(row['event_f1'])} | {fmt(row['avg_latency_sec'])} |"
            )
    lines.append("")
    lines.append("## Test-Unseen Gate")
    lines.append("")
    lines.append("| branch | forced_reason - forced_direct arg | forced_reason - forced_direct event | free - forced_direct arg | free - forced_direct event |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, branch in result["branches"].items():
        rows = {(row["mode"], row["split"]): row for row in branch["summary_rows"]}
        rd_arg = rows[("forced_reason", "test_unseen")]["argument_f1"] - rows[("forced_direct", "test_unseen")]["argument_f1"]
        rd_event = rows[("forced_reason", "test_unseen")]["event_f1"] - rows[("forced_direct", "test_unseen")]["event_f1"]
        fd_arg = rows[("free_route", "test_unseen")]["argument_f1"] - rows[("forced_direct", "test_unseen")]["argument_f1"]
        fd_event = rows[("free_route", "test_unseen")]["event_f1"] - rows[("forced_direct", "test_unseen")]["event_f1"]
        lines.append(f"| `{name}` | {fmt(rd_arg)} | {fmt(rd_event)} | {fmt(fd_arg)} | {fmt(fd_event)} |")
    lines.append("")
    lines.append("## Route Collapse")
    lines.append("")
    lines.append("| branch | split | label_reason | label_reason_routed_reason | routed_reason_rate_on_label_reason | pred_route_counts |")
    lines.append("|---|---|---:|---:|---:|---|")
    for name, branch in result["branches"].items():
        for split, payload in branch["route_behavior"].items():
            lines.append(
                f"| `{name}` | `{split}` | {payload['reason_labeled']} | {payload['reason_labeled_routed_reason']} | {fmt(payload['reason_labeled_routed_reason_rate'])} | `{payload['pred_route_counts']}` |"
            )
    lines.append("")
    lines.append("## Reason Schema Compliance")
    lines.append("")
    lines.append("| branch | split | parse_ok | has_events_key | has_type_decisions | has_role_checks | avg_role_checks | avg_status_checks | avg_absent_checks |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, branch in result["branches"].items():
        for split, payload in branch["reason_schema"].items():
            lines.append(
                f"| `{name}` | `{split}` | {fmt(payload['reason_parse_ok_rate'])} | {fmt(payload['has_events_key_rate'])} | {fmt(payload['has_type_decisions_key_rate'])} | {fmt(payload['has_role_checks_key_rate'])} | {fmt(payload['avg_role_checks_count'])} | {fmt(payload['avg_role_checks_with_status'])} | {fmt(payload['avg_absent_checks'])} |"
            )
    lines.append("")
    lines.append("## Pairwise Forced Reason vs Direct")
    lines.append("")
    lines.append("| branch | split | metric | wins | ties | losses | row_avg_delta |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for name, branch in result["branches"].items():
        for split, payload in branch["pairwise_and_errors"].items():
            for metric, stats in payload["reason_minus_direct"].items():
                lines.append(
                    f"| `{name}` | `{split}` | `{metric}` | {stats['wins']} | {stats['ties']} | {stats['losses']} | {fmt(stats['row_avg_delta'])} |"
                )
    lines.append("")
    lines.append("## Trigger-Correct But Argument-Wrong")
    lines.append("")
    lines.append("| branch | split | direct | reason | free |")
    lines.append("|---|---|---:|---:|---:|")
    for name, branch in result["branches"].items():
        for split, payload in branch["pairwise_and_errors"].items():
            tcaw = payload["trigger_correct_arg_wrong"]
            lines.append(f"| `{name}` | `{split}` | {tcaw['forced_direct']} | {tcaw['forced_reason']} | {tcaw['free_route']} |")
    lines.append("")
    lines.append("## Bucket Deltas")
    lines.append("")
    lines.append("Row-average deltas are used here for attribution; formal metrics above remain the primary result.")
    lines.append("")
    lines.append("| branch | split | bucket | n | reason-direct arg | reason-direct event |")
    lines.append("|---|---|---|---:|---:|---:|")
    for name, branch in result["branches"].items():
        for split, buckets in branch["buckets"].items():
            for bucket, payload in buckets.items():
                n = payload["direct"]["n"]
                lines.append(
                    f"| `{name}` | `{split}` | `{bucket}` | {n} | {fmt(payload['reason_minus_direct_argument_row_avg'])} | {fmt(payload['reason_minus_direct_event_row_avg'])} |"
                )
    lines.append("")
    lines.append("## Argument Error Categories")
    lines.append("")
    lines.append("| branch | split | mode | FN categories | FP categories |")
    lines.append("|---|---|---|---|---|")
    for name, branch in result["branches"].items():
        for split, payload in branch["pairwise_and_errors"].items():
            for mode, errors in payload["argument_error_breakdown"].items():
                fn = ", ".join(f"{k}:{v}" for k, v in errors["fn_categories"][:5])
                fp = ", ".join(f"{k}:{v}" for k, v in errors["fp_categories"][:5])
                lines.append(f"| `{name}` | `{split}` | `{mode}` | {fn} | {fp} |")
    lines.append("")
    lines.append("## Case Studies")
    for idx, case in enumerate(result["cases"], 1):
        lines.append("")
        lines.append(f"### Case {idx}: {case['label']}")
        lines.append("")
        lines.append(f"- branch: `{case['branch']}`")
        lines.append(f"- split: `{case['split']}`")
        lines.append(f"- id: `{case['id']}`")
        lines.append(f"- adaptive label: `{case['adaptive_route_label']}`; free route: `{case['free_route_pred']}`; forced reason valid final: `{case['valid_reason_final_json']}`")
        lines.append(f"- direct metrics: `{case['direct_metrics']}`")
        lines.append(f"- reason metrics: `{case['reason_metrics']}`")
        lines.append(f"- free metrics: `{case['free_metrics']}`")
        lines.append(f"- text: {case['text']}")
        lines.append("")
        lines.append("Gold:")
        lines.append("```json")
        lines.append(json.dumps(case["gold"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("Direct prediction:")
        lines.append("```json")
        lines.append(json.dumps(case["direct_pred"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("Reason prediction:")
        lines.append("```json")
        lines.append(json.dumps(case["reason_pred"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("Reason schema:")
        lines.append("```json")
        lines.append(json.dumps(case["reason_schema"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("Reason excerpt:")
        lines.append("```json")
        lines.append(case["reason_excerpt"])
        lines.append("```")
    lines.append("")
    lines.append("## Optimization Implications")
    lines.append("")
    lines.append("- Do not spend the next wave on router calibration alone. The action `reason` is often worse than `direct`, so a better router cannot reliably improve the final extractor.")
    lines.append("- The verify-lite target is too structurally complex for the current setup. It hurts final JSON validity in several branches and frequently copies final `events` into `<REASON>`.")
    lines.append("- The next target should separate routing from final extraction more cleanly: either a much smaller reason schema, or explicit route-balanced training that preserves direct anchors while teaching reason-only samples to output stable `<REASON>` keys.")
    lines.append("- A strong acceptance gate for the next run should be `forced_reason > forced_direct` on reason-labeled/high-confusion/high-role-rarity buckets before expanding cross-model experiments.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_base", default="outputs/stage2_adaptive_runs_user_formal_clean")
    parser.add_argument("--label_dir", default="data/stage2_adaptive_datasets/labels")
    parser.add_argument("--output_md", default="reports/2026-05-08_stage2_adaptive_verify_lite_mechanism_analysis.md")
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-08_stage2_adaptive_verify_lite_mechanism_analysis.json")
    args = parser.parse_args()

    result = build_result(args.root_base, args.label_dir)
    write_json(Path(args.output_json), result)
    write_text(Path(args.output_md), markdown_report(result))
    print(json.dumps({"output_md": args.output_md, "output_json": args.output_json}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
