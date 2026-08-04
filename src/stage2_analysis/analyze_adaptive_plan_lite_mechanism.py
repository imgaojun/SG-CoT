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
    row_key,
)


MODES = ["free_route", "forced_direct", "forced_reason"]
SPLITS = ["test", "test_seen", "test_unseen"]
PLAN_RE = re.compile(r"<PLAN>\s*(.*?)\s*</PLAN>", re.DOTALL | re.IGNORECASE)

BRANCHES = [
    {
        "name": "confrare10_heur10_plan_lite",
        "run_dir": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_plan_lite",
        "label_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10",
    },
    {
        "name": "confrare10_heur10_type_plan_lite",
        "run_dir": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_type_plan_lite",
        "label_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10",
    },
    {
        "name": "roleconf10_heur10_plan_lite",
        "run_dir": "richere_split1_qwen3_1_7b_adaptive_roleconf10_heur10_plan_lite",
        "label_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_roleconf10_heur10",
    },
    {
        "name": "confrare10_heur10_plan_lite_pairdirect",
        "run_dir": "richere_split1_qwen3_1_7b_adaptive_confrare10_heur10_plan_lite_pairdirect",
        "label_prefix": "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_confrare10_heur10",
    },
]


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


def fmt(value):
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def plan_text(payload):
    match = PLAN_RE.search(payload or "")
    return match.group(1).strip() if match else ""


def parse_plan_schema(row):
    text = plan_text(row.get("generated_payload") or row.get("generated_text") or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "has_plan": bool(text),
        "line_count": len(lines),
        "has_json_brace": "{" in text or "}" in text,
        "has_events_key": "events" in text,
        "type_lines": sum(" TYPE " in line for line in lines),
        "trigger_lines": sum(" TRIGGER " in line for line in lines),
        "contrast_lines": sum(" CONTRAST " in line for line in lines),
        "arg_lines": sum(" ARG " in line for line in lines),
    }


def load_labels(label_dir, label_prefix, split):
    path = Path(label_dir) / f"{label_prefix}_{split}_labels.jsonl"
    return {row["wnd_id"]: row for row in load_jsonl(path)}


def load_branch(root_base, branch, label_dir):
    root = Path(root_base) / branch["run_dir"]
    result = {
        "name": branch["name"],
        "run_dir": root.as_posix(),
        "summaries": {},
        "aligned": defaultdict(lambda: defaultdict(dict)),
        "labels": {},
    }
    for split in SPLITS:
        result["labels"][split] = load_labels(label_dir, branch["label_prefix"], split)
        for mode in MODES:
            summary_path = root / mode / split / "summary.json"
            pred_path = root / mode / split / "predictions.jsonl"
            result["summaries"][(mode, split)] = load_json(summary_path)
            rows = load_jsonl(pred_path)
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
    for a, b in zip(rows_a, rows_b):
        delta = float(a.get(metric, 0.0)) - float(b.get(metric, 0.0))
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


def reason_label(row):
    meta = row.get("meta", {})
    wnd_id = meta.get("wnd_id")
    return wnd_id


def missing_argument_count(rows):
    total = 0
    for row in rows:
        cats = argument_error_breakdown([row])["fn_categories"]
        total += dict(cats).get("missing_argument", 0)
    return total


def plan_compliance(rows):
    parsed = [parse_plan_schema(row) for row in rows]
    n = len(parsed)
    if not n:
        return {}
    return {
        "has_plan_rate": sum(x["has_plan"] for x in parsed) / n,
        "has_json_brace_rate": sum(x["has_json_brace"] for x in parsed) / n,
        "has_events_key_rate": sum(x["has_events_key"] for x in parsed) / n,
        "avg_line_count": sum(x["line_count"] for x in parsed) / n,
        "avg_type_lines": sum(x["type_lines"] for x in parsed) / n,
        "avg_trigger_lines": sum(x["trigger_lines"] for x in parsed) / n,
        "avg_contrast_lines": sum(x["contrast_lines"] for x in parsed) / n,
        "avg_arg_lines": sum(x["arg_lines"] for x in parsed) / n,
    }


def branch_analysis(branch):
    out = {
        "name": branch["name"],
        "formal": {},
        "pairwise_reason_direct": {},
        "missing_argument": {},
        "plan_compliance": {},
        "label_reason_bucket": {},
    }
    for split in SPLITS:
        for mode in MODES:
            out["formal"][f"{mode}/{split}"] = branch["summaries"][(mode, split)]

        keys = sorted(branch["aligned"][split]["forced_direct"])
        direct = [branch["aligned"][split]["forced_direct"][key] for key in keys]
        reason = [branch["aligned"][split]["forced_reason"][key] for key in keys]
        out["pairwise_reason_direct"][split] = {
            metric: pairwise(reason, direct, metric)
            for metric in ["trigger_f1", "argument_f1", "event_f1"]
        }
        out["missing_argument"][split] = {
            "forced_direct": missing_argument_count(direct),
            "forced_reason": missing_argument_count(reason),
        }
        out["plan_compliance"][split] = plan_compliance(reason)

        labels = branch["labels"][split]
        reason_keys = []
        for key, row in branch["aligned"][split]["forced_direct"].items():
            wnd_id = row.get("meta", {}).get("wnd_id")
            if labels.get(wnd_id, {}).get("route_label") == "reason":
                reason_keys.append(key)
        if reason_keys:
            d_bucket = [branch["aligned"][split]["forced_direct"][key] for key in reason_keys]
            r_bucket = [branch["aligned"][split]["forced_reason"][key] for key in reason_keys]
            out["label_reason_bucket"][split] = {
                "n": len(reason_keys),
                "argument_delta": sum(float(r.get("argument_f1", 0)) - float(d.get("argument_f1", 0)) for r, d in zip(r_bucket, d_bucket)) / len(reason_keys),
                "event_delta": sum(float(r.get("event_f1", 0)) - float(d.get("event_f1", 0)) for r, d in zip(r_bucket, d_bucket)) / len(reason_keys),
            }
        else:
            out["label_reason_bucket"][split] = {"n": 0, "argument_delta": 0.0, "event_delta": 0.0}
    return out


def render_report(analysis):
    lines = ["# Adaptive Plan-Lite Mechanism Analysis", ""]
    lines.append("## Formal Gate")
    lines.append("")
    lines.append("| branch | split | free reason rate | forced direct arg | forced reason arg | delta arg | forced direct event | forced reason event | delta event | reason json |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for branch in analysis["branches"]:
        for split in SPLITS:
            fd = branch["formal"][f"forced_direct/{split}"]
            fr = branch["formal"][f"forced_reason/{split}"]
            free = branch["formal"][f"free_route/{split}"]
            lines.append(
                f"| `{branch['name']}` | `{split}` | {fmt(free.get('route_reason_rate'))} | "
                f"{fmt(fd.get('argument_f1'))} | {fmt(fr.get('argument_f1'))} | {fmt(fr.get('argument_f1', 0)-fd.get('argument_f1', 0))} | "
                f"{fmt(fd.get('event_f1'))} | {fmt(fr.get('event_f1'))} | {fmt(fr.get('event_f1', 0)-fd.get('event_f1', 0))} | "
                f"{fmt(fr.get('json_valid_rate'))} |"
            )
    lines.append("")
    lines.append("## Plan Compliance")
    lines.append("")
    lines.append("| branch | split | has plan | json brace | events key | avg lines | avg arg lines |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for branch in analysis["branches"]:
        for split in SPLITS:
            comp = branch["plan_compliance"][split]
            lines.append(
                f"| `{branch['name']}` | `{split}` | {fmt(comp.get('has_plan_rate'))} | "
                f"{fmt(comp.get('has_json_brace_rate'))} | {fmt(comp.get('has_events_key_rate'))} | "
                f"{fmt(comp.get('avg_line_count'))} | {fmt(comp.get('avg_arg_lines'))} |"
            )
    lines.append("")
    lines.append("## Label-Reason Bucket")
    lines.append("")
    lines.append("| branch | split | n | reason-direct arg delta | reason-direct event delta |")
    lines.append("|---|---|---:|---:|---:|")
    for branch in analysis["branches"]:
        for split in SPLITS:
            bucket = branch["label_reason_bucket"][split]
            lines.append(
                f"| `{branch['name']}` | `{split}` | {bucket['n']} | {fmt(bucket['argument_delta'])} | {fmt(bucket['event_delta'])} |"
            )
    lines.append("")
    lines.append("## Missing Argument FN")
    lines.append("")
    lines.append("| branch | split | forced direct | forced reason | delta |")
    lines.append("|---|---|---:|---:|---:|")
    for branch in analysis["branches"]:
        for split in SPLITS:
            row = branch["missing_argument"][split]
            lines.append(
                f"| `{branch['name']}` | `{split}` | {row['forced_direct']} | {row['forced_reason']} | {row['forced_reason'] - row['forced_direct']} |"
            )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- Pass condition: forced-reason must beat forced-direct on `test_unseen` or label-reason buckets while keeping JSON valid rate at least `0.99`.")
    lines.append("- Plan compliance should show near-zero JSON braces and `events` leakage inside `<PLAN>`.")
    lines.append("- A useful plan target should reduce or at least not increase `missing_argument` false negatives.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_base", default="outputs/stage2_adaptive_runs_user_formal_clean")
    parser.add_argument("--label_dir", default="data/stage2_adaptive_datasets/labels")
    parser.add_argument("--output_md", default="reports/2026-05-09_stage2_adaptive_plan_lite_mechanism_analysis.md")
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-09_stage2_adaptive_plan_lite_mechanism_analysis.json")
    args = parser.parse_args()

    branches = [branch_analysis(load_branch(args.root_base, branch, args.label_dir)) for branch in BRANCHES]
    payload = {"branches": branches}
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), render_report(payload))
    print(json.dumps({"output_md": args.output_md, "output_json": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
