import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_analysis.analyze_adaptive_checkpoint_frontier import (  # noqa: E402
    BRANCHES,
    MODES,
    choose_adaptive_tradeoff_best,
    choose_direct_anchor_best,
    choose_reason_expert_best,
    metric,
    row_record,
    summary_path,
)
from src.stage2_analysis.analyze_adaptive_plan_lite_mechanism import load_jsonl  # noqa: E402
from src.stage2_cot.build_selective_aux_reasoning_dataset import (  # noqa: E402
    confrare_score_row,
    row_id,
)
from src.stage2_data.build_formal_stage2_dataset import load_json, load_schema_map  # noqa: E402


PROTOCOLS = ["direct_anchor_best", "reason_expert_best", "adaptive_tradeoff_best"]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def fmt(value):
    return f"{float(value):.4f}" if value is not None else "-"


def load_mode_payloads(branch, free_root, frontier_root):
    paths = {
        "free_route": summary_path(free_root, branch["run_slug"], "free_route"),
        "forced_direct": summary_path(frontier_root, branch["run_slug"], "forced_direct"),
        "forced_reason": summary_path(frontier_root, branch["run_slug"], "forced_reason"),
    }
    missing = [path.as_posix() for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing checkpoint sweep summaries:\n" + "\n".join(missing))
    return {mode: load_json(path) for mode, path in paths.items()}


def load_rows(branch, free_root, frontier_root):
    mode_payloads = load_mode_payloads(branch, free_root, frontier_root)
    rows = {}
    for mode, payload in mode_payloads.items():
        for candidate in payload.get("candidates", []):
            tag = candidate["checkpoint_tag"]
            rows.setdefault(tag, {"checkpoint_tag": tag})[mode] = candidate
    full_rows = []
    for tag, row in rows.items():
        if all(mode in row for mode in MODES):
            row["checkpoint_path"] = row["free_route"]["checkpoint_path"]
            full_rows.append(row)
    full_rows.sort(key=lambda row: int(row["checkpoint_tag"].split("-")[-1]))
    return full_rows


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for _, y in pairs))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def load_predictions(path):
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    out = {}
    for idx, row in enumerate(rows):
        wnd_id = row.get("meta", {}).get("wnd_id") or f"idx:{idx}"
        out[wnd_id] = row
    return out


def label_key_for_branch(branch_name):
    if branch_name.startswith("roleconf10_"):
        return "roleconf10_heur10"
    if branch_name.startswith("roleconf5_"):
        return "roleconf5_heur5"
    if branch_name.startswith("confrare5_"):
        return "confrare5_heur5"
    if branch_name.startswith("confrole10_"):
        return "confrole10_heur10"
    return "confrare10_heur10"


def score_buckets(eval_jsonl, schema_path, label_jsonl):
    rows = load_jsonl(Path(eval_jsonl))
    direct_rows = []
    for row in rows:
        item = dict(row)
        item["output"] = row.get("gold_output", row["output"])
        direct_rows.append(item)
    labels = {row["wnd_id"]: row for row in load_jsonl(Path(label_jsonl))} if Path(label_jsonl).exists() else {}
    schema = load_schema_map(Path(schema_path))
    from src.stage2_cot.build_selective_aux_reasoning_dataset import build_confrare_stats

    stats = build_confrare_stats(direct_rows)
    scored = [confrare_score_row(idx, row, schema, stats) for idx, row in enumerate(direct_rows)]
    score_by_id = {item["wnd_id"]: item for item in scored}
    confusion_values = sorted(item["confusion_score"] for item in scored)
    rarity_values = sorted(item["role_signature_rarity"] for item in scored)

    def percentile(values, q):
        if not values:
            return 0.0
        pos = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
        return values[pos]

    confusion_cut = percentile(confusion_values, 0.75)
    rarity_cut = percentile(rarity_values, 0.75)
    buckets = {}
    for row in direct_rows:
        rid = row_id(row)
        score = score_by_id[rid]
        label = labels.get(rid, {})
        names = ["all"]
        if label.get("route_label") == "reason":
            names.append("label_reason")
        else:
            names.append("label_direct")
        if score["confusion_score"] >= confusion_cut:
            names.append("high_confusion")
        if score["role_signature_rarity"] >= rarity_cut:
            names.append("high_role_signature_rarity")
        buckets[rid] = names
    return buckets


def bucket_metric_delta(direct_preds, reason_preds, buckets):
    grouped = {}
    for wnd_id, bucket_names in buckets.items():
        direct = direct_preds.get(wnd_id)
        reason = reason_preds.get(wnd_id)
        if direct is None or reason is None:
            continue
        for bucket in bucket_names:
            row = grouped.setdefault(bucket, {"n": 0, "argument_delta": 0.0, "event_delta": 0.0})
            row["n"] += 1
            row["argument_delta"] += float(reason.get("argument_f1", 0.0)) - float(direct.get("argument_f1", 0.0))
            row["event_delta"] += float(reason.get("event_f1", 0.0)) - float(direct.get("event_f1", 0.0))
    for row in grouped.values():
        if row["n"]:
            row["argument_delta"] /= row["n"]
            row["event_delta"] /= row["n"]
    return grouped


def prediction_path(root, branch, mode, checkpoint_tag):
    mode_slug = "free" if mode == "free_route" else mode
    return (
        Path(root)
        / f"{branch['run_slug']}_{mode_slug}_dev_seen_max512"
        / checkpoint_tag
        / "predictions.jsonl"
    )


def formal_summary(root, branch, protocol, mode, split):
    path = Path(root) / branch["formal_slug"] / f"frontier_{protocol}" / mode / split / "summary.json"
    if not path.exists():
        return None
    return load_json(path)


def analyze_branch(branch, args):
    rows = load_rows(branch, Path(args.existing_free_root), Path(args.devpick_root))
    direct_anchor = choose_direct_anchor_best(rows)
    reason_expert = choose_reason_expert_best(rows)
    tradeoff = choose_adaptive_tradeoff_best(rows)
    protocols = {
        "direct_anchor_best": direct_anchor,
        "reason_expert_best": reason_expert,
        "adaptive_tradeoff_best": tradeoff,
    }
    curve = [row_record(row) for row in rows]
    dev_reason_delta = [item["reason_direct_delta"]["argument_f1"] for item in curve]
    dev_free_delta = [item["free_direct_delta"]["argument_f1"] for item in curve]
    dev_free_arg = [item["free_route"]["argument_f1"] for item in curve]
    dev_reason_arg = [item["forced_reason"]["argument_f1"] for item in curve]

    label_key = label_key_for_branch(branch["name"])
    buckets = score_buckets(
        f"data/stage2_adaptive_datasets/{branch['dataset_prefix']}_dev_seen_pos.jsonl",
        args.schema_path,
        f"{args.label_dir}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_{label_key}_dev_seen_labels.jsonl",
    )
    bucket_by_protocol = {}
    for protocol, selected in protocols.items():
        direct_preds = load_predictions(prediction_path(args.devpick_root, branch, "forced_direct", selected["checkpoint_tag"]))
        reason_preds = load_predictions(prediction_path(args.devpick_root, branch, "forced_reason", selected["checkpoint_tag"]))
        bucket_by_protocol[protocol] = bucket_metric_delta(direct_preds, reason_preds, buckets)

    formal = {}
    for protocol in PROTOCOLS:
        formal[protocol] = {}
        for mode in MODES:
            for split in ["test", "test_seen", "test_unseen"]:
                summary = formal_summary(args.formal_root, branch, protocol, mode, split)
                if summary:
                    formal[protocol][f"{mode}/{split}"] = summary

    return {
        "name": branch["name"],
        "curve": curve,
        "correlations": {
            "dev_reason_delta_vs_dev_free_delta": corr(dev_reason_delta, dev_free_delta),
            "dev_reason_delta_vs_dev_free_arg": corr(dev_reason_delta, dev_free_arg),
            "dev_reason_delta_vs_dev_reason_arg": corr(dev_reason_delta, dev_reason_arg),
        },
        "protocols": {name: row_record(row) for name, row in protocols.items()},
        "dev_buckets": bucket_by_protocol,
        "selected_formal": formal,
    }


def render_report(payload):
    lines = ["# Adaptive Checkpoint Mechanism Deep Analysis", ""]
    lines.append("## Protocol Selection")
    lines.append("")
    lines.append("| branch | protocol | checkpoint | free reason rate | free arg | direct arg | reason arg | reason-direct arg delta | free-direct arg delta | tradeoff score |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for branch in payload["branches"]:
        for protocol in PROTOCOLS:
            row = branch["protocols"][protocol]
            lines.append(
                f"| `{branch['name']}` | `{protocol}` | `{row['checkpoint_tag']}` | "
                f"{fmt(row['free_route'].get('route_reason_rate', 0.0))} | {fmt(row['free_route']['argument_f1'])} | "
                f"{fmt(row['forced_direct']['argument_f1'])} | {fmt(row['forced_reason']['argument_f1'])} | "
                f"{fmt(row['reason_direct_delta']['argument_f1'])} | {fmt(row['free_direct_delta']['argument_f1'])} | "
                f"{fmt(row['adaptive_tradeoff_score'])} |"
            )
    lines.append("")
    lines.append("## Dev Correlations")
    lines.append("")
    lines.append("| branch | reason_delta vs free_delta | reason_delta vs free_arg | reason_delta vs reason_arg |")
    lines.append("|---|---:|---:|---:|")
    for branch in payload["branches"]:
        c = branch["correlations"]
        lines.append(
            f"| `{branch['name']}` | {fmt(c['dev_reason_delta_vs_dev_free_delta'])} | "
            f"{fmt(c['dev_reason_delta_vs_dev_free_arg'])} | {fmt(c['dev_reason_delta_vs_dev_reason_arg'])} |"
        )
    lines.append("")
    lines.append("## Dev Bucket Reason-Direct Delta")
    lines.append("")
    lines.append("| branch | protocol | bucket | n | argument delta | event delta |")
    lines.append("|---|---|---|---:|---:|---:|")
    for branch in payload["branches"]:
        for protocol, buckets in branch["dev_buckets"].items():
            for bucket in ["all", "label_reason", "high_confusion", "high_role_signature_rarity"]:
                row = buckets.get(bucket)
                if not row:
                    continue
                lines.append(
                    f"| `{branch['name']}` | `{protocol}` | `{bucket}` | {row['n']} | "
                    f"{fmt(row['argument_delta'])} | {fmt(row['event_delta'])} |"
                )
    lines.append("")
    lines.append("## Selected Formal Snapshot")
    lines.append("")
    lines.append("| branch | protocol | mode/split | json | reason rate | argument | event |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for branch in payload["branches"]:
        for protocol, summaries in branch["selected_formal"].items():
            for key in ["free_route/test_unseen", "forced_direct/test_unseen", "forced_reason/test_unseen"]:
                summary = summaries.get(key)
                if not summary:
                    continue
                lines.append(
                    f"| `{branch['name']}` | `{protocol}` | `{key}` | {fmt(summary.get('json_valid_rate'))} | "
                    f"{fmt(summary.get('route_reason_rate', 0.0))} | {fmt(summary.get('argument_f1'))} | {fmt(summary.get('event_f1'))} |"
                )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- `reason_expert_best` isolates whether the reason path can become stronger than direct at any checkpoint.")
    lines.append("- `direct_anchor_best` is the stability anchor and usually favors seen/overall extraction.")
    lines.append("- `adaptive_tradeoff_best` is the deployable compromise: it rewards free-route F1 and positive reason margin while penalizing reason-rate drift from 15%.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devpick_root", default="outputs/stage2_adaptive_runs_user_devpick_frontier")
    parser.add_argument("--existing_free_root", default="outputs/stage2_adaptive_runs_user_devpick")
    parser.add_argument("--formal_root", default="outputs/stage2_adaptive_runs_user_formal_clean")
    parser.add_argument("--label_dir", default="data/stage2_adaptive_datasets/labels")
    parser.add_argument("--schema_path", default="data/schema/richere-en.event_schema.json")
    parser.add_argument("--output_md", default="reports/2026-05-10_stage2_adaptive_checkpoint_mechanism_deep_analysis.md")
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-10_stage2_adaptive_checkpoint_mechanism_deep_analysis.json")
    parser.add_argument(
        "--branch_names",
        nargs="+",
        default=[
            "confrare10_heur10_type_plan_lite",
            "confrare10_heur10_type_plan_v2",
            "confrare10_heur10_plan_lite",
            "confrare10_heur10_plan_lite_pairdirect",
            "roleconf10_heur10_plan_lite",
        ],
    )
    args = parser.parse_args()

    branch_map = {branch["name"]: branch for branch in BRANCHES}
    branches = []
    for name in args.branch_names:
        if name not in branch_map:
            raise ValueError(f"unknown branch: {name}")
        branches.append(analyze_branch(branch_map[name], args))
    payload = {"branches": branches}
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), render_report(payload))
    print(json.dumps({"output_md": args.output_md, "output_json": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
