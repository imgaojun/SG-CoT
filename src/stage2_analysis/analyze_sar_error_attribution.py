import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_cot.build_selective_aux_reasoning_dataset import (  # noqa: E402
    build_confrare_stats,
    event_role_signature,
    parse_output_events,
    row_id,
    sample_confusion_score,
    sample_event_type_rarity,
    sample_role_signature_rarity,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl, load_schema_map  # noqa: E402


DEFAULT_BRANCHES = [
    "confrare10_typerolelite",
    "confrare10_directdup",
    "confrare20_typerolelite",
    "confrare20_directdup",
    "confrare20_hybrid_dupaux_typerolelite",
    "confrare20_directdup2x",
]
DEFAULT_PAIRS = [
    "confrare10_typerolelite=confrare10_directdup",
    "confrare20_typerolelite=confrare20_directdup",
    "confrare20_hybrid_dupaux_typerolelite=confrare20_directdup2x",
]
SPLITS = ["test", "test_seen", "test_unseen"]
METRIC_KEYS = ["trigger_f1", "argument_f1", "event_f1"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def avg(values):
    return sum(values) / len(values) if values else 0.0


def fmt(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def branch_dataset_family(prefix: str, branch: str):
    return f"{prefix}_sar_{branch}"


def branch_run_name(run_prefix: str, branch: str):
    return f"{run_prefix}_sar_{branch}_full"


def prediction_dir(best_eval_root: Path, run_name: str, split: str):
    suffix = "test_argfirst" if split == "test" else f"{split}_argfirst"
    return best_eval_root / f"{run_name}_{suffix}"


def bucket_label(value: float, threshold: float):
    return "high" if value >= threshold else "low"


def median_threshold(records, key):
    values = sorted(record[key] for record in records)
    if not values:
        return 0.0
    return values[len(values) // 2]


def prediction_events(pred_row):
    payload = pred_row.get("predicted")
    if not isinstance(payload, dict):
        return []
    events = payload.get("events", [])
    return events if isinstance(events, list) else []


def normalize_trigger_tuple(event):
    trigger = event.get("trigger", {}) if isinstance(event, dict) else {}
    if not isinstance(trigger, dict):
        trigger = {}
    return (event.get("event_type"), trigger.get("start"), trigger.get("end"))


def argument_tuples(events):
    items = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type, trigger_start, trigger_end = normalize_trigger_tuple(event)
        args = event.get("arguments", [])
        if not isinstance(args, list):
            continue
        for arg in args:
            if not isinstance(arg, dict):
                continue
            items.add((event_type, trigger_start, trigger_end, arg.get("role"), arg.get("start"), arg.get("end")))
    return items


def trigger_tuples(events):
    return {normalize_trigger_tuple(event) for event in events if isinstance(event, dict)}


def roles_from_events(events):
    roles = set()
    for event in events:
        for arg in event.get("arguments", []) if isinstance(event, dict) else []:
            if isinstance(arg, dict) and arg.get("role"):
                roles.add(arg["role"])
    return roles


def main_role(events):
    counts = Counter()
    for role in roles_from_events(events):
        counts[role] += 1
    return counts.most_common(1)[0][0] if counts else "NO_ROLE"


def exact_category(aux_value: bool, dup_value: bool):
    if aux_value and dup_value:
        return "both_correct"
    if aux_value:
        return "aux_only_correct"
    if dup_value:
        return "directdup_only_correct"
    return "both_wrong"


def compare_metric(aux_record, dup_record, key):
    aux_value = aux_record[key]
    dup_value = dup_record[key]
    if aux_value > dup_value:
        return "aux_win"
    if aux_value < dup_value:
        return "directdup_win"
    return "tie"


def build_selected_signature_sets(direct_train_rows, selected_wnd_ids):
    selected_types = set()
    selected_signatures = set()
    for row in direct_train_rows:
        if row_id(row) not in selected_wnd_ids:
            continue
        for event in parse_output_events(row):
            selected_types.add(event["event_type"])
            selected_signatures.add(event_role_signature(event))
    return selected_types, selected_signatures


def build_branch_records(
    *,
    branch: str,
    dataset_prefix: str,
    run_prefix: str,
    dataset_dir: Path,
    best_eval_root: Path,
    direct_train_rows,
    schema_by_type,
    train_stats,
):
    family = branch_dataset_family(dataset_prefix, branch)
    train_meta = load_json(dataset_dir / f"{family}_train_pos.meta.json")
    selected_wnd_ids = set(train_meta.get("selected_aux_wnd_ids", []))
    selected_types, selected_signatures = build_selected_signature_sets(direct_train_rows, selected_wnd_ids)
    run_name = branch_run_name(run_prefix, branch)

    records_by_split = {}
    for split in SPLITS:
        dataset_rows = load_jsonl(dataset_dir / f"{family}_{split}_pos.jsonl")
        pred_rows = load_jsonl(prediction_dir(best_eval_root, run_name, split) / "predictions.jsonl")
        if len(dataset_rows) != len(pred_rows):
            raise ValueError(f"{branch}/{split}: dataset rows and predictions differ in length")

        raw_records = []
        for idx, (dataset_row, pred_row) in enumerate(zip(dataset_rows, pred_rows)):
            if dataset_row["input"].strip() != pred_row["input"].strip():
                raise ValueError(f"{branch}/{split}: input mismatch at row {idx}")

            gold_events = parse_output_events(dataset_row)
            predicted_events = prediction_events(pred_row)
            gold_types = {event["event_type"] for event in gold_events}
            gold_signatures = {event_role_signature(event) for event in gold_events}
            feature_row = {
                "output": json.dumps({"events": gold_events}, ensure_ascii=False),
                "meta": dataset_row.get("meta", {}),
            }
            gold_args = argument_tuples(gold_events)
            pred_args = argument_tuples(predicted_events)
            gold_triggers = trigger_tuples(gold_events)
            pred_triggers = trigger_tuples(predicted_events)

            raw_records.append(
                {
                    "branch": branch,
                    "split": split,
                    "index": idx,
                    "wnd_id": dataset_row.get("meta", {}).get("wnd_id"),
                    "gold_event_types": sorted(gold_types),
                    "main_role": main_role(gold_events),
                    "event_type_rarity": sample_event_type_rarity(feature_row, train_stats),
                    "role_signature_rarity": sample_role_signature_rarity(feature_row, train_stats),
                    "confusion_score": sample_confusion_score(feature_row, schema_by_type),
                    "selected_type_overlap": bool(gold_types & selected_types),
                    "selected_signature_overlap": bool(gold_signatures & selected_signatures),
                    "valid_json": pred_row["valid_json"],
                    "latency_sec": pred_row["latency_sec"],
                    "trigger_f1": pred_row["trigger_f1"],
                    "argument_f1": pred_row["argument_f1"],
                    "event_f1": pred_row["event_f1"],
                    "trigger_exact": pred_row["trigger_f1"] >= 0.999999,
                    "argument_exact": pred_row["argument_f1"] >= 0.999999,
                    "event_exact": pred_row["event_f1"] >= 0.999999,
                    "trigger_correct_argument_wrong": pred_row["trigger_f1"] >= 0.999999
                    and pred_row["argument_f1"] < 0.999999,
                    "gold_argument_count": len(gold_args),
                    "pred_argument_count": len(pred_args),
                    "argument_tp": len(gold_args & pred_args),
                    "argument_fp": len(pred_args - gold_args),
                    "argument_fn": len(gold_args - pred_args),
                    "gold_trigger_count": len(gold_triggers),
                    "pred_trigger_count": len(pred_triggers),
                }
            )

        thresholds = {
            "event_type_rarity": median_threshold(raw_records, "event_type_rarity"),
            "role_signature_rarity": median_threshold(raw_records, "role_signature_rarity"),
            "confusion_score": median_threshold(raw_records, "confusion_score"),
        }
        records = []
        for record in raw_records:
            item = dict(record)
            item["event_type_rarity_bucket"] = bucket_label(item["event_type_rarity"], thresholds["event_type_rarity"])
            item["role_signature_rarity_bucket"] = bucket_label(
                item["role_signature_rarity"], thresholds["role_signature_rarity"]
            )
            item["confusion_score_bucket"] = bucket_label(item["confusion_score"], thresholds["confusion_score"])
            item["selected_signature_bucket"] = (
                "selected_signature_overlap" if item["selected_signature_overlap"] else "no_selected_signature_overlap"
            )
            records.append(item)
        records_by_split[split] = {"thresholds": thresholds, "records": records}

    return {
        "branch": branch,
        "family": family,
        "run_name": run_name,
        "selected_counts": {
            "wnd_ids": len(selected_wnd_ids),
            "event_types": len(selected_types),
            "role_signatures": len(selected_signatures),
        },
        "splits": records_by_split,
    }


def summarize_records(records):
    return {
        "n": len(records),
        "json_valid_rate": avg([1.0 if row["valid_json"] else 0.0 for row in records]),
        "trigger_f1": avg([row["trigger_f1"] for row in records]),
        "argument_f1": avg([row["argument_f1"] for row in records]),
        "event_f1": avg([row["event_f1"] for row in records]),
        "trigger_correct_argument_wrong_rate": avg(
            [1.0 if row["trigger_correct_argument_wrong"] else 0.0 for row in records]
        ),
        "argument_fn": avg([row["argument_fn"] for row in records]),
        "argument_fp": avg([row["argument_fp"] for row in records]),
    }


def summarize_pair(aux_records, dup_records):
    rows = []
    for aux, dup in zip(aux_records, dup_records):
        if aux["index"] != dup["index"] or aux["wnd_id"] != dup["wnd_id"]:
            raise ValueError("pair records are not aligned")
        row = {
            "index": aux["index"],
            "wnd_id": aux["wnd_id"],
            "main_role": aux["main_role"],
            "event_type_rarity_bucket": aux["event_type_rarity_bucket"],
            "role_signature_rarity_bucket": aux["role_signature_rarity_bucket"],
            "confusion_score_bucket": aux["confusion_score_bucket"],
            "selected_signature_bucket": aux["selected_signature_bucket"],
            "argument_delta": aux["argument_f1"] - dup["argument_f1"],
            "event_delta": aux["event_f1"] - dup["event_f1"],
            "trigger_delta": aux["trigger_f1"] - dup["trigger_f1"],
            "argument_win": compare_metric(aux, dup, "argument_f1"),
            "event_win": compare_metric(aux, dup, "event_f1"),
            "trigger_win": compare_metric(aux, dup, "trigger_f1"),
            "argument_exact_category": exact_category(aux["argument_exact"], dup["argument_exact"]),
            "event_exact_category": exact_category(aux["event_exact"], dup["event_exact"]),
            "aux_trigger_correct_argument_wrong": aux["trigger_correct_argument_wrong"],
            "directdup_trigger_correct_argument_wrong": dup["trigger_correct_argument_wrong"],
            "aux_argument_fn": aux["argument_fn"],
            "directdup_argument_fn": dup["argument_fn"],
            "aux_argument_fp": aux["argument_fp"],
            "directdup_argument_fp": dup["argument_fp"],
        }
        rows.append(row)
    return rows


def aggregate_pair_rows(rows):
    n = len(rows)
    return {
        "n": n,
        "argument_delta": avg([row["argument_delta"] for row in rows]),
        "event_delta": avg([row["event_delta"] for row in rows]),
        "trigger_delta": avg([row["trigger_delta"] for row in rows]),
        "argument_win_counts": dict(Counter(row["argument_win"] for row in rows)),
        "event_win_counts": dict(Counter(row["event_win"] for row in rows)),
        "argument_exact_categories": dict(Counter(row["argument_exact_category"] for row in rows)),
        "event_exact_categories": dict(Counter(row["event_exact_category"] for row in rows)),
        "aux_trigger_correct_argument_wrong_rate": avg(
            [1.0 if row["aux_trigger_correct_argument_wrong"] else 0.0 for row in rows]
        ),
        "directdup_trigger_correct_argument_wrong_rate": avg(
            [1.0 if row["directdup_trigger_correct_argument_wrong"] else 0.0 for row in rows]
        ),
        "argument_fn_delta": avg([row["aux_argument_fn"] - row["directdup_argument_fn"] for row in rows]),
        "argument_fp_delta": avg([row["aux_argument_fp"] - row["directdup_argument_fp"] for row in rows]),
    }


def aggregate_by(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: aggregate_pair_rows(items) for name, items in sorted(groups.items())}


def build_pair_payload(branch_payloads, pairs):
    payload = {}
    for pair in pairs:
        aux_branch, dup_branch = pair.split("=", 1)
        pair_key = f"{aux_branch}_vs_{dup_branch}"
        payload[pair_key] = {}
        for split in SPLITS:
            aux_records = branch_payloads[aux_branch]["splits"][split]["records"]
            dup_records = branch_payloads[dup_branch]["splits"][split]["records"]
            pair_rows = summarize_pair(aux_records, dup_records)
            payload[pair_key][split] = {
                "overall": aggregate_pair_rows(pair_rows),
                "by_bucket": {
                    "event_type_rarity_bucket": aggregate_by(pair_rows, "event_type_rarity_bucket"),
                    "role_signature_rarity_bucket": aggregate_by(pair_rows, "role_signature_rarity_bucket"),
                    "confusion_score_bucket": aggregate_by(pair_rows, "confusion_score_bucket"),
                    "selected_signature_bucket": aggregate_by(pair_rows, "selected_signature_bucket"),
                    "main_role": aggregate_by(pair_rows, "main_role"),
                },
                "rows": pair_rows,
            }
    return payload


def render_overall_table(branch_payloads):
    lines = [
        "| branch | split | n | json | trigger_f1 | argument_f1 | event_f1 | trig_ok_arg_wrong | arg_fn | arg_fp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for branch, payload in branch_payloads.items():
        for split in SPLITS:
            summary = summarize_records(payload["splits"][split]["records"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{branch}`",
                        split,
                        str(summary["n"]),
                        fmt(summary["json_valid_rate"]),
                        fmt(summary["trigger_f1"]),
                        fmt(summary["argument_f1"]),
                        fmt(summary["event_f1"]),
                        fmt(summary["trigger_correct_argument_wrong_rate"]),
                        fmt(summary["argument_fn"]),
                        fmt(summary["argument_fp"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def render_pair_table(pair_payload):
    lines = [
        "| pair | split | n | arg_delta | event_delta | trigger_delta | arg_fn_delta | arg_fp_delta | aux_arg_win | dup_arg_win | arg_both_correct | aux_only_arg | dup_only_arg | both_arg_wrong |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair, split_payload in pair_payload.items():
        for split in SPLITS:
            overall = split_payload[split]["overall"]
            arg_wins = overall["argument_win_counts"]
            arg_exact = overall["argument_exact_categories"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{pair}`",
                        split,
                        str(overall["n"]),
                        fmt(overall["argument_delta"]),
                        fmt(overall["event_delta"]),
                        fmt(overall["trigger_delta"]),
                        fmt(overall["argument_fn_delta"]),
                        fmt(overall["argument_fp_delta"]),
                        str(arg_wins.get("aux_win", 0)),
                        str(arg_wins.get("directdup_win", 0)),
                        str(arg_exact.get("both_correct", 0)),
                        str(arg_exact.get("aux_only_correct", 0)),
                        str(arg_exact.get("directdup_only_correct", 0)),
                        str(arg_exact.get("both_wrong", 0)),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def render_bucket_focus(pair_payload):
    sections = []
    for pair, split_payload in pair_payload.items():
        for split in SPLITS:
            sections.append(f"### {pair} / {split}")
            sections.append("| bucket_type | bucket | n | arg_delta | event_delta | arg_fn_delta | arg_fp_delta | aux_arg_win | dup_arg_win | aux_trig_ok_arg_wrong | dup_trig_ok_arg_wrong |")
            sections.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for bucket_key in ["confusion_score_bucket", "role_signature_rarity_bucket", "selected_signature_bucket"]:
                for bucket_name, summary in split_payload[split]["by_bucket"][bucket_key].items():
                    wins = summary["argument_win_counts"]
                    sections.append(
                        "| "
                        + " | ".join(
                            [
                                bucket_key,
                                bucket_name,
                                str(summary["n"]),
                                fmt(summary["argument_delta"]),
                                fmt(summary["event_delta"]),
                                fmt(summary["argument_fn_delta"]),
                                fmt(summary["argument_fp_delta"]),
                                str(wins.get("aux_win", 0)),
                                str(wins.get("directdup_win", 0)),
                                fmt(summary["aux_trigger_correct_argument_wrong_rate"]),
                                fmt(summary["directdup_trigger_correct_argument_wrong_rate"]),
                            ]
                        )
                        + " |"
                    )
            sections.append("")
    return "\n".join(sections)


def render_markdown(payload):
    lines = [
        "# Stage2 SAR Error Attribution Analysis",
        "",
        "## Scope",
        "",
        f"- Branches: {', '.join(f'`{branch}`' for branch in payload['branches'])}",
        f"- Pairs: {', '.join(f'`{pair}`' for pair in payload['pairs'])}",
        "- Correctness categories use exact row-level F1 = 1.0.",
        "- `trig_ok_arg_wrong` marks rows with exact trigger F1 but non-exact argument F1.",
        "",
        "## Branch Error Profile",
        "",
        render_overall_table(payload["branch_payloads"]),
        "",
        "## Pairwise Attribution",
        "",
        render_pair_table(payload["pairwise"]),
        "",
        "## Bucket Focus",
        "",
        render_bucket_focus(payload["pairwise"]),
    ]
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_path", default="data/schema/richere-en.event_schema.json")
    parser.add_argument(
        "--direct_train_jsonl",
        default="data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_train_pos.jsonl",
    )
    parser.add_argument("--dataset_dir", default="data/stage2_cot_datasets")
    parser.add_argument("--best_eval_root", default="outputs/stage2_full_sft_runs_stepmatch_best_eval_user")
    parser.add_argument("--dataset_prefix", default="richere_balanced_split1_oracle_mixed_noise_top10_shuffle")
    parser.add_argument("--run_prefix", default="richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle")
    parser.add_argument("--branches", nargs="+", default=DEFAULT_BRANCHES)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-05_stage2_sar_error_attribution_analysis.json")
    parser.add_argument("--output_md", default="reports/2026-05-05_stage2_sar_error_attribution_analysis.md")
    args = parser.parse_args()

    schema_by_type = load_schema_map(Path(args.schema_path))
    direct_train_rows = load_jsonl(Path(args.direct_train_jsonl))
    train_stats = build_confrare_stats(direct_train_rows)

    branch_payloads = {}
    for branch in args.branches:
        branch_payloads[branch] = build_branch_records(
            branch=branch,
            dataset_prefix=args.dataset_prefix,
            run_prefix=args.run_prefix,
            dataset_dir=Path(args.dataset_dir),
            best_eval_root=Path(args.best_eval_root),
            direct_train_rows=direct_train_rows,
            schema_by_type=schema_by_type,
            train_stats=train_stats,
        )

    pairwise = build_pair_payload(branch_payloads, args.pairs)
    payload = {
        "schema_path": args.schema_path,
        "direct_train_jsonl": args.direct_train_jsonl,
        "dataset_dir": args.dataset_dir,
        "best_eval_root": args.best_eval_root,
        "branches": args.branches,
        "pairs": args.pairs,
        "branch_payloads": branch_payloads,
        "pairwise": pairwise,
    }

    write_json(Path(args.output_json), payload)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
