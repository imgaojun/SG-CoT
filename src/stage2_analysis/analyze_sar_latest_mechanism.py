import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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


SPLITS = ["test", "test_seen", "test_unseen"]
SUBSETS = ["all", "positive", "negative"]


@dataclass(frozen=True)
class BranchSpec:
    label: str
    dataset_dir: str
    family: str
    suffix: str
    run_name: str
    selected_meta: str | None = None


@dataclass(frozen=True)
class PairSpec:
    label: str
    left: str
    right: str


@dataclass(frozen=True)
class GroupSpec:
    label: str
    direct_train_jsonl: str
    branches: list[BranchSpec]
    pairs: list[PairSpec]


def avg(values):
    return sum(values) / len(values) if values else 0.0


def fmt(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def prediction_dir(best_eval_root: Path, run_name: str, split: str):
    suffix = "test_argfirst" if split == "test" else f"{split}_argfirst"
    return best_eval_root / f"{run_name}_{suffix}"


def dataset_path(spec: BranchSpec, split: str):
    return Path(spec.dataset_dir) / f"{spec.family}_{split}{spec.suffix}.jsonl"


def prediction_events(pred_row):
    payload = pred_row.get("predicted")
    if not isinstance(payload, dict):
        return []
    events = payload.get("events", [])
    return events if isinstance(events, list) else []


def trigger_tuple(event):
    trigger = event.get("trigger", {}) if isinstance(event, dict) else {}
    if not isinstance(trigger, dict):
        trigger = {}
    return (event.get("event_type"), trigger.get("start"), trigger.get("end"))


def trigger_tuples(events):
    return {trigger_tuple(event) for event in events if isinstance(event, dict)}


def argument_tuples(events):
    items = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type, trigger_start, trigger_end = trigger_tuple(event)
        args = event.get("arguments", [])
        if not isinstance(args, list):
            continue
        for arg in args:
            if not isinstance(arg, dict):
                continue
            items.add((event_type, trigger_start, trigger_end, arg.get("role"), arg.get("start"), arg.get("end")))
    return items


def main_role(events):
    counts = Counter()
    for event in events:
        if not isinstance(event, dict):
            continue
        for arg in event.get("arguments", []):
            if isinstance(arg, dict) and arg.get("role"):
                counts[arg["role"]] += 1
    return counts.most_common(1)[0][0] if counts else "NO_ROLE"


def selected_signature_sets(selected_meta_path: str | None, direct_train_rows):
    if not selected_meta_path:
        return set(), set()
    meta = load_json(Path(selected_meta_path))
    selected_wnd_ids = set(meta.get("selected_aux_wnd_ids", []))
    selected_types = set()
    selected_signatures = set()
    for row in direct_train_rows:
        if row_id(row) not in selected_wnd_ids:
            continue
        for event in parse_output_events(row):
            selected_types.add(event["event_type"])
            selected_signatures.add(event_role_signature(event))
    return selected_types, selected_signatures


def bucket_threshold(records, key):
    positives = [record[key] for record in records if record["subset_positive"]]
    values = sorted(positives if positives else [record[key] for record in records])
    if not values:
        return 0.0
    return values[len(values) // 2]


def bucket_label(value, threshold):
    return "high" if value >= threshold else "low"


def build_branch_records(spec: BranchSpec, *, best_eval_root: Path, direct_train_rows, schema_by_type, train_stats):
    selected_types, selected_signatures = selected_signature_sets(spec.selected_meta, direct_train_rows)
    records_by_split = {}
    for split in SPLITS:
        dataset_rows = load_jsonl(dataset_path(spec, split))
        pred_rows = load_jsonl(prediction_dir(best_eval_root, spec.run_name, split) / "predictions.jsonl")
        if len(dataset_rows) != len(pred_rows):
            raise ValueError(f"{spec.label}/{split}: dataset rows != prediction rows")

        raw_records = []
        for idx, (dataset_row, pred_row) in enumerate(zip(dataset_rows, pred_rows)):
            if dataset_row["input"].strip() != pred_row["input"].strip():
                raise ValueError(f"{spec.label}/{split}: input mismatch at row {idx}")
            gold_events = parse_output_events(dataset_row)
            pred_events = prediction_events(pred_row)
            gold_types = {event["event_type"] for event in gold_events}
            gold_signatures = {event_role_signature(event) for event in gold_events}
            gold_args = argument_tuples(gold_events)
            pred_args = argument_tuples(pred_events)
            gold_triggers = trigger_tuples(gold_events)
            pred_triggers = trigger_tuples(pred_events)
            feature_row = {
                "output": json.dumps({"events": gold_events}, ensure_ascii=False),
                "meta": dataset_row.get("meta", {}),
            }
            raw_records.append(
                {
                    "branch": spec.label,
                    "split": split,
                    "index": idx,
                    "wnd_id": dataset_row.get("meta", {}).get("wnd_id"),
                    "subset_positive": bool(gold_events),
                    "gold_event_types": sorted(gold_types),
                    "main_role": main_role(gold_events),
                    "event_type_rarity": sample_event_type_rarity(feature_row, train_stats),
                    "role_signature_rarity": sample_role_signature_rarity(feature_row, train_stats),
                    "confusion_score": sample_confusion_score(feature_row, schema_by_type),
                    "selected_type_overlap": bool(gold_types & selected_types),
                    "selected_signature_overlap": bool(gold_signatures & selected_signatures),
                    "valid_json": bool(pred_row["valid_json"]),
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
                    "trigger_fp": len(pred_triggers - gold_triggers),
                    "trigger_fn": len(gold_triggers - pred_triggers),
                }
            )

        thresholds = {
            "event_type_rarity": bucket_threshold(raw_records, "event_type_rarity"),
            "role_signature_rarity": bucket_threshold(raw_records, "role_signature_rarity"),
            "confusion_score": bucket_threshold(raw_records, "confusion_score"),
        }
        records = []
        for record in raw_records:
            item = dict(record)
            item["subset"] = "positive" if item["subset_positive"] else "negative"
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
        "label": spec.label,
        "family": spec.family,
        "run_name": spec.run_name,
        "selected_meta": spec.selected_meta,
        "splits": records_by_split,
    }


def filter_subset(records, subset):
    if subset == "all":
        return records
    if subset == "positive":
        return [record for record in records if record["subset_positive"]]
    if subset == "negative":
        return [record for record in records if not record["subset_positive"]]
    raise ValueError(f"unknown subset: {subset}")


def summarize_records(records):
    return {
        "n": len(records),
        "json_valid_rate": avg([1.0 if row["valid_json"] else 0.0 for row in records]),
        "trigger_f1": avg([row["trigger_f1"] for row in records]),
        "argument_f1": avg([row["argument_f1"] for row in records]),
        "event_f1": avg([row["event_f1"] for row in records]),
        "latency_sec": avg([row["latency_sec"] for row in records]),
        "trigger_correct_argument_wrong_rate": avg(
            [1.0 if row["trigger_correct_argument_wrong"] else 0.0 for row in records]
        ),
        "argument_fn": avg([row["argument_fn"] for row in records]),
        "argument_fp": avg([row["argument_fp"] for row in records]),
        "trigger_fn": avg([row["trigger_fn"] for row in records]),
        "trigger_fp": avg([row["trigger_fp"] for row in records]),
        "pred_argument_count": avg([row["pred_argument_count"] for row in records]),
        "gold_argument_count": avg([row["gold_argument_count"] for row in records]),
    }


def metric_win(left, right, key):
    if left[key] > right[key]:
        return "left_win"
    if left[key] < right[key]:
        return "right_win"
    return "tie"


def exact_category(left, right, key):
    lval = left[key]
    rval = right[key]
    if lval and rval:
        return "both_correct"
    if lval:
        return "left_only_correct"
    if rval:
        return "right_only_correct"
    return "both_wrong"


def pair_rows(left_records, right_records):
    rows = []
    for left, right in zip(left_records, right_records):
        if left["index"] != right["index"] or left["wnd_id"] != right["wnd_id"]:
            raise ValueError("pair records are not aligned")
        rows.append(
            {
                "index": left["index"],
                "wnd_id": left["wnd_id"],
                "subset": left["subset"],
                "subset_positive": left["subset_positive"],
                "main_role": left["main_role"],
                "event_type_rarity_bucket": left["event_type_rarity_bucket"],
                "role_signature_rarity_bucket": left["role_signature_rarity_bucket"],
                "confusion_score_bucket": left["confusion_score_bucket"],
                "selected_signature_bucket": left["selected_signature_bucket"],
                "argument_delta": left["argument_f1"] - right["argument_f1"],
                "event_delta": left["event_f1"] - right["event_f1"],
                "trigger_delta": left["trigger_f1"] - right["trigger_f1"],
                "argument_win": metric_win(left, right, "argument_f1"),
                "event_win": metric_win(left, right, "event_f1"),
                "trigger_win": metric_win(left, right, "trigger_f1"),
                "argument_exact_category": exact_category(left, right, "argument_exact"),
                "event_exact_category": exact_category(left, right, "event_exact"),
                "left_trigger_correct_argument_wrong": left["trigger_correct_argument_wrong"],
                "right_trigger_correct_argument_wrong": right["trigger_correct_argument_wrong"],
                "argument_fn_delta": left["argument_fn"] - right["argument_fn"],
                "argument_fp_delta": left["argument_fp"] - right["argument_fp"],
                "trigger_fn_delta": left["trigger_fn"] - right["trigger_fn"],
                "trigger_fp_delta": left["trigger_fp"] - right["trigger_fp"],
            }
        )
    return rows


def summarize_pair_rows(rows):
    wins = Counter(row["argument_win"] for row in rows)
    exact = Counter(row["argument_exact_category"] for row in rows)
    event_exact = Counter(row["event_exact_category"] for row in rows)
    return {
        "n": len(rows),
        "argument_delta": avg([row["argument_delta"] for row in rows]),
        "event_delta": avg([row["event_delta"] for row in rows]),
        "trigger_delta": avg([row["trigger_delta"] for row in rows]),
        "argument_fn_delta": avg([row["argument_fn_delta"] for row in rows]),
        "argument_fp_delta": avg([row["argument_fp_delta"] for row in rows]),
        "trigger_fn_delta": avg([row["trigger_fn_delta"] for row in rows]),
        "trigger_fp_delta": avg([row["trigger_fp_delta"] for row in rows]),
        "left_arg_win": wins.get("left_win", 0),
        "right_arg_win": wins.get("right_win", 0),
        "arg_tie": wins.get("tie", 0),
        "arg_both_correct": exact.get("both_correct", 0),
        "left_only_arg": exact.get("left_only_correct", 0),
        "right_only_arg": exact.get("right_only_correct", 0),
        "arg_both_wrong": exact.get("both_wrong", 0),
        "event_both_correct": event_exact.get("both_correct", 0),
        "left_only_event": event_exact.get("left_only_correct", 0),
        "right_only_event": event_exact.get("right_only_correct", 0),
        "event_both_wrong": event_exact.get("both_wrong", 0),
        "left_trigger_correct_argument_wrong_rate": avg(
            [1.0 if row["left_trigger_correct_argument_wrong"] else 0.0 for row in rows]
        ),
        "right_trigger_correct_argument_wrong_rate": avg(
            [1.0 if row["right_trigger_correct_argument_wrong"] else 0.0 for row in rows]
        ),
    }


def aggregate_pair_by(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: summarize_pair_rows(items) for name, items in sorted(groups.items())}


def build_group_payload(group: GroupSpec, best_eval_root: Path, schema_by_type):
    direct_train_rows = load_jsonl(Path(group.direct_train_jsonl))
    train_stats = build_confrare_stats(direct_train_rows)
    branch_payloads = {
        spec.label: build_branch_records(
            spec,
            best_eval_root=best_eval_root,
            direct_train_rows=direct_train_rows,
            schema_by_type=schema_by_type,
            train_stats=train_stats,
        )
        for spec in group.branches
    }

    branch_summaries = {}
    for branch, payload in branch_payloads.items():
        branch_summaries[branch] = {}
        for split in SPLITS:
            branch_summaries[branch][split] = {}
            records = payload["splits"][split]["records"]
            for subset in SUBSETS:
                branch_summaries[branch][split][subset] = summarize_records(filter_subset(records, subset))

    pair_payloads = {}
    for pair in group.pairs:
        left_payload = branch_payloads[pair.left]
        right_payload = branch_payloads[pair.right]
        pair_payloads[pair.label] = {}
        for split in SPLITS:
            rows = pair_rows(left_payload["splits"][split]["records"], right_payload["splits"][split]["records"])
            pair_payloads[pair.label][split] = {}
            for subset in SUBSETS:
                subset_rows = [row for row in rows if subset == "all" or row["subset"] == subset]
                pair_payloads[pair.label][split][subset] = {
                    "overall": summarize_pair_rows(subset_rows),
                    "by_bucket": {
                        "confusion_score_bucket": aggregate_pair_by(subset_rows, "confusion_score_bucket"),
                        "role_signature_rarity_bucket": aggregate_pair_by(subset_rows, "role_signature_rarity_bucket"),
                        "selected_signature_bucket": aggregate_pair_by(subset_rows, "selected_signature_bucket"),
                        "main_role": aggregate_pair_by(subset_rows, "main_role"),
                    },
                }

    return {
        "label": group.label,
        "direct_train_jsonl": group.direct_train_jsonl,
        "branches": [spec.label for spec in group.branches],
        "pairs": [pair.label for pair in group.pairs],
        "branch_summaries": branch_summaries,
        "pairwise": pair_payloads,
    }


def branch_spec(branch):
    prefix = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
    run_prefix = "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle"
    family = f"{prefix}_sar_{branch}"
    return BranchSpec(
        label=branch,
        dataset_dir="data/stage2_cot_datasets",
        family=family,
        suffix="_pos",
        run_name=f"{run_prefix}_sar_{branch}_full",
        selected_meta=f"data/stage2_cot_datasets/{family}_train_pos.meta.json",
    )


def build_groups():
    balanced_train = "data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_train_pos.jsonl"
    balanced_prefix = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
    qwen4_run_prefix = "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle"
    full_prefix = "richere_full_split1_oracle_mixed_noise_top10_shuffle"
    full_run_prefix = "richere_full_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle"
    confrare10_family = f"{balanced_prefix}_sar_confrare10_typerolelite"
    confrare10_dup_family = f"{balanced_prefix}_sar_confrare10_directdup"
    full_ty_family = f"{full_prefix}_sar_confrare10_typerolelite"
    full_dup_family = f"{full_prefix}_sar_confrare10_directdup"

    return [
        GroupSpec(
            label="positive_only_qwen3_1_7b_matched_pairs",
            direct_train_jsonl=balanced_train,
            branches=[
                branch_spec("confrare10_typerolelite"),
                branch_spec("confrare10_directdup"),
                branch_spec("confrare20_typerolelite"),
                branch_spec("confrare20_directdup"),
                branch_spec("confrole10_typerolelite"),
                branch_spec("confrole10_directdup"),
                branch_spec("confrole10_typeonlylite"),
                branch_spec("confrole10_roleonlylite"),
                branch_spec("confrole5_typerolelite"),
                branch_spec("confrole5_directdup"),
                branch_spec("confrole10_hybrid_dupaux_typerolelite"),
                branch_spec("confrole10_directdup2x"),
            ],
            pairs=[
                PairSpec("confrare10_typerolelite_vs_directdup", "confrare10_typerolelite", "confrare10_directdup"),
                PairSpec("confrare20_typerolelite_vs_directdup", "confrare20_typerolelite", "confrare20_directdup"),
                PairSpec("confrole10_typerolelite_vs_directdup", "confrole10_typerolelite", "confrole10_directdup"),
                PairSpec("confrole10_typeonlylite_vs_directdup", "confrole10_typeonlylite", "confrole10_directdup"),
                PairSpec("confrole10_roleonlylite_vs_directdup", "confrole10_roleonlylite", "confrole10_directdup"),
                PairSpec("confrole5_typerolelite_vs_directdup", "confrole5_typerolelite", "confrole5_directdup"),
                PairSpec(
                    "confrole10_hybrid_dupaux_typerolelite_vs_directdup2x",
                    "confrole10_hybrid_dupaux_typerolelite",
                    "confrole10_directdup2x",
                ),
            ],
        ),
        GroupSpec(
            label="full_data_qwen3_1_7b",
            direct_train_jsonl=f"data/stage2_formal_datasets/{full_prefix}_train.jsonl",
            branches=[
                BranchSpec(
                    label="full_direct",
                    dataset_dir="data/stage2_formal_datasets",
                    family=full_prefix,
                    suffix="",
                    run_name=f"{full_run_prefix}_direct_full",
                ),
                BranchSpec(
                    label="full_confrare10_typerolelite",
                    dataset_dir="data/stage2_cot_datasets",
                    family=full_ty_family,
                    suffix="",
                    run_name=f"{full_run_prefix}_sar_confrare10_typerolelite_full",
                    selected_meta=f"data/stage2_cot_datasets/{full_ty_family}_train.meta.json",
                ),
                BranchSpec(
                    label="full_confrare10_directdup",
                    dataset_dir="data/stage2_cot_datasets",
                    family=full_dup_family,
                    suffix="",
                    run_name=f"{full_run_prefix}_sar_confrare10_directdup_full",
                    selected_meta=f"data/stage2_cot_datasets/{full_dup_family}_train.meta.json",
                ),
            ],
            pairs=[
                PairSpec("full_typerolelite_vs_directdup", "full_confrare10_typerolelite", "full_confrare10_directdup"),
                PairSpec("full_typerolelite_vs_direct", "full_confrare10_typerolelite", "full_direct"),
                PairSpec("full_directdup_vs_direct", "full_confrare10_directdup", "full_direct"),
            ],
        ),
        GroupSpec(
            label="qwen3_4b_cross_model",
            direct_train_jsonl=balanced_train,
            branches=[
                BranchSpec(
                    label="qwen4_direct",
                    dataset_dir="data/stage2_formal_datasets",
                    family=balanced_prefix,
                    suffix="_pos",
                    run_name=f"{qwen4_run_prefix}_direct_full",
                ),
                BranchSpec(
                    label="qwen4_confrare10_typerolelite",
                    dataset_dir="data/stage2_cot_datasets",
                    family=confrare10_family,
                    suffix="_pos",
                    run_name=f"{qwen4_run_prefix}_sar_confrare10_typerolelite_full",
                    selected_meta=f"data/stage2_cot_datasets/{confrare10_family}_train_pos.meta.json",
                ),
                BranchSpec(
                    label="qwen4_confrare10_directdup",
                    dataset_dir="data/stage2_cot_datasets",
                    family=confrare10_dup_family,
                    suffix="_pos",
                    run_name=f"{qwen4_run_prefix}_sar_confrare10_directdup_full",
                    selected_meta=f"data/stage2_cot_datasets/{confrare10_dup_family}_train_pos.meta.json",
                ),
            ],
            pairs=[
                PairSpec("qwen4_typerolelite_vs_directdup", "qwen4_confrare10_typerolelite", "qwen4_confrare10_directdup"),
                PairSpec("qwen4_typerolelite_vs_direct", "qwen4_confrare10_typerolelite", "qwen4_direct"),
                PairSpec("qwen4_directdup_vs_direct", "qwen4_confrare10_directdup", "qwen4_direct"),
            ],
        ),
    ]


def render_pair_table(payload, focus_subset="positive"):
    lines = [
        "| group | pair | split | subset | n | arg_delta | event_delta | trigger_delta | arg_fn_delta | arg_fp_delta | left_arg_win | right_arg_win | left_only_arg | right_only_arg |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name, group_payload in payload["groups"].items():
        for pair_name, split_payload in group_payload["pairwise"].items():
            for split in SPLITS:
                if focus_subset not in split_payload[split]:
                    continue
                summary = split_payload[split][focus_subset]["overall"]
                if summary["n"] == 0:
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{group_name}`",
                            f"`{pair_name}`",
                            split,
                            focus_subset,
                            str(summary["n"]),
                            fmt(summary["argument_delta"]),
                            fmt(summary["event_delta"]),
                            fmt(summary["trigger_delta"]),
                            fmt(summary["argument_fn_delta"]),
                            fmt(summary["argument_fp_delta"]),
                            str(summary["left_arg_win"]),
                            str(summary["right_arg_win"]),
                            str(summary["left_only_arg"]),
                            str(summary["right_only_arg"]),
                        ]
                    )
                    + " |"
                )
    return "\n".join(lines)


def render_branch_subset_table(payload, group_filter):
    lines = [
        "| group | branch | split | subset | n | json | trigger_f1 | argument_f1 | event_f1 | arg_fn | arg_fp | trig_ok_arg_wrong |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name, group_payload in payload["groups"].items():
        if group_filter and group_name != group_filter:
            continue
        for branch, split_payload in group_payload["branch_summaries"].items():
            for split in SPLITS:
                for subset in SUBSETS:
                    summary = split_payload[split][subset]
                    if summary["n"] == 0:
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                f"`{group_name}`",
                                f"`{branch}`",
                                split,
                                subset,
                                str(summary["n"]),
                                fmt(summary["json_valid_rate"]),
                                fmt(summary["trigger_f1"]),
                                fmt(summary["argument_f1"]),
                                fmt(summary["event_f1"]),
                                fmt(summary["argument_fn"]),
                                fmt(summary["argument_fp"]),
                                fmt(summary["trigger_correct_argument_wrong_rate"]),
                            ]
                        )
                        + " |"
                    )
    return "\n".join(lines)


def render_bucket_focus(payload):
    rows = []
    focus_pairs = {
        ("positive_only_qwen3_1_7b_matched_pairs", "confrare10_typerolelite_vs_directdup"),
        ("positive_only_qwen3_1_7b_matched_pairs", "confrole10_typerolelite_vs_directdup"),
        ("full_data_qwen3_1_7b", "full_typerolelite_vs_directdup"),
        ("qwen3_4b_cross_model", "qwen4_typerolelite_vs_directdup"),
    }
    for group_name, pair_name in focus_pairs:
        group_payload = payload["groups"][group_name]
        pair_payload = group_payload["pairwise"][pair_name]
        for split in SPLITS:
            bucket_payload = pair_payload[split]["positive"]["by_bucket"]
            for bucket_key in ["confusion_score_bucket", "role_signature_rarity_bucket", "selected_signature_bucket"]:
                for bucket_name, summary in bucket_payload[bucket_key].items():
                    if summary["n"] == 0:
                        continue
                    rows.append(
                        [
                            f"`{group_name}`",
                            f"`{pair_name}`",
                            split,
                            bucket_key,
                            bucket_name,
                            str(summary["n"]),
                            fmt(summary["argument_delta"]),
                            fmt(summary["event_delta"]),
                            fmt(summary["argument_fn_delta"]),
                            fmt(summary["argument_fp_delta"]),
                            str(summary["left_arg_win"]),
                            str(summary["right_arg_win"]),
                        ]
                    )
    lines = [
        "| group | pair | split | bucket_type | bucket | n | arg_delta | event_delta | arg_fn_delta | arg_fp_delta | left_arg_win | right_arg_win |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def value_at(payload, group, pair, split, subset, key):
    return payload["groups"][group]["pairwise"][pair][split][subset]["overall"][key]


def render_markdown(payload):
    confrare10_unseen_arg = value_at(
        payload,
        "positive_only_qwen3_1_7b_matched_pairs",
        "confrare10_typerolelite_vs_directdup",
        "test_unseen",
        "positive",
        "argument_delta",
    )
    confrare10_unseen_event = value_at(
        payload,
        "positive_only_qwen3_1_7b_matched_pairs",
        "confrare10_typerolelite_vs_directdup",
        "test_unseen",
        "positive",
        "event_delta",
    )
    full_pos_arg = value_at(
        payload, "full_data_qwen3_1_7b", "full_typerolelite_vs_directdup", "test", "positive", "argument_delta"
    )
    full_neg_fp = value_at(
        payload, "full_data_qwen3_1_7b", "full_typerolelite_vs_direct", "test", "negative", "argument_fp_delta"
    )
    qwen4_unseen_arg = value_at(
        payload, "qwen3_4b_cross_model", "qwen4_typerolelite_vs_directdup", "test_unseen", "positive", "argument_delta"
    )
    lines = [
        "# 2026-05-07 Stage2 SAR Latest Mechanism Analysis",
        "",
        "## Scope",
        "",
        "- No new training; analysis uses completed formal prediction JSONL files.",
        "- Main focus: auxiliary target vs same-selection direct replay, plus full-data and Qwen3-4B sensitivity.",
        "- For full-data runs, metrics are reported on `all`, `positive`, and `negative` subsets to expose negative-row calibration effects.",
        "",
        "## Key Findings",
        "",
        f"- Positive-only `confrare10_typerolelite` keeps the cleanest auxiliary-over-replay unseen signal: test_unseen argument `{fmt(confrare10_unseen_arg)}`, event `{fmt(confrare10_unseen_event)}`.",
        f"- Full-data SAR is not globally best, but on positive rows `full_typerolelite_vs_directdup` test argument delta is `{fmt(full_pos_arg)}`; the full-data story is therefore mixed rather than simply negative.",
        f"- Full-data `typerolelite` vs direct has negative-row argument-FP delta `{fmt(full_neg_fp)}` on test; direct remains the stronger full-distribution baseline.",
        f"- Qwen3-4B does not replicate the Qwen3-1.7B mechanism: `qwen4_typerolelite_vs_directdup` test_unseen argument delta is `{fmt(qwen4_unseen_arg)}`.",
        "- Bucket analysis should be read on positive rows; full-data all-row averages are dominated by negative examples.",
        "",
        "## Pairwise Positive-Row Attribution",
        "",
        render_pair_table(payload, focus_subset="positive"),
        "",
        "## Full-Data Subset Profile",
        "",
        render_branch_subset_table(payload, "full_data_qwen3_1_7b"),
        "",
        "## Bucket Focus On Positive Rows",
        "",
        render_bucket_focus(payload),
    ]
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_path", default="data/schema/richere-en.event_schema.json")
    parser.add_argument("--best_eval_root", default="outputs/stage2_full_sft_runs_stepmatch_best_eval_user")
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-07_stage2_sar_latest_mechanism_analysis.json")
    parser.add_argument("--output_md", default="reports/2026-05-07_stage2_sar_latest_mechanism_analysis.md")
    args = parser.parse_args()

    schema_by_type = load_schema_map(Path(args.schema_path))
    payload = {
        "schema_path": args.schema_path,
        "best_eval_root": args.best_eval_root,
        "groups": {},
    }
    for group in build_groups():
        payload["groups"][group.label] = build_group_payload(group, Path(args.best_eval_root), schema_by_type)

    write_json(Path(args.output_json), payload)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
