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
    "confrare20_typerolelite",
    "confrare20_directdup",
    "random20_seed21_typerolelite",
]
DEFAULT_PAIRS = ["confrare20_typerolelite=confrare20_directdup"]
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


def parse_output_payload(row):
    payload = row["output"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def branch_dataset_family(prefix: str, branch: str):
    return f"{prefix}_sar_{branch}"


def branch_run_name(run_prefix: str, branch: str):
    return f"{run_prefix}_sar_{branch}_full"


def prediction_dir(best_eval_root: Path, run_name: str, split: str):
    suffix = "test_argfirst" if split == "test" else f"{split}_argfirst"
    return best_eval_root / f"{run_name}_{suffix}"


def bucket_label(value: float, threshold: float):
    return "high" if value >= threshold else "low"


def quantile_threshold(records, key):
    values = sorted(record[key] for record in records)
    if not values:
        return 0.0
    return values[len(values) // 2]


def summarize(records):
    return {
        "num_examples": len(records),
        "json_valid_rate": avg([1.0 if record["valid_json"] else 0.0 for record in records]),
        "trigger_f1": avg([record["trigger_f1"] for record in records]),
        "argument_f1": avg([record["argument_f1"] for record in records]),
        "event_f1": avg([record["event_f1"] for record in records]),
        "avg_latency_sec": avg([record["latency_sec"] for record in records]),
    }


def summarize_by_bucket(records, bucket_key):
    groups = defaultdict(list)
    for record in records:
        groups[record[bucket_key]].append(record)
    return {name: summarize(rows) for name, rows in sorted(groups.items())}


def metric_delta(left_summary, right_summary):
    return {
        key: left_summary.get(key, 0.0) - right_summary.get(key, 0.0)
        for key in ["json_valid_rate", "trigger_f1", "argument_f1", "event_f1", "avg_latency_sec"]
    }


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


def build_records_for_branch(
    *,
    branch: str,
    prefix: str,
    model_tag: str,
    dataset_dir: Path,
    best_eval_root: Path,
    direct_train_rows,
    schema_by_type,
    train_stats,
):
    family = branch_dataset_family(prefix, branch)
    train_meta_path = dataset_dir / f"{family}_train_pos.meta.json"
    train_meta = load_json(train_meta_path)
    selected_wnd_ids = set(train_meta.get("selected_aux_wnd_ids", []))
    selected_types, selected_signatures = build_selected_signature_sets(direct_train_rows, selected_wnd_ids)
    run_name = branch_run_name(model_tag, branch)

    records_by_split = {}
    for split in SPLITS:
        dataset_rows = load_jsonl(dataset_dir / f"{family}_{split}_pos.jsonl")
        pred_rows = load_jsonl(prediction_dir(best_eval_root, run_name, split) / "predictions.jsonl")
        if len(dataset_rows) != len(pred_rows):
            raise ValueError(f"{branch}/{split}: dataset rows and prediction rows have different lengths.")

        raw_records = []
        for idx, (dataset_row, pred_row) in enumerate(zip(dataset_rows, pred_rows)):
            if dataset_row["input"].strip() != pred_row["input"].strip():
                raise ValueError(f"{branch}/{split}: input mismatch at row {idx}.")
            gold_events = parse_output_events(dataset_row)
            gold_types = {event["event_type"] for event in gold_events}
            gold_signatures = {event_role_signature(event) for event in gold_events}
            feature_row = {
                "output": json.dumps({"events": gold_events}, ensure_ascii=False),
                "meta": dataset_row.get("meta", {}),
            }
            raw_records.append(
                {
                    "branch": branch,
                    "split": split,
                    "index": idx,
                    "wnd_id": dataset_row.get("meta", {}).get("wnd_id"),
                    "gold_event_types": sorted(gold_types),
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
                }
            )

        thresholds = {
            "event_type_rarity": quantile_threshold(raw_records, "event_type_rarity"),
            "role_signature_rarity": quantile_threshold(raw_records, "role_signature_rarity"),
            "confusion_score": quantile_threshold(raw_records, "confusion_score"),
        }
        records = []
        for record in raw_records:
            item = dict(record)
            item["event_type_rarity_bucket"] = bucket_label(item["event_type_rarity"], thresholds["event_type_rarity"])
            item["role_signature_rarity_bucket"] = bucket_label(item["role_signature_rarity"], thresholds["role_signature_rarity"])
            item["confusion_score_bucket"] = bucket_label(item["confusion_score"], thresholds["confusion_score"])
            item["selected_type_bucket"] = "selected_type_overlap" if item["selected_type_overlap"] else "no_selected_type_overlap"
            item["selected_signature_bucket"] = (
                "selected_signature_overlap" if item["selected_signature_overlap"] else "no_selected_signature_overlap"
            )
            records.append(item)

        records_by_split[split] = {
            "thresholds": thresholds,
            "records": records,
            "overall": summarize(records),
            "by_bucket": {
                "event_type_rarity_bucket": summarize_by_bucket(records, "event_type_rarity_bucket"),
                "role_signature_rarity_bucket": summarize_by_bucket(records, "role_signature_rarity_bucket"),
                "confusion_score_bucket": summarize_by_bucket(records, "confusion_score_bucket"),
                "selected_type_bucket": summarize_by_bucket(records, "selected_type_bucket"),
                "selected_signature_bucket": summarize_by_bucket(records, "selected_signature_bucket"),
            },
        }

    return {
        "branch": branch,
        "family": family,
        "run_name": run_name,
        "train_meta": train_meta_path.as_posix(),
        "selected_counts": {
            "wnd_ids": len(selected_wnd_ids),
            "event_types": len(selected_types),
            "role_signatures": len(selected_signatures),
        },
        "splits": records_by_split,
    }


def pairwise_bucket_deltas(branch_payloads, pairs):
    deltas = {}
    for pair in pairs:
        aux_branch, dup_branch = pair.split("=", 1)
        pair_key = f"{aux_branch}_vs_{dup_branch}"
        deltas[pair_key] = {}
        for split in SPLITS:
            aux_split = branch_payloads[aux_branch]["splits"][split]
            dup_split = branch_payloads[dup_branch]["splits"][split]
            split_payload = {
                "overall": {
                    "aux": aux_split["overall"],
                    "directdup": dup_split["overall"],
                    "delta": metric_delta(aux_split["overall"], dup_split["overall"]),
                },
                "by_bucket": {},
            }
            for bucket_key in aux_split["by_bucket"]:
                split_payload["by_bucket"][bucket_key] = {}
                bucket_names = sorted(set(aux_split["by_bucket"][bucket_key]) | set(dup_split["by_bucket"][bucket_key]))
                for bucket_name in bucket_names:
                    aux_summary = aux_split["by_bucket"][bucket_key].get(bucket_name, summarize([]))
                    dup_summary = dup_split["by_bucket"][bucket_key].get(bucket_name, summarize([]))
                    split_payload["by_bucket"][bucket_key][bucket_name] = {
                        "aux": aux_summary,
                        "directdup": dup_summary,
                        "delta": metric_delta(aux_summary, dup_summary),
                    }
            deltas[pair_key][split] = split_payload
    return deltas


def fmt(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def render_branch_table(branch_payloads):
    lines = [
        "| branch | split | n | json | trigger_f1 | argument_f1 | event_f1 | latency |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch, payload in branch_payloads.items():
        for split in SPLITS:
            summary = payload["splits"][split]["overall"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{branch}`",
                        split,
                        str(summary["num_examples"]),
                        fmt(summary["json_valid_rate"]),
                        fmt(summary["trigger_f1"]),
                        fmt(summary["argument_f1"]),
                        fmt(summary["event_f1"]),
                        fmt(summary["avg_latency_sec"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def render_delta_table(pair_payloads):
    lines = [
        "| pair | split | arg_delta | event_delta | trigger_delta | latency_delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pair, split_payload in pair_payloads.items():
        for split in SPLITS:
            delta = split_payload[split]["overall"]["delta"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{pair}`",
                        split,
                        fmt(delta["argument_f1"]),
                        fmt(delta["event_f1"]),
                        fmt(delta["trigger_f1"]),
                        fmt(delta["avg_latency_sec"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def render_bucket_delta_tables(pair_payloads):
    sections = []
    for pair, split_payload in pair_payloads.items():
        for split in SPLITS:
            for bucket_key in [
                "event_type_rarity_bucket",
                "role_signature_rarity_bucket",
                "confusion_score_bucket",
                "selected_signature_bucket",
            ]:
                sections.append(f"### {pair} / {split} / {bucket_key}")
                sections.append("| bucket | n_aux | n_dup | arg_delta | event_delta | trigger_delta |")
                sections.append("|---|---:|---:|---:|---:|---:|")
                for bucket_name, payload in split_payload[split]["by_bucket"][bucket_key].items():
                    delta = payload["delta"]
                    sections.append(
                        "| "
                        + " | ".join(
                            [
                                bucket_name,
                                str(payload["aux"]["num_examples"]),
                                str(payload["directdup"]["num_examples"]),
                                fmt(delta["argument_f1"]),
                                fmt(delta["event_f1"]),
                                fmt(delta["trigger_f1"]),
                            ]
                        )
                        + " |"
                    )
                sections.append("")
    return "\n".join(sections)


def render_markdown(payload):
    lines = [
        "# 2026-05-04 Stage2 SAR Bucket Generalization Analysis",
        "",
        "## Scope",
        "",
        f"- Branches: {', '.join(f'`{branch}`' for branch in payload['branches'])}",
        f"- Pairs: {', '.join(f'`{pair}`' for pair in payload['pairs'])}",
        "- Metrics are row-averaged, matching `eval_adapter_generation.py` summaries.",
        "- `selected_signature_bucket` means the eval row contains a gold role signature observed in the selector-selected training rows.",
        "",
        "## Overall",
        "",
        render_branch_table(payload["branch_payloads"]),
        "",
        "## Auxiliary vs Direct Duplicate Delta",
        "",
        render_delta_table(payload["pairwise_deltas"]),
        "",
        "## Bucket Deltas",
        "",
        render_bucket_delta_tables(payload["pairwise_deltas"]),
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
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-04_stage2_sar_bucket_generalization_analysis.json")
    parser.add_argument("--output_md", default="reports/2026-05-04_stage2_sar_bucket_generalization_analysis.md")
    args = parser.parse_args()

    schema_by_type = load_schema_map(Path(args.schema_path))
    direct_train_rows = load_jsonl(Path(args.direct_train_jsonl))
    train_stats = build_confrare_stats(direct_train_rows)

    branch_payloads = {}
    for branch in args.branches:
        branch_payloads[branch] = build_records_for_branch(
            branch=branch,
            prefix=args.dataset_prefix,
            model_tag=args.run_prefix,
            dataset_dir=Path(args.dataset_dir),
            best_eval_root=Path(args.best_eval_root),
            direct_train_rows=direct_train_rows,
            schema_by_type=schema_by_type,
            train_stats=train_stats,
        )

    pairwise_deltas = pairwise_bucket_deltas(branch_payloads, args.pairs)
    payload = {
        "schema_path": args.schema_path,
        "direct_train_jsonl": args.direct_train_jsonl,
        "dataset_dir": args.dataset_dir,
        "best_eval_root": args.best_eval_root,
        "branches": args.branches,
        "pairs": args.pairs,
        "branch_payloads": branch_payloads,
        "pairwise_deltas": pairwise_deltas,
    }

    write_json(Path(args.output_json), payload)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
