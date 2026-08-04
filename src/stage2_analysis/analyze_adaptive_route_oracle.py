import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_data.build_formal_stage2_dataset import load_jsonl, write_json


SPLITS = ["test", "test_seen", "test_unseen"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def input_key(row):
    return hashlib.sha1(row.get("input", "").encode("utf-8")).hexdigest()


def load_predictions(path: Path):
    rows = load_jsonl(path)
    return {input_key(row): row for row in rows}


def metric(row, key):
    return float(row.get(key, 0.0) or 0.0)


def score(row):
    return metric(row, "argument_f1") + metric(row, "event_f1") + 0.25 * metric(row, "trigger_f1")


def gold_sets(row):
    gold = row.get("gold", {})
    events = gold.get("events", []) if isinstance(gold, dict) else []
    return bool(events)


def summarize_rows(rows):
    if not rows:
        return {
            "num_examples": 0,
            "trigger_f1": 0.0,
            "argument_f1": 0.0,
            "event_f1": 0.0,
            "reason_rate": 0.0,
        }
    return {
        "num_examples": len(rows),
        "trigger_f1": sum(row["trigger_f1"] for row in rows) / len(rows),
        "argument_f1": sum(row["argument_f1"] for row in rows) / len(rows),
        "event_f1": sum(row["event_f1"] for row in rows) / len(rows),
        "reason_rate": sum(1 for row in rows if row["route"] == "reason") / len(rows),
        "positive_count": sum(1 for row in rows if row["gold_positive"]),
    }


def oracle_rows(direct_rows, reason_rows, cap_rate=None):
    keys = sorted(set(direct_rows) & set(reason_rows))
    improvements = []
    base_rows = []
    for key in keys:
        direct = direct_rows[key]
        reason = reason_rows[key]
        gain = score(reason) - score(direct)
        base = {
            "key": key,
            "gain": gain,
            "direct_trigger_f1": metric(direct, "trigger_f1"),
            "direct_argument_f1": metric(direct, "argument_f1"),
            "direct_event_f1": metric(direct, "event_f1"),
            "reason_trigger_f1": metric(reason, "trigger_f1"),
            "reason_argument_f1": metric(reason, "argument_f1"),
            "reason_event_f1": metric(reason, "event_f1"),
            "gold_positive": gold_sets(direct),
        }
        base_rows.append(base)
        if gain > 1e-9 and base["gold_positive"]:
            improvements.append(base)
    improvements.sort(key=lambda item: (item["gain"], item["key"]), reverse=True)
    if cap_rate is None:
        reason_keys = {item["key"] for item in improvements}
    else:
        reason_keys = {item["key"] for item in improvements[: round(len(keys) * cap_rate)]}

    routed = []
    for item in base_rows:
        use_reason = item["key"] in reason_keys
        prefix = "reason" if use_reason else "direct"
        routed.append(
            {
                **item,
                "route": prefix,
                "trigger_f1": item[f"{prefix}_trigger_f1"],
                "argument_f1": item[f"{prefix}_argument_f1"],
                "event_f1": item[f"{prefix}_event_f1"],
            }
        )
    return routed


def split_paths(base_dir: Path, run_name: str):
    return {
        split: base_dir / f"{run_name}_{split}_argfirst" / "predictions.jsonl"
        for split in SPLITS
    }


def existing_split_paths(base_dir: Path, run_name: str):
    paths = split_paths(base_dir, run_name)
    missing = [path.as_posix() for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prediction files for {run_name}: {missing}")
    return paths


def analyze_pair(base_dir: Path, pair):
    direct_paths = existing_split_paths(base_dir, pair["direct_run"])
    reason_paths = existing_split_paths(base_dir, pair["reason_run"])
    result = {
        "pair_id": pair["id"],
        "direct_run": pair["direct_run"],
        "reason_run": pair["reason_run"],
        "splits": {},
    }
    for split in SPLITS:
        direct_rows = load_predictions(direct_paths[split])
        reason_rows = load_predictions(reason_paths[split])
        direct_only = oracle_rows(direct_rows, direct_rows, cap_rate=None)
        reason_only = oracle_rows(reason_rows, reason_rows, cap_rate=None)
        uncapped = oracle_rows(direct_rows, reason_rows, cap_rate=None)
        cap10 = oracle_rows(direct_rows, reason_rows, cap_rate=0.10)
        cap15 = oracle_rows(direct_rows, reason_rows, cap_rate=0.15)
        result["splits"][split] = {
            "direct": summarize_rows(direct_only),
            "reason": summarize_rows(reason_only),
            "oracle_uncapped": summarize_rows(uncapped),
            "oracle_cap10": summarize_rows(cap10),
            "oracle_cap15": summarize_rows(cap15),
            "oracle_uncapped_gain_vs_direct": {
                "trigger_f1": summarize_rows(uncapped)["trigger_f1"] - summarize_rows(direct_only)["trigger_f1"],
                "argument_f1": summarize_rows(uncapped)["argument_f1"] - summarize_rows(direct_only)["argument_f1"],
                "event_f1": summarize_rows(uncapped)["event_f1"] - summarize_rows(direct_only)["event_f1"],
            },
            "oracle_cap10_gain_vs_direct": {
                "trigger_f1": summarize_rows(cap10)["trigger_f1"] - summarize_rows(direct_only)["trigger_f1"],
                "argument_f1": summarize_rows(cap10)["argument_f1"] - summarize_rows(direct_only)["argument_f1"],
                "event_f1": summarize_rows(cap10)["event_f1"] - summarize_rows(direct_only)["event_f1"],
            },
            "oracle_cap15_gain_vs_direct": {
                "trigger_f1": summarize_rows(cap15)["trigger_f1"] - summarize_rows(direct_only)["trigger_f1"],
                "argument_f1": summarize_rows(cap15)["argument_f1"] - summarize_rows(direct_only)["argument_f1"],
                "event_f1": summarize_rows(cap15)["event_f1"] - summarize_rows(direct_only)["event_f1"],
            },
        }
    return result


def render_md(results, output_json_path):
    lines = [
        "# Adaptive Route Oracle Analysis",
        "",
        f"- artifact: `{output_json_path}`",
        "- oracle policy: choose reason only when the reason expert improves argument/event/trigger weighted per-sample score; ties route direct.",
        "- capped policies: choose only the top 10% or 15% positive-gain samples.",
        "",
        "## Summary",
        "",
        "| pair | split | direct arg | direct event | cap10 arg | cap10 event | cap10 reason | uncapped arg | uncapped event | uncapped reason |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for split in SPLITS:
            row = result["splits"][split]
            lines.append(
                "| {pair} | {split} | {darg:.4f} | {devt:.4f} | {carg:.4f} | {cevt:.4f} | {crate:.3f} | {uarg:.4f} | {uevt:.4f} | {urate:.3f} |".format(
                    pair=result["pair_id"],
                    split=split,
                    darg=row["direct"]["argument_f1"],
                    devt=row["direct"]["event_f1"],
                    carg=row["oracle_cap10"]["argument_f1"],
                    cevt=row["oracle_cap10"]["event_f1"],
                    crate=row["oracle_cap10"]["reason_rate"],
                    uarg=row["oracle_uncapped"]["argument_f1"],
                    uevt=row["oracle_uncapped"]["event_f1"],
                    urate=row["oracle_uncapped"]["reason_rate"],
                )
            )
    lines.extend(["", "## Gate Check", ""])
    passed = []
    for result in results:
        unseen = result["splits"]["test_unseen"]["oracle_cap10_gain_vs_direct"]
        if unseen["argument_f1"] >= 0.02 or unseen["event_f1"] >= 0.02:
            passed.append((result["pair_id"], unseen))
    if passed:
        lines.append("Wave 1 gate passed for at least one capped oracle pair:")
        for pair_id, gain in passed:
            lines.append(
                f"- `{pair_id}` test_unseen cap10 gain: argument `{gain['argument_f1']:.4f}`, event `{gain['event_f1']:.4f}`"
            )
    else:
        lines.append("Wave 1 gate did not pass under cap10 oracle.")
    lines.append("")
    return "\n".join(lines)


def default_pairs():
    return [
        {
            "id": "qwen1p7_pos_confrare10_typerole",
            "direct_run": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_direct_full",
            "reason_run": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_sar_confrare10_typerolelite_full",
        },
        {
            "id": "qwen1p7_pos_confrole10_typerole",
            "direct_run": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_direct_full",
            "reason_run": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_sar_confrole10_typerolelite_full",
        },
        {
            "id": "qwen1p7_pos_confrole10_typeonly",
            "direct_run": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_direct_full",
            "reason_run": "richere_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_sar_confrole10_typeonlylite_full",
        },
        {
            "id": "qwen1p7_full_confrare10_typerole",
            "direct_run": "richere_full_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_direct_full",
            "reason_run": "richere_full_split1_qwen3_1_7b_instruct_oracle_mixed_noise_top10_shuffle_sar_confrare10_typerolelite_full",
        },
        {
            "id": "qwen4_pos_confrare10_typerole",
            "direct_run": "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full",
            "reason_run": "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_sar_confrare10_typerolelite_full",
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_eval_dir", default="outputs/stage2_full_sft_runs_stepmatch_best_eval_user")
    parser.add_argument("--legacy_direct_eval_dir", default="outputs/stage2_full_sft_runs_epoch20_best_eval_user")
    parser.add_argument("--output_md", default="reports/2026-05-08_stage2_adaptive_route_oracle_analysis.md")
    parser.add_argument("--output_json", default="reports/artifacts/2026-05-08_stage2_adaptive_route_oracle_analysis.json")
    args = parser.parse_args()

    base_dir = Path(args.base_eval_dir)
    legacy_dir = Path(args.legacy_direct_eval_dir)
    results = []
    for pair in default_pairs():
        try:
            results.append(analyze_pair(base_dir, pair))
        except FileNotFoundError:
            if pair["direct_run"].endswith("_direct_full") and pair["direct_run"].startswith("richere_split1_qwen3_1_7b"):
                patched = dict(pair)
                patched["_direct_base_dir"] = legacy_dir.as_posix()
                result = {
                    **analyze_pair_with_dirs(legacy_dir, base_dir, patched),
                }
                results.append(result)
            else:
                raise

    artifact = {
        "base_eval_dir": base_dir.as_posix(),
        "legacy_direct_eval_dir": legacy_dir.as_posix(),
        "pairs": results,
    }
    output_json = Path(args.output_json)
    write_json(output_json, artifact)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_md(results, output_json.as_posix()), encoding="utf-8")
    print(json.dumps({"output_md": output_md.as_posix(), "output_json": output_json.as_posix()}, indent=2))


def analyze_pair_with_dirs(direct_dir: Path, reason_dir: Path, pair):
    direct_paths = existing_split_paths(direct_dir, pair["direct_run"])
    reason_paths = existing_split_paths(reason_dir, pair["reason_run"])
    result = {
        "pair_id": pair["id"],
        "direct_run": pair["direct_run"],
        "reason_run": pair["reason_run"],
        "direct_eval_dir": direct_dir.as_posix(),
        "reason_eval_dir": reason_dir.as_posix(),
        "splits": {},
    }
    for split in SPLITS:
        direct_rows = load_predictions(direct_paths[split])
        reason_rows = load_predictions(reason_paths[split])
        direct_only = oracle_rows(direct_rows, direct_rows, cap_rate=None)
        reason_only = oracle_rows(reason_rows, reason_rows, cap_rate=None)
        uncapped = oracle_rows(direct_rows, reason_rows, cap_rate=None)
        cap10 = oracle_rows(direct_rows, reason_rows, cap_rate=0.10)
        cap15 = oracle_rows(direct_rows, reason_rows, cap_rate=0.15)
        direct_summary = summarize_rows(direct_only)
        uncapped_summary = summarize_rows(uncapped)
        cap10_summary = summarize_rows(cap10)
        cap15_summary = summarize_rows(cap15)
        result["splits"][split] = {
            "direct": direct_summary,
            "reason": summarize_rows(reason_only),
            "oracle_uncapped": uncapped_summary,
            "oracle_cap10": cap10_summary,
            "oracle_cap15": cap15_summary,
            "oracle_uncapped_gain_vs_direct": {
                "trigger_f1": uncapped_summary["trigger_f1"] - direct_summary["trigger_f1"],
                "argument_f1": uncapped_summary["argument_f1"] - direct_summary["argument_f1"],
                "event_f1": uncapped_summary["event_f1"] - direct_summary["event_f1"],
            },
            "oracle_cap10_gain_vs_direct": {
                "trigger_f1": cap10_summary["trigger_f1"] - direct_summary["trigger_f1"],
                "argument_f1": cap10_summary["argument_f1"] - direct_summary["argument_f1"],
                "event_f1": cap10_summary["event_f1"] - direct_summary["event_f1"],
            },
            "oracle_cap15_gain_vs_direct": {
                "trigger_f1": cap15_summary["trigger_f1"] - direct_summary["trigger_f1"],
                "argument_f1": cap15_summary["argument_f1"] - direct_summary["argument_f1"],
                "event_f1": cap15_summary["event_f1"] - direct_summary["event_f1"],
            },
        }
    return result


if __name__ == "__main__":
    main()
