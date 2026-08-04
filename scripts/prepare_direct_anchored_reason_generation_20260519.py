#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.summarize_sampled_confident_router_dev_20260518 import pct, signed, write_json, write_text  # noqa: E402
from scripts.summarize_sampled_k2_structural_proxy_locked_validation_20260519 import (  # noqa: E402
    DEFAULT_FRESH_NLL_ROOT,
    DEFAULT_NEW_NLL_ROOT,
    DEFAULT_OLD_NLL_ROOT,
    DEFAULT_SAMPLE_ROOT,
    build_cases,
)


ADAPTIVE_DATA_DIR = REPO / "data/stage2_adaptive_datasets"
FORMAL_DATA_DIR = REPO / "data/stage2_formal_datasets"
OUTPUT_ROOT = REPO / "outputs/stage2_adaptive_direct_anchored_reason_prep_20260519"
REPORT_MD = REPO / "reports/2026-05-19_stage2_direct_anchored_reason_generation_prep.md"
REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_direct_anchored_reason_generation_prep.json"
RELAXED_REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_sampled_k2_structproxy_relaxed_selector_sweep.json"
DIRECT_PRED_ROOT = REPO / "outputs/stage2_adaptive_route_formal_execution_20260518/sampledk2_ckpt50_margin025/forced_direct"
SOURCE_DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle"
OUT_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_directanchored_reason_20260519"
SPLITS = ["test_seen", "test_unseen"]


PROMPT_VARIANTS = {
    "anchor_verify": (
        "You are doing event extraction. Use only the provided candidate event types, schema cards, text, tokens, and the supplied direct-route extraction. "
        "Verify the direct extraction against the text. Preserve every direct event and argument that is supported by the text. "
        "Only add an event or argument if it is explicitly supported by the text and uses token offsets from the token list. "
        "Return strict JSON only with top-level key `events`. If no valid event is expressed by the candidate set, output {\"events\": []}."
    ),
    "anchor_revise": (
        "You are doing event extraction. Use only the provided candidate event types, schema cards, text, tokens, and the supplied direct-route extraction. "
        "Revise the direct extraction conservatively: keep supported direct items, remove unsupported items, and add only clearly supported missing items. "
        "Do not invent new event types, triggers, argument texts, or offsets beyond the text. "
        "Return strict JSON only with top-level key `events`. If no valid event is expressed by the candidate set, output {\"events\": []}."
    ),
    "anchor_conservative": (
        "You are doing event extraction. Use only the provided candidate event types, schema cards, text, tokens, and the supplied direct-route extraction. "
        "Prefer the direct extraction. Change it only when the text gives unambiguous evidence for a correction. "
        "When uncertain, return the direct extraction unchanged. "
        "Return strict JSON only with top-level key `events`. If no valid event is expressed by the candidate set, output {\"events\": []}."
    ),
}


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def key_for(row):
    meta = row.get("meta") or {}
    return row.get("sample_key") or meta.get("wnd_id") or meta.get("doc_id")


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def load_source_rows(split):
    rows = load_jsonl(FORMAL_DATA_DIR / f"{SOURCE_DATA_PREFIX}_{split}_pos.jsonl")
    return {key_for(row): row for row in rows}


def load_direct_predictions(split):
    rows = load_jsonl(DIRECT_PRED_ROOT / split / "predictions.jsonl")
    return {key_for(row): row for row in rows}


def route_for_spec(case, spec):
    avg_margin = mean([case["fresh_margin"], case["old17_18_margin"], case["new19_20_margin"]])
    return (
        case["fresh_margin"] >= spec["fresh_margin_min"]
        and case["margin_range"] <= spec["margin_range_max"]
        and case["num_margins_ge_0p25"] >= spec["num_margins_ge_0p25_min"]
        and case["sample_arg_text_jaccard_mean"] >= spec["arg_text_jaccard_min"]
        and case["sample_event_count_delta_mean"] <= spec["event_count_delta_max"]
        and avg_margin >= spec["avg_margin_min"]
    )


def locked_rule(case):
    return (
        case["fresh_margin"] >= 0.25
        and case["margin_range"] <= 0.50
        and case["num_margins_ge_0p25"] >= 2
        and case["sample_arg_text_jaccard_mean"] >= 0.40
        and case["sample_event_count_delta_mean"] <= 0.0
    )


def select_policy_specs(args):
    report = json.loads(args.relaxed_report_json.read_text(encoding="utf-8"))
    specs = [
        {
            "policy_id": "locked_structural_proxy",
            "source_rule_id": "locked_structural_proxy",
            "kind": "locked_baseline",
            "spec": None,
        }
    ]
    for row in report["consolidated"]:
        if not (row["target_coverage"] and row["robust_positive"] and row["beats_locked_candidate"]):
            continue
        if row["rule_id"] in {spec["source_rule_id"] for spec in specs}:
            continue
        specs.append(
            {
                "policy_id": row["rule_id"],
                "source_rule_id": row["rule_id"],
                "kind": "relaxed_selector",
                "spec": row["spec"],
                "sweep_metrics": {
                    key: row[key]
                    for key in [
                        "test_score_delta",
                        "seen_score_delta",
                        "unseen_score_delta",
                        "test_reason_rate",
                        "seen_reason_rate",
                        "unseen_reason_rate",
                        "test_harm_rate",
                        "seen_harm_rate",
                        "unseen_harm_rate",
                    ]
                },
            }
        )
        if len(specs) >= args.max_policies:
            break
    return specs


def selected_cases(cases_by_split, policy):
    selected = []
    for split in SPLITS:
        for case in cases_by_split[split]:
            is_reason = locked_rule(case) if policy["spec"] is None else route_for_spec(case, policy["spec"])
            if not is_reason:
                continue
            direct = case["single_gen_execution_direct"]
            reason = case["single_gen_execution_reason"]
            selected.append(
                {
                    "split": split,
                    "key": case["key"],
                    "case_id": case["case_id"],
                    "single_gen_score_gain": reason["score"] - direct["score"],
                    "k2_expected_score_gain": case["k2_expected_reason"]["score"] - case["k2_expected_direct"]["score"],
                    "fresh_margin": case["fresh_margin"],
                    "margin_range": case["margin_range"],
                    "num_margins_ge_0p25": case["num_margins_ge_0p25"],
                    "sample_arg_text_jaccard_mean": case["sample_arg_text_jaccard_mean"],
                    "sample_event_count_delta_mean": case["sample_event_count_delta_mean"],
                }
            )
    return selected


def build_input(source_input, direct_json):
    return (
        f"{source_input}\n\n"
        "Direct-route extraction to verify or revise:\n"
        f"{json.dumps(direct_json, ensure_ascii=False)}\n\n"
        "Return JSON only."
    )


def make_eval_rows(policy, prompt_variant, selected, source_by_split, direct_by_split):
    rows = []
    for item in selected:
        source_row = source_by_split[item["split"]][item["key"]]
        direct_pred = direct_by_split[item["split"]][item["key"]]
        direct_json = direct_pred.get("final_predicted") or direct_pred.get("predicted") or {"events": []}
        meta = dict(source_row.get("meta", {}))
        meta.update(
            {
                "adaptive_source": "direct_anchored_reason_generation_prep",
                "adaptive_dataset_role": item["split"],
                "adaptive_route_mode": "forced_direct_anchored_reason",
                "adaptive_route_label": "reason",
                "direct_anchor_policy": policy["policy_id"],
                "direct_anchor_prompt_variant": prompt_variant,
                "direct_anchor_single_gen_score_gain_original_reason": item["single_gen_score_gain"],
                "direct_anchor_k2_expected_score_gain_original_reason": item["k2_expected_score_gain"],
                "direct_anchor_fresh_margin": item["fresh_margin"],
                "direct_anchor_margin_range": item["margin_range"],
                "direct_anchor_num_margins_ge_0p25": item["num_margins_ge_0p25"],
                "direct_anchor_arg_text_jaccard": item["sample_arg_text_jaccard_mean"],
                "direct_anchor_event_count_delta": item["sample_event_count_delta_mean"],
            }
        )
        rows.append(
            {
                "instruction": (
                    f"{PROMPT_VARIANTS[prompt_variant]} "
                    "First output `<ROUTE>reason</ROUTE>`, then output `<FINAL>{...}</FINAL>`. "
                    "Do not add text outside the requested tags."
                ),
                "input": build_input(source_row["input"], direct_json),
                "output": f"<ROUTE>reason</ROUTE>\n<FINAL>{source_row['output']}</FINAL>",
                "gold_output": source_row["output"],
                "response_prefix": "<ROUTE>reason</ROUTE>\n<FINAL>",
                "meta": meta,
            }
        )
    return rows


def summarize_policy(policy, selected):
    buckets = Counter()
    by_split = defaultdict(list)
    for row in selected:
        if row["single_gen_score_gain"] > 0:
            buckets["helpful"] += 1
        elif row["single_gen_score_gain"] < 0:
            buckets["harmful"] += 1
        else:
            buckets["neutral"] += 1
        by_split[row["split"]].append(row)
    return {
        "policy_id": policy["policy_id"],
        "kind": policy["kind"],
        "source_rule_id": policy["source_rule_id"],
        "selected_count": len(selected),
        "helpful": buckets["helpful"],
        "harmful": buckets["harmful"],
        "neutral": buckets["neutral"],
        "harm_rate": buckets["harmful"] / len(selected) if selected else 0.0,
        "mean_single_gen_gain": mean(row["single_gen_score_gain"] for row in selected),
        "mean_k2_expected_gain": mean(row["k2_expected_score_gain"] for row in selected),
        "by_split": {
            split: {
                "count": len(rows),
                "mean_single_gen_gain": mean(row["single_gen_score_gain"] for row in rows),
                "harm_rate": mean(1.0 if row["single_gen_score_gain"] < 0 else 0.0 for row in rows),
            }
            for split, rows in by_split.items()
        },
    }


def render_table(rows):
    lines = [
        "| policy | selected | helpful/harmful/neutral | harm | single gain | K2 gain | seen count/gain/harm | unseen count/gain/harm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        seen = row["by_split"].get("test_seen", {"count": 0, "mean_single_gen_gain": 0.0, "harm_rate": 0.0})
        unseen = row["by_split"].get("test_unseen", {"count": 0, "mean_single_gen_gain": 0.0, "harm_rate": 0.0})
        lines.append(
            f"| `{row['policy_id']}` | {row['selected_count']} | {row['helpful']}/{row['harmful']}/{row['neutral']} | "
            f"{pct(row['harm_rate'])} | {signed(row['mean_single_gen_gain'])} | {signed(row['mean_k2_expected_gain'])} | "
            f"{seen['count']}/{signed(seen['mean_single_gen_gain'])}/{pct(seen['harm_rate'])} | "
            f"{unseen['count']}/{signed(unseen['mean_single_gen_gain'])}/{pct(unseen['harm_rate'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    lines = [
        "# Direct-Anchored Reason Generation Prep",
        "",
        "This prepares direct-anchored reason-generation JSONL files for the locked structural proxy and top relaxed selector policies.",
        "No model generation is launched in this step.",
        "",
        "## Selected Policies",
        "",
        render_table(payload["policy_summaries"]),
        "",
        "## Prompt Variants",
        "",
    ]
    for name in PROMPT_VARIANTS:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
        ]
    )
    for row in payload["datasets"]:
        lines.append(f"- `{row['dataset_id']}`: `{row['path']}` (`{row['num_rows']}` rows)")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Use the relaxed policy files for the first anchored-generation smoke run because they include the added coverage that raises harm.",
            "- Compare anchored outputs against direct and original forced-reason outputs in the later combination evaluation.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['report_json']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    policies = select_policy_specs(args)
    build_args = argparse.Namespace(
        seeds=args.seeds,
        sample_root=args.sample_root,
        fresh_nll_root=args.fresh_nll_root,
        old_nll_root=args.old_nll_root,
        new_nll_root=args.new_nll_root,
        checkpoint=args.checkpoint,
    )
    cases_by_split = build_cases(build_args)
    source_by_split = {split: load_source_rows(split) for split in SPLITS}
    direct_by_split = {split: load_direct_predictions(split) for split in SPLITS}

    args.output_root.mkdir(parents=True, exist_ok=True)
    policy_summaries = []
    datasets = []
    for policy in policies:
        selected = selected_cases(cases_by_split, policy)
        policy_summaries.append(summarize_policy(policy, selected))
        selected_path = args.output_root / f"{policy['policy_id']}_selected_cases.jsonl"
        write_jsonl(selected_path, selected)
        for prompt_variant in PROMPT_VARIANTS:
            rows = make_eval_rows(policy, prompt_variant, selected, source_by_split, direct_by_split)
            dataset_id = f"{OUT_PREFIX}_{policy['policy_id']}_{prompt_variant}_test_pos"
            path = ADAPTIVE_DATA_DIR / f"{dataset_id}.jsonl"
            write_jsonl(path, rows)
            meta = {
                "dataset_id": dataset_id,
                "policy": policy,
                "prompt_variant": prompt_variant,
                "num_rows": len(rows),
                "selected_cases": selected_path.as_posix(),
            }
            write_json(path.with_suffix(".meta.json"), meta)
            datasets.append({"dataset_id": dataset_id, "path": path.as_posix(), "num_rows": len(rows), "meta": path.with_suffix(".meta.json").as_posix()})

    payload = {
        "checkpoint": args.checkpoint,
        "policies": policies,
        "policy_summaries": policy_summaries,
        "prompt_variants": PROMPT_VARIANTS,
        "datasets": datasets,
        "output_root": args.output_root.as_posix(),
        "report_md": args.report_md.as_posix(),
        "report_json": args.report_json.as_posix(),
    }
    write_json(args.report_json, payload)
    write_json(args.output_root / "summary.json", payload)
    write_text(args.report_md, render_report(payload))
    print(json.dumps({"report_md": args.report_md.as_posix(), "report_json": args.report_json.as_posix()}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relaxed-report-json", type=Path, default=RELAXED_REPORT_JSON)
    parser.add_argument("--max-policies", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 24])
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--fresh-nll-root", type=Path, default=DEFAULT_FRESH_NLL_ROOT)
    parser.add_argument("--old-nll-root", type=Path, default=DEFAULT_OLD_NLL_ROOT)
    parser.add_argument("--new-nll-root", type=Path, default=DEFAULT_NEW_NLL_ROOT)
    parser.add_argument("--checkpoint", default="checkpoint-50")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
