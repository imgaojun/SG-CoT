#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.diagnose_sampled_k2_formal_unseen_false_positives_20260519 import metric_dict  # noqa: E402
from scripts.summarize_sampled_confident_router_dev_20260518 import pct, signed, write_json, write_text  # noqa: E402
from scripts.summarize_sampled_k2_structural_proxy_locked_validation_20260519 import (  # noqa: E402
    DEFAULT_FRESH_NLL_ROOT,
    DEFAULT_NEW_NLL_ROOT,
    DEFAULT_OLD_NLL_ROOT,
    DEFAULT_SAMPLE_ROOT,
    avg_metrics,
    build_cases,
)


DEFAULT_REPORT_MD = REPO / "reports/2026-05-19_stage2_direct_anchored_reason_smoke.md"
DEFAULT_REPORT_JSON = REPO / "reports/artifacts/2026-05-19_stage2_direct_anchored_reason_smoke.json"
POLICY_ID = "fm0p25_mr0p75_n1_aj0p40_ed0p00_am0p00"
METRICS = ["argument_f1", "event_f1", "trigger_f1", "score", "valid_json"]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def key_for(row):
    meta = row.get("meta") or {}
    return row.get("sample_key") or meta.get("wnd_id") or meta.get("doc_id")


def load_anchored_by_key(predictions_path: Path, eval_jsonl: Path):
    predictions = load_jsonl(predictions_path)
    eval_rows = load_jsonl(eval_jsonl)
    if len(predictions) != len(eval_rows):
        raise ValueError((len(predictions), len(eval_rows), predictions_path, eval_jsonl))
    return {
        key_for(eval_row): metric_dict(pred_row)
        for eval_row, pred_row in zip(eval_rows, predictions)
    }


def mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def relaxed_rule(case):
    avg_margin = mean([case["fresh_margin"], case["old17_18_margin"], case["new19_20_margin"]])
    return (
        case["fresh_margin"] >= 0.25
        and case["margin_range"] <= 0.75
        and case["num_margins_ge_0p25"] >= 1
        and case["sample_arg_text_jaccard_mean"] >= 0.40
        and case["sample_event_count_delta_mean"] <= 0.0
        and avg_margin >= 0.0
    )


def summarize_case_gains(cases):
    buckets = Counter()
    for case in cases:
        gain = case["anchored"]["score"] - case["direct"]["score"]
        if gain > 0:
            buckets["helpful"] += 1
        elif gain < 0:
            buckets["harmful"] += 1
        else:
            buckets["neutral"] += 1
    return {
        "count": len(cases),
        "helpful": buckets["helpful"],
        "harmful": buckets["harmful"],
        "neutral": buckets["neutral"],
        "harm_rate": buckets["harmful"] / len(cases) if cases else 0.0,
        "mean_gain": mean(case["anchored"]["score"] - case["direct"]["score"] for case in cases),
        "mean_original_reason_gain": mean(case["original_reason"]["score"] - case["direct"]["score"] for case in cases),
    }


def avg(rows):
    return avg_metrics(rows)


def delta(left, right):
    return {metric: left[metric] - right[metric] for metric in METRICS}


def summarize_policy(cases_by_split, anchored_by_key):
    rows = []
    selected_cases = []
    for split in ["test_seen", "test_unseen"]:
        for case in cases_by_split[split]:
            direct = case["single_gen_execution_direct"]
            original_reason = case["single_gen_execution_reason"]
            if relaxed_rule(case):
                anchored = anchored_by_key.get(case["key"])
                if anchored is None:
                    raise KeyError(f"missing anchored prediction for {split}/{case['key']}")
                routed = anchored
                selected_cases.append(
                    {
                        "split": split,
                        "key": case["key"],
                        "direct": direct,
                        "original_reason": original_reason,
                        "anchored": anchored,
                    }
                )
            else:
                routed = direct
            rows.append({"split": split, "direct": direct, "routed": routed})

    out = []
    for split in ["test", "test_seen", "test_unseen"]:
        split_rows = rows if split == "test" else [row for row in rows if row["split"] == split]
        direct_avg = avg([row["direct"] for row in split_rows])
        routed_avg = avg([row["routed"] for row in split_rows])
        selected = selected_cases if split == "test" else [case for case in selected_cases if case["split"] == split]
        gains = summarize_case_gains(selected)
        out.append(
            {
                "split": split,
                "num_examples": len(split_rows),
                "pred_reason_count": len(selected),
                "pred_reason_rate": len(selected) / len(split_rows) if split_rows else 0.0,
                "direct": direct_avg,
                "routed": routed_avg,
                "routed_minus_direct": delta(routed_avg, direct_avg),
                "selected_gain": gains,
            }
        )
    return out, selected_cases


def render_summary_table(rows):
    lines = [
        "| split | reason rate | routed A/E/T/Score | delta A/E/T/Score | selected helpful/harmful/neutral | selected gain | original reason gain | harm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        selected = row["selected_gain"]
        routed = row["routed"]
        delta_row = row["routed_minus_direct"]
        lines.append(
            f"| `{row['split']}` | {pct(row['pred_reason_rate'])} | "
            f"{routed['argument_f1']:.4f}/{routed['event_f1']:.4f}/{routed['trigger_f1']:.4f}/{routed['score']:.4f} | "
            f"{signed(delta_row['argument_f1'])}/{signed(delta_row['event_f1'])}/{signed(delta_row['trigger_f1'])}/{signed(delta_row['score'])} | "
            f"{selected['helpful']}/{selected['harmful']}/{selected['neutral']} | "
            f"{signed(selected['mean_gain'])} | {signed(selected['mean_original_reason_gain'])} | {pct(selected['harm_rate'])} |"
        )
    return "\n".join(lines)


def render_report(payload):
    rows = payload["summary_rows"]
    test = next(row for row in rows if row["split"] == "test")
    lines = [
        "# Direct-Anchored Reason Smoke Summary",
        "",
        f"Policy: `{POLICY_ID}` with `{payload['prompt_variant']}`.",
        "",
        "## Summary",
        "",
        render_summary_table(rows),
        "",
        "## Reading",
        "",
        f"- Anchored routed score delta test/seen/unseen: `{rows[0]['routed_minus_direct']['score']:+.4f}/{rows[1]['routed_minus_direct']['score']:+.4f}/{rows[2]['routed_minus_direct']['score']:+.4f}`.",
        f"- Selected-case anchored mean gain on test: `{test['selected_gain']['mean_gain']:+.4f}` versus original reason mean gain `{test['selected_gain']['mean_original_reason_gain']:+.4f}`.",
        f"- Selected-case anchored harm on test: `{test['selected_gain']['harm_rate']:.1%}`.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['report_json']}`",
        f"- anchored predictions: `{payload['anchored_predictions']}`",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    anchored_predictions = args.anchored_root / "predictions.jsonl"
    if not anchored_predictions.exists():
        raise FileNotFoundError(anchored_predictions)
    anchored_by_key = load_anchored_by_key(anchored_predictions, args.eval_jsonl)
    build_args = argparse.Namespace(
        seeds=args.seeds,
        sample_root=args.sample_root,
        fresh_nll_root=args.fresh_nll_root,
        old_nll_root=args.old_nll_root,
        new_nll_root=args.new_nll_root,
        checkpoint=args.checkpoint,
    )
    cases_by_split = build_cases(build_args)
    summary_rows, selected_cases = summarize_policy(cases_by_split, anchored_by_key)
    payload = {
        "policy": POLICY_ID,
        "prompt_variant": args.prompt_variant,
        "anchored_root": args.anchored_root.as_posix(),
        "anchored_predictions": anchored_predictions.as_posix(),
        "eval_jsonl": args.eval_jsonl.as_posix(),
        "summary_rows": summary_rows,
        "selected_case_count": len(selected_cases),
        "report_md": args.report_md.as_posix(),
        "report_json": args.report_json.as_posix(),
    }
    write_json(args.report_json, payload)
    write_json(args.anchored_root / "summary_policy_eval.json", payload)
    write_text(args.report_md, render_report(payload))
    print(json.dumps({"report_md": args.report_md.as_posix(), "report_json": args.report_json.as_posix()}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-variant", default="anchor_conservative", choices=["anchor_conservative", "anchor_revise", "anchor_verify"])
    parser.add_argument("--anchored-root", type=Path, default=None)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 24])
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--fresh-nll-root", type=Path, default=DEFAULT_FRESH_NLL_ROOT)
    parser.add_argument("--old-nll-root", type=Path, default=DEFAULT_OLD_NLL_ROOT)
    parser.add_argument("--new-nll-root", type=Path, default=DEFAULT_NEW_NLL_ROOT)
    parser.add_argument("--checkpoint", default="checkpoint-50")
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    args = parser.parse_args()
    if args.anchored_root is None:
        args.anchored_root = REPO / f"outputs/stage2_adaptive_direct_anchored_reason_generation_20260519/{POLICY_ID}_{args.prompt_variant}_tagged"
    if args.eval_jsonl is None:
        args.eval_jsonl = REPO / f"data/stage2_adaptive_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_directanchored_reason_20260519_{POLICY_ID}_{args.prompt_variant}_test_pos.jsonl"
    if args.report_md == DEFAULT_REPORT_MD and args.prompt_variant != "anchor_conservative":
        args.report_md = REPO / f"reports/2026-05-19_stage2_direct_anchored_reason_smoke_{args.prompt_variant}.md"
    if args.report_json == DEFAULT_REPORT_JSON and args.prompt_variant != "anchor_conservative":
        args.report_json = REPO / f"reports/artifacts/2026-05-19_stage2_direct_anchored_reason_smoke_{args.prompt_variant}.json"
    return args


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
