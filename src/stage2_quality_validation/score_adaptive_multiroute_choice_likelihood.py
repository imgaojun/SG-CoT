import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_quality_validation.eval_adapter_generation import build_prompt, load_jsonl  # noqa: E402
from src.stage2_quality_validation.score_adaptive_route_choice_likelihood import (  # noqa: E402
    load_model,
    mean_nll_for_continuation,
)


def meta_wnd_id(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("wnd_id") or row.get("id")


def gold_route(row):
    meta = row.get("meta") or {}
    label = meta.get("adaptive_route_label")
    if label:
        return label
    out = row.get("output", "")
    start = out.find("<ROUTE>")
    end = out.find("</ROUTE>")
    if start >= 0 and end > start:
        return out[start + len("<ROUTE>") : end].strip()
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--routes", required=True, help="Comma-separated route labels.")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    routes = [item.strip() for item in args.routes.split(",") if item.strip()]
    targets = {route: f"<ROUTE>{route}</ROUTE>" for route in routes}
    rows = load_jsonl(Path(args.eval_jsonl))
    tokenizer, model = load_model(args.base_model, args.adapter_path)

    out_rows = []
    for idx, row in enumerate(rows):
        prompt = build_prompt(tokenizer, row["instruction"], row["input"])
        nll_by_route = {}
        token_counts = {}
        for route, continuation in targets.items():
            score = mean_nll_for_continuation(tokenizer, model, prompt, continuation, args.max_length)
            nll_by_route[route] = score["mean_nll"]
            token_counts[route] = score["num_tokens"]
        pred_route = min(routes, key=lambda route: nll_by_route[route])
        direct_nll = nll_by_route.get("direct")
        best_non_direct = min(
            (route for route in routes if route != "direct"),
            key=lambda route: nll_by_route[route],
            default=None,
        )
        best_non_direct_advantage = None
        if best_non_direct is not None and direct_nll is not None:
            best_non_direct_advantage = direct_nll - nll_by_route[best_non_direct]
        gold = gold_route(row)
        out_rows.append(
            {
                "index": idx,
                "wnd_id": meta_wnd_id(row),
                "gold_route": gold,
                "pred_route_argmin_nll": pred_route,
                "route_correct_argmin_nll": pred_route == gold,
                "nll_by_route": nll_by_route,
                "route_token_counts": token_counts,
                "best_non_direct_route": best_non_direct,
                "best_non_direct_advantage_vs_direct": best_non_direct_advantage,
                "meta": row.get("meta") or {},
            }
        )

    correct = sum(1 for row in out_rows if row["route_correct_argmin_nll"])
    gold_counts = {route: sum(1 for row in out_rows if row["gold_route"] == route) for route in routes}
    pred_counts = {route: sum(1 for row in out_rows if row["pred_route_argmin_nll"] == route) for route in routes}
    confusion = {
        gold: {
            pred: sum(
                1
                for row in out_rows
                if row["gold_route"] == gold and row["pred_route_argmin_nll"] == pred
            )
            for pred in routes
        }
        for gold in routes
    }
    summary = {
        "num_examples": len(out_rows),
        "routes": routes,
        "argmin_route_accuracy": correct / len(out_rows) if out_rows else 0.0,
        "gold_counts": gold_counts,
        "pred_counts": pred_counts,
        "confusion": confusion,
    }

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
