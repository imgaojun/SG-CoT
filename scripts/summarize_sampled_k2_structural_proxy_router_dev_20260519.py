#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.summarize_sampled_confident_router_dev_20260518 as base  # noqa: E402


BRANCH = "sampled_k2_structproxy_ckpt258_evidcompact_balhard_routecls_noauxwarm_lr2e6_save25"
REPORT_STEM = "2026-05-19_stage2_sampled_k2_structural_proxy_supervised_router_dev_probe"
DESCRIPTION = (
    "This dev-only report evaluates a route-only classifier trained from structural "
    "gold-free proxy supervision with compact K=2 repeated-output evidence. The dev "
    "set uses four K2 seed-pair evidence variants per confident dev example."
)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--report_stem", default=REPORT_STEM)
    parser.add_argument("--description", default=DESCRIPTION)
    parser.add_argument("--score_root", default=base.DEFAULT_SCORE_ROOT)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    label_path = REPO / base.LABEL_PATH
    route_root = REPO / (
        "outputs/stage2_adaptive_runs_user_devpick_route/"
        f"{base.RUN_PREFIX}_{args.branch}_full_route_dev_seen_seedpairs_max16"
    )
    nll_root = REPO / args.score_root / args.branch
    output_json = Path(args.output_json) if args.output_json else REPO / f"reports/artifacts/{args.report_stem}.json"
    output_md = Path(args.output_md) if args.output_md else REPO / f"reports/{args.report_stem}.md"
    label_map = base.load_label_map(label_path)

    execution_rows = []
    route_rows = []
    for item in base.route_generation_rows(route_root):
        pred_routes = base.load_generated_pred_routes(item["predictions"])
        result = base.summarize_routes(f"{item['checkpoint']}_gen", pred_routes, label_map)
        route_rows.append({"checkpoint": item["checkpoint"], **item["summary"], "sampled_summary": result})
        execution_rows.append(result)

    nll_summary = []
    for item in base.nll_rows(nll_root):
        for budget in base.BUDGETS:
            pred_routes, label = base.load_nll_pred_routes(item["scores"], budget)
            result = base.summarize_routes(f"{item['checkpoint']}_nll_{label}", pred_routes, label_map)
            execution_rows.append(result)
        nll_summary.append({"checkpoint": item["checkpoint"], **item["summary"]})

    if not route_rows:
        raise FileNotFoundError(f"no route generation outputs under {route_root}")
    if not nll_summary:
        raise FileNotFoundError(f"no route NLL outputs under {nll_root}")

    payload = {
        "branch": args.branch,
        "description": args.description,
        "label_path": label_path.as_posix(),
        "route_generation_root": route_root.as_posix(),
        "route_nll_root": nll_root.as_posix(),
        "route_generation_summaries": route_rows,
        "route_nll_summaries": nll_summary,
        "execution_results": execution_rows,
        "best": {
            "generated_route_f1": base.best(
                [row for row in execution_rows if row["name"].endswith("_gen")],
                lambda row: row["route_vs_confident_label"]["f1"],
            ),
            "nll_route_f1": base.best(
                [row for row in execution_rows if "_nll_" in row["name"]],
                lambda row: row["route_vs_confident_label"]["f1"],
            ),
            "sampled_expected_score_delta": base.best(
                execution_rows,
                lambda row: (
                    row["sampled_expected_routed_minus_direct"]["score"],
                    row["sampled_expected_routed_minus_direct"]["event_f1"],
                    row["sampled_expected_routed_minus_direct"]["argument_f1"],
                    row["sampled_expected_routed_minus_direct"]["trigger_f1"],
                ),
            ),
        },
    }
    write_json(output_json, payload)
    write_text(output_md, base.render_report(payload))
    print(json.dumps({"output_json": output_json.as_posix(), "output_md": output_md.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
