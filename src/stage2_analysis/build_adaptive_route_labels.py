import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_cot.build_selective_aux_reasoning_dataset import (  # noqa: E402
    build_confrare_stats,
    confrare_score_row,
    confrole_score_row,
    hardconf_score_row,
    parse_output_events,
    roleconf_score_row,
    row_id,
)
from src.stage2_data.build_formal_stage2_dataset import load_jsonl, load_schema_map, write_json  # noqa: E402


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prediction_id(row):
    meta = row.get("meta", {})
    if meta.get("wnd_id"):
        return meta["wnd_id"]
    if row.get("wnd_id"):
        return row["wnd_id"]
    return row_id(row)


def input_key(row):
    text = row.get("input", "")
    return "input::" + hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_prediction_map(path: Path):
    rows = load_jsonl(path)
    mapping = {}
    for row in rows:
        mapping[prediction_id(row)] = row
        if "input" in row:
            mapping[input_key(row)] = row
    return mapping


def gold_is_negative(row):
    return len(parse_output_events(row)) == 0


def metric(row, key):
    value = row.get(key)
    if value is None:
        return 0.0
    return float(value)


def outcome_score(prediction_row):
    return (
        metric(prediction_row, "argument_f1"),
        metric(prediction_row, "event_f1"),
        metric(prediction_row, "trigger_f1"),
    )


def scalar_improvement(direct_row, reason_row):
    direct = outcome_score(direct_row)
    reason = outcome_score(reason_row)
    return (
        (reason[0] - direct[0])
        + (reason[1] - direct[1])
        + 0.25 * (reason[2] - direct[2])
    )


def build_selector_scores(rows, schema_by_type, selector):
    stats = build_confrare_stats(rows)
    scored = []
    for idx, row in enumerate(rows):
        if selector == "confrare":
            item = confrare_score_row(idx, row, schema_by_type, stats)
            item["selector_score"] = item["confrare_score"]
        elif selector == "confrole":
            item = confrole_score_row(idx, row, schema_by_type, stats)
            item["selector_score"] = item["confrole_score"]
        elif selector == "roleconf":
            item = roleconf_score_row(idx, row, schema_by_type, stats)
            item["selector_score"] = item["roleconf_score"]
        elif selector == "hardconf":
            item = hardconf_score_row(idx, row, schema_by_type, stats)
            item["selector_score"] = item["hardconf_score"]
        else:
            raise ValueError(f"Unsupported selector for route labels: {selector}")
        scored.append(item)
    return scored


def build_heuristic_labels(rows, schema_by_type, source_name, reason_rate, selector):
    reason_count = round(len(rows) * reason_rate)
    scored = build_selector_scores(rows, schema_by_type, selector)
    score_by_id = {item["wnd_id"]: item for item in scored}
    ranked = sorted(scored, key=lambda item: (item["selector_score"], item["wnd_id"]), reverse=True)
    reason_ids = {item["wnd_id"] for item in ranked[:reason_count]}

    label_rows = []
    for idx, row in enumerate(rows):
        rid = row_id(row)
        score = score_by_id[rid]
        route_label = "direct" if gold_is_negative(row) else ("reason" if rid in reason_ids else "direct")
        label_rows.append(
            {
                "idx": idx,
                "wnd_id": rid,
                "route_label": route_label,
                "label_source": source_name,
                "selector": selector,
                "reason_rate_cap": reason_rate,
                "selector_score": score["selector_score"],
                "score_components": {k: v for k, v in score.items() if k not in {"idx", "wnd_id"}},
                "gold_is_negative": gold_is_negative(row),
            }
        )
    return label_rows


def build_outcome_labels(rows, direct_predictions, reason_predictions, source_name, reason_rate):
    direct_map = load_prediction_map(Path(direct_predictions))
    reason_map = load_prediction_map(Path(reason_predictions))
    candidates = []
    label_rows = []

    for idx, row in enumerate(rows):
        rid = row_id(row)
        direct_pred = direct_map.get(rid) or direct_map.get(input_key(row))
        reason_pred = reason_map.get(rid) or reason_map.get(input_key(row))
        if direct_pred is None or reason_pred is None:
            raise KeyError(f"Missing prediction for wnd_id={rid}")

        direct_score = outcome_score(direct_pred)
        reason_score = outcome_score(reason_pred)
        improvement = scalar_improvement(direct_pred, reason_pred)
        direct_wrong_reason_better = improvement > 1e-9 and reason_score > direct_score
        if gold_is_negative(row):
            direct_wrong_reason_better = False

        base = {
            "idx": idx,
            "wnd_id": rid,
            "route_label": "direct",
            "label_source": source_name,
            "reason_rate_cap": reason_rate,
            "gold_is_negative": gold_is_negative(row),
            "direct_argument_f1": direct_score[0],
            "direct_event_f1": direct_score[1],
            "direct_trigger_f1": direct_score[2],
            "reason_argument_f1": reason_score[0],
            "reason_event_f1": reason_score[1],
            "reason_trigger_f1": reason_score[2],
            "improvement_score": improvement,
            "outcome_reason_candidate": direct_wrong_reason_better,
        }
        label_rows.append(base)
        if direct_wrong_reason_better:
            candidates.append(base)

    reason_cap = round(len(rows) * reason_rate)
    candidates.sort(key=lambda item: (item["improvement_score"], item["wnd_id"]), reverse=True)
    reason_ids = {item["wnd_id"] for item in candidates[:reason_cap]}
    for item in label_rows:
        if item["wnd_id"] in reason_ids:
            item["route_label"] = "reason"
            item["outcome_selected_rank"] = 1 + next(
                rank for rank, cand in enumerate(candidates) if cand["wnd_id"] == item["wnd_id"]
            )
    return label_rows


def summarize(label_rows):
    total = len(label_rows)
    reason = sum(1 for row in label_rows if row["route_label"] == "reason")
    direct = total - reason
    improvements = [row.get("improvement_score", 0.0) for row in label_rows if row["route_label"] == "reason"]
    return {
        "num_examples": total,
        "direct_count": direct,
        "reason_count": reason,
        "reason_rate": reason / total if total else 0.0,
        "avg_selected_improvement_score": sum(improvements) / len(improvements) if improvements else 0.0,
        "max_selected_improvement_score": max(improvements) if improvements else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_jsonl", required=True)
    parser.add_argument("--schema_path", required=True)
    parser.add_argument("--label_source", required=True)
    parser.add_argument("--selector", choices=["confrare", "confrole", "roleconf", "hardconf"], default=None)
    parser.add_argument("--direct_predictions_jsonl", default=None)
    parser.add_argument("--reason_predictions_jsonl", default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary_json", required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.direct_jsonl))
    schema_by_type = load_schema_map(Path(args.schema_path))

    heuristic_rates = {
        "heuristic_confrare5": ("confrare", 0.05),
        "heuristic_confrare10": ("confrare", 0.10),
        "heuristic_confrole10": ("confrole", 0.10),
        "heuristic_roleconf5": ("roleconf", 0.05),
        "heuristic_roleconf10": ("roleconf", 0.10),
        "heuristic_hardconf10": ("hardconf", 0.10),
        "heuristic_hardconf15": ("hardconf", 0.15),
    }

    if args.label_source in heuristic_rates:
        default_selector, reason_rate = heuristic_rates[args.label_source]
        label_rows = build_heuristic_labels(
            rows,
            schema_by_type,
            args.label_source,
            reason_rate,
            args.selector or default_selector,
        )
    else:
        if args.label_source not in {"outcome_teacher10", "outcome_teacher15"}:
            raise ValueError(f"Unsupported label_source: {args.label_source}")
        if args.direct_predictions_jsonl is None or args.reason_predictions_jsonl is None:
            raise ValueError("outcome_teacher labels require --direct_predictions_jsonl and --reason_predictions_jsonl")
        reason_rate = 0.10 if args.label_source == "outcome_teacher10" else 0.15
        label_rows = build_outcome_labels(
            rows,
            args.direct_predictions_jsonl,
            args.reason_predictions_jsonl,
            args.label_source,
            reason_rate,
        )

    summary = {
        "direct_jsonl": args.direct_jsonl,
        "schema_path": args.schema_path,
        "label_source": args.label_source,
        "selector": args.selector,
        "direct_predictions_jsonl": args.direct_predictions_jsonl,
        "reason_predictions_jsonl": args.reason_predictions_jsonl,
        **summarize(label_rows),
    }
    write_jsonl(Path(args.output_jsonl), label_rows)
    write_json(Path(args.output_summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
