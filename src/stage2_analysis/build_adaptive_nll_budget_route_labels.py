import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_labels(rows, reason_rate_cap: float, label_source: str, router_checkpoint: str):
    scored = []
    seen = set()
    for row in rows:
        delta = row.get("delta_direct_minus_reason_route_nll")
        if row.get("wnd_id") is None:
            raise ValueError(f"score row missing wnd_id: {row}")
        if row["wnd_id"] in seen:
            raise ValueError(
                f"duplicate wnd_id in score rows: {row['wnd_id']}. "
                "Use a one-row-per-sample scoring dataset, not an oversampled training dataset."
            )
        seen.add(row["wnd_id"])
        if delta is None:
            delta = float("-inf")
        scored.append((float(delta), row.get("wnd_id"), row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    cap = round(len(scored) * reason_rate_cap)
    reason_ids = {wnd_id for _, wnd_id, _ in scored[:cap]}
    rank_by_id = {wnd_id: idx + 1 for idx, (_, wnd_id, _) in enumerate(scored)}

    labels = []
    for row in sorted(rows, key=lambda item: item.get("wnd_id") or ""):
        wnd_id = row["wnd_id"]
        delta = row.get("delta_direct_minus_reason_route_nll")
        labels.append(
            {
                "wnd_id": wnd_id,
                "route_label": "reason" if wnd_id in reason_ids else "direct",
                "label_source": label_source,
                "reason_rate_cap": reason_rate_cap,
                "router_checkpoint": router_checkpoint,
                "router_score_jsonl": None,
                "rank": rank_by_id[wnd_id],
                "delta_direct_minus_reason_route_nll": delta,
                "nll_direct_route": row.get("nll_direct_route"),
                "nll_reason_route": row.get("nll_reason_route"),
                "source_gold_route": row.get("gold_route"),
                "source_pred_route_argmin_nll": row.get("pred_route_argmin_nll"),
            }
        )
    return labels, scored[:cap]


def summarize(labels, selected, args):
    selected_deltas = [
        row.get("delta_direct_minus_reason_route_nll")
        for row in labels
        if row["route_label"] == "reason" and row.get("delta_direct_minus_reason_route_nll") is not None
    ]
    all_deltas = [row.get("delta_direct_minus_reason_route_nll") for row in labels if row.get("delta_direct_minus_reason_route_nll") is not None]
    return {
        "score_jsonl": args.score_jsonl,
        "output_jsonl": args.output_jsonl,
        "label_source": args.label_source,
        "router_checkpoint": args.router_checkpoint,
        "reason_rate_cap": args.reason_rate_cap,
        "num_examples": len(labels),
        "reason_count": sum(1 for row in labels if row["route_label"] == "reason"),
        "direct_count": sum(1 for row in labels if row["route_label"] != "reason"),
        "reason_rate": (
            sum(1 for row in labels if row["route_label"] == "reason") / len(labels)
            if labels
            else 0.0
        ),
        "min_selected_delta": min(selected_deltas) if selected_deltas else None,
        "max_selected_delta": max(selected_deltas) if selected_deltas else None,
        "avg_selected_delta": sum(selected_deltas) / len(selected_deltas) if selected_deltas else None,
        "avg_all_delta": sum(all_deltas) / len(all_deltas) if all_deltas else None,
        "top_selected": [
            {
                "rank": idx + 1,
                "wnd_id": wnd_id,
                "delta_direct_minus_reason_route_nll": delta,
            }
            for idx, (delta, wnd_id, _) in enumerate(selected[:20])
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--reason_rate_cap", type=float, required=True)
    parser.add_argument("--label_source", required=True)
    parser.add_argument("--router_checkpoint", required=True)
    args = parser.parse_args()

    if not 0.0 < args.reason_rate_cap < 1.0:
        raise ValueError("--reason_rate_cap must be between 0 and 1")

    rows = load_jsonl(Path(args.score_jsonl))
    labels, selected = build_labels(rows, args.reason_rate_cap, args.label_source, args.router_checkpoint)
    for row in labels:
        row["router_score_jsonl"] = args.score_jsonl
    write_jsonl(Path(args.output_jsonl), labels)
    summary = summarize(labels, selected, args)
    write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
