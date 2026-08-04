import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_labels(score_rows, reason_rate_cap: float, margin: float, label_source: str):
    ranked = [
        row for row in score_rows
        if row.get("delta_final_nll") is not None and row["delta_final_nll"] > margin
    ]
    ranked.sort(key=lambda row: (row["delta_final_nll"], row["wnd_id"]), reverse=True)
    cap = round(len(score_rows) * reason_rate_cap)
    reason_ids = {row["wnd_id"] for row in ranked[:cap]}
    labels = []
    for row in sorted(score_rows, key=lambda item: item["wnd_id"]):
        route_label = "reason" if row["wnd_id"] in reason_ids else "direct"
        labels.append(
            {
                "wnd_id": row["wnd_id"],
                "route_label": route_label,
                "label_source": label_source,
                "reason_rate_cap": reason_rate_cap,
                "margin": margin,
                "delta_final_nll": row.get("delta_final_nll"),
                "nll_direct_final": row.get("nll_direct_final"),
                "nll_reason_final": row.get("nll_reason_final"),
                "nll_reason_plan": row.get("nll_reason_plan"),
                "score_model": row.get("score_model"),
                "split": row.get("split"),
            }
        )
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--reason_rate_cap", type=float, required=True)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--label_source", required=True)
    args = parser.parse_args()

    score_rows = load_jsonl(Path(args.scores_jsonl))
    labels = build_labels(score_rows, args.reason_rate_cap, args.margin, args.label_source)
    write_jsonl(Path(args.output_jsonl), labels)
    reason_rows = [row for row in labels if row["route_label"] == "reason"]
    summary = {
        "scores_jsonl": args.scores_jsonl,
        "output_jsonl": args.output_jsonl,
        "label_source": args.label_source,
        "reason_rate_cap": args.reason_rate_cap,
        "margin": args.margin,
        "num_examples": len(labels),
        "reason_count": len(reason_rows),
        "direct_count": len(labels) - len(reason_rows),
        "reason_rate": len(reason_rows) / len(labels) if labels else 0.0,
        "min_reason_delta": min([row["delta_final_nll"] for row in reason_rows], default=None),
        "max_reason_delta": max([row["delta_final_nll"] for row in reason_rows], default=None),
    }
    write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
