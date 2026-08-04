import argparse
import json
import statistics
from pathlib import Path


def load_summary(summary_path: Path, model_name: str):
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[model_name]


def run_dir_name(dataset: str, protocol: str, split: str, part: str, tag: str):
    dataset_slug = dataset.replace("-", "_")
    protocol_slug = protocol.replace("-", "_")
    return f"{dataset_slug}_{protocol_slug}_{split}_{part}_{tag}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="outputs/candidate_type_recall_runs")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--parts", nargs="+", default=["dev_seen", "test"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_markdown", required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    metrics_to_collect = [
        "micro_recall@1",
        "micro_recall@3",
        "micro_recall@5",
        "micro_recall@10",
        "unseen_micro_recall@1",
        "unseen_micro_recall@3",
        "unseen_micro_recall@5",
        "unseen_micro_recall@10",
    ]

    payload = {"model": args.model, "protocol": args.protocol, "datasets": {}}
    markdown_lines = []

    for dataset in args.datasets:
        dataset_entry = {"splits": {}, "aggregate": {}}
        markdown_lines.append(f"## {dataset}")
        for part in args.parts:
            markdown_lines.append(f"### {part}")
            headers = ["split", "R@1", "R@3", "R@5", "R@10"]
            if part == "test":
                headers += ["uR@1", "uR@3", "uR@5", "uR@10"]
            markdown_lines.append("| " + " | ".join(headers) + " |")
            markdown_lines.append("|" + "|".join(["---"] * len(headers)) + "|")

            part_rows = []
            aggregate_buckets = {metric: [] for metric in metrics_to_collect}
            for split in args.splits:
                run_dir = runs_root / run_dir_name(dataset, args.protocol, split, part, args.tag)
                summary_path = run_dir / "summary.json"
                metrics = load_summary(summary_path, args.model)
                dataset_entry["splits"].setdefault(split, {})[part] = metrics

                row = [
                    split,
                    f"{metrics['micro_recall@1']:.4f}",
                    f"{metrics['micro_recall@3']:.4f}",
                    f"{metrics['micro_recall@5']:.4f}",
                    f"{metrics['micro_recall@10']:.4f}",
                ]
                if part == "test":
                    row += [
                        f"{metrics['unseen_micro_recall@1']:.4f}",
                        f"{metrics['unseen_micro_recall@3']:.4f}",
                        f"{metrics['unseen_micro_recall@5']:.4f}",
                        f"{metrics['unseen_micro_recall@10']:.4f}",
                    ]
                markdown_lines.append("| " + " | ".join(row) + " |")
                part_rows.append(row)

                for metric in metrics_to_collect:
                    aggregate_buckets[metric].append(metrics.get(metric, 0.0))

            aggregate = {}
            for metric, values in aggregate_buckets.items():
                if part == "dev_seen" and metric.startswith("unseen_"):
                    continue
                aggregate[metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.pstdev(values),
                }
            dataset_entry["aggregate"][part] = aggregate

            avg_row = [
                "mean",
                f"{aggregate['micro_recall@1']['mean']:.4f}",
                f"{aggregate['micro_recall@3']['mean']:.4f}",
                f"{aggregate['micro_recall@5']['mean']:.4f}",
                f"{aggregate['micro_recall@10']['mean']:.4f}",
            ]
            if part == "test":
                avg_row += [
                    f"{aggregate['unseen_micro_recall@1']['mean']:.4f}",
                    f"{aggregate['unseen_micro_recall@3']['mean']:.4f}",
                    f"{aggregate['unseen_micro_recall@5']['mean']:.4f}",
                    f"{aggregate['unseen_micro_recall@10']['mean']:.4f}",
                ]
            std_row = [
                "std",
                f"{aggregate['micro_recall@1']['std']:.4f}",
                f"{aggregate['micro_recall@3']['std']:.4f}",
                f"{aggregate['micro_recall@5']['std']:.4f}",
                f"{aggregate['micro_recall@10']['std']:.4f}",
            ]
            if part == "test":
                std_row += [
                    f"{aggregate['unseen_micro_recall@1']['std']:.4f}",
                    f"{aggregate['unseen_micro_recall@3']['std']:.4f}",
                    f"{aggregate['unseen_micro_recall@5']['std']:.4f}",
                    f"{aggregate['unseen_micro_recall@10']['std']:.4f}",
                ]
            markdown_lines.append("| " + " | ".join(avg_row) + " |")
            markdown_lines.append("| " + " | ".join(std_row) + " |")
            markdown_lines.append("")

        payload["datasets"][dataset] = dataset_entry

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    output_markdown = Path(args.output_markdown)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    with open(output_markdown, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines) + "\n")


if __name__ == "__main__":
    main()
