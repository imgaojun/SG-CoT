import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from siliconflow_rerank import (
    SiliconFlowRerankerClient,
    default_instruction,
    load_api_key,
    parse_rerank_results,
    sanitize_model_name,
)


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_schema_map(schema_path: Path):
    items = json.load(open(schema_path, "r", encoding="utf-8"))
    return {item["event_type"]: item for item in items}


def evaluate_rows(rows, ranked_types_per_row, top_ks, unseen_types):
    total_gold = 0
    total_unseen_gold = 0
    micro_hits = {k: 0 for k in top_ks}
    micro_unseen_hits = {k: 0 for k in top_ks}
    avg_window_coverage = {k: 0.0 for k in top_ks}
    avg_unseen_window_coverage = {k: 0.0 for k in top_ks}
    window_count = 0
    unseen_window_count = 0

    for row, ranked_types in zip(rows, ranked_types_per_row):
        gold_types = sorted({ev["event_type"] for ev in row["event_mentions"]})
        if not gold_types:
            continue
        window_count += 1
        unseen_gold = sorted(t for t in gold_types if t in unseen_types)
        if unseen_gold:
            unseen_window_count += 1

        total_gold += len(gold_types)
        total_unseen_gold += len(unseen_gold)
        for k in top_ks:
            topk = set(ranked_types[:k])
            hits = len(set(gold_types) & topk)
            micro_hits[k] += hits
            avg_window_coverage[k] += hits / len(gold_types)
            if unseen_gold:
                unseen_hits = len(set(unseen_gold) & topk)
                micro_unseen_hits[k] += unseen_hits
                avg_unseen_window_coverage[k] += unseen_hits / len(unseen_gold)

    metrics = {}
    for k in top_ks:
        metrics[f"micro_recall@{k}"] = micro_hits[k] / total_gold if total_gold else 0.0
        metrics[f"avg_window_coverage@{k}"] = avg_window_coverage[k] / window_count if window_count else 0.0
        metrics[f"unseen_micro_recall@{k}"] = micro_unseen_hits[k] / total_unseen_gold if total_unseen_gold else 0.0
        metrics[f"avg_unseen_window_coverage@{k}"] = (
            avg_unseen_window_coverage[k] / unseen_window_count if unseen_window_count else 0.0
        )

    metrics["evaluated_rows"] = window_count
    metrics["rows_with_unseen_gold"] = unseen_window_count
    metrics["total_gold_types"] = total_gold
    metrics["total_unseen_gold_types"] = total_unseen_gold
    return metrics


def run_model(
    client,
    model,
    rows,
    candidate_types,
    candidate_docs,
    top_ks,
    unseen_types,
    output_dir,
    num_workers,
):
    instruction = default_instruction(model)
    top_n = max(top_ks)
    ranked_types_per_row = [None] * len(rows)
    prediction_rows = [None] * len(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"

    def process_one(index_row):
        index, row = index_row
        gold_types = sorted({ev["event_type"] for ev in row["event_mentions"]})
        response = client.rerank(
            model=model,
            query=row["text"],
            documents=candidate_docs,
            top_n=top_n,
            instruction=instruction,
        )
        parsed = parse_rerank_results(response)
        ranked_types = [candidate_types[item["index"]] for item in parsed]
        ranked_scores = [item["score"] for item in parsed]
        pred = {
            "doc_id": row["doc_id"],
            "wnd_id": row["wnd_id"],
            "text": row["text"],
            "gold_types": gold_types,
            "unseen_gold_types": sorted(t for t in gold_types if t in unseen_types),
            "ranked_types": ranked_types,
            "ranked_scores": ranked_scores,
        }
        return index, ranked_types, pred

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one, (idx, row)) for idx, row in enumerate(rows)]
        completed = 0
        for future in as_completed(futures):
            index, ranked_types, pred = future.result()
            ranked_types_per_row[index] = ranked_types
            prediction_rows[index] = pred
            completed += 1
            if completed % 100 == 0 or completed == len(rows):
                print(f"{model}: {completed}/{len(rows)} done")

    with open(pred_path, "w", encoding="utf-8") as pred_fp:
        for pred in prediction_rows:
            pred_fp.write(json.dumps(pred, ensure_ascii=False) + "\n")

    metrics = evaluate_rows(rows, ranked_types_per_row, top_ks, unseen_types)
    metrics["model"] = model
    metrics["instruction_used"] = bool(instruction)
    metrics["candidate_type_count"] = len(candidate_types)
    metrics["top_ks"] = top_ks
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/processed/type_holdout")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--part", default="dev_seen")
    parser.add_argument("--schema_path", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--top_k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default="outputs/candidate_type_recall_cache")
    parser.add_argument("--api_key_env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--include_eventless",
        action="store_true",
        help="If set, rerank all windows including those without event mentions. Default keeps the earlier positive-only behavior.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_root) / args.dataset / args.protocol / args.split
    rows = load_jsonl(data_dir / f"{args.part}.jsonl")
    if not args.include_eventless:
        rows = [row for row in rows if row["event_mentions"]]
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    unseen_types = set(json.load(open(data_dir / "unseen_types.json", "r", encoding="utf-8")))
    seen_types = json.load(open(data_dir / "seen_types.json", "r", encoding="utf-8"))
    candidate_types = seen_types + sorted(unseen_types)

    schema_map = load_schema_map(Path(args.schema_path))
    missing = [t for t in candidate_types if t not in schema_map]
    if missing:
        raise ValueError(f"Schema entries missing for candidate types: {missing}")
    candidate_docs = [schema_map[t]["document"] for t in candidate_types]

    client = SiliconFlowRerankerClient(
        api_key=load_api_key(args.api_key_env),
        cache_dir=args.cache_dir,
    )

    output_root = Path(args.output_dir)
    summary = {}
    for model in args.models:
        model_dir = output_root / sanitize_model_name(model)
        metrics = run_model(
            client=client,
            model=model,
            rows=rows,
            candidate_types=candidate_types,
            candidate_docs=candidate_docs,
            top_ks=sorted(args.top_k),
            unseen_types=unseen_types,
            output_dir=model_dir,
            num_workers=args.num_workers,
        )
        summary[model] = metrics
        print(model, metrics)

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    metadata = {
        "dataset": args.dataset,
        "protocol": args.protocol,
        "split": args.split,
        "part": args.part,
        "schema_path": args.schema_path,
        "models": args.models,
        "top_k": sorted(args.top_k),
        "max_examples": args.max_examples,
        "candidate_types": candidate_types,
        "evaluated_rows": len(rows),
        "num_workers": args.num_workers,
        "include_eventless": args.include_eventless,
    }
    with open(output_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
