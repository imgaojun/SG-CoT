import argparse
import json
import random
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_dir_name(dataset: str, protocol: str, split: str, part: str, tag: str):
    dataset_slug = dataset.replace("-", "_")
    protocol_slug = protocol.replace("-", "_")
    return f"{dataset_slug}_{protocol_slug}_{split}_{part}_{tag}"


def load_schema_map(schema_path: Path):
    items = load_json(schema_path)
    return {item["event_type"]: item for item in items}


def event_family(event_type: str):
    return event_type.split(":", 1)[0] if ":" in event_type else event_type


def normalize_cue_tokens(trigger_cues):
    tokens = set()
    for cue in trigger_cues:
        for piece in cue.lower().replace("-", " ").replace("/", " ").split():
            if piece:
                tokens.add(piece)
    return tokens


def jaccard(left, right):
    union = set(left) | set(right)
    if not union:
        return 0.0
    return len(set(left) & set(right)) / len(union)


def row_rng(seed: int, wnd_id: str):
    return random.Random(f"{seed}:{wnd_id}")


def render_schema_cards(candidate_types, schema_by_type):
    cards = []
    for idx, event_type in enumerate(candidate_types, start=1):
        cards.append(f"[{idx}] {schema_by_type[event_type]['document']}")
    return "\n\n".join(cards)


def render_instruction():
    return (
        "You are doing event extraction. Use only the provided candidate event types and their schema cards. "
        "Extract all event mentions supported by the text and output strict JSON with token offsets. "
        "If no valid event is expressed by the candidate set, output {\"events\": []}."
    )


def render_input(row, candidate_types, schema_by_type):
    tokens_text = " ".join(row["tokens"])
    type_text = ", ".join(candidate_types)
    cards = render_schema_cards(candidate_types, schema_by_type)
    return (
        f"Text:\n{row['text']}\n\n"
        f"Tokens:\n{tokens_text}\n\n"
        f"Candidate event types:\n{type_text}\n\n"
        f"Schema cards:\n{cards}\n\n"
        "Return JSON only."
    )


def render_output(events):
    normalized = {"events": []}
    for event in events:
        normalized["events"].append(
            {
                "event_type": event["event_type"],
                "trigger": {
                    "text": event["trigger"]["text"],
                    "start": event["trigger"]["start"],
                    "end": event["trigger"]["end"],
                },
                "arguments": [
                    {
                        "role": arg["role"],
                        "text": arg["text"],
                        "start": arg["start"],
                        "end": arg["end"],
                    }
                    for arg in event["arguments"]
                ],
            }
        )
    return json.dumps(normalized, ensure_ascii=False)


def update_dataset_info(dataset_dir: Path, dataset_name: str, file_name: str):
    info_path = dataset_dir / "dataset_info.json"
    info = load_json(info_path) if info_path.exists() else {}
    info[dataset_name] = {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
    write_json(info_path, info)


def select_rows(rows, selection_mode: str, negative_ratio: float, seed: int):
    if selection_mode == "all":
        return rows
    positives = [row for row in rows if row["event_mentions"]]
    if selection_mode == "positive_only":
        return positives
    if selection_mode != "sampled_neg":
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    negatives = [row for row in rows if not row["event_mentions"]]
    rng = random.Random(seed)
    negatives = negatives[:]
    rng.shuffle(negatives)
    num_negatives = int(len(positives) * negative_ratio)
    sampled = positives + negatives[:num_negatives]
    rng.shuffle(sampled)
    return sampled


def build_prediction_map(prediction_path: Path):
    preds = load_jsonl(prediction_path)
    return {row["wnd_id"]: row for row in preds}


def oracle_candidate_types(row, candidate_universe, top_k, rng):
    gold_types = sorted({ev["event_type"] for ev in row["event_mentions"]})
    if top_k is None:
        return candidate_universe[:]
    if len(gold_types) > top_k:
        raise ValueError(
            f"top_k={top_k} is smaller than the number of gold types ({len(gold_types)}) for wnd_id={row['wnd_id']}"
        )
    distractors = [item for item in candidate_universe if item not in gold_types]
    rng.shuffle(distractors)
    return gold_types + distractors[: max(0, top_k - len(gold_types))]


def oracle_anchor_predicted_types(row, candidate_universe, ranked_types, top_k):
    gold_types = sorted({ev["event_type"] for ev in row["event_mentions"]})
    if top_k is None:
        top_k = len(candidate_universe)
    if len(gold_types) > top_k:
        raise ValueError(
            f"top_k={top_k} is smaller than the number of gold types ({len(gold_types)}) for wnd_id={row['wnd_id']}"
        )

    candidate_types = []
    for event_type in gold_types:
        if event_type not in candidate_types:
            candidate_types.append(event_type)

    for event_type in ranked_types:
        if event_type not in candidate_types:
            candidate_types.append(event_type)
        if len(candidate_types) >= top_k:
            return candidate_types[:top_k]

    for event_type in candidate_universe:
        if event_type not in candidate_types:
            candidate_types.append(event_type)
        if len(candidate_types) >= top_k:
            break

    return candidate_types[:top_k]


def hard_distractor_ranking(gold_types, distractors, schema_by_type, rng):
    shuffled = distractors[:]
    rng.shuffle(shuffled)

    def hardness_score(candidate_type: str):
        candidate_schema = schema_by_type[candidate_type]
        candidate_roles = set(candidate_schema.get("core_roles", []))
        candidate_cues = normalize_cue_tokens(candidate_schema.get("trigger_cues", []))
        best = 0.0
        for gold_type in gold_types:
            gold_schema = schema_by_type[gold_type]
            same_family = 1.0 if event_family(candidate_type) == event_family(gold_type) else 0.0
            role_overlap = jaccard(candidate_roles, gold_schema.get("core_roles", []))
            cue_overlap = jaccard(candidate_cues, normalize_cue_tokens(gold_schema.get("trigger_cues", [])))
            score = same_family * 10.0 + role_overlap * 3.0 + cue_overlap
            if score > best:
                best = score
        return best

    return sorted(shuffled, key=hardness_score, reverse=True)


def gold_contained_candidates(
    row,
    candidate_universe,
    schema_by_type,
    top_k,
    rng,
    noise_mode,
):
    gold_types = sorted({ev["event_type"] for ev in row["event_mentions"]})
    if top_k is None:
        top_k = len(candidate_universe)
    if len(gold_types) > top_k:
        raise ValueError(
            f"top_k={top_k} is smaller than the number of gold types ({len(gold_types)}) for wnd_id={row['wnd_id']}"
        )

    distractors = [item for item in candidate_universe if item not in gold_types]
    needed = max(0, top_k - len(gold_types))
    if noise_mode == "clean":
        selected = distractors[:needed]
    elif noise_mode == "random":
        shuffled = distractors[:]
        rng.shuffle(shuffled)
        selected = shuffled[:needed]
    elif noise_mode == "hard":
        selected = hard_distractor_ranking(gold_types, distractors, schema_by_type, rng)[:needed]
    elif noise_mode == "mixed":
        hard_ranked = hard_distractor_ranking(gold_types, distractors, schema_by_type, rng)
        hard_quota = (needed + 1) // 2
        hard_part = hard_ranked[:hard_quota]
        remaining = [item for item in distractors if item not in hard_part]
        rng.shuffle(remaining)
        selected = hard_part + remaining[: max(0, needed - len(hard_part))]
    else:
        raise ValueError(f"Unsupported noise_mode: {noise_mode}")

    return gold_types + selected


def inject_missing_gold_into_predicted(row, candidate_universe, ranked_types, top_k):
    gold_types = sorted({ev["event_type"] for ev in row["event_mentions"]})
    if top_k is None:
        top_k = len(candidate_universe)
    if len(gold_types) > top_k:
        raise ValueError(
            f"top_k={top_k} is smaller than the number of gold types ({len(gold_types)}) for wnd_id={row['wnd_id']}"
        )

    candidate_types = ranked_types[:top_k]
    present = set(candidate_types)
    missing_gold = [event_type for event_type in gold_types if event_type not in present]
    if not missing_gold:
        return candidate_types[:top_k]

    keep = [event_type for event_type in candidate_types if event_type not in missing_gold]
    max_keep = max(0, top_k - len(missing_gold))
    keep = keep[:max_keep]
    injected = keep + missing_gold

    for event_type in candidate_universe:
        if event_type not in injected:
            injected.append(event_type)
        if len(injected) >= top_k:
            break

    return injected[:top_k]


def apply_candidate_order_mode(candidate_types, order_mode, rng):
    ordered = candidate_types[:]
    if order_mode == "as_is":
        return ordered
    if order_mode == "sorted":
        return sorted(ordered)
    if order_mode == "shuffle":
        rng.shuffle(ordered)
        return ordered
    raise ValueError(f"Unsupported candidate_order_mode: {order_mode}")


def resolve_candidate_types(
    row,
    schema_by_type,
    candidate_universe,
    candidate_source,
    top_k,
    prediction_map,
    seed,
    candidate_order_mode,
):
    rng = row_rng(seed, row["wnd_id"])
    raw_predicted_topk = None
    noise_mode = None
    gold_types = {event["event_type"] for event in row["event_mentions"]}
    allowed_prediction_types = set(candidate_universe) | gold_types

    if candidate_source == "predicted":
        pred = prediction_map.get(row["wnd_id"])
        if pred is None:
            raise ValueError(f"Missing stage-1 prediction for wnd_id={row['wnd_id']}")
        filtered_ranked_types = [
            event_type for event_type in pred["ranked_types"] if event_type in allowed_prediction_types
        ]
        raw_predicted_topk = filtered_ranked_types[:top_k]
        candidate_types = raw_predicted_topk[:]
    elif candidate_source == "oracle_clean":
        noise_mode = "clean"
        candidate_types = gold_contained_candidates(
            row=row,
            candidate_universe=candidate_universe,
            schema_by_type=schema_by_type,
            top_k=top_k,
            rng=rng,
            noise_mode=noise_mode,
        )
    elif candidate_source == "oracle_anchor_predicted":
        pred = prediction_map.get(row["wnd_id"])
        if pred is None:
            raise ValueError(f"Missing stage-1 prediction for wnd_id={row['wnd_id']}")
        filtered_ranked_types = [
            event_type for event_type in pred["ranked_types"] if event_type in allowed_prediction_types
        ]
        raw_predicted_topk = filtered_ranked_types[:top_k]
        candidate_types = oracle_anchor_predicted_types(
            row=row,
            candidate_universe=candidate_universe,
            ranked_types=filtered_ranked_types,
            top_k=top_k,
        )
    elif candidate_source == "oracle_inject_missing_predicted":
        pred = prediction_map.get(row["wnd_id"])
        if pred is None:
            raise ValueError(f"Missing stage-1 prediction for wnd_id={row['wnd_id']}")
        filtered_ranked_types = [
            event_type for event_type in pred["ranked_types"] if event_type in allowed_prediction_types
        ]
        raw_predicted_topk = filtered_ranked_types[:top_k]
        candidate_types = inject_missing_gold_into_predicted(
            row=row,
            candidate_universe=candidate_universe,
            ranked_types=filtered_ranked_types,
            top_k=top_k,
        )
    elif candidate_source == "oracle":
        noise_mode = "random"
        candidate_types = gold_contained_candidates(
            row=row,
            candidate_universe=candidate_universe,
            schema_by_type=schema_by_type,
            top_k=top_k,
            rng=rng,
            noise_mode=noise_mode,
        )
    elif candidate_source == "oracle_random_noise":
        noise_mode = "random"
        candidate_types = gold_contained_candidates(
            row=row,
            candidate_universe=candidate_universe,
            schema_by_type=schema_by_type,
            top_k=top_k,
            rng=rng,
            noise_mode=noise_mode,
        )
    elif candidate_source == "oracle_hard_noise":
        noise_mode = "hard"
        candidate_types = gold_contained_candidates(
            row=row,
            candidate_universe=candidate_universe,
            schema_by_type=schema_by_type,
            top_k=top_k,
            rng=rng,
            noise_mode=noise_mode,
        )
    elif candidate_source == "oracle_mixed_noise":
        noise_mode = "mixed"
        candidate_types = gold_contained_candidates(
            row=row,
            candidate_universe=candidate_universe,
            schema_by_type=schema_by_type,
            top_k=top_k,
            rng=rng,
            noise_mode=noise_mode,
        )
    elif candidate_source == "all_schema":
        candidate_types = candidate_universe[:]
    else:
        raise ValueError(f"Unsupported candidate_source: {candidate_source}")

    candidate_types = apply_candidate_order_mode(candidate_types, candidate_order_mode, rng)
    return {
        "candidate_types": candidate_types,
        "raw_predicted_topk": raw_predicted_topk,
        "noise_mode": noise_mode,
        "candidate_order_mode": candidate_order_mode,
    }


def build_samples(
    rows,
    schema_by_type,
    candidate_universe,
    candidate_source,
    top_k,
    prediction_map,
    seed,
    dataset_meta,
    candidate_order_mode,
):
    samples = []
    for row in rows:
        resolved = resolve_candidate_types(
            row=row,
            schema_by_type=schema_by_type,
            candidate_universe=candidate_universe,
            candidate_source=candidate_source,
            top_k=top_k,
            prediction_map=prediction_map,
            seed=seed,
            candidate_order_mode=candidate_order_mode,
        )
        candidate_types = resolved["candidate_types"]
        raw_predicted_topk = resolved["raw_predicted_topk"]
        noise_mode = resolved["noise_mode"]

        gold_event_types = sorted({ev["event_type"] for ev in row["event_mentions"]})
        sample = {
            "instruction": render_instruction(),
            "input": render_input(row, candidate_types, schema_by_type),
            "output": render_output(row["event_mentions"]),
            "meta": {
                "doc_id": row["doc_id"],
                "wnd_id": row["wnd_id"],
                "candidate_types": candidate_types,
                "gold_event_types": gold_event_types,
                "raw_predicted_topk": raw_predicted_topk,
                "missing_gold_from_raw_predicted_topk": (
                    sorted(set(gold_event_types) - set(raw_predicted_topk)) if raw_predicted_topk is not None else None
                ),
                "noise_mode": noise_mode,
                "candidate_order_mode": candidate_order_mode,
                **dataset_meta,
            },
        }
        samples.append(sample)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/processed/type_holdout")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--schema_path", required=True)
    parser.add_argument(
        "--candidate_source",
        choices=[
            "predicted",
            "oracle",
            "oracle_clean",
            "all_schema",
            "oracle_anchor_predicted",
            "oracle_inject_missing_predicted",
            "oracle_random_noise",
            "oracle_hard_noise",
            "oracle_mixed_noise",
        ],
        required=True,
    )
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--candidate_scope", choices=["all", "seen_only"], default="all")
    parser.add_argument("--candidate_order_mode", choices=["as_is", "shuffle", "sorted"], default="as_is")
    parser.add_argument("--prediction_tag", default="qwen8b_main")
    parser.add_argument("--prediction_model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument(
        "--prediction_part",
        default=None,
        help="Optional stage-1 prediction part. Defaults to --part. Useful for building test_seen/test_unseen from test predictions.",
    )
    parser.add_argument("--selection_mode", choices=["positive_only", "all", "sampled_neg"], default="positive_only")
    parser.add_argument("--negative_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--dataset_dir", default="data/stage2_formal_datasets")
    parser.add_argument("--dataset_name", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_root) / args.dataset / args.protocol / args.split
    rows = load_jsonl(data_dir / f"{args.part}.jsonl")
    rows = select_rows(rows, args.selection_mode, args.negative_ratio, args.seed)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    unseen_types = set(load_json(data_dir / "unseen_types.json"))
    seen_types = load_json(data_dir / "seen_types.json")
    candidate_universe = seen_types if args.candidate_scope == "seen_only" else seen_types + sorted(unseen_types)
    schema_by_type = load_schema_map(Path(args.schema_path))

    missing = [event_type for event_type in candidate_universe if event_type not in schema_by_type]
    if missing:
        raise ValueError(f"Schema entries missing for candidate types: {missing}")

    prediction_map = None
    if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}:
        prediction_part = args.prediction_part or args.part
        prediction_dir = (
            Path("outputs/candidate_type_recall_runs")
            / run_dir_name(args.dataset, args.protocol, args.split, prediction_part, args.prediction_tag)
            / args.prediction_model.replace("/", "__")
        )
        prediction_map = build_prediction_map(prediction_dir / "predictions.jsonl")

    dataset_meta = {
        "source_dataset": args.dataset,
        "source_protocol": args.protocol,
        "source_split": args.split,
        "source_part": args.part,
        "candidate_source": args.candidate_source,
        "candidate_scope": args.candidate_scope,
        "top_k": args.top_k,
        "candidate_order_mode": args.candidate_order_mode,
        "selection_mode": args.selection_mode,
        "prediction_model": args.prediction_model
        if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}
        else None,
        "prediction_part": (args.prediction_part or args.part)
        if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}
        else None,
    }
    samples = build_samples(
        rows=rows,
        schema_by_type=schema_by_type,
        candidate_universe=candidate_universe,
        candidate_source=args.candidate_source,
        top_k=args.top_k,
        prediction_map=prediction_map,
        seed=args.seed,
        dataset_meta=dataset_meta,
        candidate_order_mode=args.candidate_order_mode,
    )

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{args.dataset_name}.jsonl"
    output_path = dataset_dir / file_name
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    update_dataset_info(dataset_dir, args.dataset_name, file_name)
    meta = {
        "dataset_name": args.dataset_name,
        "file_name": file_name,
        "num_examples": len(samples),
        "data_root": args.data_root,
        "dataset": args.dataset,
        "protocol": args.protocol,
        "split": args.split,
        "part": args.part,
        "schema_path": args.schema_path,
        "candidate_source": args.candidate_source,
        "candidate_scope": args.candidate_scope,
        "top_k": args.top_k,
        "candidate_order_mode": args.candidate_order_mode,
        "prediction_tag": args.prediction_tag
        if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}
        else None,
        "prediction_model": args.prediction_model
        if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}
        else None,
        "prediction_part": (args.prediction_part or args.part)
        if args.candidate_source in {"predicted", "oracle_anchor_predicted", "oracle_inject_missing_predicted"}
        else None,
        "selection_mode": args.selection_mode,
        "negative_ratio": args.negative_ratio if args.selection_mode == "sampled_neg" else None,
        "seed": args.seed,
        "max_examples": args.max_examples,
    }
    write_json(dataset_dir / f"{args.dataset_name}.meta.json", meta)
    print(f"wrote {output_path}")
    print(f"updated {dataset_dir / 'dataset_info.json'}")
    print(f"wrote {dataset_dir / f'{args.dataset_name}.meta.json'}")


if __name__ == "__main__":
    main()
