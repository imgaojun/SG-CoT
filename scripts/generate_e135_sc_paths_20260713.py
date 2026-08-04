#!/usr/bin/env python3
"""Generate resumable greedy plus eight-path E135 train-only samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selection_key(seed: int, source_index: int, wnd_id: str) -> str:
    payload = f"{seed}\0{source_index}\0{wnd_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def experiment_prefix(protocol: dict[str, Any]) -> str:
    prefix = protocol.get("request_prefix", "e135")
    if not isinstance(prefix, str) or re.fullmatch(r"e\d+", prefix) is None:
        raise ValueError(f"invalid experiment prefix: {prefix!r}")
    if not str(protocol.get("id", "")).startswith(prefix + "_"):
        raise ValueError("protocol id/request prefix mismatch")
    return prefix


def validate_manifest(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    selection = protocol["selection"]
    prefix = experiment_prefix(protocol)
    if len(rows) != int(selection["rows"]):
        raise ValueError("smoke manifest row count mismatch")
    wnd_ids = []
    for rank, row in enumerate(rows):
        meta = row.get("meta") or {}
        wnd_id = str(meta.get("wnd_id") or "")
        source_index = int(meta.get(f"{prefix}_source_index", -1))
        expected_key = selection_key(int(selection["seed"]), source_index, wnd_id)
        if meta.get("source_part") != selection["source_part"]:
            raise ValueError(f"non-train manifest row at rank {rank}")
        if int(meta.get(f"{prefix}_selection_rank", -1)) != rank:
            raise ValueError(f"selection rank mismatch at rank {rank}")
        if meta.get(f"{prefix}_selection_key") != expected_key:
            raise ValueError(f"selection key mismatch at rank {rank}")
        wnd_ids.append(wnd_id)
    if len(set(wnd_ids)) != len(wnd_ids):
        raise ValueError("duplicate wnd_id in smoke manifest")


def validate_existing_prefix(
    generated: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    n_samples: int,
    prefix: str = "e135",
) -> None:
    if len(generated) > len(manifest):
        raise ValueError("generation output is longer than manifest")
    for rank, row in enumerate(generated):
        meta = manifest[rank]["meta"]
        if int(row.get("selection_rank", -1)) != rank:
            raise ValueError(f"non-prefix generation at rank {rank}")
        if row.get("wnd_id") != meta["wnd_id"]:
            raise ValueError(f"generation wnd_id mismatch at rank {rank}")
        if int(row.get("source_index", -1)) != int(meta[f"{prefix}_source_index"]):
            raise ValueError(f"generation source index mismatch at rank {rank}")
        if not isinstance(row.get("greedy_text"), str):
            raise ValueError(f"missing greedy text at rank {rank}")
        sampled = row.get("sampled_texts")
        if not isinstance(sampled, list) or len(sampled) != n_samples:
            raise ValueError(f"sample count mismatch at rank {rank}")
        if not all(isinstance(text, str) for text in sampled):
            raise ValueError(f"non-string sampled text at rank {rank}")


def validate_existing_shard_prefix(
    generated: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    n_samples: int,
    expected_ranks: list[int],
    prefix: str,
) -> None:
    if len(generated) > len(expected_ranks):
        raise ValueError("generation output is longer than shard manifest")
    for position, row in enumerate(generated):
        rank = expected_ranks[position]
        meta = manifest[rank]["meta"]
        if int(row.get("selection_rank", -1)) != rank:
            raise ValueError(f"non-prefix shard generation at position {position}")
        if row.get("wnd_id") != meta["wnd_id"]:
            raise ValueError(f"shard generation wnd_id mismatch at rank {rank}")
        if int(row.get("source_index", -1)) != int(meta[f"{prefix}_source_index"]):
            raise ValueError(f"shard generation source index mismatch at rank {rank}")
        if not isinstance(row.get("greedy_text"), str):
            raise ValueError(f"missing greedy text at rank {rank}")
        sampled = row.get("sampled_texts")
        if not isinstance(sampled, list) or len(sampled) != n_samples:
            raise ValueError(f"sample count mismatch at rank {rank}")
        if not all(isinstance(text, str) for text in sampled):
            raise ValueError(f"non-string sampled text at rank {rank}")


def build_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = [
        {
            "role": "user",
            "content": f"{row['instruction']}\n{row['input']}",
        }
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return messages[0]["content"]


def generate_rows(
    protocol: dict[str, Any],
    manifest: list[dict[str, Any]],
    ranks: list[int],
    start_position: int,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = REPO_ROOT / protocol["source_model"]
    generation = protocol["generation"]
    prefix = experiment_prefix(protocol)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    n_samples = int(generation["n_samples"])
    max_new_tokens = int(generation["max_new_tokens"])
    for position in range(start_position, len(ranks)):
        rank = ranks[position]
        row = manifest[rank]
        source_index = int(row["meta"][f"{prefix}_source_index"])
        prompt = build_prompt(tokenizer, row)
        encoded = tokenizer([prompt], return_tensors="pt").to(model.device)
        prompt_length = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            greedy = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            row_seed = int(generation["seed"]) + source_index
            torch.manual_seed(row_seed)
            torch.cuda.manual_seed_all(row_seed)
            sampled = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        yield {
            "selection_rank": rank,
            "source_index": source_index,
            "wnd_id": row["meta"]["wnd_id"],
            "row_seed": row_seed,
            "prompt_tokens": prompt_length,
            "greedy_text": tokenizer.decode(
                greedy[0][prompt_length:], skip_special_tokens=True
            ),
            "sampled_texts": [
                tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
                for sequence in sampled
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_shards", type=int)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    prefix = experiment_prefix(protocol)
    model_path = REPO_ROOT / protocol["source_model"]
    if sha256_file(model_path / "model.safetensors.index.json") != protocol[
        "source_model_index_sha256"
    ]:
        raise ValueError("source model index hash mismatch")
    if sha256_file(model_path / "trainer_state.json") != protocol[
        "source_trainer_state_sha256"
    ]:
        raise ValueError("source trainer-state hash mismatch")
    trainer_state = json.loads((model_path / "trainer_state.json").read_text())
    if int(trainer_state.get("global_step", -1)) != int(protocol["source_global_step"]):
        raise ValueError("source global step mismatch")

    manifest = load_jsonl(args.manifest_jsonl)
    validate_manifest(manifest, protocol)
    registered_shards = int(protocol["generation"].get("num_shards", 1))
    num_shards = args.num_shards if args.num_shards is not None else registered_shards
    if num_shards != registered_shards:
        raise ValueError("runtime shard count differs from frozen protocol")
    if num_shards < 1 or not 0 <= args.shard_index < num_shards:
        raise ValueError("invalid generation shard specification")
    ranks = list(range(args.shard_index, len(manifest), num_shards))
    sharded = num_shards != 1 or "num_shards" in protocol["generation"]
    output_paths = {
        "config": args.output_dir / "generation_config.json",
        "raw": args.output_dir / "raw_generations.jsonl",
        "summary": args.output_dir / "summary.json",
    }
    run_config = {
        "id": protocol.get("report_ids", {}).get(
            "generation_config", "e135_sc_path_generation_smoke64_v1"
        ),
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_sha256": sha256_file(args.manifest_jsonl),
        "source_model": protocol["source_model"],
        "source_global_step": int(protocol["source_global_step"]),
        "generation": protocol["generation"],
        "test_rows_read": 0,
    }
    if sharded:
        run_config["shard"] = {
            "index": args.shard_index,
            "num_shards": num_shards,
            "global_ranks": ranks,
        }

    existing = []
    if args.output_dir.exists():
        if not args.resume:
            raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
        if not output_paths["config"].is_file() or not output_paths["raw"].is_file():
            raise ValueError("resume requires generation_config.json and raw_generations.jsonl")
        if json.loads(output_paths["config"].read_text()) != run_config:
            raise ValueError("resume configuration mismatch")
        existing = load_jsonl(output_paths["raw"])
        validate_existing_shard_prefix(
            existing,
            manifest,
            int(protocol["generation"]["n_samples"]),
            ranks,
            prefix,
        )
        if output_paths["summary"].exists() and len(existing) != len(ranks):
            raise ValueError("incomplete output unexpectedly has a summary")
    else:
        if args.resume:
            raise ValueError("cannot resume a missing output directory")
        args.output_dir.mkdir(parents=True)
        output_paths["config"].write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if len(existing) == len(ranks):
        print(f"generation already complete: {len(existing)}/{len(ranks)}")
    else:
        with output_paths["raw"].open("a", encoding="utf-8") as handle:
            for generated in generate_rows(protocol, manifest, ranks, len(existing)):
                handle.write(json.dumps(generated, ensure_ascii=False) + "\n")
                handle.flush()
                existing.append(generated)
                print(
                    f"generated {len(existing)}/{len(ranks)} "
                    f"wnd_id={generated['wnd_id']}",
                    flush=True,
                )

    validate_existing_shard_prefix(
        existing,
        manifest,
        int(protocol["generation"]["n_samples"]),
        ranks,
        prefix,
    )
    summary = {
        "id": protocol.get("report_ids", {}).get(
            "generation_summary", "e135_sc_path_generation_smoke64_summary_v1"
        ),
        "rows": len(existing),
        "required_rows": len(ranks),
        "samples_per_row": int(protocol["generation"]["n_samples"]),
        "complete": len(existing) == len(ranks),
        "test_rows_read": 0,
        "raw_sha256": sha256_file(output_paths["raw"]),
    }
    if sharded:
        summary["shard"] = {
            "index": args.shard_index,
            "num_shards": num_shards,
            "first_global_rank": ranks[0] if ranks else None,
            "last_global_rank": ranks[-1] if ranks else None,
        }
    output_paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
