#!/usr/bin/env python3
"""Score E118 on divergent-token and complete-response margins."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from score_e115_training_diagnostics_20260712 import (
    encoded_conversation,
    load_jsonl,
    message_content,
)
from src.stage2_preference.difference_masking import divergent_token_indices


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def assistant_scores(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    response: str,
    cutoff_len: int,
) -> tuple[torch.Tensor, list[int], torch.Tensor, int]:
    prompt_ids, full_ids = encoded_conversation(tokenizer, row, response)
    if len(full_ids) > cutoff_len:
        raise ValueError(f"tokenized conversation exceeds cutoff: {len(full_ids)}")
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": message_content(row)},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    rendered_encoding = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )
    if rendered_encoding["input_ids"] != full_ids:
        raise ValueError("rendered chat does not reproduce chat-template ids")
    response_start = rendered.rfind(response)
    response_end = response_start + len(response)
    content_indices = [
        index
        for index, (start, end) in enumerate(rendered_encoding["offset_mapping"])
        if end > response_start and start < response_end
    ]
    assistant_indices = list(range(len(prompt_ids), len(full_ids)))
    if not content_indices or not assistant_indices:
        raise ValueError("assistant token accounting is empty")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    token_logps = functional.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)[0]
    assistant_logps = token_logps[[index - 1 for index in assistant_indices]].cpu()
    content_logps = token_logps[[index - 1 for index in content_indices]].cpu()
    assistant_ids = [full_ids[index] for index in assistant_indices]
    if assistant_logps.numel() != len(assistant_ids):
        raise AssertionError("assistant label-token accounting mismatch")
    return assistant_logps, assistant_ids, content_logps, len(full_ids)


def aggregate(rows: list[dict[str, Any]], margin_key: str) -> dict[str, Any]:
    margins = [float(row[margin_key]) for row in rows]
    by_category: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_category[str(row["error_category"])].append(float(row[margin_key]))
    return {
        "mean_margin": mean(margins),
        "median_margin": statistics.median(margins),
        "preference_accuracy": mean([float(value > 0.0) for value in margins]),
        "category_mean_margins": {
            category: mean(values) for category, values in sorted(by_category.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--preference_jsonl", type=Path, required=True)
    parser.add_argument("--mask_manifest", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--cutoff_len", type=int, default=1536)
    parser.add_argument("--context_tokens", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    scored = []
    source_rows = load_jsonl(args.preference_jsonl)
    manifest_rows = {
        str(row["wnd_id"]): row for row in load_jsonl(args.mask_manifest)
    }
    if len(manifest_rows) != len(source_rows):
        raise ValueError("mask manifest and preference row counts differ")
    for index, row in enumerate(source_rows, start=1):
        chosen_label_logps, chosen_ids, chosen_content_logps, chosen_total = assistant_scores(
            model, tokenizer, row, row["chosen"], args.cutoff_len
        )
        rejected_label_logps, rejected_ids, rejected_content_logps, rejected_total = assistant_scores(
            model, tokenizer, row, row["rejected"], args.cutoff_len
        )
        chosen_keep, rejected_keep = divergent_token_indices(
            chosen_ids, rejected_ids, args.context_tokens
        )
        wnd_id = str(row["meta"]["wnd_id"])
        manifest = manifest_rows.get(wnd_id)
        if manifest is None:
            raise ValueError(f"window absent from mask manifest: {wnd_id}")
        if chosen_keep != manifest["chosen_keep_indices"] or rejected_keep != manifest[
            "rejected_keep_indices"
        ]:
            raise ValueError(f"runtime mask differs from frozen manifest: {wnd_id}")
        chosen_masked = float(chosen_label_logps[chosen_keep].mean().item())
        rejected_masked = float(rejected_label_logps[rejected_keep].mean().item())
        chosen_full = float(chosen_content_logps.mean().item())
        rejected_full = float(rejected_content_logps.mean().item())
        values = [chosen_masked, rejected_masked, chosen_full, rejected_full]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite score for row {index}")
        scored.append(
            {
                "wnd_id": wnd_id,
                "document_id": wnd_id.rsplit("-", 1)[0],
                "error_category": row["meta"]["error_category"],
                "chosen_masked_logp": chosen_masked,
                "rejected_masked_logp": rejected_masked,
                "masked_margin": chosen_masked - rejected_masked,
                "chosen_full_logp": chosen_full,
                "rejected_full_logp": rejected_full,
                "full_margin": chosen_full - rejected_full,
                "chosen_kept_tokens": len(chosen_keep),
                "rejected_kept_tokens": len(rejected_keep),
                "chosen_label_tokens": int(chosen_label_logps.numel()),
                "rejected_label_tokens": int(rejected_label_logps.numel()),
                "chosen_response_tokens": int(chosen_content_logps.numel()),
                "rejected_response_tokens": int(rejected_content_logps.numel()),
                "chosen_total_tokens": chosen_total,
                "rejected_total_tokens": rejected_total,
            }
        )
        if index % 10 == 0 or index == len(source_rows):
            print(f"scored E118 pairs: {index}/{len(source_rows)}", flush=True)

    result = {
        "protocol": "E118 difference-masked atomic SimPO smoke",
        "model_path": args.model_path,
        "preference_jsonl": str(args.preference_jsonl),
        "mask_manifest": str(args.mask_manifest),
        "cutoff_len": args.cutoff_len,
        "context_tokens": args.context_tokens,
        "pairs": len(scored),
        "masked": aggregate(scored, "masked_margin"),
        "full_response": aggregate(scored, "full_margin"),
        "rows": scored,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
