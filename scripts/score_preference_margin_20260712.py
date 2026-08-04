#!/usr/bin/env python3
"""Compute post-training chosen/rejected average-token log-probability margins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def message_content(row: dict[str, Any]) -> str:
    return f"{row['instruction']}\n{row['input']}"


def encoded_conversation(tokenizer: Any, row: dict[str, Any], response: str):
    user = {"role": "user", "content": message_content(row)}
    prompt_ids = tokenizer.apply_chat_template(
        [user], tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        [user, {"role": "assistant", "content": response}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant conversation does not share the generation-prompt prefix")
    return prompt_ids, full_ids


def average_response_logp(
    model: Any, tokenizer: Any, row: dict[str, Any], response: str, cutoff_len: int
) -> tuple[float, int, int]:
    prompt_ids, full_ids = encoded_conversation(tokenizer, row, response)
    if len(full_ids) > cutoff_len:
        raise ValueError(f"tokenized conversation exceeds cutoff: {len(full_ids)} > {cutoff_len}")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    token_logps = functional.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    response_start = max(len(prompt_ids) - 1, 0)
    selected = token_logps[:, response_start:]
    return float(selected.mean().item()), selected.numel(), len(full_ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--preference_jsonl", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--cutoff_len", type=int, default=1536)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    results = []
    for row in load_jsonl(args.preference_jsonl):
        chosen_logp, chosen_tokens, chosen_total = average_response_logp(
            model, tokenizer, row, row["chosen"], args.cutoff_len
        )
        rejected_logp, rejected_tokens, rejected_total = average_response_logp(
            model, tokenizer, row, row["rejected"], args.cutoff_len
        )
        results.append(
            {
                "wnd_id": row.get("meta", {}).get("wnd_id"),
                "chosen_logp": chosen_logp,
                "rejected_logp": rejected_logp,
                "margin": chosen_logp - rejected_logp,
                "reward_margin": args.beta * (chosen_logp - rejected_logp),
                "chosen_response_tokens": chosen_tokens,
                "rejected_response_tokens": rejected_tokens,
                "chosen_total_tokens": chosen_total,
                "rejected_total_tokens": rejected_total,
            }
        )
    margins = [row["margin"] for row in results]
    ordered = sorted(margins)
    summary = {
        "pairs": len(results),
        "mean_margin": sum(margins) / len(margins) if margins else 0.0,
        "mean_reward_margin": args.beta * sum(margins) / len(margins) if margins else 0.0,
        "median_margin": ordered[len(ordered) // 2] if ordered else 0.0,
        "preference_accuracy": sum(margin > 0 for margin in margins) / len(margins)
        if margins
        else 0.0,
        "positive_mean_reward_margin": bool(margins and sum(margins) > 0),
        "beta": args.beta,
        "model_path": args.model_path,
        "preference_jsonl": str(args.preference_jsonl.resolve()),
        "rows": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0 if summary["positive_mean_reward_margin"] else 6


if __name__ == "__main__":
    raise SystemExit(main())

