#!/usr/bin/env python3
"""Score E115 preference margins and matched canonical/native response NLLs."""

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
        raise ValueError("assistant conversation does not share the prompt prefix")
    return prompt_ids, full_ids


def response_logps(
    model: Any, tokenizer: Any, row: dict[str, Any], response: str, cutoff_len: int
) -> tuple[torch.Tensor, list[tuple[int, int]], int]:
    prompt_ids, full_ids = encoded_conversation(tokenizer, row, response)
    if len(full_ids) > cutoff_len:
        raise ValueError(f"tokenized conversation exceeds cutoff: {len(full_ids)} > {cutoff_len}")
    user = {"role": "user", "content": message_content(row)}
    rendered = tokenizer.apply_chat_template(
        [user, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    rendered_encoding = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if rendered_encoding["input_ids"] != full_ids:
        raise ValueError("rendered chat text does not reproduce chat-template token ids")
    response_start = rendered.rfind(response)
    if response_start < 0:
        raise ValueError("assistant response is absent from rendered chat text")
    response_end = response_start + len(response)
    content_indices = [
        index
        for index, (token_start, token_end) in enumerate(
            rendered_encoding["offset_mapping"]
        )
        if token_end > response_start and token_start < response_end
    ]
    if not content_indices:
        raise ValueError("assistant response contains no mapped chat tokens")
    offsets = [
        (
            max(int(rendered_encoding["offset_mapping"][index][0]) - response_start, 0),
            min(int(rendered_encoding["offset_mapping"][index][1]) - response_start, len(response)),
        )
        for index in content_indices
    ]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    token_logps = functional.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    selected = token_logps[0, [index - 1 for index in content_indices]].detach().cpu()
    if selected.numel() != len(content_indices) or len(offsets) != len(content_indices):
        raise AssertionError("assistant content-token accounting mismatch")
    return selected, offsets, len(full_ids)


def tagged_segments(response: str) -> dict[str, tuple[int, int]]:
    thinking_start = response.find("<thinking>")
    thinking_close = response.find("</thinking>")
    final_start = response.find("<final>")
    final_close = response.find("</final>")
    if min(thinking_start, thinking_close, final_start, final_close) < 0:
        raise ValueError("response lacks complete lowercase thinking/final tags")
    thinking_end = thinking_close + len("</thinking>")
    final_end = final_close + len("</final>")
    if not (thinking_start == 0 and thinking_end <= final_start < final_end):
        raise ValueError("response tags are out of order")
    return {
        "thinking": (thinking_start, thinking_end),
        "final": (final_start, final_end),
    }


def interval_mean(
    logps: torch.Tensor,
    offsets: list[tuple[int, int]],
    interval: tuple[int, int],
) -> tuple[float, int]:
    start, end = interval
    indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]
    if not indices:
        raise ValueError(f"response segment contains no tokens: {interval}")
    values = logps[indices]
    return float(values.mean().item()), int(values.numel())


def response_statistics(
    model: Any, tokenizer: Any, row: dict[str, Any], response: str, cutoff_len: int
) -> dict[str, Any]:
    logps, offsets, total_tokens = response_logps(
        model, tokenizer, row, response, cutoff_len
    )
    segments = tagged_segments(response)
    thinking_logp, thinking_tokens = interval_mean(logps, offsets, segments["thinking"])
    final_logp, final_tokens = interval_mean(logps, offsets, segments["final"])
    return {
        "mean_logp": float(logps.mean().item()),
        "nll": float(-logps.mean().item()),
        "response_tokens": int(logps.numel()),
        "total_tokens": total_tokens,
        "thinking_logp": thinking_logp,
        "thinking_nll": -thinking_logp,
        "thinking_tokens": thinking_tokens,
        "final_logp": final_logp,
        "final_nll": -final_logp,
        "final_tokens": final_tokens,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_margin_rows(rows: list[dict[str, Any]], beta: float) -> dict[str, Any]:
    margins = [float(row["margin"]) for row in rows]
    by_category: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_category[str(row["error_category"])].append(float(row["margin"]))
    return {
        "pairs": len(rows),
        "mean_margin": mean(margins),
        "mean_reward_margin": beta * mean(margins),
        "median_margin": statistics.median(margins) if margins else 0.0,
        "preference_accuracy": mean([float(value > 0) for value in margins]),
        "category_mean_margins": {
            category: mean(values) for category, values in sorted(by_category.items())
        },
    }


def score_preferences(
    model: Any,
    tokenizer: Any,
    path: Path,
    cutoff_len: int,
    beta: float,
) -> dict[str, Any]:
    rows = []
    source_rows = load_jsonl(path)
    for index, row in enumerate(source_rows, start=1):
        chosen = response_statistics(
            model, tokenizer, row, row["chosen"], cutoff_len
        )
        rejected = response_statistics(
            model, tokenizer, row, row["rejected"], cutoff_len
        )
        margin = chosen["mean_logp"] - rejected["mean_logp"]
        rows.append(
            {
                "wnd_id": row["meta"]["wnd_id"],
                "document_id": row["meta"]["wnd_id"].rsplit("-", 1)[0],
                "error_category": row["meta"]["error_category"],
                "chosen_logp": chosen["mean_logp"],
                "rejected_logp": rejected["mean_logp"],
                "margin": margin,
                "reward_margin": beta * margin,
                "chosen_response_tokens": chosen["response_tokens"],
                "rejected_response_tokens": rejected["response_tokens"],
                "chosen_total_tokens": chosen["total_tokens"],
                "rejected_total_tokens": rejected["total_tokens"],
            }
        )
        if index % 10 == 0 or index == len(source_rows):
            print(f"scored preference pairs: {index}/{len(source_rows)}", flush=True)
    summary = aggregate_margin_rows(rows, beta)
    summary.update({"beta": beta, "preference_jsonl": str(path), "rows": rows})
    return summary


def score_style(
    model: Any, tokenizer: Any, path: Path, cutoff_len: int
) -> dict[str, Any]:
    results = []
    source_rows = load_jsonl(path)
    for index, row in enumerate(source_rows, start=1):
        canonical = response_statistics(
            model, tokenizer, row, row["canonical"], cutoff_len
        )
        native = response_statistics(model, tokenizer, row, row["native"], cutoff_len)
        results.append(
            {
                "wnd_id": row["meta"]["wnd_id"],
                "document_id": row["meta"]["document_id"],
                "error_category": row["meta"]["error_category"],
                "canonical": canonical,
                "native": native,
                "canonical_minus_native_nll": canonical["nll"] - native["nll"],
                "thinking_nll_gap": canonical["thinking_nll"]
                - native["thinking_nll"],
                "final_nll_gap": canonical["final_nll"] - native["final_nll"],
            }
        )
        if index % 10 == 0 or index == len(source_rows):
            print(f"scored canonical/native pairs: {index}/{len(source_rows)}", flush=True)

    gap_fields = [
        "canonical_minus_native_nll",
        "thinking_nll_gap",
        "final_nll_gap",
    ]
    summary = {
        "pairs": len(results),
        "style_jsonl": str(path),
        **{
            f"mean_{field}": mean([float(row[field]) for row in results])
            for field in gap_fields
        },
        "mean_canonical_nll": mean(
            [float(row["canonical"]["nll"]) for row in results]
        ),
        "mean_native_nll": mean([float(row["native"]["nll"]) for row in results]),
        "rows": results,
    }
    if not all(
        math.isfinite(value)
        for key, value in summary.items()
        if key.startswith("mean_") and isinstance(value, float)
    ):
        raise ValueError("non-finite canonical/native NLL summary")
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--preference_jsonl", type=Path)
    parser.add_argument("--margin_output", type=Path)
    parser.add_argument("--style_jsonl", type=Path)
    parser.add_argument("--style_output", type=Path)
    parser.add_argument("--cutoff_len", type=int, default=1536)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()
    if bool(args.preference_jsonl) != bool(args.margin_output):
        parser.error("--preference_jsonl and --margin_output must be provided together")
    if bool(args.style_jsonl) != bool(args.style_output):
        parser.error("--style_jsonl and --style_output must be provided together")
    if not args.preference_jsonl and not args.style_jsonl:
        parser.error("at least one scoring task is required")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    if args.preference_jsonl:
        margin = score_preferences(
            model,
            tokenizer,
            args.preference_jsonl,
            args.cutoff_len,
            args.beta,
        )
        margin.update({"model_path": args.model_path, "cutoff_len": args.cutoff_len})
        write_json(args.margin_output, margin)
        print(
            json.dumps({key: value for key, value in margin.items() if key != "rows"}, indent=2)
        )
    if args.style_jsonl:
        style = score_style(model, tokenizer, args.style_jsonl, args.cutoff_len)
        style.update({"model_path": args.model_path, "cutoff_len": args.cutoff_len})
        write_json(args.style_output, style)
        print(
            json.dumps({key: value for key, value in style.items() if key != "rows"}, indent=2)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
