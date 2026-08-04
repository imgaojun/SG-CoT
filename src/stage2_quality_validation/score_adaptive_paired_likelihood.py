import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_quality_validation.eval_adapter_generation import (  # noqa: E402
    build_prompt,
    has_tokenizer_assets,
    is_lora_checkpoint,
    load_jsonl,
)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row, idx):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or meta.get("doc_id") or str(idx)


def split_final_target(output: str):
    start_tag = "<FINAL>"
    end_tag = "</FINAL>"
    start = output.find(start_tag)
    end = output.find(end_tag, start + len(start_tag))
    if start == -1 or end == -1:
        raise ValueError("target does not contain complete <FINAL>...</FINAL>")
    prefix = output[: start + len(start_tag)]
    final_text = output[start + len(start_tag) : end]
    suffix = output[end:]
    return prefix, final_text, suffix


def split_plan_target(output: str):
    for tag in ["PLAN", "REASON"]:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = output.find(start_tag)
        end = output.find(end_tag, start + len(start_tag))
        if start != -1 and end != -1:
            prefix = output[: start + len(start_tag)]
            content = output[start + len(start_tag) : end]
            return prefix, content
    return None, None


def load_model(base_model: str, adapter_path: str):
    candidate_path = Path(adapter_path)
    tokenizer_source = base_model
    if candidate_path.exists() and has_tokenizer_assets(candidate_path):
        tokenizer_source = candidate_path.as_posix()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if candidate_path.exists() and is_lora_checkpoint(candidate_path):
        if PeftModel is None:
            raise ImportError("peft is required to load LoRA checkpoints.")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, candidate_path.as_posix())
    else:
        model_source = candidate_path.as_posix() if candidate_path.exists() else base_model
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    return tokenizer, model


def token_count(tokenizer, text: str):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def mean_nll_for_span(tokenizer, model, context_text: str, target_text: str, max_length: int):
    if target_text == "":
        return {"mean_nll": None, "sum_nll": 0.0, "num_tokens": 0}
    context_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return {"mean_nll": None, "sum_nll": 0.0, "num_tokens": 0}
    input_ids = context_ids + target_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        if overflow >= len(context_ids):
            raise ValueError(
                f"target span cannot fit max_length={max_length}; target_tokens={len(target_ids)}"
            )
        context_ids = context_ids[overflow:]
        input_ids = context_ids + target_ids
    target_start = len(context_ids)
    tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=tensor).logits
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        next_tokens = tensor[:, 1:]
        token_log_probs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
        positions = torch.arange(1, tensor.shape[1], device=model.device)
        mask = positions >= target_start
        selected = token_log_probs[0, mask]
        sum_nll = float((-selected).sum().item())
        num_tokens = int(selected.numel())
    return {
        "mean_nll": sum_nll / num_tokens if num_tokens else None,
        "sum_nll": sum_nll,
        "num_tokens": num_tokens,
    }


def route_row_map(rows):
    by_key = {}
    for idx, row in enumerate(rows):
        meta = row.get("meta") or {}
        route = meta.get("adaptive_route_label")
        key = row_key(row, idx)
        if route not in {"direct", "reason"}:
            continue
        by_key.setdefault(key, {})[route] = row
    return by_key


def paired_map_from_files(paired_jsonl: str | None, direct_jsonl: str | None, reason_jsonl: str | None):
    if paired_jsonl:
        return route_row_map(load_jsonl(Path(paired_jsonl)))
    if not direct_jsonl or not reason_jsonl:
        raise ValueError("Provide either --paired_jsonl or both --direct_jsonl and --reason_jsonl.")
    direct_rows = load_jsonl(Path(direct_jsonl))
    reason_rows = load_jsonl(Path(reason_jsonl))
    by_key = {}
    for idx, row in enumerate(direct_rows):
        by_key.setdefault(row_key(row, idx), {})["direct"] = row
    for idx, row in enumerate(reason_rows):
        by_key.setdefault(row_key(row, idx), {})["reason"] = row
    return by_key


def score_pair(tokenizer, model, direct_row, reason_row, max_length: int):
    direct_prompt = build_prompt(tokenizer, direct_row["instruction"], direct_row["input"])
    reason_prompt = build_prompt(tokenizer, reason_row["instruction"], reason_row["input"])
    direct_prefix, direct_final, _ = split_final_target(direct_row["output"])
    reason_prefix, reason_final, _ = split_final_target(reason_row["output"])
    direct_score = mean_nll_for_span(tokenizer, model, direct_prompt + direct_prefix, direct_final, max_length)
    reason_score = mean_nll_for_span(tokenizer, model, reason_prompt + reason_prefix, reason_final, max_length)

    plan_prefix, plan_text = split_plan_target(reason_row["output"])
    plan_score = {"mean_nll": None, "sum_nll": 0.0, "num_tokens": 0}
    if plan_prefix is not None and plan_text is not None:
        plan_score = mean_nll_for_span(tokenizer, model, reason_prompt + plan_prefix, plan_text, max_length)

    return {
        "nll_direct_final": direct_score["mean_nll"],
        "nll_reason_final": reason_score["mean_nll"],
        "delta_final_nll": (
            direct_score["mean_nll"] - reason_score["mean_nll"]
            if direct_score["mean_nll"] is not None and reason_score["mean_nll"] is not None
            else None
        ),
        "sum_nll_direct_final": direct_score["sum_nll"],
        "sum_nll_reason_final": reason_score["sum_nll"],
        "num_direct_final_tokens": direct_score["num_tokens"],
        "num_reason_final_tokens": reason_score["num_tokens"],
        "nll_reason_plan": plan_score["mean_nll"],
        "sum_nll_reason_plan": plan_score["sum_nll"],
        "num_reason_plan_tokens": plan_score["num_tokens"],
        "direct_prompt_tokens": token_count(tokenizer, direct_prompt + direct_prefix),
        "reason_prompt_tokens": token_count(tokenizer, reason_prompt + reason_prefix),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--paired_jsonl", default=None)
    parser.add_argument("--direct_jsonl", default=None)
    parser.add_argument("--reason_jsonl", default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pairs = paired_map_from_files(args.paired_jsonl, args.direct_jsonl, args.reason_jsonl)
    keys = sorted(key for key, routes in pairs.items() if {"direct", "reason"} <= set(routes))
    if args.limit is not None:
        keys = keys[: args.limit]
    tokenizer, model = load_model(args.base_model, args.adapter_path)

    out_rows = []
    skipped = []
    for key in keys:
        direct_row = pairs[key]["direct"]
        reason_row = pairs[key]["reason"]
        try:
            score_payload = score_pair(tokenizer, model, direct_row, reason_row, args.max_length)
        except Exception as exc:
            skipped.append({"wnd_id": key, "error": str(exc)})
            continue
        meta = direct_row.get("meta") or {}
        out_rows.append(
            {
                "wnd_id": key,
                "split": args.split,
                "score_model": args.adapter_path,
                "base_model": args.base_model,
                "source_jsonl": args.paired_jsonl,
                "direct_jsonl": args.direct_jsonl,
                "reason_jsonl": args.reason_jsonl,
                "gold_event_types": meta.get("gold_event_types"),
                "candidate_types": meta.get("candidate_types"),
                **score_payload,
            }
        )

    write_jsonl(Path(args.output_jsonl), out_rows)
    valid_deltas = [row["delta_final_nll"] for row in out_rows if row["delta_final_nll"] is not None]
    summary = {
        "paired_jsonl": args.paired_jsonl,
        "direct_jsonl": args.direct_jsonl,
        "reason_jsonl": args.reason_jsonl,
        "output_jsonl": args.output_jsonl,
        "split": args.split,
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "num_pairs": len(keys),
        "num_scored": len(out_rows),
        "num_skipped": len(skipped),
        "avg_delta_final_nll": sum(valid_deltas) / len(valid_deltas) if valid_deltas else None,
        "positive_delta_count": sum(1 for value in valid_deltas if value > 0),
        "positive_delta_rate": sum(1 for value in valid_deltas if value > 0) / len(valid_deltas) if valid_deltas else 0.0,
        "skipped": skipped[:20],
    }
    write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
