import argparse
import json
import random
import re
import sys
import time
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
    batched,
    build_prompt,
    has_tokenizer_assets,
    is_lora_checkpoint,
    load_jsonl,
    merge_metric_dict,
    normalize_events,
    prf,
)


ROUTE_RE = re.compile(r"<ROUTE>\s*(direct|reason)\s*</ROUTE>", re.IGNORECASE)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def extract_tag(text: str, tag: str):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    if start == -1:
        return None
    content_start = start + len(start_tag)
    end = text.find(end_tag, content_start)
    if end == -1:
        return text[content_start:].strip()
    return text[content_start:end].strip()


def extract_route(text: str):
    match = ROUTE_RE.search(text)
    if not match:
        return "unknown"
    return match.group(1).lower()


def extract_final_json(text: str):
    final_text = extract_tag(text, "FINAL")
    if final_text is None:
        return None
    return extract_json(final_text)


def prediction_key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or meta.get("doc_id") or f"index-{row.get('index')}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--template_family", default="qwen")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--route_mode", choices=["direct", "reason"], required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.eval_jsonl))

    candidate_path = Path(args.adapter_path)
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"adapter_path does not exist: {candidate_path}. "
            "Refusing to fall back to base_model for sampled adaptive route evaluation."
        )
    tokenizer_source = args.base_model
    if candidate_path.exists() and has_tokenizer_assets(candidate_path):
        tokenizer_source = candidate_path.as_posix()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if candidate_path.exists() and is_lora_checkpoint(candidate_path):
        if PeftModel is None:
            raise ImportError("peft is required to evaluate LoRA checkpoints, but it is not installed.")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, candidate_path.as_posix())
    else:
        model = AutoModelForCausalLM.from_pretrained(
            candidate_path.as_posix(),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()

    predictions = []
    trigger_metrics = []
    argument_metrics = []
    event_metrics = []
    valid_final_json = 0
    route_counts = {"direct": 0, "reason": 0, "unknown": 0}
    total_time = 0.0

    prepared_rows = []
    for idx, row in enumerate(rows):
        prepared = {
            "index": idx,
            "instruction": row["instruction"],
            "input": row["input"],
            "output": row["output"],
            "gold_output": row.get("gold_output", row["output"]),
            "response_prefix": row.get("response_prefix", ""),
            "prompt": build_prompt(tokenizer, row["instruction"], row["input"]),
            "meta": row.get("meta", {}),
        }
        prepared["sample_key"] = prediction_key(prepared)
        prepared_rows.append(prepared)

    do_sample = args.temperature > 0
    for batch_rows in batched(prepared_rows, args.batch_size):
        prompts = [row["prompt"] for row in batch_rows]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_width = inputs["input_ids"].shape[1]
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update({"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k})

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - start
        total_time += elapsed

        for batch_idx, row in enumerate(batch_rows):
            generated_ids = outputs[batch_idx][prompt_width:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            response_prefix = row.get("response_prefix", "")
            generated_payload = f"{response_prefix}{generated_text}" if response_prefix else generated_text
            route_pred = extract_route(generated_payload)
            route_counts[route_pred] = route_counts.get(route_pred, 0) + 1
            pred_json = extract_final_json(generated_payload)
            gold_json = json.loads(row["gold_output"])

            if pred_json is not None:
                valid_final_json += 1

            pred_trig, pred_arg, pred_event = normalize_events(pred_json or {"events": []})
            gold_trig, gold_arg, gold_event = normalize_events(gold_json)

            trig_metric = prf(pred_trig, gold_trig)
            arg_metric = prf(pred_arg, gold_arg)
            event_metric = prf(pred_event, gold_event)
            trigger_metrics.append(trig_metric)
            argument_metrics.append(arg_metric)
            event_metrics.append(event_metric)

            predictions.append(
                {
                    "instruction": row["instruction"],
                    "input": row["input"],
                    "gold": gold_json,
                    "generated_text": generated_text,
                    "generated_payload": generated_payload,
                    "route_pred": route_pred,
                    "reason_used": route_pred == "reason" or "<REASON>" in generated_payload or "<PLAN>" in generated_payload,
                    "final_predicted": pred_json,
                    "predicted": pred_json,
                    "valid_final_json": pred_json is not None,
                    "valid_json": pred_json is not None,
                    "response_prefix": response_prefix,
                    "latency_sec": elapsed / len(batch_rows),
                    "batch_size": len(batch_rows),
                    "trigger_f1": trig_metric["f1"],
                    "argument_f1": arg_metric["f1"],
                    "event_f1": event_metric["f1"],
                    "meta": row.get("meta", {}),
                    "sample_key": row["sample_key"],
                    "sample_id": args.sample_id,
                    "sample_seed": args.seed,
                    "sample_route_mode": args.route_mode,
                    "sampling_temperature": args.temperature,
                    "sampling_top_p": args.top_p,
                    "sampling_top_k": args.top_k,
                }
            )

    total = len(rows)
    summary = {
        "num_examples": total,
        "sample_id": args.sample_id,
        "sample_seed": args.seed,
        "sample_route_mode": args.route_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "eval_jsonl": args.eval_jsonl,
        "adapter_path": args.adapter_path,
        "final_json_valid_rate": valid_final_json / total if total else 0.0,
        "json_valid_rate": valid_final_json / total if total else 0.0,
        "avg_latency_sec": total_time / total if total else 0.0,
        "route_direct_count": route_counts.get("direct", 0),
        "route_reason_count": route_counts.get("reason", 0),
        "route_unknown_count": route_counts.get("unknown", 0),
        "route_direct_rate": route_counts.get("direct", 0) / total if total else 0.0,
        "route_reason_rate": route_counts.get("reason", 0) / total if total else 0.0,
        "route_unknown_rate": route_counts.get("unknown", 0) / total if total else 0.0,
        "final_trigger_f1": merge_metric_dict(trigger_metrics, "f1"),
        "final_argument_f1": merge_metric_dict(argument_metrics, "f1"),
        "final_event_f1": merge_metric_dict(event_metrics, "f1"),
        "trigger_f1": merge_metric_dict(trigger_metrics, "f1"),
        "argument_f1": merge_metric_dict(argument_metrics, "f1"),
        "event_f1": merge_metric_dict(event_metrics, "f1"),
    }

    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
