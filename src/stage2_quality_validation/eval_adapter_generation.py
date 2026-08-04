import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.stage2_quality_validation.event_metrics import normalize_events, prf

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


def has_tokenizer_assets(path: Path):
    return any(
        (path / name).exists()
        for name in [
            "tokenizer_config.json",
            "tokenizer.json",
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json",
        ]
    )


def is_lora_checkpoint(path: Path):
    return any(
        (path / name).exists()
        for name in [
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
        ]
    )


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def merge_metric_dict(metric_list, key):
    vals = [m[key] for m in metric_list]
    return sum(vals) / len(vals) if vals else 0.0


def build_prompt(tokenizer, instruction: str, input_text: str):
    messages = [{"role": "user", "content": f"{instruction}\n{input_text}"}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{instruction}\n{input_text}"


def batched(iterable, batch_size):
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--template_family", default="qwen")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.eval_jsonl))

    candidate_path = Path(args.adapter_path)
    tokenizer_source = args.base_model
    if candidate_path.exists() and has_tokenizer_assets(candidate_path):
        tokenizer_source = candidate_path.as_posix()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if candidate_path.exists() and is_lora_checkpoint(candidate_path):
        if PeftModel is None:
            raise ImportError("peft is required to evaluate LoRA checkpoints, but it is not installed in the current Python environment.")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, candidate_path.as_posix())
    else:
        model_source = candidate_path.as_posix() if candidate_path.exists() else args.base_model
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()

    predictions = []
    trigger_metrics = []
    argument_metrics = []
    event_metrics = []
    valid_json = 0
    total_time = 0.0

    prepared_rows = []
    for idx, row in enumerate(rows):
        prepared_rows.append(
            {
                "index": idx,
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
                "gold_output": row.get("gold_output", row["output"]),
                "response_prefix": row.get("response_prefix", ""),
                "prompt": build_prompt(tokenizer, row["instruction"], row["input"]),
            }
        )

    for batch_rows in batched(prepared_rows, args.batch_size):
        prompts = [row["prompt"] for row in batch_rows]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        # Generated sequences include the full padded input width. With left
        # padding, attention-mask lengths are shorter than that width and would
        # leave prompt tails in decoded generations.
        prompt_width = inputs["input_ids"].shape[1]
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start
        total_time += elapsed

        for batch_idx, row in enumerate(batch_rows):
            generated_ids = outputs[batch_idx][prompt_width:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            response_prefix = row.get("response_prefix", "")
            generated_payload = f"{response_prefix}{generated_text}" if response_prefix else generated_text
            pred_json = extract_json(generated_payload)
            gold_json = json.loads(row["gold_output"])

            if pred_json is not None:
                valid_json += 1

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
                    "predicted": pred_json,
                    "valid_json": pred_json is not None,
                    "response_prefix": response_prefix,
                    "latency_sec": elapsed / len(batch_rows),
                    "batch_size": len(batch_rows),
                    "trigger_f1": trig_metric["f1"],
                    "argument_f1": arg_metric["f1"],
                    "event_f1": event_metric["f1"],
                }
            )

    summary = {
        "num_examples": len(rows),
        "json_valid_rate": valid_json / len(rows) if rows else 0.0,
        "avg_latency_sec": total_time / len(rows) if rows else 0.0,
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
