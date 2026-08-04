import argparse
import json
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
)


ROUTE_RE = re.compile(r"<ROUTE>\s*(direct|reason)\s*</ROUTE>", re.IGNORECASE)


def extract_route(text: str):
    match = ROUTE_RE.search(text)
    if not match:
        return "unknown"
    return match.group(1).lower()


def gold_route(row):
    meta = row.get("meta") or {}
    label = meta.get("adaptive_route_label")
    if label in {"direct", "reason"}:
        return label
    text = row.get("output", "")
    return extract_route(text)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--template_family", default="qwen")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
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
            raise ImportError("peft is required to evaluate LoRA checkpoints, but it is not installed.")
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

    prepared_rows = []
    for idx, row in enumerate(rows):
        prepared_rows.append(
            {
                "index": idx,
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
                "gold_route": gold_route(row),
                "prompt": build_prompt(tokenizer, row["instruction"], row["input"]),
                "meta": row.get("meta", {}),
            }
        )

    predictions = []
    total_time = 0.0
    counts = {
        "gold_reason": 0,
        "gold_direct": 0,
        "pred_reason": 0,
        "pred_direct": 0,
        "pred_unknown": 0,
        "correct": 0,
        "tp_reason": 0,
        "fp_reason": 0,
        "fn_reason": 0,
    }

    for batch_rows in batched(prepared_rows, args.batch_size):
        prompts = [row["prompt"] for row in batch_rows]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
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
            pred_route = extract_route(generated_text)
            gold = row["gold_route"]
            counts[f"gold_{gold}"] = counts.get(f"gold_{gold}", 0) + 1
            counts[f"pred_{pred_route}"] = counts.get(f"pred_{pred_route}", 0) + 1
            if pred_route == gold:
                counts["correct"] += 1
            if pred_route == "reason" and gold == "reason":
                counts["tp_reason"] += 1
            elif pred_route == "reason" and gold != "reason":
                counts["fp_reason"] += 1
            elif pred_route != "reason" and gold == "reason":
                counts["fn_reason"] += 1
            predictions.append(
                {
                    "instruction": row["instruction"],
                    "input": row["input"],
                    "gold_route": gold,
                    "generated_text": generated_text,
                    "route_pred": pred_route,
                    "route_correct": pred_route == gold,
                    "latency_sec": elapsed / len(batch_rows),
                    "batch_size": len(batch_rows),
                    "meta": row.get("meta", {}),
                }
            )

    total = len(prepared_rows)
    precision, recall, f1 = prf(counts["tp_reason"], counts["fp_reason"], counts["fn_reason"])
    summary = {
        "num_examples": total,
        "route_accuracy": counts["correct"] / total if total else 0.0,
        "label_reason_count": counts.get("gold_reason", 0),
        "label_direct_count": counts.get("gold_direct", 0),
        "label_reason_rate": counts.get("gold_reason", 0) / total if total else 0.0,
        "pred_reason_count": counts.get("pred_reason", 0),
        "pred_direct_count": counts.get("pred_direct", 0),
        "pred_unknown_count": counts.get("pred_unknown", 0),
        "pred_reason_rate": counts.get("pred_reason", 0) / total if total else 0.0,
        "reason_precision": precision,
        "reason_recall": recall,
        "reason_f1": f1,
        "avg_latency_sec": total_time / total if total else 0.0,
    }

    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
