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
from src.stage2_quality_validation.eval_adaptive_route_choice import gold_route, prf  # noqa: E402


ROUTE_TARGETS = {
    "direct": "<ROUTE>direct</ROUTE>",
    "reason": "<ROUTE>reason</ROUTE>",
}


def load_model(base_model: str, adapter_path: str):
    candidate_path = Path(adapter_path)
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"adapter_path does not exist: {candidate_path}. "
            "Refusing to fall back to base_model for route likelihood scoring."
        )
    tokenizer_source = base_model
    if has_tokenizer_assets(candidate_path):
        tokenizer_source = candidate_path.as_posix()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_lora_checkpoint(candidate_path):
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
        model_source = candidate_path.as_posix()
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    return tokenizer, model


def mean_nll_for_continuation(tokenizer, model, prompt: str, continuation: str, max_length: int):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + continuation_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        if overflow >= len(prompt_ids):
            raise ValueError(
                f"continuation cannot fit max_length={max_length}; continuation_tokens={len(continuation_ids)}"
            )
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + continuation_ids
    target_start = len(prompt_ids)
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


def best_threshold(rows, key: str):
    # Predict reason when score >= threshold.
    candidates = sorted({row[key] for row in rows if row.get(key) is not None})
    if not candidates:
        return {"threshold": None, "reason_precision": 0.0, "reason_recall": 0.0, "reason_f1": 0.0}
    thresholds = [candidates[0] - 1e-9] + candidates + [candidates[-1] + 1e-9]
    best = None
    for threshold in thresholds:
        tp = fp = fn = correct = pred_reason = 0
        for row in rows:
            pred = "reason" if row[key] >= threshold else "direct"
            gold = row["gold_route"]
            pred_reason += int(pred == "reason")
            correct += int(pred == gold)
            if pred == "reason" and gold == "reason":
                tp += 1
            elif pred == "reason" and gold != "reason":
                fp += 1
            elif pred != "reason" and gold == "reason":
                fn += 1
        precision, recall, f1 = prf(tp, fp, fn)
        item = {
            "threshold": threshold,
            "route_accuracy": correct / len(rows) if rows else 0.0,
            "pred_reason_rate": pred_reason / len(rows) if rows else 0.0,
            "reason_precision": precision,
            "reason_recall": recall,
            "reason_f1": f1,
        }
        if best is None or (item["reason_f1"], item["reason_recall"], item["route_accuracy"]) > (
            best["reason_f1"],
            best["reason_recall"],
            best["route_accuracy"],
        ):
            best = item
    return best


def auc_score(rows, key: str):
    positives = [row[key] for row in rows if row["gold_route"] == "reason" and row.get(key) is not None]
    negatives = [row[key] for row in rows if row["gold_route"] != "reason" and row.get(key) is not None]
    if not positives or not negatives:
        return None
    wins = ties = 0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1
            elif pos == neg:
                ties += 1
    return (wins + 0.5 * ties) / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.eval_jsonl))
    tokenizer, model = load_model(args.base_model, args.adapter_path)
    out_rows = []
    for idx, row in enumerate(rows):
        prompt = build_prompt(tokenizer, row["instruction"], row["input"])
        direct_score = mean_nll_for_continuation(
            tokenizer, model, prompt, ROUTE_TARGETS["direct"], args.max_length
        )
        reason_score = mean_nll_for_continuation(
            tokenizer, model, prompt, ROUTE_TARGETS["reason"], args.max_length
        )
        direct_nll = direct_score["mean_nll"]
        reason_nll = reason_score["mean_nll"]
        delta = direct_nll - reason_nll if direct_nll is not None and reason_nll is not None else None
        pred = "reason" if delta is not None and delta > 0 else "direct"
        gold = gold_route(row)
        meta = row.get("meta") or {}
        out_rows.append(
            {
                "index": idx,
                "wnd_id": meta.get("wnd_id"),
                "gold_route": gold,
                "pred_route_argmin_nll": pred,
                "route_correct_argmin_nll": pred == gold,
                "nll_direct_route": direct_nll,
                "nll_reason_route": reason_nll,
                "delta_direct_minus_reason_route_nll": delta,
                "direct_route_tokens": direct_score["num_tokens"],
                "reason_route_tokens": reason_score["num_tokens"],
                "meta": meta,
            }
        )

    tp = fp = fn = correct = pred_reason = 0
    for row in out_rows:
        pred = row["pred_route_argmin_nll"]
        gold = row["gold_route"]
        pred_reason += int(pred == "reason")
        correct += int(pred == gold)
        if pred == "reason" and gold == "reason":
            tp += 1
        elif pred == "reason" and gold != "reason":
            fp += 1
        elif pred != "reason" and gold == "reason":
            fn += 1
    precision, recall, f1 = prf(tp, fp, fn)
    summary = {
        "num_examples": len(out_rows),
        "label_reason_count": sum(1 for row in out_rows if row["gold_route"] == "reason"),
        "label_reason_rate": (
            sum(1 for row in out_rows if row["gold_route"] == "reason") / len(out_rows)
            if out_rows
            else 0.0
        ),
        "argmin_route_accuracy": correct / len(out_rows) if out_rows else 0.0,
        "argmin_pred_reason_count": pred_reason,
        "argmin_pred_reason_rate": pred_reason / len(out_rows) if out_rows else 0.0,
        "argmin_reason_precision": precision,
        "argmin_reason_recall": recall,
        "argmin_reason_f1": f1,
        "delta_auc": auc_score(out_rows, "delta_direct_minus_reason_route_nll"),
        "best_threshold": best_threshold(out_rows, "delta_direct_minus_reason_route_nll"),
    }

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
