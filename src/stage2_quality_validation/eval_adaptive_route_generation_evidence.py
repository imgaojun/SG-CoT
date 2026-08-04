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
    merge_metric_dict,
    normalize_events,
    prf,
)
from src.stage2_quality_validation.eval_adaptive_route_generation import (  # noqa: E402
    extract_final_json,
    extract_reasoning_budget,
    extract_route,
)
from src.stage2_preference.reasoning_preference import (  # noqa: E402
    recover_offsets_from_evidence as shared_recover_offsets_from_evidence,
)
from src.stage2_quality_validation.generation_diagnostics import (  # noqa: E402
    completion_token_diagnostics,
    output_contract_diagnostics,
)


def norm_token(token: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "", (token or "").lower())
    return cleaned


def phrase_tokens(text: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", text or "")
    return [norm_token(tok) for tok in raw if norm_token(tok)]


def parse_prompt_tokens(input_text: str) -> list[str]:
    match = re.search(r"Tokens:\n(.*?)(?:\n\nCandidate event types:|\n\nSchema cards:|\Z)", input_text, flags=re.S)
    if not match:
        return []
    return [tok for tok in match.group(1).strip().split() if tok]


def find_subseqs(haystack: list[str], needle: list[str]) -> list[tuple[int, int]]:
    if not haystack or not needle or len(needle) > len(haystack):
        return []
    out = []
    n = len(needle)
    for idx in range(0, len(haystack) - n + 1):
        if haystack[idx : idx + n] == needle:
            out.append((idx, idx + n))
    return out


def overlap_or_inside(inner: tuple[int, int], outer: tuple[int, int]) -> bool:
    return inner[0] >= outer[0] and inner[1] <= outer[1]


def choose_surface_span(tokens: list[str], surface: str, evidence: str) -> tuple[tuple[int | None, int | None], dict]:
    diagnostics = {
        "surface": surface,
        "evidence": evidence,
        "surface_candidates": 0,
        "evidence_candidates": 0,
        "method": "none",
    }
    norm_tokens = [norm_token(tok) for tok in tokens]
    surface_seq = phrase_tokens(surface)
    evidence_seq = phrase_tokens(evidence)
    surface_spans = find_subseqs(norm_tokens, surface_seq)
    evidence_spans = find_subseqs(norm_tokens, evidence_seq)
    diagnostics["surface_candidates"] = len(surface_spans)
    diagnostics["evidence_candidates"] = len(evidence_spans)
    if not surface_spans:
        return (None, None), diagnostics
    for surface_span in surface_spans:
        if any(overlap_or_inside(surface_span, evidence_span) for evidence_span in evidence_spans):
            diagnostics["method"] = "surface_inside_evidence"
            return surface_span, diagnostics
    if evidence_spans:
        evidence_mid = sum(evidence_spans[0]) / 2
        chosen = min(surface_spans, key=lambda span: abs((span[0] + span[1]) / 2 - evidence_mid))
        diagnostics["method"] = "nearest_surface_to_evidence"
        return chosen, diagnostics
    if len(surface_spans) == 1:
        diagnostics["method"] = "unique_surface"
        return surface_spans[0], diagnostics
    diagnostics["method"] = "first_surface_fallback"
    return surface_spans[0], diagnostics


def recover_offsets_from_evidence(surface_payload: dict, input_text: str) -> tuple[dict, dict]:
    return shared_recover_offsets_from_evidence(surface_payload, input_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--template_family", default="qwen")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.eval_jsonl))

    candidate_path = Path(args.adapter_path)
    if not candidate_path.exists():
        raise FileNotFoundError(f"adapter_path does not exist: {candidate_path}")
    tokenizer_source = args.base_model
    if candidate_path.exists() and has_tokenizer_assets(candidate_path):
        tokenizer_source = candidate_path.as_posix()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if candidate_path.exists() and is_lora_checkpoint(candidate_path):
        if PeftModel is None:
            raise ImportError("peft is required to evaluate LoRA checkpoints.")
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
    recovered_without_missing = 0
    route_counts = {"direct": 0, "reason": 0, "unknown": 0}
    generated_token_counts = []
    max_new_token_hits = 0
    final_tag_complete = 0
    reasoning_tag_expected = 0
    reasoning_tag_complete = 0
    surface_event_list_valid = 0
    candidate_types_valid = 0
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
                "meta": row.get("meta", {}),
            }
        )

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
            generation_diagnostics = completion_token_diagnostics(
                generated_ids.tolist(),
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
            )
            generated_token_counts.append(generation_diagnostics["generated_token_count"])
            max_new_token_hits += int(generation_diagnostics["hit_max_new_tokens"])
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            response_prefix = row.get("response_prefix", "")
            generated_payload = f"{response_prefix}{generated_text}" if response_prefix else generated_text
            meta = row.get("meta", {})
            route_pred = extract_route(generated_payload)
            reasoning_budget_pred = extract_reasoning_budget(generated_payload)
            route_counts[route_pred] = route_counts.get(route_pred, 0) + 1
            surface_pred_json = extract_final_json(generated_payload)
            expects_reasoning = meta.get("surface_mode") == "sgcot"
            contract_diagnostics = output_contract_diagnostics(
                generated_payload,
                surface_pred_json,
                candidate_types=meta.get("candidate_types", []),
                expects_reasoning=expects_reasoning,
            )
            final_tag_complete += int(contract_diagnostics["final_tag_complete"])
            surface_event_list_valid += int(
                contract_diagnostics["surface_event_list_valid"]
            )
            candidate_types_valid += int(contract_diagnostics["candidate_types_valid"])
            if expects_reasoning:
                reasoning_tag_expected += 1
                reasoning_tag_complete += int(
                    contract_diagnostics["reasoning_tag_complete"]
                )
            gold_json = json.loads(row["gold_output"])

            if surface_pred_json is not None:
                valid_final_json += 1
            recovered_json, recovery_diagnostics = recover_offsets_from_evidence(surface_pred_json or {"events": []}, row["input"])
            if recovery_diagnostics.get("missing_offsets") == 0 and surface_pred_json is not None:
                recovered_without_missing += 1

            pred_trig, pred_arg, pred_event = normalize_events(recovered_json or {"events": []})
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
                    **generation_diagnostics,
                    **contract_diagnostics,
                    "route_pred": route_pred,
                    "reasoning_budget_pred": reasoning_budget_pred,
                    "surface_final_predicted": surface_pred_json,
                    "final_predicted": recovered_json,
                    "predicted": recovered_json,
                    "recovery_diagnostics": recovery_diagnostics,
                    "valid_final_json": surface_pred_json is not None,
                    "valid_json": surface_pred_json is not None,
                    "response_prefix": response_prefix,
                    "latency_sec": elapsed / len(batch_rows),
                    "batch_size": len(batch_rows),
                    "trigger_f1": trig_metric["f1"],
                    "argument_f1": arg_metric["f1"],
                    "event_f1": event_metric["f1"],
                    "meta": row.get("meta", {}),
                }
            )

    total = len(rows)
    sorted_generated_token_counts = sorted(generated_token_counts)
    generated_token_p95 = (
        sorted_generated_token_counts[round(0.95 * (total - 1))] if total else 0
    )
    summary = {
        "num_examples": total,
        "max_new_tokens": args.max_new_tokens,
        "generated_token_count_mean": sum(generated_token_counts) / total if total else 0.0,
        "generated_token_count_p95": generated_token_p95,
        "generated_token_count_max": max(generated_token_counts, default=0),
        "hit_max_new_tokens_count": max_new_token_hits,
        "hit_max_new_tokens_rate": max_new_token_hits / total if total else 0.0,
        "final_tag_complete_rate": final_tag_complete / total if total else 0.0,
        "reasoning_tag_expected_count": reasoning_tag_expected,
        "reasoning_tag_complete_count": reasoning_tag_complete,
        "reasoning_tag_complete_rate": (
            reasoning_tag_complete / reasoning_tag_expected
            if reasoning_tag_expected
            else None
        ),
        "surface_event_list_valid_rate": surface_event_list_valid / total if total else 0.0,
        "candidate_type_valid_rate": candidate_types_valid / total if total else 0.0,
        "final_json_valid_rate": valid_final_json / total if total else 0.0,
        "json_valid_rate": valid_final_json / total if total else 0.0,
        "offset_recovery_full_rate": recovered_without_missing / total if total else 0.0,
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
