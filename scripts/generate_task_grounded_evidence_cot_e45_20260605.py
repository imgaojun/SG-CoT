#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.generate_evidence_cot_e40_20260604 as e40  # noqa: E402


BASE_HARD_VERIFY = e40.hard_verify
BASE_MAKE_EVIDENCE_ROW = e40.make_evidence_row
QWEN4_RUN_PREFIX = "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
QWEN4_WARM_START = (
    "/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user/"
    "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full/checkpoint-2064"
)


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def sentence_count(text: str) -> int:
    return len([x for x in re.split(r"[.!?]+\s*", text or "") if x.strip()])


def generator_prompt(row: dict) -> str:
    candidates, schema_cards = e40.extract_schema(row["input"])
    payload = {
        "task": "Generate general task-grounded natural-language CoT supervision for event extraction.",
        "goal": (
            "Normalize broad language-model semantic priors into schema-aligned and annotation-grounded "
            "event extraction decisions. The strategy should be generic across models, not tailored to a specific backbone."
        ),
        "important_positioning": [
            "The provided target events are authoritative for this supervision example.",
            "Do not add, remove, reorder, or modify target event types, trigger text, argument text, or roles.",
            "Do not mention model names, model errors, Direct, E40, E45, or case labels in the output.",
            "Do not write case-specific rules. Use general extraction principles that apply broadly to event extraction.",
        ],
        "required_output": "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only.",
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Start from event meanings expressed in the text, then ground them in the provided schemas and candidate event types.",
            "For each target event, state why the event type fits the schema definition or trigger cues.",
            "Explain trigger boundary control: choose the explicit textual mention that expresses the event, prefer the minimal lexical trigger, and avoid surrounding explanation, diagnosis, report, or context words unless they are themselves the event mention.",
            "Explain argument-role grounding: include only arguments explicitly supported by local evidence and state how each argument participates in the event role.",
            "Explain extraction-style control: avoid duplicate events, avoid semantically plausible but weakly grounded extra events, and avoid unsupported role filling from world knowledge.",
            "End by verifying that every final trigger and argument has short contiguous local evidence from the text.",
        ],
        "thinking_quality_requirements": [
            "The thinking should usually contain 5 to 8 natural sentences.",
            "It must mention selected event type names and trigger texts.",
            "It must mention at least one reason for excluding nearby context or extra candidates when the text contains potentially confusing context.",
            "It must describe why the evidence phrase is local and sufficient.",
        ],
        "final_json_schema": {
            "events": [
                {
                    "event_type": "copy from target",
                    "trigger": {"text": "copy from target", "evidence": "short contiguous local phrase/clause from Text containing trigger text"},
                    "arguments": [
                        {"role": "copy from target", "text": "copy from target", "evidence": "short contiguous local phrase/clause from Text containing argument text"}
                    ],
                }
            ]
        },
        "evidence_rules": [
            "Every evidence string must be an exact contiguous quote from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence should be a local phrase or short clause, not an isolated token when local context is available.",
            "Trigger evidence should show the event mention in context, such as 'she had a broken jaw', 'they met up', or '22 000 ended up in prison'.",
            "Argument evidence should show the argument's local relation to the event when possible, not only the argument token.",
            "Do not use a nearby context clause as evidence if that clause does not contain the corresponding text field.",
        ],
        "constraints": [
            "No numeric offsets, token indices, character positions, markdown, or text outside the two lowercase tags.",
            "Keep thinking natural, concise, and general; aim for 120-280 English words.",
            "Avoid explicitly saying that you are copying gold labels or that only one event is labeled; phrase decisions as target extraction style, schema alignment, and evidence grounding.",
        ],
        "input": {
            "text": e40.extract_text(row["input"]),
            "candidate_event_types": candidates,
            "schema_cards": schema_cards,
            "target_surface_events_to_copy": e40.surface_gold_json(row),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def verifier_prompt(row: dict, thinking: str, final_obj: dict) -> str:
    candidates, schema_cards = e40.extract_schema(row["input"])
    payload = {
        "task": "Strictly verify task-grounded CoT/evidence training data for event extraction.",
        "instruction": "Do not repair the answer. Return strict JSON only.",
        "pass_requirements": [
            "The final surface events exactly match the target surface events.",
            "The reasoning is task-grounded rather than a generic template.",
            "The reasoning explains schema grounding, trigger boundary control, argument-role grounding, and extraction-style control when relevant.",
            "Every evidence string is an exact contiguous quote from Text and contains the corresponding trigger or argument text.",
            "Evidence is locally informative: it should usually be a short phrase or clause showing the event or argument relation, not only an isolated token.",
            "The response does not use numeric offsets or token indices.",
        ],
        "return_contract": {
            "pass": "boolean",
            "scores": {
                "trigger_evidence": "integer 1-5",
                "argument_evidence": "integer 1-5",
                "evidence_informativeness": "integer 1-5",
                "type_discrimination": "integer 1-5",
                "trigger_boundary_control": "integer 1-5",
                "argument_role_grounding": "integer 1-5",
                "extraction_style_control": "integer 1-5",
                "final_structure_consistency": "integer 1-5",
            },
            "errors": ["short error labels, empty if pass"],
            "reason": "one concise sentence",
        },
        "input": {
            "text": e40.extract_text(row["input"]),
            "candidate_event_types": candidates,
            "schema_cards": schema_cards,
            "target_surface_events": e40.surface_gold_json(row),
            "generated_thinking": thinking,
            "generated_final": final_obj,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def evidence_quality_errors(row: dict, thinking: str | None, final_obj: dict | None) -> list[str]:
    errors = []
    if not thinking:
        return errors
    wc = len(thinking.split())
    if wc < 90:
        errors.append(f"thinking_too_short_words:{wc}")
    if sentence_count(thinking) < 4:
        errors.append(f"thinking_too_few_sentences:{sentence_count(thinking)}")
    if final_obj is None:
        return errors
    source_n = norm_text(e40.extract_text(row["input"]))
    for event_i, event in enumerate(final_obj.get("events", []) or []):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        trigger_text = trigger.get("text") or ""
        trigger_evidence = trigger.get("evidence") or ""
        trigger_words = norm_text(trigger_text).split()
        evidence_words = norm_text(trigger_evidence).split()
        if norm_text(trigger_evidence) not in source_n:
            errors.append(f"event_{event_i}_trigger_evidence_not_in_text")
        if norm_text(trigger_text) not in norm_text(trigger_evidence):
            errors.append(f"event_{event_i}_trigger_evidence_missing_text")
        if len(trigger_words) == 1 and len(evidence_words) <= 1:
            errors.append(f"event_{event_i}_trigger_evidence_too_short")
        if len(evidence_words) > 22:
            errors.append(f"event_{event_i}_trigger_evidence_too_long")
        for arg_i, arg in enumerate(event.get("arguments", []) or []):
            if not isinstance(arg, dict):
                continue
            arg_text = arg.get("text") or ""
            arg_evidence = arg.get("evidence") or ""
            arg_words = norm_text(arg_text).split()
            arg_ev_words = norm_text(arg_evidence).split()
            if norm_text(arg_evidence) not in source_n:
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_not_in_text")
            if norm_text(arg_text) not in norm_text(arg_evidence):
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_missing_text")
            if len(arg_words) == 1 and len(arg_ev_words) <= 1:
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_too_short")
            if len(arg_ev_words) > 24:
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_too_long")
    return errors


def hard_verify(row: dict, content: str):
    thinking, final_obj, errors = BASE_HARD_VERIFY(row, content)
    if content.strip() and not content.strip().lower().startswith("<thinking>"):
        errors.append("text_before_thinking")
    if content.strip() and not content.strip().lower().endswith("</final>"):
        errors.append("text_after_final")
    if re.search(r"<\s*/?\s*think\s*>", content or "", re.I):
        errors.append("malformed_think_tag")
    errors.extend(evidence_quality_errors(row, thinking, final_obj))
    return thinking, final_obj, errors


def semantic_pass(verifier_obj: dict):
    errors = []
    if not isinstance(verifier_obj, dict):
        return False, ["verifier_not_object"]
    if verifier_obj.get("pass") is not True:
        errors.append("semantic_pass_false")
    scores = verifier_obj.get("scores") if isinstance(verifier_obj.get("scores"), dict) else {}
    for key in [
        "trigger_evidence",
        "argument_evidence",
        "evidence_informativeness",
        "type_discrimination",
        "trigger_boundary_control",
        "argument_role_grounding",
        "extraction_style_control",
    ]:
        try:
            val = int(scores.get(key))
        except Exception:
            val = 0
        if val < 4:
            errors.append(f"low_{key}:{val}")
    try:
        final_consistency = int(scores.get("final_structure_consistency"))
    except Exception:
        final_consistency = 0
    if final_consistency < 5:
        errors.append(f"low_final_structure_consistency:{final_consistency}")
    verifier_errors = verifier_obj.get("errors")
    if isinstance(verifier_errors, list) and verifier_errors:
        errors.append("semantic_errors:" + ",".join(map(str, verifier_errors[:5])))
    return not errors, errors


def process_one(row: dict, args, api_key: str) -> dict:
    sample_id = row["meta"]["e40_sample_id"]
    rec = {"sample_id": sample_id, "source_index": row["meta"].get("e40_source_index"), "accepted": False, "attempts": []}
    messages = [
        {
            "role": "system",
            "content": (
                "You create faithful, general, task-grounded CoT supervision for event extraction. "
                "Output exactly two lowercase tags: <thinking>...</thinking> and <final>...</final>. "
                "Do not omit the thinking tag."
            ),
        },
        {"role": "user", "content": generator_prompt(row)},
    ]
    for attempt in range(max(1, args.max_attempts)):
        try:
            gen = e40.call_model(args.base_url, api_key, args.model, messages, args.gen_max_tokens, args.timeout)
            content = gen.get("content") or ""
            thinking, final_obj, hard_errors = hard_verify(row, content)
            attempt_rec = {
                "attempt": attempt + 1,
                "generator": gen,
                "thinking": thinking,
                "final_obj": final_obj,
                "hard_errors": hard_errors,
                "hard_ok": not hard_errors,
            }
            if hard_errors:
                rec["attempts"].append(attempt_rec)
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Regenerate the whole answer. The previous answer failed validation: "
                            + json.dumps(hard_errors, ensure_ascii=False)
                            + ". The answer must start with <thinking>, include substantive task-grounded reasoning, then output <final>{JSON}</final>. No text outside the two tags."
                        ),
                    }
                )
                continue
            ver = e40.call_model(
                args.base_url,
                api_key,
                args.verifier_model,
                [
                    {"role": "system", "content": "You are a strict verifier for event-extraction CoT/evidence data. Return strict JSON only."},
                    {"role": "user", "content": verifier_prompt(row, thinking or "", final_obj or {})},
                ],
                args.verify_max_tokens,
                args.timeout,
            )
            verifier_obj = e40.extract_json_obj(ver.get("content") or "")
            ok, semantic_errors = semantic_pass(verifier_obj)
            attempt_rec.update(
                {
                    "verifier": ver,
                    "verifier_obj": verifier_obj,
                    "semantic_errors": semantic_errors,
                    "semantic_ok": ok,
                }
            )
            rec["attempts"].append(attempt_rec)
            if ok:
                rec.update(attempt_rec)
                rec["api_ok"] = True
                rec["verifier_api_ok"] = True
                rec["accepted"] = True
                return rec
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Regenerate the whole answer. The verifier rejected it: "
                        + json.dumps(semantic_errors, ensure_ascii=False)
                        + ". Improve evidence informativeness and task-grounded reasoning while keeping the exact target surface events."
                    ),
                }
            )
        except Exception as exc:  # keep batch generation resilient
            rec["error"] = repr(exc)
            break
    if rec["attempts"]:
        last = rec["attempts"][-1]
        rec.update(last)
        rec["api_ok"] = True
        rec["verifier_api_ok"] = bool(last.get("verifier"))
    return rec


def make_evidence_row(row: dict, thinking: str, final_obj: dict, dataset_role: str, run_name: str) -> dict:
    out = BASE_MAKE_EVIDENCE_ROW(row, thinking, final_obj, dataset_role, run_name)
    out["instruction"] = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "First output `<thinking>...</thinking>` with task-grounded natural-language reasoning: ground event meanings in the schema, "
        "choose minimal textual triggers, ground arguments in local evidence, control annotation-style boundaries, and verify the final answer. "
        "Then output `<final>{...}</final>` with a surface-only JSON event list: each trigger and argument must include `text` and a short contiguous local `evidence` quote from the input text. "
        "Do not output numeric offsets, token indices, or text outside these lowercase tags."
    )
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "task_grounded_evidence_cot_e45",
            "adaptive_target_style": "task_grounded_thinking_surface_evidence_cot",
            "e45_run_name": run_name,
            "e45_generator_model": "deepseek-v4-pro",
            "e45_verifier_model": "deepseek-v4-pro",
        }
    )
    return out


def make_eval_evidence_row(row: dict, dataset_role: str, run_name: str) -> dict:
    placeholder = (
        "Ground event meanings in the schema, choose minimal textual triggers, ground arguments in local evidence, "
        "control extraction-style boundaries, and keep the final answer surface-only without numeric offsets."
    )
    return make_evidence_row(row, placeholder, e40.surface_with_empty_evidence(row), dataset_role, run_name)


def write_train_config(branch: str, train_name: str, dev_name: str) -> Path:
    path = e40.CONFIG_DIR / f"{QWEN4_RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    config = {
        "model_name_or_path": QWEN4_WARM_START,
        "template": "qwen",
        "dataset_dir": "/workspace/project/data/stage2_adaptive_datasets",
        "dataset": train_name,
        "eval_dataset": dev_name,
        "output_dir": f"/workspace/project/outputs/stage2_adaptive_runs_user/{QWEN4_RUN_PREFIX}_{branch}_full",
        "stage": "sft",
        "do_train": True,
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "logging_steps": 1,
        "report_to": "none",
        "finetuning_type": "full",
        "cutoff_len": 1536,
        "max_samples": 20000,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "packing": False,
        "learning_rate": 2.0e-6,
        "warmup_ratio": 0.05,
        "bf16": True,
        "val_size": 0.0,
        "eval_steps": 10,
        "do_eval": True,
        "save_only_model": True,
        "num_train_epochs": 3.0,
        "load_best_model_at_end": False,
        "deepspeed": "/workspace/project/configs/deepspeed/zero2_optimizer_offload_cpu.json",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="e45_qwen4_seed1500_task_grounded_cot")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=4545)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--base_url", default=e40.DEFAULT_BASE_URL)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--verifier_model", default="deepseek-v4-pro")
    parser.add_argument("--gen_max_tokens", type=int, default=8192)
    parser.add_argument("--verify_max_tokens", type=int, default=1600)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--reuse_existing", action="store_true", default=True)
    parser.add_argument("--retry_rejected", action="store_true")
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def main():
    e40.generator_prompt = generator_prompt
    e40.verifier_prompt = verifier_prompt
    e40.hard_verify = hard_verify
    e40.semantic_pass = semantic_pass
    e40.process_one = process_one
    e40.make_evidence_row = make_evidence_row
    e40.make_eval_evidence_row = make_eval_evidence_row
    e40.write_train_config = write_train_config
    e40.RUN_PREFIX = QWEN4_RUN_PREFIX

    args = parse_args()
    if args.output_dir is None:
        args.output_dir = REPO / "outputs/stage2_strategy_cot_e45" / f"{args.run_name}_20260605"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = e40.load_jsonl(e40.FORMAL_DATA_DIR / f"{e40.DATA_PREFIX}_train_pos.jsonl")
    sampled = e40.sample_rows(source_rows, args.limit, args.seed, args.run_name)
    e40.write_jsonl(args.output_dir / "sampled_rows.jsonl", sampled)
    results = e40.run_generation(sampled, args)
    dataset_info = e40.write_datasets(sampled, results, args)
    summary = e40.summarize(sampled, results, dataset_info, args)
    summary["mode"] = "task_grounded_evidence_cot_e45"
    summary["qwen4_warm_start"] = QWEN4_WARM_START
    e40.write_json(args.output_dir / "e45_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
