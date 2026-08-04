#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.generate_evidence_cot_e40_20260604 as e40  # noqa: E402
import scripts.generate_strategy_variants_cot_e47_20260606 as e47  # noqa: E402


QWEN4_RUN_PREFIX = e47.QWEN4_RUN_PREFIX
ACTIVE_PROFILE = "precision_plus"


PROFILES = {
    "full": {
        "title": "full E57-style candidate-audit operation set",
        "goal": "Teach the complete E57-style operation set for schema-conditioned event extraction.",
        "operations": [
            "candidate audit: cover plausible event mentions in text order and decide kept versus rejected candidates",
            "close-type contrast: compare neighboring candidate event types when a mention could fit more than one schema",
            "minimal trigger selection: keep trigger.text as the shortest copied lexical anchor",
            "local role grounding: include only arguments locally tied to the trigger",
            "role abstention: leave plausible but unsupported roles unfilled",
            "duplicate and extra-event suppression: reject duplicate frames and semantically plausible extras without annotation-style triggers",
            "evidence-final consistency: every final surface text must be supported by a short local evidence quote",
        ],
        "relaxed_scores": set(),
        "instruction_focus": (
            "audit candidates, contrast close types, choose minimal triggers, ground roles locally, abstain from unsupported roles, "
            "suppress duplicates or extras, and keep evidence aligned with final JSON"
        ),
        "word_range": "110-230",
    },
    "no_close": {
        "title": "candidate audit without explicit close-type contrast",
        "goal": "Ablate explicit close-type contrast while keeping candidate audit, role grounding, suppression, and evidence alignment.",
        "operations": [
            "candidate audit: cover plausible event mentions in text order and decide kept versus rejected candidates",
            "minimal trigger selection: keep trigger.text as the shortest copied lexical anchor",
            "local role grounding: include only arguments locally tied to the trigger",
            "role abstention: leave plausible but unsupported roles unfilled",
            "duplicate and extra-event suppression: reject duplicate frames and semantically plausible extras without annotation-style triggers",
            "evidence-final consistency: every final surface text must be supported by a short local evidence quote",
        ],
        "relaxed_scores": {"type_discrimination"},
        "instruction_focus": (
            "audit candidates, choose minimal triggers, ground roles locally, abstain from unsupported roles, suppress duplicates or extras, "
            "and keep evidence aligned with final JSON; do not include an explicit close-type contrast step"
        ),
        "word_range": "100-210",
    },
    "no_role": {
        "title": "candidate audit without explicit local-role grounding or role abstention rationale",
        "goal": "Ablate explicit role-grounding reasoning while keeping final labels, candidate audit, type decisions, suppression, and evidence alignment.",
        "operations": [
            "candidate audit: cover plausible event mentions in text order and decide kept versus rejected candidates",
            "close-type contrast: compare neighboring candidate event types when a mention could fit more than one schema",
            "minimal trigger selection: keep trigger.text as the shortest copied lexical anchor",
            "duplicate and extra-event suppression: reject duplicate frames and semantically plausible extras without annotation-style triggers",
            "evidence-final consistency: every final surface text must be supported by a short local evidence quote",
        ],
        "relaxed_scores": {"argument_role_grounding", "role_abstention", "argument_evidence"},
        "instruction_focus": (
            "audit candidates, contrast close types, choose minimal triggers, suppress duplicates or extras, and keep evidence aligned with final JSON; "
            "do not include an explicit local-role-grounding or role-abstention rationale"
        ),
        "word_range": "95-200",
    },
    "no_suppress": {
        "title": "candidate audit without explicit duplicate or extra-event suppression",
        "goal": "Ablate explicit no-extra-event and duplicate-suppression reasoning while keeping type, trigger, role, and evidence operations.",
        "operations": [
            "candidate audit: cover plausible event mentions in text order and decide final retained candidates",
            "close-type contrast: compare neighboring candidate event types when a mention could fit more than one schema",
            "minimal trigger selection: keep trigger.text as the shortest copied lexical anchor",
            "local role grounding: include only arguments locally tied to the trigger",
            "role abstention: leave plausible but unsupported roles unfilled",
            "evidence-final consistency: every final surface text must be supported by a short local evidence quote",
        ],
        "relaxed_scores": {"no_extra_event_gate", "candidate_coverage"},
        "instruction_focus": (
            "audit candidates, contrast close types, choose minimal triggers, ground roles locally, abstain from unsupported roles, "
            "and keep evidence aligned with final JSON; do not include an explicit duplicate-suppression or no-extra-event gate"
        ),
        "word_range": "100-210",
    },
    "precision_plus": {
        "title": "precision-oriented E57+ operation set",
        "goal": (
            "Improve seen Argument/Event precision while preserving E57-style schema-conditioned unseen generalization. "
            "Keep reasoning compact and close to final extraction decisions."
        ),
        "operations": [
            "candidate audit: briefly cover plausible event mentions in text order",
            "close-type contrast: mention close negative types only when the local text makes them plausible",
            "minimal trigger selection: keep trigger.text as the shortest copied lexical anchor",
            "local role grounding: include only arguments locally tied to the trigger",
            "role abstention: explicitly reject unsupported Agent/Place/Entity roles when they are tempting",
            "event count check: state that final events include all target-style mentions and no duplicates or extras",
            "evidence-final consistency: every final surface text must be supported by a short local evidence quote",
        ],
        "relaxed_scores": set(),
        "instruction_focus": (
            "use compact precision-oriented candidate audit: keep all target-style events, reject close negative types only when relevant, "
            "choose minimal triggers, ground roles locally, abstain from unsupported roles, check event count, and keep evidence aligned with final JSON"
        ),
        "word_range": "90-180",
    },
}


def profile() -> dict:
    return PROFILES[ACTIVE_PROFILE]


def generator_prompt(row: dict, prompt_profile: str = "standard", output_protocol: str = "xml_tags") -> str:
    candidates, schema_cards = e40.extract_schema(row["input"])
    prof = profile()
    json_wrapper = output_protocol == "json_wrapper"
    payload = {
        "task": f"Generate {prof['title']} CoT supervision for event extraction.",
        "goal": prof["goal"],
        "important_positioning": [
            "The provided target events are authoritative for this supervision example.",
            "Do not add, remove, reorder, or modify target event types, trigger text, argument text, or roles.",
            "Do not mention model names, experiment ids, Direct, E57, E60, E61, or ablation labels in the output.",
            "Use general event-extraction decisions, not case-specific repair rules.",
        ],
        "operation_profile": ACTIVE_PROFILE,
        "required_operations": prof["operations"],
        "required_output": (
            "Return one strict JSON object only with keys `thinking` and `final`. "
            "`thinking` is a natural-language string. `final` is the surface-only event JSON object."
            if json_wrapper
            else "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        ),
        "thinking_style": [
            "Write one compact natural-language paragraph, not bullet points.",
            "Keep the reasoning close to final extraction decisions; avoid abstract terminology unless needed by the operation profile.",
            "Mention selected event type names and minimal trigger texts.",
            "Mention rejected candidates only when they are locally plausible or required by the operation profile.",
            "Do not produce a long checklist; do not repeat schema definitions.",
            f"Aim for {prof['word_range']} English words.",
        ],
        "final_json_schema": {
            "events": [
                {
                    "event_type": "copy from target",
                    "trigger": {
                        "text": "copy from target",
                        "evidence": "short contiguous local phrase/clause from Text containing trigger text",
                    },
                    "arguments": [
                        {
                            "role": "copy from target",
                            "text": "copy from target",
                            "evidence": "short contiguous local phrase/clause from Text containing argument text",
                        }
                    ],
                }
            ]
        },
        "evidence_rules": [
            "Every evidence string must be an exact contiguous quote from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence should be a local phrase or short clause, not an isolated token when local context is available.",
            "Do not use ellipses, bracket insertions, paraphrases, corrected grammar, or compressed wording in evidence.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is shorter.",
        ],
        "constraints": (
            [
                "The first non-whitespace character of your answer must be `{` and the last non-whitespace character must be `}`.",
                "Return valid JSON only: no markdown fences, no headings, no XML tags, no prose before or after the JSON object.",
                "The top-level object must have exactly two keys: `thinking` and `final`.",
                "No numeric offsets, token indices, character positions, markdown, or text outside the JSON object.",
            ]
            if json_wrapper
            else [
                "The first characters of your answer must be <thinking>.",
                "The last characters of your answer must be </final>.",
                "Use exactly one <thinking> block and exactly one <final> block.",
                "No numeric offsets, token indices, character positions, markdown, or text outside the two lowercase tags.",
            ]
        ),
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
    prof = profile()
    relaxed = sorted(prof["relaxed_scores"])
    scores = {
        "trigger_evidence": "integer 1-5",
        "argument_evidence": "integer 1-5",
        "evidence_informativeness": "integer 1-5",
        "type_discrimination": "integer 1-5",
        "trigger_boundary_control": "integer 1-5",
        "argument_role_grounding": "integer 1-5",
        "candidate_coverage": "integer 1-5",
        "minimal_trigger_separation": "integer 1-5",
        "role_abstention": "integer 1-5",
        "no_extra_event_gate": "integer 1-5",
        "final_structure_consistency": "integer 1-5",
    }
    payload = {
        "task": f"Strictly verify {prof['title']} CoT/evidence training data for event extraction.",
        "instruction": "Do not repair the answer. Return strict JSON only.",
        "operation_profile": ACTIVE_PROFILE,
        "required_operations": prof["operations"],
        "relaxed_or_ablate_scores": relaxed,
        "pass_requirements": [
            "The final surface events exactly match the target surface events.",
            "The reasoning is task-grounded rather than a generic template.",
            "The reasoning follows the requested operation profile, including deliberate omissions for ablation profiles.",
            "Every evidence string is an exact contiguous quote from Text and contains the corresponding trigger or argument text.",
            "The response does not use numeric offsets or token indices.",
            "Do not fail an answer only because it omits an operation listed in relaxed_or_ablate_scores.",
        ],
        "return_contract": {
            "pass": "boolean",
            "scores": scores,
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


def semantic_pass(verifier_obj: dict):
    errors = []
    if not isinstance(verifier_obj, dict):
        return False, ["verifier_not_object"]
    if verifier_obj.get("pass") is not True:
        errors.append("semantic_pass_false")
    scores = verifier_obj.get("scores") if isinstance(verifier_obj.get("scores"), dict) else {}
    relaxed = profile()["relaxed_scores"]
    required = [
        "trigger_evidence",
        "argument_evidence",
        "evidence_informativeness",
        "type_discrimination",
        "trigger_boundary_control",
        "argument_role_grounding",
        "candidate_coverage",
        "minimal_trigger_separation",
        "role_abstention",
        "no_extra_event_gate",
    ]
    for key in required:
        if key in relaxed:
            continue
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
        filtered = [str(e) for e in verifier_errors if not any(k in str(e) for k in relaxed)]
        if filtered:
            errors.append("semantic_errors:" + ",".join(filtered[:5]))
    return not errors, errors


def make_evidence_row(row: dict, thinking: str, final_obj: dict, dataset_role: str, run_name: str) -> dict:
    out = e47.BASE_MAKE_EVIDENCE_ROW(row, thinking, final_obj, dataset_role, run_name)
    prof = profile()
    out["instruction"] = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        f"First output `<thinking>...</thinking>` with {prof['title']} reasoning: {prof['instruction_focus']}. "
        "Then output `<final>{...}</final>` with a surface-only JSON event list: each trigger and argument must include `text` and a short contiguous local `evidence` quote from the input text. "
        "Do not output numeric offsets, token indices, or text outside these lowercase tags."
    )
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "operation_study_e62",
            "adaptive_target_style": f"operation_{ACTIVE_PROFILE}_thinking_surface_evidence_cot",
            "e62_run_name": run_name,
            "e62_operation_profile": ACTIVE_PROFILE,
            "e62_generator_model": os.environ.get("E62_GENERATOR_MODEL", "glm-5.1"),
            "e62_verifier_model": os.environ.get("E62_VERIFIER_MODEL", "deepseek-v4-pro"),
        }
    )
    return out


def make_eval_evidence_row(row: dict, dataset_role: str, run_name: str) -> dict:
    prof = profile()
    placeholder = (
        f"Use {prof['title']} reasoning: {prof['instruction_focus']}. "
        "Keep final events surface-only with short local evidence and no numeric offsets."
    )
    return make_evidence_row(row, placeholder, e40.surface_with_empty_evidence(row), dataset_role, run_name)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="e62b_precision_plus_glm51_seed1500")
    ap.add_argument("--operation_profile", choices=sorted(PROFILES), default="precision_plus")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=6262)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--base_url", default=e40.DEFAULT_BASE_URL)
    ap.add_argument("--model", default="glm-5.1")
    ap.add_argument("--verifier_model", default="deepseek-v4-pro")
    ap.add_argument("--gen_max_tokens", type=int, default=8192)
    ap.add_argument("--verify_max_tokens", type=int, default=1800)
    ap.add_argument("--timeout", type=int, default=360)
    ap.add_argument("--reasoning_effort", default=None)
    ap.add_argument("--verifier_reasoning_effort", default="max")
    ap.add_argument("--prompt_profile", choices=["standard", "strict_evidence", "json_acceptance_v2", "xml_lean_v3"], default="standard")
    ap.add_argument("--repair_profile", choices=["strict_full", "concise"], default="strict_full")
    ap.add_argument("--output_protocol", choices=["xml_tags", "json_wrapper"], default="xml_tags")
    ap.add_argument("--reuse_existing", action="store_true", default=True)
    ap.add_argument("--retry_rejected", action="store_true")
    ap.add_argument("--retry_error_contains", default=None)
    ap.add_argument("--max_attempts", type=int, default=2)
    ap.add_argument("--output_dir", type=Path)
    return ap.parse_args()


def main():
    global ACTIVE_PROFILE
    args = parse_args()
    ACTIVE_PROFILE = args.operation_profile
    if args.output_dir is None:
        args.output_dir = REPO / "outputs/stage2_strategy_cot_e62" / args.run_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["E62_GENERATOR_MODEL"] = args.model
    os.environ["E62_VERIFIER_MODEL"] = args.verifier_model

    e47.generator_prompt = generator_prompt
    e47.verifier_prompt = verifier_prompt
    e47.semantic_pass = semantic_pass
    e47.make_evidence_row = make_evidence_row
    e47.make_eval_evidence_row = make_eval_evidence_row
    e40.generator_prompt = generator_prompt
    e40.verifier_prompt = verifier_prompt
    e40.hard_verify = e47.hard_verify
    e40.semantic_pass = semantic_pass
    e40.process_one = e47.process_one
    e40.run_generation = e47.run_generation
    e40.make_evidence_row = make_evidence_row
    e40.make_eval_evidence_row = make_eval_evidence_row
    e40.write_train_config = e47.write_train_config
    e40.RUN_PREFIX = QWEN4_RUN_PREFIX

    source_rows = e40.load_jsonl(e40.FORMAL_DATA_DIR / f"{e40.DATA_PREFIX}_train_pos.jsonl")
    sampled = e40.sample_rows(source_rows, args.limit, args.seed, args.run_name)
    e40.write_jsonl(args.output_dir / "sampled_rows.jsonl", sampled)
    results = e47.run_generation(sampled, args)
    dataset_info = e40.write_datasets(sampled, results, args)
    summary = e40.summarize(sampled, results, dataset_info, args)
    summary["mode"] = "operation_study_e62"
    summary["operation_profile"] = ACTIVE_PROFILE
    summary["generator_model"] = args.model
    summary["verifier_model"] = args.verifier_model
    summary["prompt_profile"] = args.prompt_profile
    summary["output_protocol"] = args.output_protocol
    e40.write_json(args.output_dir / "e62_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
