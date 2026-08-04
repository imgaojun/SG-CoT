#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.generate_strategy_variants_cot_e47_20260606 as e47  # noqa: E402
import scripts.generate_evidence_cot_e40_20260604 as e40  # noqa: E402


BASE_GENERATOR_PROMPT = e47.generator_prompt
BASE_VERIFIER_PROMPT = e47.verifier_prompt
BASE_MAKE_EVIDENCE_ROW = e47.make_evidence_row


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def factor_from_row(row: dict) -> dict:
    meta = row.get("meta") or {}
    return {
        "factor": meta.get("e60_factor", "candidate_audit"),
        "description": meta.get("e60_factor_description", ""),
        "dataset_instantiation": meta.get("e60_dataset_instantiation", ""),
        "strict_unseen_safe": bool(meta.get("e60_strict_unseen_safe", True)),
        "schema_synthetic": bool(meta.get("e60_schema_synthetic", False)),
    }


def generator_prompt(row: dict, prompt_profile: str = "standard", output_protocol: str = "xml_tags") -> str:
    payload = json.loads(BASE_GENERATOR_PROMPT(row, prompt_profile, output_protocol))
    factor = factor_from_row(row)
    payload["task"] = "Generate factor-balanced natural-language CoT supervision for event extraction."
    payload["goal"] = (
        "Teach a general event-extraction reasoning factor, not a case-specific error repair. "
        "The final event labels are authoritative for this supervision example."
    )
    payload["factor_balanced_reconstruction"] = {
        "abstract_factor": factor["factor"],
        "abstract_factor_description": factor["description"],
        "dataset_instantiation": factor["dataset_instantiation"],
        "strict_unseen_safe": factor["strict_unseen_safe"],
        "schema_synthetic": factor["schema_synthetic"],
        "how_to_use_this_factor": [
            "Explain the abstract decision in natural language before relying on dataset-specific labels.",
            "Use the RichERE instantiation only as an example of the abstract factor.",
            "Do not write a rule that only applies to one event type.",
            "Keep the reasoning faithful to the target final events and local evidence.",
        ],
    }
    payload["thinking_strategy"] = [
        "Write a substantive but concise natural-language paragraph, not a checklist.",
        "Start from the requested abstract reasoning factor and explain the structural decision it controls.",
        "Then instantiate that factor using the local text, candidate event types, and schema cards.",
        "Audit plausible event mentions and close negative candidates only when they are relevant to the factor.",
        "For retained targets, normalize each trigger to the shortest copied lexical anchor and keep broader context only in evidence.",
        "Ground each argument locally near the event mention and abstain from unsupported roles.",
        "End by checking that final events cover the target-style mentions and exclude duplicates or semantically plausible extras.",
    ]
    payload["thinking_quality_requirements"] = [
        "The thinking should usually contain 7 to 10 natural sentences.",
        "It must name the abstract factor and express it as a general event-extraction decision.",
        "It must mention selected event type names and trigger texts.",
        "It must connect the RichERE-specific instance back to the abstract factor.",
        "It must describe minimal trigger selection, local argument grounding, and final consistency when relevant.",
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def verifier_prompt(row: dict, thinking: str, final_obj: dict) -> str:
    payload = json.loads(BASE_VERIFIER_PROMPT(row, thinking, final_obj))
    factor = factor_from_row(row)
    payload["task"] = "Strictly verify factor-balanced CoT/evidence training data for event extraction."
    payload["factor_balanced_requirements"] = [
        "The thinking must be framed as a general event-extraction reasoning factor, not a case-specific repair.",
        "The RichERE-specific labels may be used as an instantiation, but the reasoning should remain abstract enough to transfer.",
        "The thinking must not claim that all future examples of this event type should follow a special rule.",
        "The final surface events must exactly match the target surface events.",
    ]
    payload["input"]["requested_factor"] = factor
    return json.dumps(payload, ensure_ascii=False, indent=2)


def make_evidence_row(row: dict, thinking: str, final_obj: dict, dataset_role: str, run_name: str) -> dict:
    out = BASE_MAKE_EVIDENCE_ROW(row, thinking, final_obj, dataset_role, run_name)
    factor = factor_from_row(row)
    out["instruction"] = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        f"First output `<thinking>...</thinking>` with natural-language reasoning for the abstract factor `{factor['factor']}`: "
        f"{factor['description']} "
        f"In this dataset, instantiate it as: {factor['dataset_instantiation']} "
        "Then output `<final>{...}</final>` with a surface-only JSON event list: each trigger and argument must include `text` and a short contiguous local `evidence` quote from the input text. "
        "Do not output numeric offsets, token indices, or text outside these lowercase tags."
    )
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": "factor_balanced_evidence_cot_e60",
            "adaptive_target_style": "factor_balanced_thinking_surface_evidence_cot",
            "e60_run_name": run_name,
            "e60_factor": factor["factor"],
            "e60_factor_description": factor["description"],
            "e60_dataset_instantiation": factor["dataset_instantiation"],
            "e60_generator_model": os.environ.get("E60_GENERATOR_MODEL", "glm-5.1"),
            "e60_verifier_model": os.environ.get("E60_VERIFIER_MODEL", "deepseek-v4-pro"),
        }
    )
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="e60a_glm51_factor_probe_w8")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
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
    ap.add_argument("--output_dir", type=Path, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = REPO / "outputs/stage2_strategy_cot_e60" / args.run_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e47.generator_prompt = generator_prompt
    e47.verifier_prompt = verifier_prompt
    e47.make_evidence_row = make_evidence_row
    e40.generator_prompt = generator_prompt
    e40.verifier_prompt = verifier_prompt
    e40.hard_verify = e47.hard_verify
    e40.semantic_pass = e47.semantic_pass
    e40.process_one = e47.process_one
    e40.run_generation = e47.run_generation
    e40.make_evidence_row = make_evidence_row
    e40.make_eval_evidence_row = e47.make_eval_evidence_row
    e40.write_train_config = e47.write_train_config
    e40.RUN_PREFIX = e47.QWEN4_RUN_PREFIX
    os.environ["E60_GENERATOR_MODEL"] = args.model
    os.environ["E60_VERIFIER_MODEL"] = args.verifier_model

    sampled = load_jsonl(args.manifest)
    e40.write_jsonl(args.output_dir / "sampled_rows.jsonl", sampled)
    results = e47.run_generation(sampled, args)
    dataset_info = e40.write_datasets(sampled, results, args)
    summary = e40.summarize(sampled, results, dataset_info, args)
    summary["mode"] = "factor_balanced_evidence_cot_e60"
    summary["manifest"] = args.manifest.as_posix()
    summary["generator_model"] = args.model
    summary["verifier_model"] = args.verifier_model
    summary["workers"] = args.workers
    summary["prompt_profile"] = args.prompt_profile
    summary["output_protocol"] = args.output_protocol
    factor_counts = {}
    accepted_factor_counts = {}
    by_id = {r["meta"]["e40_sample_id"]: r for r in sampled}
    for row in sampled:
        factor = row.get("meta", {}).get("e60_factor", "unknown")
        factor_counts[factor] = factor_counts.get(factor, 0) + 1
    for rec in results:
        if rec.get("accepted"):
            factor = by_id.get(rec.get("sample_id"), {}).get("meta", {}).get("e60_factor", "unknown")
            accepted_factor_counts[factor] = accepted_factor_counts.get(factor, 0) + 1
    summary["factor_counts"] = factor_counts
    summary["accepted_factor_counts"] = accepted_factor_counts
    e40.write_json(args.output_dir / "e60_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
