#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

from scripts.generate_strategy_natural_cot_e37_20260604 import (  # noqa: E402
    DEFAULT_BASE_URL,
    call_model,
    extract_schema,
    extract_text,
)


TZ = timezone(timedelta(hours=8))
OUT_DIR = REPO / "outputs/stage2_strategy_cot_e45/task_grounded_cot_cases_20260605"


CASE_SPECS = [
    ("seen", 4, "e40_win_marriage"),
    ("seen", 23, "direct_win_meet_span_argument"),
    ("seen", 3, "partial_marriage_divorce"),
    ("unseen", 33, "e40_win_elect_type"),
    ("unseen", 11, "direct_win_injury_trigger_drift"),
    ("unseen", 43, "partial_annotation_boundary"),
]


PRED_PATHS = {
    "seen": {
        "direct": REPO
        / "outputs/stage2_full_sft_runs_stepmatch_best_eval_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full_test_seen_argfirst/predictions.jsonl",
        "e40": REPO
        / "outputs/stage2_strategy_cot_e43/model_scaling_20260605/qwen4_e40_seed1500/checkpoint-249/test_seen/predictions.jsonl",
    },
    "unseen": {
        "direct": REPO
        / "outputs/stage2_full_sft_runs_stepmatch_best_eval_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full_test_unseen_argfirst/predictions.jsonl",
        "e40": REPO
        / "outputs/stage2_strategy_cot_e43/model_scaling_20260605/qwen4_e40_seed1500/checkpoint-166/test_unseen/predictions.jsonl",
    },
}


def now_iso() -> str:
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_offsets(payload: dict) -> dict:
    events = []
    for event in (payload or {}).get("events", []) or []:
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict):
                args.append({"role": arg.get("role"), "text": arg.get("text")})
        events.append(
            {
                "event_type": event.get("event_type"),
                "trigger": {"text": trigger.get("text")},
                "arguments": args,
            }
        )
    return {"events": events}


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>", text or "", re.I | re.S)
    return match.group(1).strip() if match else None


def extract_json_obj(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def final_surface(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {"events": []}
    events = []
    for event in payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        args = []
        for arg in event.get("arguments", []) or []:
            if isinstance(arg, dict):
                args.append({"role": arg.get("role"), "text": arg.get("text")})
        events.append({"event_type": event.get("event_type"), "trigger": {"text": trigger.get("text")}, "arguments": args})
    return {"events": events}


def canonical(payload: dict) -> str:
    events = []
    for event in payload.get("events", []) or []:
        args = sorted(event.get("arguments", []) or [], key=lambda x: (x.get("role") or "", x.get("text") or ""))
        events.append(
            {
                "event_type": event.get("event_type"),
                "trigger": event.get("trigger") or {},
                "arguments": args,
            }
        )
    events = sorted(
        events,
        key=lambda e: (
            e.get("event_type") or "",
            (e.get("trigger") or {}).get("text") or "",
            json.dumps(e.get("arguments", []), ensure_ascii=False, sort_keys=True),
        ),
    )
    return json.dumps({"events": events}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sentence_count(text: str) -> int:
    return len([x for x in re.split(r"[.!?]+\s*", text or "") if x.strip()])


def validate_content(text: str, final_obj: dict | None, gold_surface: dict, source_text: str) -> list[str]:
    errors = []
    stripped = (text or "").strip()
    if not stripped.lower().startswith("<thinking>"):
        errors.append("text_before_thinking")
    if not stripped.lower().endswith("</final>"):
        errors.append("text_after_final")
    if re.search(r"<\s*/?\s*think\s*>", stripped, re.I):
        errors.append("malformed_think_tag")
    thinking = extract_tag(stripped, "thinking") or ""
    thinking_words = len(thinking.split())
    thinking_sentences = sentence_count(thinking)
    if thinking_words < 120:
        errors.append(f"thinking_too_short_words:{thinking_words}")
    if thinking_sentences < 5:
        errors.append(f"thinking_too_few_sentences:{thinking_sentences}")
    if thinking_words > 330:
        errors.append(f"thinking_too_long_words:{thinking_words}")
    if final_obj is None:
        errors.append("final_json_parse_failed")
        return errors
    if canonical(final_surface(final_obj)) != canonical(gold_surface):
        errors.append("final_surface_mismatch")
    source_n = norm_text(source_text)
    for event_i, event in enumerate(final_obj.get("events", []) or []):
        if not isinstance(event, dict):
            errors.append(f"event_{event_i}_not_object")
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        trigger_text = trigger.get("text") or ""
        trigger_evidence = trigger.get("evidence") or ""
        trigger_text_words = norm_text(trigger_text).split()
        trigger_evidence_words = norm_text(trigger_evidence).split()
        if norm_text(trigger_text) not in norm_text(trigger_evidence):
            errors.append(f"event_{event_i}_trigger_evidence_missing_text:{trigger_text}")
        if norm_text(trigger_evidence) not in source_n:
            errors.append(f"event_{event_i}_trigger_evidence_not_in_text:{trigger_evidence}")
        if len(trigger_text_words) == 1 and len(trigger_evidence_words) <= 1:
            errors.append(f"event_{event_i}_trigger_evidence_too_short:{trigger_evidence}")
        if len(trigger_evidence_words) > 18:
            errors.append(f"event_{event_i}_trigger_evidence_too_long:{trigger_evidence}")
        for arg_i, arg in enumerate(event.get("arguments", []) or []):
            if not isinstance(arg, dict):
                errors.append(f"event_{event_i}_arg_{arg_i}_not_object")
                continue
            arg_text = arg.get("text") or ""
            arg_evidence = arg.get("evidence") or ""
            arg_text_words = norm_text(arg_text).split()
            arg_evidence_words = norm_text(arg_evidence).split()
            if norm_text(arg_text) not in norm_text(arg_evidence):
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_missing_text:{arg_text}")
            if norm_text(arg_evidence) not in source_n:
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_not_in_text:{arg_evidence}")
            if len(arg_text_words) == 1 and len(arg_evidence_words) <= 1:
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_too_short:{arg_evidence}")
            if len(arg_evidence_words) > 22:
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_too_long:{arg_evidence}")
    return errors


def case_prompt(row: dict, direct: dict, e40: dict, label: str) -> str:
    candidates, schema_cards = extract_schema(row["input"])
    payload = {
        "task": "Generate general task-grounded natural-language CoT supervision for event extraction.",
        "goal": (
            "The reasoning strategy should be generic across models and datasets. It should normalize broad "
            "language-model semantic priors into schema-aligned, annotation-grounded extraction decisions."
        ),
        "important_positioning": [
            "The gold events are authoritative for this supervision example.",
            "Do not add, remove, reorder, or modify gold event types, trigger text, argument text, or roles.",
            "Do not mention Direct, E40, model errors, or this case label in the output.",
            "Do not write case-specific rules. Use general extraction principles that would apply to most event extraction examples.",
        ],
        "required_output": "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only.",
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Start from the event meanings expressed in the text, then ground them in the provided schemas and candidate event types.",
            "For each target event, explicitly state why the event type fits the schema definition or trigger cues.",
            "Explain trigger boundary control: choose the explicit textual mention that expresses the event, prefer the minimal lexical trigger, and avoid surrounding explanation, diagnosis, report, or context words unless they are themselves the event mention.",
            "Explain argument-role grounding: include only arguments explicitly supported by local evidence and state how each argument participates in the event role.",
            "Explain extraction-style control: avoid duplicate events, avoid semantically plausible but weakly grounded extra events, and avoid unsupported role filling from world knowledge.",
            "End by verifying that every final trigger and argument has short contiguous local evidence from the text.",
        ],
        "thinking_quality_requirements": [
            "The thinking should contain 5 to 8 natural sentences.",
            "It must mention the selected event type names and trigger texts.",
            "It must mention at least one reason for excluding nearby context or extra candidates when the text contains potentially confusing context.",
            "It must describe why the evidence phrase is local and sufficient.",
            "Do not merely say that the extraction matches the schema; explain the actual local textual cue.",
        ],
        "final_json_schema": {
            "events": [
                {
                    "event_type": "copy from gold",
                    "trigger": {"text": "copy from gold", "evidence": "short contiguous quote from Text containing trigger text"},
                    "arguments": [
                        {"role": "copy from gold", "text": "copy from gold", "evidence": "short contiguous quote from Text containing argument text"}
                    ],
                }
            ]
        },
        "constraints": [
            "No numeric offsets, token indices, or character positions.",
            "Every evidence string must be an exact contiguous quote from Text and contain the corresponding text.",
            "This is mandatory: trigger.evidence must contain trigger.text exactly, and argument evidence must contain argument.text exactly.",
            "Never use a nearby context clause as trigger evidence if that clause does not contain the trigger text.",
            "Evidence should be a local phrase or short clause, not just the isolated trigger or argument token when a local phrase is available.",
            "Good trigger evidence examples: 'she had a broken jaw', 'they met up', '22 000 ended up in prison'. Bad examples: 'broken', 'met', '22 000' when local context is available.",
            "Good argument evidence usually shows the local relation to the trigger, such as 'she had a broken jaw' rather than only 'she'.",
            "Avoid explicitly saying that you are copying gold labels or that only one event is labeled; phrase it as target extraction style, schema alignment, and evidence grounding.",
            "Keep thinking natural, concise, and general; 150-280 English words.",
            "Return no markdown and no text outside the two lowercase tags.",
        ],
        "input": {
            "case_label_for_internal_selection_only": label,
            "text": extract_text(row["input"]),
            "candidate_event_types": candidates,
            "schema_cards": schema_cards,
            "gold_surface_events_to_copy": strip_offsets(row.get("gold") or {}),
            "diagnostic_context_not_to_copy": {
                "direct_prediction_surface": strip_offsets(direct.get("predicted") or {}),
                "previous_e40_prediction_surface": strip_offsets(e40.get("predicted") or {}),
                "purpose": "Use these only to understand possible pitfalls. Do not mention them in the output.",
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_markdown(records: list[dict]) -> str:
    lines = [
        "# E45 Task-Grounded CoT Case Generations",
        "",
        f"Updated: {now_iso()}",
        "",
        "## Strategy",
        "",
        "Generic task-grounded CoT: semantic understanding -> schema grounding -> mention grounding -> argument grounding -> annotation alignment -> final verification.",
        "",
        "The generated examples below are for manual review before scaling data construction.",
        "",
    ]
    for rec in records:
        lines.extend(
            [
                f"## {rec['split']} #{rec['index']} {rec['label']}",
                "",
                "### Text",
                "",
                rec["text"],
                "",
                "### Gold Surface",
                "",
                "```json",
                json.dumps(rec["gold_surface"], ensure_ascii=False, indent=2),
                "```",
                "",
                f"### {rec.get('model', 'model')} Output",
                "",
                rec.get("content", "").strip(),
                "",
                "### Parsed Final",
                "",
                "```json",
                json.dumps(rec.get("parsed_final"), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--provider-label", default=None)
    args = parser.parse_args()
    if args.provider_label and args.out_dir == OUT_DIR:
        args.out_dir = OUT_DIR.parent / f"{OUT_DIR.name}_{args.provider_label}"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    loaded = {
        split: {name: load_jsonl(path) for name, path in paths.items()}
        for split, paths in PRED_PATHS.items()
    }
    records = []
    for split, idx, label in CASE_SPECS:
        direct = loaded[split]["direct"][idx]
        e40 = loaded[split]["e40"][idx]
        prompt = case_prompt(direct, direct, e40, label)
        messages = [
            {
                "role": "system",
                "content": (
                    "You create faithful, general, task-grounded CoT supervision for event extraction. "
                    "Output only the requested lowercase tags. The reasoning must be substantive, schema-grounded, and evidence-grounded, not a terse template."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        attempts = []
        selected = None
        for attempt in range(3):
            response = call_model(args.base_url, api_key, args.model, messages, args.max_tokens, args.timeout)
            content = response.get("content") or ""
            final_text = extract_tag(content, "final")
            parsed_final = extract_json_obj(final_text or "")
            hard_errors = validate_content(content, parsed_final, strip_offsets(direct.get("gold") or {}), extract_text(direct["input"]))
            attempts.append({"attempt": attempt + 1, "response": response, "hard_errors": hard_errors})
            if not hard_errors:
                selected = {"response": response, "content": content, "parsed_final": parsed_final, "hard_errors": hard_errors}
                break
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Regenerate the whole answer. The previous answer failed hard validation: "
                        + json.dumps(hard_errors, ensure_ascii=False)
                        + ". Regenerate with a substantive 5-8 sentence thinking section of 150-280 words, local phrase/clause evidence, not isolated tokens. Output must start with <thinking>, end with </final>, and every evidence string must be an exact quote containing the corresponding text."
                    ),
                }
            )
        if selected is None:
            last = attempts[-1]
            response = last["response"]
            content = response.get("content") or ""
            final_text = extract_tag(content, "final")
            parsed_final = extract_json_obj(final_text or "")
            hard_errors = last["hard_errors"]
        else:
            response = selected["response"]
            content = selected["content"]
            parsed_final = selected["parsed_final"]
            hard_errors = selected["hard_errors"]
        records.append(
            {
                "split": split,
                "index": idx,
                "label": label,
                "model": args.model,
                "text": extract_text(direct["input"]),
                "gold_surface": strip_offsets(direct.get("gold") or {}),
                "prompt": prompt,
                "response": response,
                "attempts": attempts,
                "hard_errors": hard_errors,
                "content": content,
                "thinking": extract_tag(content, "thinking"),
                "parsed_final": parsed_final,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "records.json", {"created_at": now_iso(), "model": args.model, "records": records})
    (args.out_dir / "case_review.md").write_text(render_markdown(records), encoding="utf-8")
    print(json.dumps({"out_dir": args.out_dir.as_posix(), "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
