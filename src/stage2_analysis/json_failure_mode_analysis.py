import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def avg(values):
    return sum(values) / len(values) if values else 0.0


def build_prompt(tokenizer, instruction: str, input_text: str):
    messages = [{"role": "user", "content": f"{instruction}\n{input_text}"}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{instruction}\n{input_text}"


def detect_primary_reason(record):
    if record["valid_json"]:
        return "valid"
    if record["unclosed_brace"] or record["unclosed_bracket"]:
        if record["runaway_role_repetition"] or record["runaway_event_repetition"]:
            return "runaway_truncated_json"
        return "truncated_json"
    if record["instruction_echo"]:
        return "instruction_echo"
    if record["starts_with_think"]:
        return "prefix_think_only"
    return "other_invalid"


def classify_row(row, tokenizer=None, max_new_tokens=None):
    text = row.get("generated_text", "")
    stripped = text.lstrip()
    generated_tokens = None
    prompt_tokens = None

    if tokenizer is not None:
        generated_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        prompt = build_prompt(tokenizer, row["instruction"], row["input"])
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    instruction_echo = any(
        marker in text
        for marker in [
            "Return JSON only.\nassistant",
            "Output requirements:\n",
            "`decisions` and `events`.",
            "contains `event_type`, `trigger`, and `arguments`.",
        ]
    ) or stripped.startswith("`decisions`") or stripped.startswith("contains `event_type`")

    role_mentions = text.count('"role"')
    event_mentions = text.count('"event_type"')

    record = {
        "valid_json": bool(row.get("valid_json", False)),
        "generated_chars": len(text),
        "generated_tokens": generated_tokens,
        "prompt_tokens": prompt_tokens,
        "latency_sec": row.get("latency_sec", 0.0),
        "starts_with_think": stripped.startswith("<think>"),
        "has_assistant_think": "assistant\n<think>" in text or "assistant\n\n<think>" in text,
        "instruction_echo": instruction_echo,
        "open_braces": text.count("{"),
        "close_braces": text.count("}"),
        "open_brackets": text.count("["),
        "close_brackets": text.count("]"),
        "unclosed_brace": text.count("{") > text.count("}"),
        "unclosed_bracket": text.count("[") > text.count("]"),
        "near_max_new_tokens": (
            generated_tokens is not None and max_new_tokens is not None and generated_tokens >= max_new_tokens - 8
        ),
        "role_mentions": role_mentions,
        "event_mentions": event_mentions,
        "runaway_role_repetition": role_mentions >= 20,
        "runaway_event_repetition": event_mentions >= 8,
        "trigger_f1": row.get("trigger_f1", 0.0),
        "argument_f1": row.get("argument_f1", 0.0),
        "event_f1": row.get("event_f1", 0.0),
        "snippet": text[:800],
    }
    record["primary_reason"] = detect_primary_reason(record)
    return record


def summarize(records):
    invalid = [r for r in records if not r["valid_json"]]
    valid = [r for r in records if r["valid_json"]]
    reason_counts = Counter(r["primary_reason"] for r in records)

    payload = {
        "num_examples": len(records),
        "json_valid_rate": avg([1.0 if r["valid_json"] else 0.0 for r in records]),
        "avg_generated_chars": avg([r["generated_chars"] for r in records]),
        "avg_generated_chars_valid": avg([r["generated_chars"] for r in valid]),
        "avg_generated_chars_invalid": avg([r["generated_chars"] for r in invalid]),
        "avg_generated_tokens": avg([r["generated_tokens"] for r in records if r["generated_tokens"] is not None]),
        "avg_generated_tokens_valid": avg([r["generated_tokens"] for r in valid if r["generated_tokens"] is not None]),
        "avg_generated_tokens_invalid": avg([r["generated_tokens"] for r in invalid if r["generated_tokens"] is not None]),
        "avg_prompt_tokens": avg([r["prompt_tokens"] for r in records if r["prompt_tokens"] is not None]),
        "avg_latency_sec": avg([r["latency_sec"] for r in records]),
        "avg_trigger_f1": avg([r["trigger_f1"] for r in records]),
        "avg_argument_f1": avg([r["argument_f1"] for r in records]),
        "avg_event_f1": avg([r["event_f1"] for r in records]),
        "primary_reason_counts": dict(reason_counts),
        "invalid_feature_rates": {
            "starts_with_think": avg([1.0 if r["starts_with_think"] else 0.0 for r in invalid]),
            "has_assistant_think": avg([1.0 if r["has_assistant_think"] else 0.0 for r in invalid]),
            "instruction_echo": avg([1.0 if r["instruction_echo"] else 0.0 for r in invalid]),
            "unclosed_brace": avg([1.0 if r["unclosed_brace"] else 0.0 for r in invalid]),
            "unclosed_bracket": avg([1.0 if r["unclosed_bracket"] else 0.0 for r in invalid]),
            "near_max_new_tokens": avg([1.0 if r["near_max_new_tokens"] else 0.0 for r in invalid]),
            "runaway_role_repetition": avg([1.0 if r["runaway_role_repetition"] else 0.0 for r in invalid]),
            "runaway_event_repetition": avg([1.0 if r["runaway_event_repetition"] else 0.0 for r in invalid]),
        },
    }
    return payload


def select_examples(records, limit):
    invalid = [r for r in records if not r["valid_json"]]
    invalid = sorted(
        invalid,
        key=lambda r: (
            1 if r["near_max_new_tokens"] else 0,
            r["generated_tokens"] or r["generated_chars"],
            r["role_mentions"],
            r["event_mentions"],
        ),
        reverse=True,
    )
    examples = []
    for idx, row in enumerate(invalid[:limit], start=1):
        examples.append(
            {
                "rank": idx,
                "primary_reason": row["primary_reason"],
                "generated_chars": row["generated_chars"],
                "generated_tokens": row["generated_tokens"],
                "starts_with_think": row["starts_with_think"],
                "instruction_echo": row["instruction_echo"],
                "unclosed_brace": row["unclosed_brace"],
                "unclosed_bracket": row["unclosed_bracket"],
                "near_max_new_tokens": row["near_max_new_tokens"],
                "role_mentions": row["role_mentions"],
                "event_mentions": row["event_mentions"],
                "snippet": row["snippet"],
            }
        )
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tokenizer_model")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--example_limit", type=int, default=3)
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer_model:
        if AutoTokenizer is None:
            raise ImportError("transformers is required when --tokenizer_model is provided")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)

    rows = load_jsonl(Path(args.predictions_jsonl))
    records = [classify_row(row, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens) for row in rows]

    payload = {
        "label": args.label,
        "predictions_jsonl": args.predictions_jsonl,
        "tokenizer_model": args.tokenizer_model,
        "max_new_tokens": args.max_new_tokens,
        "summary": summarize(records),
        "examples": select_examples(records, args.example_limit),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
