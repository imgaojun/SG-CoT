#!/usr/bin/env python3
"""L2b: few-shot in-context-learning external baseline (Code4Struct/ChatIE-style prompting).

Prompts an LLM with K demonstration (input -> gold JSON) examples drawn from the TRAIN split, then
the test input, same candidate-conditioned instruction and strict-JSON output. Demonstrations are
fixed per run (seeded) and never overlap the test set. Reuses the offset scorer from the zero-shot
baseline for identical evaluation.
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "scripts")
from teacher_zeroshot_baseline_20260702 import parse_json_obj, norm_events, score  # noqa: E402


def build_shots(train_rows, k, seed):
    rng = random.Random(seed)
    # prefer demonstrations that contain at least one event (informative)
    pos = [r for r in train_rows if json.loads(r["output"]).get("events")]
    rng.shuffle(pos)
    shots = pos[:k]
    blocks = []
    for r in shots:
        blocks.append(f"Example input:\n{r['input']}\nExample output:\n{r['output']}")
    return "\n\n".join(blocks)


def call_once(client, model, shots, instruction, text_input, timeout, max_tokens):
    sys_prompt = (
        "You are an information extraction engine. Follow the task instruction and the worked examples "
        "exactly, and return ONLY one strict JSON object, no markdown, no code fences, no commentary."
    )
    user = f"{instruction}\n\n{shots}\n\nNow do the same for this input.\nInput:\n{text_input}\nOutput:"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
        temperature=0.0, max_tokens=max_tokens, timeout=timeout,
    )
    return (resp.choices[0].message.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="glm-5.1")
    ap.add_argument("--base_url", default="${LLM_BASE_URL}")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=os.environ["OPENAI_API_KEY"])
    rows = [json.loads(l) for l in open(args.eval_jsonl)]
    train = [json.loads(l) for l in open(args.train_jsonl)]
    shots = build_shots(train, args.k, args.seed)

    def work(i):
        row = rows[i]; content, parsed = "", None
        for attempt in range(args.max_retries):
            try:
                content = call_once(client, args.model, shots, row["instruction"], row["input"],
                                    args.timeout, args.max_tokens)
                parsed = parse_json_obj(content)
                if parsed is not None:
                    break
            except Exception:
                pass
            time.sleep(2 + attempt * 3)
        return {"idx": i, "raw": content, "parsed": parsed, "gold": row["output"], "meta": row.get("meta", {})}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, range(len(rows))):
            results.append(r)
    results.sort(key=lambda r: r["idx"])
    with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in results if r["parsed"] is not None)
    scored = [(norm_events(json.loads(r["gold"])), norm_events(r["parsed"] or {})) for r in results]
    A, E, T = score(scored)
    summary = {"label": args.label, "model": args.model, "k": args.k, "n": len(rows),
               "parsed_ok": ok, "argument_f1": A, "event_f1": E, "trigger_f1": T}
    json.dump(summary, open(os.path.join(args.output_dir, "summary.json"), "w"), indent=2)
    print(f"[{args.label}] k={args.k} n={len(rows)} parsed={ok}  A/E/T = {A:.4f}/{E:.4f}/{T:.4f}")


if __name__ == "__main__":
    main()
