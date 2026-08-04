#!/usr/bin/env python3
"""Build E81-format CoT eval data for the with-negatives (all-window) test sets.

Mirror of scripts/build_e81_noise_variant_eval_data_20260617.py: the E81 eval
`input` is byte-identical to the formal-pool `input`; only the `instruction`
(CoT) and `output` (placeholder) differ; `gold_output` = the formal `output`
(for negative windows this is {"events": []}).
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORMAL = REPO / "data/stage2_formal_datasets"
ADAPT = REPO / "data/stage2_adaptive_datasets"

E81_REF = ADAPT / "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_test_unseen_pos.jsonl"

SPLITS = ["test_seen", "test_unseen"]


def main():
    with E81_REF.open(encoding="utf-8") as f:
        ref = json.loads(next(f))
    instruction = ref["instruction"]
    placeholder = ref["output"]
    for split in SPLITS:
        src = FORMAL / f"richere_balanced_split1_oracle_mixed_noise_top10_shuffle_{split}_withneg.jsonl"
        with src.open(encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        out_rows = []
        for r in rows:
            out_rows.append({
                "instruction": instruction,
                "input": r["input"],
                "output": placeholder,
                "gold_output": r["output"],
                "meta": r.get("meta", {}),
            })
        out_name = f"richere_e81eval_on_mixed_{split}_withneg.jsonl"
        (ADAPT / out_name).write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows), encoding="utf-8")
        print(f"{split}: {len(out_rows)} rows -> {out_name}")


if __name__ == "__main__":
    main()
