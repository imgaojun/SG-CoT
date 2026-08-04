#!/usr/bin/env python3
"""Build E81-format CoT eval data for noise/retrieval variants (no API, no retrain).

The E81 eval `input` is byte-identical to the source direct pool `input` for a given
noise setting; only the `instruction` (CoT) and `gold_output` differ. So we can
evaluate the trained E81 model under different test-time candidate sets (clean /
random / hard noise, and predicted_top10 real retrieval) by reformatting each
variant's gold pool into the E81 eval format.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORMAL = REPO / "data/stage2_formal_datasets"
ADAPT = REPO / "data/stage2_adaptive_datasets"

# An existing E81 eval file to copy the CoT instruction + output placeholder from.
E81_REF = ADAPT / "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_test_unseen_pos.jsonl"

VARIANTS = {
    "clean": "richere_balanced_split1_oracle_clean_top10_shuffle",
    "random": "richere_balanced_split1_oracle_random_noise_top10_shuffle",
    "hard": "richere_balanced_split1_oracle_hard_noise_top10_shuffle",
    "predicted_top10": "richere_balanced_split1_predicted_top10",
}
SPLITS = ["test_seen", "test_unseen"]


def main():
    with E81_REF.open(encoding="utf-8") as f:
        ref = json.loads(next(f))
    instruction = ref["instruction"]
    placeholder = ref["output"]
    out_base = ADAPT
    manifest = {}
    for vkey, prefix in VARIANTS.items():
        for split in SPLITS:
            src = FORMAL / f"{prefix}_{split}_pos.jsonl"
            if not src.exists():
                print("MISSING", src)
                continue
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
            out_name = f"richere_e81eval_on_{vkey}_{split}_pos.jsonl"
            (out_base / out_name).write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows), encoding="utf-8")
            manifest.setdefault(vkey, {})[split] = {"path": str(out_base / out_name), "rows": len(out_rows)}
            print(f"{vkey}/{split}: {len(out_rows)} rows -> {out_name}")
    (REPO / "reports/artifacts/2026-06-17_e81_noise_variant_eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("data_base per variant: data/stage2_adaptive_datasets/richere_e81eval_on_{variant}")


if __name__ == "__main__":
    main()
