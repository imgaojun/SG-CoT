#!/usr/bin/env python3
"""M4 reviewer item: full-scale free-form CoT control (e88b).

Assemble the 1432-row train set from outputs/strengthen_20260709/freeform_pool/*.jsonl
(accepted rows from multiple generation passes; dedupe by sample_id keeping first
occurrence in sorted-filename order), converting each accepted rec to the exact train
format of the existing 526-row e88 dataset (field mapping mirrors
scripts/generate_strategy_variants_cot_e47_20260606.py::make_evidence_row /
scripts/generate_evidence_cot_e40_20260604.py::write_datasets; verified byte-exact
against the 334 rows shared with the e88 train file).

Also copies the e88 eval splits under e88b names and registers all four datasets in
data/stage2_adaptive_datasets/dataset_info.json.
"""
import glob
import json
import shutil
from pathlib import Path

R = Path("/mnt/disk/gaojun/research/progressive-ee")
POOL = R / "outputs/strengthen_20260709/freeform_pool"
DATA = R / "data/stage2_adaptive_datasets"
E88_BRANCH = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e88_freeform_nl_glm51_full1500_thinking_evidence_cot"
E88B_BRANCH = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e88b_freeform_full1432_thinking_evidence_cot"


def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # Reference rows (for instruction constant + byte-exact verification).
    e88_train = load_jsonl(DATA / f"{E88_BRANCH}_train_pos.jsonl")
    instructions = {r["instruction"] for r in e88_train}
    assert len(instructions) == 1, "e88 instruction is not constant"
    instruction = instructions.pop()
    e88_by_sid = {r["meta"]["e40_sample_id"]: r for r in e88_train}

    # Dedupe pool by sample_id, first occurrence, sorted filename order.
    pool = {}
    for fp in sorted(glob.glob(str(POOL / "*.jsonl"))):
        for rec in load_jsonl(fp):
            sid = rec["sample_id"]
            if sid not in pool:
                pool[sid] = rec
    accepted = {sid: r for sid, r in pool.items() if r.get("accepted")}
    print(f"pool unique sample_ids: {len(pool)}, accepted: {len(accepted)}")
    assert len(accepted) == 1432, f"expected 1432 accepted unique rows, got {len(accepted)}"

    # Convert to train format (mirror of e47 make_evidence_row on top of e40 base row).
    train_rows = []
    verified = 0
    for sid in sorted(accepted):
        rec = accepted[sid]
        run_name = sid.rsplit("_", 1)[0]  # per-row provenance run name
        meta = dict(rec["meta"])
        meta.update(
            {
                "adaptive_source": "strategy_variant_evidence_cot_e47b",
                "adaptive_target_style": "candidate_audit_thinking_surface_evidence_cot",
                "adaptive_dataset_role": "train",
                "e40_run_name": run_name,
                "e40_generator_model": "deepseek-v4-pro",
                "e40_verifier_model": "deepseek-v4-pro",
                "e47_run_name": run_name,
                "e47_variant": "candidate_audit",
                "e47_generator_model": "deepseek-v4-pro",
                "e47_verifier_model": "deepseek-v4-pro",
            }
        )
        row = {
            "instruction": instruction,
            "input": rec["input"],
            "output": f"<thinking>{rec['thinking'].strip()}</thinking>\n<final>{compact(rec['final_obj'])}</final>",
            "meta": meta,
            "gold_output": rec["gold_output"],
        }
        if sid in e88_by_sid:
            ref = e88_by_sid[sid]
            assert row["instruction"] == ref["instruction"]
            assert row["input"] == ref["input"]
            assert row["output"] == ref["output"]
            assert row["gold_output"] == ref["gold_output"]
            verified += 1
        train_rows.append(row)
    print(f"byte-exact verified against existing e88 train rows: {verified}")

    train_path = DATA / f"{E88B_BRANCH}_train_pos.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(train_rows)} rows -> {train_path}")

    # Copy eval splits verbatim under e88b names.
    names = [f"{E88B_BRANCH}_train_pos"]
    for split in ("dev_seen", "test_seen", "test_unseen"):
        src = DATA / f"{E88_BRANCH}_{split}_pos.jsonl"
        dst = DATA / f"{E88B_BRANCH}_{split}_pos.jsonl"
        shutil.copyfile(src, dst)
        names.append(f"{E88B_BRANCH}_{split}_pos")
        print(f"copied {src.name} -> {dst.name}")

    # Register in dataset_info.json mirroring the e88 entries.
    info_path = DATA / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for name in names:
        info[name] = {
            "file_name": f"{name}.jsonl",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"registered {len(names)} datasets in {info_path}")


if __name__ == "__main__":
    main()
