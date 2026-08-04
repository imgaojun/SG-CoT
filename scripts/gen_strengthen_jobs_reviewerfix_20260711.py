#!/usr/bin/env python3
"""Reviewer-fix work items 2026-07-11: generate file-queue jobs (prefix 13_/14_).

Same job format / docker template as scripts/gen_strengthen_jobs_20260709.py.

Item 1: with-negatives test sets (Direct base/r1/r2 + E81 5 seeds).
Item 2: E77b row-matched same-data control (3 seed trainings + evals).
Item 3: Direct @ predicted top-10 (matched retrieval comparison).
Item 4: Audit-CoT (E56 xml_cot) 5-seed re-eval on mixed test sets.
"""
from pathlib import Path

R = Path("/mnt/disk/gaojun/research/progressive-ee")
JOBS = R / "scripts/strengthen_jobs_20260709/pending"
JOBS.mkdir(parents=True, exist_ok=True)
LOGS = "/workspace/project/outputs/strengthen_20260709/logs"
(R / "outputs/strengthen_20260709/logs").mkdir(parents=True, exist_ok=True)

DOCKER = (
    'docker run --rm --user root --ipc host --shm-size 16g --gpus "device=${GPU}" '
    "-v /mnt/disk/gaojun/research/progressive-ee:/workspace/project "
    "-v /mnt/disk/gaojun/models:/workspace/models "
    "-v /mnt/disk/gaojun/research/llamafactory-lab/cache/huggingface:/workspace/.cache/huggingface "
    "-v /mnt/disk/gaojun/research/llamafactory-lab/cache/torch_extensions:/workspace/.cache/torch_extensions "
    "-e PYTHONUNBUFFERED=1 -e HF_HOME=/workspace/.cache/huggingface "
    "-e TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions "
    "-w /workspace/project llamafactory-lab:0.9.4-py3.12 "
)
BASE = "/workspace/models/LLM-Research/Qwen3-4B"
EVID = "src/stage2_quality_validation/eval_adaptive_route_generation_evidence.py"
DIRE = "src/stage2_quality_validation/eval_adapter_generation.py"
ADAPT = "/workspace/project/outputs/stage2_adaptive_runs_user"
SFT = "/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user"
DATA_A = "/workspace/project/data/stage2_adaptive_datasets"
DATA_F = "/workspace/project/data/stage2_formal_datasets"
OUT = "/workspace/project/outputs/strengthen_20260709"

jobs = {}


def train(name, cfg, dep=None):
    body = f"bash -lc 'FORCE_TORCHRUN=1 llamafactory-cli train {cfg} 2>&1 | tee {LOGS}/{name}.log'"
    jobs[name] = (dep, DOCKER + body)


def ev(name, script, model, jsonl, out, batch, maxnew, dep=None):
    body = (f"bash -lc 'python {script} --base_model {BASE} --adapter_path {model} "
            f"--eval_jsonl {jsonl} --output_dir {out} --batch_size {batch} --max_new_tokens {maxnew} "
            f"2>&1 | tee {LOGS}/{name}.log'")
    jobs[name] = (dep, DOCKER + body)


DIRECT_RUNS = {
    "base": f"{SFT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full",
    "repeat1": f"{SFT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_repeat1_full",
    "repeat2": f"{SFT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_repeat2_full",
}
E81_RUNS = {
    "base": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full",
    "r1": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat1_full",
    "r2": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat2_full",
    "r3": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat3_full",
    "r4": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat4_full",
}

# ---------- Item 1: with-negatives test sets ----------
for tag, run in DIRECT_RUNS.items():
    for sp in ("unseen", "seen"):
        ev(f"13_withneg_direct_{tag}_{sp}", DIRE, run,
           f"{DATA_F}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_test_{sp}_withneg.jsonl",
           f"{OUT}/withneg/direct_{tag}/test_{sp}", 8, 512)
for tag, run in E81_RUNS.items():
    for sp in ("unseen", "seen"):
        ev(f"13_withneg_e81_{tag}_{sp}", EVID, run,
           f"{DATA_A}/richere_e81eval_on_mixed_test_{sp}_withneg.jsonl",
           f"{OUT}/withneg/e81_{tag}/test_{sp}", 4, 1024)

# ---------- Item 3: Direct @ predicted top-10 ----------
for tag, run in DIRECT_RUNS.items():
    for sp in ("unseen", "seen"):
        ev(f"13_robust_direct_{tag}_predicted_{sp}", DIRE, run,
           f"{DATA_F}/richere_balanced_split1_predicted_top10_test_{sp}_pos.jsonl",
           f"{OUT}/robust_direct/{tag}_predicted_{sp}", 8, 512)

# ---------- Item 4: Audit-CoT 5-seed re-eval ----------
AUDIT_RUNS = {
    "base": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_full",
    "r1": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat1_full",
    "r2": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat2_full",
    "r3": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat3_full",
    "r4": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_repeat4_full",
}
AUDIT_JSONL = f"{DATA_A}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e56_full1500_glm51_candidate_audit_xml_cot_thinking_evidence_cot_test_%s_pos.jsonl"
for tag, run in AUDIT_RUNS.items():
    for sp in ("unseen", "seen"):
        ev(f"13_eval_auditcot_{tag}_{sp}", EVID, run, AUDIT_JSONL % sp,
           f"{OUT}/auditcot/{tag}/test_{sp}", 4, 1024)

# ---------- Item 2: E77b row-matched control (train + eval) ----------
DEP_DIRECT = str(R / "outputs/stage2_full_sft_runs_stepmatch_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full/train_results.json")
E77B_JSONL = f"{DATA_A}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e77b_e81_rowmatched_control_test_%s_pos.jsonl"
for stag in ("base", "s8322", "s8333"):
    seed_tag = "" if stag == "base" else f"_{stag}"
    cfg = ("configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_"
           f"top10_shuffle_adaptive_e77b_e81_rowmatched_control{seed_tag}_full_stepmatch.yaml")
    train(f"13_train_e77b_{stag}", cfg, dep=DEP_DIRECT)
    m = f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e77b_e81_rowmatched_control{seed_tag}_full"
    dep = m.replace("/workspace/project", str(R)) + "/train_results.json"
    for sp in ("unseen", "seen"):
        ev(f"14_eval_e77b_{stag}_{sp}", EVID, m, E77B_JSONL % sp,
           f"{OUT}/e77b/{stag}/test_{sp}", 4, 1024, dep=dep)

# ---------- 写文件 ----------
for name, (dep, cmd) in sorted(jobs.items()):
    lines = []
    if dep:
        lines.append(f"#DEP={dep}")
    lines.append(cmd)
    (JOBS / f"{name}.job").write_text("\n".join(lines) + "\n")
print(f"共生成 {len(jobs)} 个作业:")
import collections
c = collections.Counter(n.split("_")[0] for n in jobs)
for k in sorted(c):
    print(f"  {k}: {c[k]}")
for n in sorted(jobs):
    print(" ", n)
