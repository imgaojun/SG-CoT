#!/usr/bin/env python3
"""加固战役:生成文件队列作业(scripts/strengthen_jobs_20260709/pending/)。

作业 = 单个 shell 片段,runner 按文件名排序领取($GPU 由 runner 注入)。
可选首行 `#DEP=<path>`:该路径存在才可领取(训练完成标记 train_results.json)。
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

# ---------- 01 Direct base 重训(恢复被删 warm-start;也是 Tier0 的 Direct base seed)----------
train("01_train_direct_base",
      "configs/generated/stage2_cot/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full_stepmatch.yaml")

# ---------- 02-04 无依赖训练 ----------
for s in (8322, 8333):
    train(f"02_train_gollie_s{s}",
          f"configs/generated/stage2_cot/richere_split1_qwen3_4b_gollie_style_direct_s{s}_full_stepmatch.yaml")
    train(f"03_train_hybrid_s{s}",
          f"configs/generated/stage2_adaptive/richere_split1_qwen3_4b_gollie_style_adaptive_hybrid_sgcot_thinking_evidence_cot_s{s}_full_stepmatch.yaml")
    train(f"04_train_e90sgcot_s{s}",
          f"configs/generated/stage2_adaptive/richere_contactfam_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e83sd_contactfam_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_s{s}_full_stepmatch.yaml")
    train(f"04_train_e91sgcot_s{s}",
          f"configs/generated/stage2_adaptive/richere_movementfam_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e83sd_movementfam_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_s{s}_full_stepmatch.yaml")

# ---------- 05 Tier1:E81 5 seeds x 4 条件 x unseen(82 行,快,先做)----------
E81_RUNS = {
    "base": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_full",
    "r1": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat1_full",
    "r2": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat2_full",
    "r3": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat3_full",
    "r4": f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_repeat4_full",
}
CONDS = ["clean", "random", "hard", "predicted_top10"]
for tag, run in E81_RUNS.items():
    for c in CONDS:
        ev(f"05_eval_e81_{tag}_{c}_unseen", EVID, run,
           f"{DATA_A}/richere_e81eval_on_{c}_test_unseen_pos.jsonl",
           f"{OUT}/robust/e81_{tag}/{c}_unseen", 4, 1024)

# ---------- 06 Tier0:被删 predictions 重建(mixed 主条件)----------
E81_MIXED = f"{DATA_A}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e81_trigger_locked_arbitration_glm51_full1500_thinking_evidence_cot_test_%s_pos.jsonl"
for tag, run in E81_RUNS.items():
    for sp in ("unseen", "seen"):
        ev(f"06_eval_e81_{tag}_mixed_{sp}", EVID, run, E81_MIXED % sp,
           f"{OUT}/mixed/e81_{tag}/test_{sp}", 4, 1024)
E83R = f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e83_richere_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_full"
E83_JSONL = f"{DATA_A}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e83_richere_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_test_%s_pos.jsonl"
for sp in ("unseen", "seen"):
    ev(f"06_eval_e83richere_mixed_{sp}", EVID, E83R, E83_JSONL % sp,
       f"{OUT}/mixed/e83_richere/test_{sp}", 4, 1024)
# Direct repeats(root=ck2064,协议偏差:论文用 dev-selected ck1806,已删,note 说明)
DIRECT_JSONL = "/workspace/project/data/stage2_formal_datasets/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_test_%s_pos.jsonl"
for rep in ("repeat1", "repeat2"):
    run = f"{SFT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_{rep}_full"
    for sp in ("unseen", "seen"):
        ev(f"06_eval_direct_{rep}_mixed_{sp}", DIRE, run, DIRECT_JSONL % sp,
           f"{OUT}/mixed/direct_{rep}/test_{sp}", 8, 512)
# e77 原 seed(root=ck279)predictions 重建
E77 = f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e77_e76_direct_control_full"
E77_JSONL = f"{DATA_A}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e77_e76_direct_control_test_%s_pos.jsonl"
for sp in ("unseen", "seen"):
    ev(f"06_eval_e77_base_mixed_{sp}", DIRE, E77, E77_JSONL % sp,
       f"{OUT}/mixed/e77_base/test_{sp}", 8, 512)

# ---------- 06b Tier1:SG-CoT-SC pooled unseen(3 seeds)----------
G9 = {
    "base": f"{ADAPT}/richere_split1_qwen3_4b_gollie_style_dualmode_g9_cotcalib_full/checkpoint-91",
    "s8322": f"{ADAPT}/richere_split1_qwen3_4b_gollie_style_dualmode_g9_s8322_cotcalib_full/checkpoint-91",
    "s8333": f"{ADAPT}/richere_split1_qwen3_4b_gollie_style_dualmode_g9_s8333_cotcalib_full/checkpoint-91",
}
for tag, mp in G9.items():
    body = (f"bash -lc 'python scripts/self_consistency_eval_20260703.py --model_path {mp} "
            f"--eval_jsonl /workspace/project/data/stage2_formal_datasets/richere_balanced_pooled15_gollie_style_test_unseen_pos.jsonl "
            f"--output_dir /workspace/project/outputs/stage2_analysis/self_consistency_g9_{tag}_pooled_20260709 "
            f"--vote_k 3 2>&1 | tee {LOGS}/06b_sc_g9_{tag}_pooled.log'")
    jobs[f"06b_sc_g9_{tag}_pooled"] = (None, DOCKER + body)

# ---------- 07 依赖 Direct base 的训练 ----------
DEP_BASE = str(R / "outputs/stage2_full_sft_runs_stepmatch_user/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full/train_results.json")
for s in (8322, 8333):
    train(f"07_train_e77_s{s}",
          f"configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e77_e76_direct_control_s{s}_full_stepmatch.yaml",
          dep=DEP_BASE)
    train(f"07_train_e95_s{s}",
          f"configs/generated/stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e95_autocluster_glm51_full1500_thinking_evidence_cot_s{s}_full_stepmatch.yaml",
          dep=DEP_BASE)

# ---------- 08 Tier3 Direct 臂(16ep,重)----------
for s in (8322, 8333):
    train(f"08_train_e90direct_s{s}",
          f"configs/generated/stage2_cot/richere_contactfam_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_s{s}_full_stepmatch.yaml")
    train(f"08_train_e91direct_s{s}",
          f"configs/generated/stage2_cot/richere_movementfam_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_s{s}_full_stepmatch.yaml")

# ---------- 09 robust seen 补全(361 行,慢,后置)----------
for tag, run in E81_RUNS.items():
    for c in CONDS:
        ev(f"09_eval_e81_{tag}_{c}_seen", EVID, run,
           f"{DATA_A}/richere_e81eval_on_{c}_test_seen_pos.jsonl",
           f"{OUT}/robust/e81_{tag}/{c}_seen", 4, 1024)

# ---------- 10 新 seed 模型的评测(依赖各自训练完成)----------
def dep_of(outdir_container):
    return outdir_container.replace("/workspace/project", str(R)) + "/train_results.json"

for s in (8322, 8333):
    m = f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e77_e76_direct_control_s{s}_full"
    for sp in ("unseen", "seen"):
        ev(f"10_eval_e77_s{s}_{sp}", DIRE, m, E77_JSONL % sp, f"{OUT}/new/e77_s{s}/test_{sp}", 8, 512, dep=dep_of(m))
    m = f"{SFT}/richere_split1_qwen3_4b_gollie_style_direct_s{s}_full"
    GJ = "/workspace/project/data/stage2_formal_datasets/richere_balanced_split1_gollie_style_top10_shuffle_test_%s_pos.jsonl"
    for sp in ("unseen", "seen"):
        ev(f"10_eval_gollie_s{s}_{sp}", DIRE, m, GJ % sp, f"{OUT}/new/gollie_s{s}/test_{sp}", 8, 512, dep=dep_of(m))
    m = f"{ADAPT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e95_autocluster_glm51_full1500_thinking_evidence_cot_s{s}_full"
    E95J = f"{DATA_A}/richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive_e95_autocluster_glm51_full1500_thinking_evidence_cot_test_%s_pos.jsonl"
    for sp in ("unseen", "seen"):
        ev(f"10_eval_e95_s{s}_{sp}", EVID, m, E95J % sp, f"{OUT}/new/e95_s{s}/test_{sp}", 4, 1024, dep=dep_of(m))
    m = f"{ADAPT}/richere_split1_qwen3_4b_gollie_style_adaptive_hybrid_sgcot_thinking_evidence_cot_s{s}_full"
    HJ = f"{DATA_A}/richere_balanced_split1_gollie_style_top10_shuffle_adaptive_hybrid_sgcot_thinking_evidence_cot_test_%s_pos.jsonl"
    for sp in ("unseen", "seen"):
        ev(f"10_eval_hybrid_s{s}_{sp}", EVID, m, HJ % sp, f"{OUT}/new/hybrid_s{s}/test_{sp}", 4, 1024, dep=dep_of(m))
    for fam, famtag in (("contactfam", "e90"), ("movementfam", "e91")):
        m = f"{ADAPT}/richere_{fam}_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e83sd_{fam}_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_s{s}_full"
        FJ = f"{DATA_A}/richere_{fam}_split1_oracle_mixed_noise_top10_shuffle_adaptive_e83sd_{fam}_glm51_full1500_thinking_evidence_cot_test_%s_pos.jsonl"
        for sp in ("unseen", "seen"):
            ev(f"10_eval_{famtag}sgcot_s{s}_{sp}", EVID, m, FJ % sp, f"{OUT}/new/{famtag}sgcot_s{s}/test_{sp}", 4, 1024, dep=dep_of(m))
        m = f"{SFT}/richere_{fam}_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_s{s}_full"
        DFJ = f"/workspace/project/data/stage2_formal_datasets/richere_{fam}_split1_oracle_mixed_noise_top10_shuffle_test_%s_pos.jsonl"
        for sp in ("unseen", "seen"):
            ev(f"10_eval_{famtag}direct_s{s}_{sp}", DIRE, m, DFJ % sp, f"{OUT}/new/{famtag}direct_s{s}/test_{sp}", 8, 512, dep=dep_of(m))

# Direct base 重训后的评测(Tier0 Direct base seed)
m = f"{SFT}/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full"
for sp in ("unseen", "seen"):
    ev(f"10_eval_direct_base_{sp}", DIRE, m, DIRECT_JSONL % sp, f"{OUT}/new/direct_base/test_{sp}", 8, 512, dep=dep_of(m))

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
