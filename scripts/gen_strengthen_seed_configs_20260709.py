#!/usr/bin/env python3
"""加固战役:生成 +2 seeds(8322/8333)训练 config。

规则(从既有 _s8322/_s8333 变体 diff 得出):
- 加一行 `seed: NNNN`(插在 cutoff_len 附近任意稳定位置——这里统一插在
  `model_name_or_path` 行之后,LlamaFactory 只看键不看位置);
- `output_dir` 末段 `..._full` -> `..._sNNNN_full`;
- 其余字段一律不动(warm-start 保持原 config 指向)。
仅生成文件并打印 diff,不启动任何训练。
"""
import sys
from pathlib import Path

R = Path("/mnt/disk/gaojun/research/progressive-ee")
GEN = R / "configs/generated"

# (源 config 相对路径, 简称)
SOURCES = [
    # Tier 2
    ("stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e77_e76_direct_control_full_stepmatch.yaml", "e77"),
    ("stage2_cot/richere_split1_qwen3_4b_gollie_style_direct_full_stepmatch.yaml", "gollie"),
    ("stage2_adaptive/richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e95_autocluster_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml", "e95"),
    ("stage2_adaptive/richere_split1_qwen3_4b_gollie_style_adaptive_hybrid_sgcot_thinking_evidence_cot_full_stepmatch.yaml", "hybrid"),
    # Tier 3(两臂 x 两家族)
    ("stage2_cot/richere_contactfam_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full_stepmatch.yaml", "e90direct"),
    ("stage2_adaptive/richere_contactfam_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e83sd_contactfam_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml", "e90sgcot"),
    ("stage2_cot/richere_movementfam_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full_stepmatch.yaml", "e91direct"),
    ("stage2_adaptive/richere_movementfam_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive_e83sd_movementfam_trigger_locked_schema_driven_glm51_full1500_thinking_evidence_cot_full_stepmatch.yaml", "e91sgcot"),
]
SEEDS = [8322, 8333]


def variant_lines(lines, seed):
    out = []
    seeded = False
    for ln in lines:
        if ln.startswith("seed:"):
            raise SystemExit(f"源 config 已含 seed 行: {ln}")
        if ln.startswith("output_dir:"):
            raw = ln.split(":", 1)[1].strip()
            quoted = raw.startswith('"')
            path = raw.strip('"')
            assert path.endswith("_full"), path
            new = f"{path[:-5]}_s{seed}_full"
            ln = f'output_dir: "{new}"' if quoted else f"output_dir: {new}"
        out.append(ln)
        if ln.startswith("model_name_or_path:") and not seeded:
            out.append(f"seed: {seed}")
            seeded = True
    assert seeded, "未找到 model_name_or_path 行"
    return out


def main():
    made = []
    for rel, tag in SOURCES:
        src = GEN / rel
        text = src.read_text().splitlines()
        for seed in SEEDS:
            dst = src.with_name(src.name.replace("_full_stepmatch.yaml", f"_s{seed}_full_stepmatch.yaml"))
            dst.write_text("\n".join(variant_lines(text, seed)) + "\n")
            made.append((tag, seed, dst))
    print("生成的 config 与源的 diff(应只有 seed 行 + output_dir):\n")
    import subprocess
    for tag, seed, dst in made:
        src = GEN / dict(SOURCES)[[k for k, v in dict(SOURCES).items() if v == tag][0]] if False else None
    for rel, tag in SOURCES:
        src = GEN / rel
        for seed in SEEDS:
            dst = src.with_name(src.name.replace("_full_stepmatch.yaml", f"_s{seed}_full_stepmatch.yaml"))
            d = subprocess.run(["diff", str(src), str(dst)], capture_output=True, text=True)
            print(f"--- {tag} s{seed} ({dst.name})")
            print(d.stdout.strip() or "(无差异!错误)")
            print()


if __name__ == "__main__":
    main()
