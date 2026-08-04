import json
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
BASE_MODEL = "/workspace/models/LLM-Research/Qwen3-1.7B"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
FORMAL_ROOT = "outputs/stage2_adaptive_runs_user_formal_clean"
SELECTION_ROOT = "outputs/stage2_adaptive_runs_user_devpick_frontier/protocol_selections"
OUT = REPO / "configs/generated/stage2_adaptive/richere_qwen3_1_7b_adaptive_outcome_calibrated_execution_gate_formal_manifest.json"

BRANCHES = [
    "outcome15cal_nlltop10_type_role_hint_plan_lite_routeaux2x_reasonos2",
    "outcome15cal_nlltop15_type_role_hint_plan_lite_routeaux2x_reasonos2",
]
MODES = ["free_route", "forced_direct", "forced_reason"]
SPLITS = ["test", "test_seen", "test_unseen"]


def eval_spec(branch: str, mode: str, split: str):
    mode_prefix = "" if mode == "free_route" else f"_{mode}"
    return {
        "name": f"{mode}/{split}",
        "eval_jsonl": f"data/stage2_adaptive_datasets/{DATA_PREFIX}_{branch}{mode_prefix}_{split}_pos.jsonl",
        "output_dir": f"{FORMAL_ROOT}/richere_split1_qwen3_1_7b_adaptive_{branch}/frontier_execution_gate/{mode}/{split}",
    }


def gate_passed(selection_path: Path):
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = payload.get("frontier_selected_checkpoint") or {}
    return bool(selected.get("gate_pass"))


def main():
    runs = []
    skipped = []
    for branch in BRANCHES:
        selection = f"{SELECTION_ROOT}/{branch}__execution_gate/selection_summary.json"
        selection_path = REPO / selection
        if not selection_path.exists():
            skipped.append({"branch": branch, "reason": "missing_selection_summary"})
            continue
        if not gate_passed(selection_path):
            skipped.append({"branch": branch, "reason": "execution_gate_failed"})
            continue
        runs.append(
            {
                "tag": f"{branch}__execution_gate",
                "selection_summary": selection,
                "evals": [eval_spec(branch, mode, split) for mode in MODES for split in SPLITS],
            }
        )
    payload = {"base_model": BASE_MODEL, "runs": runs, "skipped": skipped}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": OUT.as_posix(), "runs": len(runs), "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
