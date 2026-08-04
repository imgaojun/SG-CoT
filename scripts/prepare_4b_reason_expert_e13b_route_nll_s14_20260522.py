import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO = Path("/mnt/disk/gaojun/research/progressive-ee")
DATA_DIR = REPO / "data/stage2_adaptive_datasets"
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
SOURCE_BRANCH = "confrare10_heur10_typeonlylite"
E13B_BRANCH = "confrare10_typeonlylite_directwarm_retention_e13b"
S14_BRANCH = "confrare10_typeonlylite_directwarm_retention_e13b_s14"
TZ = timezone(timedelta(hours=8))


def now_iso():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_dataset_info(dataset_name, file_name):
    info_path = DATA_DIR / "dataset_info.json"
    info = load_json(info_path)
    info[dataset_name] = {
        "file_name": file_name,
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
    write_json(info_path, info)


def clone(row, role, route_mode=None, route_label=None):
    out = json.loads(json.dumps(row, ensure_ascii=False))
    meta = out.setdefault("meta", {})
    meta["adaptive_source"] = "e13b_directwarm_retention_s14"
    meta["e13b_branch"] = E13B_BRANCH
    meta["s14_source_dataset"] = role
    if route_mode is not None:
        meta["adaptive_route_mode"] = route_mode
    if route_label is not None:
        meta["adaptive_route_label"] = route_label
    return out


def make_dataset(target_name, source_name, role, route_mode=None, route_label=None):
    rows = load_jsonl(DATA_DIR / f"{source_name}.jsonl")
    out_rows = [clone(row, role, route_mode, route_label) for row in rows]
    file_name = f"{target_name}.jsonl"
    write_jsonl(DATA_DIR / file_name, out_rows)
    update_dataset_info(target_name, file_name)
    write_json(
        DATA_DIR / f"{target_name}.meta.json",
        {
            "dataset": target_name,
            "source_dataset": source_name,
            "num_examples": len(out_rows),
            "created_at": now_iso(),
            "purpose": "E13B S14 route-NLL selector calibration",
        },
    )
    return target_name, len(out_rows)


def main():
    made = {}
    for split in ["dev_seen", "test_seen", "test_unseen"]:
        source = f"{DATA_PREFIX}_{SOURCE_BRANCH}_{split}_pos"
        target = f"{DATA_PREFIX}_{S14_BRANCH}_{split}_pos"
        made[target] = make_dataset(target, source, f"neutral_{split}", "free_route", None)[1]

    for mode, label in [("forced_direct", "direct"), ("forced_reason", "reason")]:
        source = f"{DATA_PREFIX}_{SOURCE_BRANCH}_{mode}_dev_seen_pos"
        target = f"{DATA_PREFIX}_{S14_BRANCH}_{mode}_dev_seen_pos"
        made[target] = make_dataset(target, source, f"{mode}_dev_seen", mode, label)[1]

    print(json.dumps({"created": made}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
