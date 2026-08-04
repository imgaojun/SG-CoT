import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_path(run_dir: Path, tag: str):
    if tag == "final":
        return run_dir
    return run_dir / tag


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint_tags", nargs="+", required=True)
    parser.add_argument("--gpu_ids", nargs="+", required=True)
    parser.add_argument("--metric_keys", nargs="+", default=["argument_f1", "event_f1", "trigger_f1"])
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--eval_script",
        default="src/stage2_quality_validation/eval_adapter_generation.py",
        help="Evaluation script to run for each checkpoint. The script must write summary.json with the requested metric keys.",
    )
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--status_json", required=True)
    parser.add_argument("--reuse_existing", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    queue = []
    for tag in args.checkpoint_tags:
        eval_dir = output_root / tag
        summary_path = eval_dir / "summary.json"
        if args.reuse_existing and summary_path.exists():
            log(f"skip existing {tag}")
            continue
        queue.append(tag)

    running = {}
    completed = []
    failed = []

    while queue or running:
        used_gpus = {item["gpu"] for item in running.values()}
        free_gpus = [gpu for gpu in args.gpu_ids if gpu not in used_gpus]

        while queue and free_gpus:
            gpu = free_gpus.pop(0)
            tag = queue.pop(0)
            ckpt = checkpoint_path(run_dir, tag)
            eval_dir = output_root / tag
            eval_dir.mkdir(parents=True, exist_ok=True)
            child_log = eval_dir / "launcher.log"
            cmd = [
                sys.executable,
                args.eval_script,
                "--base_model",
                args.base_model,
                "--adapter_path",
                ckpt.as_posix(),
                "--eval_jsonl",
                args.eval_jsonl,
                "--output_dir",
                eval_dir.as_posix(),
                "--batch_size",
                str(args.batch_size),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            fh = open(child_log, "a", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[proc.pid] = {"proc": proc, "tag": tag, "gpu": gpu, "fh": fh}
            log(f"launched {tag} on gpu={gpu} pid={proc.pid}")

        time.sleep(10)
        finished = []
        for pid, item in running.items():
            ret = item["proc"].poll()
            if ret is None:
                continue
            item["fh"].close()
            record = {"tag": item["tag"], "gpu": item["gpu"], "pid": pid, "returncode": ret}
            if ret == 0:
                completed.append(record)
                log(f"completed {item['tag']} on gpu={item['gpu']}")
            else:
                failed.append(record)
                log(f"failed {item['tag']} on gpu={item['gpu']} returncode={ret}")
            finished.append(pid)
        for pid in finished:
            del running[pid]

        write_json(
            Path(args.status_json),
            {
                "pending": queue,
                "running": [{"pid": pid, **{k: v for k, v in item.items() if k != "proc" and k != "fh"}} for pid, item in running.items()],
                "completed": completed,
                "failed": failed,
            },
        )

    if failed:
        raise SystemExit(f"shortlist evaluation failed for {len(failed)} checkpoints")

    rows = []
    best = None
    for tag in args.checkpoint_tags:
        summary_path = output_root / tag / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        score_tuple = tuple(summary[key] for key in args.metric_keys)
        row = {
            "checkpoint_tag": tag,
            "checkpoint_path": checkpoint_path(run_dir, tag).as_posix(),
            "eval_dir": (output_root / tag).as_posix(),
            "metric_keys": args.metric_keys,
            "score_tuple": score_tuple,
            "summary": summary,
        }
        rows.append(row)
        if best is None:
            best = row
        else:
            if args.greater_is_better:
                if score_tuple > tuple(best["score_tuple"]):
                    best = row
            else:
                if score_tuple < tuple(best["score_tuple"]):
                    best = row

    payload = {
        "run_dir": run_dir.as_posix(),
        "eval_jsonl": args.eval_jsonl,
        "metric_keys": args.metric_keys,
        "greater_is_better": args.greater_is_better,
        "batch_size": args.batch_size,
        "checkpoint_tags": args.checkpoint_tags,
        "best": best,
        "candidates": rows,
        "selection_protocol": "parallel_shortlist_dev_select",
    }
    write_json(output_root / "selection_summary.json", payload)
    log(f"wrote selection_summary.json best={best['checkpoint_tag'] if best else 'none'}")


if __name__ == "__main__":
    main()
